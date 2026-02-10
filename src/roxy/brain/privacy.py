"""Privacy gateway for PII detection and redaction.

Enforces Roxy's privacy model by detecting, redacting, and logging sensitive data.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


# PII Risk levels for classification
class PIIRiskLevel(str, Enum):
    """Risk levels for PII data types."""

    LOW = "low"  # Publicly available or low sensitivity
    MEDIUM = "medium"  # Semi-private, could be used for profiling
    HIGH = "high"  # Sensitive personal information
    CRITICAL = "critical"  # Financial, medical, or identity theft risk


# Mapping of PII patterns to risk levels
PII_RISK_LEVELS: dict[str, PIIRiskLevel] = {
    "email": PIIRiskLevel.MEDIUM,
    "phone_na": PIIRiskLevel.MEDIUM,
    "phone_swedish": PIIRiskLevel.MEDIUM,
    "phone_uk": PIIRiskLevel.MEDIUM,
    "phone_intl": PIIRiskLevel.MEDIUM,
    "ssn": PIIRiskLevel.CRITICAL,
    "personnummer_swedish": PIIRiskLevel.CRITICAL,
    "credit_card": PIIRiskLevel.CRITICAL,
    "ip_address": PIIRiskLevel.MEDIUM,
    "ipv6_address": PIIRiskLevel.MEDIUM,
    "password": PIIRiskLevel.CRITICAL,
    "pin": PIIRiskLevel.CRITICAL,
    "account_number": PIIRiskLevel.HIGH,
    "address": PIIRiskLevel.HIGH,
    "url": PIIRiskLevel.LOW,
    "mac_address": PIIRiskLevel.LOW,
    "iban": PIIRiskLevel.HIGH,
    "passport_us": PIIRiskLevel.CRITICAL,
    "date_of_birth": PIIRiskLevel.HIGH,
    "medical_id": PIIRiskLevel.CRITICAL,
    "drivers_license": PIIRiskLevel.HIGH,
    "api_key": PIIRiskLevel.CRITICAL,
    "token": PIIRiskLevel.CRITICAL,
}


class ConsentMode(str, Enum):
    """User consent modes for cloud LLM access."""

    ASK = "ask"
    ALWAYS = "always"
    NEVER = "never"


@dataclass
class PIIMatch:
    """A detected PII instance."""

    pattern_name: str
    original: str
    start: int
    end: int
    placeholder: str
    risk_level: PIIRiskLevel = PIIRiskLevel.MEDIUM

    def __str__(self) -> str:
        return f"PIIMatch({self.pattern_name}, {self.placeholder}, risk={self.risk_level.value})"


@dataclass
class RedactionResult:
    """Result of PII redaction."""

    redacted_text: str
    pii_matches: list[PIIMatch] = field(default_factory=list)
    was_redacted: bool = False

    def __str__(self) -> str:
        return f"RedactionResult(redacted={self.was_redacted}, matches={len(self.pii_matches)})"


class PrivacyGateway:
    """
    Gateway that enforces Roxy's privacy model.

    Responsibilities:
    1. Detect PII in text using regex patterns
    2. Redact PII with placeholders
    3. Restore PII in responses
    4. Log all cloud requests
    5. Handle user consent for cloud access
    """

    # PII regex patterns
    PATTERNS: dict[str, re.Pattern] = {
        "email": re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            re.IGNORECASE,
        ),
        # North American phone (US/Canada) - more permissive
        "phone_na": re.compile(
            r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        ),
        # Swedish phone: +46 XX XXX XXXX or 0XX XXX XX XX or 0XX-XXX XX XX
        "phone_swedish": re.compile(
            r"(?:\+46|0)[\s-]?\d{1,3}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}",
        ),
        # UK phone: +44 XXXX XXXXXX or 0XXX XXX XXXX
        "phone_uk": re.compile(
            r"(?:\+44|0)[\s-]?\d{3,4}[\s-]?\d{3}[\s-]?\d{3,4}",
        ),
        # Generic international phone: +XX XXX XXX XXXX
        "phone_intl": re.compile(
            r"\+\d{1,4}[\s-]?\d{2,4}[\s-]?\d{2,4}[\s-]?\d{2,6}",
        ),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        # Swedish personnummer: YYMMDD-XXXX or YYYYMMDD-XXXX
        "personnummer_swedish": re.compile(
            r"\b(?:19|20)?\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])[-]\d{4}\b",
        ),
        "credit_card": re.compile(
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        ),
        "ip_address": re.compile(
            r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        ),
        # Contextual PII: password
        "password": re.compile(
            r"(?:password|passwd|pwd)[\s=:]+[^\s'\"]+",
            re.IGNORECASE,
        ),
        # Contextual PII: PIN code
        "pin": re.compile(
            r"(?:pin|passcode)[\s=:]+\d{4,8}",
            re.IGNORECASE,
        ),
        # Contextual PII: account number
        "account_number": re.compile(
            r"(?:account|acct|customer)[\s#]+(?:number|no|num)?[\s#:]*\d{6,}",
            re.IGNORECASE,
        ),
        # Enhanced address pattern (international)
        "address": re.compile(
            r"\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Court|Ct|Place|Pl)[,\s]+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?[,\s]+[A-Z]{2}\s+\d{5}",
            re.MULTILINE,
        ),
        # IPv6 address (simplified - covers common formats)
        "ipv6_address": re.compile(
            r"(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,7}|::1|::",
        ),
        # URL pattern (http/https)
        "url": re.compile(
            r"https?://(?:[-\w.]|(?:%[0-9a-fA-F]{2}))+[/\w .?=&%+-]*",
        ),
        # MAC address
        "mac_address": re.compile(
            r"(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}",
        ),
        # IBAN (International Bank Account Number) - simplified pattern
        "iban": re.compile(
            r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,35}\b",
        ),
        # US Passport number
        "passport_us": re.compile(
            r"\b\d{9}\b",  # Contextual - might need enhancement
        ),
        # Date of birth (ISO format: YYYY-MM-DD or similar)
        "date_of_birth": re.compile(
            r"\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b",
        ),
        # Medical ID / Health insurance number
        "medical_id": re.compile(
            r"(?:medical|health|insurance)[\s#]+(?:id|record|number|no|num)?[\s#:]*[A-Z0-9]{5,}",
            re.IGNORECASE,
        ),
        # Driver's license (US format - state-specific patterns)
        "drivers_license": re.compile(
            r"(?:driver'?s?[\s-]*?license|dl)[\s#]*:?\s*[A-Z0-9]{6,15}",
            re.IGNORECASE,
        ),
        # API key pattern (sk-, pk-, etc.)
        "api_key": re.compile(
            r"\b(sk|pk|ak|api)[_-]?[a-zA-Z0-9]{20,}\b",
        ),
        # JWT token
        "token": re.compile(
            r"eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+",
        ),
    }

    def __init__(
        self,
        redact_patterns: list[str] | None = None,
        consent_mode: ConsentMode = ConsentMode.ASK,
        log_path: Path | str = "data/cloud_requests.log",
    ) -> None:
        """Initialize privacy gateway.

        Args:
            redact_patterns: List of pattern names to redact. None uses all.
            consent_mode: User consent mode for cloud access.
            log_path: Path to cloud request log file.
        """
        self.consent_mode = consent_mode
        self.log_path = Path(log_path).expanduser()
        self._pattern_names = redact_patterns or list(self.PATTERNS.keys())

        # Ensure log directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Counter for generating unique placeholders
        self._placeholder_counter = 0

    async def can_use_cloud(self) -> tuple[bool, str | None]:
        """
        Check if cloud LLM can be used based on consent mode.

        Returns:
            Tuple of (allowed, message). Message is for user notification.
        """
        if self.consent_mode == ConsentMode.NEVER:
            return False, "Cloud access is disabled."

        if self.consent_mode == ConsentMode.ALWAYS:
            return True, None

        # ASK mode - caller should prompt user
        return True, "This request requires cloud processing. Allow cloud access?"

    def redact(self, text: str) -> RedactionResult:
        """
        Detect and redact PII from text.

        Args:
            text: Text to check for PII.

        Returns:
            RedactionResult with redacted text and list of PII matches.
        """
        if not text:
            return RedactionResult(redacted_text=text, pii_matches=[], was_redacted=False)

        redacted_text = text
        pii_matches: list[PIIMatch] = []
        offset = 0  # Track position changes due to replacements

        # Reset counter for this redaction pass
        self._placeholder_counter = 0

        for pattern_name in self._pattern_names:
            # Handle backward compatibility aliases
            actual_pattern_name = pattern_name
            if pattern_name == "phone":
                actual_pattern_name = "phone_na"  # Map old "phone" to "phone_na"

            if actual_pattern_name not in self.PATTERNS:
                logger.warning(f"Unknown PII pattern: {pattern_name}")
                continue

            pattern = self.PATTERNS[actual_pattern_name]

            for match in pattern.finditer(text):
                original = match.group()
                start = match.start() + offset
                end = match.end() + offset

                # For backward compatibility, use original pattern name in placeholder
                placeholder = self._generate_placeholder(pattern_name)
                risk_level = PII_RISK_LEVELS.get(actual_pattern_name, PIIRiskLevel.MEDIUM)
                pii_match = PIIMatch(
                    pattern_name=actual_pattern_name,
                    original=original,
                    start=start,
                    end=end,
                    placeholder=placeholder,
                    risk_level=risk_level,
                )
                pii_matches.append(pii_match)

                # Replace in redacted text
                redacted_text = redacted_text[:start] + placeholder + redacted_text[end:]

                # Update offset for subsequent replacements
                offset += len(placeholder) - len(original)

        was_redacted = len(pii_matches) > 0

        if was_redacted:
            logger.info(
                f"Redacted {len(pii_matches)} PII instances using patterns: {self._pattern_names}"
            )

        return RedactionResult(
            redacted_text=redacted_text,
            pii_matches=pii_matches,
            was_redacted=was_redacted,
        )

    def restore(self, text: str, pii_matches: list[PIIMatch]) -> str:
        """
        Restore PII placeholders with original values.

        Args:
            text: Text containing PII placeholders.
            pii_matches: List of PII matches from redaction.

        Returns:
            Text with PII restored.
        """
        if not text or not pii_matches:
            return text

        restored_text = text

        # Sort matches by start position in reverse order to avoid position shifts
        sorted_matches = sorted(pii_matches, key=lambda m: m.start, reverse=True)

        for match in sorted_matches:
            if match.placeholder in restored_text:
                restored_text = restored_text.replace(match.placeholder, match.original, 1)

        return restored_text

    async def log_cloud_request(
        self,
        original_prompt: str,
        redacted_prompt: str,
        provider: str,
        model: str,
        response_summary: str,
        pii_matches: list[PIIMatch] | None = None,
    ) -> None:
        """
        Log cloud request to file for audit.

        Args:
            original_prompt: Original user prompt (before redaction).
            redacted_prompt: Redacted prompt sent to cloud.
            provider: Cloud provider name.
            model: Model name used.
            response_summary: Summary of the response.
            pii_matches: List of PII matches that were redacted.
        """
        try:
            timestamp = datetime.now().isoformat()

            # Sanitize input for logging
            safe_prompt = redacted_prompt[:500]  # Limit length
            safe_response = response_summary[:500]  # Limit length

            # Calculate PII risk summary
            risk_summary = self._calculate_risk_summary(pii_matches) if pii_matches else {}

            log_entry = (
                f"\n{'=' * 80}\n"
                f"Timestamp: {timestamp}\n"
                f"Provider: {provider}\n"
                f"Model: {model}\n"
                f"Original Prompt: {original_prompt[:200]}...\n"
                f"Redacted Prompt: {safe_prompt}\n"
                f"Response: {safe_response}\n"
            )

            # Add PII risk summary if available
            if risk_summary:
                log_entry += "PII Risk Summary:\n"
                for risk_level, count in risk_summary.items():
                    log_entry += f"  - {risk_level}: {count} instances\n"

            log_entry += f"{'=' * 80}\n"

            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(log_entry)

            logger.debug(f"Logged cloud request to {self.log_path}")

        except Exception as e:
            logger.error(f"Failed to log cloud request: {e}")

    def _generate_placeholder(self, pattern_name: str) -> str:
        """
        Generate a unique placeholder for redacted PII.

        Args:
            pattern_name: Type of PII (email, phone, etc.)

        Returns:
            Placeholder string like [REDACTED_EMAIL_1]
        """
        self._placeholder_counter += 1
        return f"[REDACTED_{pattern_name.upper()}_{self._placeholder_counter}]"

    def add_pattern(self, name: str, pattern: str | re.Pattern, activate: bool = True) -> None:
        """
        Add a custom PII pattern.

        Args:
            name: Name for the pattern.
            pattern: Regex pattern (string or compiled).
            activate: Whether to immediately activate this pattern for redaction.
        """
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        self.PATTERNS[name] = pattern
        if activate and name not in self._pattern_names:
            self._pattern_names.append(name)
        logger.debug(f"Added custom PII pattern: {name}")

    def remove_pattern(self, name: str) -> None:
        """
        Remove a PII pattern.

        Args:
            name: Pattern name to remove.
        """
        if name in self.PATTERNS:
            del self.PATTERNS[name]
            logger.debug(f"Removed PII pattern: {name}")

    def _calculate_risk_summary(self, pii_matches: list[PIIMatch]) -> dict[str, int]:
        """
        Calculate a risk summary from PII matches.

        Args:
            pii_matches: List of PII matches.

        Returns:
            Dictionary mapping risk level to count.
        """
        risk_counts: dict[str, int] = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }

        for match in pii_matches:
            risk_level = match.risk_level.value
            if risk_level in risk_counts:
                risk_counts[risk_level] += 1

        # Remove empty levels
        return {k: v for k, v in risk_counts.items() if v > 0}

    def get_max_risk_level(self, pii_matches: list[PIIMatch]) -> PIIRiskLevel:
        """
        Get the maximum risk level from a list of PII matches.

        Args:
            pii_matches: List of PII matches.

        Returns:
            The highest risk level found.
        """
        if not pii_matches:
            return PIIRiskLevel.LOW

        risk_order = [
            PIIRiskLevel.LOW,
            PIIRiskLevel.MEDIUM,
            PIIRiskLevel.HIGH,
            PIIRiskLevel.CRITICAL,
        ]
        max_risk = PIIRiskLevel.LOW

        for match in pii_matches:
            current_index = risk_order.index(match.risk_level)
            max_index = risk_order.index(max_risk)
            if current_index > max_index:
                max_risk = match.risk_level

        return max_risk

    @classmethod
    def detect_address(cls, text: str) -> list[tuple[str, int, int]]:
        """
        Detect potential addresses in text.

        This is a simplified pattern-based approach. More sophisticated
        NLP-based detection could be added for better accuracy.

        Args:
            text: Text to search for addresses.

        Returns:
            List of (address, start, end) tuples.
        """
        # Simplified address pattern: street number + street name + city/state/zip
        # This is intentionally conservative to avoid false positives
        address_pattern = re.compile(
            r"\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Court|Ct|Place|Pl)[,\s]+(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)[,\s]+[A-Z]{2}\s+\d{5}(?:-\d{4})?\b",
            re.MULTILINE,
        )

        return [(m.group(), m.start(), m.end()) for m in address_pattern.finditer(text)]
