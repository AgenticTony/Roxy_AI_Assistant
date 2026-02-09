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
from typing import Literal

logger = logging.getLogger(__name__)


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

    def __str__(self) -> str:
        return f"PIIMatch({self.pattern_name}, {self.placeholder})"


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
        "phone": re.compile(
            r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        ),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "credit_card": re.compile(
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        ),
        "ip_address": re.compile(
            r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
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
            if pattern_name not in self.PATTERNS:
                logger.warning(f"Unknown PII pattern: {pattern_name}")
                continue

            pattern = self.PATTERNS[pattern_name]

            for match in pattern.finditer(text):
                original = match.group()
                start = match.start() + offset
                end = match.end() + offset

                placeholder = self._generate_placeholder(pattern_name)
                pii_match = PIIMatch(
                    pattern_name=pattern_name,
                    original=original,
                    start=start,
                    end=end,
                    placeholder=placeholder,
                )
                pii_matches.append(pii_match)

                # Replace in redacted text
                redacted_text = redacted_text[:start] + placeholder + redacted_text[end:]

                # Update offset for subsequent replacements
                offset += len(placeholder) - len(original)

        was_redacted = len(pii_matches) > 0

        if was_redacted:
            logger.info(f"Redacted {len(pii_matches)} PII instances using patterns: {self._pattern_names}")

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
    ) -> None:
        """
        Log cloud request to file for audit.

        Args:
            original_prompt: Original user prompt (before redaction).
            redacted_prompt: Redacted prompt sent to cloud.
            provider: Cloud provider name.
            model: Model name used.
            response_summary: Summary of the response.
        """
        try:
            timestamp = datetime.now().isoformat()

            # Sanitize input for logging
            safe_prompt = redacted_prompt[:500]  # Limit length
            safe_response = response_summary[:500]  # Limit length

            log_entry = (
                f"\n{'='*80}\n"
                f"Timestamp: {timestamp}\n"
                f"Provider: {provider}\n"
                f"Model: {model}\n"
                f"Original Prompt: {original_prompt[:200]}...\n"
                f"Redacted Prompt: {safe_prompt}\n"
                f"Response: {safe_response}\n"
                f"{'='*80}\n"
            )

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

    def add_pattern(self, name: str, pattern: str | re.Pattern) -> None:
        """
        Add a custom PII pattern.

        Args:
            name: Name for the pattern.
            pattern: Regex pattern (string or compiled).
        """
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        self.PATTERNS[name] = pattern
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