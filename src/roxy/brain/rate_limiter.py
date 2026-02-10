"""Persistent rate limiting for Roxy's cloud API calls.

Implements file-based rate limit tracking that persists across restarts.
Tracks requests per minute and per hour with automatic cleanup of old records.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class RateLimitRecord:
    """A single API request record."""

    timestamp: float
    provider: str
    model: str
    success: bool
    status_code: int | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "provider": self.provider,
            "model": self.model,
            "success": self.success,
            "status_code": self.status_code,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RateLimitRecord:
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            provider=data["provider"],
            model=data["model"],
            success=data["success"],
            status_code=data.get("status_code"),
        )


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    # Limits (configurable per provider if needed)
    requests_per_minute: int = 60
    requests_per_hour: int = 1000

    # Storage settings
    storage_path: str = "data/rate_limits.json"

    # Cleanup settings
    cleanup_interval_seconds: int = 300  # Clean up old records every 5 minutes
    retention_hours: int = 24  # Keep records for 24 hours


class RateLimiter:
    """
    Persistent rate limiter for cloud API calls.

    Tracks request timestamps in a JSON file that persists across restarts.
    Thread-safe for concurrent access.
    """

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        """Initialize rate limiter.

        Args:
            config: Rate limit configuration. If None, uses defaults.
        """
        self._config = config or RateLimitConfig()
        self._lock = Lock()
        self._records: list[RateLimitRecord] = []
        self._last_cleanup = 0.0

        # Ensure storage directory exists
        self._storage_path = Path(self._config.storage_path).expanduser()
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing records on startup
        self._load_records()

    def _load_records(self) -> None:
        """Load existing rate limit records from storage."""
        if not self._storage_path.exists():
            logger.debug(f"No existing rate limit data at {self._storage_path}")
            return

        try:
            with self._storage_path.open("r") as f:
                data = json.load(f)

            self._records = [RateLimitRecord.from_dict(r) for r in data.get("records", [])]

            # Clean up old records on load
            self._cleanup_old_records()

            logger.debug(
                f"Loaded {len(self._records)} rate limit records from {self._storage_path}"
            )

        except (OSError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load rate limit data: {e}. Starting fresh.")
            self._records = []

    def _save_records(self) -> None:
        """Save current records to storage."""
        try:
            data = {
                "last_updated": time.time(),
                "records": [r.to_dict() for r in self._records],
            }

            # Write to temp file first, then atomic rename
            temp_path = self._storage_path.with_suffix(".tmp")
            with temp_path.open("w") as f:
                json.dump(data, f, indent=2)

            temp_path.replace(self._storage_path)

        except OSError as e:
            logger.error(f"Failed to save rate limit data: {e}")

    def _cleanup_old_records(self) -> None:
        """Remove records older than retention period."""
        cutoff_time = time.time() - (self._config.retention_hours * 3600)

        original_count = len(self._records)
        self._records = [r for r in self._records if r.timestamp >= cutoff_time]

        removed = original_count - len(self._records)
        if removed > 0:
            logger.debug(f"Cleaned up {removed} old rate limit records")

        self._last_cleanup = time.time()

    def _maybe_cleanup(self) -> None:
        """Run cleanup if enough time has passed."""
        now = time.time()
        if now - self._last_cleanup >= self._config.cleanup_interval_seconds:
            self._cleanup_old_records()
            self._save_records()

    def _count_requests_in_window(
        self,
        provider: str,
        window_seconds: int,
    ) -> int:
        """Count requests from a provider within the time window."""
        cutoff_time = time.time() - window_seconds

        return sum(
            1 for r in self._records if r.provider == provider and r.timestamp >= cutoff_time
        )

    def check_rate_limit(
        self,
        provider: str,
    ) -> tuple[bool, str]:
        """
        Check if a request is allowed under rate limits.

        Args:
            provider: The cloud provider name (e.g., "zai", "openrouter").

        Returns:
            Tuple of (allowed: bool, message: str).
            If allowed is True, the request can proceed.
            If allowed is False, the message explains the limit.
        """
        with self._lock:
            # Periodic cleanup
            self._maybe_cleanup()

            # Check per-minute limit
            minute_count = self._count_requests_in_window(provider, 60)
            if minute_count >= self._config.requests_per_minute:
                return (
                    False,
                    f"Rate limit exceeded: {minute_count} requests in the last minute "
                    f"(limit: {self._config.requests_per_minute}/min)",
                )

            # Check per-hour limit
            hour_count = self._count_requests_in_window(provider, 3600)
            if hour_count >= self._config.requests_per_hour:
                return (
                    False,
                    f"Rate limit exceeded: {hour_count} requests in the last hour "
                    f"(limit: {self._config.requests_per_hour}/hour)",
                )

            return (True, "Rate limit OK")

    def record_request(
        self,
        provider: str,
        model: str,
        success: bool,
        status_code: int | None = None,
    ) -> None:
        """
        Record an API request.

        Args:
            provider: The cloud provider name.
            model: The model used.
            success: Whether the request succeeded.
            status_code: HTTP status code if applicable.
        """
        with self._lock:
            record = RateLimitRecord(
                timestamp=time.time(),
                provider=provider,
                model=model,
                success=success,
                status_code=status_code,
            )
            self._records.append(record)

            # Save after each request (could be batched for performance)
            self._save_records()

    def get_stats(self, provider: str) -> dict:
        """
        Get rate limit statistics for a provider.

        Args:
            provider: The cloud provider name.

        Returns:
            Dictionary with current usage statistics.
        """
        with self._lock:
            now = time.time()

            return {
                "provider": provider,
                "requests_last_minute": self._count_requests_in_window(provider, 60),
                "requests_last_hour": self._count_requests_in_window(provider, 3600),
                "requests_today": self._count_requests_in_window(provider, 86400),
                "limit_per_minute": self._config.requests_per_minute,
                "limit_per_hour": self._config.requests_per_hour,
                "total_records": len(self._records),
            }

    def reset(self, provider: str | None = None) -> None:
        """
        Reset rate limit records.

        Args:
            provider: If specified, only reset records for this provider.
                     If None, reset all records.
        """
        with self._lock:
            if provider:
                self._records = [r for r in self._records if r.provider != provider]
                logger.info(f"Reset rate limit records for provider: {provider}")
            else:
                self._records = []
                logger.info("Reset all rate limit records")

            self._save_records()

    def get_failed_requests(
        self,
        provider: str,
        hours: int = 1,
    ) -> list[RateLimitRecord]:
        """
        Get failed requests for a provider within the time window.

        Args:
            provider: The cloud provider name.
            hours: Lookback period in hours.

        Returns:
            List of failed request records.
        """
        with self._lock:
            cutoff_time = time.time() - (hours * 3600)

            return [
                r
                for r in self._records
                if r.provider == provider and not r.success and r.timestamp >= cutoff_time
            ]


class RateLimiterAware:
    """
    Mixin for classes that need rate limiting support.

    Provides common rate limiting methods for cloud client implementations.
    """

    def __init__(self, rate_limiter: RateLimiter | None = None) -> None:
        """Initialize with optional rate limiter.

        Args:
            rate_limiter: Rate limiter instance. If None, creates a default one.
        """
        self._rate_limiter = rate_limiter or RateLimiter()

    @property
    def rate_limiter(self) -> RateLimiter:
        """Get the rate limiter instance."""
        return self._rate_limiter

    def check_rate_limit(self, provider: str) -> tuple[bool, str]:
        """Delegate to rate limiter."""
        return self._rate_limiter.check_rate_limit(provider)

    def record_request(
        self,
        provider: str,
        model: str,
        success: bool,
        status_code: int | None = None,
    ) -> None:
        """Delegate to rate limiter."""
        self._rate_limiter.record_request(provider, model, success, status_code)
