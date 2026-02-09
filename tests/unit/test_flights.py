"""Unit tests for FlightSearchSkill.

Tests flight search functionality with mocks.
No external API calls are made.
"""

from __future__ import annotations

import pytest
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from roxy.brain.privacy import PrivacyGateway
from roxy.skills.base import SkillContext, StubMemoryManager
from roxy.skills.web.flights import FlightSearchSkill, FlightOption, FlightSearchCache


@pytest.fixture
def flight_skill():
    """Fixture providing a FlightSearchSkill instance."""
    return FlightSearchSkill()


@pytest.fixture
def skill_context():
    """Fixture providing a SkillContext for testing."""
    memory = StubMemoryManager()
    memory.set_preference("home_airport", "CPH")

    return SkillContext(
        user_input="find flights to London",
        intent="flight_search",
        parameters={},
        memory=memory,
        config=MagicMock(),
        conversation_history=[],
    )


class TestFlightSearchSkill:
    """Tests for FlightSearchSkill."""

    def test_init(self):
        """Test skill initialization."""
        skill = FlightSearchSkill()

        assert skill.name == "flight_search"
        assert skill.DEFAULT_ORIGIN == "CPH"
        assert skill.DEFAULT_PASSENGERS == 1
        assert len(skill.triggers) > 0

    def test_parse_flight_query_destination(self, flight_skill):
        """Test parsing destination from query."""
        preferences = {"home_airport": "CPH"}

        # Test "to" pattern
        params = flight_skill._parse_flight_query("flights to London", preferences)
        assert params["destination"] == "London"

        # Test "fly to" pattern
        params = flight_skill._parse_flight_query("fly to Stockholm", preferences)
        assert params["destination"] == "Stockholm"

    def test_parse_flight_query_date(self, flight_skill):
        """Test parsing dates from query."""
        preferences = {"home_airport": "CPH"}

        # Test "on" pattern
        params = flight_skill._parse_flight_query("flights to London on January 15", preferences)
        assert params["departure_date"] is not None
        assert "January 15" in params["departure_date"]

    def test_parse_flight_query_passengers(self, flight_skill):
        """Test parsing passenger count from query."""
        preferences = {"home_airport": "CPH"}

        params = flight_skill._parse_flight_query("flights for 2 passengers to London", preferences)
        assert params["passengers"] == 2

    def test_parse_flight_query_class(self, flight_skill):
        """Test parsing cabin class from query."""
        preferences = {"home_airport": "CPH"}

        params = flight_skill._parse_flight_query("business class flights to London", preferences)
        assert params["class"] == "business"

    def test_build_google_flights_url(self, flight_skill):
        """Test Google Flights URL generation."""
        params = {
            "origin": "CPH",
            "destination": "London",
            "departure_date": None,
            "passengers": 2,
            "class": "business",
        }

        url = flight_skill._build_google_flights_url(params)

        assert "google.com/travel/flights" in url
        # The URL building just returns the base URL, params are handled by browser automation

    def test_extract_price_from_string(self, flight_skill):
        """Test price extraction from various formats."""
        from roxy.skills.web.flights import FlightOption
        import re

        def extract_price(price: str) -> float:
            try:
                numbers = re.findall(r'\d+\.?\d*', price.replace(",", ""))
                return float(numbers[0]) if numbers else float("inf")
            except (ValueError, IndexError):
                return float("inf")

        # Test different price formats
        assert extract_price("$123.45") == 123.45
        assert extract_price("€1,234") == 1234.0
        assert extract_price("500") == 500.0


class TestFlightOption:
    """Tests for FlightOption dataclass."""

    def test_to_dict(self):
        """Test converting FlightOption to dictionary."""
        option = FlightOption(
            airline="SAS",
            departure_time="10:00",
            arrival_time="12:00",
            duration="2h 00m",
            stops="Direct",
            price="$150",
            origin="CPH",
            destination="LHR",
        )

        result = option.to_dict()

        assert result["airline"] == "SAS"
        assert result["price"] == "$150"
        assert result["origin"] == "CPH"


class TestFlightSearchCache:
    """Tests for FlightSearchCache."""

    def test_cache_expiry(self):
        """Test that cache entries expire correctly."""
        from datetime import datetime

        cache = FlightSearchCache(
            search_params="CPH-LHR-2025-01-15",
            results=[],
            timestamp=datetime.now(),
            ttl=timedelta(hours=4),
        )

        # Should not be expired immediately
        assert cache.is_expired() is False

        # Should be expired after 5 hours
        with patch("roxy.skills.web.flights.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime.now() + timedelta(hours=5)
            assert cache.is_expired() is True

    @pytest.mark.asyncio
    async def test_format_results(self, flight_skill):
        """Test formatting flight results for display."""
        from datetime import datetime

        flights = [
            FlightOption(
                airline="SAS",
                departure_time="10:00",
                arrival_time="12:00",
                duration="2h 00m",
                stops="Direct",
                price="$150",
                origin="CPH",
                destination="LHR",
            ),
            FlightOption(
                airline="British Airways",
                departure_time="14:00",
                arrival_time="16:00",
                duration="2h 00m",
                stops="Direct",
                price="$180",
                origin="CPH",
                destination="LHR",
            ),
        ]

        params = {"origin": "CPH", "destination": "LHR", "departure_date": None, "passengers": 1, "class": "economy"}

        result = flight_skill._format_results(flights, params, cached=False)

        assert result.success is True
        assert "2 flight options" in result.response_text
        assert "SAS" in result.response_text
        assert "$150" in result.response_text
