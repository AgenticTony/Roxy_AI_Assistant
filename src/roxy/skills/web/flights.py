"""Flight search skill for Roxy.

Searches for flights using Google Flights via Browser-Use.
Results are cached for 4 hours.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from roxy.skills.base import Permission, RoxySkill, SkillContext, SkillResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from roxy.brain.privacy import PrivacyGateway


@dataclass
class FlightOption:
    """A flight search result option."""

    airline: str
    departure_time: str
    arrival_time: str
    duration: str
    stops: str
    price: str
    origin: str
    destination: str
    url: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "airline": self.airline,
            "departure_time": self.departure_time,
            "arrival_time": self.arrival_time,
            "duration": self.duration,
            "stops": self.stops,
            "price": self.price,
            "origin": self.origin,
            "destination": self.destination,
            "url": self.url,
        }


@dataclass
class FlightSearchCache:
    """Cache entry for flight search results."""

    search_params: str  # Hash of search parameters
    results: list[FlightOption]
    timestamp: datetime
    ttl: timedelta = timedelta(hours=4)

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() > self.timestamp + self.ttl


class FlightSearchSkill(RoxySkill):
    """
    Flight search skill using Google Flights via Browser-Use.

    Features:
    - Parses flight queries for origin, destination, dates, passengers
    - Returns top 5 options sorted by price
    - 4-hour cache for results
    - PrivacyGateway for PII redaction
    - Default origin: Copenhagen (CPH) - user configurable
    """

    name: str = "flight_search"
    description: str = "Search for flights"
    triggers: list[str] = [
        "find flights",
        "cheap flights",
        "book flights",
        "flight to",
        "fly to",
        "airfare",
    ]
    permissions: list[Permission] = [Permission.NETWORK]
    requires_cloud: bool = False

    # Default settings
    DEFAULT_ORIGIN: str = "CPH"  # Copenhagen
    DEFAULT_PASSENGERS: int = 1
    DEFAULT_CLASS: str = "economy"

    # Cache
    CACHE_TTL: timedelta = timedelta(hours=4)

    def __init__(self, privacy_gateway: PrivacyGateway | None = None) -> None:
        """Initialize flight search skill.

        Args:
            privacy_gateway: PrivacyGateway for PII checks.
        """
        super().__init__()
        self.privacy_gateway = privacy_gateway
        self._cache: dict[str, FlightSearchCache] = {}

    def _parse_flight_query(self, query: str, user_preferences: dict) -> dict:
        """Parse flight search query into parameters.

        Args:
            query: User's flight search query.
            user_preferences: User preferences from memory.

        Returns:
            Dict with origin, destination, dates, passengers, class.
        """
        params = {
            "origin": user_preferences.get("home_airport", self.DEFAULT_ORIGIN),
            "destination": None,
            "departure_date": None,
            "return_date": None,
            "passengers": self.DEFAULT_PASSENGERS,
            "class": self.DEFAULT_CLASS,
        }

        # Extract destination
        # Look for "to [city]" or "flights to [city]"
        to_pattern = r"(?:to|flights? to|fly to) ([A-Za-z\s]+?)(?:\s+(?:on|from|for|departing|$|\d))"
        to_match = re.search(to_pattern, query, re.IGNORECASE)

        if to_match:
            params["destination"] = to_match.group(1).strip()
        else:
            # Fallback: look for city codes or names
            words = query.split()
            for i, word in enumerate(words):
                if word.upper() in ["LHR", "JFK", "ARN", "CDG", "FRA", "AMS", "MAD", "BCN"]:
                    params["destination"] = word.upper()
                    break
                # Check for "to" followed by city name
                if word.lower() == "to" and i + 1 < len(words):
                    params["destination"] = words[i + 1].title()
                    break

        # Extract dates
        # Look for "on [date]", "[date]", "departing [date]"
        date_pattern = r"(?:on|departing|from)\s+([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?|\d{1,2}/\d{1,2}|\d{4}-\d{2}-\d{2}|tomorrow|today|next week)"
        date_match = re.search(date_pattern, query, re.IGNORECASE)

        if date_match:
            params["departure_date"] = date_match.group(1).strip()

        # Extract passengers
        passenger_pattern = r"(\d+)\s+(?:passengers?|people|travelers?|adults?)"
        passenger_match = re.search(passenger_pattern, query, re.IGNORECASE)

        if passenger_match:
            params["passengers"] = int(passenger_match.group(1))

        # Extract class
        class_pattern = r"(business|first|economy|premium)\s+class?"
        class_match = re.search(class_pattern, query, re.IGNORECASE)

        if class_match:
            params["class"] = class_match.group(1).lower()

        return params

    def _build_google_flights_url(self, params: dict) -> str:
        """Build Google Flights search URL from parameters.

        Args:
            params: Parsed flight search parameters.

        Returns:
            Google Flights URL.
        """
        # Base URL
        url = "https://www.google.com/travel/flights"

        # Build query parameters
        query_parts = []

        # Origin and destination
        if params["origin"]:
            query_parts.append(f"flt={params['origin']}")
        if params["destination"]:
            query_parts.append(f"to={params['destination']}")

        # Dates (simplified - Google uses specific format)
        # For now, we'll let Browser-Use handle the date selection
        # by navigating to the base URL and filling in the form

        # Passengers
        if params["passengers"] > 1:
            query_parts.append(f"num={params['passengers']}")

        # Class
        if params["class"] != "economy":
            class_codes = {"business": "2", "first": "1", "premium": "3"}
            query_parts.append(f"class={class_codes.get(params['class'], '4')}")

        return url

    async def _search_flights_google(self, params: dict) -> list[FlightOption]:
        """Search Google Flights using Browser-Use.

        Args:
            params: Flight search parameters.

        Returns:
            List of FlightOption results.
        """
        try:
            from browser_use import Agent
            from langchain_openai import ChatOpenAI

            # Use local Ollama
            llm = ChatOpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
                model="qwen3:8b",
                temperature=0.3,  # Lower temperature for structured data
            )

            # Build search task
            url = self._build_google_flights_url(params)

            task = f"""Go to {url} and search for flights with these parameters:
- Origin: {params['origin']}
- Destination: {params.get('destination', 'Any')}
- Departure date: {params.get('departure_date', 'Next available')}
- Passengers: {params['passengers']}
- Class: {params['class']}

Extract the top 5 flight options sorted by price. For each option, provide:
1. Airline name
2. Departure time
3. Arrival time
4. Duration
5. Number of stops
6. Price

Format the results as a JSON array of objects with these keys: airline, departure_time, arrival_time, duration, stops, price, origin, destination.
"""

            # Create and run agent
            agent = Agent(
                task=task,
                llm=llm,
            )

            result = await asyncio.wait_for(
                agent.run(),
                timeout=60.0,
            )

            # Clean up
            await agent.browser.close()

            # Parse results
            return self._parse_flight_results(str(result), params)

        except ImportError:
            logger.warning("browser-use not available")
            return []
        except asyncio.TimeoutError:
            logger.error("Flight search timed out")
            return []
        except Exception as e:
            logger.error(f"Flight search error: {e}")
            return []

    def _parse_flight_results(self, results_text: str, params: dict) -> list[FlightOption]:
        """Parse flight results from browser output.

        Args:
            results_text: Raw results from browser agent.
            params: Original search parameters.

        Returns:
            List of FlightOption objects.
        """
        options = []

        try:
            # Try to parse as JSON
            import json

            # Extract JSON from the text
            json_pattern = r'\[[\s\S]*\]'
            json_match = re.search(json_pattern, results_text)

            if json_match:
                data = json.loads(json_match.group())

                for item in data[:5]:  # Top 5 only
                    options.append(FlightOption(
                        airline=item.get("airline", "Unknown"),
                        departure_time=item.get("departure_time", ""),
                        arrival_time=item.get("arrival_time", ""),
                        duration=item.get("duration", ""),
                        stops=item.get("stops", "Unknown"),
                        price=item.get("price", "Price unavailable"),
                        origin=params["origin"],
                        destination=params.get("destination", "Unknown"),
                    ))

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse flight results as JSON: {e}")

            # Fallback: parse text format
            options = self._parse_text_flight_results(results_text, params)

        # Sort by price (remove currency symbols and convert)
        def extract_price(price: str) -> float:
            try:
                # Extract numbers from price string
                numbers = re.findall(r'\d+\.?\d*', price.replace(",", ""))
                return float(numbers[0]) if numbers else float("inf")
            except (ValueError, IndexError):
                return float("inf")

        options.sort(key=lambda f: extract_price(f.price))

        return options[:5]

    def _parse_text_flight_results(self, text: str, params: dict) -> list[FlightOption]:
        """Parse flight results from plain text format.

        Args:
            text: Raw text results.
            params: Search parameters.

        Returns:
            List of FlightOption objects.
        """
        options = []

        # Look for patterns like:
        # "1. Airline - $XXX - Dep: XX:XX - Arr: XX:XX"
        pattern = r'(?:\d+\.|\*)\s*([A-Za-z\s]+?)\s*[-–]\s*\$?(\d+(?:\.\d+)?)\s*[-–]\s*Dep:\s*(\d{1,2}:\d{2}\s*[AP]M?)\s*[-–]\s*Arr:\s*(\d{1,2}:\d{2}\s*[AP]M?)'

        matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:
            airline = match.group(1).strip()
            price = f"${match.group(2)}"
            departure = match.group(3)
            arrival = match.group(4)

            options.append(FlightOption(
                airline=airline,
                departure_time=departure,
                arrival_time=arrival,
                duration="Unknown",
                stops="Unknown",
                price=price,
                origin=params["origin"],
                destination=params.get("destination", "Unknown"),
            ))

        return options

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute flight search.

        Args:
            context: Skill execution context.

        Returns:
            SkillResult with flight options.
        """
        user_input = context.user_input

        # Get user preferences
        user_preferences = await context.memory.get_user_preferences()

        # Parse query
        params = self._parse_flight_query(user_input, user_preferences)

        if not params["destination"]:
            return SkillResult(
                success=False,
                response_text="Where would you like to fly to?",
                follow_up="Please tell me your destination.",
            )

        # Check cache
        cache_key = f"{params['origin']}-{params.get('destination', '')}-{params.get('departure_date', '')}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if not cached.is_expired():
                logger.info(f"Flight search cache hit: {cache_key}")
                return self._format_results(cached.results, params, cached=True)

        # Redact any PII from the query before searching
        if self.privacy_gateway:
            redaction_result = self.privacy_gateway.redact(user_input)
            if redaction_result.was_redacted:
                logger.info(f"Redacted PII from flight search query")

        # Perform search
        try:
            logger.info(f"Searching flights: {params}")

            results = await self._search_flights_google(params)

            if not results:
                return SkillResult(
                    success=False,
                    response_text=f"I couldn't find any flights from {params['origin']} to {params.get('destination', 'your destination')}. Please check the airport codes or try a different search.",
                )

            # Cache results
            self._cache[cache_key] = FlightSearchCache(
                search_params=cache_key,
                results=results,
                timestamp=datetime.now(),
                ttl=self.CACHE_TTL,
            )

            return self._format_results(results, params, cached=False)

        except Exception as e:
            logger.error(f"Flight search error: {e}")
            return SkillResult(
                success=False,
                response_text=f"Sorry, I couldn't search for flights. Error: {e}",
            )

    def _format_results(
        self,
        results: list[FlightOption],
        params: dict,
        cached: bool,
    ) -> SkillResult:
        """Format flight results for user response.

        Args:
            results: List of flight options.
            params: Search parameters used.
            cached: Whether results are from cache.

        Returns:
            Formatted SkillResult.
        """
        if not results:
            return SkillResult(
                success=False,
                response_text="No flights found for your search.",
            )

        # Build response text
        lines = [
            f"Found {len(results)} flight option{'s' if len(results) > 1 else ''} from {params['origin']} to {params.get('destination', 'your destination')}:",
            "",
        ]

        for i, flight in enumerate(results, 1):
            lines.append(f"{i}. {flight.airline}")
            lines.append(f"   {flight.departure_time} → {flight.arrival_time} ({flight.duration})")
            lines.append(f"   {flight.stops} | {flight.price}")
            lines.append("")

        if cached:
            lines.append("(Results from cache - may not reflect live pricing)")

        return SkillResult(
            success=True,
            response_text="\n".join(lines),
            data={
                "flights": [f.to_dict() for f in results],
                "search_params": params,
                "cached": cached,
            },
        )
