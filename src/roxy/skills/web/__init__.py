"""Web access skills for Roxy.

This package contains skills that provide web access capabilities:
- WebSearchSkill: Search the web using Brave/SearXNG
- BrowseSkill: Browse websites using Browser-Use
- FlightSearchSkill: Search for flights using Google Flights
"""

from roxy.skills.web.browse import BrowseSkill
from roxy.skills.web.flights import FlightSearchSkill
from roxy.skills.web.search import WebSearchSkill

__all__ = [
    "WebSearchSkill",
    "BrowseSkill",
    "FlightSearchSkill",
]
