"""Unit tests for CalendarSkill.

Tests calendar functionality with AppleScript mocks.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from roxy.macos.applescript import AppleScriptRunner
from roxy.skills.base import SkillContext, StubMemoryManager
from roxy.skills.productivity.calendar import CalendarSkill


@pytest.fixture
def mock_applescript():
    """Fixture providing a mock AppleScriptRunner."""
    mock = MagicMock(spec=AppleScriptRunner)

    # Mock availability check
    mock._check_available = MagicMock(return_value=True)
    mock._available = True

    return mock


@pytest.fixture
def calendar_skill(mock_applescript):
    """Fixture providing a CalendarSkill with mocked AppleScript."""
    skill = CalendarSkill()
    skill._applescript = mock_applescript
    return skill


@pytest.fixture
def skill_context():
    """Fixture providing a SkillContext for testing."""
    return SkillContext(
        user_input="what's on my calendar today",
        intent="calendar",
        parameters={},
        memory=StubMemoryManager(),
        config=MagicMock(),
        conversation_history=[],
    )


class TestCalendarSkill:
    """Tests for CalendarSkill."""

    def test_init(self):
        """Test skill initialization."""
        skill = CalendarSkill()

        assert skill.name == "calendar"
        assert len(skill.triggers) > 0
        assert "what's on my calendar" in skill.triggers

    def test_parse_date_today(self, calendar_skill):
        """Test parsing 'today' from input."""
        date_str = calendar_skill._parse_date("what's on my calendar today")

        assert "today" in date_str.lower() or "current date" in date_str.lower()

    def test_parse_date_tomorrow(self, calendar_skill):
        """Test parsing 'tomorrow' from input."""
        date_str = calendar_skill._parse_date("what's on my calendar tomorrow")

        assert "tomorrow" in date_str.lower() or "1" in date_str

    def test_parse_date_this_week(self, calendar_skill):
        """Test parsing 'this week' from input."""
        date_str = calendar_skill._parse_date("what's on my calendar this week")

        assert "week" in date_str.lower() or "current date" in date_str.lower()

    @pytest.mark.asyncio
    async def test_get_events_with_results(self, calendar_skill, mock_applescript):
        """Test getting events when events exist."""
        # Mock AppleScript response - format from listToString is comma-separated, events separated by |||
        mock_applescript.run.return_value = "Team Standup, Monday January 15 2024 at 9:00:00 AM, Monday January 15 2024 at 9:30:00 AM, Conference Room A|||Lunch with Sarah, Monday January 15 2024 at 12:00:00 PM, Monday January 15 2024 at 1:00:00 PM,"

        events = await calendar_skill.get_events()

        assert len(events) == 2
        assert events[0]["summary"] == "Team Standup"
        assert events[0]["location"] == "Conference Room A"
        assert events[1]["summary"] == "Lunch with Sarah"

    @pytest.mark.asyncio
    async def test_get_events_empty(self, calendar_skill, mock_applescript):
        """Test getting events when no events exist."""
        mock_applescript.run.return_value = ""

        events = await calendar_skill.get_events()

        assert events == []

    @pytest.mark.asyncio
    async def test_create_event(self, calendar_skill, mock_applescript):
        """Test creating a new event."""
        mock_applescript.run.return_value = ""  # Success returns empty

        success = await calendar_skill.create_event(
            title="Test Event",
            start_date="date (current date)",
            end_date="date (current date) + 1 * hours",
            location="Office",
        )

        assert success is True
        mock_applescript.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_get_events(self, calendar_skill, mock_applescript, skill_context):
        """Test executing skill to get events."""
        # Mock empty calendar
        mock_applescript.run.return_value = ""

        result = await calendar_skill.execute(skill_context)

        assert result.success is True
        assert "today" in result.response_text.lower() or "event" in result.response_text.lower()

    @pytest.mark.asyncio
    async def test_execute_create_event(self, calendar_skill, mock_applescript):
        """Test executing skill to create an event."""
        mock_applescript.run.return_value = ""

        context = SkillContext(
            user_input="create event called Team Meeting",
            intent="calendar",
            parameters={"title": "Team Meeting", "start_date": "date (current date)"},
            memory=StubMemoryManager(),
            config=MagicMock(),
            conversation_history=[],
        )

        result = await calendar_skill.execute(context)

        assert result.success is True
        assert "Team Meeting" in result.response_text

    @pytest.mark.asyncio
    async def test_execute_with_events_found(self, calendar_skill, mock_applescript, skill_context):
        """Test executing skill when events are found."""
        # Mock AppleScript with events
        mock_applescript.run.return_value = (
            "Team Standup, Monday at 9:00 AM, Monday at 9:30 AM, Room A|||"
        )

        result = await calendar_skill.execute(skill_context)

        assert result.success is True
        assert "1 event" in result.response_text or "Team Standup" in result.response_text
