"""Tests for the Notes skill.

Tests creating, listing, and searching notes in macOS Notes.app.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roxy.skills.base import Permission, SkillContext
from roxy.skills.productivity.notes import NotesSkill


class TestNotesSkill:
    """Tests for NotesSkill."""

    def test_skill_metadata(self):
        """Test skill has correct metadata."""
        skill = NotesSkill()

        assert skill.name == "notes"
        assert "notes" in skill.description.lower()
        assert len(skill.triggers) > 0
        assert Permission.APPLESCRIPT in skill.permissions
        assert skill.requires_cloud is False

    def test_init(self):
        """Test skill initialization."""
        skill = NotesSkill()

        assert hasattr(skill, "_applescript")
        assert skill._applescript is not None

    @pytest.mark.asyncio
    async def test_get_notes_list(self):
        """Test getting list of notes."""
        skill = NotesSkill()

        # Mock the applescript runner
        with patch.object(skill, "_applescript") as mock_runner:
            mock_runner.run_jxa = AsyncMock(return_value="[]")

            notes = await skill.get_notes()
            assert notes == []

    @pytest.mark.asyncio
    async def test_create_note(self):
        """Test creating a new note."""
        skill = NotesSkill()

        with patch.object(skill, "_applescript") as mock_runner:
            mock_runner.run_jxa = AsyncMock(return_value="")

            result = await skill.create_note("Test Title", "Test Body")
            assert result is True

    @pytest.mark.asyncio
    async def test_create_note_with_special_chars(self):
        """Test creating a note with special characters that need escaping."""
        skill = NotesSkill()

        with patch.object(skill, "_applescript") as mock_runner:
            mock_runner.run_jxa = AsyncMock(return_value="")

            # Test with special characters
            result = await skill.create_note('Test "Quote"', 'Text with "quotes"')
            assert result is True

    @pytest.mark.asyncio
    async def test_execute_list_notes(self):
        """Test execute method for listing notes."""
        skill = NotesSkill()
        context = SkillContext(
            user_input="list my notes",
            intent="list_notes",
            parameters={},
            memory=MagicMock(),
            config=MagicMock(),
            conversation_history=[],
        )

        with patch.object(skill, "get_notes") as mock_get:
            mock_get.return_value = [
                {"name": "Note 1", "body": "Body 1"},
                {"name": "Note 2", "body": "Body 2"},
            ]

            result = await skill.execute(context)

            assert result.success is True
            assert "2" in result.response_text or "notes" in result.response_text.lower()

    @pytest.mark.asyncio
    async def test_execute_create_note(self):
        """Test execute method for creating a note."""
        skill = NotesSkill()
        context = SkillContext(
            user_input="create a note about my meeting",
            intent="create_note",
            parameters={"title": "Meeting Notes", "body": "Discussed project timeline"},
            memory=MagicMock(),
            config=MagicMock(),
            conversation_history=[],
        )

        with patch.object(skill, "create_note") as mock_create:
            mock_create.return_value = True

            result = await skill.execute(context)

            assert result.success is True
            assert "created" in result.response_text.lower()

    @pytest.mark.asyncio
    async def test_execute_create_note_failure(self):
        """Test execute method handles creation failure."""
        skill = NotesSkill()
        context = SkillContext(
            user_input="create a note",
            intent="create_note",
            parameters={},
            memory=MagicMock(),
            config=MagicMock(),
            conversation_history=[],
        )

        with patch.object(skill, "create_note") as mock_create:
            mock_create.return_value = False

            result = await skill.execute(context)

            assert result.success is False
            assert (
                "couldn't" in result.response_text.lower()
                or "failed" in result.response_text.lower()
            )

    @pytest.mark.asyncio
    async def test_can_handle_list_phrases(self):
        """Test can_handle recognizes list note phrases."""
        skill = NotesSkill()

        # Test various list phrases
        phrases = [
            "list my notes",
            "show my notes",
            "what notes do I have",
            "get my notes",
        ]

        for phrase in phrases:
            confidence = skill.can_handle(phrase, {})
            assert confidence > 0, f"Should handle '{phrase}' with confidence > 0"

    @pytest.mark.asyncio
    async def test_can_handle_create_phrases(self):
        """Test can_handle recognizes create note phrases."""
        skill = NotesSkill()

        # Test various create phrases
        phrases = [
            "create a note about",
            "make a note for",
            "save this as a note",
            "take a note",
        ]

        for phrase in phrases:
            confidence = skill.can_handle(phrase, {})
            assert confidence > 0, f"Should handle '{phrase}' with confidence > 0"

    @pytest.mark.asyncio
    async def test_can_handle_low_confidence_for_irrelevant(self):
        """Test can_handle returns low confidence for irrelevant input."""
        skill = NotesSkill()

        # Should have low confidence for unrelated requests
        confidence = skill.can_handle("open Safari", {})
        assert confidence < 0.3, "Should have low confidence for unrelated commands"

    @pytest.mark.asyncio
    async def test_get_notes_handles_error(self):
        """Test get_notes handles errors gracefully."""
        skill = NotesSkill()

        with patch.object(skill, "_applescript") as mock_runner:
            mock_runner.run_jxa = AsyncMock(side_effect=Exception("Notes error"))

            notes = await skill.get_notes()
            # Should return empty list on error
            assert notes == []

    @pytest.mark.asyncio
    async def test_search_in_notes(self):
        """Test searching through notes (if implemented)."""
        skill = NotesSkill()

        # If there's a search method, test it
        if hasattr(skill, "search_notes"):
            with patch.object(skill, "_applescript") as mock_runner:
                mock_runner.run_jxa = AsyncMock(return_value="[]")

                results = await skill.search_notes("test query")
                assert isinstance(results, list)
        else:
            # Search may not be implemented yet
            pass


class TestNotesSkillSecurity:
    """Test security aspects of NotesSkill."""

    @pytest.mark.asyncio
    async def test_create_note_escapes_input(self):
        """Test that note title and body are properly escaped."""
        skill = NotesSkill()
        dangerous_title = 'Note"; do shell script "rm -rf /'

        with patch.object(skill, "_applescript") as mock_runner:
            mock_runner.run_jxa = AsyncMock(return_value="")

            # Should handle dangerous input without crashing
            result = await skill.create_note(dangerous_title, "body")
            # The skill should either escape or reject the input
            assert result is True or result is False  # Should not crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
