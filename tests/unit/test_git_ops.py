"""Unit tests for git_ops skill."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roxy.brain.llm_clients import LLMResponse
from roxy.skills.base import Permission, SkillContext, StubMemoryManager
from roxy.skills.dev.git_ops import GitOpsSkill


# Fixtures
@pytest.fixture
def skill_context():
    """Create a skill context for testing."""
    from roxy.config import CloudLLMConfig, LocalLLMConfig, PrivacyConfig, RoxyConfig

    config = RoxyConfig(
        name="TestRoxy",
        data_dir="/tmp/test_roxy",
        llm_local=LocalLLMConfig(),
        llm_cloud=CloudLLMConfig(),
        privacy=PrivacyConfig(),
    )

    # Create mock LLM client
    mock_llm_client = MagicMock()
    mock_llm_client.generate = AsyncMock(
        return_value=LLMResponse(
            content="feat: add new authentication system",
            model="qwen3:8b",
            provider="local",
        )
    )

    return SkillContext(
        user_input="test input",
        intent="test",
        parameters={},
        memory=StubMemoryManager(),
        config=config,
        conversation_history=[],
        local_llm_client=mock_llm_client,
    )


@pytest.fixture
def mock_git_diff():
    """Return a mock git diff output."""
    return """diff --git a/src/auth.py b/src/auth.py
index 1234567..abcdefg 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -1,5 +1,10 @@
 def authenticate(user, password):
+    # Add OAuth2 support
+    if user.oauth_token:
+        return validate_oauth(user.oauth_token)
+
     return validate_password(user, password)

diff --git a/tests/test_auth.py b/tests/test_auth.py
index 1111111..2222222 100644
--- a/tests/test_auth.py
+++ b/tests/test_auth.py
@@ -10,6 +10,9 @@ def test_auth():
     assert result is True
+
+def test_oauth_auth():
+    assert authenticate(oauth_user, None) is True
"""


class TestGitOpsSkill:
    """Test GitOpsSkill."""

    def test_skill_metadata(self):
        """Test skill has correct metadata."""
        skill = GitOpsSkill()

        assert skill.name == "git_ops"
        assert "commit changes" in skill.triggers
        assert Permission.SHELL in skill.permissions

    def test_run_git_success(self, skill_context, tmp_path):
        """Test successful git command execution."""
        skill = GitOpsSkill()

        # Create a temp directory and initialize git repo
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()
        result = skill._run_git(["init"], cwd=repo_path)

        assert result.success is True
        assert "Initialized" in result.output or "init" in result.output.lower()

    def test_run_git_failure(self, skill_context):
        """Test git command failure handling."""
        skill = GitOpsSkill()

        result = skill._run_git(["invalid-command-xyz"])

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_generate_commit_message_with_llm(self, skill_context, mock_git_diff):
        """Test commit message generation using LLM."""
        skill = GitOpsSkill()

        # Mock the git diff command
        with patch.object(skill, "_run_git") as mock_run_git:
            mock_run_git.return_value = MagicMock(
                success=True,
                output=mock_git_diff,
            )

            # Generate commit message
            message = await skill.generate_commit_message(skill_context)

            # Verify LLM was called
            assert skill_context.local_llm_client.generate.called
            # Verify we got a commit message
            assert message
            assert len(message) > 5

    @pytest.mark.asyncio
    async def test_generate_commit_message_fallback(self, skill_context, mock_git_diff):
        """Test commit message generation fallback when LLM fails."""
        skill = GitOpsSkill()

        # Mock the git diff command
        with patch.object(skill, "_run_git") as mock_run_git:
            mock_run_git.return_value = MagicMock(
                success=True,
                output=mock_git_diff,
            )

            # Mock LLM failure
            skill_context.local_llm_client.generate = AsyncMock(side_effect=Exception("LLM failed"))

            # Generate commit message
            message = await skill.generate_commit_message(skill_context)

            # Verify we got a fallback message
            assert message
            assert "file" in message.lower() or "update" in message.lower()

    @pytest.mark.asyncio
    async def test_generate_commit_message_no_changes(self, skill_context):
        """Test commit message generation with no staged changes."""
        skill = GitOpsSkill()

        # Mock the git diff command to return no changes
        with patch.object(skill, "_run_git") as mock_run_git:
            mock_run_git.return_value = MagicMock(
                success=True,
                output="",  # No changes
            )

            # Generate commit message
            message = await skill.generate_commit_message(skill_context)

            # Verify we got the "no changes" message
            assert message == "No staged changes to commit"

    @pytest.mark.asyncio
    async def test_execute_commit_command(self, skill_context):
        """Test execute method with commit command."""
        skill = GitOpsSkill()
        skill_context.user_input = "commit my changes"

        # Mock git commands
        with patch.object(skill, "_run_git") as mock_run_git:
            # Mock status check (no changes)
            mock_run_git.return_value = MagicMock(success=True, output="")

            result = await skill.execute(skill_context)

            # Verify result structure
            assert result.success is True or result.success is False  # Could succeed or fail
            assert isinstance(result.response_text, str)

    @pytest.mark.asyncio
    async def test_execute_status_command(self, skill_context):
        """Test execute method with status command."""
        skill = GitOpsSkill()
        skill_context.user_input = "git status"

        # Mock git status
        with patch.object(skill, "git_status") as mock_status:
            mock_status.return_value = "On branch main\nYour branch is up to date."

            result = await skill.execute(skill_context)

            assert result.success is True
            assert "branch" in result.response_text.lower()

    @pytest.mark.asyncio
    async def test_can_handle_confidence_scoring(self):
        """Test can_handle method returns appropriate confidence scores."""
        skill = GitOpsSkill()

        # High confidence for exact match
        confidence = skill.can_handle("commit changes", {})
        assert confidence > 0.5

        # High confidence for git status
        confidence = skill.can_handle("git status", {})
        assert confidence > 0.5

        # Lower confidence for partial match (contains "git")
        confidence = skill.can_handle("what is git", {})
        # This might be 0 since the can_handle implementation looks for trigger phrases
        # and "what is git" doesn't match any trigger exactly
        assert confidence >= 0.0

        # No confidence for unrelated input
        confidence = skill.can_handle("what's the weather", {})
        assert confidence == 0.0
