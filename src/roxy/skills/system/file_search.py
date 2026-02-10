"""File search skill using macOS Spotlight.

Provides fast file and metadata search capabilities.
"""

from __future__ import annotations

import logging
import re

from roxy.skills.base import Permission, RoxySkill, SkillContext, SkillResult

logger = logging.getLogger(__name__)


class FileSearchSkill(RoxySkill):
    """Skill for searching files using Spotlight."""

    name = "file_search"
    description = "Search for files using macOS Spotlight"
    triggers = [
        "find file",
        "search for file",
        "look for",
        "open the file",
        "find document",
        "search files",
    ]
    permissions = [Permission.FILESYSTEM]
    requires_cloud = False

    async def execute(self, context: SkillContext) -> SkillResult:
        """
        Execute the file search skill.

        Args:
            context: Skill execution context.

        Returns:
            SkillResult with search results.
        """
        user_input = context.user_input
        parameters = context.parameters

        # Extract search query and options
        query = self._extract_query(user_input, parameters)
        kind = parameters.get("kind")
        recent_days = parameters.get("recent_days")
        limit = parameters.get("limit", 10)

        if not query:
            return SkillResult(
                success=False,
                response_text="What would you like me to search for?",
                speak=False,
            )

        logger.info(f"Searching for files: query='{query}', kind={kind}, recent={recent_days}")

        try:
            from roxy.macos.spotlight import get_spotlight_search

            spotlight = get_spotlight_search()

            # Perform search based on parameters
            if recent_days:
                results = await spotlight.search_recent(query, days=recent_days, limit=limit)
            elif kind:
                results = await spotlight.search_by_kind(query, kind=kind, limit=limit)
            else:
                results = await spotlight.search(query, limit=limit)

            if not results:
                return SkillResult(
                    success=True,
                    response_text=f"No files found matching '{query}'",
                    speak=True,
                    data={"results": []},
                )

            # Check if user wants to open a file
            should_open = self._should_open_file(user_input)

            if should_open and results:
                # Open the first result
                first_file = results[0]
                success = await self._open_file(first_file["path"])

                if success:
                    response = f"Opening {first_file['name']}"
                else:
                    response = f"Found {len(results)} files, but couldn't open {first_file['name']}"
            else:
                # Just return results
                count = len(results)
                response = f"Found {count} file{'s' if count != 1 else ''} matching '{query}'"

                # List first few results
                if results:
                    response += ":"
                    for i, result in enumerate(results[:3], 1):
                        response += f"\n{i}. {result['name']} ({result['kind']})"
                    if count > 3:
                        response += f"\n... and {count - 3} more"

            return SkillResult(
                success=True,
                response_text=response,
                speak=True,
                data={"results": results, "count": len(results)},
            )

        except Exception as e:
            logger.error(f"Error searching files: {e}")
            return SkillResult(
                success=False,
                response_text=f"Sorry, I encountered an error searching for files: {e}",
                speak=True,
            )

    def _extract_query(self, user_input: str, parameters: dict) -> str | None:
        """
        Extract search query from user input.

        Args:
            user_input: User's input string.
            parameters: Extracted parameters from LLM.

        Returns:
            Search query or None if not found.
        """
        # First check if LLM extracted a query parameter
        if "query" in parameters:
            return parameters["query"]

        # Try to extract query from common patterns

        # Pattern: "find file about [topic]", "search for [query]", "look for [query]"
        patterns = [
            r"find\s+(?:file|document)\s+(?:about|containing|named)?\s+['\"]?(.+?)['\"]?(?:\s|$|\?)",
            r"search\s+(?:for\s+)?(?:files?\s+)?(?:about|containing|named)?\s+['\"]?(.+?)['\"]?(?:\s|$|\?)",
            r"look\s+for\s+['\"]?(.+?)['\"]?(?:\s|$|\?)",
            r"open\s+(?:the\s+)?(?:file|document)\s+(?:about|containing|named)?\s+['\"]?(.+?)['\"]?(?:\s|$|\?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                query = match.group(1).strip()
                # Remove trailing words like "please"
                query = re.split(r"\s+(?:please|thanks)$", query)[0].strip()
                return query if query else None

        return None

    def _should_open_file(self, user_input: str) -> bool:
        """
        Determine if user wants to open a found file.

        Args:
            user_input: User's input string.

        Returns:
            True if user wants to open the file.
        """
        open_indicators = ["open the file", "open it", "launch it"]
        return any(indicator in user_input.lower() for indicator in open_indicators)

    async def _open_file(self, file_path: str) -> bool:
        """
        Open a file using the default application.

        Args:
            file_path: Path to the file.

        Returns:
            True if successfully opened.
        """
        import subprocess

        try:
            # Validate the file path before opening
            from roxy.macos.path_validation import validate_path

            validated_path = validate_path(file_path, must_exist=True)

            result = subprocess.run(
                ["open", str(validated_path)],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return False
        except ValueError as e:
            logger.error(f"Invalid file path '{file_path}': {e}")
            return False
        except Exception as e:
            logger.error(f"Error opening file {file_path}: {e}")
            return False

    def can_handle(self, intent: str, parameters: dict) -> float:
        """
        Calculate confidence that this skill can handle the intent.

        Args:
            intent: Classified intent string.
            parameters: Extracted parameters.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        # Check for direct trigger matches
        if any(trigger in intent.lower() for trigger in self.triggers):
            return 0.9

        # Check for file-related keywords
        file_keywords = ["file", "document", "folder", "search", "find"]
        if any(keyword in intent.lower() for keyword in file_keywords):
            return 0.6

        return 0.0
