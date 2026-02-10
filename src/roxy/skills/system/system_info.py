"""System info skill for getting macOS system status.

Provides information about running processes, system resources, and system status.
"""

from __future__ import annotations

import logging

from roxy.skills.base import Permission, RoxySkill, SkillContext, SkillResult

logger = logging.getLogger(__name__)


class SystemInfoSkill(RoxySkill):
    """Skill for getting system information and status."""

    name = "system_info"
    description = "Get system information and running processes"
    triggers = [
        "what's running",
        "system status",
        "running processes",
        "what applications are open",
        "system info",
        "what's on",
    ]
    permissions = [Permission.APPLESCRIPT]
    requires_cloud = False

    async def execute(self, context: SkillContext) -> SkillResult:
        """
        Execute the system info skill.

        Args:
            context: Skill execution context.

        Returns:
            SkillResult with system information.
        """
        user_input = context.user_input.lower()
        parameters = context.parameters

        # Determine what information to get
        if "running" in user_input or "process" in user_input or "application" in user_input:
            return await self._get_running_apps()
        elif "system" in user_input or "info" in user_input:
            return await self._get_system_info()
        elif "memory" in user_input or "ram" in user_input:
            return await self._get_memory_info()
        elif "disk" in user_input or "storage" in user_input:
            return await self._get_disk_info()
        else:
            # Default: get everything
            return await self._get_full_status()

    async def _get_running_apps(self) -> SkillResult:
        """
        Get list of running applications.

        Returns:
            SkillResult with running apps list.
        """
        try:
            from roxy.macos.applescript import get_applescript_runner

            runner = get_applescript_runner()
            apps = await runner.get_running_apps()

            if not apps:
                return SkillResult(
                    success=True,
                    response_text="No applications are currently running",
                    speak=True,
                )

            # Sort by name
            apps.sort(key=lambda a: a.get("name", ""))

            # Format response
            response = f"You have {len(apps)} application{'s' if len(apps) != 1 else ''} running:"

            # List apps
            for app in apps:
                name = app.get("name", "Unknown")
                frontmost = " (active)" if app.get("frontmost") else ""
                response += f"\n• {name}{frontmost}"

            # Limit length for speech
            speech_response = f"You have {len(apps)} applications running"

            return SkillResult(
                success=True,
                response_text=response,
                response_text_for_speech=speech_response,
                speak=True,
                data={"apps": apps, "count": len(apps)},
            )

        except Exception as e:
            logger.error(f"Error getting running apps: {e}")
            return SkillResult(
                success=False,
                response_text=f"Sorry, I couldn't get the list of running applications: {e}",
                speak=True,
            )

    async def _get_system_info(self) -> SkillResult:
        """
        Get system information.

        Returns:
            SkillResult with system info.
        """
        try:
            from roxy.macos.pyobjc_bridge import get_macos_bridge

            bridge = get_macos_bridge()
            info = await bridge.get_system_info()

            response = "System Information:"

            if "os_version" in info:
                response += f"\n• macOS: {info['os_version']}"
            if "computer_name" in info:
                response += f"\n• Computer: {info['computer_name']}"
            if "cpu" in info:
                response += f"\n• Processor: {info['cpu']}"
            if "ram_gb" in info:
                response += f"\n• Memory: {info['ram_gb']} GB"

            return SkillResult(
                success=True,
                response_text=response,
                speak=True,
                data={"system_info": info},
            )

        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return SkillResult(
                success=False,
                response_text=f"Sorry, I couldn't get system information: {e}",
                speak=True,
            )

    async def _get_memory_info(self) -> SkillResult:
        """
        Get memory usage information.

        Returns:
            SkillResult with memory info.
        """
        try:
            import subprocess

            # Get memory stats using vm_stat
            result = subprocess.run(
                ["vm_stat"],
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse output
            pages_free = 0
            pages_active = 0
            pages_wired = 0

            for line in result.stdout.split("\n"):
                if "Pages free:" in line:
                    pages_free = int(line.split(":")[1].strip().replace(".", ""))
                elif "Pages active:" in line:
                    pages_active = int(line.split(":")[1].strip().replace(".", ""))
                elif "Pages wired:" in line:
                    pages_wired = int(line.split(":")[1].strip().replace(".", ""))

            # Convert to GB (page size is 4096 bytes)
            page_size = 4096
            free_gb = (pages_free * page_size) / (1024**3)
            active_gb = (pages_active * page_size) / (1024**3)
            wired_gb = (pages_wired * page_size) / (1024**3)
            used_gb = active_gb + wired_gb

            response = f"Memory Usage:\n• Free: {free_gb:.1f} GB\n• Used: {used_gb:.1f} GB"

            return SkillResult(
                success=True,
                response_text=response,
                speak=True,
                data={
                    "memory": {
                        "free_gb": round(free_gb, 1),
                        "used_gb": round(used_gb, 1),
                    }
                },
            )

        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return SkillResult(
                success=False,
                response_text=f"Sorry, I couldn't get memory information: {e}",
                speak=True,
            )

    async def _get_disk_info(self) -> SkillResult:
        """
        Get disk usage information.

        Returns:
            SkillResult with disk info.
        """
        try:
            import subprocess

            result = subprocess.run(
                ["df", "-h", "/"],
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse output
            lines = result.stdout.split("\n")
            if len(lines) >= 2:
                parts = lines[1].split()
                if len(parts) >= 5:
                    total = parts[1]
                    used = parts[2]
                    available = parts[3]
                    percent = parts[4]

                    response = f"Disk Usage:\n• Total: {total}\n• Used: {used} ({percent})\n• Available: {available}"

                    return SkillResult(
                        success=True,
                        response_text=response,
                        speak=True,
                        data={
                            "disk": {
                                "total": total,
                                "used": used,
                                "available": available,
                                "percent": percent,
                            }
                        },
                    )

            return SkillResult(
                success=False,
                response_text="Sorry, I couldn't parse disk information",
                speak=True,
            )

        except Exception as e:
            logger.error(f"Error getting disk info: {e}")
            return SkillResult(
                success=False,
                response_text=f"Sorry, I couldn't get disk information: {e}",
                speak=True,
            )

    async def _get_full_status(self) -> SkillResult:
        """
        Get full system status.

        Returns:
            SkillResult with complete system status.
        """
        # Combine all info
        apps_result = await self._get_running_apps()
        system_result = await self._get_system_info()

        combined_response = system_result.response_text + "\n\n" + apps_result.response_text

        return SkillResult(
            success=True,
            response_text=combined_response,
            speak=True,
            data={
                "apps": apps_result.data.get("apps", []) if apps_result.data else [],
                "system_info": system_result.data.get("system_info", {})
                if system_result.data
                else {},
            },
        )
