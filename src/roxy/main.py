"""Roxy main entry point.

Provides a text-mode REPL, voice mode, and server mode for Roxy.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from enum import Enum
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .config import RoxyConfig, ConsentMode
from .brain.orchestrator import RoxyOrchestrator
from .skills import get_registry
from .skills.registry import SkillRegistry
from .voice.pipeline import VoicePipeline, create_voice_pipeline
from .macos.menubar import RoxyMenuBar, get_menubar_app
from .mcp.servers import MCPServerManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Rich console for output
console = Console()


class RunMode(str, Enum):
    """Roxy execution modes."""

    TEXT = "text"  # Interactive text REPL
    VOICE = "voice"  # Voice mode with wake word
    SERVER = "server"  # Background service with menu bar


class RoxyInitializer:
    """Handles initialization of Roxy components."""

    def __init__(
        self,
        config: RoxyConfig,
        mode: RunMode,
    ) -> None:
        """Initialize the initializer.

        Args:
            config: Roxy configuration.
            mode: Execution mode (text, voice, or server).
        """
        self.config = config
        self.mode = mode
        self.orchestrator: RoxyOrchestrator | None = None
        self.voice_pipeline: VoicePipeline | None = None
        self.menubar: RoxyMenuBar | None = None
        self.mcp_manager: MCPServerManager | None = None

    async def initialize(
        self,
        use_memory: bool = True,
        cloud_mode: str | None = None,
    ) -> tuple[RoxyOrchestrator, VoicePipeline | None, RoxyMenuBar | None]:
        """Initialize all Roxy components.

        Args:
            use_memory: Whether to initialize the memory system.
            cloud_mode: Override cloud consent mode (never|ask|always).

        Returns:
            Tuple of (orchestrator, voice_pipeline, menubar).
        """
        console.print(f"[bold cyan]Initializing {self.config.name}...[/bold cyan]")

        # Apply cloud mode override if specified
        if cloud_mode:
            try:
                original_consent = self.config.privacy.cloud_consent
                self.config.privacy.cloud_consent = ConsentMode(cloud_mode)
                logger.info(f"Cloud consent mode overridden to: {cloud_mode}")
            except ValueError:
                logger.warning(f"Invalid cloud mode: {cloud_mode}")

        try:
            # 1. Verify Ollama is available
            await self._verify_ollama()

            # 2. Initialize skill registry
            await self._initialize_skills()

            # 3. Initialize MCP servers (if configured)
            await self._initialize_mcp()

            # 4. Initialize orchestrator (includes memory)
            self.orchestrator = RoxyOrchestrator(
                config=self.config,
                skill_registry=get_registry(),
            )
            await self.orchestrator.initialize()
            console.print("[green]✓[/green] Orchestrator initialized")

            # 5. Initialize voice pipeline if in voice mode
            if self.mode == RunMode.VOICE:
                self.voice_pipeline = create_voice_pipeline(
                    orchestrator=self.orchestrator,
                    config=self.config.voice,
                )
                console.print("[green]✓[/green] Voice pipeline ready")

            # 6. Initialize menu bar if in server mode
            if self.mode == RunMode.SERVER:
                self.menubar = get_menubar_app(self.config.name)
                console.print("[green]✓[/green] Menu bar ready")

            console.print()
            console.print(f"[green]{self.config.name} is ready![/green]")
            console.print()

        except Exception as e:
            console.print(f"[red]✗[/red] Failed to initialize: {e}")
            raise

        return self.orchestrator, self.voice_pipeline, self.menubar

    async def _verify_ollama(self) -> None:
        """Verify Ollama is running and models are available."""
        import httpx

        ollama_host = self.config.llm_local.host
        logger.info(f"Checking Ollama at {ollama_host}...")

        try:
            # Check Ollama is running
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{ollama_host}/api/tags")
                if response.status_code != 200:
                    raise RuntimeError(f"Ollama returned status {response.status_code}")

            # Check for required models
            models_response = await client.get(f"{ollama_host}/api/tags")
            models_data = models_response.json()

            available_models = {
                model.get("name", "").split(":")[0]
                for model in models_data.get("models", [])
            }

            required_models = [
                self.config.llm_local.model.split(":")[0],
                self.config.llm_local.router_model.split(":")[0],
            ]

            missing = [m for m in required_models if m not in available_models]

            if missing:
                console.print(
                    f"[yellow]Warning: Missing Ollama models: {', '.join(missing)}[/yellow]"
                )
                console.print(
                    f"[yellow]Run: [bold]ollama pull {' && ollama pull '.join(missing)}[/bold][/yellow]"
                )
                console.print()

        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {ollama_host}. "
                f"Start Ollama with: [bold]ollama serve[/bold]"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama check failed: {e}")

    async def _initialize_skills(self) -> None:
        """Register all available skills."""
        from roxy.skills import (
            AppLauncherSkill,
            FileSearchSkill,
            WindowManagerSkill,
            SystemInfoSkill,
            ClipboardSkill,
            ShortcutsSkill,
            WebSearchSkill,
            BrowseSkill,
            FlightSearchSkill,
            CalendarSkill,
            EmailSkill,
            NotesSkill,
            RemindersSkill,
            GitOpsSkill,
            ClaudeCodeSkill,
        )

        registry = get_registry()

        # System skills
        for skill_cls in [
            AppLauncherSkill,
            FileSearchSkill,
            WindowManagerSkill,
            SystemInfoSkill,
            ClipboardSkill,
            ShortcutsSkill,
        ]:
            registry.register(skill_cls)

        # Web skills - inject privacy gateway
        from .brain.privacy import PrivacyGateway

        # Create privacy gateway for skills that need it
        privacy = PrivacyGateway(
            redact_patterns=self.config.privacy.redact_patterns,
            consent_mode=self.config.privacy.cloud_consent,
            log_path=f"{self.config.data_dir}/cloud_requests.log",
        )

        # Web skills with privacy gateway
        for skill_cls, needs_privacy in [
            (WebSearchSkill, True),
            (BrowseSkill, True),
            (FlightSearchSkill, True),
        ]:
            if needs_privacy:
                # Create skill instance with privacy gateway
                skill_instance = skill_cls(privacy_gateway=privacy)
                # Manually register the instance
                registry._skills[skill_instance.name] = skill_cls
                registry._skill_instances[skill_instance.name] = skill_instance
            else:
                registry.register(skill_cls)

        # Productivity skills
        for skill_cls in [
            CalendarSkill,
            EmailSkill,
            NotesSkill,
            RemindersSkill,
        ]:
            registry.register(skill_cls)

        # Developer skills
        for skill_cls in [
            GitOpsSkill,
            ClaudeCodeSkill,
        ]:
            registry.register(skill_cls)

        # Discover and register any other skill modules
        import roxy.skills
        skills_dir = Path(roxy.skills.__file__).parent
        registry.discover(str(skills_dir))

        # Log registered skills
        skills_list = registry.list_skills()
        logger.info(f"Registered {len(skills_list)} skills:")
        for skill in skills_list:
            logger.debug(f"  - {skill['name']}: {skill['description']}")

    async def _initialize_mcp(self) -> None:
        """Initialize MCP servers if configured."""
        try:
            self.mcp_manager = MCPServerManager()

            # Auto-start enabled MCP servers
            results = await self.mcp_manager.start_all()

            started = sum(1 for success in results.values() if success)
            if started > 0:
                console.print(f"[green]✓[/green] Started {started} MCP server(s)")
            else:
                console.print("[dim]No MCP servers configured[/dim]")

        except Exception as e:
            logger.warning(f"MCP initialization failed: {e}")
            console.print("[yellow]⚠[/yellow] MCP servers not available")

    async def shutdown(self) -> None:
        """Shutdown initialized components."""
        # Stop voice pipeline
        if self.voice_pipeline:
            await self.voice_pipeline.stop()

        # Stop menu bar
        if self.menubar:
            self.menubar.stop()

        # Stop MCP servers
        if self.mcp_manager:
            await self.mcp_manager.stop_all()

        # Shutdown orchestrator
        if self.orchestrator:
            await self.orchestrator.shutdown()


class CommandHandler:
    """Handles REPL slash commands."""

    def __init__(
        self,
        config: RoxyConfig,
        orchestrator: RoxyOrchestrator | None,
    ) -> None:
        """Initialize the command handler.

        Args:
            config: Roxy configuration.
            orchestrator: The orchestrator instance (may be None).
        """
        self.config = config
        self.orchestrator = orchestrator

    async def handle(self, command: str) -> bool:
        """Handle a REPL command.

        Args:
            command: Command string starting with '/'.

        Returns:
            True if the command was handled, False otherwise.
        """
        parts = command.split()
        cmd = parts[0].lower()

        if cmd in ("/quit", "/exit"):
            return False  # Signal to exit

        if cmd == "/help":
            self._print_help()
        elif cmd == "/stats":
            await self._print_stats()
        elif cmd == "/config":
            self._print_config()
        elif cmd == "/memory":
            await self._print_memory()
        else:
            console.print(f"[red]Unknown command: {cmd}[/red]")
            console.print("Type [bold]/help[/bold] for available commands")

        return True  # Continue running

    def _print_help(self) -> None:
        """Print help message."""
        help_table = Table(title="Available Commands")
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description")

        commands = [
            ("/help", "Show this help message"),
            ("/stats", "Show routing and usage statistics"),
            ("/config", "Show current configuration"),
            ("/memory", "Search and display memories"),
            ("/quit, /exit", "Exit the REPL"),
        ]

        for cmd, desc in commands:
            help_table.add_row(cmd, desc)

        console.print(help_table)

    async def _print_stats(self) -> None:
        """Print statistics."""
        if not self.orchestrator:
            console.print("[red]Orchestrator not initialized[/red]")
            return

        stats = await self.orchestrator.get_statistics()

        stats_table = Table(title="Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value")

        stats_table.add_row("Conversation Length", str(stats["conversation_length"]))
        stats_table.add_row("Skills Registered", str(stats["skills_registered"]))

        routing = stats["routing_stats"]
        stats_table.add_row("Total Requests", str(routing["total_requests"]))
        stats_table.add_row("Local Requests", str(routing["local_requests"]))
        stats_table.add_row("Cloud Requests", str(routing["cloud_requests"]))
        stats_table.add_row("Local Rate", f"{routing['local_rate']:.1%}")
        stats_table.add_row("Cloud Rate", f"{routing['cloud_rate']:.1%}")

        console.print(stats_table)

    def _print_config(self) -> None:
        """Print current configuration."""
        config_table = Table(title="Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value")

        config_table.add_row("Name", self.config.name)
        config_table.add_row("Version", self.config.version)
        config_table.add_row("Data Directory", self.config.data_dir)
        config_table.add_row("Log Level", self.config.log_level)
        config_table.add_row("", "")
        config_table.add_row("Local Model", self.config.llm_local.model)
        config_table.add_row("Router Model", self.config.llm_local.router_model)
        config_table.add_row("Local Host", self.config.llm_local.host)
        config_table.add_row("", "")
        config_table.add_row("Cloud Provider", self.config.llm_cloud.provider.value)
        config_table.add_row("Cloud Model", self.config.llm_cloud.model)
        config_table.add_row("Confidence Threshold", str(self.config.llm_cloud.confidence_threshold))
        config_table.add_row("", "")
        config_table.add_row("PII Redaction", str(self.config.privacy.pii_redaction_enabled))
        config_table.add_row("Cloud Consent", self.config.privacy.cloud_consent.value)

        console.print(config_table)

    async def _print_memory(self) -> None:
        """Print memory contents."""
        if not self.orchestrator:
            console.print("[red]Orchestrator not initialized[/red]")
            return

        memories = await self.orchestrator.get_memory("")

        if not memories:
            console.print("[yellow]No memories stored yet[/yellow]")
            return

        memory_table = Table(title="Stored Memories")
        memory_table.add_column("#", style="cyan")
        memory_table.add_column("Content")

        for i, memory in enumerate(memories[:10], 1):
            memory_table.add_row(str(i), memory[:100])

        console.print(memory_table)


class RoxyREPL:
    """Interactive REPL for Roxy."""

    def __init__(self, config: RoxyConfig, mode: RunMode = RunMode.TEXT) -> None:
        """Initialize the REPL.

        Args:
            config: Roxy configuration.
            mode: Execution mode (text, voice, or server).
        """
        self.config = config
        self.mode = mode
        self.running = False
        self.initializer = RoxyInitializer(config, mode)
        self.command_handler = CommandHandler(config, None)

        # These will be set after initialization
        self.orchestrator: RoxyOrchestrator | None = None
        self.voice_pipeline: VoicePipeline | None = None
        self.menubar: RoxyMenuBar | None = None

    async def initialize(self, use_memory: bool = True, cloud_mode: str | None = None) -> None:
        """Initialize all Roxy components.

        Args:
            use_memory: Whether to initialize the memory system.
            cloud_mode: Override cloud consent mode (never|ask|always).
        """
        self.orchestrator, self.voice_pipeline, self.menubar = await self.initializer.initialize(
            use_memory=use_memory,
            cloud_mode=cloud_mode,
        )
        # Update command handler with the initialized orchestrator
        self.command_handler = CommandHandler(self.config, self.orchestrator)

    async def run(self) -> None:
        """Run the appropriate mode."""
        self.running = True

        if self.mode == RunMode.TEXT:
            await self._run_repl()
        elif self.mode == RunMode.VOICE:
            await self._run_voice()
        elif self.mode == RunMode.SERVER:
            await self._run_server()

    async def _run_repl(self) -> None:
        """Run text REPL mode."""
        self._print_welcome()

        while self.running:
            try:
                user_input = Prompt.ask(
                    "[bold yellow]You[/bold yellow]",
                    console=console,
                )

                if not user_input.strip():
                    continue

                if user_input.startswith("/"):
                    should_continue = await self.command_handler.handle(user_input)
                    if not should_continue:
                        self.running = False
                        console.print("[yellow]Goodbye![/yellow]")
                    continue

                await self._process_input(user_input)

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupt received[/yellow]")
                continue
            except EOFError:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                logger.exception("REPL error")

    async def _run_voice(self) -> None:
        """Run voice mode."""
        if not self.voice_pipeline:
            console.print("[red]Voice pipeline not initialized[/red]")
            return

        console.print("[bold cyan]Voice mode active[/bold cyan]")
        console.print("Say 'Hey Roxy' to activate, or press Ctrl+C to exit")
        console.print()

        try:
            await self.voice_pipeline.start()

            # Keep running until interrupted
            while self.running:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping voice mode[/yellow]")

        finally:
            if self.voice_pipeline:
                await self.voice_pipeline.stop()

    async def _run_server(self) -> None:
        """Run server mode with menu bar."""
        if not self.menubar:
            console.print("[red]Menu bar not initialized[/red]")
            return

        # Start voice pipeline in background if available
        if self.voice_pipeline:
            asyncio.create_task(self._voice_background_task())

        # Setup menu bar callbacks
        self.menubar.set_quit_callback(self._shutdown)
        self.menubar.set_listening_toggle_callback(self._toggle_listening)

        # Run menu bar (blocking)
        self.menubar.run()

    async def _voice_background_task(self) -> None:
        """Run voice pipeline in background during server mode."""
        try:
            if self.voice_pipeline:
                await self.voice_pipeline.start()

                while self.running:
                    await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Voice background task error: {e}")

    def _toggle_listening(self) -> None:
        """Toggle voice listening state."""
        if not self.voice_pipeline:
            return

        if self.voice_pipeline.is_running:
            # Would need to add pause/resume functionality
            pass

    async def _process_input(self, user_input: str) -> None:
        """Process user input through the orchestrator.

        Args:
            user_input: User's input text.
        """
        if not self.orchestrator:
            console.print("[red]Orchestrator not initialized[/red]")
            return

        with console.status("[bold cyan]Thinking...[/bold cyan]"):
            response = await self.orchestrator.process(user_input)

        console.print()
        console.print(Panel(
            Markdown(response),
            title=f"[bold cyan]{self.config.name}[/bold cyan]",
            border_style="cyan",
        ))
        console.print()

    def _print_welcome(self) -> None:
        """Print welcome message."""
        welcome_text = f"""
[bold cyan]Welcome to {self.config.name}[/bold cyan]
[bold]Version:[/bold] {self.config.version}
[bold]Mode:[/bold] {self.mode.value}
[bold]Local Model:[/bold] {self.config.llm_local.model}
[bold]Cloud Provider:[/bold] {self.config.llm_cloud.provider.value}

Type your message to chat, or [bold]/help[/bold] for commands.
"""
        console.print(Panel(welcome_text, border_style="cyan"))

    async def shutdown(self) -> None:
        """Shutdown the REPL and cleanup."""
        self.running = False
        await self.initializer.shutdown()

    def _shutdown(self) -> None:
        """Sync shutdown callback for menu bar."""
        # Create event loop if needed and run shutdown
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.shutdown())
            else:
                asyncio.run(self.shutdown())
        except Exception:
            pass


def setup_signal_handlers(repl: RoxyREPL) -> None:
    """Setup signal handlers for graceful shutdown.

    Args:
        repl: REPL instance to shutdown on signal.
    """

    def signal_handler(sig, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {sig}, shutting down...")
        repl.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to YAML config file",
)
@click.option(
    "--voice",
    is_flag=True,
    help="Start in voice mode (wake word + speech I/O)",
)
@click.option(
    "--server",
    is_flag=True,
    help="Start as background service with menu bar",
)
@click.option(
    "--no-memory",
    is_flag=True,
    help="Disable memory system (for debugging)",
)
@click.option(
    "--cloud",
    type=click.Choice(["never", "ask", "always"]),
    help="Override cloud consent mode",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose/debug logging",
)
def main(
    config: str | None,
    voice: bool,
    server: bool,
    no_memory: bool,
    cloud: str | None,
    verbose: bool,
) -> None:
    """
    Roxy — Local-first, voice-controlled AI assistant for macOS.

    Default mode is text REPL. Use --voice for voice mode or --server
    for background service with menu bar.
    """
    # Set debug logging if requested
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Determine run mode
    if server:
        mode = RunMode.SERVER
    elif voice:
        mode = RunMode.VOICE
    else:
        mode = RunMode.TEXT

    # Load configuration
    try:
        if config:
            roxy_config = RoxyConfig.load(yaml_path=Path(config))
        else:
            roxy_config = RoxyConfig.load()

        logger.info(f"Loaded configuration for {roxy_config.name}")
    except Exception as e:
        console.print(f"[red]Failed to load configuration: {e}[/red]")
        sys.exit(1)

    # Run REPL
    async def run_roxy() -> None:
        repl = RoxyREPL(roxy_config, mode=mode)
        setup_signal_handlers(repl)

        try:
            await repl.initialize(use_memory=not no_memory, cloud_mode=cloud)
            await repl.run()
        finally:
            await repl.shutdown()

    try:
        asyncio.run(run_roxy())
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        logger.exception("Fatal error in main")
        sys.exit(1)


if __name__ == "__main__":
    main()
