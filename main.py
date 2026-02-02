"""
Research Agent - Interactive CLI

Usage:
    python main.py                    # Interactive mode
    python main.py -q "What is AI?"   # Single query
    python main.py -v                 # Verbose mode
"""
import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

load_dotenv()

console = Console()


def print_banner():
    banner = """
[bold cyan]Research Agent[/bold cyan] - AI-Powered Research Assistant
[dim]Type 'quit' to exit, 'help' for commands[/dim]
    """
    console.print(Panel(banner, border_style="cyan"))


async def run_query(query: str, verbose: bool = False):
    """Run a single query using the async agent."""
    import os
    from research_agent import AsyncResearchAgent, create_agent

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY not set[/red]")
        console.print("Set it with: export ANTHROPIC_API_KEY='your-key'")
        return None

    try:
        # Create async agent
        agent = create_agent(verbose=verbose, async_mode=True)

        with console.status("[bold green]Thinking...[/bold green]"):
            response = await agent.run(query)

        # Display response
        if response.answer:
            console.print(Panel(
                Markdown(response.answer),
                title="[bold green]Response[/bold green]",
                border_style="green",
            ))

            # Show confidence
            confidence_color = "green" if response.confidence > 0.7 else "yellow" if response.confidence > 0.4 else "red"
            console.print(f"[{confidence_color}]Confidence: {response.confidence:.0%}[/{confidence_color}]")

            # Show sources if any
            if response.sources:
                console.print("\n[bold]Sources:[/bold]")
                for source in response.sources:
                    if source:
                        console.print(f"  - {source}")

            # Show reasoning steps in verbose mode
            if verbose and response.reasoning_steps:
                console.print("\n[bold]Reasoning Steps:[/bold]")
                for i, step in enumerate(response.reasoning_steps, 1):
                    console.print(f"  {i}. [cyan]{step.thought}[/cyan]")
                    if step.action:
                        console.print(f"     Action: {step.action}")
                    if step.observation:
                        obs_preview = step.observation[:100] + "..." if len(step.observation) > 100 else step.observation
                        console.print(f"     Observation: {obs_preview}")

        return response

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())

    return None


async def interactive_mode(verbose: bool = False):
    """Run interactive chat mode."""
    print_banner()

    while True:
        try:
            console.print()
            query = console.input("[bold cyan]You:[/bold cyan] ").strip()

            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                console.print("\n[bold blue]Goodbye![/bold blue]")
                break
            if query.lower() == "help":
                console.print("""
[bold]Commands:[/bold]
  help  - Show this message
  quit  - Exit the program

[bold]Tips:[/bold]
  - Ask research questions
  - The agent will search and provide answers
  - Sources will be cited when available
""")
                continue

            console.print()
            await run_query(query, verbose=verbose)

        except KeyboardInterrupt:
            console.print("\n\n[bold blue]Goodbye![/bold blue]")
            break


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Research Agent - AI-powered research assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                     Interactive mode
  %(prog)s -q "What is AI?"    Single query
  %(prog)s -v                  Verbose mode (show reasoning)
        """,
    )

    parser.add_argument(
        "-q", "--query",
        help="Run a single query (non-interactive)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output (show reasoning steps)",
    )

    args = parser.parse_args()

    # Query mode
    if args.query:
        asyncio.run(run_query(args.query, args.verbose))
        return

    # Interactive mode
    asyncio.run(interactive_mode(args.verbose))


if __name__ == "__main__":
    main()
