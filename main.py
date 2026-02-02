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
    [white]Research Agent[/white] - AI-Powered Research Assistant
    [dim]Type 'quit' to exit, 'help' for commands[/dim]
    """
    console.print(banner)
    
async def run_query(query: str, stream: bool = False, verbose: bool = False):
    """Run a single query"""
    import os
    from research_agent import AgentService, Agentstatus
    
    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red] ANTHROPIC_API_KEY not set[/red]")
        console.print("Set it with: export ANTHROPIC_API_KEY='your-key'") 
        return None
    
    try:
        async with AgentService() as agent:
            if stream:
                console.print("[bold]Progress:[/bold]")
                final_state = None
                
                async for event in agent.stream(query):
                    if event.type == "node_start":
                        console.print(f"  [cyan]> {event.node_id}[/cyan]")
                    elif event.type == "complete":
                        final_state = event.state
            
            else:
                with console.status("[bold green]Thinking..."): 
                  final_state = await agent.run(query)        
            
            if final_state:
                if final_state.status == AgentStatus.Completed:
                    console.print(Panel(
                        Markdown(final_state.final_answer or "No answer"),
                        title="[bold green]✅ Response[/bold green]",
                        border_style="green",
                    ))
                else:
                    console.print(Panel(
                        final_state.error or "Unknown error",
                        title="[bold red]❌ Error[/bold red]",
                        border_style="red",
                    ))
                return final_state
    
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
    
    return None


async def interactive_mode(stream: bool = False, verbose: bool = False):
    """Run interactive chat mode."""
    print_banner()
    
    while True:
        try:
            console.print()
            query = console.input("[bold cyan]🔍 You:[/bold cyan] ").strip()
            
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                console.print("\n[bold blue]👋 Goodbye![/bold blue]")
                break
            if query.lower() == "help":
                console.print("""
[bold]Commands:[/bold]
  help  - Show this message
  quit  - Exit the program
  
[bold]Tips:[/bold]
  • Ask research questions
  • The agent can search the web
  • Sources will be cited
""")
                continue
            
            console.print()
            await run_query(query, stream=stream, verbose=verbose)
            
        except KeyboardInterrupt:
            console.print("\n\n[bold blue]👋 Goodbye![/bold blue]")
            break


def start_server(host: str, port: int, reload: bool):
    """Start the API server."""
    from research_agent.api import run_server
    
    console.print(f"[bold green]🚀 Starting API server on {host}:{port}[/bold green]")
    console.print(f"[dim]   Docs: http://localhost:{port}/docs[/dim]")
    console.print(f"[dim]   Health: http://localhost:{port}/health[/dim]")
    console.print()
    
    run_server(host=host, port=port, reload=reload)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Research Agent - AI-powered research assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           Interactive mode
  %(prog)s -q "What is AI?"          Single query
  %(prog)s -q "Explain ML" --stream  With streaming
  %(prog)s --server                  Start API server
  %(prog)s --server --port 3000      Custom port
        """,
    )
    
    # Query options
    parser.add_argument(
        "-q", "--query",
        help="Run a single query (non-interactive)",
    )
    parser.add_argument(
        "-s", "--stream",
        action="store_true",
        help="Show streaming progress",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output (show errors)",
    )
    
    # Server options
    parser.add_argument(
        "--server",
        action="store_true",
        help="Start the API server",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development)",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    from research_agent.utils import configure_logging
    configure_logging(level="DEBUG" if args.verbose else "INFO")
    
    # Server mode
    if args.server:
        start_server(args.host, args.port, args.reload)
        return
    
    # Query mode
    if args.query:
        asyncio.run(run_query(args.query, args.stream, args.verbose))
        return
    
    # Interactive mode
    asyncio.run(interactive_mode(args.stream, args.verbose))


if __name__ == "__main__":
    main()
                           