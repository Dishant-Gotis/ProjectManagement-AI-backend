import sys
import os
import json
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.llm_service import llm_service
from services import tool_service

console = Console()

def display_card(card):
    """
    Displays the Case Study Card in a beautiful Rich format or normal chat text.
    """
    console.print("\n")
    
    if isinstance(card, str):
        console.print(Panel(card, title="[bold green]Chatbot[/bold green]", border_style="green"))
        return
        
    table = Table(title=f"[bold cyan]Case Study Card: {card.concept}[/bold cyan]", show_header=False, box=None)
    
    table.add_row("[bold yellow]Story:[/bold yellow]", card.story)
    table.add_row("[bold yellow]Problem:[/bold yellow]", card.problem)
    table.add_row("[bold yellow]Decision Point:[/bold yellow]", card.decision_point)
    table.add_row("[bold yellow]Concept Mapping:[/bold yellow]", card.concept_mapping)
    
    lessons = "\n".join([f"• {l}" for l in card.key_lessons])
    table.add_row("[bold yellow]Key Lessons:[/bold yellow]", lessons)
    
    table.add_row("[bold green]Think About This:[/bold green]", f"[italic]{card.think_about_this}[/italic]")
    
    console.print(Panel(table, border_style="cyan", expand=True))

def run_gui():
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]PM-Project: AI Case Study Agent[/bold cyan]\n[italic white]Interactive Testing Terminal[/italic white]",
        border_style="magenta"
    ))
    
    # Patch tool_service to print tool executions to our console
    original_execute_tool = tool_service.execute_tool
    def patched_execute_tool(name, args):
        console.print(f"[bold magenta]▶  Agent is invoking tool:[/bold magenta] [yellow]{name}[/yellow] ([dim]{args}[/dim])")
        return original_execute_tool(name, args)
    
    tool_service.execute_tool = patched_execute_tool
    
    # We use a real UUID from the Supabase auth.users table to satisfy foreign key memory constraints
    user_id = "5fe1a9e7-7496-480a-8d3a-d4d8e0576eef"
    
    while True:
        try:
            query = Prompt.ask("\n[bold green]Enter your PM query[/bold green] (or 'q' to quit)")
            
            if query.lower() in ['q', 'quit', 'exit']:
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            with console.status("[bold blue]Agent is reasoning and building your card...[/bold blue]"):
                card = llm_service.process_query(query, user_id=user_id)
            
            display_card(card)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")

if __name__ == "__main__":
    run_gui()
