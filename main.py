"""Support assistant.

Command:      uv run python main.py "How long do I have to return an unopened product?"
Interactive:  uv run python main.py
Batch:        uv run python main.py --questions [path]   (default: utils/questions.yaml)

The answer streams as it is written, then the full JSON result is printed.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import yaml
from rich import print, print_json
from rich.markup import escape

from agent import Agent
from agent.tools import FinalAnswer, SearchArgs, finalize_answer, search_knowledge_base

QUESTIONS_FILE = Path(__file__).parent / "utils" / "questions.yaml"


def build_agent() -> Agent:
    agent = Agent()
    agent.register_tool(search_knowledge_base, SearchArgs)
    agent.register_tool(finalize_answer, FinalAnswer, final=True, stream_arg="answer")
    return agent


def error(message: str) -> dict:
    return {"answer": message, "source_ids": [], "status": "error"}


def ask(agent: Agent, question: str, on_text: Callable[[str], None] | None = None) -> dict:
    """Return the JSON-compatible result. `on_text` receives answer text as it streams."""
    if not question.strip():
        return error("Please enter a question.")
    result = None
    try:
        for item in agent.stream(question):
            if isinstance(item, str):
                if on_text:
                    on_text(item)
            else:
                result = item
    except Exception as e:
        # Exception type only: messages from provider SDKs can include request details.
        return error(f"The assistant could not answer right now ({type(e).__name__}). Please try again later.")
    if not isinstance(result, dict):
        return error("The assistant did not return a final answer. Please try again.")
    return result


def respond(agent: Agent, question: str) -> None:
    streamed = False

    def on_text(text: str) -> None:
        nonlocal streamed
        streamed = True
        print(escape(text), end="")  # escape so "[...]" in answers isn't read as rich markup

    result = ask(agent, question, on_text)
    if streamed:
        print("\n")
    print("[bold cyan]Actual output:[/bold cyan]")
    print_json(data=result)


def run_questions(agent: Agent, path: Path) -> None:
    questions = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    for i, question in enumerate(questions, 1):
        agent.clear_memory()  # answer each question independently
        print(f"\n[bold]Q{i}: {escape(str(question)) or '(empty input)'}[/bold]")
        respond(agent, str(question))


def interactive(agent: Agent) -> None:
    print("Support assistant. Type a question, 'clear' to forget the conversation, or 'exit' to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.lower() in {"exit", "quit"}:
            return
        if question.lower() == "clear":
            agent.clear_memory()
            print("Conversation cleared.")
            continue
        respond(agent, question)


def main():
    parser = argparse.ArgumentParser(description="Ask the support assistant a question.")
    parser.add_argument("question", nargs="*", help="question to ask; omit to start interactive mode")
    parser.add_argument(
        "--questions", nargs="?", const=QUESTIONS_FILE, type=Path, metavar="PATH",
        help=f"answer every question in a YAML list (default: {QUESTIONS_FILE.relative_to(Path(__file__).parent)})",
    )
    args = parser.parse_args()

    agent = build_agent()
    if args.questions:
        run_questions(agent, args.questions)
    elif args.question:
        respond(agent, " ".join(args.question))
    else:
        interactive(agent)


if __name__ == "__main__":
    main()
