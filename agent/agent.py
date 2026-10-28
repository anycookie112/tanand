"""Minimal tool-calling agent built on LangChain chat models.

Every tool call the model makes is validated against the tool's pydantic
schema before it runs. Invalid arguments are not executed; the validation
errors are sent back to the model as a ToolMessage so it can correct itself.
A valid call to a tool registered with `final=True` ends the run.
"""

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

PROMPTS_FILE = Path(__file__).with_name("prompts.yaml")
DEFAULT_MODELS = {"anthropic": "claude-haiku-5-5"}


class MaxIterationsError(RuntimeError):
    """The model kept calling tools past the iteration limit."""


@dataclass
class _Tool:
    func: Callable[..., Any]
    schema: type[BaseModel]
    spec: StructuredTool  # only used to describe the tool to the model
    final: bool  # a valid call ends the run and returns the tool's output
    stream_arg: str | None  # string argument whose text `stream` yields while the model writes it


class Agent:
    """Tool-calling agent.

    If no `llm` is passed, one is built from environment variables (.env is loaded):
      LLM_PROVIDER  "anthropic" (default) or "openai" (any OpenAI-compatible API)
      LLM_MODEL     model name; defaults to claude-haiku-5-5 for anthropic
      openai:    OPENAI_API_KEY, optional OPENAI_BASE_URL for compatible endpoints
      anthropic: ANTHROPIC_API_KEY, optional ANTHROPIC_BASE_URL
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        prompts_path: Path = PROMPTS_FILE,
        prompt_name: str = "system",
        max_iterations: int = 5,
        temperature: float | None = None,  # some models (e.g. Claude Haiku 5.5) reject this; leave None for those
        timeout: float = 30.0,
        memory_turns: int = 5,
    ):
        if llm is None:
            load_dotenv()
            provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
            model = os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(provider)
            if not model:
                raise ValueError(f"LLM_MODEL is not set and provider '{provider}' has no default model")
            kwargs = {"model": model, "timeout": timeout, "max_retries": 1}
            if temperature is not None:
                kwargs["temperature"] = temperature
            if provider == "openai":
                llm = ChatOpenAI(**kwargs)
            elif provider == "anthropic":
                llm = ChatAnthropic(**kwargs)
            else:
                raise ValueError(f"Unsupported LLM_PROVIDER '{provider}' (expected 'openai' or 'anthropic')")
        self.llm = llm
        self.prompts: dict[str, str] = yaml.safe_load(prompts_path.read_text(encoding="utf-8")) or {}
        if prompt_name not in self.prompts:
            raise KeyError(f"No '{prompt_name}' prompt in {prompts_path.name}. Found: {', '.join(self.prompts) or 'none'}")
        self.system_prompt = self.prompts[prompt_name]
        self.max_iterations = max_iterations
        # Memory keeps the last `memory_turns` questions and final replies, not tool calls,
        # so each turn must search again before it can cite a document.
        self.memory_turns = memory_turns
        self.memory: list[BaseMessage] = []
        self._tools: dict[str, _Tool] = {}

    def clear_memory(self) -> None:
        self.memory = []

    def register_tool(
        self,
        func: Callable[..., Any],
        args_schema: type[BaseModel],
        name: str | None = None,
        description: str | None = None,
        final: bool = False,
        stream_arg: str | None = None,
    ) -> None:
        """Expose `func` to the model. Its arguments must match `args_schema`.

        If `final` is True, a valid call ends the run and `invoke` returns the tool's output.
        If `stream_arg` is set, `stream` yields that argument's text as the model writes it.
        """
        name = name or func.__name__
        description = description or func.__doc__ or args_schema.__doc__ or name
        spec = StructuredTool.from_function(
            func=func, name=name, description=description, args_schema=args_schema
        )
        if stream_arg:
            # Anthropic buffers tool input by default; this streams it as it is written.
            # The API then skips validating the input, which verify_output does instead.
            spec.extras = {"eager_input_streaming": True}
        self._tools[name] = _Tool(
            func=func, schema=args_schema, spec=spec, final=final, stream_arg=stream_arg
        )

    def verify_output(self, tool_call: ToolCall, messages: list[BaseMessage]) -> BaseModel | str:
        """Validate a tool call from the model.

        Successful tool outputs so far are passed to the schema as validation
        context (`tool_outputs`), so a final-answer schema can check grounding.
        Returns the parsed arguments, or an error message to pass back to the model.
        """
        tool = self._tools.get(tool_call["name"])
        if tool is None:
            return f"Unknown tool '{tool_call['name']}'. Available tools: {', '.join(self._tools) or 'none'}."
        tool_outputs = [m.text for m in messages if isinstance(m, ToolMessage) and m.status == "success"]
        try:
            return tool.schema.model_validate(tool_call["args"], context={"tool_outputs": tool_outputs})
        except ValidationError as e:
            return f"Invalid arguments for '{tool_call['name']}': {e}\nFix the arguments and call the tool again."

    def invoke(self, user_input: str) -> Any:
        """Run the agent. Returns the final tool's output, or the reply text if the model answers directly."""
        messages = self._initial_messages(user_input)
        llm = self._bound_llm()
        for _ in range(self.max_iterations):
            response = llm.invoke(messages)
            messages.append(response)
            if response.tool_calls:
                output = self._run_tool_calls(response, messages)
                if output is not None:
                    self._remember(user_input, str(output))
                    return output
                continue
            self._remember(user_input, response.text)
            return response.text
        raise MaxIterationsError(f"No final answer after {self.max_iterations} iterations")

    def stream(self, user_input: str) -> Iterator[Any]:
        """Like `invoke`, but yields text as the model produces it.

        Text includes the model's plain replies and the `stream_arg` of any tool
        call as it is written. If a final tool ends the run, the last item
        yielded is its output (not a str).
        """
        messages = self._initial_messages(user_input)
        llm = self._bound_llm()
        for _ in range(self.max_iterations):
            response = None
            streamed = False
            arg_streamed = ""
            for chunk in llm.stream(messages):
                response = chunk if response is None else response + chunk
                if chunk.text:
                    streamed = True
                    yield chunk.text
                # Tool-call args arrive as partial JSON; yield only the newly written text.
                arg_text = self._stream_arg_text(response)
                if len(arg_text) > len(arg_streamed) and arg_text.startswith(arg_streamed):
                    streamed = True
                    yield arg_text[len(arg_streamed) :]
                    arg_streamed = arg_text
            if response is None:
                return
            messages.append(response)
            if response.tool_calls:
                output = self._run_tool_calls(response, messages)
                if output is not None:
                    self._remember(user_input, str(output))
                    yield output
                    return
                if streamed:
                    yield "\n\n"  # separate this round's text from the next round's
                continue
            self._remember(user_input, response.text)
            return
        raise MaxIterationsError(f"No final answer after {self.max_iterations} iterations")

    def _initial_messages(self, user_input: str) -> list[BaseMessage]:
        # The user's text only ever goes in a HumanMessage, never in the system prompt.
        return [SystemMessage(self.system_prompt), *self.memory, HumanMessage(user_input)]

    def _stream_arg_text(self, response: AIMessage) -> str:
        for call in response.tool_calls:
            tool = self._tools.get(call["name"])
            if tool and tool.stream_arg and isinstance(call["args"].get(tool.stream_arg), str):
                return call["args"][tool.stream_arg]
        return ""

    def _remember(self, user_input: str, reply: str) -> None:
        self.memory += [HumanMessage(user_input), AIMessage(reply)]
        self.memory = self.memory[-2 * self.memory_turns :] if self.memory_turns > 0 else []

    def _bound_llm(self):
        if not self._tools:
            return self.llm
        specs = [t.spec for t in self._tools.values()]
        if any(t.final for t in self._tools.values()):
            # Force a tool call every turn, so the model has to finish through the final tool.
            return self.llm.bind_tools(specs, tool_choice="any")
        return self.llm.bind_tools(specs)

    def _run_tool_calls(self, response: AIMessage, messages: list[BaseMessage]) -> Any:
        """Run the tool calls in `response`, appending results to `messages`.

        Returns the output of a final tool if one was called successfully, otherwise None.
        """
        for call in response.tool_calls:
            args = self.verify_output(call, messages)
            if isinstance(args, str):
                messages.append(ToolMessage(args, tool_call_id=call["id"], status="error"))
                continue
            tool = self._tools[call["name"]]
            try:
                output = tool.func(**dict(args))
            except Exception as e:
                messages.append(
                    ToolMessage(f"Tool '{call['name']}' failed: {e}", tool_call_id=call["id"], status="error")
                )
                continue
            if tool.final:
                return output
            messages.append(ToolMessage(str(output), tool_call_id=call["id"]))
        return None
