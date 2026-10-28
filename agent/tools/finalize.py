"""Final answer tool. The model calls this once to finish; its arguments are validated before the run ends."""

from typing import Literal

from pydantic import BaseModel, ValidationInfo, model_validator


class FinalAnswer(BaseModel):
    answer: str
    source_ids: list[str]
    status: Literal["answered", "insufficient_information", "error"]

    @model_validator(mode="after")
    def check_sources(self, info: ValidationInfo) -> "FinalAnswer":
        if self.status == "answered" and not self.source_ids:
            raise ValueError("status 'answered' needs at least one source_id")
        if self.status != "answered" and self.source_ids:
            raise ValueError(f"status '{self.status}' must have an empty source_ids list")
        # When the agent validates a tool call it passes the search results seen so far;
        # only documents actually returned by a search may be cited.
        if info.context and "tool_outputs" in info.context:
            retrieved = "\n".join(info.context["tool_outputs"])
            not_retrieved = [i for i in self.source_ids if f"[{i}]" not in retrieved]
            if not_retrieved:
                raise ValueError(f"source_ids {not_retrieved} were not returned by any search; cite only retrieved documents")
        return self


def finalize_answer(answer: str, source_ids: list[str], status: str) -> dict:
    """Submit your final answer. Call this once, after searching, to finish."""
    return {"answer": answer, "source_ids": source_ids, "status": status}
