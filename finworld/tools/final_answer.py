from finworld.tools.tools import AsyncTool, ToolResult
from finworld.registry import TOOL

@TOOL.register_module(force=True)
class FinalAnswerTool(AsyncTool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    parameters = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The final answer to the problem.",
            },
        },
        "required": ["answer"],
    }
    output_type = "any"

    async def forward(self, answer: str) -> ToolResult:
        result = ToolResult(
            output=answer,
            error=None,
        )
        return result