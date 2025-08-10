"""Core module for the MCPAdapt library.

This module contains the core functionality for the MCPAdapt library. It provides the
basic interfaces and classes for adapting tools from MCP to the desired Agent framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine
from fastmcp.tools import Tool
import json5
import keyword
import re

from finworld.tools import AsyncTool

def _sanitize_function_name(name):
    """
    A function to sanitize function names to be used as a tool name.
    Prevent the use of dashes or other python keywords as function names by tool.
    """
    # Replace dashes with underscores
    name = name.replace("-", "_")

    # Remove any characters that aren't alphanumeric or underscore
    name = re.sub(r"[^\w_]", "", name)

    # Ensure it doesn't start with a number
    if name[0].isdigit():
        name = f"_{name}"

    # Check if it's a Python keyword
    if keyword.iskeyword(name):
        name = f"{name}_"

    return name

class ToolAdapter(ABC):
    def adapt(
        self,
        client,
        Tool
    ) -> Any:
        pass

class AsyncToolAdapter(ToolAdapter):
    """Adapter for the `smolagents` framework.

    Note that the `smolagents` framework do not support async tools at this time so we
    write only the adapt method.

    Warning: if the mcp tool name is a python keyword, starts with digits or contains
    dashes, the tool name will be sanitized to become a valid python function name.

    """
    async def adapt(
        self,
        client,
        tool: Tool,
    ) -> AsyncTool:

        class MCPAdaptTool(AsyncTool):
            def __init__(
                self,
                name: str,
                description: str,
                parameters: dict[str, dict[str, str]],
                output_type: str,
            ):
                self.name = _sanitize_function_name(name)
                self.description = description
                self.parameters = parameters
                self.output_type = output_type
                self.is_initialized = True
                self.skip_forward_signature_validation = True

                super(MCPAdaptTool, self).__init__()

            async def forward(self, *args, **kwargs) -> str:
                if len(args) > 0:
                    if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
                        async with client:
                            mcp_output = await client.call_tool(self.name, arguments=args[0])
                    else:
                        raise ValueError(
                            f"tool {self.name} does not support multiple positional arguments or combined positional and keyword arguments"
                        )
                else:
                    async with client:
                        mcp_output = await client.call_tool(name = self.name, arguments=kwargs)

                return json5.loads(mcp_output[0].text)

        name = tool.name
        description = tool.description or "No description."

        # make sure jsonref are resolved
        input_schema = tool.inputSchema

        # make sure mandatory `description` and `type` is provided for each arguments:
        for k, v in input_schema["properties"].items():
            if "description" not in v:
                input_schema["properties"][k]["description"] = "See tool description"
            if "type" not in v:
                input_schema["properties"][k]["type"] = "string"
            if 'title' in v:
                # remove title as it is not used in MCPAdaptTool
                del input_schema["properties"][k]['title']

        parameters = input_schema
        if parameters is not None:
            parameters['type'] = 'object'
        else:
            parameters = {}
        output_type = "any"

        tool = MCPAdaptTool(
            name=name,
            description=description,
            parameters= parameters,
            output_type= output_type,
        )

        return tool