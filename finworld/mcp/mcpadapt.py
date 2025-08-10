"""Core module for the MCPAdapt library.

This module contains the core functionality for the MCPAdapt library. It provides the
basic interfaces and classes for adapting tools from MCP to the desired Agent framework.
"""

from functools import partial
from typing import Any, Dict
from fastmcp import Client
import asyncio

from finworld.mcp.adapter import AsyncToolAdapter, ToolAdapter

class MCPAdapt:
    def __init__(
        self,
        config: Dict[str, Any],
        adapter: ToolAdapter,
    ):
        """
        Manage the MCP server / client lifecycle and expose tools adapted with the adapter.

        Args:
            serverparams (StdioServerParameters | dict[str, Any] | list[StdioServerParameters | dict[str, Any]]):
                MCP server parameters (stdio or sse). Can be a list if you want to connect multiple MCPs at once.
            adapter (ToolAdapter): Adapter to use to convert MCP tools call into agentic framework tools.
            connect_timeout (int): Connection timeout in seconds to the mcp server (default is 30s).
            client_session_timeout_seconds: Timeout for MCP ClientSession calls

        Raises:
            TimeoutError: When the connection to the mcp server time out.
        """
        self.config = config
        self.adapter = adapter
        self.client = Client(config)

    async def tools(self):
        """Returns the tools from the MCP server adapted to the desired Agent framework.

        This is what is yielded if used as a context manager otherwise you can access it
        directly via this method.

        Only use this when you start the client in synchronous context or by :meth:`start`.

        An equivalent async method is available if your Agent framework supports it:
        see :meth:`atools`.

        """
        async with self.client as client:
            mcp_tools = await client.list_tools()

        mcp_tools = await asyncio.gather(*[
            self.adapter.adapt(client, tool)
            for tool in mcp_tools
        ])

        mcp_tools = {
            tool.name: tool
            for tool in mcp_tools
        }
        return mcp_tools

async def main():
    config = {
        "mcpServers": {
            # Local stdio server
            "weather": {
                "command": "python",
                "args": ["server.py"],
                "env": {"DEBUG": "true"}
            },
        }
    }

    mcpadapt = MCPAdapt(
        config,
        AsyncToolAdapter()
    )

    tools = await mcpadapt.tools()
    for name, tool in tools.items():
        print(f"Tool Name: {name}")
        print(f"Tool Description: {tool.description}")
        print(f"Tool Input Schema: {tool.inputSchema}")
        print("-" * 40)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())