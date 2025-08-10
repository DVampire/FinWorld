from fastmcp import Client
from openai import OpenAI
import json

def convert2function(tool):
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

    tool = {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": parameters,
    }

    return tool

async def main():
    async with Client("server.py") as client:
        tools = await client.list_tools()
        tools = [convert2function(tool) for tool in tools]

        model_client = OpenAI("<your-api-key>")

        response = model_client.responses.create(
            model="gpt-4.1",
            input=[{"role": "user", "content": "What is the weather like in Paris today?"}],
            tools=tools
        )
        output = response.output[0]
        name = output.name
        arguments = json.loads(output.arguments)

        print(name, arguments)

        res = await client.call_tool(name, arguments)
        print(res)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())