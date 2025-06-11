#!/usr/bin/env python
"""
This file implements a LangChain MCP client that:
  - Loads configuration from a JSON file specified by the THEAILANGUAGE_CONFIG environment variable.
  - Connects to one or more MCP servers defined in the config.
  - Loads available MCP tools from each connected server.
  - Uses the Google Gemini API (via LangChain) to create a React agent with access to all tools.
  - Runs an interactive chat loop where user queries are processed by the agent.
"""

import asyncio
import os
import sys
import json
from contextlib import AsyncExitStack

# ---------------------------
# MCP Client Imports
# ---------------------------
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---------------------------
# Agent and LLM Imports
# ---------------------------
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------
# Environment Setup
# ---------------------------
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# Custom JSON Encoder for LangChain objects
# ---------------------------
class CustomEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle non-serializable objects returned by LangChain.
    If the object has a 'content' attribute (such as HumanMessage or ToolMessage), serialize it accordingly.
    """

    def default(self, o):
        if hasattr(o, "content"):
            return {"type": o.__class__.__name__, "content": o.content}
        return super().default(o)


# ---------------------------
# Function: read_config_json
# ---------------------------
def read_config_json():
    """
    Reads the MCP server configuration JSON.

    Priority:
      1. Try to read the path from the SERVER_CONFIG environment variable.
      2. If not set, fallback to a default file 'config.json' in the same directory.

    Returns:
        dict: Parsed JSON content with MCP server definitions.
    """
    config_path = os.getenv("SERVER_CONFIG")

    if not config_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        print(
            f"⚠️  Config environment variable not set not set. Falling back to: {config_path}"
        )

    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to read config file at '{config_path}': {e}")
        sys.exit(1)


# ---------------------------
# Google Gemini LLM Instantiation
# ---------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_retries=2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


# ---------------------------
# Main Function: run_agent
# ---------------------------
async def run_agent():
    """
    Connects to all MCP servers defined in the configuration, loads their tools, creates a unified React agent,
    and starts an interactive loop to query the agent.
    """

    config = read_config_json()
    mcp_servers = config.get("mcpServers", {})
    if not mcp_servers:
        print("❌ No MCP servers found in the configuration.")
        return

    tools = []

    async with AsyncExitStack() as stack:

        for server_name, server_info in mcp_servers.items():
            print(f"\n🔗 Connecting to MCP Server: {server_name}...")

            server_params = StdioServerParameters(
                command=server_info["command"], args=server_info["args"]
            )

            try:
                read, write = await stack.enter_async_context(
                    stdio_client(server_params)
                )
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                server_tools = await load_mcp_tools(session)

                for tool in server_tools:
                    print(f"\n🔧 Loaded tool: {tool.name}")
                    tools.append(tool)

                print(f"\n✅ {len(server_tools)} tools loaded from {server_name}.")

            except Exception as e:
                print(f"❌ Failed to connect to server {server_name}: {e}")

        if not tools:
            print("❌ No tools loaded from any server. Exiting.")
            return

        agent = create_react_agent(llm, tools)

        print("\n🚀 MCP Client Ready! Type 'quit' to exit.")

        while True:
            query = input("\nQuery: ").strip()

            if query.lower() == "quit":
                break

            response = await agent.ainvoke({"messages": query})

            print("\nResponse:")

            try:
                formatted = json.dumps(response, indent=2, cls=CustomEncoder)
                print(formatted)

            except Exception:
                print(str(response))


# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    # Run the asynchronous run_agent function using asyncio's event loop
    asyncio.run(run_agent())
