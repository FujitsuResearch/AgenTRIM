import os
import sys
import argparse
import json
import mlflow
from dotenv import load_dotenv
from datetime import datetime
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
import inspect


def setup_paths():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_directory)
    grandparent_dir = os.path.dirname(parent_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    if grandparent_dir not in sys.path:
        sys.path.append(grandparent_dir)
setup_paths()

from langraph_react_tools import *


def load_mcp_client(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    for server in config.values():
        if server.get("command") in {"python", "python3"}:
            server["command"] = sys.executable
    return MultiServerMCPClient(config)

async def main():
    load_dotenv(override=False)
    mlflow.langchain.autolog()
    mlflow.set_tracking_uri("http://localhost:5000")

    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
        model=os.getenv("AZURE_MODEL_NAME"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY"),
        temperature=1,
    )

    parser = argparse.ArgumentParser(description="Run the ReAct agent with a custom user task.")
    parser.add_argument("--task", type=str, required=True, help="The user task to pass to the agent.")
    parser.add_argument("--tool_file", type=str, default="agent_scripts/langraph_react_tools_list.json", help="Path to the JSON file containing tool names.")
    parser.add_argument("--exp_name", type=str, default="langraph_react_partial_random_list", help="name of current experiments, for mlflow tracing.")
    args = parser.parse_args()

    
    # Start and await MCP client
    mcp_server_file = "agent_scripts/langraph_react_mcp_servers.json"
    mcp_client = load_mcp_client(mcp_server_file)
    
    mcp_tools = await mcp_client.get_tools()
    mcp_tool_dict = {tool.name: tool for tool in mcp_tools}
    
    # Load tool names from file
    with open(args.tool_file, "r") as f:
        tool_names = json.load(f)

    tool_list = []
    for name in tool_names:
        if name in mcp_tool_dict:
            tool_list.append(mcp_tool_dict[name])
        elif name in globals():
            tool_list.append(globals()[name])
        else:
            print(f"⚠️ Warning: Tool '{name}' not found in MCP or globals.")
    
    # Build and run agent
    agent = create_react_agent(llm, tools=tool_list, prompt="you are a helpfull asistant")
    # graph = agent.get_graph()
    # print("\n=== NODES ===")
    # for node_id, node in graph.nodes.items():
    #     print(f"- {node_id} : {node}")

    # print("\n=== EDGES ===")
    # for edge in graph.edges:
    #     print(f"- {edge}")

    # agent_node = graph.nodes["agent"]

    # print("\n=== AGENT NODE RAW ===")
    # print(agent_node)

    # print("\n=== AGENT NODE DATA (CALLABLE) ===")
    # print(agent_node.data)

    # print("\n=== DIR ===")
    # print(dir(agent_node.data))

    # print("\n=== __DICT__ ===")
    # print(getattr(agent_node.data, "__dict__", {}))

    # # Try common places where the underlying function might live
    # for attr in ("func", "bound", "target", "afunc"):
    #     if hasattr(agent_node.data, attr):
    #         inner = getattr(agent_node.data, attr)
    #         print(f"\n=== UNDERLYING {attr} ===")
    #         print(inner)
    #         try:
    #             print("Signature:", inspect.signature(inner))
    #         except Exception as e:
    #             print("Could not get signature:", e)
    # print(agent_node.data.input_schema())
    # print(agent_node.data.output_schema())
    # print(agent_node.data.config_schema())
    # exit()

    mlflow.set_experiment(args.exp_name)
    # Start manual MLflow run
    with mlflow.start_run() as run:

        # Run the agent as usual
        final_response = None
        observed_tool_calls = []
        async for response in agent.astream(
            {
                "input": args.task,
                "agent": {},
                "messages": [{"role": "user", "content": args.task}],
            },
            stream_mode="values",
        ):
            response["messages"][-1].pretty_print()
            final_response = response
            for message in response.get("messages", []):
                for tool_call in getattr(message, "tool_calls", []) or []:
                    name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
                    if name and name not in observed_tool_calls:
                        observed_tool_calls.append(name)

        print("Response:", final_response['messages'][-1].content)
        print("AGENT_TOOL_CALLS_JSON:", json.dumps(observed_tool_calls))


if __name__ == "__main__":
    asyncio.run(main())