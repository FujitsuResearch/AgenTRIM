"""Run AGENTRIM tool extraction in static-only or full verification mode."""

import argparse
import json
from pathlib import Path

try:
    from .extractor_utils import (
        discover_additional_tools,
        extract_tools_code,
        generate_test_queries,
        generate_test_queries_additional,
        test_tool,
    )
except ImportError:
    from extractor_utils import (
        discover_additional_tools,
        extract_tools_code,
        generate_test_queries,
        generate_test_queries_additional,
        test_tool,
    )


def _write_json(path: str, value: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main(
    agent_file_path: str,
    output_json_path: str,
    code_output_json_path: str,
    tool_list: str,
    trace_exp_id: str,
    static_only: bool = False,
) -> dict:
    print("Extracting initial tool list...")
    Path(code_output_json_path).parent.mkdir(parents=True, exist_ok=True)
    tools = extract_tools_code(agent_file_path, code_output_json_path)

    if static_only:
        _write_json(output_json_path, tools)
        print(f"Static extraction saved {len(tools)} tools to {output_json_path}")
        return tools

    verified_tools = {}
    if tools:
        print("Generating and executing one verification query per tool...")
        for tool_name, query in generate_test_queries(tools).items():
            tool_results = test_tool(agent_file_path, query, tool_list, trace_exp_id)
            verified_tools.update(tool_results)
            if tool_name not in tool_results:
                retry = f"You must call {tool_name} with no arguments and report the response."
                verified_tools.update(test_tool(agent_file_path, retry, tool_list, trace_exp_id))
    else:
        print("No tools were found during static analysis.")

    print("Discovering additional possible tools...")
    additional_tools = discover_additional_tools(agent_file_path, verified_tools)
    if additional_tools:
        for query in generate_test_queries_additional(additional_tools).values():
            verified_tools.update(test_tool(agent_file_path, query, tool_list, trace_exp_id))

    _write_json(output_json_path, verified_tools)
    print(f"Verified extraction saved {len(verified_tools)} tools to {output_json_path}")
    return verified_tools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and verify an agent's tools.")
    parser.add_argument("--agent-path", required=True, help="Path to the agent Python file.")
    parser.add_argument("--output-json", required=True, help="Final extraction output.")
    parser.add_argument("--code-output-json", required=True, help="Static extraction output.")
    parser.add_argument("--tool-file", required=True, help="JSON list of tools exposed to the agent.")
    parser.add_argument("--trace-exp-id", default="", help="MLflow experiment ID used for verification traces.")
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Run AST extraction without model calls or MLflow traces.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        agent_file_path=args.agent_path,
        output_json_path=args.output_json,
        code_output_json_path=args.code_output_json,
        tool_list=args.tool_file,
        trace_exp_id=args.trace_exp_id,
        static_only=args.static_only,
    )
