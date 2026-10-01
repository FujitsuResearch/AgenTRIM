"""Generate and evaluate 500 deterministic perturbations of an agent tool set."""

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path


def generate_tool_subsets(tools: list[str], total: int, seed: int) -> list[list[str]]:
    tools = list(dict.fromkeys(tools))
    if not tools:
        raise ValueError("The tool universe must not be empty.")
    maximum = (2 ** len(tools)) - 1
    if total > maximum:
        raise ValueError(f"Cannot generate {total} non-empty unique subsets from {len(tools)} tools.")

    subsets = [tools.copy()]
    if total == 500:
        subsets.extend([[tool] for tool in tools])

    seen = {frozenset(subset) for subset in subsets}
    rng = random.Random(seed)
    while len(subsets) < total:
        size = rng.randint(1, len(tools))
        subset = rng.sample(tools, size)
        key = frozenset(subset)
        if key not in seen:
            seen.add(key)
            subsets.append(subset)
    return subsets[:total]


def normalize(name: str) -> str:
    return name.removeprefix("functions.")


def load_extracted(path: Path) -> list[str]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data) if isinstance(data, dict) else data


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, default=root / "outputs" / "extractor_500")
    parser.add_argument("--tool-file", type=Path, default=root / "agent_scripts" / "langraph_react_tools_list.json")
    parser.add_argument("--agent-file", type=Path, default=root / "agent_scripts" / "langraph_react_agent.py")
    parser.add_argument("--extractor", type=Path, default=root / "extractor" / "full_extractor_pipeline.py")
    parser.add_argument("--trace-exp-id", default="")
    parser.add_argument("--total-number", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--max-runs", type=int, help="Limit executions for a quick integration check.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tools = json.loads(args.tool_file.read_text(encoding="utf-8"))
    subsets = generate_tool_subsets(tools, args.total_number, args.seed)

    generated_dir = args.exp_dir / "generated_lists"
    extracted_dir = args.exp_dir / "extracted_tools"
    logs_dir = args.exp_dir / "run_logs"
    for directory in (generated_dir, extracted_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for index, subset in enumerate(subsets):
        (generated_dir / f"tools_list_{index}.json").write_text(
            json.dumps(subset, indent=2), encoding="utf-8"
        )

    manifest = {
        "seed": args.seed,
        "total_number": len(subsets),
        "tool_universe_size": len(tools),
        "unique_subsets": len({frozenset(value) for value in subsets}),
    }
    (args.exp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated {len(subsets)} unique perturbations in {generated_dir}")
    if args.generate_only:
        return

    run_count = len(subsets) if args.max_runs is None else min(args.max_runs, len(subsets))
    summary = args.exp_dir / "comparison_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instance_number", "number of tools", "true_matches", "only_in_input", "only_in_extracted"])
        for index, subset in enumerate(subsets[:run_count]):
            input_path = generated_dir / f"tools_list_{index}.json"
            output_path = extracted_dir / f"extracted_tools_{index}.json"
            code_path = extracted_dir / f"extracted_tools_code_{index}.json"
            log_path = logs_dir / f"run_log_{index}.txt"
            command = [
                sys.executable,
                str(args.extractor),
                "--agent-path", str(args.agent_file),
                "--output-json", str(output_path),
                "--code-output-json", str(code_path),
                "--tool-file", str(input_path),
                "--trace-exp-id", args.trace_exp_id,
            ]
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

            expected = {normalize(name) for name in subset}
            extracted = {normalize(name) for name in load_extracted(output_path)}
            writer.writerow([
                index,
                len(expected),
                len(expected & extracted),
                len(expected - extracted),
                len(extracted - expected),
            ])
            handle.flush()
    print(f"Completed {run_count} extractor runs; summary saved to {summary}")


if __name__ == "__main__":
    main()
