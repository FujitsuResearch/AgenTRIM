import os
import json
import csv
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description="Extract and verify agent tools.")
parser.add_argument("--exp_dir", type=str, default="extractor_eval/test_eval", help="Name of folder of evaluations.")
args = parser.parse_args()

root_dir = args.exp_dir
GENERATED_FOLDER = os.path.join(root_dir, "generated_lists")
EXTRACTED_FOLDER = os.path.join(root_dir, "extracted_tools")
SUMMARY_CSV = os.path.join(root_dir, "tool_occurrence_summary.csv")

# Map index -> filename for alignment
generated_map = {
    int(f.split("_")[-1].split(".")[0]): f
    for f in os.listdir(GENERATED_FOLDER)
    if f.endswith(".json")
}
extracted_map = {
    int(f.split("_")[-1].split(".")[0]): f
    for f in os.listdir(EXTRACTED_FOLDER)
    if f.endswith(".json")
}

all_indices = sorted(generated_map.keys())  # process every generated list

tool_stats = defaultdict(lambda: {
    "in_generated": 0,
    "in_generated_and_extracted": 0,
    "in_generated_not_extracted": 0,
    "in_extracted_not_generated": 0
})

for idx in all_indices:
    gen_file = os.path.join(GENERATED_FOLDER, generated_map[idx])
    ext_file = os.path.join(EXTRACTED_FOLDER, f"extracted_tools_{idx}.json")

    # Load generated tools
    with open(gen_file, "r") as f:
        gen_tools = json.load(f)
    gen_set = set(gen_tools)

    # Check if extracted file exists
    if os.path.exists(ext_file):
        with open(ext_file, "r") as f:
            ext_data = json.load(f)
            ext_tools = list(ext_data.keys()) if isinstance(ext_data, dict) else ext_data
        ext_set = set(ext_tools)
    else:
        # No extraction file: nothing was extracted
        ext_set = set()

    # Update stats for tools in generated list
    for tool in gen_tools:
        tool_stats[tool]["in_generated"] += 1
        if tool in ext_set:
            tool_stats[tool]["in_generated_and_extracted"] += 1
        else:
            tool_stats[tool]["in_generated_not_extracted"] += 1

    # Update stats for tools extracted but not in generated list
    for tool in (ext_set - gen_set):
        tool_stats[tool]["in_extracted_not_generated"] += 1

# Save summary to CSV
with open(SUMMARY_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "tool_name",
        "in_generated",
        "in_generated_and_extracted",
        "in_generated_not_extracted",
        "in_extracted_not_generated"
    ])
    for tool, stats in sorted(tool_stats.items()):
        writer.writerow([
            tool,
            stats["in_generated"],
            stats["in_generated_and_extracted"],
            stats["in_generated_not_extracted"],
            stats["in_extracted_not_generated"]
        ])

print(f"✅ Tool occurrence summary saved to {SUMMARY_CSV}")

analyse_specific = False
if analyse_specific:
    tool_to_find_in_extracted = "unit_conversion_tool"
    tool_to_find_not_extracted = "wiki_wiki_scrape"

    dalle_tool_extracted_indices = []
    gmail_read_tool_missing_indices = []

    # Map index -> file for alignment
    generated_map = {
        int(f.split("_")[-1].split(".")[0]): f
        for f in os.listdir(GENERATED_FOLDER)
        if f.endswith(".json")
    }
    extracted_map = {
        int(f.split("_")[-1].split(".")[0]): f
        for f in os.listdir(EXTRACTED_FOLDER)
        if f.endswith(".json")
    }

    all_indices = sorted(generated_map.keys())

    for idx in all_indices:
        gen_file = os.path.join(GENERATED_FOLDER, generated_map[idx])
        ext_file = os.path.join(EXTRACTED_FOLDER, f"extracted_tools_{idx}.json")

        # Load generated tools
        with open(gen_file, "r") as f:
            gen_tools = json.load(f)

        # Load extracted tools if file exists
        if os.path.exists(ext_file):
            with open(ext_file, "r") as f:
                ext_data = json.load(f)
                ext_tools = list(ext_data.keys()) if isinstance(ext_data, dict) else ext_data
        else:
            ext_tools = []

        # 1️⃣ Check for dalle_tool in extracted
        if tool_to_find_in_extracted in ext_tools:
            dalle_tool_extracted_indices.append(idx)

        # 2️⃣ Check for gmail_read_tool in generated but NOT in extracted
        if tool_to_find_not_extracted in gen_tools and tool_to_find_not_extracted not in ext_tools:
            gmail_read_tool_missing_indices.append(idx)

    print(f"Indices where {tool_to_find_in_extracted} was extracted:")
    print(dalle_tool_extracted_indices)
    print(f"\nIndices where {tool_to_find_not_extracted} was in generated list but not extracted:")
    print(gmail_read_tool_missing_indices)