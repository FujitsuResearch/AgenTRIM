# --------------------------------------------------------------------------------------
# AgentDojo Evaluation
# --------------------------------------------------------------------------------------
# This script runs automated AgentDojo benchmarks across multiple suites, agents,
# defenses, and attack configurations.
#
# It supports:
#   - Loading and running benchmark suites (workspace, travel, banking, etc.)
#   - Dynamic pipeline construction (baseline, base adapter, dynamic planner)
#   - Optional attack and defense combinations
#   - Tool description masking for controlled LLM exposure
#   - Incremental result logging to JSONL summaries
#
# Usage:
#   python run_benchmark.py --suites all --agents dynamic_planner --attacks both
#                           --defenses-list all --tool-desc-json ./tool_overrides
#
# The script assumes an initialized environment (.env) with OpenAI or Azure keys.
# --------------------------------------------------------------------------------------

import os
import re
import warnings
from pathlib import Path
import importlib.util
import importlib
import argparse
import json
import copy
from typing import Dict
from pydantic import create_model, Field
import openai
from openai import AzureOpenAI
from dotenv import load_dotenv
import sys
def setup_paths():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_directory)
    grandparent_dir = os.path.dirname(parent_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    if grandparent_dir not in sys.path:
        sys.path.append(grandparent_dir)
setup_paths()
load_dotenv(dotenv_path=Path(".env"), override=False)

# --- AgentDojo: suite + benchmark ---
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
import agentdojo.logging as dojo_logging
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig

# Attacks registry & loader
from agentdojo.attacks.attack_registry import ATTACKS
from agentdojo.attacks import load_attack
import agentdojo.attacks as attacks_pkg  # force-import submodules for registration
from agentdojo.agent_pipeline.pi_detector import PromptInjectionDetector
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement

# --- Tracer you already have ---
from logging_script import attach_logging

# --- Adapters ---
from baseline_adapter import BaselineCloneAdapter
# from tool_orchestrator_adapter import DynamicAdapter
from tool_orchestrator_adapter import DynamicAdapter

warnings.filterwarnings(
    "ignore",
    message=r"^Cannot log executed benchmark",
    category=UserWarning,
    module="agentdojo.logging",
)

# --------------------------------------------------------------------------------------
# Benchmark resolver
# --------------------------------------------------------------------------------------
def _resolve_bench_fns():
    import agentdojo.benchmark as B
    bench_no = getattr(B, "benchmark_suite_without_injections", None)
    bench_yes = getattr(B, "benchmark_suite_with_injections", None)
    if not callable(bench_no) or not callable(bench_yes):
        raise RuntimeError("Expected benchmark_suite_without_injections / benchmark_suite_with_injections in agentdojo.benchmark.")
    return bench_no, bench_yes

bench_no_attacks, bench_with_attacks = _resolve_bench_fns()

# --------------------------------------------------------------------------------------
# Arg parsing
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run AgentDojo eval over suites/agents with/without attacks.")
    p.add_argument("--suite-version", "-V", default="v1", help="Suite version (default: 'v1')")
    p.add_argument(
        "--suites", "-S",
        default="slack",
        help="Comma-separated suite names or 'all' to run every suite in the version (e.g., workspace, travel, banking, slack, all))"
    )
    p.add_argument(
        "--agents", "-A",
        # choices=["base_adap", "baseline", "dynamic_planner", "all"],
        default="dynamic_planner_e",
        help="Which agent(s) to run (default: dynamic_planner)"
    )
    p.add_argument(
        "--attacks", "-T",
        choices=["none", "with", "both"],
        default="both",
        help="Run without attacks, with attacks, or both (default: both)"
    )
    p.add_argument(
        "--attacks-list", "-L",
        default="important_instructions",
        help="Comma-separated list of attacks to run from the allowed set "
             "(direct, ignore_previous, system_message, important_instructions, tool_knowledge). "
             "Default: all"
    )
    p.add_argument(
        "--defenses-list", "-D",
        default="none",
        help="Comma-separated list of defenses to run from the allowed set "
             "(none, tool_filter, transformers_pi_detector, repeat_user_prompt, spotlighting_with_delimiting). "
             "Default: none"
    )
    p.add_argument(
        "--logdir",
        default="runs_dojo_tests_4_1",
        help="Root output dir for suite run logs (default: ./dojo_runs/runs_dojo_<version>)"
    )
    p.add_argument(
        "--tool-desc-json",
        default="",
        help="Directory with per-suite JSON files named '<suite>.json' (agentdojo_scripts/tool_overrides). to use with no iverrides, keep empty."
    )
    p.add_argument("--force-rerun", action="store_true", help="Force re-run benchmarks even if cached")
    return p.parse_args()

# --------------------------------------------------------------------------------------
# Patches & helpers
# --------------------------------------------------------------------------------------
def normalize_agent_kind(agent_kind) -> str:
    """
    Normalize various agent kind identifiers to a canonical form.

    - Lowercases and trims the input for robust matching.
    - If the token 'dynamic' appears anywhere, maps to 'dynamic_planner'.
      (Keeps backward compatibility with multiple dynamic-* variants.)
    """
    # Make it a clean lowercase string for matching
    s = str(agent_kind).strip().lower()
    # If it contains "dynamic" anywhere, collapse to the canonical name
    if re.search(r"dynamic", s):
        return "dynamic_planner"
    return str(agent_kind).strip()

def build_openai_llm() -> OpenAILLM:
    """
    Build an OpenAI client wrapped in the AgentDojo OpenAILLM adapter.

    Reads deployment name from AZURE_DEPLOYMENT_NAME, defaulting to 'gpt-4o-2024-05-13'.
    Assumes environment variables for authentication are already configured.
    """

    client = openai.OpenAI()
    model = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-2024-05-13")
    return OpenAILLM(client, model)

def patch_null_logger_has_logdir() -> None:
    """
    Ensure dojo_logging.NullLogger exposes a 'logdir' attribute.

    Some code paths may expect a .logdir attribute; this patch adds it
    if missing to avoid AttributeError without changing logger behavior.
    """

    if not hasattr(dojo_logging.NullLogger, "logdir"):
        dojo_logging.NullLogger.logdir = None

def patch_agentdojo_runtime_for_baseline() -> None:
    """
    Apply baseline-friendly patches to AgentDojo runtime.

    - Forces NullLogger.logdir to None (no-op logging directory).
    - Disables prompt injection detection by overriding detect() with a no-op.
      Useful for ablation/baseline runs where guards are intentionally off.
    """

    dojo_logging.NullLogger.logdir = None
    PromptInjectionDetector.detect = lambda self, text: False  # no-op

def _maybe_route_openai_to_azure():
    """
    If AZURE_OPENAI_ENDPOINT is defined, monkey-patch openai.OpenAI()
    to return an AzureOpenAI client instead.

    This provides transparent routing for environments configured to use
    Azure OpenAI without touching upstream call sites.
    """

    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        openai.OpenAI = lambda: AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION", "2024-06-01"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

def apply_tool_desc_overrides(suite, overrides: dict[str, str]) -> dict[str, str]:
    """Mutate suite.tools descriptions; return {tool_name: original_desc} for restoration.

    Notes:
        - Expects `suite.tools` to be an iterable of tool-like objects with `.name` and `.description`.
        - Only tools present in both `suite.tools` and `overrides` are modified.
        - Missing tools are warned about (stdout) but do not raise.
        - Returns a mapping of original descriptions for later restoration.
    """
    saved = {}
    # Index tools by name for O(1) lookup
    by_name = {getattr(t, "name", None): t for t in getattr(suite, "tools", [])}
    for name, new_desc in overrides.items():
        tool = by_name.get(name)
        if tool is None:
            print(f"[desc-override] WARN: tool '{name}' not found in suite '{getattr(suite,'name',None)}'")
            continue
        orig = getattr(tool, "description", None)
        saved[name] = orig
        try:
            setattr(tool, "description", new_desc)              # Mutate in place
        except Exception:
            # Protect against read-only attributes or unexpected tool shapes
            print(f"[desc-override] WARN: failed to set description for '{name}'")
    return saved

def restore_tool_desc_overrides(suite, saved: dict[str, str]) -> None:
    """Restore original descriptions after run.

    Args:
        suite: Suite whose tools were previously overridden.
        saved: Mapping {tool_name: original_description} returned by apply_tool_desc_overrides().
    """
    
    # Rebuild name→tool index (suite.tools may have changed references)
    by_name = {getattr(t, "name", None): t for t in getattr(suite, "tools", [])}
    for name, orig in saved.items():
        tool = by_name.get(name)
        if tool is None:
            continue                # Tool removed or renamed; silently skip
        try:
            setattr(tool, "description", orig)          # Best-effort restore
        except Exception:
            pass                    # Ignore restore failures to avoid interrupting teardown

def build_baseline_pipeline():
    """Load and return the AgentPipeline instance from the baseline example.

    Behavior:
        - Optionally reroutes OpenAI client construction to Azure if env is set.
        - Dynamically imports `agentdojo/examples/pipeline.py`.
        - Scans the loaded module for an exported AgentPipeline instance and returns it.

    Raises:
        FileNotFoundError: If the baseline example file is missing.
        RuntimeError: If the module does not export an AgentPipeline instance.
    """
    
    _maybe_route_openai_to_azure()
    module_path = Path("agentdojo/examples/pipeline.py").resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Baseline example not found at {module_path}")

    # Import the example module by path without adding to sys.path
    spec = importlib.util.spec_from_file_location("baseline_mod", str(module_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Return the first AgentPipeline instance defined in the module
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, AgentPipeline):
            return obj

    raise RuntimeError("No AgentPipeline instance exported by agentdojo/examples/pipeline.py")

def build_base_adap_pipeline(defense: str | None = None):
    """
    Construct an AgentPipeline using the BaselineCloneAdapter, optionally with a defense.

    Args:
        defense: One of:
            - "none" (default): plain baseline clone adapter
            - "spotlighting_with_delimiting": use adapter-implemented spotlighting defense
            - any defense supported by PipelineConfig (e.g., "tool_filter",
              "transformers_pi_detector", "repeat_user_prompt")

    Notes:
        - Reuses a single LLM object across components (compatible with Azure/OpenAI).
        - The `.name` attribute is set on the LLM if missing to satisfy tool-filter paths.
    """

    _maybe_route_openai_to_azure()
    defense = (defense or "none").lower()

    # Use the same LLM object everywhere (works with Azure/OpenAI)
    llm = build_openai_llm()
    model_name = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-2024-05-13")
    if not getattr(llm, "name", None):
        llm.name = model_name  # required by tool_filter path

    if defense == "spotlighting_with_delimiting":
        # Bypass AgentDojo factory recursion bug; our adapter implements spotlighting safely
        return AgentPipeline([
            BaselineCloneAdapter(
                llm=llm,
                defense_name="spotlighting_with_delimiting",
            )
        ])

    if defense != "none":
        # Build via config to enable built-in defenses
        cfg = PipelineConfig(
            llm=llm,                          # pass the object to avoid ModelsEnum issues
            system_message_name="default",
            system_message="You are a helpful assistant.",
            tool_output_format="yaml",
            defense=defense,                  # tool_filter | transformers_pi_detector | repeat_user_prompt
        )
        return AgentPipeline.from_config(cfg)

    # No defense
    return AgentPipeline([BaselineCloneAdapter(llm=llm, defense_name="none")])

def build_dynamic_adap_pipeline(system_message: str = "You are a helpful assistant.") -> AgentPipeline:
    """
    Construct an AgentPipeline using the DynamicAdapter (planner + restricted tools).

    Args:
        system_message: System prompt provided to the dynamic adapter/LLM.

    Notes:
        - Ensures Azure routing when configured.
        - Ensures LLM has a `.name` attribute (used by tool filter code paths).
        - DynamicAdapter internally handles planning and tool restriction.
    """

    # Adapter builds the OpenAI/Azure LLM internally via build_openai_llm()
    _maybe_route_openai_to_azure()
    llm = build_openai_llm()
    model_name = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-2024-05-13")
    if not getattr(llm, "name", None):
        llm.name = model_name  # required by tool_filter path
    
    return AgentPipeline([
        DynamicAdapter(
            llm=llm,
            prints=False
        )
    ])

def build_pipeline(agent_kind: str, defense: str | None = None) -> AgentPipeline:
    """
    Unified factory for building a pipeline based on agent kind.

    Args:
        agent_kind: "baseline" | "dynamic_planner" | anything else (falls back to base adapter).
        defense: Optional defense string forwarded to the baseline branch.

    Behavior:
        - "baseline": patches runtime for baseline behavior and loads the baseline example pipeline.
        - "dynamic_planner": patches logger and returns a DynamicAdapter pipeline.
        - default: patches logger and returns a base adapter pipeline with the chosen defense.
    """

    if agent_kind == "baseline":
        patch_agentdojo_runtime_for_baseline()
        return build_baseline_pipeline()
    elif agent_kind == "dynamic_planner":
        patch_null_logger_has_logdir()
        return build_dynamic_adap_pipeline()
    else:
        patch_null_logger_has_logdir()
        return build_base_adap_pipeline(defense or "none")

def _ensure_attacks_registered():
    """
    Force import-time side effects to register all attack sets.

    Accessing attributes on attacks_pkg ensures that the submodules are loaded
    and their registration code runs (even if we don't use the values here).
    """
    _ = attacks_pkg.baseline_attacks, attacks_pkg.important_instructions_attacks, attacks_pkg.dos_attacks

ALLOWED_ATTACKS = {
    "direct",
    "ignore_previous",
    "system_message",
    "important_instructions",
    "tool_knowledge",
}

def _get_attack_names(selected: str) -> list:
    """
    Parse a user-specified attack selection into a validated list of attack names.

    - "all" → all known attacks intersected with ALLOWED_ATTACKS.
    - Comma-separated list → validated against ALLOWED_ATTACKS.
    """
    
    _ensure_attacks_registered()
    if selected.strip().lower() == "all":
        return [name for name in ATTACKS.keys() if name in ALLOWED_ATTACKS]
    else:
        req = [s.strip() for s in selected.split(",") if s.strip()]
        for r in req:
            if r not in ALLOWED_ATTACKS:
                raise ValueError(f"Invalid attack '{r}'. Allowed: {sorted(ALLOWED_ATTACKS)}")
        return req

ALLOWED_DEFENSES = [
    "none",
    "tool_filter",
    "transformers_pi_detector",
    "repeat_user_prompt",
    "spotlighting_with_delimiting",
]

def _get_defense_names(spec: str) -> list[str]:
    """
    Normalize/validate a defense specification.

    - "all" → every defense except "none".
    - Comma-separated list → each entry must be in ALLOWED_DEFENSES.
    - Empty/None → ["none"].
    """
    
    s = (spec or "none").strip().lower()
    if s == "all":
        # all built-ins except "none"
        return [d for d in ALLOWED_DEFENSES if d != "none"]
    req = [x.strip() for x in s.split(",") if x.strip()]
    for r in req:
        if r not in ALLOWED_DEFENSES:
            raise ValueError(f"Invalid defense '{r}'. Allowed: {ALLOWED_DEFENSES}")
    return req or ["none"]

def _append_summary_line(root_logdir: str, row: dict) -> None:
    """
    Append a single JSON row to <root_logdir>/summary.jsonl, creating directories as needed.

    The file is newline-delimited JSON (JSONL), one record per line.
    """

    summary_path = os.path.join(root_logdir, "summary.jsonl")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_desc_overrides_for_suite(dirpath: str | None, suite_name: str) -> dict:
    """Load {tool_name: new_description} for a specific suite from dirpath.

    Search order (first match wins):
      1) <dirpath>/<suite_name>.json
      2) <dirpath>/<suite_name.lower()>.json
      3) <dirpath>/<suite_name>/overrides.json

    Returns:
        dict: Mapping of tool_name → description. Empty dict if none found/valid.
    """

    if not dirpath:
        return {}
    base = Path(dirpath)
    
    # Try a few sensible patterns
    candidates = [
        base / f"{suite_name}.json",
        base / f"{suite_name.lower()}.json",
        base / suite_name / "overrides.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                try:
                    m = json.load(f)
                    if isinstance(m, dict):
                        print(f"[desc-override] Loaded {len(m)} overrides from {p}")
                        return m
                    else:
                        print(f"[desc-override] WARN: {p} is not a JSON object; ignoring.")
                except Exception as e:
                    print(f"[desc-override] WARN: failed to read {p}: {e}")
            break
    print(f"[desc-override] No overrides found for suite '{suite_name}' in {dirpath}")
    return {}

def _one_string_model(field: str = "expr"):
    """
    Build a minimal Pydantic model class with a single required string field.

    Purpose:
        - Dynamically creates a Pydantic model whose schema contains only one
          required string field (named by `field`).
        - Returns a subclass that overrides `model_json_schema` to remove the
          top-level "title" from the generated JSON schema (to avoid leaking
          hints in prompts or tool specs).

    Args:
        field: Name of the single string field to include (default: "expr").

    Returns:
        A Pydantic model class (subclass of the dynamically created `Base`)
        that has one required `str` field and no schema title.
    """
    
    Base = create_model("MaskedParams", **{field: (str, Field(..., description=""))})
    class MaskedParams(Base):  # strip schema title to avoid leaking hints
        @classmethod
        def model_json_schema(cls, *a, **k):
            
            # Delegate to base schema generation, then blank out the title
            s = super().model_json_schema(*a, **k)
            s["title"] = ""
            return s
    return MaskedParams

# --------------------------------------------------------------------------------------
# Helper Classes
# --------------------------------------------------------------------------------------
class _MultiToolMaskRuntime:
    """
    Wraps a FunctionsRuntime and, for tools listed in `overrides`,
    presents:
      - a new description (overrides[name])
      - a minimal one-string Pydantic `parameters` model

    Execution is unchanged (delegates to base runtime).

    Notes:
        - This is a *presentation-only* mask: underlying tool execution and true
          parameter schemas are preserved in `self._base`.
        - Useful for spotlighting or simplifying the LLM-facing interface while
          keeping real runtime behavior intact.
    """

    def __init__(self, base_runtime, overrides: Dict[str, str], field: str = "expr"):
        # Base runtime that performs actual execution
        self._base = base_runtime
        # Mapping: tool_name -> replacement description string
        self._overrides = overrides
        # Name of the single string parameter field for masked schemas
        self._field = field
        # Cached, lazily constructed view of function specs as presented to the LLM
        self._presented = None

    def _build(self):
        """
        Lazily construct a masked function-spec dictionary:
        - Shallow-copy specs so the base runtime is never mutated.
        - For overridden names, replace .description and .parameters (if settable).
        """
        
        if self._presented is not None:
            return
        presented = {}
        for name, spec in self._base.functions.items():
            if name in self._overrides:
                s = copy.copy(spec)             # shallow clone; do not mutate real spec
                try: setattr(s, "parameters", _one_string_model(self._field))
                except Exception: pass          # tolerate specs without a settable parameters attribute
                try: setattr(s, "description", self._overrides[name])
                except Exception: pass          # tolerate specs without a settable description
                presented[getattr(s, "name", name)] = s
            else:
                presented[getattr(spec, "name", name)] = spec
        self._presented = presented

    @property
    def functions(self):
        """LLM-facing function map with masked descriptions/parameters."""
        self._build()
        return self._presented

    def list_functions(self):
        """Return a list of function specs as presented to the LLM."""
        self._build()
        return list(self._presented.values())

    def run_function(self, env, name: str, args: dict):
        # Execution unchanged: delegate to the base runtime.
        return self._base.run_function(env, name, args)

class PreLLMToolMasker(BasePipelineElement):
    """
    Pipeline element that wraps the runtime right before the LLM sees tools.

    `overrides` maps tool_name -> new_description.

    Use cases:
        - Spotlighting: simplify or sharpen tool descriptions for the planner.
        - Safety: redact internal details without affecting execution behavior.
    """

    def __init__(self, overrides: Dict[str, str], field: str = "expr"):
        self.overrides = overrides
        self.field = field

    def query(self, query, runtime, env, messages, extra_args):
        """
        Inject a masked runtime in the pipeline (presentation-only) so that subsequent
        planning steps see modified tool descriptions and a minimal parameter schema.
        """
        if self.overrides:
            runtime = _MultiToolMaskRuntime(runtime, self.overrides, field=self.field)
        return query, runtime, env, messages, extra_args

# --------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------
def main():
    """
    Orchestrate benchmark runs across suites, agents, defenses, and attack modes.

    Workflow:
      - Parse CLI args and resolve suite/version, agent kinds, allowed attacks/defenses.
      - Optionally load per-suite tool description overrides (for LLM-facing masking).
      - For each (suite, agent, defense, attack_mode) combination:
          * Build pipeline (baseline/base_adap/planner_adap).
          * Optionally inject PreLLMToolMasker for presentation-only tool overrides.
          * Run with or without attacks; collect utility/security metrics.
          * Write a JSONL summary row incrementally to <root_logdir>/summary.jsonl.
      - Ensure tool descriptions are restored even on exceptions.
    """
    
    args = parse_args()
    SUITE_VERSION = args.suite_version
    selected_attacks = _get_attack_names(args.attacks_list)
    selected_defenses = _get_defense_names(args.defenses_list)

    # Resolve suites
    all_suites_dict = get_suites(SUITE_VERSION)
    if args.suites.strip().lower() == "all":
        suite_names = list(all_suites_dict.keys())
    else:
        suite_names = [s.strip() for s in args.suites.split(",") if s.strip()]
        missing = [s for s in suite_names if s not in all_suites_dict]
        if missing:
            raise KeyError(
                f"Suites not found in version '{SUITE_VERSION}': {missing}\nAvailable: {list(all_suites_dict.keys())}"
            )
    print("Suites:", suite_names)

    # Agent selection: "all" expands to three families, else single choice
    agent_kinds = ["base_adap", "dynamic_adapter", "baseline"] if args.agents == "all" else [args.agents]
    print("Agents:", agent_kinds)

    # Attack toggle semantics: none / with / both → boolean list
    attack_modes = {
        "none": [False],
        "with": [True],
        "both": [False, True],
    }[args.attacks]

    print("Attacks Checks:", attack_modes)
    if any(attack_modes):
        print(f"Attacks to run: {selected_attacks}")
        print("Defenses:", selected_defenses)

    # Root logging directory (per suite/agent/defense/mode subfolders created later)
    root_logdir = args.logdir or os.path.abspath(f"./runs_dojo_{SUITE_VERSION}")
    os.makedirs(root_logdir, exist_ok=True)

    # Aggregated results (kept in-memory and also streamed to JSONL)
    agg = []

    for suite_name in suite_names:
        suite = all_suites_dict[suite_name]

        # Load per-suite overrides, e.g. <dir>/workspace.json
        # NOTE: use the existing CLI arg name (directory path passed via --tool-desc-json)
        desc_overrides = load_desc_overrides_for_suite(args.tool_desc_json, suite_name)

        # Incrementally write overrides: 1 tool, then 2, then 3, ...
        tool_items = sorted(desc_overrides.items(), key=lambda kv: kv[0])  # stable order by tool name
        override_batches = []  # collect metadata for later logging
        if desc_overrides:
            tool_num = len(tool_items) + 1
        else:
            tool_num = 2
        
        # Currently a single pass (k=0). Kept as a loop for future incremental override batches.
        for k in range(1):

            saved_descs = {}
            if desc_overrides:
                incremental_overrides = {name: spec for name, spec in tool_items[:k]}
                overwrite_count = k  # how many tools are overwritten in this batch
                saved_descs = apply_tool_desc_overrides(suite, incremental_overrides)
                if saved_descs:
                    print(f"[desc-override] Applied to suite '{suite_name}': {list(saved_descs.keys())}")
            else:
                overwrite_count = 0
            
            try:
                for agent_kind in agent_kinds:

                    # For baseline we force a single "none" defense; for base_adap we iterate selected_defenses
                    defenses_for_agent = selected_defenses if agent_kind == "base_adap" else ["none"]

                    for defense_name in defenses_for_agent:
                        for with_attacks in attack_modes:
                            mode = "with_attacks" if with_attacks else "no_attacks"

                            # Include defense in the directory structure so runs don't overwrite each other
                            base_mode_dir = os.path.join(
                                root_logdir, f"{suite_name}", f"{agent_kind}", f"defense_{defense_name}", mode
                            )
                            Path(base_mode_dir).mkdir(parents=True, exist_ok=True)

                            # Attack-enabled branch: iterate over selected attacks
                            if with_attacks:
                                for attack_name in selected_attacks:
                                    
                                    agent_kind_ = normalize_agent_kind(agent_kind)
                                    pipeline = build_pipeline(agent_kind_, defense_name if agent_kind == "base_adap" else None)
                                    pipeline.name = "gpt-4o-2024-05-13"

                                    # Inject LLM-facing parameters mask only for our adapters
                                    if desc_overrides and agent_kind in ("base_adap", "planner_adap"):
                                        try:
                                            pipeline.elements.insert(0, PreLLMToolMasker(incremental_overrides, field="expr"))
                                            print(f"[mask] Injected PreLLMToolMasker for suite '{suite_name}' ({len(incremental_overrides)} tools).")
                                        except Exception as e:
                                            print(f"[mask] WARN: failed to inject masker: {e}")

                                    run_dir = os.path.join(base_mode_dir, attack_name)
                                    traces_dir = os.path.join(run_dir, "_traces")
                                    print(traces_dir)
                                    if Path(traces_dir).exists():
                                        print("skipped")
                                        continue  # skip this loop iteration

                                    # Prepare logging hooks and run the benchmark with attacks
                                    Path(traces_dir).mkdir(parents=True, exist_ok=True)
                                    attach_logging(pipeline, logdir=traces_dir)
                                    attack = load_attack(attack_name, task_suite=suite, target_pipeline=pipeline)

                                    results = bench_with_attacks(
                                        agent_pipeline=pipeline,
                                        suite=suite,
                                        attack=attack,
                                        logdir=Path(run_dir),
                                        force_rerun=args.force_rerun,
                                        user_tasks=None,
                                        injection_tasks=None,
                                        verbose=True,
                                        benchmark_version=SUITE_VERSION,
                                    )

                                    # Summarize metrics (utility/security). Support dict or object results.
                                    util = results.get("utility_results", {}) if isinstance(results, dict) else getattr(results, "utility_results", {})
                                    sec = results.get("security_results", {}) if isinstance(results, dict) else getattr(results, "security_results", {})
                                    util_ok = sum(1 for v in util.values() if v)
                                    asr_count = sum(1 for v in sec.values() if v)
                                    util_total = len(util)
                                    sec_total = len(sec)

                                    print(f"[Suite={suite_name} | Agent={agent_kind} | Defense={defense_name} | Mode={mode}:{attack_name}]")
                                    print(f"  Utility (pass): {util_ok} / {util_total}")
                                    print(f"  Attack success (ASR count): {asr_count} / {sec_total}")

                                    # Stream a JSONL row and also keep in-memory aggregation
                                    row = {
                                        "suite_version": SUITE_VERSION,
                                        "suite": suite_name,
                                        "agent": agent_kind,
                                        "defense": defense_name,
                                        "mode": f"{mode}:{attack_name}",
                                        "overwrite_count": overwrite_count,
                                        "utility_pass": util_ok,
                                        "utility_total": util_total,
                                        "attack_success": asr_count,
                                        "attack_total": sec_total,
                                        "utility_results": {str(k): v for k, v in util.items()},
                                        "security_results": {str(k): v for k, v in sec.items()},
                                        "logdir": run_dir,
                                        "traces_dir": traces_dir,
                                    }
                                    agg.append(row)
                                    _append_summary_line(root_logdir, row)

                            else:
                                # No-attack branch: just run utility evaluation
                                agent_kind_ = normalize_agent_kind(agent_kind)
                                pipeline = build_pipeline(agent_kind_, defense_name if agent_kind == "base_adap" else None)
                                if getattr(pipeline, "name", None) in (None, ""):
                                    pipeline.name = f"{agent_kind}_{suite_name}_noattacks"

                                # Inject LLM-facing parameters mask only for our adapters
                                if desc_overrides and agent_kind in ("base_adap", "planner_adap"):
                                    try:
                                        pipeline.elements.insert(0, PreLLMToolMasker(incremental_overrides, field="expr"))
                                        print(f"[mask] Injected PreLLMToolMasker for suite '{suite_name}' ({len(incremental_overrides)} tools).")
                                    except Exception as e:
                                        print(f"[mask] WARN: failed to inject masker: {e}")

                                run_dir = base_mode_dir
                                traces_dir = os.path.join(run_dir, "_traces")
                                print(traces_dir)
                                if Path(traces_dir).exists():
                                    print("skipped")
                                    continue  # skip this loop iteration
                                
                                # Prepare logging hooks and run the benchmark without attacks
                                Path(traces_dir).mkdir(parents=True, exist_ok=True)
                                attach_logging(pipeline, logdir=traces_dir)
                                results = bench_no_attacks(
                                    agent_pipeline=pipeline,
                                    suite=suite,
                                    user_tasks=None,
                                    logdir=Path(run_dir),
                                    force_rerun=args.force_rerun,
                                    benchmark_version=SUITE_VERSION,
                                )

                                # Summarize utility metrics
                                util = results.get("utility_results", {}) if isinstance(results, dict) else getattr(results, "utility_results", {})
                                util_ok = sum(1 for v in util.values() if v)
                                util_total = len(util)
                                print(f"[Suite={suite_name} | Agent={agent_kind} | Defense={defense_name} | Mode={mode}]")
                                print(f"  Utility (pass): {util_ok} / {util_total}")
                                print(f"  Attack success (ASR count): N/A")

                                # Stream a JSONL row and also keep in-memory aggregation
                                row = {
                                    "suite_version": SUITE_VERSION,
                                    "suite": suite_name,
                                    "agent": agent_kind,
                                    "defense": defense_name,
                                    "mode": mode,
                                    "overwrite_count": overwrite_count,
                                    "utility_pass": util_ok,
                                    "utility_total": util_total,
                                    "attack_success": None,
                                    "attack_total": None,
                                    "utility_results": {str(k): v for k, v in util.items()},      
                                    "security_results": None,    
                                    "logdir": run_dir,
                                    "traces_dir": traces_dir,
                                }
                                agg.append(row)
                                _append_summary_line(root_logdir, row)

            finally:
                # Restore originals so other suites/runs aren’t contaminated
                if saved_descs:
                    restore_tool_desc_overrides(suite, saved_descs)

if __name__ == "__main__":
    main()
