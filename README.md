# AGENTRIM Official Code

Official implementation of **AGENTRIM**, accepted to **Findings of EMNLP 2026**.

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2601.12449)

![AGENTRIM overview](repo_image.png)

This repository contains the real LangGraph agent used in the extractor experiments, the tool extractor and its 500-perturbation experiment, and the AgentDojo adapter for the tool orchestrator.

## Installation

Python 3.12 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

The Gmail tools in this public artifact use deterministic mock data and do not require OAuth credentials. The screenshot tool may require `python -m playwright install firefox`.

## Tool Extractor experiment

The full experiment generates verification queries with Azure OpenAI, invokes the real agent with each perturbed tool list, and reads its MLflow traces. Configure `.env`, start the MLflow service used by the agent, then run:

```bash
python extractor/run_500_experiment.py \
  --trace-exp-id YOUR_MLFLOW_EXPERIMENT_ID \
  --total-number 500
```

The generator uses seed `42`, begins with the full tool set and each singleton, then fills the remainder with unique non-empty random subsets. Outputs go to `outputs/extractor_500/`.

Analyze a completed run:

```bash
python extractor/analyze_results.py --exp_dir outputs/extractor_500
python extractor/compute_metrics.py --root_dir outputs --total_tools 20
```

## AgentDojo benchmark

Configure `.env`, then run:

```bash
python agentdojo_orchestrator/benchmark_eval.py \
  --suites workspace,slack,travel,banking \
  --agents dynamic_planner \
  --attacks both \
  --attacks-list important_instructions \
  --defenses-list none \
  --logdir outputs/agentdojo
```

Repeat with `--agents baseline` for comparison. 

## Contents

- `agent_scripts/`: real LangGraph ReAct agent, tool list, and MCP configuration.
- `tool_files/`: real local and MCP tool implementations.
- `extractor/`: extractor, 500-perturbation runner, and analysis scripts.
- `agentdojo_orchestrator/`: dynamic adapter, baseline, benchmark, and suite inventories.

## Citation

If you use AGENTRIM in your research, please cite:

```bibtex
@inproceedings{betser2026agentrim,
  title={Agentrim: Tool risk mitigation for agentic ai},
  author={Betser, Roy and Giloni, Amit and Bose, Shamik and Padakandla, Sindhu and Picardi, Chiara and Erez, Lidor and Vainshtein, Roman},
  booktitle={EMNLP 2026 (Findings)},
  year={2026}
}
```

## License

AGENTRIM is available for noncommercial use. Third-party software and assets remain subject to their respective licenses.
