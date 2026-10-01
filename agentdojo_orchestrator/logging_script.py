# ===================== logging_script.py =====================
# Per-task JSONL + Markdown tracing for AgentDojo pipelines.

import os, json, dataclasses
from typing import Any, Sequence, Dict, Optional, List
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop
from datetime import datetime
from uuid import uuid4

# -------- helpers --------
def _to_jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, float, str)): return x
    if dataclasses.is_dataclass(x):
        try: return _to_jsonable(dataclasses.asdict(x))
        except Exception: return repr(x)
    if hasattr(x, "model_dump"):
        try: return _to_jsonable(x.model_dump())
        except Exception: pass
    if hasattr(x, "dict"):
        try: return _to_jsonable(x.dict())
        except Exception: pass
    if isinstance(x, dict): return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)): return [_to_jsonable(v) for v in x]
    if hasattr(x, "__dict__"):
        try: return {k: _to_jsonable(v) for k, v in x.__dict__.items() if not k.startswith("_")}
        except Exception: return repr(x)
    return repr(x)

def _serialize_messages(msgs: Any) -> Any:
    try: return _to_jsonable(msgs)
    except Exception: return str(msgs)

def _as_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict): return x
    if hasattr(x, "model_dump"):
        try: return x.model_dump()
        except Exception: pass
    if hasattr(x, "dict"):
        try: return x.dict()
        except Exception: pass
    try:
        return {k: getattr(x, k) for k in dir(x) if not k.startswith("_") and not callable(getattr(x, k, None))}
    except Exception:
        return {"repr": repr(x)}

def _msg_get(msg: Any, key: str, default=None):
    if isinstance(msg, dict): return msg.get(key, default)
    return getattr(msg, key, default)

def _content_text(msg: Any) -> str:
    c = _msg_get(msg, "content")
    if isinstance(c, list):
        parts = []
        for it in c:
            if isinstance(it, dict) and it.get("type") == "text":
                parts.append(str(it.get("content", "")))
        return "\n".join(parts).strip()
    if isinstance(c, str): return c.strip()
    return ""

def _truncate(s: str, n: int = 220) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rstrip() + " …"

# -------- tracing state --------
class _TraceState:
    def __init__(self, logdir: str):
        self.logdir = logdir; self.counter = 0; self.current_task_id: Optional[str] = None; self.step = 0
        os.makedirs(self.logdir, exist_ok=True)
    def _paths(self, tid: str): return (os.path.join(self.logdir, f"{tid}.jsonl"), os.path.join(self.logdir, f"{tid}.compact.md"))
    def start_task(self, extra_args: Optional[Dict[str, Any]], user_query: str):
        provided = (extra_args or {}).get("user_task_id") or (extra_args or {}).get("task_id")
        if provided:
            self.current_task_id = str(provided)
        else:
            self.counter += 1
            self.current_task_id = f"task_{self.counter:03d}"
        self.step = 0
        jl, md = self._paths(self.current_task_id)
        with open(md, "w", encoding="utf-8") as f:
            f.write(f"# {self.current_task_id}\n\n**User:** {user_query}\n\n## Steps\n")
        open(jl, "a", encoding="utf-8").close()
    def end_task(self): self.current_task_id = None; self.step = 0
    def log_jsonl(self, obj: Dict[str, Any]):
        tid = self.current_task_id or "task_unknown"; jl, _ = self._paths(tid)
        with open(jl, "a", encoding="utf-8") as f: f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
    def log_compact(self, line: str):
        tid = self.current_task_id or "task_unknown"; _, md = self._paths(tid)
        with open(md, "a", encoding="utf-8") as f: f.write(line + "\n")
    def next_step(self) -> int: self.step += 1; return self.step

# -------- instrumented elements --------
class _ElementLogger(BasePipelineElement):
    def __init__(self, inner: BasePipelineElement, label: str, state: _TraceState):
        super().__init__(); self.inner = inner; self.label = label; self.state = state
    def query(self, query, runtime, env, messages, extra_args):
        self.state.log_jsonl({"event":"element_start","element":self.label,"query_in":query,"messages_in":_serialize_messages(messages)})
        before_len = len(messages) if isinstance(messages, list) else 0
        q2, r2, e2, msgs2, extra2 = self.inner.query(query, runtime, env, messages, extra_args)
        self.state.log_jsonl({"event":"element_end","element":self.label,"query_out":q2,"messages_out":_serialize_messages(msgs2)})
        if isinstance(msgs2, list) and len(msgs2) >= before_len:
            for m in msgs2[before_len:]:
                role = _msg_get(m, "role") or getattr(m, "type", None)
                tcs = _msg_get(m, "tool_calls") or ((_msg_get(m, "additional_kwargs") or {}).get("tool_calls"))
                if role == "assistant" and tcs:
                    for tc in tcs:
                        d = _as_dict(tc); fn = d.get("function") or d.get("name") or d.get("tool_name") or "unknown_tool"
                        args = d.get("args") or {}
                        try: args_str = json.dumps(args, ensure_ascii=False, default=str)
                        except Exception: args_str = str(args)
                        self.state.log_compact(f"{self.state.next_step():02d}. → **CALL** `{fn}` {args_str}")
                elif role == "tool":
                    meta = _msg_get(m, "tool_call"); d = _as_dict(meta) if meta is not None else {}
                    fn = d.get("function") or d.get("name") or _msg_get(m, "name") or "tool"
                    out = _truncate(_content_text(m))
                    self.state.log_compact(f"{self.state.next_step():02d}. ✔ **RESULT** `{fn}` → {('`'+out+'`') if out else '(no text)'}")
                elif role == "assistant" and _msg_get(m, "content") is not None:
                    txt = _truncate(_content_text(m))
                    if txt: self.state.log_compact(f"{self.state.next_step():02d}. 🗣 **ASSISTANT** → {txt}")
        return q2, r2, e2, msgs2, extra2

class _TapStart(BasePipelineElement):
    def __init__(self, state: _TraceState): super().__init__(); self.state = state
    def query(self, query, runtime, env, messages, extra_args):
        self.state.start_task(extra_args, user_query=query)
        self.state.log_jsonl({"event":"task_start","task_id":self.state.current_task_id,"query":query,"messages":_serialize_messages(messages)})
        return query, runtime, env, messages, extra_args

class _TapEnd(BasePipelineElement):
    def __init__(self, state: _TraceState): super().__init__(); self.state = state
    def query(self, query, runtime, env, messages, extra_args):
        last_assistant = None
        if isinstance(messages, list):
            for m in reversed(messages):
                if isinstance(m, dict) and m.get("role") == "assistant": last_assistant = m; break
        self.state.log_jsonl({"event":"task_end","final_query":query,"final_messages":_serialize_messages(messages),"final_assistant":_to_jsonable(last_assistant)})
        final_text = _truncate(_content_text(last_assistant or {}))
        if final_text: self.state.log_compact(f"\n**FINAL:** {final_text}\n")
        self.state.end_task(); return query, runtime, env, messages, extra_args

# -------- public API --------
def attach_logging(pipeline, logdir: str = "./runs_traces"):
    state = _TraceState(logdir); new_elements = [_TapStart(state)]
    for el in pipeline.elements:
        if isinstance(el, ToolsExecutionLoop) and isinstance(getattr(el, "elements", None), list):
            el.elements = [_ElementLogger(inner, f"ToolsLoop::{inner.__class__.__name__}", state) for inner in el.elements]
        new_elements.append(_ElementLogger(el, el.__class__.__name__, state))
    new_elements.append(_TapEnd(state)); pipeline.elements = new_elements
    return pipeline

# --- ADD to logging_script.py (no edits to existing functions) ---
def attach_logging_offline(pipeline, logroot: str = "./runs_traces"):
    """
    Attach your existing tracer to a temporary subdir, then provide a
    finalize() function that moves the produced JSONL into `logroot`
    as a uniquely named single file: trace_<ts>_<uuid>.jsonl.
    Usage:
        pipe, finalize = attach_logging_unique_file(pipeline, "./runs_traces")
        # ... run ONE agent query ...
        final_path = finalize()  # returns path to the unique JSONL in logroot
    """
    import os, glob, shutil, time
    from tempfile import mkdtemp
    from datetime import datetime
    from uuid import uuid4

    # 1) temp dir for the existing tracer (keeps current behavior intact)
    tmp_dir = mkdtemp(prefix="trace_tmp_", dir=logroot if os.path.isdir(logroot) else None)
    os.makedirs(logroot, exist_ok=True)

    # 2) attach your existing logger to that temp dir
    attach_logging(pipeline, logdir=tmp_dir)

    # 3) prepare a unique target filename in the SINGLE destination dir
    def _unique_name():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"trace_{ts}_{uuid4().hex[:6]}.json"

    target_path = os.path.join(logroot, _unique_name())

    def finalize():
        """Convert the freshest .jsonl in tmp_dir to a single .json file in logroot with a unique name."""
        import os, glob, time, json

        time.sleep(0.05)  # allow buffers to flush

        jsonls = [p for p in glob.glob(os.path.join(tmp_dir, "*.jsonl")) if os.path.isfile(p)]
        if not jsonls:
            candidate = os.path.join(tmp_dir, "events.jsonl")
            if os.path.isfile(candidate):
                jsonls = [candidate]
        if not jsonls:
            return None

        src = max(jsonls, key=os.path.getmtime)

        # ensure target_path uses .json and is unique
        nonlocal target_path
        base, _ = os.path.splitext(target_path)
        target_path = base + ".json"
        while os.path.exists(target_path):
            target_path = base + "_dup.json"
            base = base + "_dup"

        # read JSONL -> list[json]
        with open(src, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        # write pretty JSON
        with open(target_path, "w", encoding="utf-8") as out:
            json.dump(records, out, ensure_ascii=False, indent=2)

        # optional pointer to latest
        latest_ptr = os.path.join(logroot, "LATEST_PATH.txt")
        try:
            with open(latest_ptr, "w", encoding="utf-8") as f:
                f.write(target_path)
        except Exception:
            pass

        # cleanup temp dir
        try:
            for p in glob.glob(os.path.join(tmp_dir, "*")):
                try:
                    os.remove(p)
                except Exception:
                    pass
            os.rmdir(tmp_dir)
        except Exception:
            pass

        return target_path

    return pipeline, finalize