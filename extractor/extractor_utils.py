from mlflow.tracking import MlflowClient
import mlflow
from typing import Set
import json
import subprocess
import ast
import os
import re
import math
import sys
import argparse
from typing import Any, Dict, List, Union
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# ========== Setup ==========
def setup_paths():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_directory)
    grandparent_dir = os.path.dirname(parent_dir)

    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    if grandparent_dir not in sys.path:
        sys.path.append(grandparent_dir)

setup_paths()
load_dotenv(override=False)

# ========== Agent Runner ==========
def run_agent(agent_file_path, query, tool_list):
    """Run the agent with a specific query and return its response."""
    result = subprocess.run(
        [sys.executable, agent_file_path, "--task", query, "--tool_file", tool_list],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Agent exited with status {result.returncode}:\n{result.stderr.strip()}")
    return result.stdout

# ========== Tool Extraction from query ==========
def extract_tools(agent_file_path, tool_list):
    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
        model=os.getenv("AZURE_MODEL_NAME"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY"),
        temperature=1,
    )

    agent_query = "List all the tools you have access to and explain briefly what each one does."
    agent_response = run_agent(agent_file_path, agent_query, tool_list)
    print("\nAgent Response:\n", agent_response)

    extraction_prompt = f"""
    You will be provided with the agent's answer that lists its available tools.

    Extract the tool names, descriptions, and assign each tool a default permission of "usage".

    Return your answer in the following JSON format:
    {{
        "tool_name": {{
            "description": "What the tool does.",
        }},
        ...
    }}


    Agent Response:
    {agent_response}
    """

    llm_response = llm.invoke(extraction_prompt)
    raw_content = llm_response.content.strip()

    if raw_content.startswith("```json"):
        raw_content = re.sub(r"```json", "", raw_content)
    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]

    try:
        tools = json.loads(raw_content.strip())
    except json.JSONDecodeError:
        print("\nFailed to parse JSON from LLM response. Cleaned response:\n", raw_content)
        return {}
    print(tools)
    return tools

# ========== Tool Extraction code ==========
def extract_imported_modules(tree):
    """Extract imported module names from the AST tree."""
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split('.')[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split('.')[0])
    return list(imported)

def resolve_module_path(module_name, base_dir):
    """Try to resolve a module like `gmail_tools` to a .py file path relative to base_dir."""
    module_parts = module_name.split('.')
    path = os.path.join(base_dir, *module_parts) + ".py"
    return path if os.path.isfile(path) else None

def extract_files_from_mcp_config(file_path):
    """Extract .py filenames from args in MultiServerMCPClient({...}) block."""
    filenames = []
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and hasattr(node.func, "id") and node.func.id == "MultiServerMCPClient":
            if node.args and isinstance(node.args[0], ast.Dict):
                config_dict = node.args[0]
                for entry in config_dict.values:
                    if isinstance(entry, ast.Dict):
                        for key_node, val_node in zip(entry.keys, entry.values):
                            if isinstance(key_node, ast.Constant) and key_node.value == "args":
                                if isinstance(val_node, ast.List):
                                    for elt in val_node.elts:
                                        if isinstance(elt, ast.Constant) and elt.value.endswith(".py"):
                                            filenames.append(elt.value)
    return filenames

def extract_json_paths_from_code(file_path: str) -> set:
    base_dir = os.path.dirname(os.path.abspath(file_path))
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    json_paths: set[str] = set()
    assigned_strings: dict[str, str] = {}

    def norm(path: str) -> str:
        # resolve relative to the file's directory
        if not os.path.isabs(path):
            cwd_path = os.path.abspath(path)
            path = cwd_path if os.path.exists(cwd_path) else os.path.normpath(os.path.join(base_dir, path))
        return path

    def add_if_json(s: str):
        if isinstance(s, str) and s.endswith(".json"):
            json_paths.add(norm(s))

    def eval_simple(node):
        """Best-effort constant string evaluator for common cases:
        - string constants
        - names pointing to previously assigned strings
        - os.path.join(<parts>)
        - Path(...) / 'x.json' (very simple)
        """
        # Constant "..." 
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value

        # Name -> previously assigned string
        if isinstance(node, ast.Name):
            return assigned_strings.get(node.id)

        # os.path.join("a","b","c.json")
        if isinstance(node, ast.Call):
            # function name or attribute chain endswith 'join'
            fn = node.func
            fn_name = None
            if isinstance(fn, ast.Name):
                fn_name = fn.id
            elif isinstance(fn, ast.Attribute):
                fn_name = fn.attr
            if fn_name == "join":
                parts = []
                for a in node.args:
                    s = eval_simple(a)
                    if s is None:
                        return None
                    parts.append(s)
                try:
                    return os.path.join(*parts)
                except Exception:
                    return None

            # Path("a") / "b.json" also appears as a Call: Path("a","b.json")
            # Handle Path("...json") directly
            if (isinstance(fn, ast.Name) and fn.id in {"Path", "PurePath"}) or \
               (isinstance(fn, ast.Attribute) and fn.attr in {"Path", "PurePath"}):
                if not node.args:
                    return None
                # If multiple args, join them
                parts = []
                for a in node.args:
                    s = eval_simple(a)
                    if s is None:
                        return None
                    parts.append(s)
                return os.path.join(*parts)

        # Path("a") / "b.json" => BinOp( left=Call(Path(...)), op=Div, right=Constant )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left = eval_simple(node.left)
            right = eval_simple(node.right)
            if left and right:
                return os.path.join(left, right)

        return None

    # Pass 1: capture simple string assignments: x = "file.json"
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id
                val = eval_simple(node.value)
                if isinstance(val, str):
                    assigned_strings[var_name] = val
                    add_if_json(val)

    # Pass 2: function calls (positional and keyword args), including argparse defaults
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # positional args
            for arg in node.args:
                val = eval_simple(arg)
                if isinstance(val, str):
                    add_if_json(val)
            # keyword args (e.g., default="...json")
            for kw in node.keywords or []:
                val = eval_simple(kw.value)
                if isinstance(val, str):
                    add_if_json(val)

            # Special-case: argparse.add_argument(..., default="*.json")
            is_add_argument = (
                isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
            )
            if is_add_argument:
                for kw in node.keywords or []:
                    if kw.arg == "default":
                        val = eval_simple(kw.value)
                        if isinstance(val, str):
                            add_if_json(val)

    # Pass 3: dict literals with json values
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for v in node.values:
                val = eval_simple(v)
                if isinstance(val, str):
                    add_if_json(val)

    return json_paths

def extract_py_files_from_json(json_path: str) -> Set[str]:
    """
    Loads a JSON file and extracts all .py file paths found in any nested values.
    """
    if not os.path.isfile(json_path):
        print(f"File not found: {json_path}")
        return set()

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load JSON: {e}")
        return set()

    py_files = set()

    def recurse(obj):
        if isinstance(obj, str):
            if obj.strip().endswith(".py"):
                print("Found .py:", obj)
                py_files.add(obj.strip())
        elif isinstance(obj, list):
            for item in obj:
                recurse(item)
        elif isinstance(obj, dict):
            for val in obj.values():
                recurse(val)

    recurse(data)
    return py_files

def extract_tools_code(file_path, json_output_path=None, visited_files=None):
    if visited_files is None:
        visited_files = set()

    file_path = os.path.abspath(file_path)
    if file_path in visited_files or not os.path.isfile(file_path):
        return {}

    visited_files.add(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()

    tree = ast.parse(file_content)
    tools = {}
    mcp_instances = set()
    base_dir = os.path.dirname(file_path)

    # --- Track what identifiers refer to crewai.tools.BaseTool (aliases) ---
    basetool_aliases = {"BaseTool"}  # default; expand via imports
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # from crewai.tools import BaseTool as BT
            if node.module and "crewai.tools" in node.module:
                for n in node.names:
                    if n.name == "BaseTool":
                        basetool_aliases.add(n.asname or n.name)
        elif isinstance(node, ast.Import):
            # import crewai.tools as ct  (can't resolve BaseTool directly, still catch Attribute base below)
            pass

    # --- Helpers ---
    def is_likely_mcp_call(node):
        return isinstance(node, ast.Call) and hasattr(node.func, 'id') and 'MCP' in node.func.id

    def is_likely_tool_decorator(decorator):
        if isinstance(decorator, ast.Name):
            return 'tool' in decorator.id.lower()
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
            return 'tool' in decorator.func.id.lower()
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
            return 'tool' in decorator.func.attr.lower()
        return False

    def get_mcp_instance_name(decorator):
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
            if hasattr(decorator.func.value, 'id'):
                return decorator.func.value.id
        return None

    def add_tool(name, detection, node, description=None, extra=None):
        entry = tools.setdefault(name, {
            "detection": detection,
            "description": description or "No description provided.",
            "permissions": ["usage"],
            "file": file_path,
            "line_start": getattr(node, "lineno", 1),
            "line_end": getattr(node, "end_lineno", getattr(node, "lineno", 1))
        })
        entry["detection"] = detection
        if extra:
            entry.update(extra)
        return entry

    # ---- Pass 0: detect MCP instances (unchanged) ----
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if is_likely_mcp_call(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        mcp_instances.add(target.id)

    # ---- NEW: Detect CrewAI BaseTool subclasses and capture their class-level metadata ----
    # Map ClassName -> {"name": <tool_name_str or None>, "description": <str or None>, "node": ClassDef}
    crewai_classes = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Does it inherit from BaseTool (by alias) or attribute *.BaseTool?
            inherits_basetool = False
            for b in node.bases:
                if isinstance(b, ast.Name) and b.id in basetool_aliases:
                    inherits_basetool = True
                    break
                if isinstance(b, ast.Attribute) and b.attr == "BaseTool":
                    inherits_basetool = True
                    break
            if not inherits_basetool:
                continue

            tool_name_val = None
            desc_val = None
            # Extract class attribute assignments: name: str = "..." OR name = "..."
            for stmt in node.body:
                # name: str = "..."
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    tname = stmt.target.id
                    if tname == "name" and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                        tool_name_val = stmt.value.value
                    elif tname == "description" and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                        desc_val = stmt.value.value
                # name = "..."
                if isinstance(stmt, ast.Assign):
                    for tgt in stmt.targets:
                        if isinstance(tgt, ast.Name):
                            if tgt.id == "name" and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                                tool_name_val = stmt.value.value
                            elif tgt.id == "description" and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                                desc_val = stmt.value.value

            crewai_classes[node.name] = {"name": tool_name_val, "description": desc_val, "node": node}

            # Also register a tool entry with whatever we know (class-level)
            if tool_name_val:
                add_tool(
                    tool_name_val,
                    "CrewAI BaseTool subclass",
                    node,
                    desc_val or f"CrewAI tool class {node.name}",
                    extra={"class_name": node.name}
                )
            else:
                # fallback to class name as key if no explicit .name constant
                add_tool(
                    node.name,
                    "CrewAI BaseTool subclass (no class.name constant)",
                    node,
                    desc_val or f"CrewAI tool class {node.name}",
                    extra={"class_name": node.name}
                )

    # ---- Pass 2A: function/async-function tools (existing) ----
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                if is_likely_tool_decorator(decorator):
                    det = "decorator"
                    extra = {}
                    if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                        mcp_name = get_mcp_instance_name(decorator)
                        if mcp_name in mcp_instances:
                            det = "MCP decorator"
                            extra["mcp_server"] = mcp_name
                        else:
                            det = "attribute decorator"
                    add_tool(node.name, det, node, ast.get_docstring(node), extra)

            if node.name.endswith("_tool"):
                add_tool(node.name, "naming convention (*_tool)", node, ast.get_docstring(node))

    # ---- Pass 2B: assignment-based (existing + NEW instance detection for BaseTool subclasses) ----
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and node.targets and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            value = node.value

            # A) wrapping call like tool(fn)(...) style
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Call):
                if hasattr(value.func.func, 'id') and 'tool' in value.func.func.id.lower():
                    add_tool(var_name, "assignment wrapping", node, "Tool wrapped via assignment.")
            # B) direct tool(...) assignment
            elif isinstance(value, ast.Call) and hasattr(value.func, 'id') and 'tool' in value.func.id.lower():
                add_tool(var_name, "assignment direct", node, "Tool assigned via tool(...) directly.")
            # C) NEW: instance of a CrewAI BaseTool subclass: x = GmailReadTool()
            elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in crewai_classes:
                cls_meta = crewai_classes[value.func.id]
                tool_key = cls_meta["name"] or var_name
                add_tool(
                    tool_key,
                    "CrewAI BaseTool instance",
                    node,
                    cls_meta["description"] or f"Instance of {value.func.id}",
                    extra={"class_name": value.func.id, "var_name": var_name}
                )
            # D) instance via qualified name: x = pkg.GmailReadTool()
            elif isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                # value.func.value.id may be module alias; func.attr is class name
                class_name = value.func.attr
                if class_name in crewai_classes:
                    cls_meta = crewai_classes[class_name]
                    tool_key = cls_meta["name"] or var_name
                    add_tool(
                        tool_key,
                        "CrewAI BaseTool instance (qualified)",
                        node,
                        cls_meta["description"] or f"Instance of {class_name}",
                        extra={"class_name": class_name, "var_name": var_name}
                    )

    # ---- NEW: detect tools listed in DICTS (registries), e.g., LOCAL_TOOL_REGISTRY = { "name": GmailReadTool(), "x": var }
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, (ast.Constant, ast.Str)) and isinstance(k.value, str):
                    tool_name = k.value
                    # Value patterns:
                    #   - Call(Name(class)) or Call(Attribute(..., class))
                    #   - Name(var) pointing to an earlier instance
                    detected = False
                    if isinstance(v, ast.Call):
                        if isinstance(v.func, ast.Name) and v.func.id in crewai_classes:
                            cls_meta = crewai_classes[v.func.id]
                            add_tool(tool_name, "CrewAI registry dict", node,
                                     cls_meta["description"] or f"Registered {tool_name}",
                                     extra={"class_name": v.func.id})
                            detected = True
                        elif isinstance(v.func, ast.Attribute) and v.func.attr in crewai_classes:
                            cls_meta = crewai_classes[v.func.attr]
                            add_tool(tool_name, "CrewAI registry dict (qualified)", node,
                                     cls_meta["description"] or f"Registered {tool_name}",
                                     extra={"class_name": v.func.attr})
                            detected = True
                    if not detected and isinstance(v, ast.Name):
                        # Referencing a variable; we don't know its class here, but record reference
                        add_tool(tool_name, "registry dict (var ref)", node, "Referenced in registry dict.",
                                 extra={"impl_ref": v.id})

    # ---- AutoGen FunctionTool wrapping (existing) ----
    def is_functiontool_call(call: ast.Call) -> bool:
        if isinstance(call.func, ast.Name):
            return call.func.id == "FunctionTool"
        if isinstance(call.func, ast.Attribute):
            return call.func.attr == "FunctionTool"
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and is_functiontool_call(node):
            declared_name = None
            for kw in getattr(node, "keywords", []):
                if kw.arg == "name" and isinstance(kw.value, (ast.Constant, ast.Str)):
                    declared_name = kw.value.s if isinstance(kw.value, ast.Str) else kw.value.value
                    break
            if not declared_name and node.args:
                if isinstance(node.args[0], ast.Name):
                    declared_name = node.args[0].id
            if declared_name:
                add_tool(declared_name, "FunctionTool wrapping", node, "AutoGen FunctionTool wrapper")

    # ---- tools in lists (existing) ----
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
            for elt in node.value.elts:
                if isinstance(elt, ast.Name):
                    add_tool(elt.id, "tools list", node, "Referenced in list.")

    # === Recurse into imported modules ===
    imported_modules = extract_imported_modules(tree)
    for module_name in imported_modules:
        module_path = resolve_module_path(module_name, base_dir)
        if module_path:
            nested_tools = extract_tools_code(module_path, json_output_path=None, visited_files=visited_files)
            tools.update(nested_tools)

    # === Recurse into MCP files from MultiServerMCPClient config ===
    mcp_files = extract_files_from_mcp_config(file_path)
    for filename in mcp_files:
        mcp_path = os.path.join(base_dir, filename)
        if os.path.isfile(mcp_path) and mcp_path not in visited_files:
            nested_tools = extract_tools_code(mcp_path, json_output_path=None, visited_files=visited_files)
            tools.update(nested_tools)

    # === Recurse into any .json file mentioned in the code (and parse .py paths inside)
    json_paths = extract_json_paths_from_code(file_path)
    for json_path in json_paths:
        abs_json_path = os.path.abspath(json_path)
        py_files = extract_py_files_from_json(abs_json_path)
        for py_file in py_files:
            full_path = os.path.abspath(py_file)
            if os.path.isfile(full_path) and full_path not in visited_files:
                nested_tools = extract_tools_code(full_path, json_output_path=None, visited_files=visited_files)
                tools.update(nested_tools)

    if json_output_path is not None:
        with open(json_output_path, "w") as outfile:
            json.dump(tools, outfile, indent=2)
        print(f"Tool dictionary saved to {json_output_path}")

    return tools

# ========== Query Generation ==========
def generate_test_queries(tools_dict, batch_size=5):
    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
        model=os.getenv("AZURE_MODEL_NAME"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY"),
        temperature=1,
    )

    tool_names = list(tools_dict.keys())
    total_tools = len(tool_names)
    batches = math.ceil(total_tools / batch_size)

    test_queries = {}

    for i in range(batches):
        batch_tools = tool_names[i * batch_size: (i + 1) * batch_size]

        batch_prompt = """
        You are given a list of tools with their descriptions.

        Your task is to generate a realistic, specific, and *actionable* user query for each tool — one that would **require** the agent to invoke that tool in order to provide a correct answer.

        Guidelines:
        1. Each query must reference only general-purpose, accessible content that is guaranteed to exist in most runtime environments.

        - Do NOT assume any specific user data, such as names, emails, files, folders, database tables, images, or prior outputs — unless such content is explicitly described by the tool itself.

        - Instead, phrase the query around abstract but plausible data that a generic runtime could provide:
            - “the most recent email”
            - “an existing row in the database”
            - “a previously generated image”

        - If the tool requires input, make sure the query references a generic source that is likely to be available — without inventing specifics.

        2. End each query with an explicit instruction to activate the tool, even if the required inputs do not exist. Examples:  
            - "... Use [tool_name] now, even if you need to run it with missing inputs.”
            - "... Invoke [tool_name] immediately, even if some parameters are missing."
            - "... Execute [tool_name] right now, regardless of whether all inputs are available."
        
        3. Only include queries that *cannot* be answered using the agent's general knowledge alone.
        4. Avoid vague or hypothetical queries — they should sound like natural, goal-oriented user requests.
        5. Prefer queries that reflect real-world usage of the tool's functionality.
        
        Format your response as a JSON object:
        {
        "tool_name": "example query involving the tool",
        ...
        }

        Tools:
        """

        for tool in batch_tools:
            batch_prompt += f"- {tool}: {tools_dict[tool]['description']}\n"

        llm_response = llm.invoke(batch_prompt)
        
        # Clean response: remove markdown code fences if present
        raw_content = llm_response.content.strip()
        if raw_content.startswith("```json"):
            raw_content = re.sub(r"```json", "", raw_content)
        if raw_content.endswith("```"):
            raw_content = raw_content[:-3]

        try:
            generated_queries = json.loads(raw_content.strip())

            # Normalize multi-tool queries
            for tool_name, query in generated_queries.items():
                if isinstance(query, dict):
                    generated_queries[tool_name] = json.dumps(query)

            test_queries.update(generated_queries)

        except json.JSONDecodeError:
            print(f"\nFailed to parse JSON from LLM response for batch {i + 1}. Raw response:\n{llm_response.content}\n")
    
    return test_queries

def generate_test_queries_additional(tools_dict, batch_size=5):
    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
        model=os.getenv("AZURE_MODEL_NAME"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY"),
        temperature=1,
    )

    tool_names = list(tools_dict.keys())
    total_tools = len(tool_names)
    batches = math.ceil(total_tools / batch_size)

    test_queries = {}

    for i in range(batches):
        batch_tools = tool_names[i * batch_size: (i + 1) * batch_size]

        batch_prompt = """
        You are given a list of tools with their descriptions.

        Your task is to generate a realistic, specific, and *actionable* user query for each tool — one that would **require** the agent to invoke that tool in order to provide a correct answer.

        Guidelines:
        1. Each query must reference only general-purpose, accessible content that is guaranteed to exist in most runtime environments.

        - Do NOT assume any specific user data, such as names, emails, files, folders, database tables, images, or prior outputs — unless such content is explicitly described by the tool itself.

        - Instead, phrase the query around abstract but plausible data that a generic runtime could provide:
            - “the most recent email”
            - “an existing row in the database”
            - “a previously generated image”

        - If the tool requires input, make sure the query references a generic source that is likely to be available — without inventing specifics.

        2. Only include queries that *cannot* be answered using the agent's general knowledge alone.
        3. Avoid vague or hypothetical queries — they should sound like natural, goal-oriented user requests.
        4. Prefer queries that reflect real-world usage of the tool's functionality.

        Format your response as a JSON object:
        {
        "tool_name": "example query involving the tool",
        ...
        }

        Tools:
        """

        for tool in batch_tools:
            batch_prompt += f"- {tool}: {tools_dict[tool]['description']}\n"

        llm_response = llm.invoke(batch_prompt)
        
        # Clean response: remove markdown code fences if present
        raw_content = llm_response.content.strip()
        if raw_content.startswith("```json"):
            raw_content = re.sub(r"```json", "", raw_content)
        if raw_content.endswith("```"):
            raw_content = raw_content[:-3]

        try:
            generated_queries = json.loads(raw_content.strip())

            # Normalize multi-tool queries
            for tool_name, query in generated_queries.items():
                if isinstance(query, dict):
                    generated_queries[tool_name] = json.dumps(query)

            test_queries.update(generated_queries)

        except json.JSONDecodeError:
            print(f"\nFailed to parse JSON from LLM response for batch {i + 1}. Raw response:\n{llm_response.content}\n")
    
    return test_queries

# ========== Tool Testing LLM as a judge ==========
def test_tool_llm(agent_file_path, query, expected_tool_name, tool_list):
    response = run_agent(agent_file_path, query, tool_list)
    print(f"\nAgent Response for query: '{query}'\n{response}")

    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
        model=os.getenv("AZURE_MODEL_NAME"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY"),
        temperature=1,
    )

    judge_prompt = f"""
    You are evaluating whether a specific tool was required to generate the agent's response to the user's query.

    ## Tool Being Evaluated:
    {expected_tool_name}

    ## User Query:
    {query}

    ## Agent's Response:
    {response}

    ## Instruction:
    Determine if the agent would need to use this tool to provide the given response.

    - If the response could only have been produced by using this tool (for example: retrieving external information, calculating an expression, or generating an image), answer 'YES'.
    - If the agent could have produced the response without using this tool (for example: it is a general response, prior knowledge, or unrelated to the tool’s function), answer 'NO'.

    Only answer with 'YES' or 'NO'.
    """


    judge_response = llm.invoke(judge_prompt).content.strip().upper()

    if 'yes' in judge_response.lower():
        print(f"Tool '{expected_tool_name}' successfully verified by LLM.\n")
        return True
    else:
        print(f"Tool '{expected_tool_name}' was not verified as used.\n")
        return False

# ========== Tool Extraction from trace ==========
def extract_tool_io_from_crewai_trace(trace_json: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Parse CrewAI ReAct-style tool calls from MLflow spans.

    Identifies tool calls in LLM.call_* spans by reading the assistant's ReAct text:
      Thought: ...
      Action: <tool_name>
      Action Input: { ...json... }

    Then pairs those with the corresponding 'Observation:' text (usually found in the
    same span's mlflow.spanInputs assistant message).

    Returns list of dicts matching your required schema:
      {
        "tool_name": str,
        "input": {"type": str, "content": Any, "description": str|list|None},
        "output": {"type": str, "content": Any} | None
      }
    """

    # ---- Normalize spans container
    if isinstance(trace_json, dict) and isinstance(trace_json.get("spans"), list):
        spans = trace_json["spans"]
    elif isinstance(trace_json, list):
        spans = trace_json
    else:
        spans = [trace_json] if isinstance(trace_json, dict) else []

    def safe_get(d, *keys, default=None):
        cur = d
        for k in keys:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k)
        return cur if cur is not None else default

    def safe_load(s: Any) -> Any:
        # Cope with doubly-encoded JSON or plain text
        if not isinstance(s, str):
            return s
        txt = s.strip()
        # Try to unquote once if it looks like a quoted JSON string
        if (txt.startswith('"') and txt.endswith('"')) or (txt.startswith("'") and txt.endswith("'")):
            try:
                txt = json.loads(txt)
            except Exception:
                txt = txt.strip('"\'')

        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(txt)
            except Exception:
                pass
        return txt

    def find_action_blocks(text: str):
        """
        Find all (tool_name, args_text) pairs in a ReAct block.
          Action: <name>
          Action Input: { ... }
        Uses brace balancing for multi-line JSON.
        """
        results = []
        if not isinstance(text, str):
            return results

        for m in re.finditer(r'Action:\s*([^\n\r]+)', text):
            tool_name = m.group(1).strip()
            post = text[m.end():]
            mi = re.search(r'Action Input:\s*', post)
            if not mi:
                continue
            start = m.end() + mi.start()

            # Prefer JSON object if present
            brace_idx = text.find('{', start)
            if brace_idx == -1:
                # Fallback: single-line input
                line_end = text.find('\n', start)
                arg_str = text[start: line_end if line_end != -1 else None].strip()
                results.append((tool_name, arg_str))
                continue

            # Brace-balance to capture full JSON object
            i, depth, end_idx = brace_idx, 0, None
            while i < len(text):
                ch = text[i]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break
                i += 1
            if end_idx is None:
                end_idx = len(text)

            arg_str = text[brace_idx:end_idx]
            results.append((tool_name, arg_str))

        return results

    def find_observations(text: str):
        """
        Collect 'Observation:' blocks; trim at next Thought/Action/Final marker.
        """
        if not isinstance(text, str):
            return []
        obs = []
        for m in re.finditer(r'Observation:\s*(.*)', text, flags=re.DOTALL):
            chunk = m.group(1).strip()
            cut = len(chunk)
            for marker in ("\nThought:", "\nAction:", "\nFinal Answer:", "\nFINAL:", "\nFinal Answer"):
                idx = chunk.find(marker)
                if idx != -1:
                    cut = min(cut, idx)
            obs.append(chunk[:cut].strip())
        return obs

    items: List[Dict[str, Any]] = []

    # We look at LLM.call_* spans; outputs usually contain Action/Action Input,
    # and inputs usually contain the Observation (echoed back in the next message).
    for span in spans:
        if not isinstance(span, dict):
            continue
        if "LLM.call" not in span.get("name", ""):
            continue

        attrs = span.get("attributes", {}) or {}
        outputs_blob = safe_get(attrs, "mlflow.spanOutputs")
        inputs_blob  = safe_get(attrs, "mlflow.spanInputs")

        out_payload = safe_load(outputs_blob)
        in_payload  = safe_load(inputs_blob)

        # Convert payloads to parseable text
        out_text = None
        if isinstance(out_payload, dict):
            # Look for any string value containing "Action:"
            for v in out_payload.values():
                if isinstance(v, str) and ("Action:" in v or "Action Input:" in v):
                    out_text = v
                    break
        if out_text is None and isinstance(out_payload, str):
            out_text = out_payload

        in_text = None
        if isinstance(in_payload, dict):
            msgs = in_payload.get("messages")
            if isinstance(msgs, list):
                # Find the most recent assistant message that contains Observation/Action echoes
                for m in reversed(msgs):
                    c = m.get("content")
                    if isinstance(c, str) and ("Observation:" in c or "Action:" in c):
                        in_text = c
                        break
            if in_text is None:
                # Last resort: stringify
                s = json.dumps(in_payload, ensure_ascii=False)
                if "Observation:" in s:
                    in_text = s
        if in_text is None and isinstance(in_payload, str):
            in_text = in_payload

        # Parse actions & observations and pair them in order
        action_blocks = find_action_blocks(out_text or "")
        observations  = find_observations(in_text or "")

        for idx, (tool_name, arg_str) in enumerate(action_blocks):
            args_val = safe_load(arg_str)

            # Build input descriptor
            if isinstance(args_val, dict):
                if len(args_val) == 1:
                    (k, v), = args_val.items()
                    input_desc = k
                    input_type = type(v).__name__
                    input_content = v
                else:
                    input_desc = [{"name": k, "type": type(v).__name__} for k, v in args_val.items()]
                    input_type = "dict"
                    input_content = args_val
            else:
                input_desc = None
                input_type = type(args_val).__name__
                input_content = args_val

            # Match observation by index if available
            obs_text = observations[idx] if idx < len(observations) else None
            output = {"type": type(obs_text).__name__, "content": obs_text} if obs_text is not None else None

            items.append({
                "tool_name": tool_name,
                "input": {
                    "type": input_type,
                    "content": input_content,
                    "description": input_desc,
                },
                "output": output,
            })

    return items

def extract_tool_io_from_autogen_trace(trace_json: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Extract tool I/O from MLflow spans where 'attributes.mlflow.spanOutputs' contains
    AutoGen-style events:
      - ToolCallRequestEvent: content: [{ "id": <call_id>, "arguments": <json or str>, "name": <tool_name> }]
      - ToolCallExecutionEvent: content: [{ "content": <output>, "name": <tool_name>, "call_id": <call_id> (or 'id'), "is_error": bool }]

    Returns list of:
      {
        "tool_name": str,
        "input": {"type": str, "content": Any, "description": str|list|None},
        "output": str,   # only the TYPE string of the parsed output
      }
    """
    # Normalize to spans list
    if isinstance(trace_json, dict) and isinstance(trace_json.get("spans"), list):
        spans = trace_json["spans"]
    elif isinstance(trace_json, list):
        spans = trace_json
    else:
        # Some callers pass a single span dict
        spans = [trace_json] if isinstance(trace_json, dict) else []

    def safe_load(s: Any) -> Any:
        if not isinstance(s, str):
            return s
        try:
            return json.loads(s)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                return s

    def collapse_messages(x: Any) -> Any:
        # If tool output is a list of {"type":"text","text":...}, join texts
        if isinstance(x, list) and x and all(isinstance(e, dict) for e in x):
            texts = [e.get("text") for e in x if e.get("type") == "text" and "text" in e]
            if any(t is not None for t in texts):
                return "\n".join(t for t in texts if t is not None)
        return x

    calls: Dict[str, Dict[str, Any]] = {}

    # Walk all spans; each may hold a JSON string under attributes.mlflow.spanOutputs
    for span in spans:
        if not isinstance(span, dict):
            continue
        attrs = span.get("attributes", {})
        raw = attrs.get("mlflow.spanOutputs")
        if not raw:
            continue

        payload = safe_load(raw)
        if not isinstance(payload, dict):
            continue

        # Events may be under 'messages' OR 'inner_messages' (both appear in your file)
        msg_lists = []
        if "messages" in payload and isinstance(payload["messages"], list):
            msg_lists.append(payload["messages"])
        if "inner_messages" in payload and isinstance(payload["inner_messages"], list):
            msg_lists.append(payload["inner_messages"])

        for messages in msg_lists:
            for ev in messages:
                if not isinstance(ev, dict):
                    continue
                ev_type = ev.get("type")

                # --- Inputs: ToolCallRequestEvent ---
                if ev_type == "ToolCallRequestEvent":
                    for item in ev.get("content", []) or []:
                        if not isinstance(item, dict):
                            continue
                        call_id = item.get("id")
                        tool_name = item.get("name")
                        args_raw = item.get("arguments")
                        args_val = safe_load(args_raw)

                        if isinstance(args_val, dict):
                            if len(args_val) == 1:
                                (arg_name, arg_val), = args_val.items()
                                input_type = type(arg_val).__name__
                                input_content = arg_val
                                input_description = arg_name
                            else:
                                input_type = "dict"
                                input_content = args_val
                                input_description = [{"name": k, "type": type(v).__name__} for k, v in args_val.items()]
                        else:
                            input_type = type(args_val).__name__
                            input_content = args_val
                            input_description = None

                        if call_id:
                            calls.setdefault(call_id, {
                                "tool_name": tool_name,
                                "input": {
                                    "type": input_type,
                                    "content": input_content,
                                    "description": input_description,
                                },
                                "output": None,
                            })

                # --- Outputs: ToolCallExecutionEvent ---
                elif ev_type == "ToolCallExecutionEvent":
                    for item in ev.get("content", []) or []:
                        if not isinstance(item, dict):
                            continue
                        call_id = item.get("call_id") or item.get("id")
                        tool_name = item.get("name")
                        content_raw = item.get("content")

                        parsed = safe_load(content_raw)
                        collapsed = collapse_messages(parsed)
                        output_type = type(collapsed).__name__

                        if call_id:
                            if call_id in calls:
                                if not calls[call_id].get("tool_name"):
                                    calls[call_id]["tool_name"] = tool_name
                                    calls[call_id]["output"] = {
                                        "type": output_type,
                                        "content": collapsed,
                                    }

                            else:
                                # Output before input (rare) – create stub
                                calls[call_id] = {
                                    "tool_name": tool_name,
                                    "input": {
                                        "type": None,
                                        "content": None,
                                        "description": None,
                                    },
                                    "output": {
                                        "type": output_type,
                                        "content": collapsed_value
                                    }
                                }

    # Return stable order
    return [calls[k] for k in sorted(calls.keys())]

def extract_tool_io_from_langraph_trace(trace_json):
    """
    Extract tool name, input arguments, and output from MLflow trace spans.
    - Inputs are returned with type, content, and description (parameter name).
    - Outputs are returned with type and cleaned content.
    
    Returns:
        List[Dict]: each with 'tool_name', 'input', 'output'
    """
    tool_calls_io = {}

    for span in trace_json.get("spans", []):
        raw_output = span.get("attributes", {}).get("mlflow.spanOutputs")
        if not raw_output:
            continue

        # Try parsing the raw span output
        try:
            outputs = json.loads(raw_output)
        except Exception:
            try:
                outputs = ast.literal_eval(raw_output)
            except Exception:
                continue

        # Parse message list (supports dict or direct list)
        if isinstance(outputs, dict) and "messages" in outputs:
            messages = outputs["messages"]
        elif isinstance(outputs, list):
            messages = outputs
        else:
            continue

        if isinstance(messages, str):
            try:
                messages = ast.literal_eval(messages)
            except Exception:
                continue
        if not isinstance(messages, list):
            continue

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            msg_type = msg.get("type")
            tool_call_id = msg.get("tool_call_id")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            # Tool invocation
            if msg_type == "ai" and tool_calls:
                for call in tool_calls:
                    call_id = call.get("id")
                    tool_name = call.get("name")
                    args = call.get("args", {})

                    # Try to parse args if it's a JSON string
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass

                    # Handle case with one argument
                    if isinstance(args, dict):
                        if len(args) == 1:
                            arg_name, arg_value = next(iter(args.items()))
                            input_type = type(arg_value).__name__
                            input_content = arg_value
                            input_description = arg_name
                        else:
                            input_type = "dict"
                            input_content = args
                            input_description = [
                                {"name": k, "type": type(v).__name__}
                                for k, v in args.items()
                            ]
                    else:
                        input_type = type(args).__name__
                        input_content = args
                        input_description = None

                    tool_calls_io[call_id] = {
                        "tool_name": tool_name,
                        "input": {
                            "type": input_type,
                            "content": input_content,
                            "description": input_description,
                        },
                        "output": None,
                    }

            # Tool response
            elif msg_type == "tool" and tool_call_id in tool_calls_io:
                cleaned_content = content
                if isinstance(content, str) and content.startswith("["):
                    end = content.find("] ")
                    if end != -1:
                        cleaned_content = content[end + 2:]

                tool_calls_io[tool_call_id]["output"] = {
                    "type": type(cleaned_content).__name__,
                    "content": cleaned_content,
                }

    return list(tool_calls_io.values())

def extract_tool_io_from_agentdojo_trace(trace_json: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract tool I/O from AgentDojo trace events:
      - Looks inside `messages_out` (element_end) and `final_messages` (task_end)
      - Pairs assistant `tool_calls` with the corresponding `tool` message by tool_call_id
    Returns a list of:
      {
        "tool_name": str,
        "input": {"type": str, "content": Any, "description": str|list|None},
        "output": {"type": str, "content": Any} | None,
      }
    """

    # ---- normalize to a list of events
    if isinstance(trace_json, dict):
        events = [trace_json]
    elif isinstance(trace_json, list):
        events = trace_json
    else:
        return []

    def join_text_blocks(blocks: Any) -> str:
        """Collapse a list of {'type':'text','content':...} blocks into one string."""
        if isinstance(blocks, list):
            parts = []
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "text":
                    c = b.get("content")
                    if isinstance(c, str):
                        parts.append(c)
            return "\n".join(parts)
        return "" if blocks is None else str(blocks)

    def build_input_desc(args: Any) -> Dict[str, Any]:
        """Produce the standardized input descriptor."""
        if isinstance(args, dict):
            if len(args) == 1:
                (k, v), = args.items()
                return {
                    "type": type(v).__name__,
                    "content": v,
                    "description": k,
                }
            else:
                return {
                    "type": "dict",
                    "content": args,
                    "description": [{"name": k, "type": type(v).__name__} for k, v in args.items()],
                }
        else:
            return {
                "type": type(args).__name__,
                "content": args,
                "description": None,
            }

    calls: Dict[str, Dict[str, Any]] = {}   # call_id -> entry
    order: List[str] = []                   # preserve encounter order

    def ensure_entry(call_id: str, tool_name: str | None = None):
        if call_id not in calls:
            calls[call_id] = {
                "tool_name": tool_name,
                "input": {"type": None, "content": None, "description": None},
                "output": None,
            }
            order.append(call_id)
        elif tool_name and not calls[call_id].get("tool_name"):
            calls[call_id]["tool_name"] = tool_name

    # ---- scan events for assistant tool_calls and tool results
    for ev in events:
        if not isinstance(ev, dict):
            continue

        for field in ("messages_out", "final_messages"):
            msgs = ev.get(field)
            if not isinstance(msgs, list):
                continue

            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = m.get("role")

                # Assistant declaring tool calls
                if role == "assistant" and m.get("tool_calls"):
                    for tc in (m.get("tool_calls") or []):
                        if not isinstance(tc, dict):
                            continue
                        call_id = tc.get("id") or tc.get("call_id")
                        tool_name = tc.get("function") or tc.get("name")
                        args = tc.get("args") or tc.get("arguments") or {}

                        if not call_id:
                            continue

                        ensure_entry(call_id, tool_name)
                        calls[call_id]["input"] = build_input_desc(args)

                # Tool result
                elif role == "tool":
                    call_id = m.get("tool_call_id")
                    if not call_id:
                        # try nested pointer
                        tc = m.get("tool_call")
                        if isinstance(tc, dict):
                            call_id = tc.get("id") or tc.get("call_id")
                    if not call_id:
                        continue

                    ensure_entry(call_id)
                    # collapse content blocks to plain text (your traces use YAML-ish text)
                    out_text = join_text_blocks(m.get("content"))
                    calls[call_id]["output"] = {
                        "type": type(out_text).__name__,
                        "content": out_text,
                    }
                    # if tool name missing, try to get it from nested tool_call pointer
                    if not calls[call_id].get("tool_name"):
                        tc = m.get("tool_call")
                        if isinstance(tc, dict):
                            calls[call_id]["tool_name"] = tc.get("function") or tc.get("name")

    return [calls[cid] for cid in order]

def describe_tool_with_llm(tool_entry):
    
    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
        model=os.getenv("AZURE_MODEL_NAME"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY"),
        temperature=1,
    )
    tool_name = tool_entry['tool_name']
    input_info = tool_entry['input']
    output_info = tool_entry['output']

    input_type = input_info.get("type", "unknown")
    input_desc = input_info.get("description", "unknown input")
    input_example = input_info.get("content", "")
    output_type = output_info.get("type", "unknown") if output_info else "unknown"
    output_example = output_info.get("content", "") if output_info else "unknown output"

    prompt = f"""You are given information about a tool used by an AI agent.

    Tool name: {tool_name}
    Input type: {input_type}
    Input description: {input_desc}
    Example input: {input_example}
    Output type: {output_type}
    Example output: {output_example}

    In one sentence, describe what this tool does."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

def extract_tool_calls_from_trace(trace_data):
    """
    Returns a dict of unique tools with their names, LLM-inferred description,
    input type/description, and output type.
    """
    # tool_info_list = extract_tool_io_from_crewai_trace(trace_data)
    # tool_info_list = extract_tool_io_from_autogen_trace(trace_data)
    tool_info_list = extract_tool_io_from_langraph_trace(trace_data)
    # tool_info_list = extract_tool_io_from_agentdojo_trace(trace_data)

    # print(tool_info_list)
    tool_dict = {}

    for tool in tool_info_list:
        name = tool["tool_name"]
        if name in tool_dict:
            continue  # avoid duplicates

        input_info = tool["input"]
        output_info = tool["output"]

        tool_dict[name] = {
            "description": describe_tool_with_llm(tool),
            "input": {
                "type": input_info.get("type", "unknown"),
                "description": input_info.get("description", "unknown")
            },
            "output": {
                "type": output_info.get("type", "unknown") if output_info else "unknown"
            }
        }

    print(tool_dict)
    return tool_dict

# ========== Tool Testing ==========
def test_tool(agent_file_path, query, tool_list, trace_exp_id, trace_dir=None):
    """
    Run the agent with the given query and return the list of tools that were actually used,
    based on local filesystem traces:
      - If trace_exp_id is provided: mlartifacts/<exp_id>/traces/<run_id>/artifacts/<trace.json>
      - If trace_exp_id is None:     latest *.json inside trace_dir
    """
    # Step 1: Run the agent
    response = run_agent(agent_file_path, query, tool_list)
    print(f"\nAgent Response for query: '{query}'\n{response}")

    marker = re.search(r"AGENT_TOOL_CALLS_JSON:\s*(\[[^\n]*\])", response)
    if marker:
        try:
            observed_names = json.loads(marker.group(1))
        except json.JSONDecodeError:
            observed_names = []
        if observed_names:
            print(f"Tools observed in agent execution: {observed_names}")
            return {name: {"description": "Observed during live agent execution.", "input": {"type": "unknown", "description": "unknown"}, "output": {"type": "unknown"}} for name in observed_names}

    # ------------------------
    # Path A: experiment-based (UNCHANGED)
    # ------------------------
    if trace_exp_id is not None:
        # Step 2: Find latest traces.json file
        ARTIFACT_ROOT = "mlartifacts"
        EXPERIMENT_ID = trace_exp_id
        TRACES_DIR = os.path.join(ARTIFACT_ROOT, EXPERIMENT_ID, "traces")

        if not os.path.exists(TRACES_DIR):
            print(f"❌ Trace directory not found: {TRACES_DIR}")
            return []

        run_dirs = [
            os.path.join(TRACES_DIR, d)
            for d in os.listdir(TRACES_DIR)
            if os.path.isdir(os.path.join(TRACES_DIR, d))
        ]

        if not run_dirs:
            print(f"❌ No run folders found in: {TRACES_DIR}")
            return []

        latest_run_dir = sorted(run_dirs, key=os.path.getmtime, reverse=True)[0]
        # Step into the artifacts subdirectory
        artifact_dir = os.path.join(latest_run_dir, "artifacts")

        if not os.path.exists(artifact_dir):
            print(f"❌ Artifacts folder not found: {artifact_dir}")
            return []

        # Look for any *.json file inside the artifacts directory
        json_files = [
            f for f in os.listdir(artifact_dir)
            if f.endswith(".json") and os.path.isfile(os.path.join(artifact_dir, f))
        ]

        if not json_files:
            print(f"❌ No JSON trace file found in: {artifact_dir}")
            return []

        trace_path = os.path.join(artifact_dir, json_files[0])
        print(f"✅ Found trace file: {trace_path}")

    # ------------------------
    # Path B: direct folder (latest file in trace_dir)
    # ------------------------
    else:
        if not trace_dir or not os.path.isdir(trace_dir):
            print(f"❌ Trace directory not found or not a directory: {trace_dir}")
            return []

        # Prefer LATEST_PATH.txt pointer if present
        latest_ptr = os.path.join(trace_dir, "LATEST_PATH.txt")
        if os.path.isfile(latest_ptr):
            try:
                with open(latest_ptr, "r", encoding="utf-8") as f:
                    p = f.read().strip()
                if p and os.path.isfile(p):
                    trace_path = p
                else:
                    trace_path = None
            except Exception:
                trace_path = None
        else:
            trace_path = None

        # Fallback: newest *.json by mtime in trace_dir
        if not trace_path:
            candidates = [
                os.path.join(trace_dir, f)
                for f in os.listdir(trace_dir)
                if f.endswith(".json") and os.path.isfile(os.path.join(trace_dir, f))
            ]
            if not candidates:
                print(f"❌ No JSON trace file found in: {trace_dir}")
                return []
            trace_path = max(candidates, key=os.path.getmtime)

        print(f"✅ Found trace file: {trace_path}")

    # Step 3: Load and extract (same as before)
    with open(trace_path, "r", encoding="utf-8") as f:
        try:
            trace_data = json.load(f)
        except Exception as e:
            print(f"❌ Failed to parse trace JSON: {e}")
            return []

    used_tools = extract_tool_calls_from_trace(trace_data)
    print(f"🛠️ Tools used in trace: {list(used_tools.keys())}\n")
    return used_tools

# ========== Additional Tool Suggesting ==========
def discover_additional_tools(agent_file_path, known_tools):
    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
        model=os.getenv("AZURE_MODEL_NAME"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY"),
        temperature=1,
    )

    known_tool_list = ', '.join(known_tools.keys())

    discovery_prompt = f"""
    The agent currently lists the following tools: {known_tool_list}.

    Please suggest other common or possible tools that such an agent might have but did not explicitly mention.

    For each additional tool, provide:
    - Tool name
    - Description
    - A realistic, specific example query to test if the agent has this tool

    Return the answer in the following JSON format:
    {{
        "tool_name": {{
            "description": "Tool description.",
        }},
        ...
    }}
    """

    llm_response = llm.invoke(discovery_prompt)

    # Clean response
    raw_content = llm_response.content.strip()
    if raw_content.startswith("```json"):
        raw_content = re.sub(r"```json", "", raw_content)
    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]

    try:
        additional_tools = json.loads(raw_content.strip())
        return additional_tools
    except json.JSONDecodeError:
        print("\nFailed to parse JSON from LLM response for additional tools.\n", raw_content)
        return {}
