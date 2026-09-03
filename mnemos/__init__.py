# -*- coding: utf-8 -*-
"""ShineMnemos — память для агентов с узлами правды (П1-П6).

Первый кирпич по роадмапу RECON_PRODUCTS: локальный MCP memory server +
шина агентов + truth-gate. Только stdlib, без внешних зависимостей.
"""

from .budget import (
    RELS,
    ROUTERS,
    BudgetSearch,
    Graph,
    build_system_prompt,
    classify_query,
)
from .bus import Bus, locked_file
from .context_engine import (
    CanonicalPrefix,
    ContextDefragmenter,
    HierarchicalCompactor,
    estimate_tokens,
)
from .gates import (
    GateResult,
    run_read_gates,
    run_write_gates,
)
from .grounding import (
    SYSTEM_PROMPT_TEMPLATE,
    GroundLog,
    SessionTracker,
    graph_first,
    ground_answer,
    split_claims,
    verify_claim,
)
from .model import KINDS, MemoryNode, make_node, new_id, now_iso
from .plugins import (
    DEFAULT_ENABLED,
    ENV_NAME,
    PluginManager,
    known_plugin_names,
    resolve_enabled_plugins,
)
from .server import MCPHttpServer, MnemosCore, run, serve_in_thread
from .store import (
    BLANK_KEYWORD,
    BLANK_TEMPLATE_NAME,
    DEFAULT_STORE_NAME,
    STORE_ENV,
    STORE_PATH_ENV,
    Store,
    assert_blank_target_empty,
    blank_target,
    blank_template_path,
    blank_template_text,
    create_blank_store,
    resolve_store_path,
)
from .truth_gate import (
    PASS_THRESHOLD,
    TruthResult,
    check_and_update,
    check_claim,
)

__version__ = "0.4.0"

__all__ = [
    "BudgetSearch",
    "Graph",
    "RELS",
    "ROUTERS",
    "build_system_prompt",
    "classify_query",
    "Bus",
    "locked_file",
    "SYSTEM_PROMPT_TEMPLATE",
    "GroundLog",
    "SessionTracker",
    "graph_first",
    "ground_answer",
    "split_claims",
    "verify_claim",
    "CanonicalPrefix",
    "ContextDefragmenter",
    "DEFAULT_ENABLED",
    "ENV_NAME",
    "GateResult",
    "HierarchicalCompactor",
    "KINDS",
    "MemoryNode",
    "make_node",
    "new_id",
    "now_iso",
    "MCPHttpServer",
    "MnemosCore",
    "PluginManager",
    "known_plugin_names",
    "resolve_enabled_plugins",
    "run",
    "serve_in_thread",
    "Store",
    "BLANK_KEYWORD",
    "BLANK_TEMPLATE_NAME",
    "DEFAULT_STORE_NAME",
    "STORE_ENV",
    "STORE_PATH_ENV",
    "assert_blank_target_empty",
    "blank_target",
    "blank_template_path",
    "blank_template_text",
    "create_blank_store",
    "resolve_store_path",
    "PASS_THRESHOLD",
    "TruthResult",
    "check_and_update",
    "check_claim",
    "estimate_tokens",
    "run_read_gates",
    "run_write_gates",
    "__version__",
]
