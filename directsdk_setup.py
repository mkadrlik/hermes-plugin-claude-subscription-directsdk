"""Setup-time probes of the user's Claude CLI: login state and its own model picker.

Both are offline with respect to Anthropic: ``auth status`` reads the local credential store, and
the ``initialize`` handshake enumerates the picker without a Messages request (the admission relay
proves it by counting upstream calls). Anything unexpected returns ``None``/``False`` so callers
fall back to the pinned catalog rather than failing setup.
"""
import json
import os
import shutil
import subprocess
import tempfile

try:
    from .admission import Admission
    from .model_catalog import (DEFAULT_CONTEXT_WINDOW, LONG_CONTEXT_WINDOW, MODEL_METADATA,
                                learn, native_model)
except ImportError:
    from admission import Admission
    from model_catalog import (DEFAULT_CONTEXT_WINDOW, LONG_CONTEXT_WINDOW, MODEL_METADATA,
                               learn, native_model)

INSTALL_HINT = ("Claude Code is not installed (no `claude` on PATH). Install it with "
                "`npm install -g @anthropic-ai/claude-code` or set CLAUDE_SUBSCRIPTION_DIRECTSDK_COMMAND to the binary.")
LOGIN_HINT = "Claude Code is installed but not logged in. Run `claude auth login`, then select this provider again."


def _resolve(command, env):
    command = list(command) if command else [env.get("CLAUDE_SUBSCRIPTION_DIRECTSDK_COMMAND") or "claude"]
    head = command[0]
    exe = head if os.path.isabs(head) and os.access(head, os.X_OK) else shutil.which(head, path=env.get("PATH") or os.defpath)
    return ([exe] + command[1:]) if exe else None


def _child_env(env):
    child = dict(env)
    config = child.pop("CLAUDE_SUBSCRIPTION_DIRECTSDK_CONFIG_DIR", None)
    if config:
        child["CLAUDE_CONFIG_DIR"] = config
    child.update(CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", DISABLE_TELEMETRY="1", DISABLE_ERROR_REPORTING="1")
    return child


def _plan_label(raw):
    """``"pro"`` (auth status) and ``"Claude Pro"`` (handshake) name the same plan."""
    raw = str(raw or "").strip()
    if not raw:
        return ""
    return raw if raw.lower().startswith("claude") else "Claude " + raw.replace("_", " ").title()


def setup_status(command=None, env=None, timeout=20):
    """``{available, logged_in, plan, detail, login_command}`` from the CLI's own ``auth status``."""
    env = dict(env if env is not None else os.environ)
    resolved = _resolve(command, env)
    if resolved is None:
        return {"available": False, "logged_in": False, "plan": "", "detail": INSTALL_HINT, "login_command": None}
    login_command = resolved + ["auth", "login"]
    try:
        run = subprocess.run(resolved + ["auth", "status"], env=_child_env(env), stdin=subprocess.DEVNULL,
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        auth = json.loads(run.stdout) if run.stdout.strip().startswith("{") else {}
    except (OSError, ValueError, subprocess.SubprocessError):
        auth = {}
    logged_in = auth.get("loggedIn") is True
    plan = str(auth.get("subscriptionType") or "")
    detail = "" if logged_in else LOGIN_HINT
    return {"available": True, "logged_in": logged_in, "plan": _plan_label(plan), "detail": detail,
            "login_command": login_command}


def discover_models(command=None, env=None, timeout=40):
    """The account's live picker as ``[{id, label, note, upstream_requests}]`` in Hermes route ids,
    or ``None`` when the CLI is missing, logged out, or the handshake fails."""
    env = dict(env if env is not None else os.environ)
    resolved = _resolve(command, env)
    # Logged out, the handshake still answers with a generic default list; only a signed-in
    # account's picker reflects its entitlements.
    if resolved is None or not setup_status(command=resolved, env=env, timeout=timeout)["logged_in"]:
        return None
    child = _child_env(env)
    gate = Admission("https://api.anthropic.com", timeout)
    try:
        child["ANTHROPIC_BASE_URL"] = gate.url
        argv = resolved + ["-p", "--model", "sonnet", "--input-format", "stream-json", "--output-format", "stream-json",
                           "--verbose", "--tools", "", "--setting-sources", "", "--strict-mcp-config",
                           "--mcp-config", '{"mcpServers":{}}', "--disable-slash-commands", "--no-session-persistence"]
        handshake = json.dumps({"type": "control_request", "request_id": "hermes-picker",
                                "request": {"subtype": "initialize"}}) + "\n"
        with tempfile.TemporaryDirectory(prefix="claude-directsdk-picker-") as cwd:
            run = subprocess.run(argv, input=handshake, env=child, cwd=cwd, capture_output=True,
                                 text=True, encoding="utf-8", errors="replace", timeout=timeout)
        rows = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
        response = next(r["response"] for r in rows if r.get("type") == "control_response")
        native = response.get("response", {}).get("models") or []
        upstream = int(bool(gate.used))
    except (OSError, ValueError, StopIteration, KeyError, subprocess.SubprocessError):
        return None
    finally:
        gate.close()
    if upstream or not native:
        return None
    # Pro / Team-standard seats bill Fable to usage credits from the first request; the CLI only
    # says so at request time, so apply the documented plan rule here.
    plan = str((response.get("response", {}).get("account") or {}).get("subscriptionType") or "").lower()
    credit_billed_on_plan = {"claude-fable-5-1"} if plan and "max" not in plan else set()

    # Read the picker's own metadata rather than intersecting it with the pins: a model released
    # after this plugin shipped is exactly the one absent from the pins, and dropping it here is
    # why a new model used to need a catalog commit before it could be selected. The window is
    # not guessed — the CLI states it by resolving a route to `<id>[1m]` when that model is
    # offered at 1M, and reports thinking support per row.
    observed, aliases = {}, {}
    for row in native:
        resolved = str(row.get("resolvedModel") or row.get("value") or "")
        if not resolved:
            continue
        base = resolved.removesuffix("[1m]")
        facts = observed.setdefault(base, {"context_window": None,
                                           "supports_adaptive_thinking": None})
        # `[1m]` anywhere in the row is positive evidence of the long window; its ABSENCE is no
        # evidence at all. The CLI has no way to state "this route is only 200K", so reading
        # silence as the short window is a guess wearing a derivation's clothes. From 2.1.284
        # the current generation is announced as bare family aliases (`opus`, `sonnet`) with the
        # suffix left on secondary full-id rows only, so that guess downgraded every newly
        # released 1M model — Opus 5.5 and Sonnet 5.5 both, the day they shipped.
        # Read both fields: the suffix rides `value` while `resolvedModel` stays bare (Fable).
        if resolved.endswith("[1m]") or str(row.get("value") or "").endswith("[1m]"):
            facts["context_window"] = LONG_CONTEXT_WINDOW
        # Same asymmetry: a row that omits the key says nothing, and must not be read as False.
        if row.get("supportsAdaptiveThinking"):
            facts["supports_adaptive_thinking"] = True
        elif "supportsAdaptiveThinking" in row and facts["supports_adaptive_thinking"] is None:
            facts["supports_adaptive_thinking"] = False
        # `value` carries the family shortcut the account can type (`opus`, `sonnet`); the
        # "Default (recommended)" row names no model and a self-referential value is an id.
        shortcut = str(row.get("value") or "").removesuffix("[1m]")
        if shortcut and shortcut != "default" and shortcut != base:
            aliases[shortcut] = base
    if not observed:
        return None
    # Before native_model(), which reads the pins this teaches.
    learn(observed, aliases)

    routes = {}
    for row in native:
        base = str(row.get("resolvedModel") or row.get("value") or "").removesuffix("[1m]")
        if base not in observed:
            continue
        route = native_model(base)
        # Model names, not the native "Default (recommended)" alias row.
        label = str(row.get("description") or "").split("·")[0].strip() or route
        # An unknown window is surfaced, never hidden: the route still runs natively inside the
        # 200K gateway default, and the note is what lets a human (or claude-model-drift) spot a
        # model whose real window this build cannot state. Loudly wrong beats silently wrong.
        note = "" if route in MODEL_METADATA else "unpinned"
        entry = routes.setdefault(route, {"id": route, "label": label, "note": note, "upstream_requests": upstream})
        if "usage credit" in str(row.get("description") or "").lower() or base in credit_billed_on_plan:
            entry["note"] = "usage credits"
    return list(routes.values()) or None
