"""Setup for the provider is driven by the Claude CLI itself: its auth status gates the flow and its
own model picker (initialize handshake) supplies the list, with the pinned catalog as fallback."""
import json
import os
import sys
import textwrap


FAKE_CLI = textwrap.dedent('''
    import json, os, sys
    state = json.loads(os.environ["FAKE_STATE"])
    if sys.argv[1:3] == ["auth", "status"]:
        print(json.dumps(state["auth"])); sys.exit(0 if state["auth"]["loggedIn"] else 1)
    assert "-p" in sys.argv and "--input-format" in sys.argv, sys.argv
    req = json.loads(sys.stdin.readline())
    assert req["request"]["subtype"] == "initialize"
    if state.get("hang_upstream"):
        import urllib.request
        urllib.request.urlopen(os.environ["ANTHROPIC_BASE_URL"] + "/v1/messages", data=b"{}")
    print(json.dumps({"type": "control_response", "response": {"subtype": "success", "request_id": req["request_id"],
          "response": {"models": state["models"], "account": state["account"]}}}))
''')


def _cli(tmp_path, state):
    path = tmp_path / "claude.py"
    path.write_text(FAKE_CLI)
    return [sys.executable, str(path)], {**os.environ, "FAKE_STATE": json.dumps(state), "PATH": os.defpath}


def test_setup_status_reports_login_and_models_from_the_cli(profile, tmp_path):
    state = {"auth": {"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "pro"},
             "account": {"subscriptionType": "Claude Pro"},
             "models": [
                 {"value": "sonnet[1m]", "resolvedModel": "claude-sonnet-5[1m]", "displayName": "Sonnet 5 (1M context)", "description": "Sonnet 5 for long sessions"},
                 {"value": "opus", "resolvedModel": "claude-opus-5-5", "displayName": "Opus", "description": "Opus 5.5 · Best for everyday, complex tasks"},
                 {"value": "opus[1m]", "resolvedModel": "claude-opus-5-5[1m]", "displayName": "Opus (1M context)", "description": "Opus 5.5 with 1M context · Draws from usage credits · $4/$20 per Mtok"},
                 {"value": "haiku", "resolvedModel": "claude-haiku-4-5-20251001", "displayName": "Haiku", "description": "Haiku 4.5 · Fastest for quick answers"},
             ]}
    command, env = _cli(tmp_path, state)
    status = profile.setup_status(command=command, env=env)
    assert status["available"] and status["logged_in"] and status["plan"] == "Claude Pro"
    assert status["login_command"] == command + ["auth", "login"]

    models = profile.discover_models(command=command, env=env)
    ids = [m["id"] for m in models]
    # Native picker rows are deduplicated to their Hermes route ids (opus and opus[1m] both -> opus 1M)
    assert ids == ["claude-sonnet-5[1m]", "claude-opus-5-5[1m]", "claude-haiku-4-5-20251001"]
    assert [m["label"] for m in models] == ["Sonnet 5 for long sessions", "Opus 5.5", "Haiku 4.5"]
    assert models[1]["note"] == "usage credits"
    assert models[2]["note"] == ""
    # Discovery goes through the admission relay with zero upstream requests
    assert all(m["upstream_requests"] == 0 for m in models)


def test_a_model_the_pins_never_saw_arrives_from_the_picker(profile, tmp_path):
    """The point of the whole path: shipping a new Claude must not require a catalog commit.

    Both routes below are invented, so neither can be in the pinned catalog — they stand in for
    whatever Anthropic releases next. The CLI states the window by resolving a route to `<id>[1m]`
    when it offers that model at 1M, so nothing here is guessed.
    """
    state = {"auth": {"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "max"},
             "account": {"subscriptionType": "Claude Max"},
             "models": [
                 {"value": "zephyr[1m]", "resolvedModel": "claude-zephyr-9[1m]", "displayName": "Zephyr",
                  "description": "Zephyr 9 · Brand new", "supportsAdaptiveThinking": True},
                 {"value": "quill", "resolvedModel": "claude-quill-2", "displayName": "Quill",
                  "description": "Quill 2 · Short context"},
             ]}
    command, env = _cli(tmp_path, state)
    models = profile.discover_models(command=command, env=env)

    assert [m["id"] for m in models] == ["claude-zephyr-9[1m]", "claude-quill-2"]
    assert profile.get_model_context_length("claude-zephyr-9[1m]") == 1_000_000
    # Not blanket-promoted to 1M: a route the CLI offers only at the short window keeps it.
    assert profile.get_model_context_length("claude-quill-2") == 200_000
    # The family shortcut the account can actually type resolves to the new route too.
    assert profile.get_model_context_length("zephyr") == 1_000_000
    # A pinned route the picker has stopped listing survives the merge rather than being dropped.
    assert profile.get_model_context_length("claude-opus-4-8[1m]") == 1_000_000


def test_thinking_support_is_read_from_the_picker_not_the_pin(profile, tmp_path):
    """`supportsAdaptiveThinking` is stated per row; a new model that rejects adaptive thinking
    must be recorded as such without anyone editing the hardcoded quirk set."""
    import importlib

    state = {"auth": {"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "max"},
             "account": {"subscriptionType": "Claude Max"},
             "models": [
                 {"value": "zephyr[1m]", "resolvedModel": "claude-zephyr-9[1m]", "displayName": "Zephyr",
                  "description": "Zephyr 9", "supportsAdaptiveThinking": True},
                 {"value": "quill", "resolvedModel": "claude-quill-2", "displayName": "Quill",
                  "description": "Quill 2"},
             ]}
    command, env = _cli(tmp_path, state)
    assert profile.discover_models(command=command, env=env)

    catalog = importlib.import_module(type(profile).__module__ + ".model_catalog")
    assert catalog.supports_adaptive_thinking("claude-zephyr-9[1m]") is True
    # The Quill row omits `supportsAdaptiveThinking` entirely, which states nothing — and silence
    # must not be read as False. Only a row that says so disables adaptive thinking; otherwise a
    # future model is downgraded by a field the CLI simply did not send, which is exactly the
    # asymmetry that downgraded 1M windows. Known quirks stay covered by the pin below.
    assert catalog.supports_adaptive_thinking("claude-quill-2") is True
    # Absent from the picker entirely, the pinned quirk still stands.
    assert catalog.supports_adaptive_thinking("claude-haiku-4-5-20251001") is False


def test_a_bare_alias_row_never_downgrades_a_pinned_long_window(profile, tmp_path):
    """Silence is not evidence of a short window, and reading it as one loses 800K tokens.

    These are the shapes Claude Code 2.1.287 actually emits. From 2.1.284 the current generation
    is announced as a bare family alias -- `value: "opus"`, `resolvedModel: "claude-opus-5-5"`,
    no `[1m]` anywhere -- while the suffix survives only on secondary full-id rows, and there on
    `value` with `resolvedModel` left bare (Fable). Defaulting an unstated window to 200K
    therefore downgraded precisely the newest 1M models, silently and with no error on either
    side: Opus 5.5 and Sonnet 5.5 both, the day each shipped.
    """
    state = {"auth": {"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "max"},
             "account": {"subscriptionType": "Claude Max"},
             "models": [
                 # Pinned at 1M, announced bare. The pin must survive the merge.
                 {"value": "opus", "resolvedModel": "claude-opus-5-5", "displayName": "Opus",
                  "description": "Opus 5.5 · Best for everyday", "supportsAdaptiveThinking": True},
                 # The suffix rides `value` while `resolvedModel` stays bare: still 1M.
                 {"value": "claude-fable-5-1[1m]", "resolvedModel": "claude-fable-5-1",
                  "displayName": "Fable", "description": "Fable 5.1 · Hardest tasks",
                  "supportsAdaptiveThinking": True},
                 # Invented, unpinned, announced bare: genuinely unknown, so it must not be
                 # promoted to 1M either -- it runs at the gateway default and says so.
                 {"value": "sonnet", "resolvedModel": "claude-marlow-7", "displayName": "Marlow",
                  "description": "Marlow 7 · Brand new", "supportsAdaptiveThinking": True},
             ]}
    command, env = _cli(tmp_path, state)
    models = profile.discover_models(command=command, env=env)
    by_id = {m["id"]: m for m in models}

    # The regression: a bare alias row left these at 200K before.
    assert "claude-opus-5-5[1m]" in by_id
    assert profile.get_model_context_length("claude-opus-5-5[1m]") == 1_000_000
    assert "claude-fable-5-1[1m]" in by_id
    assert profile.get_model_context_length("claude-fable-5-1[1m]") == 1_000_000
    # Unknown stays unknown: routed bare at the gateway default, and flagged rather than hidden.
    assert "claude-marlow-7" in by_id
    assert by_id["claude-marlow-7"]["note"] == "unpinned"
    assert profile.get_model_context_length("claude-marlow-7") == 200_000
    # Discovery costs nothing upstream.
    assert all(m["upstream_requests"] == 0 for m in models)


def test_logged_out_or_missing_cli_degrades_to_pinned_catalog(profile, tmp_path):
    command, env = _cli(tmp_path, {"auth": {"loggedIn": False, "authMethod": "none"}, "account": {}, "models": []})
    status = profile.setup_status(command=command, env=env)
    assert status["available"] and not status["logged_in"]
    assert profile.discover_models(command=command, env=env) is None

    missing = profile.setup_status(command=[str(tmp_path / "nope")], env=env)
    assert not missing["available"] and not missing["logged_in"]
    assert "install" in missing["detail"].lower()
