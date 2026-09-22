"""Catalog capacity must agree with the actual native --model selection.

Contracts, not a snapshot. The route list is data that is *expected* to change — a Claude release
adds one, and `learn()` folds it in from the live picker at runtime — so freezing ids and windows
here would fail on every routine refresh without covering anything. Assert how the pieces relate.
"""
import json
import os
import sys

from test_directsdk import FAKE


def test_every_catalog_route_reports_a_usable_window(profile):
    from agent.model_metadata import get_model_context_length
    from model_catalog import DEFAULT_CONTEXT_WINDOW, LONG_CONTEXT_WINDOW

    assert profile.fallback_models, 'the offline fallback list must not be empty'
    for model in profile.fallback_models:
        window = profile.get_model_context_length(model)
        assert window in (DEFAULT_CONTEXT_WINDOW, LONG_CONTEXT_WINDOW)
        assert get_model_context_length(model, provider=profile.name) == window
        assert get_model_context_length(model, provider=profile.name, config_context_length=200000) == 200000
        # The suffix is what selects the long window on a loopback gateway, so carrying it and
        # claiming 1M are the same statement.
        assert model.endswith('[1m]') == (window == LONG_CONTEXT_WINDOW)
        canonical = profile.model_metadata[model]['canonical_model']
        assert canonical.startswith('claude-')
        assert profile.get_model_context_length(canonical) == window
    assert profile.get_model_context_length('unqualified-future-model') is None


def test_aux_model_and_aliases_stay_inside_the_catalog(profile):
    """A default or alias pointing at a route the catalog dropped is how aux silently breaks."""
    assert profile.default_aux_model in profile.fallback_models
    for alias, route in profile.model_aliases.items():
        assert route in profile.fallback_models, f'alias {alias!r} points outside the catalog'


def test_native_model_is_idempotent(profile):
    """Core may hand back an already-resolved id; resolving twice must not append `[1m][1m]`."""
    from model_catalog import native_model

    for model in profile.fallback_models:
        assert native_model(model) == model
        assert native_model(native_model(model)) == model


def test_native_argv_selects_the_catalog_route_for_every_model_and_alias(profile, tmp_path):
    capture = tmp_path / 'argv.json'
    native = tmp_path / 'native.py'
    native.write_text(FAKE.replace('rows=[]', "pathlib.Path(os.environ['ARGV_CAPTURE']).write_text(json.dumps(sys.argv))\nrows=[]"))
    # Derived from the catalog rather than restated, so a new route is covered the day it lands.
    selections = {model: model for model in profile.fallback_models}
    selections.update(profile.model_aliases)
    # An id the catalog has never seen is passed through untouched rather than guessed at.
    selections['unqualified-future-model'] = 'unqualified-future-model'

    with_client = profile.create_client(command=[sys.executable, str(native)],
                                        env={'PATH': os.defpath, 'HOME': str(tmp_path), 'ARGV_CAPTURE': str(capture)})
    try:
        for requested, expected in selections.items():
            with_client.create(model=requested, messages=[{'role': 'user', 'content': 'fixture'}],
                               tools=[{'type': 'function', 'function': {'name': 'probe', 'description': 'TAIL', 'parameters': {'type': 'object', 'properties': {'value': {'type': 'string'}}}}}])
            argv = json.loads(capture.read_text())
            assert argv[argv.index('--model') + 1] == expected
    finally:
        with_client.close()
