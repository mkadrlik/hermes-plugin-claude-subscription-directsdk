"""Native routes: pinned as the offline fallback, refreshed from the account's own picker.

The pins answer what the picker cannot. It is unavailable when Claude Code is missing or logged
out, and it stops listing a version-pinned route once that route leaves the menu even though the
route still resolves. ``learn()`` folds a live reading in on top, so a model released after this
file was written arrives on its own instead of waiting for a catalog commit.
"""
DEFAULT_CONTEXT_WINDOW = 200_000
LONG_CONTEXT_WINDOW = 1_000_000

CONTEXT_WINDOWS = {
    'claude-sonnet-5': LONG_CONTEXT_WINDOW,
    'claude-haiku-4-5-20251001': DEFAULT_CONTEXT_WINDOW,
    'claude-opus-5-5': LONG_CONTEXT_WINDOW,
    'claude-opus-5': LONG_CONTEXT_WINDOW,
    'claude-opus-4-8': LONG_CONTEXT_WINDOW,
    'claude-fable-5-1': LONG_CONTEXT_WINDOW,
}
ALIASES = {
    'sonnet': 'claude-sonnet-5',
    'haiku': 'claude-haiku-4-5-20251001',
    'claude-haiku-4-5': 'claude-haiku-4-5-20251001',
    'opus': 'claude-opus-5-5',
    'fable': 'claude-fable-5-1',
}
# Haiku 4.5 rejects `thinking: {'type': 'adaptive'}` with a 400 upstream; its effort signal
# still applies. Unknown routes keep adaptive so future models are not silently downgraded.
# A live picker states this per model as `supportsAdaptiveThinking`, which `learn()` trusts
# over the pin. A set, not a frozenset, because `learn()` corrects it in place.
NO_ADAPTIVE_THINKING = {'claude-haiku-4-5-20251001'}


def native_model(model):
    base = model.removesuffix('[1m]')
    canonical = ALIASES.get(base, base)
    window = CONTEXT_WINDOWS.get(canonical)
    if window == LONG_CONTEXT_WINDOW:
        return canonical + '[1m]'
    if window == DEFAULT_CONTEXT_WINDOW:
        if model.endswith('[1m]'):
            raise ValueError('Haiku 4.5 does not support a 1M context window')
        return canonical
    return model


def supports_adaptive_thinking(model):
    # Body assembly precedes model validation, so an absent or invalid route answers True
    # and still reaches the existing `model is required` error.
    base = model.removesuffix('[1m]') if isinstance(model, str) else ''
    return ALIASES.get(base, base) not in NO_ADAPTIVE_THINKING


def _metadata():
    return {native_model(model): {'canonical_model': model, 'context_window': window}
            for model, window in CONTEXT_WINDOWS.items()}


MODEL_METADATA = _metadata()


def learn(models, aliases=()):
    """Fold a live picker reading into the pins, in place.

    In place because the provider profile holds ``MODEL_METADATA`` itself and core reads it for
    every context-length lookup; rebinding the name here would leave the profile on the stale
    dict. The picker wins for the routes it lists and says nothing about the rest, so a pinned
    route it has stopped offering (Opus 4.8) survives the merge.

    ``models`` maps a native base id to ``{context_window, supports_adaptive_thinking}``, either
    of which may be ``None`` for "the picker did not say". Only a stated fact moves a pin: the
    CLI can announce a route at 1M, but it has no way to announce one as *only* 200K, so an
    unknown window must leave the pin alone. Overwriting it with a default is how a derived
    catalog silently downgrades a 1M model to a fifth of its window with no error anywhere.
    """
    for base, facts in models.items():
        window = facts.get('context_window')
        if window:
            CONTEXT_WINDOWS[base] = window
        adaptive = facts.get('supports_adaptive_thinking')
        if adaptive is True:
            NO_ADAPTIVE_THINKING.discard(base)
        elif adaptive is False:
            NO_ADAPTIVE_THINKING.add(base)
    ALIASES.update(aliases)
    MODEL_METADATA.clear()
    MODEL_METADATA.update(_metadata())
