"""Provider abstraction: rent a GPU, reach it, pay for it, give it back.

Nothing in this package is imported for its side effects, and ``infra.types`` imports
nothing internal, so ``store.ledger`` can persist an :class:`~infervolt.infra.types.Instance`
without a cycle back into ``infra.base``.
"""
