# -*- coding: utf-8 -*-
"""Validation functions for the gentrie package."""

from collections.abc import Sequence

from .protocols import TrieKeyToken, GeneralizedKey


def is_triekeytoken(token: TrieKeyToken) -> bool:
    """Tests token for whether it is a valid :class:`TrieKeyToken`.

    A valid :class:`TrieKeyToken` is a hashable object (implements both ``__eq__()`` and ``__hash__()`` methods).

    Examples:
    :class:`bool`, :class:`bytes`, :class:`float`, :class:`frozenset`,
    :class:`int`, :class:`str`, :class:`None`, :class:`tuple`.

    Args:
        token (GeneralizedKey): Object for testing.

    Returns:
        :class:`bool`: ``True`` if a valid :class:`TrieKeyToken`, ``False`` otherwise.
    """
    return isinstance(
        token, TrieKeyToken)  # type: ignore[reportUnnecessaryIsInstance]


def is_hashable(token: TrieKeyToken) -> bool:
    """is_hashable is deprecated and will be removed in a future version.

    This function is a wrapper for :func:`is_triekeytoken` and is only provided for backward compatibility.

    Use :func:`is_triekeytoken` instead.
    """
    return is_triekeytoken(token)


def is_generalizedkey(key: GeneralizedKey) -> bool:
    """Tests key for whether it is a valid `GeneralizedKey`.

    A valid :class:`GeneralizedKey` is a :class:`Sequence` that returns
    :class:`TrieKeyToken` protocol conformant objects when
    iterated. It must have at least one token.

    Parameters:
        key (GeneralizedKey): Key for testing.

    Returns:
        :class:`bool`: ``True`` if a valid :class:`GeneralizedKey`, ``False`` otherwise.
    """
    return (
        isinstance(key, Sequence)  # type: ignore[reportUnnecessaryIsInstance]
        and len(key)
        and all(isinstance(t, TrieKeyToken) for t in key)  # type: ignore[reportUnnecessaryIsInstance]
    )
