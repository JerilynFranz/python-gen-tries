#  -*- coding: utf-8 -*-
"""Custom exceptions for the gentrie package."""


class InvalidTrieKeyTokenError(TypeError):
    """Raised when a token in a key is not a valid :class:`TrieKeyToken` object.

    This is a sub-class of :class:`TypeError`."""


class InvalidGeneralizedKeyError(TypeError):
    """Raised when a key is not a valid :class:`GeneralizedKey` object.

    This is a sub-class of :class:`TypeError`."""


class DuplicateKeyError(KeyError):
    """Raised when an attempt is made to add a key that is already in the trie
    with a different associated value.

    This is a sub-class of :class:`KeyError`."""
