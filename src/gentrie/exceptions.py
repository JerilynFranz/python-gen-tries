#  -*- coding: utf-8 -*-
"""Custom exceptions for the gentrie package."""


class TrieTypeError(TypeError):
    """Base class for all trie-related type errors.

    It differs from a standard TypeError by the addition of a
    tag code used to very specifically identify where the error
    was thrown in the code for testing and development support.

    This tag code does not have a direct semantic meaning except to identify
    the specific code throwing the exception for tests.
    """
    def __init__(self, msg: str, tag: str = '') -> None:
        """Create a new TrieTypeError.

        Args:
            msg (str): The error message.
            tag (str): The tag code.
        """
        self.tag_code: str = tag.upper()
        super().__init__(msg)


class TrieKeyError(KeyError):
    """Base class for all trie-related key errors.

    It differs from a standard KeyError by the addition of a
    tag code used to very specifically identify where the error
    was thrown in the code for testing and development support.

    This tag code does not have a direct semantic meaning except to identify
    the specific code throwing the exception for tests.

    Args:
        msg (str): The error message.
        tag (str): The tag code.
    """
    def __init__(self, msg: str, tag: str = '') -> None:
        self.tag_code: str = tag.upper()
        super().__init__(msg)


class InvalidTrieKeyTokenError(TrieTypeError):
    """Raised when a token in a key is not a valid :class:`TrieKeyToken` object.

    This is a sub-class of :class:`TrieTypeError`."""


class InvalidGeneralizedKeyError(TrieTypeError):
    """Raised when a key is not a valid :class:`GeneralizedKey` object.

    This is a sub-class of :class:`TrieTypeError`."""


class DuplicateKeyError(TrieKeyError):
    """Raised when an attempt is made to add a key that is already in the trie
    with a different associated value.

    This is a sub-class of :class:`TrieKeyError`."""
