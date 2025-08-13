# -*- coding: utf-8 -*-
"""Collection operations for GeneralizedTrie."""

from typing import Any, Generator

from ..types import TrieEntry, TrieId


class TrieCollectionMixin:
    """Mixin providing collection operations (__len__, __iter__, keys, values, items)."""

    # Type hints for expected attributes (will be provided by mixing class)
    _trie_index: dict[TrieId, Any]
    _trie_entries: dict[TrieId, TrieEntry]

    def __len__(self) -> int:
        """Returns the number of keys in the trie.

        Returns:
            :class:`int`: Number of keys in the trie.

        Usage::

            n_keys: int = len(trie)

        """
        return len(self._trie_index)

    def __iter__(self) -> Generator[TrieId, None, None]:
        """Returns an iterator for the trie.

        The generator yields the :class:`TrieId`for each key in the trie.

        Returns:
            :class:`Generator[TrieId, None, None]`: Generator for the trie.
        """
        return (entry for entry in self._trie_entries.keys())  # pylint: disable=consider-iterating-dictionary

    def keys(self) -> Generator[TrieId, None, None]:
        """Returns an iterator for all the TrieIds in the trie.

        [Full docstring from original implementation]
        """
        return (entry for entry in self._trie_entries.keys())  # pylint: disable=consider-iterating-dictionary

    def values(self) -> Generator[TrieEntry, None, None]:
        """Returns an iterator for all the TrieEntry entries in the trie.

        The generator yields the :class:`TrieEntry` for each key in the trie.

        Returns:
            :class:`Generator[TrieEntry, None, None]`: Generator for the trie.
        """
        return (entry for entry in self._trie_entries.values())

    def items(self) -> Generator[tuple[TrieId, TrieEntry], None, None]:
        """Returns an iterator for the trie.

        The keys are the TrieIds and the values are the TrieEntry instances.

        The generator yields the :class:`TrieId` and :class:`TrieEntry` for each key in the trie.

        Returns:
            :class:`Generator[tuple[TrieId, TrieEntry], None, None]`: Generator for the trie.
        """
        return ((key, value) for key, value in self._trie_entries.items())
