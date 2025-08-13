# -*- coding: utf-8 -*-
"""Iterator operations for the trie."""

from typing import Generator

from ..types import TrieEntry, TrieId


class TrieIterationMixin:
    """Mixin providing iteration operations."""

    # Type hints for expected attributes (will be provided by mixing class)
    _trie_entries: dict[TrieId, TrieEntry]

    def __iter__(self) -> Generator[TrieId, None, None]:
        """Returns an iterator for the trie.

        The generator yields the :class:`TrieId`for each key in the trie.

        Returns:
            :class:`Generator[TrieId, None, None]`: Generator for the trie.
        """
        return (entry for entry in self._trie_entries.keys())

    def keys(self) -> Generator[TrieId, None, None]:
        """Returns an iterator for all the TrieIds in the trie.

        The generator yields the :class:`TrieId` for each key in the trie.

        It returns TrieIds instead of GeneralizedKeys because TrieIds are

        1. Faster: Lookups using TrieIds are *O(1)* for time regardless
           of the length of the GeneralizedKey they are associated with vs *O(n)*
           to the length of keys for operations using GeneralizedKeys to look
           up entries.

        2. More efficient memory usage: TrieIds are typically smaller in size
           compared to GeneralizedKeys, leading to reduced memory overhead
           when storing and processing keys in the trie.

        3. Guaranteed stable even with key modifications: TrieIds remain
           consistent even if the underlying GeneralizedKey changes, making
           them more reliable for long-term storage and retrieval.

        Returns:
            :class:`Generator[TrieId, None, None]`: Generator for the trie.
        """
        return (entry for entry in self._trie_entries.keys())

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
