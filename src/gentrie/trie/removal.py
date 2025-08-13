# -*- coding: utf-8 -*-
"""Entry removal operations for the trie."""
from abc import ABC, abstractmethod
from typing import Any, Optional

from ..protocols import GeneralizedKey, TrieKeyToken
from ..types import TrieEntry, TrieId
from ..validation import is_generalizedkey


class TrieRemovalMixin(ABC):
    """Mixin providing entry removal operations."""

    # Type hints for expected attributes (will be provided by mixing class)
    _trie_index: dict[TrieId, Any]
    _trie_entries: dict[TrieId, TrieEntry]

    @abstractmethod
    def __getitem__(self, key: TrieId | GeneralizedKey) -> TrieEntry:
        """__getitem__ method must be implemented by the mixing class."""

    def remove(self, key: TrieId | GeneralizedKey) -> None:
        """Remove the specified key from the trie.

        Removes the key from the trie. If the key is not found, it raises a KeyError.
        The key can be specified either as a :class:`TrieId` or as a :class:`GeneralizedKey`.

        Args:
            key (TrieId | GeneralizedKey): identifier for key to remove.

        Raises:
            TypeError ([GTR001]): if the key arg is not a :class:`TrieId` or a valid :class:`GeneralizedKey`.
            KeyError ([GTR002]): if the key arg does not match the id or trie key of any entries in the trie.
        """
        ident: Optional[TrieId] = None
        if isinstance(key, TrieId):
            ident = key
        elif is_generalizedkey(key):
            try:
                ident = self[key].ident
            except KeyError:
                ident = None
            except TypeError as exc:
                raise RuntimeError("[GTR003] failed lookup of key because of unexpected exception") from exc
        else:
            raise TypeError(
                "[GTR001] key arg must be of type TrieId or a valid GeneralizedKey"
            )

        if ident is None or ident not in self._trie_index:
            raise KeyError("[GTR002] key not found")

        # Get the node and delete its id from the trie index and entries
        # and remove the node from the trie.
        node = self._trie_index[ident]
        del self._trie_index[ident]
        del self._trie_entries[ident]

        # Remove the id from the node
        node.ident = TrieId(0)

        # If the node still has other trie ids or children, we're done: return
        if node.children:
            return

        # No trie ids or children are left for this node, so prune
        # nodes up the trie tree as needed.
        token: Optional[TrieKeyToken] = node.token
        parent = node.parent
        while parent is not None:
            del parent.children[token]
            # explicitly break possible cyclic references
            node.parent = node.token = None

            # If the parent node has a trie id or children, we're done: return
            if parent.ident or parent.children:
                return
            # Keep purging nodes up the tree
            token = parent.token
            node = parent
            parent = node.parent
        return
