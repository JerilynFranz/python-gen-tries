# -*- coding: utf-8 -*-
"""Base trie functionality and initialization."""

from copy import deepcopy
from textwrap import indent
from typing import Any, Optional

from ..nodes import Node
from ..protocols import TrieKeyToken
from ..types import TrieEntry, TrieId


class TrieBase:
    """Base class providing core trie structure and utilities."""

    def __init__(self) -> None:
        self.token: Optional[TrieKeyToken] = None
        self.value: Optional[Any] = None
        self.parent: Optional["TrieBase | Node"] = None
        self.children: dict[TrieKeyToken, Node] = {}
        self.ident: Optional[TrieId] = TrieId(0)
        # Counter for the next unique identifier to assign to a key in the trie.
        self._ident_counter: int = 0
        # Mapping of unique identifiers to their corresponding trie nodes.
        self._trie_index: dict[TrieId, Node] = {}
        # Mapping of unique identifiers to their corresponding TrieEntry instances.
        self._trie_entries: dict[TrieId, TrieEntry] = {}

    def clear(self) -> None:
        """Clears all keys from the trie."""
        self.ident = TrieId(0)
        self.token = None
        self.value = None
        self.parent = None
        self.children.clear()
        self._trie_index.clear()
        self._trie_entries.clear()
        # Reset the ident counter
        self._ident_counter = TrieId(0)

    def __len__(self) -> int:
        """Returns the number of keys in the trie."""
        return len(self._trie_index)

    def __str__(self) -> str:
        """Generates a stringified version of the trie for visual examination."""
        output: list[str] = ["{"]
        output.append(f"  trie number = {self._ident_counter}")
        if self.children:
            output.append("  children = {")
            for child_key, child_value in self.children.items():
                output.append(f"    {repr(child_key)} = " + indent(str(child_value), "    ").lstrip())
            output.append("  }")
        output.append(f"  trie index = {self._trie_index.keys()}")
        output.append("}")
        return "\n".join(output)

    def _as_dict(self) -> dict[str, Any]:
        """Converts the trie to a dictionary representation."""
        # pylint: disable=protected-access, no-member
        return deepcopy({
            "ident": self.ident,
            "children": {
                k: v._as_dict()  # type: ignore[reportPrivateUsage]
                for k, v in self.children.items()
            },
            "trie_index": sorted(self._trie_index.keys()),
            "trie_entries": {
                k: repr(v)
                for k, v in self._trie_entries.items()
            }
        })
