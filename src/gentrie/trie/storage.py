# -*- coding: utf-8 -*-
"""Entry storage operations for the trie."""

from typing import Any, Optional, TYPE_CHECKING

from ..exceptions import DuplicateKeyError, InvalidGeneralizedKeyError
from ..nodes import Node
from ..protocols import GeneralizedKey
from ..types import TrieEntry, TrieId
from ..validation import is_generalizedkey

if TYPE_CHECKING:
    from gentrie.trie.base import TrieBase as GeneralizedTrie


class TrieStorageMixin:
    """Mixin providing entry storage operations."""

    # Type hints for expected attributes (will be provided by mixing class)
    _trie_index: dict[TrieId, Any]
    _trie_entries: dict[TrieId, TrieEntry]
    ident: Optional[TrieId]
    children: dict[Any, Node]
    parent: Optional["GeneralizedTrie | Node"]
    value: Optional[Any]

    def add(self, key: GeneralizedKey, value: Optional[Any] = None) -> TrieId:
        """Adds the key to the trie.

        .. warning:: **Keys Must Be Immutable**

            Once a key is added to the trie, neither the key sequence itself nor any of its
            constituent tokens should be mutated. Modifying a key after it has been added
            can corrupt the internal state of the trie, leading to unpredictable behavior
            and making entries unreachable. The trie does not create a deep copy of keys
            for performance reasons.

            If you need to modify a key, you should remove the old key and add a new one
            with the modified value.

        Args:
            key (GeneralizedKey): Must be an object that can be iterated and that when iterated
                returns elements conforming to the :class:`TrieKeyToken` protocol.
            value (Optional[Any], default=None): Optional value to associate with the key.

        Raises:
            InvalidGeneralizedKeyError ([GTU001]):
                If key is not a valid :class:`GeneralizedKey`.
            DuplicateKeyError ([GTU002]):
                If the key is already in the trie.

        Returns:
            :class:`TrieId`: Id of the inserted key. If the key was not in the trie,
            it returns the id of the new entry. If the key was already in the trie,
            it raises a :class:`DuplicateKeyError`.
        """
        return self._store_entry(key=key, value=value, allow_value_update=False)

    def update(self, key: GeneralizedKey, value: Optional[Any] = None) -> TrieId:
        """Updates the key/value pair in the trie.

        .. warning:: **Keys Must Be Immutable**

            Once a key is added to the trie, neither the key sequence itself nor any of its
            constituent tokens should be mutated. Modifying a key after it has been added
            can corrupt the internal state of the trie, leading to unpredictable behavior
            and making entries unreachable. The trie does not create a deep copy of keys
            for performance reasons.

            If you need to modify a key, you should remove the old key and add a new one
            with the modified value.

        Args:
            key (GeneralizedKey): Must be an object that can be iterated and that when iterated
                returns elements conforming to the :class:`TrieKeyToken` protocol.
            value (Optional[Any], default=None): Optional value to associate with the key.

        Raises:
            InvalidGeneralizedKeyError ([GTSE001]):
                If key is not a valid :class:`GeneralizedKey`.

        Returns:
            :class:`TrieId`: Id of the inserted key. If the key was already in the trie with the same value
            it returns the id for the already existing entry. If the key was not already in the trie,
            it returns the id for a new entry.
        """
        return self._store_entry(key=key, value=value, allow_value_update=True)

    def _store_entry(self, key: GeneralizedKey, value: Any, allow_value_update: bool) -> TrieId:
        """Stores a key/value pair entry in the trie.

        Args:
            key (GeneralizedKey): Must be an object that can be iterated and that when iterated
                returns elements conforming to the :class:`TrieKeyToken` protocol.
            value (Optional[Any], default=None): Optional value to associate with the key.
            allow_value_update (bool):
                Whether to allow overwriting the value with a different value if the key already exists.
        Raises:
            InvalidGeneralizedKeyError ([GTSE001]):
                If key is not a valid :class:`GeneralizedKey`.
            DuplicateKeyError ([GTSE002]):
                If the key is already in the trie but with a different value and allow_value_update
                is False.

        Returns:
            :class:`TrieId`: Id of the inserted key. If the key was already in the trie with the same value
            it returns the id for the already existing entry. If the key was not in the trie,
            it returns the id of the new entry. If the key was already in the trie and allow_value_update
            is False, it raises a DuplicateKeyError. If allow_value_update is True, it replaces the value
            and returns the id of the existing entry.
        """
        if not is_generalizedkey(key):
            raise InvalidGeneralizedKeyError("[GTSE001] key is not a valid `GeneralizedKey`")

        # Traverse the trie to find the insertion point for the key,
        # creating nodes as necessary.
        current_node = self
        for token in key:
            if token not in current_node.children:
                child_node = Node(token=token, parent=current_node)  # type: ignore[reportArgumentType]
                current_node.children[token] = child_node
            current_node = current_node.children[token]

        # This key is already in the trie (it has a trie id)
        if current_node.ident:
            # If we allow updating, update the value and return the existing id
            if allow_value_update:
                current_node.value = value
                self._trie_entries[current_node.ident] = TrieEntry(current_node.ident, key, value)
                return current_node.ident

            # The key is already in the trie but we are not allowing updating values - so raise an error
            raise DuplicateKeyError(
                "[GTSE002] Attempted to store a key with a value that is already in the trie with "
                " - use `update()` to change the value of an existing key.")

        # Assign a new trie id for the node and set the value
        current_counter = getattr(self, '_ident_counter', 0)
        current_counter += 1
        setattr(self, '_ident_counter', current_counter)
        new_ident = TrieId(current_counter)
        current_node.ident = new_ident
        current_node.value = value
        trie_index = getattr(self, '_trie_index', {})
        trie_index[new_ident] = current_node
        setattr(self, '_trie_index', trie_index)
        trie_entries = getattr(self, '_trie_entries', {})
        trie_entries[new_ident] = TrieEntry(new_ident, key, value)
        setattr(self, '_trie_entries', trie_entries)
        return new_ident
