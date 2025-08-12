# -*- coding: utf-8 -*-
# pylint: disable=too-many-lines
"""Package providing a generalized trie implementation.

This package includes classes and functions to create and manipulate a generalized trie
data structure. Unlike common trie implementations that only support strings as keys,
this generalized trie can handle various types of tokens, as long as they are hashable.

.. warning:: **gentrie IS NOT thread-safe**

    It is not designed for concurrent use. If you need a thread-safe trie, you must
    use an external locking mechanism to ensure that either only read-only threads are
    accessing the trie at the same time or that only one write thread and no read threads
    are accessing the trie at the same time.

    One way to do this is to create TWO instances of the trie, one for read-only access and one for write access.

    The read-only instance can be used by multiple threads at the same time, while the write instance can
    be used by only one thread at a time. After a write operation, a new read-only instance can be created
    from the write instance by forking it using :func:`copy.deepcopy()` which can then be used by the
    read-only threads.

Usage
=======

You can create a trie using the `GeneralizedTrie` class:

.. code-block:: python
    :linenos:

    from gentrie import GeneralizedTrie

    # Create a new trie instance
    trie = GeneralizedTrie()

There are three ways to add entries:

1. Using the `trie[key] = value` syntax

This allows you to assign a value directly to a key and will create a
new TrieEntry if the key does not already exist. If the key already exists,
it will update the value associated with that key.

.. code-block:: python
    :linenos:
    :caption: Examples of using the `trie[key] = value` syntax

    # Assigns 'value' to 'key'
    # (tokenized as characters 'k','e','y')
    trie['key'] = 'value'

    # Assigns value 'value2' to key 'another_key' (tokenized as
    # 'a','n','o','t','h','e','r','_','k','e','y','2')
    trie['another_key'] = 'another_value'

    # Changes the value for 'key' (tokenized as 'k', 'e', 'y')
    # to 'new_value'
    trie['key'] = 'new_value'

    # Assigns a tuple of int (tokenized as
    # 128, 96, 160, 0) as a key with the value'value5'
    trie[(128, 96, 160, 0)] = 'value5'

    # Assigns a tuple with mixed value types (tokenized
    # as 128, 'a') as a key with the value 'value5b'
    trie[(128, 'a')] = 'value5b'

    # Assigns a list of words (tokenized as 'hello', 'world')
    # as a key with the value 'value6'
    trie[['hello', 'world']] = 'value6'

2. Using the `trie.add(key, value)` method

    This method adds a new entry to the trie and returns the TrieId
    for the new entry. If the key already exists, it will throw an
    error. The value argument is optional, and if not provided
    the entry will be created with a value of `None`.

3. Using the `trie.update(key, value)` method

    This method adds a new entry or updates an existing entry,
    returning the TrieId for the entry and returns the TrieId of the entry.

    This is the same as using the `trie[key] = value` syntax,
    but it is more explicit about the intention to update or add
    an entry and returns the TrieId of the entry.

    The value argument is optional, and if not provided, the
    entry will be created or updated with a value of `None`.

You can use the `in` operator to check if a key exists in the trie,
e.g., `if key in trie:`. This will return `True` if the key exists,
and `False` otherwise.

There are two ways to directly retrieve entries using their keys:

1. Using the `trie[key | TrieId]` syntax.

    This retrieves the `TrieEntry` associated with the key or TrieId. It
    will raise a `KeyError` if the key/TrieId does not exist. It will raise
    a `TypeError` if the key is not either a valid `TrieId` or `GeneralizedKey`.

    The returned `TrieEntry` contains the key, value, and an identifier
    (ident - of type `TrieId`) that uniquely identifies the entry in the trie.

2. Using the `trie.get(key | TrieId, [default])` method
    This retrieves the `TrieEntry` associated with the key or TrieId,
    returning `None` if the key/TrieId does not exist. This could be
    preferable in cases where you want to avoid exceptions
    for missing keys although it cannot distinguish between a key
    that does not exist and a key that exists with a `None` value
    in the trie by default.

    You *can* provide a different default value to return if the key
    does not exist, which can be useful for handling cases where
    you want to return a specific value instead of `None`.

You can also retrieve all entries that are prefixed by or prefixes for a given key:

- `trie.prefixed_by(key)` returns a Generator of `TrieEntry` objects that
  are prefixed_by of the given key.
- `trie.prefixes(key)` returns a Generator of `TrieEntry` objects that
  are prefixes of the given key.

These methods are useful for searching and retrieving entries
that match a specific pattern or structure in the trie.

Example 1 - Basic Usage
------------------------

.. code-block:: python
    :linenos:

    from gentrie import GeneralizedTrie, TrieEntry

    trie = GeneralizedTrie()
    trie.add(['ape', 'green', 'apple'])
    trie.add(['ape', 'green'])
    matches: set[TrieEntry] = set(trie.prefixes(['ape', 'green']))
    print(matches)

Value of 'matches'::

    {TrieEntry(ident=2, key=['ape', 'green'], value=None)}

Example 2 - Trie of URLs
--------------------------

.. code-block:: python
    :linenos:

    from gentrie import GeneralizedTrie, TrieEntry

    # Create a trie to store website URLs
    url_trie = GeneralizedTrie()

    # Add some URLs with different components (protocol, domain, path)
    url_trie.add(["https", "com", "example", "www", "/", "products", "clothing"], value="Clothing Store")
    url_trie.add(["http", "org", "example", "blog", "/", "2023", "10", "best-laptops"], value="Best Laptops 2023")
    url_trie.add(["ftp", "net", "example", "ftp", "/", "data", "images"], value="FTP Data Images")

    # Find all https URLs with "example.com" domain
    prefixed_by: set[TrieEntry] = set(url_trie.prefixed_by(["https", "com", "example"]))
    print(prefixed_by)

Value of 'prefixed_by'::

    {
        TrieEntry(
            ident=1,
            key=['https', 'com', 'example', 'www', '/', 'products', 'clothing'],
            value='Clothing Store')
    }

Example 3 - Entries prefixed by a key
-------------------------------------

.. code-block:: python
    :linenos:

    from gentrie import GeneralizedTrie, TrieEntry

    trie = GeneralizedTrie()
    trie.add('abcdef')
    trie.add('abc')
    trie.add('qrf')
    matches: set[TrieEntry] = set(trie.prefixed_by('ab'))
    print(matches)

Value of 'matches'::

    {
        TrieEntry(ident=2, key='abc', value=None),
        TrieEntry(ident=1, key='abcdef', value=None)
    }

"""

from collections import deque
from collections.abc import Sequence
from copy import deepcopy
from textwrap import indent
from typing import Any, runtime_checkable, Generator, Optional, Protocol, NamedTuple, TypeAlias

# Constants for TrieEntry fields (performance optimization)
TRIE_IDENT: int = 0
"""Alias for field number 0 (ident) in TrieEntry. It is faster to use this than accessing the field by name."""
TRIE_KEY: int = 1
"""Alias for field number 1 (key) in TrieEntry. It is faster to use this than accessing the field by name."""
TRIE_VALUE: int = 2
"""Alias for field number 2 (value) in TrieEntry. It is faster to use this than accessing the field by name."""


class InvalidTrieKeyTokenError(TypeError):
    """Raised when a token in a key is not a valid :class:`TrieKeyToken` object.

    This is a sub-class of :class:`TypeError`."""


class InvalidGeneralizedKeyError(TypeError):
    """Raised when a key is not a valid :class:`GeneralizedKey` object.

    This is a sub-class of :class:`TypeError`."""


class DuplicateKeyError(KeyError):
    """Raised when an attempt is made to add a key that is already in the trie with a different associated value.

    This is a sub-class of :class:`KeyError`."""


@runtime_checkable
class TrieKeyToken(Protocol):
    """:class:`TrieKeyToken` is a protocol that defines key tokens that are usable with a :class:`GeneralizedTrie`.

    The protocol requires that a token object be *hashable*. This means that it
    implements both an ``__eq__()`` method and a ``__hash__()`` method.

    Some examples of built-in types that are suitable for use as tokens in a key:

    * :class:`str`
    * :class:`bytes`
    * :class:`int`
    * :class:`float`
    * :class:`complex`
    * :class:`frozenset`
    * :class:`tuple`
    * :class:`None`

    Note: frozensets and tuples are only hashable *if their contents are hashable*.

    Usage:

    .. code-block:: python
        :linenos:

        from gentrie import TrieKeyToken

        token = SomeTokenClass()
        if isinstance(token, TrieKeyToken):
            print("supports the TrieKeyToken protocol")
        else:
            print("does not support the TrieKeyToken protocol")

    .. warning:: **Using User Defined Classes As Tokens In Keys**

        User-defined classes are hashable by default, but you should implement the
        ``__eq__()`` and ``__hash__()`` dunder methods in a content-aware way (the hash and eq values
        must depend on the content of the object) if you want to use them as tokens in a key. The default
        implementation of ``__eq__()`` and ``__hash__()`` uses the memory address of the object, which
        means that two different instances of the same class will not be considered equal.

    """
    def __eq__(self, value: object, /) -> bool:
        ...

    def __hash__(self) -> int:
        ...


@runtime_checkable
class Hashable(TrieKeyToken, Protocol):
    """The Hashable protocol is deprecated and will be removed in a future version.

    This protocol is a sub-class of :class:`TrieKeyToken` and is only provided for backward compatibility.

    Use :class:`TrieKeyToken` instead.
    """
    def __eq__(self, value: object, /) -> bool:
        ...

    def __hash__(self) -> int:
        ...


GeneralizedKey: TypeAlias = Sequence[TrieKeyToken | str]
"""A :class:`GeneralizedKey` is an object of any class that is a :class:`Sequence` and
that when iterated returns tokens conforming to the :class:`TrieKeyToken` protocol.


.. warning:: **Keys Must Be Immutable**

    Once a key is added to the trie, neither the key sequence itself nor any of its
    constituent tokens should be mutated. Modifying a key after it has been added
    can corrupt the internal state of the trie, leading to unpredictable behavior
    and making entries unreachable. The trie does not create a deep copy of keys
    for performance reasons.

    If you need to modify a key, you should remove the old key and add a new one
    with the modified value.

**Examples of valid :class:`GeneralizedKey` types**

* :class:`str`
* :class:`bytes`
* :class:`list[bool]`
* :class:`list[int]`
* :class:`list[bytes]`
* :class:`list[str]`
* :class:`list[Optional[str]]`
* :class:`tuple[int, int, str]`

"""


class TrieId(int):
    """Unique identifier for a key in a trie."""
    __slots__ = ()

    def __new__(cls, value: int):
        return int.__new__(cls, value)

    def __str__(self) -> str:
        """Returns a string representation of the TrieId."""
        return f'TrieId({int(self)})'

    def __repr__(self) -> str:
        """Returns a string representation of the TrieId for debugging."""
        return f'TrieId({int(self)})'


class TrieEntry(NamedTuple):
    """A :class:`TrieEntry` is a :class:`NamedTuple` containing the unique identifer and key for an entry in the trie.
    """

    ident: TrieId
    """:class:`TrieId` Unique identifier for a key in the trie. Alias for field number 0."""
    key: GeneralizedKey
    """:class:`GeneralizedKey` Key for an entry in the trie. Alias for field number 1."""
    value: Optional[Any] = None
    """Optional value for the entry in the trie. Alias for field number 2."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrieEntry):
            return False
        return self.ident == other.ident and tuple(self.key) == tuple(
            other.key)

    def __hash__(self) -> int:
        return hash((self.ident, tuple(self.key)))


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


class _Node:  # pylint: disable=too-few-public-methods
    """A node in the trie.

    A node is a container for a key in the trie. It has a unique identifier
    and a reference to the key.

    Attributes:
        ident (TrieId): Unique identifier for the key.
        token (TrieKeyToken): Token for the key.
        parent (Optional[GeneralizedTrie | _Node): Reference to the parent node.
        children (dict[TrieKeyToken, _Node]): Dictionary of child nodes.
    """

    def __init__(self, token: TrieKeyToken, parent: "GeneralizedTrie | _Node", value: Optional[Any] = None) -> None:
        self.ident: Optional[TrieId] = None
        self.token: TrieKeyToken = token
        self.value: Optional[Any] = value
        self.parent: Optional[GeneralizedTrie | _Node] = parent
        self.children: dict[TrieKeyToken, _Node] = {}

    def __str__(self) -> str:
        """Generates a stringified version of the trie for visual examination.

        The output IS NOT executable code but more in the nature of debug and testing support."""
        output: list[str] = ["{"]
        if self.parent is None:
            output.append("  parent = None")
        elif isinstance(self.parent, GeneralizedTrie):
            output.append("  parent = root node")
        else:
            output.append(f"  parent = {repr(self.parent.token)}")
        output.append(f"  node token = {repr(self.token)}")
        if self.ident:
            output.append(f"  trie id = {self.ident}")
        if self.children:
            output.append("  children = {")
            for child_key, child_value in self.children.items():
                output.append(f"    {repr(child_key)} = " + indent(str(child_value), "    ").lstrip())
            output.append("  }")
        output.append("}")
        return "\n".join(output)

    def _as_dict(self) -> dict[str, Any]:
        """Converts the node to a dictionary representation.

        This is useful for tests and debugging purposes and is not intended
        for general purpose serialization of the trie. It's output is not
        suitable for use with :func:`json.dumps()` or similar functions and
        is subject to change without notice. This is NOT a public API - it is
        intended for internal use by tests only.

        Returns:
            :class:`dict[str, Any]`: Dictionary representation of the node.
            The dictionary contains the following keys:
                - "ident": The unique identifier of the node.
                - "token": The token of the node.
                - "value": The value associated with the node.
                - "parent": The token of the parent node, or None if there is no parent.
                - "children": A dictionary of child nodes, where the keys are the tokens
                  of the child nodes and the values are dictionaries representing the child nodes.
        """
        # pylint: disable=protected-access
        # Using deepcopy to ensure that the dictionary is a copy of the data in the trie,
        # not a dictionary of live references to it
        return deepcopy(
            {
                "ident": self.ident,
                "token": self.token,
                "value": self.value,
                "parent": self.parent.token if self.parent else None,
                "children": {str(k): v._as_dict() for k, v in self.children.items()},
            }
        )


class GeneralizedTrie:  # pylint: disable=too-many-instance-attributes
    """A general purpose trie.

    Unlike many trie implementations which only support strings as keys
    and token match only at the character level, it is agnostic as to the
    types of tokens used to key it and thus far more general purpose.

    It requires only that the indexed tokens be hashable. This is verified
    at runtime using the :class:`gentrie.TrieKeyToken` protocol.

    Tokens in a key do NOT have to all be the same type as long as they
    can be compared for equality.

    It can handle a :class:`Sequence` of :class:`TrieKeyToken` conforming objects as keys
    for the trie out of the box.

    You can 'mix and match' types of objects used as token in a key as
    long as they all conform to the :class:`TrieKeyToken` protocol.

    The code emphasizes robustness and correctness.

    .. warning:: **GOTCHA: Using User Defined Classes As Tokens In Keys**

        Objects of user-defined classes are conformant with the :class:`TrieKeyToken` protocol
        by default, but **this will not work as naively expected.** The hash value of an object
        is based on its memory address by default. This results in the hash value of an object changing
        every time the object is created and means that the object will not be found in
        the trie unless you have a reference to the original object.

        If you want to use a user-defined class as a token in a key to look up by value
        instead of the instance, you must implement the ``__eq__()`` and ``__hash__()``
        dunder methods in a content aware way (the hash and eq values must depend on the
        content of the object).

        .. tip:: **Using `dataclasses.dataclass` For Content-Aware User Defined Classes**

            A simple way to implement a user-defined class that is content aware hashable
            is to use the :class:`dataclasses.dataclass` decorator using the ``frozen=True`` and
            ``eq=True`` options . This will automatically implement appropriate ``__eq__()``
            and ``__hash__()`` methods for you.

            .. code-block:: python
                :linenos:
                :caption: Example of a content-aware user-defined class

                from dataclasses import dataclass

                from gentrie import TrieKeyToken

                @dataclass(frozen=True, eq=True)
                class MyTokenClass:
                    name: str
                    value: int

                # Create an instance of the token class
                token = MyTokenClass(name="example", value=42)

                # Check if the token is hashable
                if isinstance(token, TrieKeyToken):
                    print("token is usable as a TrieKeyToken")
                else:
                    print("token is not usable as a TrieKeyToken")

    """

    def __init__(self) -> None:
        self.token: Optional[TrieKeyToken] = None
        self.value: Optional[Any] = None
        self.parent: Optional[GeneralizedTrie | _Node] = None
        self.children: dict[TrieKeyToken, _Node] = {}
        self.ident: TrieId = TrieId(0)
        # Counter for the next unique identifier to assign to a key in the trie.
        self._ident_counter: int = 0
        # Mapping of unique identifiers to their corresponding trie nodes.
        self._trie_index: dict[TrieId, _Node] = {}
        # Mapping of unique identifiers to their corresponding TrieEntry instances.
        self._trie_entries: dict[TrieId, TrieEntry] = {}

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
        return self._store_entry(key=key,
                                 value=value,
                                 allow_value_update=False)

    def update(self,
               key: GeneralizedKey,
               value: Optional[Any] = None) -> TrieId:
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

    def _store_entry(self, key: GeneralizedKey, value: Any,
                     allow_value_update: bool) -> TrieId:
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
                child_node = _Node(token=token, parent=current_node)
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
        self._ident_counter += 1
        new_ident = TrieId(self._ident_counter)
        current_node.ident = new_ident
        current_node.value = value
        self._trie_index[new_ident] = current_node  # type: ignore[assignment]
        self._trie_entries[new_ident] = TrieEntry(new_ident, key, value)
        return new_ident

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
        node: GeneralizedTrie | _Node = self._trie_index[ident]
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

    def prefixes(self, key: GeneralizedKey) -> Generator[TrieEntry, None, None]:
        """Yields TrieEntry instances for all keys in the trie that are a prefix of the passed key.

        Searches the trie for all keys that are prefix matches
        for the key and yields their TrieEntry instances.

        .. note::

            The `prefixes` method finds all keys that are prefixes of the passed
            key.  For example, `trie.prefixes('apple')` will find entries for
            keys like 'a', 'apple' and 'app'.

        .. warning:: **GOTCHA: Generators**

            Because generators are not executed until the first iteration,
            they may not behave as expected if not consumed properly. For example,
            exceptions will not be raised until the generator is iterated over.

        Args:
            key (GeneralizedKey): Key for matching.

        Yields:
            :class:`TrieEntry`: The next matching :class:`TrieEntry` instance.

        Raises:
            InvalidGeneralizedKeyError ([GTM001]):
                If key is not a valid :class:`GeneralizedKey`
                (is not a :class:`Sequence` of :class:`TrieKeyToken` objects).

        Usage::

            from gentrie import GeneralizedTrie, TrieEntry

            trie: GeneralizedTrie = GeneralizedTrie()
            keys: list[str] = ['abcdef', 'abc', 'a', 'abcd', 'qrs']
            for entry in keys:
                trie.add(entry)
            matches_generator: Generator[TrieEntry, None, None] = trie.prefixes('abcd')
            for trie_entry in sorted(list(matches_generator)):
                print(f'{trie_entry.ident}: {trie_entry.key}')

            # 2: abc
            # 3: a
            # 4: abcd

        """
        if not is_generalizedkey(key):
            raise InvalidGeneralizedKeyError("[GTM001] key is not a valid `GeneralizedKey`")

        current_node = self

        for token in key:
            if current_node.ident:
                yield self._trie_entries[current_node.ident]
            if token not in current_node.children:
                return  # no match in children, so the generator is done
            current_node = current_node.children[token]

        # If we reached here, we have a match for the full key
        # Yield the current node's entry if it has an ident
        # This is the case where the key is an exact match for a key in the trie
        # and not just a prefix match.
        # If the key is a prefix of a key in the trie, it will not have
        # an ident, so we do not yield it.
        if current_node.ident:
            yield self._trie_entries[current_node.ident]

    def prefixed_by(self, key: GeneralizedKey, depth: int = -1) -> Generator[TrieEntry, None, None]:
        """Yields all entries in the trie that are prefixed by the given key, up to a specified depth.

        Searches the trie for all keys that start with the provided key and yields their
        :class:`TrieEntry` instances.

        .. note::
            The `prefixed_by` method finds all keys that start with the given
            prefix. For example, `trie.prefixed_by('app')` will find entries for
            keys like 'apple' and 'application'.

        .. warning:: **GOTCHA: Generators**

            Because generators are not executed until the first iteration,
            they may not behave as expected if not consumed properly. For example,
            exceptions will not be raised until the generator is iterated over.

        Args:
            key (GeneralizedKey): Key for matching.
            depth (`int`, default=-1): Depth starting from the matched key to include.
                The depth determines how many 'layers' deeper into the trie to look for prefixed_by.:
                * A depth of -1 (the default) includes ALL entries for the exact match and all children nodes.
                * A depth of 0 only includes the entries for the *exact* match for the key.
                * A depth of 1 includes entries for the exact match and the next layer down.
                * A depth of 2 includes entries for the exact match and the next two layers down.

        Yields:
            :class:`TrieEntry`: The next matching :class:`TrieEntry` instance.

        Raises:
            InvalidGeneralizedKeyError ([GTS001]):
                If key arg is not a GeneralizedKey.
            TypeError ([GTS002]):
                If depth arg is not an int.
            ValueError ([GTS003]):
                If depth arg is less than -1.
            InvalidGeneralizedKeyError ([GTS004]):
                If a token in the key arg does not conform to the :class:`TrieKeyToken` protocol.

        Usage::

            from gentrie import GeneralizedTrie, TrieEntry

            trie = GeneralizedTrie()
            keys: list[str] = ['abcdef', 'abc', 'a', 'abcd', 'qrs']
            for entry in keys:
                trie.add(entry)
            matches_generator = trie.prefixed_by('abcd')

            for trie_entry in sorted(list(matches_generator)):
                print(f'{trie_entry.ident}: {trie_entry.key}')

            # 1: abcdef
            # 4: abcd

        """
        if not is_generalizedkey(key):
            raise InvalidGeneralizedKeyError("[GTS001] key arg is not a valid GeneralizedKey")

        if not isinstance(depth, int):  # type: ignore[reportUnnecessaryIsInstance]
            raise TypeError("[GTS002] depth must be an int")
        if depth < -1:
            raise ValueError("[GTS003] depth cannot be less than -1")

        current_node = self
        try:
            for token in key:
                current_node = current_node.children[token]
        except KeyError:
            return  # No match, so the generator is empty

        # Perform a breadth-first search to collect prefixed keys up to the specified depth
        queue = deque([(current_node, depth)])

        while queue:
            node, current_depth = queue.popleft()
            if node.ident:
                yield self._trie_entries[node.ident]
            if current_depth != 0:
                for child in node.children.values():
                    queue.append((child, current_depth - 1))

    def clear(self) -> None:
        """Clears all keys from the trie.

        Usage::

            trie_obj.clear()

        """
        self.ident = TrieId(0)
        self.token = None
        self.value = None
        self.parent = None
        self.children.clear()
        self._trie_index.clear()
        self._trie_entries.clear()
        # Reset the ident counter
        self._ident_counter = TrieId(0)

    def __contains__(self, key_or_ident: GeneralizedKey | TrieId) -> bool:
        """Returns True if the trie contains a GeneralizedKey or TrieId matching the passed key.

        This method checks if the trie contains a key that matches the provided key_or_ident.
        The key can be specified either as a :class:`GeneralizedKey` or as a :class:`TrieId`.

        A lookup by :class:`TrieId` is a fast operation (*O(1)* time) while a lookup by :class:`GeneralizedKey`
        involves traversing the trie structure to find a matching key (*O(n)* time in the worst case,
        where n is the key length).

        Args:
            key_or_ident (GeneralizedKey | TrieId): Key or TrieId for matching.

        Returns:
            :class:`bool`: True if there is a matching GeneralizedKey/TrieId in the trie. False otherwise.

        Raises:
            :class:`TypeError`:
                If key arg is not a GeneralizedKey or TrieId.

        Usage::

            trie = GeneralizedTrie()
            keys: list[str] = ['abcdef', 'abc', 'a', 'abcd', 'qrs']
            idents: list[TrieId] = []
            for entry in keys:
                idents.append(trie.add(entry))

            if 'abc' in trie:
                print('"abc" is in the trie')

            if idents[0] in trie:
                print(f'Ident {idents[0]} is in the trie')
        """
        if isinstance(key_or_ident, TrieId):
            # If it's a TrieId, check if it exists in the trie index
            return key_or_ident in self._trie_index

        if not is_generalizedkey(key_or_ident):
            raise InvalidGeneralizedKeyError(
                "[GTC001] key_or_ident is not a valid `GeneralizedKey` or `TrieId`")

        current_node = self
        for token in key_or_ident:
            if token not in current_node.children:
                return False
            current_node = current_node.children[token]

        return current_node.ident is not None

    def __len__(self) -> int:
        """Returns the number of keys in the trie.

        Returns:
            :class:`int`: Number of keys in the trie.

        Usage::

            n_keys: int = len(trie)

        """
        return len(self._trie_index)

    def __str__(self) -> str:
        """Generates a stringified version of the trie for visual examination.

        The output IS NOT executable code but more in the nature of debug and testing support."""
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
        """Converts the trie to a dictionary representation.

        This is used for tests and debugging purposes and is not intended
        for general purpose serialization of the trie. It's output is not
        suitable for use with :func:`json.dumps()` or similar functions and
        is subject to change without notice. This is NOT a public API - it is
        for internal use by package tests only. It is intended to provide a
        snapshot of the current state of the trie for tests and debugging purposes.

        Returns:
            :class:`dict[str, Any]`: Dictionary representation of the trie.
            The dictionary contains the following keys:
                - "ident": The unique identifier of the trie.
                - "children": A dictionary of child nodes, where the keys are the tokens
                  of the child nodes and the values are dictionaries representing the child nodes.
                - "trie_index": A dictionary mapping TrieId to _Node objects.
                - "trie_entries": A dictionary mapping TrieId to TrieEntry objects.
        """
        # pylint: disable=protected-access
        # Using deepcopy to ensure that the returned dictionary is a copy of the data in the trie,
        # not a dictionary of live references to it
        return deepcopy({
            "ident": self.ident,
            "children": {
                k: v._as_dict()  # type: ignore[protected-access]
                for k, v in self.children.items()
            },  # type: ignore[protected-access]
            "trie_index": sorted(self._trie_index.keys()),
            "trie_entries": {
                k: repr(v)
                for k, v in self._trie_entries.items()
            }
        })

    def __iter__(self) -> Generator[TrieId, None, None]:
        """Returns an iterator for the trie.

        The generator yields the :class:`TrieId`for each key in the trie.

        Returns:
            :class:`Generator[TrieId, None, None]`: Generator for the trie.
        """
        return (entry for entry in self._trie_entries.keys())  # pylint: disable=consider-iterating-dictionary

    def __setitem__(self, key: GeneralizedKey, value: Any) -> None:
        """Adds or updates an entry in the trie using subscript notation.

        This is a convenience wrapper for the :meth:`update` method.

        Example:
            trie[['ape', 'green', 'apple']] = "A green apple"

        Args:
            key (GeneralizedKey): The key for the entry.
            value (Any): The value to associate with the key.

        Raises:
            InvalidGeneralizedKeyError: If key is not a valid :class:`GeneralizedKey`.
        """
        self.update(key, value)

    def __delitem__(self, key: TrieId | GeneralizedKey) -> None:
        """Removes an entry from the trie using subscript notation.

        This is a convenience wrapper for the :meth:`remove` method.

        Example:
            del trie[['ape', 'green', 'apple']]

        Args:
            key (TrieId | GeneralizedKey): The identifier for the entry to remove.

        Raises:
            KeyError: if the key does not exist in the trie.
            TypeError: if the key is not a :class:`TrieId` or a valid :class:`GeneralizedKey`.
        """
        self.remove(key)

    def __getitem__(self, key: TrieId | GeneralizedKey) -> TrieEntry:
        """Returns the :class:`TrieEntry` for the ident or key with the passed identifier.

        The identifier can be either the :class:`TrieId` (ident) or the :class:`GeneralizedKey` (key)
        for the entry.

        Example:
            trie = GeneralizedTrie()
            ident: TrieId = trie.add(['ape', 'green', 'apple'])

            entry: TrieEntry = trie[ident]
            print(entry)
            # Output: TrieEntry(ident=1, key=['ape', 'green', 'apple'], value=None)

            entry: TrieEntry = trie[['ape', 'green', 'apple']]
            print(entry)
            # Output: TrieEntry(ident=1, key=['ape', 'green', 'apple'], value=None

        Args:
            key (TrieId | GeneralizedKey): the identifier to retrieve.

        Returns: :class:`TrieEntry`: TrieEntry for the key with the passed identifier.

        Raises:
            KeyError ([GTGI001]): if the key arg does not match any keys/idents in the trie.
            TypeError ([GTGI002]): if the key arg is neither a :class:`TrieId` or a valid :class:`GeneralizedKey`.
        """
        if isinstance(key, TrieId):
            if key not in self._trie_index:
                raise KeyError(
                    "[GTGI001] key does not match any idents or keys in the trie"
                )
            # Return the TrieEntry for the TrieId
            return self._trie_entries[key]

        if is_generalizedkey(key):
            # Find the TrieId for the key
            current_node = self
            for token in key:
                if token not in current_node.children:
                    raise KeyError(
                        "[GTGI001] key does not match any idents or keys in the trie"
                    )
                current_node = current_node.children[token]
            if current_node.ident:
                # Return the TrieEntry for the TrieId
                return self._trie_entries[current_node.ident]
            raise KeyError(
                "[GTGI001] key does not match any idents or keys in the trie")

        # If we reach here, the passed key was neither a TrieId nor a GeneralizedKey
        raise TypeError(
            "[GTGI002] key must be either a :class:TrieId or a :class:`GeneralizedKey`"
        )

    def get(self,
            key: TrieId | GeneralizedKey,
            default: Optional[Any] = None) -> Optional[TrieEntry | Any]:
        """Returns the :class:`TrieEntry` for the ident or key with the passed identifier.

        The identifier can be either the :class:`TrieId` (ident) or the :class:`GeneralizedKey` (key)
        for the entry.

        If the key is not found, it returns the default value if provided or None if not provided.

        Example:
            trie = GeneralizedTrie()
            ident: TrieId = trie.add(['ape', 'green', 'apple'])

            entry: TrieEntry = trie.get(ident)
            print(entry)
            # Output: TrieEntry(ident=1, key=['ape', 'green', 'apple'], value=None)

            trie[['ape', 'green', 'apple']] = "A green apple"
            entry = trie.get(['ape', 'green', 'apple'])
            print(entry)
            # Output: TrieEntry(ident=1, key=['ape', 'green', 'apple'], value="A green apple")

            entry = trie.get(['non', 'existent', 'key'], default="Not Found")
            print(entry)
            # Output: "Not Found"

        Args:
            key (TrieId | GeneralizedKey): the identifier to retrieve.
            default (Optional[TrieEntry | Any], default=None): The default value to return if the key is not found.

        Returns: :class:`TrieEntry`: TrieEntry for the key with the passed identifier or the default value if not found.

        Raises:
            TypeError ([GTG002]): if the key arg is neither a :class:`TrieId` or a valid :class:`GeneralizedKey`.
        """
        try:
            return self[key]
        except KeyError:
            return default
        except TypeError as exc:
            # Re-raise the TypeError if the key type is invalid, as this is a usage error, not a "not found" case.
            if "[GTGI002]" in str(exc):
                raise TypeError(
                    "[GTG002] key must be either a :class:TrieId or a :class:`GeneralizedKey`"
                ) from exc
            # For other TypeErrors that might arise from key validation, treat as not found.
            return default

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
