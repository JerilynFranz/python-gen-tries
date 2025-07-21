"""Package providing a generalized trie implementation.

This package includes classes and functions to create and manipulate a generalized trie
data structure. Unlike common trie implementations that only support strings as keys,
this generalized trie can handle various types of tokens, as long as they are hashable.

.. warning:: gentrie IS NOT thread-safe. It is not designed for concurrent use.
It is a single-threaded data structure. If you need a thread-safe trie, you must
use an external locking mechanism to ensure that either only read-only threads are
accessing the trie at the same time or that only one write thread and no read threads
are accessing the trie at the same time.

Usage:

    Example 1::

        from gentrie import GeneralizedTrie, TrieEntry

        trie = GeneralizedTrie()
        trie.add(['ape', 'green', 'apple'])
        trie.add(['ape', 'green'])
        matches: set[TrieEntry] = trie.prefixes(['ape', 'green'])
        print(matches)

    Example 1 Output::

        {TrieEntry(ident=2, key=['ape', 'green'])}

    Example 2::

        from gentrie import GeneralizedTrie, TrieEntry

        # Create a trie to store website URLs
        url_trie = GeneralizedTrie()

        # Add some URLs with different components (protocol, domain, path)
        url_trie.add(["https", "com", "example", "www", "/", "products", "clothing"])
        url_trie.add(["http", "org", "example", "blog", "/", "2023", "10", "best-laptops"])
        url_trie.add(["ftp", "net", "example", "ftp", "/", "data", "images"])

        # Find all https URLs with "example.com" domain
        suffixes: set[TrieEntry] = url_trie.suffixes(["https", "com", "example"])
        print(suffixes)

    Example 2 Output::

        {TrieEntry(ident=1, key=['https', 'com', 'example', 'www', '/', 'products', 'clothing'])}}

    Example 3::

        from gentrie import GeneralizedTrie, TrieEntry

        trie = GeneralizedTrie()
        trie.add('abcdef')
        trie.add('abc')
        trie.add('qrf')
        matches: set[TrieEntry] = trie.suffixes('ab')
        print(matches)

    Example 3 Output::

        {TrieEntry(ident=2, key='abc'), TrieEntry(ident=1, key='abcdef')}

"""
# pylint: disable=protected-access

from collections.abc import Sequence
from textwrap import indent
from typing import runtime_checkable, Generator, Optional, Protocol, NamedTuple, TypeAlias
from warnings import deprecated


class InvalidTrieKeyTokenError(TypeError):
    """Raised when a token in a key is not a valid :class:`TrieKeyToken` object.

    This is a sub-class of :class:`TypeError`."""


class InvalidGeneralizedKeyError(TypeError):
    """Raised when a key is not a valid :class:`GeneralizedKey` object.

    This is a sub-class of :class:`TypeError`."""

@runtime_checkable
class TrieKeyToken(Protocol):
    """:class:`TrieKeyToken` is a protocol that defines key tokens that are usable with a :class:`GeneralizedTrie`.

    The protocol requires that a token object be *hashable*. This means that it
    implements both an ``__eq__()`` method and a ``__hash__()`` method.

    Some examples of built-in types suitable for use as tokens in a key:

        :class:`str`
        :class:`bytes`
        :class:`int`
        :class:`float`
        :class:`complex`
        :class:`frozenset`
        :class:`tuple`
        :class:`None`

    Note: frozensets and tuples are only hashable *if their contents are hashable*.

   .. warning:: User-defined classes are hashable by default, but you should implement the ``__eq__()`` and ``__hash__()``
    dunder methods in a content-aware way (the hash and eq values must depend on the
    content of the object) if you want to use them as tokens in a key. The default implementation
    of ``__eq__()`` and ``__hash__()`` uses the memory address of the object, which means that two
    different instances of the same class will not be considered equal.
    
    Usage::

        from gentrie import TrieKeyToken
        if isinstance(token, TrieKeyToken):
            print("token supports the TrieKeyToken protocol")
        else:
            print("token does not support the TrieKeyToken protocol")

    """
    def __eq__(self, value: object, /) -> bool: ...
    def __hash__(self) -> int: ...


@deprecated("The Hashable protocol is deprecated and will be removed in a future version. Use :class:`TrieKeyToken` instead.")
@runtime_checkable
class Hashable(TrieKeyToken, Protocol):
    """The Hashable protocol is deprecated and will be removed in a future version.
    
    Use :class:`TrieKeyToken` instead."""


GeneralizedKey: TypeAlias = Sequence[TrieKeyToken | str]
"""A :class:`GeneralizedKey` is an object of any class that is a :class:`Sequence` and
that when iterated returns tokens conforming to the :class:`TrieKeyToken` protocol.

Examples:

    * :class:`str`
    * :class:`bytes`
    * :class:`list[bool]`
    * :class:`list[int]`
    * :class:`list[bytes]`
    * :class:`list[str]`
    * :class:`list[Optional[str]]`
    * :class:`tuple[int, int, str]`

"""


TrieId: TypeAlias = int
"""Unique identifier for a key in a trie."""


class TrieEntry(NamedTuple):
    """A :class:`TrieEntry` is a :class:`NamedTuple` containing the unique identifer and key for an entry in the trie.
    """
    ident: TrieId
    """:class:`TrieId` Unique identifier for a key in the trie. Alias for field number 0."""
    key: GeneralizedKey
    """:class:`GeneralizedKey` Key for an entry in the trie. Alias for field number 1."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrieEntry):
            return False
        return self.ident == other.ident and tuple(self.key) == tuple(other.key)

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
    return isinstance(token, TrieKeyToken)  # type: ignore[reportUnnecessaryIsInstance]


@deprecated("The is_hashable function is deprecated and will be removed in a future version. Use :func:`is_triekeytoken` instead.")
def is_hashable(token: TrieKeyToken) -> bool:
    """is_hashable is deprecated and will be removed in a future version.
    
    Use :func:`is_triekeytoken` instead."""
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
        isinstance(key, Sequence) and  # type: ignore[reportUnnecessaryIsInstance]
        len(key) and
        all(isinstance(t, TrieKeyToken) for t in key))  # type: ignore[reportGeneralTypeIssues]


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
    def __init__(self, token: TrieKeyToken, parent: 'GeneralizedTrie | _Node') -> None:
        self.ident: Optional[TrieId] = None
        self.token: TrieKeyToken = token
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
                output.append(
                    f"    {repr(child_key)} = " + indent(str(child_value), "    ").lstrip()
                )
            output.append("  }")
        output.append("}")
        return "\n".join(output)


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

        Objects of user-defined classes are :class:`TrieKeyToken` by default, but **this
        will not work as naively expected.** The hash value of an object is based on its
        memory address by default. This results in the hash value of an object changing
        every time the object is created and means that the object will not be found in
        the trie unless you have a reference to the original object.

        If you want to use a user-defined class as a token in a key to look up by value
        instead of the instance, you must implement the ``__eq__()`` and ``__hash__()``
        dunder methods in a content aware way (the hash and eq values must depend on the
        content of the object).

        One way to do this is to use the :class:`dataclasses.dataclass` decorator
        to create a class that is content aware. This will automatically implement the
        ``__eq__()`` and ``__hash__()`` dunder methods for you

    """
    def __init__(self) -> None:
        self.token: Optional[TrieKeyToken] = None
        self.parent: Optional[GeneralizedTrie | _Node] = None
        self.children: dict[TrieKeyToken, _Node] = {}
        self.ident: TrieId = 0
        self._ident_counter: TrieId = 0
        self._trie_index: dict[TrieId, _Node] = {}
        self._trie_entries: dict[TrieId, TrieEntry] = {}

    def add(self, key: GeneralizedKey) -> TrieId:
        """Adds the key to the trie.

        Args:
            key (GeneralizedKey): Must be an object that can be iterated and that when iterated
                returns elements conforming to the :class:`TrieKeyToken` protocol.

        Raises:
            InvalidGeneralizedKeyError ([GTA001]):
                If key is not a valid :class:`GeneralizedKey`.

        Returns:
            :class:`TrieId`: Id of the inserted key. If the key was already in the trie,
            it returns the id for the already existing entry.
        """
        if not is_generalizedkey(key):
            raise InvalidGeneralizedKeyError("[GTA001] key is not a valid `GeneralizedKey`")

        # Traverse the trie to find the insertion point for the key
        current_node = self
        for token in key:
            if token not in current_node.children:
                child_node = _Node(token=token, parent=current_node)
                current_node.children[token] = child_node
            current_node = current_node.children[token]

        # If the node already has a trie id, return it
        if current_node.ident:
            return current_node.ident

        # Assign a new trie id for the node
        self._ident_counter += 1
        current_node.ident = self._ident_counter
        self._trie_index[self._ident_counter] = current_node  # type: ignore[assignment]
        self._trie_entries[self._ident_counter] = TrieEntry(self._ident_counter, key)
        return self._ident_counter

    def remove(self, ident: TrieId) -> None:
        """Remove the key with the passed ident from the trie.

        Args:
            ident (TrieId): id of the key to remove.

        Raises:
            TypeError ([GTR001]): if the ident arg is not a :class:`TrieId`.
            ValueError ([GTR002]): if the ident arg is not a legal value.
            KeyError ([GTR003]): if the ident does not match the id of any keys.
        """
        if not isinstance(ident, TrieId):  # type: ignore[reportUnnecessaryIsInstance]
            raise TypeError("[GTR001] ident arg must be of type TrieId")
        if ident < 1:
            raise KeyError("[GTR002] ident is not valid")

        # Not a known trie id
        if ident not in self._trie_index:
            raise KeyError("[GTR003] ident arg does not match any key ids")

        # Find the node and delete its id from the trie index
        node: GeneralizedTrie | _Node = self._trie_index[ident]
        del self._trie_index[ident]
        del self._trie_entries[ident]

        # Remove the id from the node
        node.ident = 0

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

    def prefixes(self, key: GeneralizedKey) -> set[TrieEntry]:
        """Returns a set of TrieEntry instances for all keys in the trie that are a prefix of the passed key.

        Searches the trie for all keys that are prefix matches
        for the key and returns their TrieEntry instances as a set.

        Args:
            key (GeneralizedKey): Key for matching.

        Returns:
            :class:`set[TrieEntry]`: :class:`set` containing TrieEntry instances for keys that are prefixes of the key.
            This will be an empty set if there are no matches.

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
            matches: set[TrieEntry] = trie.prefixes('abcd')
            for trie_entry in sorted(list(matches)):
                print(f'{trie_entry.ident}: {trie_entry.key}')

            # 2: abc
            # 3: a
            # 4: abcd

        """
        if not is_generalizedkey(key):
            raise InvalidGeneralizedKeyError("[GTM001] key is not a valid `GeneralizedKey`")

        matched: set[TrieEntry] = set()
        current_node = self

        for token in key:
            if current_node.ident:
                matched.add(self._trie_entries[current_node.ident])
            if token not in current_node.children:
                break
            current_node = current_node.children[token]

        if current_node.ident:
            matched.add(self._trie_entries[current_node.ident])

        return matched

    def suffixes(self, key: GeneralizedKey, depth: int = -1) -> set[TrieEntry]:
        """Returns the ids of all suffixes of the trie_key up to depth.

        Searches the trie for all keys that are suffix matches for the key up
        to the specified depth below the key match and returns their ids as a set.

        Args:
            key (GeneralizedKey): Key for matching.
            depth (`int`, default=-1): Depth starting from the matched key to include.
                The depth determines how many 'layers' deeper into the trie to look for suffixes.:
                * A depth of -1 (the default) includes ALL entries for the exact match and all children nodes.
                * A depth of 0 only includes the entries for the *exact* match for the key.
                * A depth of 1 includes entries for the exact match and the next layer down.
                * A depth of 2 includes entries for the exact match and the next two layers down.

        Returns:
            :class:`set[TrieId]`: Set of TrieEntry instances for keys that are suffix matches for the key.
            This will be an empty set if there are no matches.

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
            matches: set[TrieEntry] = trie.suffixes('abcd')

            for trie_entry in sorted(list(matches)):
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
        for token in key:
            if token not in current_node.children:
                return set()  # no match
            current_node = current_node.children[token]

        # Perform a breadth-first search to collect suffixes up to the specified depth
        queue = [(current_node, depth)]
        matches: set[TrieEntry] = set()

        while queue:
            node, current_depth = queue.pop(0)
            if node.ident:
                matches.add(self._trie_entries[node.ident])
            if current_depth != 0:
                for child in node.children.values():
                    queue.append((child, current_depth - 1))

        return matches

    def clear(self) -> None:
        """Clears all keys from the trie.

        Usage::

            trie_obj.clear()

        """
        all_ids = list(self._trie_index.keys())
        for ident in all_ids:
            self.remove(ident)
        self.ident = 0
        self.token = None
        self.parent = None
        self.children = {}
        self._trie_index = {}
        self._trie_entries = {}
        self._ident_counter = 0

    def __contains__(self, key: GeneralizedKey) -> bool:
        """Returns True if the trie contains a GeneralizedKey matching the passed key.

        Args:
            key (GeneralizedKey): Key for matching.
                key for matching.

        Returns:
            :class:`bool`: True if there is a matching GeneralizedKey in the trie. False otherwise.

        Raises:
            :class:`TypeError`:
                If key arg is not a GeneralizedKey.

        Usage::

            trie = GeneralizedTrie()
            keys: list[str] = ['abcdef', 'abc', 'a', 'abcd', 'qrs']
            for entry in keys:
                trie.add(entry)

            if 'abc' in trie:
                print('"abc" is in the trie')

        """
        return bool(self.suffixes(key, 0))

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
                output.append(
                    f"    {repr(child_key)} = " + indent(str(child_value), "    ").lstrip()
                )
            output.append("  }")
        output.append(f"  trie index = {self._trie_index.keys()}")
        output.append("}")
        return "\n".join(output)

    def __iter__(self) -> Generator[TrieId, None, None]:
        """Returns an iterator for the trie.

        The generator yields the :class:`TrieId`for each key in the trie.

        Returns:
            :class:`Generator[TrieId, None, None]`: Generator for the trie.
        """
        return (entry for entry in self._trie_entries.keys())  # pylint: disable=consider-iterating-dictionary

    def __getitem__(self, ident: TrieId) -> TrieEntry:
        """Returns the TrieEntry for the key with the passed ident.

        Args:
            ident (TrieId): Id of the key to retrieve.

        Returns: :class:`TrieEntry`: TrieEntry for the key with the passed ident.
        """
        return self._trie_entries[ident]

    def keys(self) -> Generator[TrieId, None, None]:
        """Returns an iterator for all the TrieId keys in the trie.

        The generator yields the :class:`TrieId` for each key in the trie.

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

        The generator yields the :class:`TrieId` and :class:`TrieEntry` for each key in the trie.

        Returns:
            :class:`Generator[tuple[TrieId, TrieEntry], None, None]`: Generator for the trie.
        """
        return ((key, value) for key, value in self._trie_entries.items())
