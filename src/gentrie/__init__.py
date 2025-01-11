"""Package providing a generalized trie implementation.

This package includes classes and functions to create and manipulate a generalized trie
data structure. Unlike common trie implementations that only support strings as keys,
this generalized trie can handle various types of tokens, as long as they are hashable.

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


class InvalidHashableError(TypeError):
    """Raised when a token in a key is not a valid :class:`Hashable` object.

    This is a sub-class of :class:`TypeError`."""


class InvalidGeneralizedKeyError(TypeError):
    """Raised when a key is not a valid :class:`GeneralizedKey` object.

    This is a sub-class of :class:`TypeError`."""


@runtime_checkable
class Hashable(Protocol):
    """:class:`Hashable` is a protocol that defines key tokens that are usable with a :class:`GeneralizedTrie`.

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

    User-defined classes are hashable by default.

    Usage::

        from gentrie import Hashable
        if isinstance(token, Hashable):
            print("token supports the Hashable protocol")
        else:
            print("token does not support the Hashable protocol")

    """
    def __eq__(self, value: object, /) -> bool: ...
    def __hash__(self) -> int: ...


GeneralizedKey: TypeAlias = Sequence[Hashable | str]
"""A :class:`GeneralizedKey` is an object of any class that is a :class:`Sequence` and
that when iterated returns tokens conforming to the :class:`Hashable` protocol.

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


def is_hashable(token: Hashable) -> bool:
    """Tests token for whether it is a valid :class:`Hashable`.

    A valid :class:`Hashable` is a hashable object.

    Examples:
    :class:`bool`, :class:`bytes`, :class:`float`, :class:`frozenset`,
    :class:`int`, :class:`str`, :class:`None`, :class:`tuple`.

    Args:
        token (GeneralizedKey): Object for testing.

    Returns:
        :class:`bool`: ``True`` if a valid :class:`Hashable`, ``False`` otherwise.
    """
    return isinstance(token, Hashable)  # type: ignore[reportUnnecessaryIsInstance]]


def is_generalizedkey(key: GeneralizedKey) -> bool:
    """Tests key for whether it is a valid `GeneralizedKey`.

    A valid :class:`GeneralizedKey` is a :class:`Sequence` that returns
    :class:`Hashable` protocol conformant objects when
    iterated. It must have at least one token.

    Parameters:
        key (GeneralizedKey): Key for testing.

    Returns:
        :class:`bool`: ``True`` if a valid :class:`GeneralizedKey`, ``False`` otherwise.
    """
    return (
        isinstance(key, Sequence) and  # type: ignore[reportUnnecessaryIsInstance]
        len(key) and
        all(isinstance(t, Hashable) for t in key))  # type: ignore[reportGeneralTypeIssues]


class GeneralizedTrie:  # pylint: disable=too-many-instance-attributes
    """A general purpose trie.

    Unlike many trie implementations which only support strings as keys
    and token match only at the character level, it is agnostic as to the
    types of tokens used to key it and thus far more general purpose.

    It requires only that the indexed tokens be hashable. This is verified
    at runtime using the :class:`gentrie.Hashable` protocol.

    Tokens in a key do NOT have to all be the same type as long as they
    can be compared for equality.

    It can handle a :class:`Sequence` of :class:`Hashable` conforming objects as keys
    for the trie out of the box.

    You can 'mix and match' types of objects used as token in a key as
    long as they all conform to the :class:`Hashable` protocol.

    The code emphasizes robustness and correctness.

   .. warning:: **GOTCHA: Using User Defined Classes As Tokens In Keys**

        Objects of user-defined classes are :class:`Hashable` by default, but **this
        will not work as naively expected.** The hash value of an object is based on its
        memory address by default. This results in the hash value of an object changing
        every time the object is created and means that the object will not be found in
        the trie unless you have a reference to the original object.

        If you want to use a user-defined class as a token in a key to look up by value
        instead of the instance, you must implement the ``__eq__()`` and ``__hash__()``
        dunder methods in a content aware way (the hash and eq values must depend on the
        content of the object).

    """
    def __init__(self) -> None:
        self._root_node: bool = True
        self._node_token: Optional[Hashable] = None
        self._parent: Optional[GeneralizedTrie] = None
        self._children: dict[Hashable, GeneralizedTrie] = {}
        self._trie_index: dict[TrieId, GeneralizedTrie] = {}
        self._trie_entries: dict[TrieId, TrieEntry] = {}
        self._trie_id: TrieId = 0
        self._trie_id_counter: dict[str, TrieId] = {"trie_number": 0}

    def add(self, key: GeneralizedKey) -> TrieId:
        """Adds the key to the trie.

        Args:
            key (GeneralizedKey): Must be an object that can be iterated and that when iterated
                returns elements conforming to the :class:`Hashable` protocol.

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
            if token not in current_node._children:
                child_node = GeneralizedTrie()  # type: ignore[reportArgumentType]
                child_node._root_node = False
                child_node._node_token = token  # type: ignore[reportAttributeAccess]
                child_node._parent = current_node
                current_node._children[token] = child_node  # type: ignore[reportArgumentType]
            current_node = current_node._children[token]  # type: ignore[reportArgumentType]

        # If the node already has a trie id, return it
        if current_node._trie_id:
            return current_node._trie_id

        # Assign a new trie id for the node
        new_trie_id: TrieId = self._trie_id_counter["trie_number"] + 1
        current_node._trie_id = new_trie_id
        self._trie_id_counter["trie_number"] = new_trie_id
        self._trie_index[new_trie_id] = current_node
        self._trie_entries[new_trie_id] = TrieEntry(new_trie_id, key)
        return new_trie_id

    def remove(self, trie_id: TrieId) -> None:
        """Remove the key with the passed trie_id from the trie.

        Args:
            trie_id (TrieId): id of the key to remove.

        Raises:
            TypeError ([GTR001]): if the trie_id arg is not a :class:`TrieId`.
            ValueError ([GTR002]): if the trie_id arg is not a legal value.
            KeyError ([GTR003]): if the trie_id does not match the id of any keys.
        """
        # pylint: disable=protected-access
        if not isinstance(trie_id, TrieId):  # type: ignore
            raise TypeError("[GTR001] trie_id arg must be of type TrieId")
        if trie_id < 1:
            raise KeyError("[GTR002] trie_id is not valid")

        # Not a known trie id
        if trie_id not in self._trie_index:
            raise KeyError("[GTR003] trie_id arg does not match any key ids")

        # Find the node and delete its id from the trie index
        node: GeneralizedTrie = self._trie_index[trie_id]
        del self._trie_index[trie_id]
        del self._trie_entries[trie_id]

        # Remove the id from the node
        node._trie_id = 0

        # If the node still has other trie ids or children, we're done: return
        if node._children:
            return

        # No trie ids or children are left for this node, so prune
        # nodes up the trie tree as needed.
        node_token: Optional[Hashable] = node._node_token
        parent_node = node._parent
        while parent_node is not None:
            del parent_node._children[node_token]  # type: ignore[arg-type, reportArgumentType]
            # explicitly break possible cyclic references
            node._parent = node._node_token = None

            # If the parent node has a trie id or children, we're done: return
            if parent_node._trie_id or parent_node._children:
                return
            # Keep purging nodes up the tree
            node_token = parent_node._node_token
            node = parent_node
            parent_node = node._parent
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
                (is not a :class:`Sequence` of :class:`Hashable` objects).

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
            if current_node._trie_id:
                matched.add(self._trie_entries[current_node._trie_id])
            if token not in current_node._children:
                break
            current_node = current_node._children[token]

        if current_node._trie_id:
            matched.add(self._trie_entries[current_node._trie_id])

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
                If a token in the key arg does not conform to the :class:`Hashable` protocol.

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

        if not isinstance(depth, int):  # type: ignore
            raise TypeError("[GTS002] depth must be an int")
        if depth < -1:
            raise ValueError("[GTS003] depth cannot be less than -1")

        current_node = self
        for token in key:
            if token not in current_node._children:
                return set()  # no match
            current_node = current_node._children[token]  # type: ignore[reportGeneralTypeIssues]

        # Perform a breadth-first search to collect suffixes up to the specified depth
        queue = [(current_node, depth)]
        matches: set[TrieEntry] = set()

        while queue:
            node, current_depth = queue.pop(0)
            if node._trie_id:
                matches.add(self._trie_entries[node._trie_id])
            if current_depth != 0:
                for child in node._children.values():
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
        self._root_node = True
        self._node_token = None
        self._parent = None
        self._children = {}
        self._trie_index = {}
        self._trie_entries = {}
        self._trie_id = 0
        self._trie_id_counter = {"trie_number": 0}

    def __contains__(self, key: TrieId) -> bool:
        """Returns True if the trie contains a TrieId matching the passed key.

        Args:
            key (TrieId):
                Id key for matching.

        Returns:
            :class:`bool`: True if there is a matching TrieId in the trie. False otherwise.

        Raises:
            :class:`TypeError`:
                If key arg is not a TrieId.

        Usage::

            trie = GeneralizedTrie()
            keys: list[str] = ['abcdef', 'abc', 'a', 'abcd', 'qrs']
            for entry in keys:
                trie.add(entry)

            if 1 in trie:
                print('id 1 is in the trie')

        """
        return bool(key in self._trie_index)

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
        if self._root_node:
            output.append(f"  trie number = {self._trie_id_counter['trie_number']}")
        else:
            if self._parent is None:
                output.append("  parent = None")
            elif self._parent._root_node:
                output.append("  parent = root node")
            else:
                output.append(f"  parent = {repr(self._parent._node_token)}")
        output.append(f"  node token = {repr(self._node_token)}")
        if self._trie_id:
            output.append(f"  trie id = {self._trie_id}")
        if self._children:
            output.append("  children = {")
            for child_key, child_value in self._children.items():
                output.append(
                    f"    {repr(child_key)} = " + indent(str(child_value), "    ").lstrip()
                )
            output.append("  }")
        if self._root_node:
            output.append(f"  trie index = {self._trie_index.keys()}")
        output.append("}")
        return "\n".join(output)

    def __del__(self) -> None:
        """Deletes the trie object."""
        self.clear()
        del self

    def __iter__(self) -> Generator[TrieId, None, None]:
        """Returns an iterator for the trie.

        The generator yields the :class:`TrieId`for each key in the trie.

        Returns:
            :class:`Generator[TrieId, None, None]`: Generator for the trie.
        """
        return (entry for entry in self._trie_entries.keys())  # pylint: disable=consider-iterating-dictionary

    def __getitem__(self, trie_id: TrieId) -> TrieEntry:
        """Returns the TrieEntry for the key with the passed trie_id.

        Args:
            trie_id (TrieId): Id of the key to retrieve.

        Returns: :class:`TrieEntry`: TrieEntry for the key with the passed trie_id.
        """
        return self._trie_entries[trie_id]

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
