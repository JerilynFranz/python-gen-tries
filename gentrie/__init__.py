"""Package providing a generalized trie implementation."""

from collections.abc import Sequence, Iterator
from textwrap import indent
from typing import Any, runtime_checkable, Optional, Protocol, NamedTuple, TypeAlias


class InvalidGeneralizedTokenError(TypeError):
    """Raised when a token in a key is not a valid `GeneralizedToken`.

    This is a sub-class of `TypeError`."""


class InvalidGeneralizedKeyError(TypeError):
    """Raised when a key is not a valid `GeneralizedKey`.

    This is a sub-class of `TypeError`."""


@runtime_checkable
class GeneralizedToken(Protocol):
    """GeneralizedToken is a protocal that defines key tokens that are usable with a `GeneralizedTrie`.

    The protocol requires that a token object implements both an __eq__()
    method and a __hash__() method. This generally means that only immutable types
    are suitable for use as tokens.

    Some examples of types suitable for use as tokens in a key:
        str  bytes  int  float  complex  frozenset  tuple

    Usage:
        from gentrie import GeneralizedToken
        if isinstance(token, GeneralizedToken):
            print("token supports the GeneralizedToken protocol")
        else:
            print("token does not support the GeneralizedToken protocol")
    """
    def __eq__(self, value: Any) -> bool: ...
    def __hash__(self) -> int: ...


GeneralizedKey: TypeAlias = Sequence[GeneralizedToken | str]
"""A GeneralizedKey is an object of any class that is a `Sequence` and
that when iterated returns tokens conforming to the `GeneneralizedToken` protocol.

Examples:
    `str`
    `bytes`
    `list[bool]`
    `list[int]`
    `list[bytes]`
    `list[str]`
    `tuple[int, int, str]`

"""


TrieId = int
"""Unique identifier for a key in a trie"""


class TrieEntry(NamedTuple):
    """A TrieEntry is a tuple containing the unique identifer and key for an entry in the trie."""
    ident: TrieId
    """Unique identifier for a key in the trie"""
    key: GeneralizedKey
    """Key for an entry in the trie"""


def is_generalizedtoken(token: GeneralizedToken) -> bool:
    """Tests token for whether it is a valid `GeneralizedToken`.

    A valid GeneralizedToken is a hashable object that
    can have its value compared for equality.

    Examples: `bool`, `bytes`, `float`, `frozenset`, `int`, `str`

    Args:
        token (GeneralizedKey): Object for testing.

    Returns:
        bool: True if a valid GeneralizedToken, False otherwise.
    """
    return isinstance(token, GeneralizedToken)  # type: ignore[reportUnnecessaryIsInstance]]


def is_generalizedkey(key: GeneralizedKey) -> bool:
    """Tests key for whether it is a valid `GeneralizedKey`.

    A valid `GeneralizedKey` is a `Sequence` that returns
    `GeneralizedToken` protocol conformant objects when
    iterated. It must have at least one token.

    Args;
        key (GeneralizedKey): Key for testing.

    Returns:
        bool: True if a valid GeneralizedKey, False otherwise.
    """
    if not isinstance(key, Sequence):  # type: ignore[reportUnnecessaryIsInstance]
        return False
    if len(key) == 0:
        return False
    for token in key:
        if not isinstance(token, GeneralizedToken):  # type: ignore[reportGeneralTypeIssues]
            return False
    return True


class GeneralizedTrie:  # pylint: disable=too-many-instance-attributes
    """A general purpose trie.

    Unlike many Trie implementations which only support strings as keys
    and token match only at the character level, it is agnostic as to the
    types of tokens used to key it and thus far more general purpose.

    It requires only that the indexed tokens be comparable for equality
    and hashable. This is verified at runtime using the
    `gentrie.GeneralizedToken` protocol.

    This generally means that immutable values and bytes can be used as tokens
    in a key. i.e: a frozenset() works as a token, but a set() does not.

    Tokens in a key do NOT have to all be the same type as long as they
    can be compared for equality.

    You can make new classes of objects able to be used with the
    `GeneralizedTrie` just by defining __hash__ and __eq__ dunder methods on them.

    You should generally **ONLY** do this for immutable objects (objects that cannot
    be changed once created).

    It can handle `Sequence`s of `GeneralizedToken` conforming objects as keys
    for the trie out of the box.

    As long as the tokens returned by a sequence are comparable and
    hashable, it largely 'just works'.

    You can 'mix and match' types of objects used as token in a key as
    long as they all conform to the `GeneralizedToken` protocol.

    The code emphasizes robustness and correctness.

    Usage:

    Example 1:
        ```
        from gentrie import GeneralizedTrie

        trie  = GeneralizedTrie()
        trie_id_1: TrieEntry = trie.add(['ape', 'green', 'apple'])
        trie_id_2: TrieEntry = trie.add(['ape', 'green'])
        matches: list[TrieEntry] = trie.prefixes(['ape', 'green'])
        ```

    Example 2:
        ```
        from gentrie import GeneralizedTrie

        # Create a trie to store website URLs
        url_trie = GeneralizedTrie()

        # Add some URLs with different components (protocol, domain, path)
        url_trie.add(["https", "com", "example", "www", "/", "products", "clothing"])
        url_trie.add(["http", "org", "example", "blog", "/", "2023", "10", "best-laptops"])
        url_trie.add(["ftp", "net", "example", "ftp", "/", "data", "images"])

        # Find all https URLs with "example.com" domain
        prefixes: list[TrieEntry] = url_trie.prefixes(["https", "com", "example"])
        print(f"Found URL prefixes: {prefixes}")  # Output: Found URL prefixes: {1}
        ```
    """

    def __init__(self) -> None:
        self._root_node: bool = True
        self._node_token: Optional[GeneralizedToken] = None
        self._parent: Optional[GeneralizedTrie] = None
        self._children: dict[GeneralizedToken, GeneralizedTrie] = {}
        self._trie_index: dict[TrieId, GeneralizedTrie] = {}
        self._trie_id: TrieId = 0
        self._trie_id_counter: dict[str, TrieId] = {"trie_number": 0}

    def _add_new_child(self, /,
                       node_token: GeneralizedToken,
                       key: Iterator[GeneralizedToken]) -> TrieId:
        """Creates and adds a new GeneralizedTrie node to the node's _children.

        The new node is initialized with the passed arguments. This is used
        recursively by the add() method to actually add the key to the trie.

        Args:
            node_token (GeneralizedToken):
                The node_token for the new child.
            key (Iterator[GeneralizedToken]):
                Remaining tokens (if any) in the key.

        Returns:
            TrieId:
                Assigned id for the key in the trie.

        Raises:
            AssertionError [GTANC001]:
                If node_token does not conform to the GeneralizedToken protocol
                or if the key is not an Iterator.
            InvalidGeneralizedTokenError [GTAFBT003]:
                If a token in the key does not conform to the GeneralizedToken protocol.
        """
        # pylint: disable=protected-access
        # trunk-ignore(bandit/B101)
        assert isinstance(node_token, GeneralizedToken) and isinstance(
            key, Iterator), "[GTANC001] incorrect arguments passed to _add_new_child()"
        new_child = GeneralizedTrie()
        new_child._root_node = False
        new_child._node_token = node_token
        new_child._parent = self
        new_child._trie_index = self._trie_index
        new_child._trie_id_counter = self._trie_id_counter
        trie_id: TrieId = new_child._add_iter(key)
        self._children[node_token] = new_child
        return trie_id

    def _add_iter(self, key: Iterator[GeneralizedToken]) -> TrieId:
        """Adds a key defined by the passed key Iterator to the trie.

        This private method handles the iteration as a performance oriented
        'helper' internal method.

        Args:
            key (Iterator[GeneralizedToken):
                Must be an Iterator that returns elements conforming
                to the `GeneralizedToken` protocol.

        Returns:
            TrieId: Id of the inserted key. If the key was already in the
                    trie, it returns the id of the already existing entry.
        """
        try:
            first_token: GeneralizedToken = next(key)  # type: ignore
        except StopIteration:
            # already existing key
            if self._trie_id:
                return self._trie_id

            # new key
            new_trie_id: TrieId = self._trie_id_counter["trie_number"] + 1
            self._trie_id = new_trie_id
            self._trie_id_counter["trie_number"] = new_trie_id
            self._trie_index[new_trie_id] = self
            return new_trie_id

        # there is an existing child trie we can use
        if first_token in self._children:
            return self._children[first_token]._add_iter(key)  # pylint: disable=protected-access

        # we need a new sub-trie
        return self._add_new_child(node_token=first_token, key=key)

    def add(self, key: GeneralizedKey) -> TrieId:
        """Adds the key to the trie.

        Args:
            key (GeneralizedKey):
                Must be an object that can be iterated and that when iterated
                returns elements conforming to the **GeneralizedToken** protocol.

        Raises:
            InvalidGeneralizedKeyError [GTA001]:
                If key is not a valid `GeneralizedKey`.

        Returns:
            TrieId: id of the inserted key. If the key was already in the
                 trie, it returns the id for the already existing entry.
        """
        if not is_generalizedkey(key):
            raise InvalidGeneralizedKeyError("[GTA001] key is not a valid `GeneralizedKey`")
        return self._add_iter(key=iter(key))  # type: ignore

    def remove(self, trie_id: TrieId) -> None:
        """Remove the key with the passed trie_id from the trie.

        Args:
            trie_id (TrieId): id of the key to remove.

        Raises:
            TypeError:
                trie_id arg is not a `TrieId`.
            ValueError:
                trie_id arg is not a legal value.
            KeyError:
                trie_id does not match the id of any keys.
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
        del node._trie_index[trie_id]

        # Remove the id from the node
        node._trie_id = 0

        # If the node still has other trie ids or children, we're done: return
        if node._children:
            return

        # No trie ids or children are left for this node, so prune
        # nodes up the trie tree as needed.
        node_token: Optional[GeneralizedToken] = node._node_token
        parent_node = node._parent
        while parent_node is not None:
            del parent_node._children[node_token]  # type: ignore[arg-type, reportArgumentType]
            # explicitly break possible cyclic references
            node._parent = node._node_token = None
            node._trie_id_counter = {}
            # If the parent node has a trie id or children, we're done: return
            if parent_node._trie_id or parent_node._children:
                return
            # Keep purging nodes up the tree
            node_token = parent_node._node_token
            node = parent_node
            parent_node = node._parent
        return

    def prefixes(self, key: GeneralizedKey) -> set[TrieId]:
        """Returns the ids of all keys in the trie that are a prefix of the passed key.

        Searches the trie for all keys that are prefix matches
        for the key and returns their ids as a set.

        Usage:
            trie: GeneralizedTrie = GeneralizedTrie()
            keys: list[str] = ['abcdef', 'abc', 'a', 'abcd', 'qrs']
            keys_index: dict[TrieId, str] = {}
            for entry in keys:
                key_index[trie.add(entry)] = entry
            matches: set[TrieId] = trie.prefix_key_ids('abcd')

            # matches now contains the set {2, 3, 4}, corresponding
            # to the keys 'abc', 'a', and 'abcd' - all of which are
            prefix matches for 'abcd'.

            # 2: abcd
            # 3: a
            # 4: abcd
            for trie_id in sorted(list(matches)):
                print(f'{trie_id}: {keys_index[trie_id]}')

        Args:
            key (Any):
                key for matching.

        Returns:
            set[TrieId]:
                Set of ids for keys that are prefixes of
                the key. This will be an empty set if there
                are no matches.

        Raises:
            InvalidGeneralizedKeyError [GTM001]:
                If key is not a valid `GeneralizedKey` (is not a `Sequence` of `GeneralizedToken`).
            InvalidGeneralizedKeyError [GTM002]:
                If a token in the key does not conform to the `GeneralizedToken` protocol.
        """
        if not isinstance(key, Iterator):
            try:
                key = iter(key)  # type: ignore[reportAssignmentType]
            except TypeError as err:
                raise InvalidGeneralizedKeyError(f"[GTM001] key is not a valid `GeneralizedKey`: {err}") from err

        matched: set[TrieId] = set([self._trie_id]) if self._trie_id else set()
        try:
            token = next(key)  # type: ignore[reportAssignmentType]
            if not isinstance(token, GeneralizedToken):  # type: ignore[unnecessaryIsInstance]
                raise InvalidGeneralizedKeyError(
                    "[GTM002] key contains a token that does not conform with the `GeneralizedToken` protocol")
            if token in self._children:
                matched = matched.union(self._children[token].prefixes(key))
        except StopIteration:
            pass

        return matched

    def _suffixes_iter(self, key: Iterator[GeneralizedToken], depth: int) -> set[TrieId]:
        """Returns the ids of all suffixs of the trie_key up to depth.

        Searches the trie for all keys that are suffix matches for
        the key up to the specified depth and returns their ids as a set.

        This is a performance oriented internal 'helper' method for trie.suffixes()

        Args:
            key (`GeneralizedKey`):
                Key for matching.
            depth (`int`, default=-1):
                Depth starting from the matched key to include.

                The depth determines how many 'layers' deeper into the trie to look for ids:
                    * A depth of -1 (the default) includes ALL ids for the exact match and all children nodes.
                    * A depth of 0 only includes the ids for the *exact* match for the key.
                    * A depth of 1 includes ids for the exact match and the next layer down.
                    * A depth of 2 includes ids for the exact match and the next two layers down.

        Returns:
            `set[TrieId]`:
                Set of ids for keys that are suffix matchs for the key.
                This will be an empty set if there are no matches.
        """
        try:
            token: GeneralizedToken = next(key)  # type: ignore
        except StopIteration:  # found the match
            return self._contained_ids(depth=depth)
        if token in self._children:  # looking for match
            return self._children[token]._suffixes_iter(key, depth)
        return set()  # no match

    def suffixes(self, key: GeneralizedKey, depth: int = -1) -> set[TrieId]:
        """Returns the ids of all suffixs of the trie_key up to depth.

        Searches the trie for all keys that are suffix matches for
        the key up to the specified depth and returns their ids as a set.

        Args:
            key (`GeneralizedKey`):
                Key for matching.
            depth (`int`, default=-1):
                Depth starting from the matched key to include.

                The depth determines how many 'layers' deeper into the trie to look for ids:
                    * A depth of -1 (the default) includes ALL ids for the exact match and all children nodes.
                    * A depth of 0 only includes the ids for the *exact* match for the key.
                    * A depth of 1 includes ids for the exact match and the next layer down.
                    * A depth of 2 includes ids for the exact match and the next two layers down.

        Returns:
            `set[TrieId]`:
                Set of ids for keys that are suffix matchs for the key.
                This will be an empty set if there are no matches.

        Raises:
            InvalidGeneralizedKeyError (GTS001):
                If key arg is not a GeneralizedKey.
            TypeError (GTS002):
                If depth arg is not an int.
            ValueError (GTS003):
                If depth arg is less than -1.
            InvalidGeneralizedKeyError (GTS004):
                If a token in the key arg does not conform with the GeneralizedToken protocol.

        Usage:
            ```python
            trie = GeneralizedTrie()
            keys: list[str] = ['abcdef', 'abc', 'a', 'abcd', 'qrs']
            trie_keys_index: dict[TrieId, str] = {}
            for entry in keys:
                trie_key_index[trie.add(entry)] = entry
            matches: set[TrieId] = trie.token_suffixes('abcd')
            ```

            After running the code above, `matches` contains the set {1, 4},
            corresponding to the keys 'abcdef' and 'abcd' - each of which are
            suffix matches to 'abcd'.

            ```python
            for trie_id in sorted(list(matches)):
                print(f'{trie_id}: {trie_keys_index[trie_id]}')
            ```
            Output:
                1: abcdef
                4: abc
        """
        if not is_generalizedkey(key):
            raise InvalidGeneralizedKeyError("[GTS001] key arg is not a valid GeneralizedKey")
        try:
            key = iter(key)  # type: ignore[reportAssignmentType]
        except TypeError as err:
            raise InvalidGeneralizedKeyError(
                f"[GTS001] key arg is not a valid GeneralizedKey: {err}") from err

        if not isinstance(depth, int):  # type: ignore
            raise TypeError("[GTS002] depth must be an int")
        if depth < -1:
            raise ValueError("[GTS003] depth cannot be less than -1")
        try:
            token: GeneralizedToken = next(key)  # type: ignore
        except StopIteration:  # found the match
            return self._contained_ids(depth=depth)

        if token in self._children:  # looking for match
            return self._children[token]._suffixes_iter(key, depth)  # type: ignore
        return set()  # no match

    def _contained_ids_helper(self, ids: set[TrieId], depth: int) -> set[TrieId]:
        """'helper' method returns a set contains all key ids defined for this node and/or its children.

        Args:
            ids (TrieId):
                Set containing the collected TrieIds
            depth (int):
                depth counter to limit the depth of contained ids to include.

                * A negative (-1 or lower) depth includes ALL ids for this node and all children
                    nodes.
                * A depth of 0 includes ONLY the ids for this node.
                * A depth of 1 includes ids for this node and its direct child nodes.
                * A depth of 2 includes ids for this node and the next two layers below it.
                * and so on.

        Returns:
            set[TrieId]:
                Set containing the ids of all contained keys.
        """
        if self._trie_id:
            ids.add(self._trie_id)
        if depth:
            depth -= 1
            for node in self._children.values():
                # pylint: disable=protected-access
                node._contained_ids_helper(ids, depth)
        return ids

    def _contained_ids(self, depth: int = -1) -> set[TrieId]:
        """Returns a set contains all key ids defined for this node and/or its children up to
           the requested depth.

        Args:
            ids (TrieId):
                Set containing the collected TrieIds
            depth (int):
                depth counter to limit the depth of contained ids to include.

                * A negative (-1 or lower) depth includes ALL ids for this node and all children
                    nodes.
                * A depth of 0 includes ONLY the ids for this node.
                * A depth of 1 includes ids for this node and its direct child nodes.
                * A depth of 2 includes ids for this node and the next two layers below it.
                * and so on.

        Returns:
            set[TrieId]:
                Set containing the ids of all contained keys.
        """
        # trunk-ignore(bandit/B101)
        assert isinstance(depth, int), "[GTCI002] depth arg must be an int or sub-class"
        ids: set[TrieId] = set()
        return self._contained_ids_helper(ids=ids, depth=depth)

    def __contains__(self,
                     key: Sequence[GeneralizedToken]) -> set[TrieId]:
        """Returns True if the trie contains a key matching the passed key.

        Usage:
            trie = GeneralizedTrie()
            keys: list[str] = ['abcdef', 'abc', 'a', 'abcd', 'qrs']
            keys_index: dict[TrieId, str] = {}
            for entry in keys:
                key_index[trie.add(entry)] = entry

            if 'abc' in trie:
                print('abc is in the trie')

        Args:
            key (GeneralizedKey):
                key for matching.

        Returns:
            bool:
                (False):
                    Trie does not contain a key matching the passed key.
                (True):
                    Trie contains a key matching the passed key.

        Raises:
            TypeError:
                If key arg is not a Sequence.
            InvalidGeneralizedTokenError:
                If a token in the key arg does not conform with the GeneralizedToken protocol.
        """
        return bool(self.suffixes(key, 0))  # type: ignore

    def __len__(self) -> int:
        """Returns the number of keys in the trie.

        Usage:
            n_keys: int = len(trie)

        Returns:
            (int) number of keys in the trie.
        """
        return len(self._trie_index)

    def __str__(self) -> str:
        """Generates a stringified version of the trie for visual examination.

        The output IS NOT executable code but more in the nature of debug and testing support."""
        output: list[str] = ["{"]
        if self._root_node:
            output.append(f"  trie number = {self._trie_id_counter['trie_number']}")
        elif self._parent:
            if self._parent._root_node:
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
