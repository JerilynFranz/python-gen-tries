"""Module providing a generalized trie implementation."""

from textwrap import indent
from typing import Any, Dict, Iterator, List, Optional, Set

from . import GeneralizedToken


class GeneralizedTrie:  # pylint: disable=too-many-instance-attributes
    """GeneralizedTrie

    A general purpose trie.

    Unlike many Trie implementations, which only support strings as keys
    and token match only at the character level, it is agnostic as to the
    types of tokens used to key it and thus much more general purpose.

    It requires only that the indexed tokens be comparable for equality
    and hashable. This is verified at runtime using the GeneralizedToken
    protocol.

    This generally means that only immutable values can be used as tokens
    in a trie key. i.e: a frozenset() works as a token, but a set() does not.
    Tokens in a trie key do NOT have to all be the same type as long as they
    can be compared for equality.

    It can handle strings, bytes, lists, sequences, and iterables of token
    objects as keys for the trie. As long as the tokens used, whether characters
    in a string or frozensets in a list, are comparable and hashable, it 'just works'.

    The code emphasizes robustness and correctness.

    Usage:

    Example 1:
        from gentries.generalized import GeneralizedTrie

        trie: GeneralizedTrie = GeneralizedTrie()
        trie_id_1 = trie.add(['ape', 'green', 'apple'])
        trie_id_2 = trie.add(['ape', 'green'])
        matches = trie.prefixes(['ape', 'green'])

    Example 2:
        from gentries.generalized import GeneralizedTrie

        # Create a trie to store website URLs
        url_trie = GeneralizedTrie()

        # Add some URLs with different components (protocol, domain, path)
        url_trie.add(["https", "com", "example", "www", "/", "products", "clothing"])
        url_trie.add(["http", "org", "example", "blog", "/" "2023", "10", "best-laptops"])
        url_trie.add(["ftp", "net", "example", "ftp", "/", "data", "images"])

        # Find all https URLs with "example.com" domain
        prefixes = url_trie.prefixes(["https", "com", "example"])
        print(f"Found URL prefixes: {prefixes}")  # Output: Found URL prefixes: {1}
    """

    def __init__(self) -> None:
        self._root_node: bool = True
        self._node_token: Optional[GeneralizedToken] = None
        self._parent: Optional[GeneralizedTrie] = None
        self._children: Dict[GeneralizedToken, GeneralizedTrie] = {}
        self._trie_index: Dict[int, GeneralizedTrie] = {}
        self._trie_id: int = 0
        self._trie_id_counter: Dict[str, int] = {"trie_number": 0}

    def _add_new_child(self, /, node_token: GeneralizedToken, trie_key: Any) -> int:
        """Creates and adds a new GeneralizedTrie node to the node's _children.

        The new node is initialized with the passed arguments. This is used
        recursively by the add() method to actually add the trie key to the trie.

        Args:
            node_token (GeneralizedToken):
                The node_token for the new child.
            trie_key (Iterator):
                Remaining tokens (if any) in the trie key.

        Returns:
            int:
                Id number of the new GeneralizedTrie key.

        Raises:
            AssertionError:
                If node_token does not conform to the GeneralizedToken protocol.
            AssertionError:
                If trie_key is not an Iterator.
            TypeError:
                If entries in trie_key do not conform to the GeneralizedToken protocol.
        """
        # pylint: disable=protected-access
        # trunk-ignore(bandit/B101)
        assert isinstance(node_token, GeneralizedToken) and isinstance(
            trie_key, Iterator
        ), "[GTANC001] incorrect arguments passed to _add_new_child()"
        new_child: GeneralizedTrie = GeneralizedTrie()
        new_child._root_node = False
        new_child._node_token = node_token
        new_child._parent = self
        new_child._trie_index = self._trie_index
        new_child._trie_id_counter = self._trie_id_counter
        trie_id: int = new_child.add(trie_key)
        self._children[node_token] = new_child
        return trie_id

    @property
    def _trie_number(self) -> int:
        """Getter for the _trie_number property.

        Returns:
            int:
                the current _trie_number property value.
        """
        return self._trie_id_counter["trie_number"]

    @_trie_number.setter
    def _trie_number(self, value: int) -> None:
        """Setter for the _trie_number property.

        Args:
            value (int): non-negative integer value.

        Raises:
            AssertionError:
                If value is not of type int.
            AssertionError:
                If value is negative.
        """
        # trunk-ignore(bandit/B101)
        assert isinstance(
            value, int
        ), "[GTTNS001] attempted to set _trie_number to a non-int type value"
        # trunk-ignore(bandit/B101)
        assert (
            value >= 0
        ), "[GTTNS002] attempted to set _trie_number to a negative value"
        self._trie_id_counter["trie_number"] = value

    def add(self, trie_key: Any) -> int:
        """Adds a trie key defined by the passed trie_key to the trie.

        Args:
            trie_key (Any):
                Must be an object that can be iterated and contains entries
                conforming to the GeneralizedToken protocol.

        Raises:
            TypeError:
                If trie_key cannot be iterated on.
            KeyError:
                If trie_key has no tokens.
            TypeError:
                If entries in trie_key do not conform to the GeneralizedToken protocol.

        Returns:
            int: id number of the inserted trie key.
        """
        if not isinstance(trie_key, Iterator):
            try:
                trie_key = iter(trie_key)
            except TypeError as err:
                raise TypeError(
                    f"[GTAFBT001] trie_key arg is not iterable: {err}"
                ) from err

        try:
            first_token: Any = next(trie_key)  # type: ignore
        except StopIteration:
            if self._root_node:
                raise ValueError("[GTAFBT002] empty trie_key passed") from None
            # already existing trie key
            if self._trie_id:
                return self._trie_id
            # new trie key
            new_trie_id: int = self._trie_number + 1
            self._trie_id = new_trie_id
            self._trie_number = new_trie_id
            self._trie_index[new_trie_id] = self
            return new_trie_id

        if not isinstance(first_token, GeneralizedToken):
            raise TypeError(
                "[GTAFBT003] entry in trie_key arg does not support the GeneralizedToken protocol"
            )

        # there is an existing child trie we can use
        if first_token in self._children:
            return self._children[first_token].add(trie_key)

        # we need a new sub-trie
        return self._add_new_child(node_token=first_token, trie_key=trie_key)

    def remove(self, trie_id: int) -> None:
        """Remove the trie key with the passed trie_id from the trie.

        Args:
            trie_id (int): id of the trie key to remove.

        Raises:
            TypeError:
                trie_id arg is not of type int or an int sub-class.
            ValueError:
                trie_id arg is less than 1.
            KeyError:
                trie_id does not match the id of any trie keys.
        """
        # pylint: disable=protected-access
        if not isinstance(trie_id, int):  # type: ignore
            raise TypeError("[GTR001] trie_id arg must be type int or an int sub-class")
        if trie_id < 1:
            raise KeyError("[GTR002] trie_id arg must be 1 or greater")

        # Not a known trie id
        if trie_id not in self._trie_index:
            raise KeyError("[GTR003] trie_id arg does not match any trie key ids")

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
        node_token: Any = node._node_token
        parent_node = node._parent
        while parent_node is not None:
            del parent_node._children[node_token]
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

    def prefixes(self, trie_key: Any) -> Set[int]:
        """Returns the ids of all keys in the trie that are a prefix of the passed trie_key.

        Searches the trie for all trie keys that are prefix matches
        for the trie_key and returns their ids as a set.

        Usage:
            trie: GeneralizedTrie = GeneralizedTrie()
            keys: List[str] = ['abcdef', 'abc', 'a', 'abcd', 'qrs']
            trie_keys_index: Dict[int, str] = {}
            for entry in keys:
                trie_key_index[trie.add(entry)] = entry
            matches: Set[int] = trie.prefix_key_ids('abcd')

            # matches now contains the set {2, 3, 4}, corresponding
            # to the trie keys 'abc', 'a', and 'abcd' - all of which are
            prefix matches for 'abcd'.

            # 2: abcd
            # 3: a
            # 4: abcd
            for trie_id in sorted(list(matches)):
                print(f'{trie_id}: {trie_keys_index[trie_id]}')

        Args:
            trie_key (Any):
                trie key for matching.

        Returns:
            Set[int]:
                Set of ids for trie keys that are prefixes of
                the trie_key. This will be an empty set if there
                are no matches.

        Raises:
            TypeError:
                If trie_key arg is not iterable.
            TypeError:
                If entries in the trie_key arg do not support the
                GeneralizedToken protocol.
        """
        if not isinstance(trie_key, Iterator):
            try:
                trie_key = iter(trie_key)
            except TypeError as err:
                raise TypeError(
                    f"[GTM001] trie_key arg cannot be iterated: {err}"
                ) from err

        matched: Set[int] = set([self._trie_id]) if self._trie_id else set()
        try:
            token: Any = next(trie_key)  # type: ignore
            if token in self._children:
                matched = matched.union(self._children[token].prefixes(trie_key))
        except StopIteration:
            pass

        return matched

    def suffixes(self, trie_key: Any, depth: int = -1) -> Set[int]:
        """Returns the ids of all suffixs of the trie_key up to depth.

        Searches the trie for all trie keys that are suffix matches for
        the trie_key up to the specified depth and returns their ids as a set.

        Usage:
            trie: GeneralizedTrie = GeneralizedTrie()
            keys: List[str] = ['abcdef', 'abc', 'a', 'abcd', 'qrs']
            trie_keys_index: Dict[int, str] = {}
            for entry in keys:
                trie_key_index[trie.add(entry)] = entry
            matches: Set[int] = trie.token_suffixes('abcd')

            # matches now contains the set {1, 4}, corresponding
            # to the trie keys 'abcdef' and 'abcd' - each of which are
            # suffix matches to 'abcd'.

            # 1: abcdef
            # 4: abc
            for trie_id in sorted(list(matches)):
                print(f'{trie_id}: {trie_keys_index[trie_id]}')


        Args:
            trie_key (Any):
                trie_key for matching.
            depth (int):
                depth starting from the matched trie key to include.

                The depth determines how many 'layers' deeper into the trie
                to look for ids:
                    * A depth of -1 (default) includes ALL ids for the exact match and all children
                      nodes.
                    * A depth of 0 only includes the ids for the *exact* match for the trie key.
                    * A depth of 1 includes ids for the exact match and the next layer down.
                    * A depth of 2 includes ids for the exact match and the next two layers down.

        Returns:
            Set[int]:
                Set of ids for trie keys that are suffix matchs for
                the trie_key. This will be an empty set if there
                are no matches.

        Raises:
            TypeError:
                If trie_key arg is not iterable.
            TypeError:
                If depth is not an int or a sub-class of int.
            ValueError:
                If depth is less than -1.
            TypeError:
                If entries in the trie_key arg do not support the
                GeneralizedToken protocol.
        """
        if not isinstance(trie_key, Iterator):
            try:
                trie_key = iter(trie_key)
            except TypeError as err:
                raise TypeError(
                    f"[GTS001] trie_key arg cannot be iterated: {err}"
                ) from err

        if not isinstance(depth, int):  # type: ignore
            raise TypeError("[GTS002] depth must be of type int or a sub-class")
        if depth < -1:
            raise ValueError("[GTS003] depth cannot be less than -1")
        try:
            token: Any = next(trie_key)  # type: ignore
        except StopIteration:  # found the match
            return self._contained_ids(ids=set(), depth=depth)
        if not isinstance(token, GeneralizedToken):
            raise TypeError(
                "[GTS004] token found that does not conform to GeneralizedToken protocol"
            )

        if token in self._children:  # looking for match
            return self._children[token].suffixes(trie_key, depth)
        return set()  # no match

    def _contained_ids(self, ids: set[int], depth: int = -1) -> Set[int]:
        """Returns a set contains all trie key ids defined for this node and/or its children up to
           the requested depth.
                    * A negative (-1 or lower) depth includes ALL ids for this node and all children
                      nodes.
                    * A depth of 0 includes ONLY the ids for this node.
                    * A depth of 1 includes ids for this node and its direct child nodes.
                    * A depth of 2 includes ids for this node and the next two layers below it.
                    * and so on.

        Returns:
            Set[int]:
                Set containing the ids of all contained trie keys.
        """
        # trunk-ignore(bandit/B101)
        assert isinstance(ids, set), "[GTCI001] ids arg must be a set or sub-class"
        # trunk-ignore(bandit/B101)
        assert isinstance(depth, int), "[GTCI002] depth arg must be an int or sub-class"
        if self._trie_id:
            ids.add(self._trie_id)
        if depth:
            depth -= 1
            for node in self._children.values():
                # pylint: disable=protected-access
                node._contained_ids(ids, depth)
        return ids

    def __contains__(self, trie_key: Any) -> Set[int]:
        """Returns True if the trie contains a key matching the trie_key.

        Usage:
            trie: GeneralizedTrie = GeneralizedTrie()
            keys: List[str] = ['abcdef', 'abc', 'a', 'abcd', 'qrs']
            trie_keys_index: Dict[int, str] = {}
            for entry in keys:
                trie_key_index[trie.add(entry)] = entry

            if 'abc' in trie:
                print('abc is in the trie')

        Args:
            trie_key (Any):
                trie key for matching.

        Returns:
            bool:
                (False):
                    Trie does not contain a key matching the trie_key.
                (True):
                    Trie contains a key matching the trie_key.

        Raises:
            TypeError:
                If trie_key arg is not iterable.
            TypeError:
                If entries in the trie_key arg do not support the
                GeneralizedToken protocol.
        """
        return bool(self.suffixes(trie_key, 0))  # type: ignore

    def __len__(self) -> int:
        """Returns the number of keys in the trie.

        Usage:
            n_trie_keys: int = len(trie)

        Returns:
            (int) number of keys in the trie.
        """
        return len(self._trie_index)

    def __str__(self) -> str:
        """Generates a stringified version of the trie for visual examination.

        The output IS NOT executable code but more in the nature of debug support."""
        output: List[str] = ["{"]
        if self._root_node:
            output.append(f"  trie number = {self._trie_number}")
        elif self._parent:
            if self._parent._root_node:
                output.append("  parent = root node")
            else:
                output.append(f"  parent = {self._parent._node_token}")
        output.append(f"  node token = {self._node_token}")
        if self._trie_id:
            output.append(f"  trie id = {self._trie_id}")
        if self._children:
            output.append("  children = {")
            for child_key, child_value in self._children.items():
                output.append(
                    f"    {child_key} = " + indent(str(child_value), "    ").lstrip()
                )
            output.append("  }")
        if self._root_node:
            output.append(f"  trie index = {self._trie_index.keys()}")
        output.append("}")
        return "\n".join(output)
