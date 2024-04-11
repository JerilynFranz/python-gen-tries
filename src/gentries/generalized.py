from copy import copy
from textwrap import indent
from typing import Any, Dict, Iterator, Set

from . import GeneralizedToken


class GeneralizedTrie:
    """Implementation of a general purpose trie.

    Unlike many Trie implementations, which only support strings as entries,
    and match only at the character level, it is agnostic as to the types of
    tokens used to key it and thus much more general purpose.

    It requires only that the indexed tokens be comparable for equality
    and hashable. This is verified at runtime using the GeneralizedToken
    protocol.

    This generally means that only immutable values can be used as tokens.
    i.e: a frozenset() works as a token, but a set() does not. Tokens
    in a trie key do NOT have to all be the same type as long as they can be
    compared for equality.

    It can handle strings, bytes, lists, sequences, and iterables of token
    objects. As long as the tokens used, whether characters in a string or
    a list of objects, are comparable and hashable, it 'just works'.

    The code emphasizes robustness and correctness.

    Usage Examples:

    Example 1:
        from gentries.generalized import GeneralizedTrie

        trie: GeneralizedTrie = GeneralizedTrie()
        trie_id_1 = trie.add(['ape', 'green', 'apple'])
        trie_id_2 = trie.add(['ape', 'green'])
        matches = trie.token_prefixes(['ape', 'green'])

    Example 2:
        from gentries.generalized import GeneralizedTrie

        # Create a trie to store website URLs
        url_trie = GeneralizedTrie()

        # Add some URLs with different components (protocol, domain, path)
        url_trie.add(["https", "com", "example", "www", "/", "products", "clothing"])
        url_trie.add(["http", "org", "example", "blog", "/" "2023", "10", "best-laptops"])
        url_trie.add(["ftp", "net", "example", "ftp", "/", "data", "images"])

        # Find all https URLs with "example.com" domain
        prefixes = url_trie.key_prefixes(["https", "com", "example"])
        print(f"Found URL prefixes: {prefixes}")  # Output: Found URL prefixes: {1}

        # Check if a specific URL exists (including all path components)
        url_id = url_trie.add(["http", "org", "example", "blog", "/", "2023", "10", "best-laptops"])
        print(f"URL exists: {url_id}")  # Output: URL exists: 3

    """
    def __init__(self):
        self._root_node: bool = True
        self._node_token: GeneralizedToken = None
        self._parent: GeneralizedTrie = None
        self._children: Dict[GeneralizedToken, GeneralizedTrie] = {}
        self._trie_index: Dict[int, GeneralizedTrie] = {}
        self._trie_ids: Set[int] = set()
        self._trie_id_counter: Dict[str, int] = {'trie_number': 0}

    def _add_new_child(self, /,
                       node_token: GeneralizedToken,
                       trie_key: Iterator) -> 'GeneralizedTrie':
        """Creates and adds a new GeneralizedTrie node to the node's _children.

        The new node is initialized with the passed arguments.

        Args:
            node_token (GeneralizedToken): The node_token for the new child.
            trie_key (Iterator): Remaining tokens (if any) in the trie key.

        Returns:
            int: Id number of the new GeneralizedTrie key.

        Raises:
            AssertionError:
                If node_token does not conform to the GeneralizedToken
                protocol.
            AssertionError:
                If trie_key is not an Iterator.
            TypeError:
                If entries in trie_key do not conform to the GeneralizedToken
                protocol.
        """
        # trunk-ignore(bandit/B101)
        assert (
            (isinstance(node_token, GeneralizedToken) and
             isinstance(trie_key, Iterator))), (
             '[GTANC001] incorrect arguments passed to _add_new_child()')
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
            int: the current _trie_number property value.
        """
        return self._trie_id_counter['trie_number']

    @_trie_number.setter
    def _trie_number(self, value: int) -> None:
        """Setter for the _trie_number property.

        Args:
            value (int): non-negative integer value.

        Raises:
            AssertionError: If value is not of type int.
            AssertionError: If value is negative."""
        # trunk-ignore(bandit/B101)
        assert isinstance(value, int), (
            '[GTTNS001] attempted to set _trie_number to a non-int type value')
        # trunk-ignore(bandit/B101)
        assert value >= 0, (
            '[GTTNS002] attempted to set _trie_number to a negative value')
        self._trie_id_counter['trie_number'] = value

    def add(self, trie_key: Any) -> int:
        """Adds a trie key defined by the passed trie_key to the trie.

        Args:
            trie_key (Any): Must be an object that can be used in iteration and
                            containing entries conforming to the GeneralizedToken
                            protocol.

        Raises:
            TypeError: If trie_key cannot be iterated on.
            TypeError: If entries in trie_key do not conform to the
                       GeneralizedToken protocol.

        Returns:
            int: id number of the inserted trie key.
        """
        if not isinstance(trie_key, Iterator):
            try:
                trie_key = iter(trie_key)
            except TypeError as err:
                raise TypeError(
                    '[GTAFBT001] trie_key arg cannot '
                    f'be iterated: {err}') from err

        first_token: Any = next(trie_key, None)
        # if first_token is None, we have run out of tokens in the trie key to iterate
        if first_token is None:
            if self._root_node:
                raise ValueError('[GTAFBT002] empty trie_key passed')
            new_trie_id: int = self._trie_number + 1
            self._trie_ids.add(new_trie_id)
            self._trie_number = new_trie_id
            self._trie_index[new_trie_id] = self
            return new_trie_id

        if not isinstance(first_token, GeneralizedToken):
            raise TypeError(
                '[GTAFBT003] entry in trie_key arg does not support the '
                'GeneralizedToken protocol')

        # there is an existing child trie we can use
        if first_token in self._children:
            return self._children[first_token].add(trie_key)

        # we need a new sub-trie
        return self._add_new_child(node_token=first_token, trie_key=trie_key)

    def remove(self, trie_id: int):
        """Remove the trie key with the passed trie_id from the trie.

        Args:
            trie_id (int): id of the trie key to remove.

        Raises:
            TypeError: if trie_id arg is not type int or an int sub-class
            ValueError: if trie_id arg is less than 1.
            ValueError: if trie_id does not match the id of any trie keys.
        """
        if not isinstance(trie_id, int):
            raise TypeError(
                '[GTR001] trie_id arg must be type int or an int sub-class')
        if trie_id < 1:
            raise ValueError(
                '[GTR002] trie_id arg must be 1 or greater')

        # Not a known trie id
        if trie_id not in self._trie_index:
            raise ValueError(
                '[GTR003] trie_id arg does not match any trie keys')

        # Find the node and delete its id from the trie index
        trie_node: GeneralizedTrie = self._trie_index[trie_id]
        node_token: Any = trie_node._node_token
        del trie_node._trie_index[trie_id]
        trie_node._trie_index = None
        parent_node: GeneralizedTrie = trie_node._parent
        trie_node._parent = None

        # If the node still has other trie ids or children, return.
        if trie_node._trie_ids or trie_node._children:
            return

        # No trie ids or children are left for this node, so purge
        # nodes up the trie tree as needed. Explicitly cleaning up
        # references to prevent generating orphans.
        while parent_node:
            if node_token in parent_node._children:
                del parent_node._children[node_token]
            if parent_node._children or parent_node._trie_ids:
                break
            # Nothing left here. Purge and move up.
            parent_node._trie_index = None
            parent_node._trie_id_counter = None
            node_token = parent_node._node_token
            next_parent_node: GeneralizedTrie = parent_node._parent
            parent_node._parent = None
            parent_node = next_parent_node

        return

    def token_prefixes(self, tokens: Any) -> Set[int]:
        """Returns the ids of all trie keys that are a prefix of the tokens.

        Searches the trie for all trie keys that are prefix subsets
        of the tokens and returns their ids.

        Example:
            trie: GeneralizedTrie = GeneralizedTrie()
            keys: List[str] = ['abcdef', 'abc', 'a', 'qrs']
            trie_keys_index: Dict[int, str] = {}
            for entry in keys:
                trie_key_index[trie.add(entry)] = entry
            matches: Set[int] = trie.prefix_key_ids('abc')

            # matches now contains the set {2, 3}, corresponding
            # to the trie keys 'abc' and 'a'

        Args:
            tokens (Any): Ordered trie_key for matching.

        Returns:
            Set[int]: Set of ids for trie keys that are prefixes of
                      the trie_key. This will be an empty set if there
                      are no matches.

        Raises:
            TypeError: If trie_key arg is not iterable.
            TypeError: If entries in the trie_key arg do not support the
                       GeneralizedToken protocol.
        """
        if not isinstance(tokens, Iterator):
            try:
                tokens = iter(tokens)
            except TypeError as err:
                raise TypeError(
                    f'[GTM001] trie_key arg cannot be iterated: {err}') from err

        matched: Set[int] = copy(self._trie_ids) if self._trie_ids else set()
        token_entry = next(tokens, None)
        if token_entry in self._children:
            matched = matched.union(
                self._children[token_entry].token_prefixes(tokens))
        return matched

    def __len__(self) -> int:
        """Returns the number of keys in the trie.

        Usage:
            n_trie_keys: int = len(trie)

        Returns:
            (int) number of keys in the trie.
        """
        return len(self._trie_ids)

    def __str__(self) -> str:
        """Generates a stringified version of the trie for visual examination.

        The output IS NOT executable code but more in the nature of debug support."""
        output: str = ['{']
        if self._root_node:
            output.append(f'  trie number = {self._trie_number}')
        output.append(f'  node token = {self._node_token}')
        trie_ids: str = str(self._trie_ids) if self._trie_ids else '{}'
        output.append(f'  trie ids = {trie_ids}')
        output.append('  children = {')
        for child_key, child_value in self._children.items():
            output.append(
                f'    {child_key} = ' +
                indent(str(child_value), '    ').lstrip())
        output.append('  }')
        if self._root_node:
            output.append(f'  trie index = {self._trie_index.keys()}')
        trie_ids: str = str(self._trie_ids) if self._trie_ids else '{}'
        output.append('}')
        return '\n'.join(output)
