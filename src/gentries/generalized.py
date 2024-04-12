from copy import copy
from textwrap import indent
from typing import Any, Dict, Iterator, Set

from . import GeneralizedToken, InvalidTokenError


class GeneralizedTrie:
    """Implementation of a general purpose trie.

    Unlike many Trie implementations, which only support strings as entries
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
        prefixes = url_trie.key_prefixes(["https", "com", "example"])
        print(f"Found URL prefixes: {prefixes}")  # Output: Found URL prefixes: {1}
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
                       trie_key: Iterator) -> int:
        """Creates and adds a new GeneralizedTrie node to the node's _children.

        The new node is initialized with the passed arguments. This is used
        recursively by the add() method to actually add a trie key to the trie.

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
            int:
                the current _trie_number property value.
        """
        return self._trie_id_counter['trie_number']

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
        assert isinstance(value, int), '[GTTNS001] attempted to set _trie_number to a non-int type value'
        # trunk-ignore(bandit/B101)
        assert value >= 0, '[GTTNS002] attempted to set _trie_number to a negative value'
        self._trie_id_counter['trie_number'] = value

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
            InvalidTokenError:
                If entries in trie_key do not conform to the GeneralizedToken protocol.

        Returns:
            int: id number of the inserted trie key.
        """
        if not isinstance(trie_key, Iterator):
            try:
                trie_key = iter(trie_key)
            except TypeError as err:
                raise TypeError(
                    f'[GTAFBT001] trie_key arg is not iterable: {err}') from err

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
            raise InvalidTokenError(
                '[GTAFBT003] entry in trie_key arg does not support the GeneralizedToken protocol')

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
        if not isinstance(trie_id, int):
            raise TypeError(
                '[GTR001] trie_id arg must be type int or an int sub-class')
        if trie_id < 1:
            raise KeyError(
                '[GTR002] trie_id arg must be 1 or greater')

        # Not a known trie id
        if trie_id not in self._trie_index:
            raise KeyError(
                '[GTR003] trie_id arg does not match any trie key ids')

        # Find the node and delete its id from the trie index
        node: GeneralizedTrie = self._trie_index[trie_id]
        del node._trie_index[trie_id]

        # Remove the id from the node
        node._trie_ids.remove(trie_id)

        # If the node still has other trie ids or children, we're done: return
        if node._trie_ids or node._children:
            return

        # No trie ids or children are left for this node, so prune
        # nodes up the trie tree as needed.
        node_token: Any = node._node_token
        parent_node = node._parent
        while parent_node is not None:
            del parent_node._children[node_token]
            # explicitly break any possible cyclic references
            node._parent = node._node_token = node._trie_ids = node._trie_id_counter = None
            # If the node still has other trie ids or children, we're done: return
            if parent_node._trie_ids or parent_node._children:
                return
            # Keep purging nodes up the tree
            node_token = parent_node._node_token
            node = parent_node
            parent_node = node._parent
        return

    def prefixes(self, tokens: Any) -> Set[int]:
        """Returns the ids of all trie keys that are a prefix of the tokens.

        Searches the trie for all trie keys that are prefix matches
        for the tokens and returns their ids as a set.

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
            tokens (Any):
                trie_key for matching.

        Returns:
            Set[int]:
                Set of ids for trie keys that are prefixes of
                the tokens. This will be an empty set if there
                are no matches.

        Raises:
            TypeError:
                If tokens arg is not iterable.
            InvalidTokenError:
                If entries in the tokens arg do not support the
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
                self._children[token_entry].prefixes(tokens))
        return matched

    def suffixes(self, tokens: Any) -> Set[int]:
        """Returns the ids of all trie keys that are suffixs of the tokens.

        Searches the trie for all trie keys that are suffix matches for
        the tokens and returns their ids as a set.

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
            tokens (Any):
                trie_key for matching.

        Returns:
            Set[int]:
                Set of ids for trie keys that are suffix matchs for
                the tokens. This will be an empty set if there
                are no matches.

        Raises:
            TypeError:
                If tokens arg is not iterable.
            InvalidTokenError:
                If entries in the tokens arg do not support the
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
                self._children[token_entry].prefixes(tokens))
        return matched

    def __contains__(self, tokens: Any) -> Set[int]:
        """Returns True if the trie contains a key matching the tokens.

        Usage:
            trie: GeneralizedTrie = GeneralizedTrie()
            keys: List[str] = ['abcdef', 'abc', 'a', 'abcd', 'qrs']
            trie_keys_index: Dict[int, str] = {}
            for entry in keys:
                trie_key_index[trie.add(entry)] = entry

            if 'abc' in trie:
                print('abc is in the trie')

        Args:
            tokens (Any):
                trie key for matching.

        Returns:
            bool:
                (False):
                    Trie does not contain a key matching the tokens.
                (True):
                    Trie contains a key matching the tokens.

        Raises:
            TypeError:
                If tokens arg is not iterable.
            InvalidTokenError:
                If entries in the tokens arg do not support the
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
                self._children[token_entry].prefixes(tokens))
        return matched

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
        output: str = ['{']
        if self._root_node:
            output.append(f'  trie number = {self._trie_number}')
        else:
            if self._parent._root_node:
                output.append('  parent = root node')
            else:
                output.append(f'  parent = {self._parent._node_token}')
        output.append(f'  node token = {self._node_token}')
        trie_ids: str = str(self._trie_ids) if self._trie_ids else '{}'
        output.append(f'  trie ids = {trie_ids}')
        output.append('  children = {')
        for child_key, child_value in self._children.items():
            output.append(f'    {child_key} = ' + indent(str(child_value), '    ').lstrip())
        output.append('  }')
        if self._root_node:
            output.append(f'  trie index = {self._trie_index.keys()}')
        trie_ids: str = str(self._trie_ids) if self._trie_ids else '{}'
        output.append('}')
        return '\n'.join(output)
