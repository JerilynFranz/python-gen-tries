from copy import copy
from typing import Any, Dict, Iterator, Set


from . import GeneralizedToken


class GeneralizedTrie:
    """Implementation of a general purpose trie.

    Unlike many Trie implementations, which only support strings as entries,
    and match only at the character level, it is agnostic as to the types of
    tokens used to key it and thus much more general purpose.

    It requires only that the indexed tokens be comparable and hashable. This
    is verified at runtime using the GeneralizedToken protocol.

    It can handle strings, bytes, lists, sequences, and iterables of token
    objects. As long as the tokens used, whether characters in a string or
    a list of objects, are comparable and hashable, it 'just works'.

    The code emphasizes robustness and correctness.

    Usage:
        trie: GeneralizedTrie = GeneralizedTrie()
        trie_id_1 = trie.add(['ape', 'green', 'apple'])
        trie_id_2 = trie.add(['ape', 'green'])
        matches = trie.match(['ape', 'green'])
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
                       tokens: Iterator) -> 'GeneralizedTrie':
        """Creates and adds a new GeneralizedTrie node to the node's _children.

        The new node is initialized with the passed arguments.

        Args:
            node_token (GeneralizedToken): The node_token for the new child.
            tokens (Iterator): Remaining tokens (if any) in the trie key.

        Returns:
            int: Id number of the new GeneralizedTrie key.

        Raises:
            AssertionError:
                If node_token does not conform to the GeneralizedToken
                protocol.
            AssertionError:
                If tokens are not an Iterator.
            TypeError:
                If entries in tokens do not conform to the GeneralizedToken
                protocol.
        """
        # trunk-ignore(bandit/B101)
        assert (
            (isinstance(node_token, GeneralizedToken) and
             isinstance(tokens, Iterator))), (
             '[GTANC001] incorrect arguments passed to _add_new_child()')
        new_child: GeneralizedTrie = GeneralizedTrie()
        new_child._root_node = False
        new_child._node_token = node_token
        new_child._parent = self
        new_child._trie_index = self._trie_index
        new_child._trie_id_counter = self._trie_id_counter
        trie_id: int = new_child.add(tokens)
        self._children[trie_id] = new_child
        return trie_id

    @property
    def _trie_number(self) -> int:
        """Getter for the _trie_number property.

        Returns:
            int: the current _trie_number property value.
        """
        return self._trie_id_counter('trie_number') + 1

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

    def _validate_token_protocol(self, token: GeneralizedToken) -> None:
        """Validates that the passed token supports __eq__ and __hash__.

        This is required to allow matching tokens and using them
        as keys in hashes.

        Args:
            token (Any):  for validation

        Raises:
            TypeError: If does not support __eq__ method
            TypeError: If does not support __hash__ method
        """
        if not (hasattr(token, '__eq__')
                and callable(token.__eq__)):
            raise TypeError('missing an __eq__ method')
        if not (hasattr(token, '__hash__')
                and callable(token.__hash__)):
            raise TypeError('missing a __hash__ method')

    def add(self, tokens: Any) -> int:
        """Adds a trie key defined by the passed tokens to the trie.

        Args:
            tokens (Any): Must be an object that can be used in iteration and
                          containing entries conforming to the GeneralizedToken
                          protocol.

        Raises:
            TypeError: If tokens cannot be iterated on.
            TypeError: If entries in tokens do not conform to the
                       GeneralizedToken protocol.

        Returns:
            int: id number of the inserted trie key.
        """
        if not isinstance(tokens, Iterator):
            try:
                tokens = iter(tokens)
            except TypeError as err:
                raise TypeError(
                    '[GTAFBT001] tokens arg cannot '
                    f'be iterated: {err}') from err

        # When passed None, we have run out of tokens to iterate
        if tokens is None:
            new_trie_id: int = self._trie_number + 1
            self._trie_ids.add(new_trie_id)
            self._trie_number = new_trie_id
            self._trie_index[new_trie_id] = self
            return new_trie_id

        # We will always have at least one token here.
        first_token: Any = next(tokens)
        if not isinstance(first_token, GeneralizedToken):
            raise TypeError(
                '[GTAFBT002] entry in tokens arg does not support the '
                'GeneralizedToken protocol')

        # there is an existing child trie we can use
        if first_token in self._children:
            return self._children[first_token].add(tokens)

        # we need a new sub-trie
        return self._add_new_child(node_token=first_token, tokens=tokens)

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

    def match(self, tokens: Any) -> Set[int]:
        """Search the trie for all trie entries that match the given tokens.

        Args:
            tokens (Any): Ordered tokens for matching.

        Returns:
            Set[int]: Set of trie key ids that match the given tokens. This
                      will be an empty set if there are no matches.

        Raises:
            TypeError: If tokens arg is not iterable.
            TypeError: If entries in the tokens arg do not support the
                       GeneralizedToken protocol.
        """
        if not isinstance(tokens, Iterator):
            try:
                tokens = iter(tokens)
            except TypeError as err:
                raise TypeError(
                    f'[GTM001] tokens arg cannot be iterated: {err}') from err

        matched: Set[int] = copy(self._trie_ids) if self._trie_ids else set()
        token_entry = next(tokens, default=None)
        while token_entry is not None:
            if token_entry in self._children:
                matched = matched.union(
                    self._children[token_entry].match(tokens=tokens))
            token_entry = next(tokens, default=None)
        return matched
