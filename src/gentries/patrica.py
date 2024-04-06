from copy import copy
from typing import Any, Dict, Iterator, Optional, Set


class PatriciaTrie:
    """Implementation of a PATRICIA Trie data structure.


    Usage:
        trie = PatriciaTrie()
        trie_id = trie.add(tokens=['ape', 'green', 'apple'])

    Raises:
        TypeError: If passed a tokens argument that does not support iteration.
        TypeError: If entries in the tokens argument do not support both
                   __eq__() and __hash__().
        TypeError: _description_
        TypeError: _description_
        TypeError: _description_
        TypeError: _description_
        TypeError: _description_
        ValueError: _description_
        ValueError: _description_
        TypeError: _description_
        TypeError: _description_
        TypeError: _description_
        ValueError: _description_
        TypeError: _description_

    Returns:
       gentries.patrica.PatriciaTrie: Instance of PatraciaTrie.
    """
    def __init__(self, /,
                 **kwargs) -> None:
        # pylint: disable=too-many-branches
        self._root_node: bool = True
        self._node_token: Any = None
        self._parent: Optional[PatriciaTrie] = None
        self._children: Dict[Any, PatriciaTrie] = {}
        self._trie_index: Dict[int, PatriciaTrie] = {}
        self._trie_ids: Set[int] = set()
        self._trie_id_counter: Dict[str, int]

        if kwargs:
            if 'root_node' in kwargs:
                root_node: bool = kwargs['root_node']
                if not isinstance(root_node, bool):
                    raise TypeError(
                        '[PT001] root_node arg must be type bool '
                        'or a sub-class')
                self._root_node = root_node
            else:
                raise KeyError('[PT002] missing root_node arg')

            if 'node_token' in kwargs:
                node_token: Any = kwargs['node_token']
                try:
                    self._validate_token(token=node_token)
                except TypeError as err:
                    raise TypeError(
                        f'[PT003] node_token {err}') from err
                self._node_token = node_token
            else:
                raise KeyError('[PT004] missing node_token arg')

            if 'parent' in kwargs:
                parent: PatriciaTrie = kwargs['parent']
                if not isinstance(parent, PatriciaTrie):
                    raise TypeError(
                        '[PT003] parent arg must be of '
                        'type PatriciaTrie or a sub-class')
            else:
                raise KeyError('[PT004] missing parent arg')

            if 'trie_index' in kwargs:
                trie_index: Dict[int, PatriciaTrie] = kwargs[
                                                            'trie_index']
                if not isinstance(trie_index, Dict):
                    raise TypeError(
                        '[PT005] trie_index arg must be of '
                        'type Dict')
                self._trie_index = trie_index
            else:
                raise KeyError('[PT006] missing trie_index arg')

            if 'trie_id_counter' in kwargs:
                trie_id_counter = kwargs['trie_id_counter']
                if not isinstance(trie_id_counter, Dict):
                    raise TypeError(
                        '[PT007] trie_id_counter arg must be a Dict or '
                        'a sub-class')
                if 'trie_number' not in trie_id_counter:
                    raise ValueError(
                        '[PT008] missing trie_number key in trie_id_counter '
                        'arg')
                if not isinstance(trie_id_counter['trie_number'], int):
                    raise ValueError(
                        '[PT009]trie_number key in trie_id_counter must be '
                        'an int or sub-class')
                if trie_id_counter['trie_number'] < 0:
                    raise ValueError(
                        '[PT010] trie_number value in trie_id_counter must be '
                        'non-negative')
                self._trie_id_counter = trie_id_counter
            else:
                raise KeyError('[PT011] missing trie_id_counter arg')
        else:
            self._trie_id_counter = {'trie_number': 0}

    @property
    def _trie_number(self) -> int:
        return self._trie_id_counter('trie_number') + 1

    @_trie_number.setter
    def _trie_number(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError('_trie_number must be of type int or a sub-class')
        if value < 0:
            raise ValueError('_trie_number must be non-negative')
        self._trie_id_counter['trie_number'] = value

    def _validate_token(self, /, token: Any) -> None:
        """Validates that the passed token supports __eq__ and __hash__.

        This is required to allow matching tokens and using them
        as keys.

        Args:
            token (Any): Token for validation

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

    def add(self, tokens: Any) -> None:
        if not isinstance(tokens, Iterator):
            try:
                tokens = iter(tokens)
            except TypeError as err:
                raise TypeError(
                    '[PTAFBT001] tokens arg cannot '
                    f'be iterated: {err}') from err

        if tokens is None:
            new_trie_id: int = self._trie_number + 1
            self._trie_ids.add(new_trie_id)
            self._trie_number = new_trie_id
            self._trie_index[new_trie_id] = self
            return

        first_token: Any = next(tokens)
        try:
            self._validate_token(token=first_token)
        except TypeError as err:
            raise TypeError('[PTAFBT005] entries in tokens arg must '
                            f'support __eq__ and __hash__ methods: '
                            f'{err}') from err

        if first_token in self._children:
            return self._children[first_token].add(tokens)

        new_trie: PatriciaTrie = PatriciaTrie(
                                    root_node=False,
                                    node_token=first_token,
                                    parent=self,
                                    trie_index=self._trie_index,
                                    trie_id_counter=self._trie_id_counter)
        trie_id = new_trie.add(tokens)
        self._children[first_token] = new_trie
        return trie_id

    def remove(self, trie_id: int) -> bool:
        """Remove the trie entry with the given trie_id from the trie.

        Args:
            trie_id (int): id of the trie entry to remove.

        Raises:
            TypeError: if trie_id arg is not type int or an int sub-class
            ValueError: if trie_id arg is less than 1.

        Returns:
            bool: True if the trie entry was removed, False otherwise.
        """
        # pylint: disable=protected-access
        if not isinstance(trie_id, int):
            raise TypeError(
                '[PTRF001] trie_id arg must be type int or an int sub-class')
        if trie_id < 1:
            raise ValueError(
                '[PTRF002] trie_id arg cannot be less than 1')

        # Not a known trie id
        if trie_id not in self._trie_index:
            return False

        # Find the node and delete its id from the trie index
        trie_node: PatriciaTrie = self._trie_index[trie_id]
        node_token: Any = trie_node._node_token
        del trie_node._trie_index[trie_id]
        trie_node._trie_index = None
        parent_node: PatriciaTrie = trie_node._parent
        trie_node._parent = None

        # If the node still has other trie ids or children, return.
        if trie_node._trie_ids or trie_node._children:
            return True

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
            next_parent_node: PatriciaTrie = parent_node._parent
            parent_node._parent = None
            parent_node = next_parent_node

        return True

    def match(self, tokens: Any) -> Set[int]:
        """Search the trie for all trie entries that match the given tokens.

        Args:
            tokens (Any): Ordered tokens for matching.

        Returns:
            Set[int]: Set of trie ids that match the given tokens.

        Raises:
            TypeError: If tokens arg is not iterable.
            TypeError: If entries in tokens arg do not support __eq__ and
                       __hash__ methods.
        """
        # pylint: disable=protected-access
        if not isinstance(tokens, Iterator):
            try:
                tokens = iter(tokens)
            except TypeError as err:
                raise TypeError(
                    '[PTAFBT001] tokens arg cannot '
                    f'be iterated: {err}') from err

        matched: Set[int] = copy(self._trie_ids) if self._trie_ids else set()
        token_entry = next(tokens, default=None)
        while token_entry is not None:
            if token_entry in self._children:
                matched = matched.union(
                    self._children[token_entry].match(tokens=tokens))
            token_entry = next(tokens, default=None)
        return matched
