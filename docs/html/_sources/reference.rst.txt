API Reference
=============

This section provides detailed information about the API of **gen-trie**. 
It includes descriptions of the main classes, methods, and attributes
available in the library.

gen-trie is a Python library for creating and manipulating
'generalized' trie data structures. Unlike common trie implementations that
only support strings or some other specific types as keys, this generalized
trie can handle various types of tokens in keys, including integers, floats,
tuples, lists, frozen sets,  and even custom objects, as long as they are hashable.

It is designed to be flexible and efficient, allowing for complex data structures
to be represented and manipulated easily.

.. index::

.. py:module:: gentrie
    :synopsis: Generalized Trie Data Structure


.. class:: GeneralizedTrie

    .. method:: insert(key, value)

        Inserts a key-value pair into the trie.

    .. method:: search(key)

        Searches for a key in the trie and returns its associated value, if found.

    .. method:: delete(key)

        Deletes a key-value pair from the trie.

