==============
Using gen-trie
==============

.. _gentrie-installation:

------------
Installation
------------

**Via PyPI**::

    pip3 install gen-tries
    
**From source**::

    git clone https://github.com/JerilynFranz/python-gen-tries
    cd python-gen-tries
    pip3 install .

-----
Usage
-----
The `gentrie` module provides a `GeneralizedTrie` class that allows you to create a trie structure for
storing and searching sequences of items, such as strings, lists, or tuples.

Below are some examples of how to use the `GeneralizedTrie` class.

Examples
========

By Letter
----------------

.. include:: ../../examples/by_letter.py
   :code: python

By Number
----------------

.. include:: ../../examples/by_number.py
   :code: python

By Tuple
----------------

.. include:: ../../examples/by_tuple.py
   :code: python

By Word
----------------

.. include:: ../../examples/by_word.py
   :code: python


Key In Trie
----------------

.. include:: ../../examples/key_in_trie.py
   :code: python


Prefixes
----------------

.. include:: ../../examples/prefixes_example.py
   :code: python

Suffixes
----------------

.. include:: ../../examples/suffixes_example.py
   :code: python

URL Suffixes
----------------

.. include:: ../../examples/url_suffixes.py
   :code: python

Word Suggestions
----------------

.. include:: ../../examples/word_suggestion.py
   :code: python


Trie of numeric vectors
------------------------

.. code-block:: python

    from gentrie import GeneralizedTrie, TrieEntry

    trie = GeneralizedTrie()
    entries = [
        [128, 256, 512],
        [128, 256],
        [512, 1024],
    ]
    for item in entries:
        trie.add(item)
    suffixes: set[TrieEntry] = trie.suffixes([128])
    print(f'suffixes = {suffixes}')

    prefixes: set[TrieEntry] = trie.prefixes([128, 256, 512, 1024])
    print(f'prefixes = {prefixes}')

    # suffixes = {TrieEntry(ident=1, key=[128, 256, 512]),
    #             TrieEntry(ident=2, key=[128, 256])}
    # prefixes = {TrieEntry(ident=1, key=[128, 256, 512]),
    #             TrieEntry(ident=2, key=[128, 256])}

