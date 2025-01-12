==============
Using gen-trie
==============

Stuff goes here

.. _gentrie-installation:

------------
Installation
------------

``pip3 install gen-tries``

-----
Usage
-----

Example 1 - trie of words::

    from gentrie import GeneralizedTrie, TrieEntry

    trie = GeneralizedTrie()
    trie.add(['ape', 'green', 'apple'])
    trie.add(['ape', 'green'])
    matches: set[TrieEntry] = trie.prefixes(['ape', 'green'])
    print(matches)


Example 1 Output::

    {TrieEntry(ident=2, key=['ape', 'green'])}}


Example 2 - trie of words used for URL search::

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

    {TrieEntry(ident=1, key=['https', 'com', 'example', 'www', '/', 'products', 'clothing'])}


Example 3 - trie of letters from string::

    from gentrie import GeneralizedTrie, TrieEntry

    trie = GeneralizedTrie()
    trie.add('abcdef')
    trie.add('abc')
    trie.add('qrf')
    matches: set[TrieEntry] = trie.suffixes('ab')
    print(matches)


Example 3 Output::

    {TrieEntry(ident=2, key='abc'), TrieEntry(ident=1, key='abcdef')}


Example 4 - trie of numeric vectors::

    from gentrie import GeneralizedTrie, TrieEntry

    trie = GeneralizedTrie()
    entries = [
        [128, 256, 512],
        [128, 256],
        [512, 1024],
    ]
    for item in entries:
        trie.add(item)
    matches: set[TrieEntry] = trie.suffixes(128)
    print(matches)


Example 4 Output::

    {TrieEntry(ident=1, key=[128, 256, 512]), TrieEntry(ident=2, key=[128, 256])}

Example 5 - trie of tuples::

    from gentrie import GeneralizedTrie, TrieEntry

    trie = GeneralizedTrie()
    entries = [
        [(1, 2), (3, 4), (5, 6)],
        [(1, 2), (3, 4)],
        [(5, 6), (7, 8)],
    ]
    for item in entries:
        trie.add(item)
    matches: set[TrieEntry] = trie.suffixes([(1, 2)])
    print(matches)

Example 5 Output::

    {TrieEntry(ident=1, key=[(1, 2), (3, 4), (5, 6)]), TrieEntry(ident=2, key=[(1, 2), (3, 4)])}
