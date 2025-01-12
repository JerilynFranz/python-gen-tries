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

Example 1::

    from gentrie import GeneralizedTrie, TrieEntry

    trie = GeneralizedTrie()
    trie.add(['ape', 'green', 'apple'])
    trie.add(['ape', 'green'])
    matches: set[TrieEntry] = trie.prefixes(['ape', 'green'])
    print(matches)


Example 1 Output::

    {TrieEntry(ident=2, key=['ape', 'green'])}}


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

    {TrieEntry(ident=1, key=['https', 'com', 'example', 'www', '/', 'products', 'clothing'])}


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
