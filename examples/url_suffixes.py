#!/usr/bin/env python3

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
