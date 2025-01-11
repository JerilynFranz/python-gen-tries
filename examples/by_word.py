#!/usr/bin/env python3

from gentrie import GeneralizedTrie, TrieEntry

trie = GeneralizedTrie()
trie.add(['ape', 'green', 'apple'])
trie.add(['ape', 'green'])
matches: set[TrieEntry] = trie.prefixes(['ape', 'green'])
print(matches)
