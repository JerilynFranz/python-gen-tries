#!/usr/bin/env python3

from gentrie import GeneralizedTrie, TrieEntry

trie = GeneralizedTrie()
trie.add('abcdef')
trie.add('abc')
trie.add('qrf')
matches: set[TrieEntry] = trie.suffixes('ab')
print(matches)
