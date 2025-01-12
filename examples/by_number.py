#!/usr/bin/env python3

from gentrie import GeneralizedTrie, TrieEntry

trie = GeneralizedTrie()
entries = [
    [128, 256, 512],
    [128, 256],
    [512, 1024],
]
for item in entries:
    trie.add(item)
matches: set[TrieEntry] = trie.suffixes([128])
print(matches)

# {TrieEntry(ident=1, key=[128, 256, 512]), TrieEntry(ident=2, key=[128, 256])}
