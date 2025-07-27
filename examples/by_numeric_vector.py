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
suffixes: set[TrieEntry] = trie.suffixes([128])
print(f'suffixes = {suffixes}')

prefixes: set[TrieEntry] = trie.prefixes([128, 256, 512, 1024])
print(f'prefixes = {prefixes}')

# suffixes = {TrieEntry(ident=1, key=[128, 256, 512], value=None),
#             TrieEntry(ident=2, key=[128, 256], value=None)}
# prefixes = {TrieEntry(ident=1, key=[128, 256, 512], value=None),
#             TrieEntry(ident=2, key=[128, 256], value=None)}
