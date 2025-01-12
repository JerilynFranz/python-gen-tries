#!/usr/bin/env python3

from gentrie import GeneralizedTrie, TrieEntry

trie = GeneralizedTrie()
entries = [
    [(1, 2), (3, 4), (5, 6)],
    [(1, 2), (3, 4)],
    [(5, 6), (7, 8)],
]
for item in entries:
    trie.add(item)
suffixes: set[TrieEntry] = trie.suffixes([(1, 2)])
print(f'suffixes = {suffixes}')
prefixes: set[TrieEntry] = trie.prefixes([(1, 2), (3, 4), (5, 6), (7, 8)])
print(f'prefixes = {prefixes}')

# suffixes = {TrieEntry(ident=1, key=[(1, 2), (3, 4), (5, 6)]),
#             TrieEntry(ident=2, key=[(1, 2), (3, 4)])}
# prefixes = {TrieEntry(ident=1, key=[(1, 2), (3, 4), (5, 6)]),
#             TrieEntry(ident=2, key=[(1, 2), (3, 4)])}
