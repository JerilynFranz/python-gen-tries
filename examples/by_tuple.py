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
matches: set[TrieEntry] = trie.suffixes([(1, 2)])
print(matches)

# {TrieEntry(ident=1, key=[(1, 2), (3, 4), (5, 6)]), TrieEntry(ident=2, key=[(1, 2), (3, 4)])}
