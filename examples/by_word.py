#!/usr/bin/env python3
"""Example of using a GeneralizedTrie for indexing sequences of words
"""
from gentrie import GeneralizedTrie, TrieEntry

trie = GeneralizedTrie()
entries: list[list[str]] = [
    ['ape', 'green', 'apple'],
    ['ape', 'green'],
    ['ape', 'green', 'pineapple'],
]
for item in entries:
    trie.add(item)
prefixes: set[TrieEntry] = trie.prefixes(['ape', 'green', 'apple'])
print(f'prefixes = {prefixes}')
suffixes: set[TrieEntry] = trie.suffixes(['ape', 'green'])
print(f'suffixes = {suffixes}')

# prefixes = {
#   TrieEntry(ident=TrieId(1), key=['ape', 'green', 'apple'], value=None),
#   TrieEntry(ident=TrieId(2), key=['ape', 'green'], value=None)
# }
# suffixes = {
#   TrieEntry(ident=TrieId(1), key=['ape', 'green', 'apple'], value=None),
#   TrieEntry(ident=TrieId(2), key=['ape', 'green'], value=None),
#   TrieEntry(ident=TrieId(3), key=['ape', 'green', 'pineapple'], value=None)
# }
