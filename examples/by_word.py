#!/usr/bin/env python3

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

# prefixes = {TrieEntry(ident=1, key=['ape', 'green', 'apple'], value=None),
#             TrieEntry(ident=2, key=['ape', 'green'])}
# suffixes = {TrieEntry(ident=1, key=['ape', 'green', 'apple'], value=None),
#             TrieEntry(ident=3, key=['ape', 'green', 'pineapple'], value=None),
#             TrieEntry(ident=2, key=['ape', 'green'], value=None)}
