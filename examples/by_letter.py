#!/usr/bin/env python3

from gentrie import GeneralizedTrie, TrieEntry

trie = GeneralizedTrie()
entries: list[str] = [
    'abcdef',
    'abc',
    'abcd',
    'qrf',
]
for item in entries:
    trie.add(item)

suffixes: set[TrieEntry] = trie.suffixes('abcd')
print(f'suffixes = {suffixes}')

prefixes: set[TrieEntry] = trie.prefixes('abcdefg')
print(f'prefixes = {prefixes}')

# suffixes = {TrieEntry(ident=1, key='abcdef'),
#             TrieEntry(ident=3, key='abcd')}
# prefixes = {TrieEntry(ident=1, key='abcdef'),
#             TrieEntry(ident=3, key='abcd'),
#             TrieEntry(ident=2, key='abc')}
