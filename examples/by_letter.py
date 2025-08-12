#!/usr/bin/env python3
"""Example of using a GeneralizedTrie for 'by letter' string indexing.
"""

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

# suffixes = {
#   TrieEntry(ident=TrieId(1), key='abcdef', value=None),
#   TrieEntry(ident=TrieId(3), key='abcd', value=None)}
# prefixes = {
#   TrieEntry(ident=TrieId(2), key='abc', value=None),
#   TrieEntry(ident=TrieId(3), key='abcd', value=None),
#   TrieEntry(ident=TrieId(1), key='abcdef', value=None)}
