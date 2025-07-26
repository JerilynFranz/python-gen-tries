#!/usr/bin/env python3

from gentrie import GeneralizedTrie, TrieEntry

trie = GeneralizedTrie()
entries: list[str] = [
    'hell',
    'hello',
    'help',
    'do',
    'dog',
    'doll',
    'dolly',
    'dolphin',
    'do'
]
for item in entries:
    trie.add(item)

suggestions: set[TrieEntry] = trie.suffixes('do', depth=2)
print(f'+2 letter suggestions for "do" = {suggestions}')

suggestions = trie.suffixes('do', depth=3)
print(f'+3 letter suggestions for "do" = {suggestions}')

# +2 letter suggestions for "do" = {
#     TrieEntry(ident=6, key='doll', value=None),
#     TrieEntry(ident=5, key='dog', value=None),
#     TrieEntry(ident=4, key='do', value=None)}
#
# +3 letter suggestions for "do" = {
#     TrieEntry(ident=6, key='doll', value=None),
#     TrieEntry(ident=5, key='dog', value=None),
#     TrieEntry(ident=4, key='do', value=None),
#     TrieEntry(ident=7, key='dolly', value=None)}
