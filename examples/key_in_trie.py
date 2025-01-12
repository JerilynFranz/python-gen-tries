#!/usr/bin/env python3

from gentrie import GeneralizedTrie

trie = GeneralizedTrie()
entries: list[str] = [
    'abcdef',
    'abc',
    'abcd',
    'qrf',
]
for item in entries:
    trie.add(item)

if 'abc' in trie:
    print('abc is in trie')
else:
    print('error: abc is not in trie')

if 'abcde' not in trie:
    print('abcde is not in trie')
else:
    print('error: abcde is in trie')

if 'qrf' not in trie:
    print('error: qrf is not in trie')
else:
    print('qrf is in trie')

if 'abcdef' not in trie:
    print('error: abcdef is not in trie')
else:
    print('abcdef is in trie')

# abc is in trie
# abcde is not in trie
# qrf is in trie
# abcdef is in trie
