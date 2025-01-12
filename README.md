# gen-tries

## Name

gen-tries

## Description

A generalized trie implementation for python 3.10 or later that provides classes and
functions to create and manipulate a generalized trie data structure. 

Unlike many Trie implementations which only support strings as keys
and token match only at the character level, it is agnostic as to the
types of tokens used to key it and thus far more general purpose.

It requires only that the indexed tokens be hashable (this means that the have
__eq__ and __hash__ methods). This is verified at runtime using the `gentrie.Hashable` protocol.

Tokens in a key do NOT have to all be the same type as long as they
can be compared for equality.

Note that objects of user-defined classes are Hashable by default, but this
may not work as naively expected unless they are immutable.

It can handle `Sequence`s of `Hashable` conforming objects as keys
for the trie out of the box.

As long as the tokens returned by a sequence are hashable, it largely 'just works'.

You can 'mix and match' types of objects used as token in a key as
long as they all conform to the `Hashable` protocol.


## Usage

### Example 1:
```python
from gentrie import GeneralizedTrie, TrieEntry

trie = GeneralizedTrie()
trie.add(['ape', 'green', 'apple'])
trie.add(['ape', 'green'])
matches: set[TrieEntry] = trie.prefixes(['ape', 'green'])
print(matches)
```

### Example 1 Output:
```python
{TrieEntry(ident=2, key=['ape', 'green'])}}
```

### Example 2:
```python
from gentrie import GeneralizedTrie, TrieEntry

# Create a trie to store website URLs
url_trie = GeneralizedTrie()

# Add some URLs with different components (protocol, domain, path)
url_trie.add(["https", "com", "example", "www", "/", "products", "clothing"])
url_trie.add(["http", "org", "example", "blog", "/", "2023", "10", "best-laptops"])
url_trie.add(["ftp", "net", "example", "ftp", "/", "data", "images"])

# Find all https URLs with "example.com" domain
suffixes: set[TrieEntry] = url_trie.suffixes(["https", "com", "example"])
print(suffixes)
```

### Example 2 Output:
```python
{TrieEntry(ident=1, key=['https', 'com', 'example', 'www', '/', 'products', 'clothing'])}
```

### Example 3:
```python
from gentrie import GeneralizedTrie, TrieEntry

trie = GeneralizedTrie()
trie.add('abcdef')
trie.add('abc')
trie.add('qrf')
matches: set[TrieEntry] = trie.suffixes('ab')
print(matches)
```

### Example 3 Output:
```python
{TrieEntry(ident=2, key='abc'), TrieEntry(ident=1, key='abcdef')}
```
## Authors and acknowledgment

- Jerilyn Franz

## Copyright

Copyright 2025 Jerilyn Franz

## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
