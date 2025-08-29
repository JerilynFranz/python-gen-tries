#!env python3
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, too-many-instance-attributes, line-too-long
# pylint: disable=too-many-positional-arguments, too-many-arguments, too-many-locals
# pyright: reportUnnecessaryTypeIgnoreComment=warning

# Note: pytest-benchmark does not expose proper typing hence the many type
# ignores on lines related to benchmark and benchmark.info
'''bench_performance.py

Benchmark for the Generalized Trie implementation.

This script runs a series of tests to measure the performance of the Generalized Trie
against a set of predefined test cases.


See the documentation for pytest-benchmark at https://pytest-benchmark.readthedocs.io/
for more information on how to use it.
'''
import gc
import gzip
import itertools
from pathlib import Path
import time
import sys
from typing import Any, cast, Optional, Sequence

import pytest

from gentrie import GeneralizedTrie, GeneralizedKey

# More robust benchmark configuration
BENCHMARK_CONFIG: dict[str, Any] = {
    'warmup': True,
    'min_rounds': 100,
    'min_time': 1,
    'max_time': 10,
    'timer': time.perf_counter_ns
}

# Apply to all benchmarks
#  pytestmark = pytest.mark.benchmark(**BENCHMARK_CONFIG)

SYMBOLS: str = '0123'  # Define the symbols for the trie


def generate_test_data(depth: int, symbols: str, max_keys: int) -> list[str]:
    '''Generate test data for the Generalized Trie.

    Args:
        depth (int): The depth of the keys to generate.
        symbols (str): The symbols to use in the keys.
        max_keys (int): The maximum number of keys to generate.'''
    test_data: list[str] = []
    for key in itertools.product(symbols, repeat=depth):
        key_string = ''.join(key)
        test_data.append(key_string)
        if len(test_data) >= max_keys:
            break
    return test_data


TEST_DATA: dict[int, list[str]] = {}
TEST_DEPTHS: list[int] = [2, 3, 4, 5, 6, 7, 8, 9]  # Depths to test - '1' is generally omitted due to low key count
TEST_MAX_KEYS: int = len(SYMBOLS) ** max(TEST_DEPTHS)  # Limit to a manageable number of keys
for gen_depth in TEST_DEPTHS:
    max_keys_for_depth = len(SYMBOLS) ** gen_depth  # pylint: disable=invalid-name
    TEST_DATA[gen_depth] = generate_test_data(gen_depth, SYMBOLS, max_keys=max_keys_for_depth)


def generate_test_trie(depth: int, symbols: str, max_keys: int, value: Optional[Any] = None) -> GeneralizedTrie:
    '''Generate a test Generalized Trie for the given depth and symbols.

    Args:
        depth (int): The depth of the trie.
        symbols (str): The symbols to use in the trie.
        max_keys (int): The maximum number of keys to generate.
        value (Optional[Any]): The value to assign to each key in the trie.
    '''
    test_data = generate_test_data(depth, symbols, max_keys)
    trie = GeneralizedTrie(runtime_validation=False)

    for key in test_data:
        trie[key] = value
    return trie


def generate_test_trie_from_data(data: Sequence[GeneralizedKey], value: Optional[Any] = None) -> GeneralizedTrie:
    '''Generate a test Generalized Trie from the passed Sequence of GeneralizedKey.

    Args:
        data (Sequence[GeneralizedKey]): The sequence of keys to insert into the trie.
        value (Optional[Any]): The value to assign to each key in the trie.
    '''
    trie = GeneralizedTrie(runtime_validation=False)
    for key in data:
        trie[key] = value
    return trie


# We generate the TEST_TRIES from the TEST_DATA for synchronization
TEST_TRIES: dict[int, GeneralizedTrie] = {}
for gen_depth in TEST_DEPTHS:
    TEST_TRIES[gen_depth] = generate_test_trie_from_data(TEST_DATA[gen_depth], None)


def generate_trie_with_missing_key_from_data(
        test_data: Sequence[GeneralizedKey], value: Optional[Any] = None) -> tuple[GeneralizedTrie, Any]:
    """Generate a GeneralizedTrie and a key that is not in the trie.

    The generated trie will contain all keys from the test_data except for the last one.

    Args:
    test_data: The test data to populate the trie.
    value: The value to associate with the keys in the trie.
    """
    trie = generate_test_trie_from_data(data=test_data, value=value)
    missing_key = test_data[-1]  # Use the last key as the missing key
    trie.remove(missing_key)  # Ensure the key is not actually in the trie
    return trie, missing_key


# We generate the TEST_MISSING_KEY_TRIES from the TEST_DATA for synchronization
TEST_MISSING_KEY_TRIES: dict[int, tuple[GeneralizedTrie, str]] = {}
for gen_depth in TEST_DEPTHS:
    TEST_MISSING_KEY_TRIES[gen_depth] = generate_trie_with_missing_key_from_data(TEST_DATA[gen_depth], None)


def generate_fully_populated_trie(max_depth: int, value: Optional[Any] = None) -> GeneralizedTrie:
    '''Generate a fully populated Generalized Trie for the given max_depth.

    A fully populated trie contains all possible keys up to the specified depth.
    It uses the pregenerated TEST_DATA as the source of truth for the keys for each depth
    because it contains all the possible keys for the depth and symbol set.

    Args:
        max_depth (int): The maximum depth of the trie.
        value (Optional[Any], default=None): The value to assign to each key in the trie.
    '''
    trie = GeneralizedTrie(runtime_validation=False)
    # Use precomputed TEST_DATA if available for performance
    for depth, data in TEST_DATA.items():
        if depth <= max_depth:
            for key in data:
                trie[key] = value

    # Generate any requested depths NOT included in TEST_DATA
    for depth in range(1, max_depth + 1):
        if depth not in TEST_DATA:
            # Generate all possible keys for this depth
            for key in generate_test_data(depth, SYMBOLS, len(SYMBOLS) ** depth):
                trie[key] = value

    return trie


TEST_FULLY_POPULATED_TRIES: dict[int, GeneralizedTrie] = {}
for gen_depth in TEST_DEPTHS:
    TEST_FULLY_POPULATED_TRIES[gen_depth] = generate_fully_populated_trie(max_depth=gen_depth)


def english_words():
    """Imports English words from a gzipped text file.

    The file contains a bit over 278 thousand words in English
    (one per line).
    """
    words_file = Path(__file__).parent.joinpath("english_words.txt.gz")
    return list(map(str.rstrip, gzip.open(words_file, "rt")))


TEST_ENGLISH_WORDS: list[str] = english_words()
TEST_ENGLISH_WORDS_TRIE: GeneralizedTrie = generate_test_trie_from_data(TEST_ENGLISH_WORDS, None)


@pytest.mark.benchmark(group="Build trie from English wordset using update()", **BENCHMARK_CONFIG)
@pytest.mark.parametrize('runtime_validation', [False, True])
def test_organic_build_with_update_from_english_words_list(
       benchmark,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
       runtime_validation: bool):
    '''Benchmark the adding of a list of english words to the trie using update()

    This test checks the performance of adding keys from a list of english words to the trie.
    '''
    def helper_create_dictionary(words: Sequence[str], runtime_validation: bool) -> GeneralizedTrie:
        trie = GeneralizedTrie(runtime_validation=runtime_validation)
        for word in words:
            trie[word] = None
        return trie

    gc.collect()
    benchmark.extra_info['number_of_words'] = len(TEST_ENGLISH_WORDS)  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['runtime_validation'] = runtime_validation  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['average_word_length'] = (  # pyright: ignore[reportUnknownMemberType]
        sum(len(word) for word in TEST_ENGLISH_WORDS) / len(TEST_ENGLISH_WORDS) if TEST_ENGLISH_WORDS else 0)
    benchmark(helper_create_dictionary,
              words=TEST_ENGLISH_WORDS,
              runtime_validation=runtime_validation)


@pytest.mark.benchmark(group="Microbenchmark update() building trie from English words", **BENCHMARK_CONFIG)
@pytest.mark.parametrize('runtime_validation', [False, True])
def test_microbenchmarking_update_for_build_from_english_words_list(
       benchmark,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
       runtime_validation: bool):
    '''Benchmark the adding of keys to the trie using update()

    This test checks the performance of adding keys to the trie using update()
    with a list of English words but in a micro-benchmarking context (isolating
    the update() calls for performance testing rather than benchmarking the entire
    process).
    '''
    benchmark_trie: GeneralizedTrie = GeneralizedTrie(runtime_validation=runtime_validation)
    key_iter = iter(TEST_ENGLISH_WORDS)

    def setup():
        return (), {'key': next(key_iter)}  # Will crash when exhausted
    rounds = len(TEST_ENGLISH_WORDS)  # Rounds limited to prevent exhaustion

    gc.collect()
    benchmark.extra_info['number_of_words'] = len(TEST_ENGLISH_WORDS)  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['runtime_validation'] = runtime_validation  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['average_word_length'] = (  # pyright: ignore[reportUnknownMemberType]
        sum(len(word) for word in TEST_ENGLISH_WORDS) / len(TEST_ENGLISH_WORDS) if TEST_ENGLISH_WORDS else 0)

    benchmark.pedantic(benchmark_trie.update,  # pyright: ignore[reportUnknownMemberType]
                       setup=setup,
                       rounds=rounds,
                       iterations=1)


@pytest.mark.benchmark(group="Microbenchmark update() building trie from synthetic data",
                       **BENCHMARK_CONFIG)
@pytest.mark.parametrize('runtime_validation', [False, True])
@pytest.mark.parametrize('depth', TEST_DEPTHS)
def test_build_with_update(
       benchmark,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
       runtime_validation: bool,
       depth: int):
    '''Benchmark the adding of keys to the trie using update()

    This test checks the performance of adding keys to the trie using update().
    '''
    benchmark_trie: GeneralizedTrie = GeneralizedTrie(runtime_validation=runtime_validation)
    key_iter = iter(TEST_DATA[depth])

    def setup():
        return (), {'key': next(key_iter)}  # Will crash when exhausted
    rounds = len(TEST_DATA[depth])  # Rounds limited to prevent exhaustion

    gc.collect()
    benchmark.extra_info['depth'] = depth  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['number_of_words'] = len(TEST_DATA[depth])  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['runtime_validation'] = runtime_validation  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['average_word_length'] = (  # pyright: ignore[reportUnknownMemberType]
        sum(len(word) for word in TEST_DATA[depth]) / len(TEST_DATA[depth]) if TEST_DATA[depth] else 0)
    benchmark.pedantic(benchmark_trie.update,  # pyright: ignore[reportUnknownMemberType]
                       setup=setup,
                       rounds=rounds,
                       iterations=1)


@pytest.mark.benchmark(group="Microbenchmark add() building trie from synthetic data",
                       **BENCHMARK_CONFIG)
@pytest.mark.parametrize('runtime_validation', [False, True])
@pytest.mark.parametrize('depth', TEST_DEPTHS)
def test_build_with_add(
       benchmark,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
       runtime_validation: bool,
       depth: int):
    '''Benchmark the adding of keys to the trie using add()

    This test checks the performance of adding keys to the trie using the add() method.
    '''
    benchmark_trie: GeneralizedTrie = GeneralizedTrie(runtime_validation=runtime_validation)
    key_iter = iter(TEST_DATA[depth])

    def setup():
        return (), {'key': next(key_iter)}  # Will crash when exhausted
    rounds = len(TEST_DATA[depth])

    gc.collect()
    benchmark.extra_info['depth'] = depth  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['number_of_words'] = len(TEST_DATA[depth])  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['runtime_validation'] = runtime_validation  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['average_word_length'] = (  # pyright: ignore[reportUnknownMemberType]
        sum(len(word) for word in TEST_DATA[depth]) / len(TEST_DATA[depth]) if TEST_DATA[depth] else 0)
    benchmark.pedantic(benchmark_trie.add,  # pyright: ignore[reportUnknownMemberType]
                       setup=setup,
                       rounds=rounds,
                       iterations=1)


@pytest.mark.benchmark(group="Microbenchmark update() updating trie from synthetic data",
                       **BENCHMARK_CONFIG)
@pytest.mark.parametrize('runtime_validation', [False, True])
@pytest.mark.parametrize('depth', TEST_DEPTHS)
def test_updating_trie(
        benchmark,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        runtime_validation: bool,
        depth: int):
    '''Benchmark the update value for a key operation on a populated trie.

    This test checks the performance of updating keys in the trie.
    '''
    benchmark_trie = TEST_TRIES[depth]
    benchmark_trie.runtime_validation = runtime_validation
    benchmark_key: str = TEST_DATA[depth][0]  # Use the first key for benchmarking
    # for idempotency we reuse the orignal value for the updated value
    benchmark_value: Any = benchmark_trie[benchmark_key]

    gc.collect()
    benchmark(benchmark_trie.update, benchmark_key, benchmark_value)


@pytest.mark.benchmark(group="Microbenchmark '__contains__()' method using synthetic data",
                       **BENCHMARK_CONFIG)
@pytest.mark.parametrize('runtime_validation', [False, True])
@pytest.mark.parametrize('depth', TEST_DEPTHS)
def test_key_in_trie(benchmark,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
                     runtime_validation: bool,
                     depth: int) -> None:
    '''Benchmark using keys with the in operator for GeneralizedTrie.

    This test checks the performance of key lookups in the trie using the in operator.
    '''
    benchmark_trie: GeneralizedTrie = TEST_TRIES[depth]
    benchmark_key: str = TEST_DATA[depth][-1]  # Use the last key for benchmarking
    gc.collect()
    benchmark_trie.runtime_validation = runtime_validation
    benchmark.extra_info['depth'] = depth  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['number_of_words'] = len(TEST_DATA[depth])  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['runtime_validation'] = runtime_validation  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['average_word_length'] = (  # pyright: ignore[reportUnknownMemberType]
        sum(len(word) for word in TEST_DATA[depth]) / len(TEST_DATA[depth]) if TEST_DATA[depth] else 0)
    benchmark(benchmark_trie.__contains__, benchmark_key)


@pytest.mark.benchmark(group="Microbenchmark __contains__() for missing keys using synthetic data", **BENCHMARK_CONFIG)
@pytest.mark.parametrize('runtime_validation', [False, True])
@pytest.mark.parametrize('depth', TEST_DEPTHS)
def test_key_not_in_trie(benchmark,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
                         runtime_validation: bool,
                         depth: int) -> None:
    '''Benchmark missing keys with the in operator for GeneralizedTrie.

    This test checks the performance of missing key lookups in the trie using the in operator.
    '''
    benchmark_trie, missing_key = TEST_MISSING_KEY_TRIES[depth]
    benchmark_trie.runtime_validation = runtime_validation

    gc.collect()
    benchmark.extra_info['depth'] = depth  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['number_of_words'] = len(TEST_DATA[depth])  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['runtime_validation'] = runtime_validation  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['average_word_length'] = (  # pyright: ignore[reportUnknownMemberType]
        sum(len(word) for word in TEST_DATA[depth]) / len(TEST_DATA[depth]) if TEST_DATA[depth] else 0)
    benchmark(benchmark_trie.__contains__, missing_key)


@pytest.mark.benchmark(group="Microbenchmark remove() method using synthetic data", **BENCHMARK_CONFIG)
@pytest.mark.parametrize('runtime_validation', [False, True])
@pytest.mark.parametrize('depth', TEST_DEPTHS)
def test_remove_key_from_trie(benchmark,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
                              runtime_validation: bool,
                              depth: int) -> None:
    '''Benchmark remove() method for GeneralizedTrie.

    This test checks the performance of the remove() method.
    '''
    # Generate a NEW GeneralizedTrie from the test data to keep from corrupting the
    # pre-built test tries with the deletions.
    test_data = TEST_DATA[depth]
    benchmark_trie: GeneralizedTrie = generate_test_trie_from_data(data=test_data, value=None)
    benchmark_trie.runtime_validation = runtime_validation
    key_iter = iter(test_data)

    def setup():
        return (), {'key': next(key_iter)}  # Will crash when exhausted
    rounds = len(test_data)  # Rounds limited to prevent exhaustion

    gc.collect()
    benchmark.extra_info['depth'] = depth  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['number_of_words'] = len(TEST_DATA[depth])  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['runtime_validation'] = runtime_validation  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['average_word_length'] = (  # pyright: ignore[reportUnknownMemberType]
        sum(len(word) for word in test_data) / len(test_data) if test_data else 0)
    benchmark.pedantic(benchmark_trie.remove,  # pyright: ignore[reportUnknownMemberType]
                       setup=setup,
                       rounds=rounds)


@pytest.mark.benchmark(group="Microbenchmark get() for trie from synthetic data",
                       **BENCHMARK_CONFIG)
@pytest.mark.parametrize('runtime_validation', [False, True])
@pytest.mark.parametrize('depth', TEST_DEPTHS)
def test_get(benchmark,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
             runtime_validation: bool,
             depth: int) -> None:
    '''Benchmark get() method for GeneralizedTrie.

    This test checks the performance of the get() method.
    '''
    test_data = TEST_DATA[depth]
    benchmark_trie: GeneralizedTrie = TEST_TRIES[depth]
    benchmark_trie.runtime_validation = runtime_validation
    key_iter = iter(test_data)

    def setup():
        return (), {'key': next(key_iter)}  # Will crash when exhausted
    rounds = len(test_data)  # Rounds limited to prevent exhaustion

    gc.collect()
    benchmark.extra_info['depth'] = depth  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['number_of_words'] = len(TEST_DATA[depth])  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['runtime_validation'] = runtime_validation  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['average_word_length'] = (  # pyright: ignore[reportUnknownMemberType]
        sum(len(word) for word in test_data) / len(test_data) if test_data else 0)
    benchmark.pedantic(benchmark_trie.get,  # pyright: ignore[reportUnknownMemberType]
                       setup=setup,
                       rounds=rounds)


@pytest.mark.benchmark(group="Microbenchmark prefixes() using synthetic data", **BENCHMARK_CONFIG)
@pytest.mark.parametrize('runtime_validation', [False, True])
@pytest.mark.parametrize('depth', [3, 4, 5, 6, 7, 8, 9])
def test_prefixes(benchmark,  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
                  runtime_validation: bool,
                  depth: int):
    """Benchmark trie prefixes() method.

    This test checks the performance of the prefixes() method on fully populated tries.
    Because the potential number of matching keys in the trie increases linearly with depth
    and the full runtime of a prefix search is dominated by the number of keys found,
    this test aims to measure the impact of this growth on the performance of the prefixes()
    method.

    Args:
        runtime_validation (bool): Whether to enable runtime validation.
        depth (int): The depth of the trie to test.
    """
    trie = TEST_FULLY_POPULATED_TRIES[depth]
    trie.runtime_validation = runtime_validation
    search_key = TEST_DATA[depth][0]

    def helper_prefixes(trie: GeneralizedTrie, search_key: GeneralizedKey) -> list[GeneralizedKey]:
        return list(trie.prefixes(search_key))

    gc.collect()
    # The helper function is needed because prefixes() is a generator
    # which requires iteration to access all results.
    # This results in a list of all matched keys being created
    # with additional overhead vs the generator approach.

    results: list[GeneralizedKey] = cast(list[GeneralizedKey], benchmark(helper_prefixes, trie, search_key))
    benchmark.extra_info['number_of_matches_per_query'] = len(results)  # pyright: ignore
    benchmark.extra_info['keys_in_trie'] = len(trie)  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['depth'] = depth  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['runtime_validation'] = runtime_validation  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.benchmark(group="Microbenchmark prefixed_by() using synthetic data",
                       **BENCHMARK_CONFIG)
@pytest.mark.parametrize('runtime_validation', [False, True])
@pytest.mark.parametrize('trie_depth', [7])
@pytest.mark.parametrize('key_depth', [2, 3, 4])  # Focus on manageable depths
@pytest.mark.parametrize('search_depth', [1, 2, 3])  # Focus on manageable depths
def test_prefixed_by(benchmark,  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
                     runtime_validation: bool,
                     trie_depth: int,
                     key_depth: int,
                     search_depth: int):
    """Benchmark trie prefixed_by() method.

    This test checks the performance of the prefixed_by() method on fully populated tries.

    prefixed_by() finds all keys in the trie that are prefixed by a given key up to a specified search depth.

    Args:
        runtime_validation (bool): Whether to enable runtime validation.
        trie_depth (int): The depth of the trie to test.
        key_depth (int): The depth of the key to test.
        search_depth (int): The depth to search for prefixed keys starting from key_depth.
    """
    trie = TEST_FULLY_POPULATED_TRIES[trie_depth]
    trie.runtime_validation = runtime_validation

    # Use a prefix that matches multiple keys
    search_key = TEST_DATA[key_depth][-1]  # last key of the key_depth

    def helper_prefixed_by(trie: GeneralizedTrie,
                           search_key: GeneralizedKey,
                           search_depth: int) -> list[GeneralizedKey]:
        return list(trie.prefixed_by(search_key, search_depth))

    gc.collect()
    # The helper function is needed because prefixed_by() is a generator
    # which requires iteration to access all results.
    # This results in a list of all matched keys being created
    # with additional overhead vs the generator approach.
    results: list[GeneralizedKey] = cast(list[GeneralizedKey],
                                         benchmark(helper_prefixed_by, trie, search_key, search_depth))
    benchmark.extra_info['number_of_matches_per_query'] = len(results)  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['keys_in_trie'] = len(trie)  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['trie_depth'] = trie_depth  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['search_depth'] = search_depth  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['key_depth'] = key_depth  # pyright: ignore[reportUnknownMemberType]
    benchmark.extra_info['runtime_validation'] = runtime_validation  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    sys.exit(pytest.main([
        __file__,
        '--benchmark-columns=ops,rounds,iterations',
        '--benchmark-sort=name',
        '--benchmark-group-by=group',
        '--benchmark-histogram=histogram/benchmark',
        '--benchmark-time-unit=ns',
        '--benchmark-timer=time.perf_counter_ns',
        '--benchmark-save=gentri',
        # '--benchmark-save-data'
        ],
        plugins=[]))
