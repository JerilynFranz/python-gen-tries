#!env python3
# -*- coding: utf-8 -*-
'''
Benchmark for the Generalized Trie implementation.
This script runs a series of tests to measure the performance of the Generalized Trie
against a set of predefined test cases.
'''
# pylint: disable=wrong-import-position, too-many-instance-attributes
# pylint: disable=too-many-positional-arguments, too-many-arguments, too-many-locals


import gc
import itertools
import time
from typing import Any


import sys
from pathlib import Path
sys.path.insert(0, str(Path('../src').resolve()))

import pytest  # noqa: E402

from gentrie import GeneralizedTrie  # noqa: E402


# More robust benchmark configuration
BENCHMARK_CONFIG: dict[str, Any] = {
    'warmup': True,
    'warmup_iterations': 3,
    'max_time': 10,
    'min_rounds': 5,
    'timer': time.process_time_ns  # More precise timing
}

# Apply to all benchmarks
pytestmark = pytest.mark.benchmark(**BENCHMARK_CONFIG)

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
TEST_DEPTHS: list[int] = [2, 3, 4, 5, 6, 7, 8, 9]
TEST_MAX_KEYS: int = len(SYMBOLS) ** max(TEST_DEPTHS)  # Limit to a manageable number of keys
for gen_depth in TEST_DEPTHS:
    max_keys_for_depth = len(SYMBOLS) ** gen_depth
    TEST_DATA[gen_depth] = generate_test_data(gen_depth, SYMBOLS, max_keys=max_keys_for_depth)

TEST_TRIES: dict[int, GeneralizedTrie] = {}
for gen_depth in TEST_DEPTHS:
    TEST_TRIES[gen_depth] = GeneralizedTrie(runtime_validation=True)

    for gen_key in TEST_DATA[gen_depth]:
        TEST_TRIES[gen_depth][gen_key] = gen_key  # Assign the key to itself


def generate_test_trie(depth: int, symbols: str, max_keys: int) -> GeneralizedTrie:
    '''Generate a test Generalized Trie for the given depth and symbols.'''
    test_data = generate_test_data(depth, symbols, max_keys)
    trie = GeneralizedTrie(runtime_validation=False)

    for key in test_data:
        trie[key] = key  # Assign the key to itself
    return trie


@pytest.mark.benchmark(warmup=True, warmup_iterations=1, max_time=5)
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
    benchmark.pedantic(benchmark_trie.update,  # pyright: ignore[reportUnknownMemberType]
                       setup=setup,
                       rounds=rounds,
                       iterations=1)


@pytest.mark.benchmark(warmup=True, warmup_iterations=1, max_time=5)
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
    benchmark.pedantic(benchmark_trie.add,  # pyright: ignore[reportUnknownMemberType]
                       setup=setup,
                       rounds=rounds,
                       iterations=1)


@pytest.mark.benchmark(warmup=True, warmup_iterations=1, max_time=5)
@pytest.mark.parametrize('runtime_validation', [False, True])
@pytest.mark.parametrize('depth', TEST_DEPTHS)
def test_updating_trie(
        benchmark,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        runtime_validation: bool,
        depth: int):
    '''Benchmark the update value for a key operation on a populated trie.

    This test checks the performance of updating keys in the trie.
    '''
    benchmark_trie: GeneralizedTrie = GeneralizedTrie(runtime_validation=True)
    for key in TEST_DATA[depth]:
        benchmark_trie.add(key, value=None)
    benchmark_key: str = TEST_DATA[depth][0]  # Use the first key for benchmarking
    gc.collect()
    benchmark_trie.runtime_validation = runtime_validation
    benchmark(benchmark_trie.update, benchmark_key, 1)


@pytest.mark.benchmark(warmup=True, warmup_iterations=1, max_time=5)
@pytest.mark.parametrize('runtime_validation', [False, True])
@pytest.mark.parametrize('depth', TEST_DEPTHS)
def test_key_in_trie(benchmark,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
                     runtime_validation: bool,
                     depth: int) -> None:
    '''Benchmark using keys with the in operator for GeneralizedTrie.

    This test checks the performance of key lookups in the trie using the in operator.
    '''
    benchmark_trie: GeneralizedTrie = TEST_TRIES[depth]
    benchmark_key: str = TEST_DATA[depth][0]  # Use the first key for benchmarking
    gc.collect()
    benchmark_trie.runtime_validation = runtime_validation
    benchmark(benchmark_trie.__contains__, benchmark_key)


@pytest.mark.benchmark(warmup=True, warmup_iterations=1, max_time=5)
@pytest.mark.parametrize('runtime_validation', [False, True])
@pytest.mark.parametrize('depth', TEST_DEPTHS)
def test_key_not_in_trie(benchmark,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
                         runtime_validation: bool,
                         depth: int) -> None:
    '''Benchmark missing keys with the in operator for GeneralizedTrie.

    This test checks the performance of missing key lookups in the trie using the in operator.
    '''
    benchmark_trie: GeneralizedTrie = GeneralizedTrie(runtime_validation=True)
    for key in TEST_DATA[depth]:
        benchmark_trie.add(key, value=None)
    benchmark_key: str = TEST_DATA[depth][0]  # Use the first key for benchmarking
    benchmark_trie.remove(benchmark_key)  # Ensure the key is not in the trie
    gc.collect()
    benchmark_trie.runtime_validation = runtime_validation
    benchmark(benchmark_trie.__contains__, benchmark_key)


@pytest.mark.benchmark(**BENCHMARK_CONFIG)
@pytest.mark.parametrize('runtime_validation', [False, True])
@pytest.mark.parametrize('depth', TEST_DEPTHS[4:8])  # Focus on larger tries
def test_traversal_performance(benchmark,  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
                               runtime_validation: bool,
                               depth: int):
    """Benchmark trie traversal operations."""
    trie = TEST_TRIES[depth]
    trie.runtime_validation = runtime_validation

    # Use a prefix that matches multiple keys
    prefix_key = TEST_DATA[depth][0][:max(1, depth-2)]

    gc.collect()
    benchmark(lambda: list(trie.prefixed_by(prefix_key)))
