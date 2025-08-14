#!env python3 -m scalene
# -*- coding: utf-8 -*-
"""
Benchmark for the Generalized Trie implementation.
This script runs a series of tests to measure the performance of the Generalized Trie
against a set of predefined test cases.
"""

from dataclasses import dataclass
from itertools import permutations
import time
from typing import NamedTuple, Sequence

from gentrie import GeneralizedTrie, GeneralizedKey


SYMBOLS: str = '0123456789ABCDEFGHIJKLMNIOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'  # Define the symbols for the trie


def generate_test_data(depth: int, symbols: str, max_keys: int) -> list[str]:
    """Generate test data for the Generalized Trie.

    Args:
        depth (int): The depth of the keys to generate.
        symbols (str): The symbols to use in the keys.
        max_keys (int): The maximum number of keys to generate."""
    test_data: list[str] = []
    seen: set[str] = set()
    for key in permutations(symbols, depth):
        key_string = ''.join(key)
        # Avoid duplicates by using a set
        if key_string in seen:
            continue
        seen.add(key_string)
        test_data.append(key_string)
        if len(test_data) >= max_keys:
            break
    return test_data


def generate_test_trie(depth: int, symbols: str, max_keys: int) -> GeneralizedTrie:
    """Generate a test Generalized Trie for the given depth and symbols."""
    test_data = generate_test_data(depth, symbols, max_keys)
    trie = GeneralizedTrie(runtime_validation=False)
    for key in test_data:
        trie[key] = key  # Assign the key to itself
    return trie


class TestCase(NamedTuple):
    name: str
    description: str
    data: Sequence[GeneralizedKey]
    iterations: int = 10


@dataclass
class TestResults:
    name: str
    description: str
    data: Sequence[GeneralizedKey]
    elapsed: int
    n: int
    iterations: int
    per_second: float


def benchmark_null_loop(iterations: int = 10, size: int = 10_000_000) -> TestResults:
    elapsed: int = 0
    for _ in range(iterations):
        timer_start = time.process_time_ns()
        for _ in range(size):
            pass
        timer_end = time.process_time_ns()
        elapsed += (timer_end - timer_start)
    return TestResults(
        name='null_loop',
        description='Timing for a null loop',
        data=[],
        elapsed=elapsed,
        n=size,
        iterations=iterations,
        per_second=float(iterations * size / (elapsed / 1e9))
    )


def benchmark_add_with_validation(test_data: Sequence[GeneralizedKey],
                                  iterations: int,
                                  depth: int) -> TestResults:
    n: int = len(test_data)
    elapsed: int = 0

    for _ in range(iterations):
        trie = GeneralizedTrie()
        timer_start = time.process_time_ns()
        for key in test_data:
            trie.add(key)
        timer_end = time.process_time_ns()
        elapsed += (timer_end - timer_start)
    return TestResults(
        name='add() (validated keys)',
        description='Timing for GeneralizedTrie.add() with key validation',
        data=test_data,
        elapsed=elapsed,
        n=n,
        iterations=iterations,
        per_second=n * iterations / (elapsed / 1e9)
    )


def benchmark_add_without_validation(test_data: Sequence[GeneralizedKey],
                                     iterations: int,
                                     depth: int) -> TestResults:
    n: int = len(test_data)
    elapsed: int = 0

    for _ in range(iterations):
        trie = GeneralizedTrie(runtime_validation=False)
        timer_start = time.process_time_ns()
        for key in test_data:
            trie.add(key)
        timer_end = time.process_time_ns()
        elapsed += (timer_end - timer_start)
    return TestResults(
        name='add() (non-validated keys)',
        description='Timing for GeneralizedTrie.add() without runtime key validation',
        data=test_data,
        elapsed=elapsed,
        n=n,
        iterations=iterations,
        per_second=n * iterations / (elapsed / 1e9)
    )


def benchmark_trie_key_assignment_with_validation(
        test_data: Sequence[GeneralizedKey],
        iterations: int,
        depth: int) -> TestResults:
    n: int = len(test_data)
    elapsed: int = 0
    for _ in range(iterations):
        trie = GeneralizedTrie()
        timer_start = time.process_time_ns()
        for key in test_data:
            trie[key] = key  # Assign the key to itself
        timer_end = time.process_time_ns()
        elapsed += (timer_end - timer_start)
    return TestResults(
        name='trie[key] = key (validated keys)',
        description='Timing for trie[key] = key with key validation',
        data=test_data,
        elapsed=elapsed,
        n=n,
        iterations=iterations,
        per_second=n * iterations / (elapsed / 1e9)
    )


def benchmark_trie_key_assignment_without_validation(
        test_data: Sequence[GeneralizedKey],
        iterations: int,
        depth: int) -> TestResults:
    n: int = len(test_data)
    elapsed: int = 0
    for _ in range(iterations):
        trie = GeneralizedTrie(runtime_validation=False)
        timer_start = time.process_time_ns()
        for key in test_data:
            trie[key] = key  # Assign the key to itself
        timer_end = time.process_time_ns()
        elapsed += (timer_end - timer_start)
    return TestResults(
        name='trie[key] = key (non-validated keys)',
        description='Timing for trie[key] = key without key validation',
        data=test_data,
        elapsed=elapsed,
        n=n,
        iterations=iterations,
        per_second=n * iterations / (elapsed / 1e9)
    )


def benchmark_update_with_validation(test_data: Sequence[GeneralizedKey],
                                     iterations: int = 10,
                                     depth: int = 3) -> TestResults:
    n: int = len(test_data)
    elapsed: int = 0

    for _ in range(iterations):
        trie = GeneralizedTrie()
        timer_start = time.process_time_ns()
        for key in test_data:
            trie.update(key, value=key)  # Update the key with itself as value
        timer_end = time.process_time_ns()
        elapsed += (timer_end - timer_start)
    return TestResults(
        name='update() (validated keys)',
        description='Timing for GeneralizedTrie.update() with key validation',
        data=test_data,
        elapsed=elapsed,
        n=n,
        iterations=iterations,
        per_second=n * iterations / (elapsed / 1e9)
    )


def benchmark_update_without_validation(test_data: Sequence[GeneralizedKey],
                                        iterations: int = 10,
                                        depth: int = 3) -> TestResults:
    n: int = len(test_data)
    elapsed: int = 0

    for _ in range(iterations):
        trie = GeneralizedTrie(runtime_validation=False)
        timer_start = time.process_time_ns()
        for key in test_data:
            trie.update(key, value=key)  # Update the key with itself as value
        timer_end = time.process_time_ns()
        elapsed += (timer_end - timer_start)
    return TestResults(
        name='update() (non-validated keys)',
        description='Timing for GeneralizedTrie.update() without key validation',
        data=test_data,
        elapsed=elapsed,
        n=n,
        iterations=iterations,
        per_second=n * iterations / (elapsed / 1e9)
    )


def benchmark_key_in_trie(
        runtime_validation: bool = True,
        iterations: int = 10,
        depth: int = 3,
        symbols: str = SYMBOLS,
        max_keys: int = 1_000_000) -> TestResults:
    """Benchmark the in operator for GeneralizedTrie."""
    elapsed: int = 0
    trie = generate_test_trie(depth, symbols, max_keys)
    trie.runtime_validation = runtime_validation
    trie_keys: list[GeneralizedKey] = list(entry.key for entry in trie.values())
    n: int = len(trie_keys)

    key: GeneralizedKey
    for _ in range(iterations):
        timer_start = time.process_time_ns()
        for key in trie_keys:
            if key in trie:
                pass  # Just checking membership
            else:
                raise KeyError(f"Key {key} not found in trie")
        timer_end = time.process_time_ns()
        elapsed += (timer_end - timer_start)

    return TestResults(
        name=f'in operator on trie (runtime validation: {runtime_validation})',
        description=f'Timing for checking membership in GeneralizedTrie (runtime validation: {runtime_validation})',
        data=[],
        elapsed=elapsed,
        n=n,
        iterations=iterations,
        per_second=n * iterations / (elapsed / 1e9)
    )


if __name__ == '__main__':
    default_depth: int = 15  # Default depth for test data generation
    default_max_keys: int = 1_000_000  # Default maximum number of keys to generate
    default_iterations: int = 1  # Number of iterations for each test case
    default_size: int = 10_000_000  # Number of elements for the tests
    default_test_data = generate_test_data(default_depth, SYMBOLS, default_max_keys)

    all_results: list[TestResults] = []

    all_results.append(benchmark_key_in_trie(
                        runtime_validation=True,
                        iterations=default_iterations,
                        depth=default_depth,
                        symbols=SYMBOLS,
                        max_keys=default_max_keys))
    all_results.append(benchmark_key_in_trie(
                        runtime_validation=False,
                        iterations=default_iterations,
                        depth=default_depth,
                        symbols=SYMBOLS,
                        max_keys=default_max_keys))

    # all_results.append(benchmark_null_loop(iterations=default_iterations, size=default_size))
    # all_results.append(benchmark_add_with_validation(test_data=default_test_data,
    #                                                  iterations=default_iterations,
    #                                                  depth=default_depth))
    # all_results.append(benchmark_add_without_validation(test_data=default_test_data,
    #                                                     iterations=default_iterations,
    #                                                     depth=default_depth))

    # all_results.append(benchmark_trie_key_assignment_with_validation(test_data=default_test_data,
    #                                                                  iterations=default_iterations,
    #                                                                  depth=default_depth))

    # all_results.append(benchmark_trie_key_assignment_without_validation(test_data=default_test_data,
    #                                                                     iterations=default_iterations,
    #                                                                     depth=default_depth))

    # all_results.append(benchmark_update_with_validation(test_data=default_test_data,
    #                                                    iterations=default_iterations,
    #                                                     depth=default_depth))

    # all_results.append(benchmark_update_without_validation(test_data=default_test_data,
    #                                                        iterations=default_iterations,
    #                                                       depth=default_depth))

    # Display the results
    for result in all_results:
        # Print the results for each test case
        print("=" * 40)
        print(f"{result.name}: {result.description}")
        print(f"  Key depth: {default_depth}")
        print(f"  Data size: {result.n}, Iterations: {result.iterations}")
        print(f"  Elapsed time: {result.elapsed / 1e9:.2f} seconds")
        print(f"  Operations per second: {result.per_second:.2f}\n")
