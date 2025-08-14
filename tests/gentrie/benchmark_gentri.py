#!env python3.10
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


def generate_test_data(depth: int, symbols: str) -> list[str]:
    """Generate test data for the Generalized Trie.

    Args:
        depth (int): The depth of the keys to generate.
        symbols (str): The symbols to use in the keys."""
    test_data: list[str] = []
    seen: set[str] = set()
    for key in permutations(symbols, depth):
        key_string = ''.join(key)
        # Avoid duplicates by using a set
        if key_string in seen:
            continue
        seen.add(key_string)
        test_data.append(key_string)
    return test_data


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


if __name__ == '__main__':
    iterations: int = 10  # Number of iterations for each test case
    size: int = 10_000_000  # Number of elements for the tests
    all_results: list[TestResults] = []

    timer_start: int = time.process_time_ns()
    for _ in range(iterations):
        for i in range(size):
            pass
    timer_end: int = time.process_time_ns()
    elapsed: int = timer_end - timer_start
    null_timer = TestResults(name='null_loop',
                             description='Null loop timing',
                             data=[],
                             elapsed=elapsed,
                             n=size,
                             iterations=iterations,
                             per_second=float(iterations * size / (elapsed / 1e9))
                             )
    all_results.append(null_timer)

    default_depth: int = 3  # Default depth for test data generation

    # Time the Generalized Trie add operation with validated keys
    iterations: int = 10
    depth: int = default_depth
    test_data = generate_test_data(depth, SYMBOLS)
    n: int = len(test_data)
    elapsed: int = 0
    for _ in range(iterations):
        trie = GeneralizedTrie()
        timer_start = time.process_time_ns()
        for key in test_data:
            trie.add(key)
        timer_end = time.process_time_ns()
        elapsed += (timer_end - timer_start)
    result = TestResults(
        name='add() (validated keys)',
        description='Timing for GeneralizedTrie.add() with key validation',
        data=test_data,
        elapsed=elapsed,
        n=n,
        iterations=iterations,
        per_second=n * iterations / (elapsed / 1e9)
    )
    all_results.append(result)

    # Time the Generalized Trie add operation with nonvalidated keys
    iterations: int = 10
    depth: int = default_depth
    n: int = len(test_data)
    elapsed: int = 0
    for _ in range(iterations):
        trie = GeneralizedTrie(runtime_validation=False)  # Disable runtime validation for performance testing
        # Note: This is a performance test, so we assume the keys are valid
        # In a real-world scenario, you would want to validate keys before adding them
        timer_start = time.process_time_ns()
        for key in test_data:
            trie.add(key)
        timer_end = time.process_time_ns()
        elapsed += (timer_end - timer_start)
    result = TestResults(
        name='add() (non-validated keys)',
        description='Timing for GeneralizedTrie.add() without key validation',
        data=test_data,
        elapsed=elapsed,
        n=n,
        iterations=iterations,
        per_second=n * iterations / (elapsed / 1e9)
    )
    all_results.append(result)

    # Time the Generalized Trie hash assignment operation with validated keys
    iterations: int = 10
    depth: int = default_depth
    n: int = len(test_data)
    elapsed: int = 0
    for _ in range(iterations):
        trie = GeneralizedTrie()
        timer_start = time.process_time_ns()
        for key in test_data:
            trie[key] = key  # Assign the key to itself
        timer_end = time.process_time_ns()
        elapsed += (timer_end - timer_start)
    result = TestResults(
        name='trie[key] = key (validated keys)',
        description='Timing for trie[key] = key with key validation',
        data=test_data,
        elapsed=elapsed,
        n=n,
        iterations=iterations,
        per_second=n * iterations / (elapsed / 1e9)
    )
    all_results.append(result)

    # Time the Generalized Trie hash assignment operation without validated keys
    iterations: int = 10
    depth: int = default_depth
    n: int = len(test_data)
    elapsed: int = 0
    for _ in range(iterations):
        trie = GeneralizedTrie(runtime_validation=False)  # Disable runtime validation for performance testing
        # Note: This is a performance test, so we assume the keys are valid
        # In a real-world scenario, you would want to validate keys before assigning them
        timer_start = time.process_time_ns()
        for key in test_data:
            trie[key] = key  # Assign the key to itself
        timer_end = time.process_time_ns()
        elapsed += (timer_end - timer_start)
    result = TestResults(
        name='trie[key] = key (non-validated keys)',
        description='Timing for trie[key] = key without key validation',
        data=test_data,
        elapsed=elapsed,
        n=n,
        iterations=iterations,
        per_second=n * iterations / (elapsed / 1e9)
    )
    all_results.append(result)

    # Time the Generalized Trie update operation with validated keys
    iterations: int = 10
    depth: int = default_depth
    n: int = len(test_data)
    elapsed: int = 0
    for _ in range(iterations):
        trie = GeneralizedTrie()
        timer_start = time.process_time_ns()
        for key in test_data:
            trie.add(key)
        timer_end = time.process_time_ns()
        elapsed += (timer_end - timer_start)
    result = TestResults(
        name='update() (validated keys)',
        description='Timing for GeneralizedTrie.update() with key validation',
        data=test_data,
        elapsed=elapsed,
        n=n,
        iterations=iterations,
        per_second=n * iterations / (elapsed / 1e9)
    )
    all_results.append(result)

    # Time the Generalized Trie update operation with nonvalidated keys
    iterations: int = 10
    depth: int = default_depth
    n: int = len(test_data)
    elapsed: int = 0
    for _ in range(iterations):
        trie = GeneralizedTrie(runtime_validation=False)  # Disable runtime validation for performance testing
        # Note: This is a performance test, so we assume the keys are valid
        # In a real-world scenario, you would want to validate keys before adding them
        timer_start = time.process_time_ns()
        for key in test_data:
            trie.add(key)
        timer_end = time.process_time_ns()
        elapsed += (timer_end - timer_start)
    result = TestResults(
        name='update() (non-validated keys)',
        description='Timing for GeneralizedTrie.update() without key validation',
        data=test_data,
        elapsed=elapsed,
        n=n,
        iterations=iterations,
        per_second=n * iterations / (elapsed / 1e9)
    )
    all_results.append(result)

    # Display the results
    for result in all_results:
        # Print the results for each test case
        print("=" * 40)
        print(f"{result.name}: {result.description}")
        print(f"  Key depth: {default_depth}")
        print(f"  Data size: {result.n}, Iterations: {result.iterations}")
        print(f"  Elapsed time: {result.elapsed / 1e9:.2f} seconds")
        print(f"  Operations per second: {result.per_second:.2f}\n")
