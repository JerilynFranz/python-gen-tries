#!env python3
# -*- coding: utf-8 -*-
'''
Benchmark for the Generalized Trie implementation.
This script runs a series of tests to measure the performance of the Generalized Trie
against a set of predefined test cases.
'''
# pylint: disable=wrong-import-position, too-many-instance-attributes
# pylint: disable=too-many-positional-arguments, too-many-arguments, too-many-locals

from dataclasses import dataclass, field
import gc
import gzip
import itertools
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, NamedTuple, Optional, Sequence


sys.path.insert(0, str(Path('../src').resolve()))
from gentrie import GeneralizedTrie, GeneralizedKey, TrieId  # noqa: E402

# A minimum of 3 iterations is required to allow statistical analysis
MIN_MEASURED_ITERATIONS: int = 3

DEFAULT_ITERATIONS: int = 10
DEFAULT_TIMER = time.perf_counter_ns

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
TEST_MARKS: list[int] = [3, 4, 5, 6, 7, 8, 9]  # Marks to test - 1 and 2 are omitted due to low key counts
TEST_MAX_KEYS: int = len(SYMBOLS) ** max(TEST_MARKS)  # Limit to a manageable number of keys
for gen_depth in TEST_MARKS:
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
for gen_depth in TEST_MARKS:
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
for gen_depth in TEST_MARKS:
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
for gen_depth in TEST_MARKS:
    TEST_FULLY_POPULATED_TRIES[gen_depth] = generate_fully_populated_trie(max_depth=gen_depth)


def english_words():
    """Imports English words from a gzipped text file.

    The file contains a bit over 278 thousand words in English
    (one per line).
    """
    words_file = Path(__file__).parent.joinpath("english_words.txt.gz")
    return list(map(str.rstrip, gzip.open(words_file, "rt")))


ENGLISH_WORDS = english_words()
TEST_ORGANIC_DATA: dict[str, list[str]] = {
    'english': ENGLISH_WORDS
}
TEST_ORGANIC_TRIES: dict[str, GeneralizedTrie] = {
    'english': generate_test_trie_from_data(ENGLISH_WORDS, None)
}
TEST_ORGANIC_MARKS: dict[str, int] = {
    'english': max(len(word) for word in ENGLISH_WORDS)
}


class BenchGroup(NamedTuple):
    '''Declaration of a benchmark group.

    A benchmark group is a collection of benchmark cases
    that share a common purpose or theme.
    '''
    id: str
    name: str
    description: str
    mark_label: str


@dataclass(kw_only=True)
class BenchIteration:
    '''Container for the results of a single benchmark iteration.'''
    n: int = 0
    elapsed_ns: int = 0
    ops_per_second: float = 0.0


@dataclass(kw_only=True)
class BenchOperationsPerSecond:
    '''Container for the operations per second statistics of a benchmark.
    
    Attributes:
        average (float): The average operations per second.
        median (float): The median operations per second.
        minimum (float): The minimum operations per second.
        maximum (float): The maximum operations per second.
        standard_deviation (float): The standard deviation of operations per second.
        relative_standard_deviation (float): The relative standard deviation of operations per second.
        percentiles (dict[int, float]): Percentiles of operations per second.
    '''
    average: float = 0.0
    median: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    standard_deviation: float = 0.0
    relative_standard_deviation: float = 0.0
    percentiles: dict[int, float] = field(default_factory=dict[int, float])


@dataclass(kw_only=True)
class BenchResults:
    '''Container for the results of a single benchmark test.'''
    group: BenchGroup
    name: str
    mark: int | str
    description: str
    n: int
    runtime_validation: bool
    iterations: list[BenchIteration] = field(default_factory=list[BenchIteration])
    ops_per_second: BenchOperationsPerSecond = field(default_factory=BenchOperationsPerSecond)
    total_elapsed_ns: int = 0


@dataclass(kw_only=True)
class BenchCase:
    '''Declaration of a benchmark case.

    kwargs_variations are used to describe the variations in keyword arguments for the benchmark.
    All combinations of these variations will be tested.

    kwargs_variations example:
        kwargs_variations={
            'mark': [1, 2, 3],
            'runtime_validation': [True, False]
        }

    Args:
        name (str): The name of the benchmark case.
        group (BenchGroup): The reporting group to which the benchmark case belongs.
        mark (int | str | None): The identifying mark for the benchmark case.
        description (str): A brief description of the benchmark case.
        action (Callable[..., Any]): The action to perform for the benchmark.
        min_time (float): The minimum time for the benchmark in seconds.
        max_time (float): The maximum time for the benchmark in seconds.
        kwargs_variations (dict[str, list[Any]]): Variations of keyword arguments for the benchmark.
        runner (Optional[Callable[..., Any]]): A custom runner for the benchmark.
        verbose (bool): Whether to enable verbose output.
    '''
    name: str
    group: BenchGroup
    mark: int | str | None = None
    description: str
    action: Callable[..., Any]
    min_time: float = 5.0  # seconds
    max_time: float = 10.0  # seconds
    kwargs_variations: dict[str, list[Any]] = field(default_factory=dict[str, list[Any]])
    runner: Optional[Callable[..., Any]] = None
    verbose: bool = False

    def __post_init__(self):
        self.results: list[BenchResults] = []

    @property
    def expanded_kwargs_variations(self) -> list[dict[str, Any]]:
        '''All combinations of keyword arguments from the specified kwargs_variations.

        Returns:
            A list of dictionaries, each representing a unique combination of keyword arguments.
        '''
        keys = self.kwargs_variations.keys()
        values = [self.kwargs_variations[key] for key in keys]
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def run(self):
        """Run the benchmark tests.

        This method will execute the benchmark for each combination of
        keyword arguments and collect the results. After running the
        benchmarks, the results will be stored in the `self.results` attribute.
        """
        collected_results: list[BenchResults] = []
        for kwargs in self.expanded_kwargs_variations:
            bench_args: dict[str, Any] = {
                'group': self.group,
                'name': self.name,
                'mark': self.mark,
                'description': self.description,
                'min_time': self.min_time,
                'max_time': self.max_time
            }
            # merge the generated kwargs (this allows overriding BenchCase attributes
            # such as the mark with variations)
            bench_args.update(kwargs)
            if self.verbose:
                name = bench_args["name"]
                formatted_name = name.format(**bench_args)
                mark = bench_args["mark"]
                print(f'Running benchmark "{formatted_name}" for mark {mark}')
            results: BenchResults = self.action(**bench_args)
            collected_results.append(results)
        self.results = collected_results

    def results_as_text_table(self) -> str:
        """
        Returns benchmark results in a text table format if available.

        This method will format the benchmark results into a human-readable text table.
        If the tests have not yet been run, it will indicate that no results are available.

        Returns:
            A string representation of the benchmark results in a printable text table format
            or an indication that no results are available.
        """
        if not self.results:
            return "No benchmark results available."

        output_text_lines: list[str] = []
        output_text_lines.append(f'{self.group.name}\n')
        output_text_lines.append(f'{self.group.description}\n')

        header_line0: str = (
            f'{"":^8s}'
            f' | {"":^6s}'
            f' | {"Elapsed":^7s}'
            f' | {"KOps/Second (avg/median/min/max/5th/95th/std dev)":^74s}'
            f' | {"":^6s}'
            f' | { "Runtime":^10s}'
            f' | {"":^15s}'
        )
        header_line1: str = (
            f'{"N":^8s}'
            f' | {"Iter":^6s}'
            f' | {"seconds":^7s}'
            f' | {"avg":^8s}'
            f' | {"median":^8s}'
            f' | {"min":^8s}'
            f' | {"max":^8s}'
            f' | {"5th":^8s}'
            f' | {"95th":^8s}'
            f' | {"std dev":^8s}'
            f' | {"rsd%":^6s}'
            f' | { "Validate":^10s}'
            f' | {self.group.mark_label:^15s}'
        )
        output_text_lines.append('=' * max(len(header_line0), len(header_line1)))
        output_text_lines.append(header_line0)
        output_text_lines.append(header_line1)
        output_text_lines.append('-' * max(len(header_line0), len(header_line1)))
        for result in self.results:
            output_text_lines.append(
                f'{result.n:>8d}'
                f' |{len(result.iterations):>7d}'
                f' |  {result.total_elapsed_ns/1e9:>5.2f} '
                f' | {result.ops_per_second.average / 1000:8.1f}'
                f' | {result.ops_per_second.median / 1000:8.1f}'
                f' | {result.ops_per_second.minimum / 1000:8.1f}'
                f' | {result.ops_per_second.maximum / 1000:8.1f}'
                f' | {result.ops_per_second.percentiles[5] / 1000:8.1f}'
                f' | {result.ops_per_second.percentiles[95] / 1000:8.1f}'
                f' | {result.ops_per_second.standard_deviation / 1000:8.1f}'
                f' | {result.ops_per_second.relative_standard_deviation:5.1f}%'
                f' | {str(result.runtime_validation):^10s}'
                f' | {str(result.mark):^15s}'
            )
        output_text_lines.append('')
        return '\n'.join(output_text_lines)


class BenchmarkRunner():
    """A class to run benchmarks for various actions.
    """
    @staticmethod
    def default_runner(
            action: Callable[..., Any],
            n: int,
            group: BenchGroup,
            name: str,
            mark: int | str,
            description: str,
            min_time: float,
            max_time: float,
            runtime_validation: bool,
            iterations: int) -> BenchResults:
        """Run a generic benchmark using the specified action and test data for rounds.

        This function will execute the benchmark for the given action and
        collect the results. It is designed for macro-benchmarks (i.e., benchmarks
        that measure the performance of a function over multiple iterations) where
        the overhead of the function call is not significant compared with the work
        done inside the function.

        Micro-benchmarks (i.e., benchmarks that measure the performance of a fast function
        over a small number of iterations) require more complex handling to account
        for the overhead of the function call.

        Args:
            action (Callable[..., Any]): The action to benchmark.
            n (int): The number of test rounds run.
            test_data (Sequence[dict[str, Any] | list[Any] | tuple[Any, ...]]):
                The test data to use for the benchmark rounds.
            group (BenchGroup): The reporting group to which the benchmark case belongs.
            name (str): The name of the benchmark case.
            mark (int | str): The identifying mark for the benchmark case.
            description (str): A brief description of the benchmark case.
            min_time (float): The minimum time for the benchmark in seconds.
            max_time (float): The maximum time for the benchmark in seconds.
            runtime_validation (bool): Whether to perform runtime validation.
            iterations (int): The number of iterations to run.
        """
        benchmark_results = BenchResults(
            group=group,
            name=name.format(runtime_validation=runtime_validation, mark=mark, n=n),
            description=description.format(runtime_validation=runtime_validation, mark=mark, n=n),
            mark=mark,
            runtime_validation=runtime_validation,
            n=n)
        iteration_pass: int = 0
        time_start: int = DEFAULT_TIMER()
        max_stop_at: int = int(max_time * 1e9) + time_start
        min_stop_at: int = int(min_time * 1e9) + time_start
        wall_time: int = DEFAULT_TIMER() - time_start
        iterations_min: int = max(MIN_MEASURED_ITERATIONS, iterations)

        gc.collect()

        while ((iteration_pass <= iterations_min or wall_time < min_stop_at)
                and wall_time < max_stop_at):
            iteration_pass += 1
            iteration_result = BenchIteration()
            iteration_result.elapsed_ns = 0

            # Timer for benchmarked code
            timer_start: int = DEFAULT_TIMER()
            action()
            timer_end: int = DEFAULT_TIMER()

            if iteration_pass == 1:
                # Warmup iteration, not included in final stats
                continue
            iteration_result.elapsed_ns += (timer_end - timer_start)
            iteration_result.ops_per_second = n / (iteration_result.elapsed_ns / 1e9)
            iteration_result.n = n
            benchmark_results.total_elapsed_ns += iteration_result.elapsed_ns
            benchmark_results.iterations.append(iteration_result)
            wall_time = DEFAULT_TIMER()

        average_ops = statistics.mean(iter.ops_per_second for iter in benchmark_results.iterations)
        median_ops = statistics.median(iter.ops_per_second for iter in benchmark_results.iterations)
        standard_deviation: float = 0.0
        if len(benchmark_results.iterations) > 1:
            standard_deviation = statistics.stdev(iter.ops_per_second for iter in benchmark_results.iterations)
        benchmark_results.ops_per_second = BenchOperationsPerSecond(
            average=average_ops,
            median=median_ops,
            minimum=min(iter.ops_per_second for iter in benchmark_results.iterations),
            maximum=max(iter.ops_per_second for iter in benchmark_results.iterations),
            standard_deviation=standard_deviation,
            relative_standard_deviation=standard_deviation / average_ops * 100 if average_ops else 0
        )

        # Calculate percentiles if we have enough data points
        for percentile in [5, 10, 25, 50, 75, 90, 95]:
            if len(benchmark_results.iterations) > 1:
                benchmark_results.ops_per_second.percentiles[percentile] = statistics.quantiles(
                    [iter.ops_per_second for iter in benchmark_results.iterations],
                    n=100)[percentile - 1]
            else:
                benchmark_results.ops_per_second.percentiles[percentile] = float('nan')
        return benchmark_results


def benchmark_build_with_add(
        group: BenchGroup,
        name: str,
        mark: int | str,
        description: str,
        min_time: float,
        max_time: float,
        runtime_validation: bool,
        test_data: dict[int | str, Sequence[GeneralizedKey]],
        iterations: int) -> BenchResults:
    '''Benchmark the addition of keys to the trie.

    Args:
        group (str): The reporting group to which the benchmark case belongs.
        name (str): The name of the benchmark case.
        mark (int | str): The identifying mark for the benchmark case.
        description (str): A brief description of the benchmark case.
        min_time (float): The minimum time for the benchmark in seconds.
        max_time (float): The maximum time for the benchmark in seconds.
        runtime_validation (bool): Whether to enable runtime validation.
        test_data (dict[int | str, Sequence[GeneralizedKey]]): The test data to use for the benchmark.
        iterations (int): The number of iterations to run the benchmark.

    Returns (BenchResults):
        The results of the benchmark.
    '''
    test_keys = test_data[mark]

    def action_to_benchmark():
        trie = GeneralizedTrie(runtime_validation=runtime_validation)
        for key in test_keys:
            trie.add(key, None)

    return BenchmarkRunner.default_runner(
        action=action_to_benchmark,
        n=len(test_keys),
        group=group,
        name=name,
        mark=mark,
        description=description,
        min_time=min_time,
        max_time=max_time,
        runtime_validation=runtime_validation,
        iterations=iterations
    )


def benchmark_build_with_assign(
        name: str,
        group: BenchGroup,
        mark: int | str,
        description: str,
        min_time: float,
        max_time: float,
        runtime_validation: bool,
        test_data: dict[int | str, Sequence[GeneralizedKey]],
        iterations: int) -> BenchResults:
    '''Benchmark the assignment of keys to the trie.

    Args:
        group (str): The reporting group to which the benchmark case belongs.
        name (str): The name of the benchmark case.
        mark (int | str): The identifying mark for the benchmark case.
        min_time (float): The minimum time for the benchmark in seconds.
        max_time (float): The maximum time for the benchmark in seconds.
        description (str): A brief description of the benchmark case.
        runtime_validation (bool): Whether to enable runtime validation.
        test_data (dict[int | str, Sequence[GeneralizedKey]]): The test data to use for the benchmark.
        iterations (int): The number of iterations to run the benchmark.

    Returns (BenchResults):
        The results of the benchmark.
    '''
    test_keys = test_data[mark]

    def action_to_benchmark():
        trie = GeneralizedTrie(runtime_validation=runtime_validation)
        for key in test_keys:
            trie[key] = None

    return BenchmarkRunner.default_runner(
        action=action_to_benchmark,
        n=len(test_keys),
        group=group,
        name=name,
        mark=mark,
        description=description,
        min_time=min_time,
        max_time=max_time,
        runtime_validation=runtime_validation,
        iterations=iterations
    )


def benchmark_build_with_update(
        group: BenchGroup,
        name: str,
        mark: int | str,
        description: str,
        min_time: float,
        max_time: float,
        runtime_validation: bool,
        test_data: dict[int | str, Sequence[GeneralizedKey]],
        iterations: int) -> BenchResults:
    '''Benchmark the building of a trie using update().

    Args:
        group (BenchGroup): The reporting group to which the benchmark case belongs.
        name (str): The name of the benchmark case.
        mark (int | str): The identifying mark for the benchmark case.
        description (str): A brief description of the benchmark case.
        min_time (float): The minimum time for the benchmark in seconds.
        max_time (float): The maximum time for the benchmark in seconds.
        runtime_validation (bool): Whether to enable runtime validation.
        test_data (dict[int | str, Sequence[GeneralizedKey]]): The test data to use for the benchmark.
        iterations (int): The number of iterations to run the benchmark.

    Returns (BenchResults):
        The results of the benchmark.
    '''
    test_keys = test_data[mark]

    def action_to_benchmark():
        trie = GeneralizedTrie(runtime_validation=runtime_validation)
        for key in test_keys:
            trie.update(key, None)

    return BenchmarkRunner.default_runner(
        action=action_to_benchmark,
        n=len(test_keys),
        group=group,
        name=name,
        mark=mark,
        description=description,
        min_time=min_time,
        max_time=max_time,
        runtime_validation=runtime_validation,
        iterations=iterations
    )


def benchmark_updating_trie(
        group: BenchGroup,
        name: str,
        mark: int | str,
        description: str,
        min_time: float,
        max_time: float,
        runtime_validation: bool,
        test_data: dict[str | int, Sequence[GeneralizedKey]],
        iterations: int) -> BenchResults:
    '''Benchmark update() operations on keys in an existing trie.

    This is effectively a benchmark of code like this where all test keys
    are already in the trie and updated to the same value.

    ```
    for key in test_keys:
        trie.update(key, 1)
    ```
    Args:
        name (str): The name of the benchmark case.
        group (BenchGroup): The reporting group to which the benchmark case belongs.
        mark (int | str): The identifying mark for the benchmark case.
        description (str): A brief description of the benchmark case.
        min_time (float): The minimum time for the benchmark in seconds.
        max_time (float): The maximum time for the benchmark in seconds.
        runtime_validation (bool): Whether to enable runtime validation.
        test_data (dict[str | int, Sequence[GeneralizedKey]]): The test data to use for the benchmark.
        iterations (int): The number of iterations to run the benchmark.

    Returns:
        A list of BenchResults containing the benchmark results.
    '''
    # Build the prefix tree - built here because we are modifying it
    # and don't want to modify the pre-generated test tries
    test_keys = test_data[mark]
    test_args_data: list[tuple[GeneralizedKey, int]] = list([(key, 1) for key in test_keys])
    if len(test_keys) != len(test_args_data):
        raise ValueError("Test keys and args data length mismatch")
    trie = generate_test_trie_from_data(data=test_keys, value=None)
    trie.runtime_validation = runtime_validation
    test_keys = test_data[mark]

    def action_to_benchmark():
        for key in test_keys:
            trie.update(key, None)

    return BenchmarkRunner.default_runner(
        action=action_to_benchmark,
        n=len(test_keys),
        group=group,
        name=name,
        mark=mark,
        description=description,
        min_time=min_time,
        max_time=max_time,
        runtime_validation=runtime_validation,
        iterations=iterations
    )


def benchmark_key_in_trie(
        group: BenchGroup,
        name: str,
        mark: int | str,
        description: str,
        min_time: float,
        max_time: float,
        runtime_validation: bool,
        test_tries: dict[int | str, GeneralizedTrie],
        test_data: dict[int | str, list[GeneralizedKey]],
        iterations: int) -> BenchResults:
    '''Benchmark '<key> in <trie>' operations.

    Args:
        name (str): The name of the benchmark case.
        group (BenchGroup): The reporting group to which the benchmark case belongs.
        mark (int | str): The identifying mark for the benchmark case.
        description (str): A brief description of the benchmark case.
        min_time (float): The minimum time for the benchmark in seconds.
        max_time (float): The maximum time for the benchmark in seconds.
        runtime_validation (bool): Whether to enable runtime validation.
        test_tries (dict[int | str, GeneralizedTrie]): The test data to use for the benchmark.
        test_data (dict[int | str, list[GeneralizedKey]]): The test keys to use for the benchmark.
        iterations (int): The number of iterations to run the benchmark.

    Returns:
        A list of BenchResults containing the benchmark results.
    '''
    trie: GeneralizedTrie = test_tries[mark]
    test_keys: list[GeneralizedKey] = test_data[mark]
    trie.runtime_validation = runtime_validation

    def action_to_benchmark():
        for key in test_keys:
            _ = key in trie

    return BenchmarkRunner.default_runner(
        action=action_to_benchmark,
        n=len(test_keys),
        group=group,
        name=name,
        mark=mark,
        description=description,
        min_time=min_time,
        max_time=max_time,
        runtime_validation=runtime_validation,
        iterations=iterations
    )


def benchmark_id_in_trie(
        group: BenchGroup,
        name: str,
        mark: int | str,
        description: str,
        min_time: float,
        max_time: float,
        runtime_validation: bool,
        test_tries: dict[int | str, GeneralizedTrie],
        iterations: int) -> BenchResults:
    '''Benchmark '<TrieId> in trie' operations.

    Args:
        name (str): The name of the benchmark case.
        group (str): The reporting group to which the benchmark case belongs.
        mark (int | str): The identifying mark for the benchmark case.
        description (str): A brief description of the benchmark case.
        min_time (float): The minimum time for the benchmark in seconds.
        max_time (float): The maximum time for the benchmark in seconds.
        runtime_validation (bool): Whether to enable runtime validation.
        test_data (GeneralizedTrie): The test data to use for the benchmark.
        iterations (int): The number of iterations to run the benchmark.

    Returns:
        A list of BenchResults containing the benchmark results.
    '''
    trie: GeneralizedTrie = test_tries[mark]
    test_keys: list[TrieId] = list(trie.keys())  # pyright: ignore[reportAssignmentType]]
    trie.runtime_validation = runtime_validation

    def action_to_benchmark():
        for key in test_keys:
            _ = key in trie  # pylint: disable=unnecessary-dunder-call

    return BenchmarkRunner.default_runner(
        action=action_to_benchmark,
        n=len(test_keys),
        group=group,
        name=name,
        mark=mark,
        description=description,
        min_time=min_time,
        max_time=max_time,
        runtime_validation=runtime_validation,
        iterations=iterations
    )


def get_benchmark_cases() -> list[BenchCase]:
    """
    Define the benchmark cases to be run.
    """
    benchmark_groups_list: list[BenchGroup] = [
        BenchGroup(
            id='synthetic-id-in-trie',
            name='Synthetic "<TrieId> in trie"',
            description='Key lookup using "<TrieId> in trie" and synthetic data',
            mark_label='Depth'
        ),
        BenchGroup(
            id='synthetic-key-in-trie',
            name='Synthetic "<key> in trie"',
            description='Key lookup using "<key> in trie" and synthetic data',
            mark_label='Depth'
        ),
        BenchGroup(
            id='synthetic-building-trie-add()',
            name='Synthetic building trie using add()',
            description=('Building a trie using synthetic data and the add() method '
                         '(trie.add(key, value))'),
            mark_label='Depth'
        ),
        BenchGroup(
            id='synthetic-building-trie-update()',
            name='Synthetic building trie using update()',
            description=('Building a trie using synthetic data and the update() method '
                         '(trie.update(key, value))'),
            mark_label='Depth'
        ),
        BenchGroup(
            id='synthetic-building-trie-assign',
            name='Synthetic building trie using assignment',
            description=('Building a trie using synthetic data and '
                         'assignment (trie[key] = value)'),
            mark_label='Depth'
        ),
        BenchGroup(
            id='synthetic-updating-trie-update()',
            name='Synthetic updating trie',
            description=('Updating a trie using synthetic data and the update() method '
                         '(trie.update(key, value))'),
            mark_label='Depth'
        ),
        BenchGroup(
            id='english-dictionary-id-in-trie',
            name='English Dictionary "<TrieId> in trie")',
            description=(
                '"<TrieId> in trie" operation for words from the English dictionary'
            ),
            mark_label='Data Set'
        ),
        BenchGroup(
            id='english-dictionary-key-in-trie',
            name='English Dictionary "<key> in trie"',
            description='"<key> in trie" operation for words from the English dictionary',
            mark_label='Data Set'
        )
    ]

    benchmark_groups: dict[str, BenchGroup] = {}
    for group in benchmark_groups_list:
        benchmark_groups[group.id] = group

    benchmark_cases_list: list[BenchCase] = [
        BenchCase(
            name='<key> in trie (synthetic data, runtime validation: {runtime_validation})',
            group=benchmark_groups['synthetic-key-in-trie'],
            description='Existence check using "<key> in trie" and synthetic data',
            action=benchmark_key_in_trie,
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_tries': [TEST_TRIES],
                'test_data': [TEST_DATA],
                'iterations': [DEFAULT_ITERATIONS],
                'mark': TEST_MARKS,
            }
        ),
        BenchCase(
            name='<TrieId> in trie (English dictionary, runtime validation: {runtime_validation})',
            group=benchmark_groups['english-dictionary-id-in-trie'],
            description=('Existence check using "<TrieId> in trie" '
                         'and words from the English dictionary'),
            action=benchmark_id_in_trie,
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_tries': [TEST_ORGANIC_TRIES],
                'iterations': [DEFAULT_ITERATIONS],
                'mark': ['english'],
            }
        ),
        BenchCase(
            name='<key> in trie (English dictionary, runtime validation: {runtime_validation})',
            group=benchmark_groups['english-dictionary-key-in-trie'],
            description=('Existence check using "<key> in trie" '
                         'and words from the English dictionary'),
            action=benchmark_key_in_trie,
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_tries': [TEST_ORGANIC_TRIES],
                'test_data': [TEST_ORGANIC_DATA],
                'iterations': [DEFAULT_ITERATIONS],
                'mark': ['english'],
            }
        ),
        BenchCase(
            name='trie build with add() (synthetic data, runtime validation: {runtime_validation})',
            group=benchmark_groups['synthetic-building-trie-add()'],
            description='Building a trie using the add() method and synthetic data',
            action=benchmark_build_with_add,
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_data': [TEST_DATA],
                'iterations': [DEFAULT_ITERATIONS],
                'mark': TEST_MARKS,
            }
        ),
        BenchCase(
            name='trie build using update() method (synthetic data, runtime validation: {runtime_validation})',
            group=benchmark_groups['synthetic-building-trie-update()'],
            description='Building a trie using the update() method and synthetic data',
            action=benchmark_build_with_update,
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_data': [TEST_DATA],
                'iterations': [DEFAULT_ITERATIONS],
                'mark': TEST_MARKS,
            }
        ),
        BenchCase(
            name='trie build with trie[key] = key (synthetic data, runtime validation: {runtime_validation})',
            group=benchmark_groups['synthetic-building-trie-assign'],
            description='Building a trie using "trie[<key>] = <value>" assignment and synthetic data',
            action=benchmark_build_with_assign,
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_data': [TEST_DATA],
                'iterations': [DEFAULT_ITERATIONS],
                'mark': TEST_MARKS,
            }
        ),
        BenchCase(
            name='trie updating with update() method (runtime validation: {runtime_validation})',
            group=benchmark_groups['synthetic-updating-trie-update()'],
            description='Updating a trie using the update() method',
            action=benchmark_updating_trie,
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_data': [TEST_DATA],
                'iterations': [DEFAULT_ITERATIONS],
                'mark': TEST_MARKS,
            }
        ),
    ]
    return benchmark_cases_list


def run_benchmarks():
    """Run the benchmark tests and print the results.
    """
    benchmark_cases: list[BenchCase] = get_benchmark_cases()
    for case in benchmark_cases:
        case.run()
        print(case.results_as_text_table() + '\n')


def main():
    """Main entry point for running benchmarks."""
    run_benchmarks()


if __name__ == '__main__':
    main()
