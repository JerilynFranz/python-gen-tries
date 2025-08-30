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
import time
from typing import Any, Callable, NamedTuple, Optional, Sequence

from rich.console import Console
from rich.table import Table

from gentrie import GeneralizedTrie, GeneralizedKey, TrieId

# A minimum of 3 iterations is required to allow statistical analysis
MIN_MEASURED_ITERATIONS: int = 3

DEFAULT_ITERATIONS: int = 20

DEFAULT_TIMER = time.perf_counter_ns
DEFAULT_INTERVAL_SCALE: float = 1e-9
DEFAULT_INTERVAL_UNIT: str = 'ns'
DEFAULT_OPS_PER_INTERVAL_SCALE: float = 1.0
DEFAULT_OPS_PER_INTERVAL_UNIT: str = 'Ops/s'

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


def generate_test_trie(depth: int,
                       symbols: str,
                       max_keys: int,
                       value: Optional[Any] = None,
                       runtime_validation: bool = True) -> GeneralizedTrie:
    '''Generate a test Generalized Trie for the given depth and symbols.

    Args:
        depth (int): The depth of the trie.
        symbols (str): The symbols to use in the trie.
        max_keys (int): The maximum number of keys to generate.
        value (Optional[Any]): The value to assign to each key in the trie.
        runtime_validation (bool): Whether to enable runtime validation on the returned trie. (default: True)

    Returns:
        GeneralizedTrie: The generated trie with the specified keys and values.
    '''
    return generate_test_trie_from_data(
        data=generate_test_data(depth, symbols, max_keys),
        value=value,
        runtime_validation=runtime_validation)


def generate_test_trie_from_data(
        data: Sequence[GeneralizedKey],
        value: Optional[Any] = None,
        runtime_validation: bool = True) -> GeneralizedTrie:
    '''Generate a test Generalized Trie from the passed Sequence of GeneralizedKey.

    Args:
        data (Sequence[GeneralizedKey]): The sequence of keys to insert into the trie.
        value (Optional[Any]): The value to assign to each key in the trie.
        runtime_validation (bool): Whether to enable runtime validation on the returned trie. (default: True)

    Returns:
        GeneralizedTrie: The generated trie with the specified keys and values.
    '''
    trie = GeneralizedTrie(runtime_validation=False)
    for key in data:
        trie[key] = value
    trie.runtime_validation = runtime_validation
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
    '''Container for the results of a single benchmark iteration.

    Properties:
        n (int): The number of operations performed.
        elapsed (float): The elapsed time for the operations.
        unit (str): The unit of measurement for the elapsed time.
        scale (float): The scale factor for the elapsed time.
        ops_per_second (float): The number of operations per second. (read only)
    '''
    n: int = 0
    elapsed: int = 0
    unit: str = DEFAULT_INTERVAL_UNIT
    scale: float = DEFAULT_INTERVAL_SCALE

    @property
    def ops_per_second(self) -> float:
        '''The number of operations per second.

        This is calculated as the inverse of the elapsed time.

        The edge cases of 0 elapsed time or n results in a returned value of 0.
        This would otherwise be an impossible value and so flags a measurement error.
        '''
        if not self.elapsed:
            return 0
        return self.n / (self.elapsed * self.scale)


class BenchStatistics:
    '''Generic container for statistics on a benchmark.

    Attributes:
        unit (str): The unit of measurement for the benchmark (e.g., "ops/s").
        scale (float): The scale factor for the interval (e.g. 1 for seconds).
        data: list[int | float] = field(default_factory=list[int | float])
        mean (float): The mean operations per time interval. (read only)
        median (float): The median operations per time interval. (read only)
        minimum (float): The minimum operations per time interval. (read only)
        maximum (float): The maximum operations per time interval. (read only)
        standard_deviation (float): The standard deviation of operations per time interval. (read only)
        relative_standard_deviation (float): The relative standard deviation of ops per time interval. (read only)
        percentiles (dict[int, float]): Percentiles of operations per time interval. (read only)
    '''
    def __init__(self, unit: str = '', scale: float = 0.0, data: Optional[list[int | float]] = None):
        self.unit: str = unit
        self.scale: float = scale
        self.data: list[int | float] = data if data is not None else []

    @property
    def mean(self) -> float:
        '''The mean of the data.'''
        return statistics.mean(self.data) if self.data else 0.0

    @property
    def median(self) -> float:
        '''The median of the data.'''
        return statistics.median(self.data) if self.data else 0.0

    @property
    def minimum(self) -> float:
        '''The minimum of the data.'''
        return float(min(self.data)) if self.data else 0.0

    @property
    def maximum(self) -> float:
        '''The maximum of the data.'''
        return float(max(self.data)) if self.data else 0.0

    @property
    def standard_deviation(self) -> float:
        '''The standard deviation of the data.'''
        return statistics.stdev(self.data) if len(self.data) > 1 else 0.0

    @property
    def relative_standard_deviation(self):
        '''The relative standard deviation of the data.'''
        return self.standard_deviation / self.mean * 100 if self.mean else 0.0

    @property
    def percentiles(self) -> dict[int, float]:
        '''Percentiles of the data.

        Computes the 5th, 10th, 25th, 50th, 75th, 90th, and 95th percentiles
        and returns them as a dictionary keyed by percent.
        '''
        # Calculate percentiles if we have enough data points
        if not self.data:
            return {p: float('nan') for p in [5, 10, 25, 50, 75, 90, 95]}
        percentiles: dict[int, float] = {}
        for percent in [5, 10, 25, 50, 75, 90, 95]:
            percentiles[percent] = statistics.quantiles(self.data, n=100)[percent - 1]
        return percentiles


class BenchOperationsPerInterval(BenchStatistics):
    '''Container for the operations per time interval statistics of a benchmark.

    Attributes:
        unit (str): The unit of measurement for the benchmark (e.g., "ops/s").
        scale (float): The scale factor for the interval (e.g. 1 for seconds).
        data: list[int] = field(default_factory=list[int])
        mean (float): The mean operations per time interval. (read only)
        median (float): The median operations per time interval. (read only)
        minimum (float): The minimum operations per time interval. (read only)
        maximum (float): The maximum operations per time interval. (read only)
        standard_deviation (float): The standard deviation of operations per time interval. (read only)
        relative_standard_deviation (float): The relative standard deviation of ops per time interval. (read only)
        percentiles (dict[int, float]): Percentiles of operations per time interval. (read only)
    '''
    def __init__(self,
                 unit: str = DEFAULT_OPS_PER_INTERVAL_UNIT,
                 scale: float = DEFAULT_OPS_PER_INTERVAL_SCALE,
                 data: Optional[list[int | float]] = None):
        super().__init__(unit=unit, scale=scale, data=data)


class BenchOperationTimings(BenchStatistics):
    '''Container for the operation timing statistics of a benchmark.

    Attributes:
        unit (str): The unit of measurement for the timings (e.g., "ns").
        scale (float): The scale factor for the timings (e.g., "1e-9" for nanoseconds).
        mean (float): The mean time per operation.
        median (float): The median time per operation.
        minimum (float): The minimum time per operation.
        maximum (float): The maximum time per operation.
        standard_deviation (float): The standard deviation of the time per operation.
        relative_standard_deviation (float): The relative standard deviation of the time per operation.
        percentiles (dict[int, float]): Percentiles of time per operation.
        data: list[float | int] = field(default_factory=list[float | int])
    '''
    def __init__(self,
                 unit: str = DEFAULT_INTERVAL_UNIT,
                 scale: float = DEFAULT_INTERVAL_SCALE,
                 data: Optional[list[int | float]] = None):
        super().__init__(unit=unit, scale=scale, data=data)


@dataclass(kw_only=True)
class BenchResults:
    '''Container for the results of a single benchmark test.

    Properties:
        group (BenchGroup): The reporting group to which the benchmark case belongs.
        name (str): The name of the benchmark case.
        mark (int | str): The identifying mark for the benchmark case.
        description (str): A brief description of the benchmark case.
        n (int): The number of rounds the benchmark ran per iteration.
        runtime_validation (bool): Whether runtime validation was enabled
        interval_unit (str): The unit of measurement for the interval (e.g. "ns").
        interval_scale (float): The scale factor for the interval (e.g. 1e-9 for nanoseconds).
        ops_per_interval_unit (str): The unit of measurement for operations per interval (e.g. "ops/s").
        ops_per_interval_scale (float): The scale factor for operations per interval (e.g. 1.0 for ops/s).
        total_elapsed (int): The total elapsed time for the benchmark.
        extra_info (dict[str, Any]): Additional information about the benchmark run.
    '''
    group: BenchGroup
    name: str
    mark: int | str
    description: str
    n: int
    runtime_validation: bool
    interval_unit: str = DEFAULT_INTERVAL_UNIT
    interval_scale: float = DEFAULT_INTERVAL_SCALE
    ops_per_interval_unit: str = DEFAULT_INTERVAL_UNIT
    ops_per_interval_scale: float = DEFAULT_INTERVAL_SCALE
    iterations: list[BenchIteration] = field(default_factory=list[BenchIteration])
    ops_per_second: BenchOperationsPerInterval = field(default_factory=BenchOperationsPerInterval)
    op_timings: BenchOperationTimings = field(default_factory=BenchOperationTimings)
    total_elapsed: int = 0
    extra_info: dict[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self):
        if self.iterations:
            self.op_timings.data = list([iteration.elapsed for iteration in self.iterations])
            self.ops_per_second.data = list([iteration.ops_per_second for iteration in self.iterations])


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
        min_time (float): The minimum time for the benchmark in seconds. (default: 5.0)
        max_time (float): The maximum time for the benchmark in seconds. (default: 20.0)
        kwargs_variations (dict[str, list[Any]]): Variations of keyword arguments for the benchmark.
        runner (Optional[Callable[..., Any]]): A custom runner for the benchmark.
    '''
    name: str
    group: BenchGroup
    mark: int | str | None = None
    description: str
    action: Callable[..., Any]
    min_time: float = 5.0  # seconds
    max_time: float = 20.0  # seconds
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
                'max_time': self.max_time,
                # 'verbose': self.verbose
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

    def _scale_for(self, numbers: list[float], base_unit: str) -> tuple[str, float]:
        """Scale a list of numbers by a given factor.

        Args:
            numbers: A list of numbers to scale.
            base_unit: The base unit to use for scaling.

        Returns:
            A tuple containing the scaled unit and the scaling factor.
        """
        min_n: float = min(numbers)
        unit: str = ''
        scale: float = 1.0
        if min_n >= 1e9:
            unit, scale = 'G' + base_unit, 1e-9
        elif min_n >= 1e6:
            unit, scale = 'M' + base_unit, 1e-6
        elif min_n >= 1e3:
            unit, scale = 'K' + base_unit, 1e-3
        elif min_n >= 1e0:
            unit, scale = base_unit, 1.0
        elif min_n >= 1e-3:
            unit, scale = 'm' + base_unit, 1e3
        elif min_n >= 1e-6:
            unit, scale = 'μ' + base_unit, 1e6
        elif min_n >= 1e-9:
            unit, scale = 'n' + base_unit, 1e9
        return unit, scale

    def results_as_rich_table(self) -> Table:
        """Returns benchmark results in a rich table format if available.
        """
        mean_unit, mean_scale = self._scale_for(
            numbers=[result.ops_per_second.mean for result in self.results],
            base_unit='Ops')
        median_unit, median_scale = self._scale_for(
            numbers=[result.ops_per_second.median for result in self.results],
            base_unit='Ops')
        min_unit, min_scale = self._scale_for(
            numbers=[result.ops_per_second.minimum for result in self.results],
            base_unit='Ops')
        max_unit, max_scale = self._scale_for(
            numbers=[result.ops_per_second.maximum for result in self.results],
            base_unit='Ops')
        p5_unit, p5_scale = self._scale_for(
            numbers=[result.ops_per_second.percentiles[5] for result in self.results],
            base_unit='Ops')
        p95_unit, p95_scale = self._scale_for(
            numbers=[result.ops_per_second.percentiles[95] for result in self.results],
            base_unit='Ops')
        stddev_unit, stddev_scale = self._scale_for(
            numbers=[result.ops_per_second.standard_deviation for result in self.results],
            base_unit='Ops')

        table = Table(title=(self.name + '\n\n' + self.description),
                      show_header=True,
                      title_style='bold green1',
                      header_style='bold magenta')
        table.add_column('N', justify='center')
        table.add_column('Iterations', justify='center')
        table.add_column('Elapsed Seconds', justify='center', max_width=7)
        table.add_column(f'mean {mean_unit}', justify='center', vertical='bottom', overflow='fold')
        table.add_column(f'median {median_unit}', justify='center', vertical='bottom', overflow='fold')
        table.add_column(f'min {min_unit}', justify='center', vertical='bottom', overflow='fold')
        table.add_column(f'max {max_unit}', justify='center', vertical='bottom', overflow='fold')
        table.add_column(f'5th {p5_unit}', justify='center', vertical='bottom', overflow='fold')
        table.add_column(f'95th {p95_unit}', justify='center', vertical='bottom', overflow='fold')
        table.add_column(f'std dev {stddev_unit}', justify='center', vertical='bottom', overflow='fold')
        table.add_column('rsd%', justify='center', vertical='bottom', overflow='fold')
        table.add_column('Runtime Validate', justify='center', vertical='bottom', overflow='fold', max_width=9)
        table.add_column(self.group.mark_label, justify='center', vertical='bottom', overflow='fold')
        for result in self.results:
            table.add_row(
                f'{result.n:>6d}',
                f'{len(result.iterations):>6d}',
                f'{result.total_elapsed * DEFAULT_INTERVAL_SCALE:>4.2f}',
                f'{result.ops_per_second.mean * mean_scale:>6.2f}',
                f'{result.ops_per_second.median * median_scale:>6.2f}',
                f'{result.ops_per_second.minimum * min_scale:>6.2f}',
                f'{result.ops_per_second.maximum * max_scale:>6.2f}',
                f'{result.ops_per_second.percentiles[5] * p5_scale:>6.2f}',
                f'{result.ops_per_second.percentiles[95] * p95_scale:>6.2f}',
                f'{result.ops_per_second.standard_deviation * stddev_scale:>6.2f}',
                f'{result.ops_per_second.relative_standard_deviation:>3.2f}%',
                f'{result.runtime_validation!s}',
                f'{result.mark!s}')
        return table


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
            iterations: int,
            setup: Optional[Callable[..., Any]] = None,
            teardown: Optional[Callable[..., Any]] = None,
            verbose: bool = False) -> BenchResults:
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
            setup (Optional[Callable[..., Any]]): A setup function to run before each iteration.
            teardown (Optional[Callable[..., Any]]): A teardown function to run after each iteration.
            verbose (bool): Whether to print verbose output. (default = False)
        """
        iteration_pass: int = 0
        time_start: int = DEFAULT_TIMER()
        max_stop_at: int = int(max_time / DEFAULT_INTERVAL_SCALE) + time_start
        min_stop_at: int = int(min_time / DEFAULT_INTERVAL_SCALE) + time_start
        wall_time: int = DEFAULT_TIMER() - time_start
        iterations_min: int = max(MIN_MEASURED_ITERATIONS, iterations)

        gc.collect()

        total_elapsed: int = 0
        iterations_list: list[BenchIteration] = []
        while ((iteration_pass <= iterations_min or wall_time < min_stop_at)
                and wall_time < max_stop_at):
            iteration_pass += 1
            iteration_result = BenchIteration()
            iteration_result.elapsed = 0

            if isinstance(setup, Callable):
                setup()

            # Timer for benchmarked code
            timer_start: int = DEFAULT_TIMER()
            action()
            timer_end: int = DEFAULT_TIMER()

            if isinstance(teardown, Callable):
                teardown()

            if iteration_pass == 1:
                # Warmup iteration, not included in final stats
                continue
            iteration_result.elapsed += (timer_end - timer_start)
            iteration_result.n = n
            total_elapsed += iteration_result.elapsed
            iterations_list.append(iteration_result)
            wall_time = DEFAULT_TIMER()

        benchmark_results = BenchResults(
            group=group,
            name=name.format(runtime_validation=runtime_validation, mark=mark, n=n),
            description=description.format(runtime_validation=runtime_validation, mark=mark, n=n),
            mark=mark,
            runtime_validation=runtime_validation,
            n=n,
            iterations=iterations_list,
            total_elapsed=total_elapsed,
            extra_info={})

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
        iterations=iterations,
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
        iterations=iterations,
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
        iterations=iterations,
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
        iterations=iterations,
    )


def benchmark_remove_key_from_trie(
        group: BenchGroup,
        name: str,
        mark: int | str,
        description: str,
        min_time: float,
        max_time: float,
        runtime_validation: bool,
        test_data: dict[str | int, Sequence[GeneralizedKey]],
        iterations: int) -> BenchResults:
    '''Benchmark remove() operations on keys in an existing trie.

    This is effectively a benchmark of code like this where all test keys
    are already in the trie and updated to the same value.

    ```
    for key in test_keys:
        trie.remove(key)
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

    test_keys = test_data[mark]
    trie: GeneralizedTrie = GeneralizedTrie(runtime_validation=runtime_validation)

    def setup():
        "Setup the trie with test keys. clear() first for safety."
        trie.clear()
        for key in test_keys:
            trie.update(key, None)

    def action_to_benchmark():
        "Remove all test keys from the trie."
        for key in test_keys:
            trie.remove(key)

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
        iterations=iterations,
        setup=setup,
    )


def benchmark_del_key_from_trie(
        group: BenchGroup,
        name: str,
        mark: int | str,
        description: str,
        min_time: float,
        max_time: float,
        runtime_validation: bool,
        test_data: dict[str | int, Sequence[GeneralizedKey]],
        iterations: int) -> BenchResults:
    '''Benchmark "del trie[<key>] operations on keys in an existing trie.

    This is effectively a benchmark of code like this where all test keys
    are already in the trie.

    ```
    for key in test_keys:
        del trie[key]
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
    test_keys: Sequence[GeneralizedKey] = test_data[mark]
    trie: GeneralizedTrie = GeneralizedTrie(runtime_validation=runtime_validation)

    def setup():
        "Setup the trie with test keys."
        trie.clear()
        for key in test_keys:
            trie.update(key, None)

    def action_to_benchmark():
        "Remove all test keys from the trie using del operator"
        for key in test_keys:
            del trie[key]

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
        iterations=iterations,
        setup=setup,
    )


def benchmark_del_id_from_trie(
        group: BenchGroup,
        name: str,
        mark: int | str,
        description: str,
        min_time: float,
        max_time: float,
        runtime_validation: bool,
        test_data: dict[str | int, Sequence[GeneralizedKey]],
        iterations: int) -> BenchResults:
    '''Benchmark "del trie[<key>] operations on keys in an existing trie.

    This is effectively a benchmark of code like this where all test keys
    are already in the trie.

    ```
    for key in test_keys:
        del trie[key]
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
    test_keys: Sequence[GeneralizedKey] = test_data[mark]
    test_ids: list[TrieId] = []
    trie: GeneralizedTrie = GeneralizedTrie(runtime_validation=runtime_validation)

    def setup():
        "Setup the trie with test keys. clear() first for safety."
        trie.clear()
        for key in test_keys:
            trie.update(key, None)
        test_ids.clear()
        test_ids.extend(trie.keys())

    def action_to_benchmark():
        "Remove all test keys from the trie."
        for trie_id in test_ids:
            del trie[trie_id]

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
        iterations=iterations,
        setup=setup,
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
        iterations=iterations,
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
    test_keys: list[TrieId] = list(trie.keys())

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
        iterations=iterations,
    )


def benchmark_trie_prefixes_key(
        group: BenchGroup,
        name: str,
        mark: int | str,
        description: str,
        min_time: float,
        max_time: float,
        runtime_validation: bool,
        test_tries: dict[str | int, GeneralizedTrie],
        test_data: dict[str | int, Sequence[GeneralizedKey]],
        iterations: int) -> BenchResults:
    '''Benchmark trie prefixes() method.

    This test checks the performance of the prefixes() method on fully populated tries.
    Because the potential number of matching keys in the trie increases linearly with depth
    and the full runtime of a prefix search is dominated by the number of keys found,
    this test aims to measure the impact of this growth on the performance of the
    prefixes() method.

    Because prefixes() returns a Generator, we need to exhaust it to measure its performance.
    This is done by converting the generator to a list.

    Interpreting performance here is tricky because the number of keys found per prefix can vary
    significantly and they can have a large impact on the overall measurement.
    ```
    for key in test_keys:
        _ = list(trie.prefixes(key))
    ```
    Args:
        name (str): The name of the benchmark case.
        group (BenchGroup): The reporting group to which the benchmark case belongs.
        mark (int | str): The identifying mark for the benchmark case.
        description (str): A brief description of the benchmark case.
        min_time (float): The minimum time for the benchmark in seconds.
        max_time (float): The maximum time for the benchmark in seconds.
        runtime_validation (bool): Whether to enable runtime validation.
        test_trie (dict[str | int, GeneralizedTrie]): The test tries to use for the benchmark.
        test_data (dict[str | int, Sequence[GeneralizedKey]]): The test data to use for the benchmark.
        iterations (int): The number of iterations to run the benchmark.

    Returns:
        A list of BenchResults containing the benchmark results.
    '''
    test_keys = test_data[mark]
    trie = test_tries[mark]
    trie.runtime_validation = runtime_validation
    n: int = len(test_keys)

    def action_to_benchmark():
        for key in test_keys:
            _ = list(trie.prefixes(key))

    return BenchmarkRunner.default_runner(
        action=action_to_benchmark,
        n=n,
        group=group,
        name=name,
        mark=mark,
        description=description,
        min_time=min_time,
        max_time=max_time,
        runtime_validation=runtime_validation,
        iterations=iterations,
    )


def benchmark_trie_prefixed_by_key(
        group: BenchGroup,
        name: str,
        mark: int | str,
        description: str,
        min_time: float,
        max_time: float,
        runtime_validation: bool,
        test_trie: GeneralizedTrie,
        test_keys: Sequence[GeneralizedKey],
        iterations: int) -> BenchResults:
    '''Benchmark trie prefixes_by() method.

    This test checks the performance of the prefixed_by() method on fully populated tries
    at various search depths.

    Because the potential number of matching keys in the trie increases exponentially with depth
    and the full runtime of a prefix search is dominated by the sheer number of keys found,
    this test aims to measure the impact of this growth on the performance of the prefixes() method.

    Because prefixed_by() returns a Generator, we need to exhaust it to measure its performance.
    This is done by converting the generator to a list.

    Interpreting performance here is tricky because the number of keys found per prefix can vary
    significantly by depth and they can have a large impact on the overall measurement.
    ```
    for key in test_keys:
        _ = list(trie.prefixed_by(key, depth))
    ```
    Args:
        name (str): The name of the benchmark case.
        group (BenchGroup): The reporting group to which the benchmark case belongs.
        mark (int | str): The identifying mark for the benchmark case.
        description (str): A brief description of the benchmark case.
        min_time (float): The minimum time for the benchmark in seconds.
        max_time (float): The maximum time for the benchmark in seconds.
        runtime_validation (bool): Whether to enable runtime validation.
        test_trie (GeneralizedTrie): The test trie to use for the benchmark.
        test_keys (Sequence[GeneralizedKey]): The target keys to use for the benchmark.
        iterations (int): The number of iterations to run the benchmark.

    Returns:
        A list of BenchResults containing the benchmark results.
    '''
    trie = test_trie
    if not isinstance(mark, int):
        raise TypeError(f"Expected 'mark' to be int, got {type(mark).__name__}")
    search_depth = mark
    trie.runtime_validation = runtime_validation
    n: int = len(test_keys)

    def action_to_benchmark():
        for key in test_keys:
            _ = list(trie.prefixed_by(key, search_depth))

    return BenchmarkRunner.default_runner(
        action=action_to_benchmark,
        n=n,
        group=group,
        name=name,
        mark=mark,
        description=description,
        min_time=min_time,
        max_time=max_time,
        runtime_validation=runtime_validation,
        iterations=iterations,
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
            name='Synthetic building trie using trie.add(<key>, <value>)',
            description=('Building a trie using synthetic data and the add() method '
                         '(trie.add(<key>, <value>))'),
            mark_label='Depth'
        ),
        BenchGroup(
            id='synthetic-building-trie-update()',
            name='Synthetic building trie using trie.update(<key>, <value>)',
            description=('Building a trie using synthetic data and the update() method '
                         '(trie.update(<key>, <value>))'),
            mark_label='Depth'
        ),
        BenchGroup(
            id='synthetic-building-trie-assign',
            name='Synthetic building trie using trie[<key>] = <value>',
            description=('Building a trie using synthetic data and '
                         'assignment (trie[<key>] = <value>)'),
            mark_label='Depth'
        ),
        BenchGroup(
            id='synthetic-updating-trie-update()',
            name='Synthetic updating trie using trie.update(<key>, <value>)',
            description=('Updating a trie using synthetic data and the update() method '
                         '(trie.update(<key>, <value>))'),
            mark_label='Depth'
        ),
        BenchGroup(
            id='synthetic-trie-remove()-key',
            name='Synthetic removing keys from trie using trie.remove(<key>)',
            description=('Deleting keys using synthetic data and the remove() method '
                         '(trie.remove(<key>))'),
            mark_label='Depth'
        ),
        BenchGroup(
            id='synthetic-trie-del-key',
            name='Synthetic deleting keys using "del trie[<key>]"',
            description=('Deleting keys using synthetic data and the del operator '
                         '(del trie[<key>])'),
            mark_label='Depth'
        ),
        BenchGroup(
            id='synthetic-trie-del-id',
            name='Synthetic deleting keys using "del trie[<TrieId>]"',
            description=('Deleting keys using synthetic data and the del operator '
                         '(del trie[<TrieId>])'),
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
        ),
        BenchGroup(
            id='synthetic-trie-prefixes(<key>)',
            name='Synthetic trie.prefixes(<key>)',
            description=('Finding keys using trie.prefixes(<key>) method'),
            mark_label='Depth'
        ),
        BenchGroup(
            id='synthetic-trie-prefixed_by(<key>, <search_depth>)',
            name='Synthetic trie.prefixed_by(<key>, <search_depth>)',
            description=('Finding keys using trie.prefixed_by(<key>, <search_depth>) method'),
            mark_label='Search Depth'
        ),
    ]

    benchmark_groups: dict[str, BenchGroup] = {}
    for group in benchmark_groups_list:
        benchmark_groups[group.id] = group

    benchmark_cases_list: list[BenchCase] = [
        BenchCase(
            name='<key> in trie (Synthetic)',
            group=benchmark_groups['synthetic-key-in-trie'],
            description='Timing [yellow bold]<key> in trie[/yellow bold] with synthetic data',
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
            name='<TrieId> in trie (English)',
            group=benchmark_groups['english-dictionary-id-in-trie'],
            description=(
                'Timing [yellow bold]<TrieId> in trie[/yellow bold] with words from the English dictionary'),
            action=benchmark_id_in_trie,
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_tries': [TEST_ORGANIC_TRIES],
                'iterations': [DEFAULT_ITERATIONS],
                'mark': ['english'],
            }
        ),
        BenchCase(
            name='<key> in trie (English)',
            group=benchmark_groups['english-dictionary-key-in-trie'],
            description=('Timing [yellow bold]<key> in trie[/yellow bold] with words from the English dictionary'),
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
            name='trie.add(<key>, <value>) (Synthetic)',
            group=benchmark_groups['synthetic-building-trie-add()'],
            description=('Timing [yellow bold]trie.add(<key>, <value>)[/yellow bold] '
                         'while building a newtrie with synthetic data'),
            action=benchmark_build_with_add,
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_data': [TEST_DATA],
                'iterations': [DEFAULT_ITERATIONS],
                'mark': TEST_MARKS,
            }
        ),
        BenchCase(
            name='trie.update(<key>, <value>) (Synthetic)',
            group=benchmark_groups['synthetic-building-trie-update()'],
            description=('Timing [yellow bold]trie.update(<key>, <value>)[/yellow bold] '
                         'while building a new trie with synthetic data'),
            action=benchmark_build_with_update,
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_data': [TEST_DATA],
                'iterations': [DEFAULT_ITERATIONS],
                'mark': TEST_MARKS,
            }
        ),
        BenchCase(
            name='trie[<key>] = <value> (Synthetic)',
            group=benchmark_groups['synthetic-building-trie-assign'],
            description=('Timing [yellow bold]trie[<key>] = <value>[/yellow bold] '
                         'while building a new trie with synthetic data'),
            action=benchmark_build_with_assign,
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_data': [TEST_DATA],
                'iterations': [DEFAULT_ITERATIONS],
                'mark': TEST_MARKS,
            }
        ),
        BenchCase(
            name='trie.update(<key>, <value>) (Synthetic)',
            group=benchmark_groups['synthetic-updating-trie-update()'],
            description=('Timing [yellow bold]trie.update(<key>, <value>)[/yellow bold] '
                         'while updating values for existing keys with synthetic data'),
            action=benchmark_updating_trie,
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_data': [TEST_DATA],
                'iterations': [DEFAULT_ITERATIONS],
                'mark': TEST_MARKS,
            }
        ),
        BenchCase(
            name='trie.remove(<key>) (Synthetic)',
            group=benchmark_groups['synthetic-trie-remove()-key'],
            description=('Timing [yellow bold]trie.remove(<key>)[/yellow bold] '
                         'while removing keys from a trie with synthetic data'),
            action=benchmark_remove_key_from_trie,
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_data': [TEST_DATA],
                'iterations': [DEFAULT_ITERATIONS],
                'mark': TEST_MARKS,
            },
        ),
        BenchCase(
            name='del trie[<key>] (Synthetic)',
            group=benchmark_groups['synthetic-trie-del-key'],
            description=('Timing [yellow bold]del trie[<key>][/yellow bold] '
                         'while deleting keys from a trie with synthetic data'),
            action=benchmark_del_key_from_trie,
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_data': [TEST_DATA],
                'iterations': [DEFAULT_ITERATIONS],
                'mark': TEST_MARKS,
            },
        ),
        BenchCase(
            name='del trie[<TrieId>] (Synthetic)',
            group=benchmark_groups['synthetic-trie-del-id'],
            description=('Timing [yellow bold]del trie[<TrieId>][/yellow bold] '
                         'while deleting keys from a trie with synthetic data'),
            action=benchmark_del_id_from_trie,
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_data': [TEST_DATA],
                'iterations': [DEFAULT_ITERATIONS],
                'mark': TEST_MARKS,
            },
        ),
        BenchCase(
            name='trie.prefixes(<key>) (Synthetic)',
            group=benchmark_groups['synthetic-trie-prefixes(<key>)'],
            description=('Timing [yellow bold]trie.prefixes(<key>)[/yellow bold] '
                         'while finding keys matching a specific prefix in a trie with synthetic data'),
            action=benchmark_trie_prefixes_key,
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_tries': [TEST_FULLY_POPULATED_TRIES],
                'test_data': [TEST_DATA],
                'iterations': [DEFAULT_ITERATIONS],
                'mark': TEST_MARKS,
            },
        ),
        BenchCase(
            name='trie.prefixed_by(<key>, <search_depth>) (Synthetic)',
            group=benchmark_groups['synthetic-trie-prefixed_by(<key>, <search_depth>)'],
            description=('Timing [yellow bold]trie.prefixed_by(<key>, <search_depth>)[/yellow bold] '
                         'in a fully populated trie'),
            action=benchmark_trie_prefixed_by_key,
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_trie': [TEST_FULLY_POPULATED_TRIES[9]],
                'test_keys': [TEST_DATA[5]],
                'iterations': [DEFAULT_ITERATIONS],
                'mark': [1, 2, 3],
            },
        ),
    ]
    return benchmark_cases_list


def run_benchmarks():
    """Run the benchmark tests and print the results.
    """
    benchmark_cases: list[BenchCase] = get_benchmark_cases()
    console = Console()
    for case in benchmark_cases:
        case.run()
        if case.results:
            console.print(case.results_as_rich_table())
        else:
            console.print('No results available')


def main():
    """Main entry point for running benchmarks."""
    run_benchmarks()


if __name__ == '__main__':
    main()
