#!env python3
# -*- coding: utf-8 -*-
'''
Benchmark for the Generalized Trie implementation.
This script runs a series of tests to measure the performance of the Generalized Trie
against a set of predefined test cases.
'''
# pylint: disable=wrong-import-position, too-many-instance-attributes
# pylint: disable=too-many-positional-arguments, too-many-arguments, too-many-locals

from argparse import ArgumentParser, Namespace
from functools import cache
from dataclasses import dataclass, field
import gc
import gzip
import itertools
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Optional, Sequence

from rich.progress import Progress, TaskID
from rich.table import Table

from gentrie import GeneralizedTrie, GeneralizedKey, TrieId

PROGRESS = Progress(refresh_per_second=5)
"""Progress bar for benchmarking."""

TASKS: dict[str, TaskID] = {}
"""Task IDs for the progress bar."""

MIN_MEASURED_ITERATIONS: int = 3
"""Minimum number of iterations for statistical analysis."""

DEFAULT_ITERATIONS: int = 20
"""Default number of iterations for benchmarking."""

DEFAULT_TIMER = time.perf_counter_ns
"""Default timer function for benchmarking."""

DEFAULT_INTERVAL_SCALE: float = 1e-9
"""Default scaling factor for time intervals (nanoseconds -> seconds)."""

DEFAULT_INTERVAL_UNIT: str = 'ns'
"""Default unit for time intervals (nanoseconds)."""

DEFAULT_OPS_PER_INTERVAL_SCALE: float = 1.0
"""Default scaling factor for operations per interval (1.0 -> 1.0)."""

DEFAULT_OPS_PER_INTERVAL_UNIT: str = 'Ops/s'
"""Default unit for operations per interval (operations per second)."""


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


def generate_fully_populated_trie(test_data: dict[int, list[str]],
                                  symbols: str,
                                  max_depth: int,
                                  value: Optional[Any] = None) -> GeneralizedTrie:
    '''Generate a fully populated Generalized Trie for the given max_depth.

    A fully populated trie contains all possible keys up to the specified depth.
    It uses the pregenerated test_data as the source of truth for the keys for each depth
    because it contains all the possible keys for the depth and symbol set.

    Args:
        max_depth (int): The maximum depth of the trie.
        value (Optional[Any], default=None): The value to assign to each key in the trie.
    '''
    trie = GeneralizedTrie(runtime_validation=False)
    # Use precomputed test_data if available for performance
    for depth, data in test_data.items():
        if depth <= max_depth:
            for key in data:
                trie[key] = value

    # Generate any requested depths NOT included in test_data
    for depth in range(1, max_depth + 1):
        if depth not in test_data:
            # Generate all possible keys for this depth
            for key in generate_test_data(depth, symbols, len(symbols) ** depth):
                trie[key] = value

    return trie


@cache
def load_english_words():
    """Imports English words from a gzipped text file.

    The file contains a bit over 278 thousand words in English
    (one per line).
    """
    words_file = Path(__file__).parent.joinpath("english_words.txt.gz")
    return list(map(str.rstrip, gzip.open(words_file, "rt")))


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
        group (str): The reporting group to which the benchmark case belongs.
        title (str): The name of the benchmark case.
        description (str): A brief description of the benchmark case.
        n (int): The number of rounds the benchmark ran per iteration.
        variation_cols (dict[str, str]): The columns to use for labelling kwarg variations in the benchmark.
        interval_unit (str): The unit of measurement for the interval (e.g. "ns").
        interval_scale (float): The scale factor for the interval (e.g. 1e-9 for nanoseconds).
        ops_per_interval_unit (str): The unit of measurement for operations per interval (e.g. "ops/s").
        ops_per_interval_scale (float): The scale factor for operations per interval (e.g. 1.0 for ops/s).
        total_elapsed (int): The total elapsed time for the benchmark.
        extra_info (dict[str, Any]): Additional information about the benchmark run.
    '''
    group: str
    title: str
    description: str
    n: int
    variation_cols: dict[str, str] = field(default_factory=dict[str, str])
    interval_unit: str = DEFAULT_INTERVAL_UNIT
    interval_scale: float = DEFAULT_INTERVAL_SCALE
    ops_per_interval_unit: str = DEFAULT_INTERVAL_UNIT
    ops_per_interval_scale: float = DEFAULT_INTERVAL_SCALE
    iterations: list[BenchIteration] = field(default_factory=list[BenchIteration])
    ops_per_second: BenchOperationsPerInterval = field(default_factory=BenchOperationsPerInterval)
    op_timings: BenchOperationTimings = field(default_factory=BenchOperationTimings)
    total_elapsed: int = 0
    variation_marks: dict[str, Any] = field(default_factory=dict[str, Any])
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
            'search_depth': [1, 2, 3],
            'runtime_validation': [True, False]
        }

    Args:
        group (str): The benchmark reporting group to which the benchmark case belongs.
        title (str): The name of the benchmark case.
        description (str): A brief description of the benchmark case.
        action (Callable[..., Any]): The action to perform for the benchmark.
        iterations (int): The number of iterations to run for the benchmark.
        min_time (float): The minimum time for the benchmark in seconds. (default: 5.0)
        max_time (float): The maximum time for the benchmark in seconds. (default: 20.0)
        variation_cols (dict[str, str]): kwargs to be used for cols to denote kwarg variations.
        kwargs_variations (dict[str, list[Any]]): Variations of keyword arguments for the benchmark.
        runner (Optional[Callable[..., Any]]): A custom runner for the benchmark.
        verbose (bool): Enable verbose output.
        progress (bool): Enable progress output.
    '''
    group: str
    title: str
    description: str
    action: Callable[..., Any]
    iterations: int = DEFAULT_ITERATIONS
    min_time: float = 5.0  # seconds
    max_time: float = 20.0  # seconds
    variation_cols: dict[str, str] = field(default_factory=dict[str, str])
    kwargs_variations: dict[str, list[Any]] = field(default_factory=dict[str, list[Any]])
    runner: Optional[Callable[..., Any]] = None
    verbose: bool = False
    progress: bool = False
    variations_task: Optional[TaskID] = None

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
        all_variations = self.expanded_kwargs_variations
        task_name: str = 'variations'
        if task_name not in TASKS and self.progress:
            TASKS[task_name] = PROGRESS.add_task(
                description=f'[cyan] Running case {self.title}',
                total=len(all_variations))
        if task_name in TASKS:
            PROGRESS.update(task_id=TASKS[task_name],
                            description=f'[cyan] Running case {self.title}',
                            total=len(all_variations))
        if task_name in TASKS:
            PROGRESS.start_task(TASKS[task_name])
        collected_results: list[BenchResults] = []
        kwargs: dict[str, Any]
        for variations_counter, kwargs in enumerate(all_variations):
            benchmark: BenchmarkRunner = BenchmarkRunner(case=self, kwargs=kwargs)
            results: BenchResults = self.action(benchmark)
            collected_results.append(results)
            if task_name in TASKS:
                PROGRESS.update(task_id=TASKS[task_name],
                                description=(f'[cyan] Running case {self.title} '
                                             f'({variations_counter + 1}/{len(all_variations)})'),
                                completed=variations_counter + 1,
                                refresh=True)
        if task_name in TASKS:
            PROGRESS.stop_task(TASKS[task_name])
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

    def ops_results_as_rich_table(self) -> None:
        """Prints the benchmark results in a rich table format if available.
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

        table = Table(title=(self.title + '\n\n' + self.description),
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
        for value in self.variation_cols.values():
            table.add_column(value, justify='center', vertical='bottom', overflow='fold')
        for result in self.results:
            row: list[str] = [
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
                f'{result.ops_per_second.relative_standard_deviation:>3.2f}%'
            ]
            for value in result.variation_marks.values():
                row.append(f'{value!s}')
            table.add_row(*row)
        PROGRESS.console.print(table)


class BenchmarkRunner():
    """A class to run benchmarks for various actions.
    """
    def __init__(self,
                 case: BenchCase,
                 kwargs: dict[str, Any],
                 runner: Optional[Callable[..., Any]] = None):
        self.case: BenchCase = case
        self.kwargs: dict[str, Any] = kwargs
        self.run: Callable[..., Any] = runner if runner is not None else self.default_runner

    @property
    def variation_marks(self) -> dict[str, Any]:
        '''Return the variation marks for the benchmark.

        The variation marks identify the specific variations being tested in a run
        from the kwargs values.
        '''
        return {key: self.kwargs.get(key, None) for key in self.case.variation_cols.keys()}

    def default_runner(
            self,
            n: int,
            action: Callable[..., Any],
            setup: Optional[Callable[..., Any]] = None,
            teardown: Optional[Callable[..., Any]] = None) -> BenchResults:
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
            variation_cols (dict[str, str]): The variation columns to use for the benchmark.
            n (int): The number of test rounds that will be run by the action on each iteration.
            action (Callable[..., Any]): The action to benchmark.
            setup (Optional[Callable[..., Any]]): A setup function to run before each iteration.
            teardown (Optional[Callable[..., Any]]): A teardown function to run after each iteration.
        """
        group: str = self.case.group
        title: str = self.case.title
        description: str = self.case.description
        min_time: float = self.case.min_time
        max_time: float = self.case.max_time
        iterations: int = self.case.iterations

        iteration_pass: int = 0
        time_start: int = DEFAULT_TIMER()
        max_stop_at: int = int(max_time / DEFAULT_INTERVAL_SCALE) + time_start
        min_stop_at: int = int(min_time / DEFAULT_INTERVAL_SCALE) + time_start
        wall_time: int = DEFAULT_TIMER()
        iterations_min: int = max(MIN_MEASURED_ITERATIONS, iterations)

        gc.collect()

        tasks_name = 'runner'

        progress_max: float = 100.0
        if self.case.progress and tasks_name not in TASKS:
            TASKS[tasks_name] = PROGRESS.add_task(
                            description=f'[green] Benchmarking {group}',
                            total=progress_max)
        if tasks_name in TASKS:
            PROGRESS.update(TASKS[tasks_name],
                            completed=5.0,
                            description=f'[green] Benchmarking {group} (iteration {iteration_pass:<6d}; '
                                        f'time {0.00:<3.2f}s)')
            PROGRESS.start_task(TASKS[tasks_name])
        total_elapsed: float = 0
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

            # Update progress display if showing progress
            if tasks_name in TASKS:
                iteration_completion: float = progress_max * iteration_pass / iterations_min
                wall_time_elapsed_seconds: float = (wall_time - time_start) * DEFAULT_INTERVAL_SCALE
                time_completion: float = progress_max * (wall_time - time_start) / (min_stop_at - time_start)
                progress_current = min(iteration_completion, time_completion)
                PROGRESS.update(TASKS[tasks_name],
                                completed=progress_current,
                                description=(
                                    f'[green] Benchmarking {group} (iteration {iteration_pass:6d}; '
                                    f'time {wall_time_elapsed_seconds:<3.2f}s)'))

        benchmark_results = BenchResults(
            group=group,
            title=title,
            description=description,
            variation_marks=self.variation_marks,
            n=n,
            iterations=iterations_list,
            total_elapsed=total_elapsed,
            extra_info={})

        if tasks_name in TASKS:
            PROGRESS.stop_task(TASKS[tasks_name])

        return benchmark_results


def benchmark_build_with_add(benchmark: BenchmarkRunner) -> BenchResults:
    '''Benchmark the addition of keys to the trie.

    Args:
        benchmark (BenchmarkRunner): The benchmark runner instance.

    Returns (BenchResults):
        The results of the benchmark.
    '''
    kwargs = benchmark.kwargs
    depth = kwargs['depth']
    test_keys = kwargs['test_data'][depth]
    runtime_validation = kwargs['runtime_validation']

    def action_to_benchmark():
        trie = GeneralizedTrie(runtime_validation=runtime_validation)
        for key in test_keys:
            trie.add(key, None)

    return benchmark.run(action=action_to_benchmark, n=len(test_keys))


def benchmark_build_with_assign(benchmark: BenchmarkRunner) -> BenchResults:
    '''Benchmark the assignment of keys to the trie.

    Args:
        benchmark (BenchmarkRunner): The benchmark runner instance.

    Returns (BenchResults):
        The results of the benchmark.
    '''
    kwargs: dict[str, Any] = benchmark.kwargs
    depth = kwargs['depth']
    test_keys = kwargs['test_data'][depth]
    runtime_validation = kwargs['runtime_validation']

    def action_to_benchmark():
        trie = GeneralizedTrie(runtime_validation=runtime_validation)
        for key in test_keys:
            trie[key] = None

    return benchmark.run(n=len(test_keys), action=action_to_benchmark)


def benchmark_build_with_update(benchmark: BenchmarkRunner) -> BenchResults:
    '''Benchmark the building of a trie using update().

    Args:
        benchmark (BenchmarkRunner): The benchmark runner instance.

    Returns (BenchResults):
        The results of the benchmark.
    '''
    kwargs: dict[str, Any] = benchmark.kwargs
    depth = kwargs['depth']
    test_keys = kwargs['test_data'][depth]
    runtime_validation = kwargs['runtime_validation']

    def action_to_benchmark():
        trie = GeneralizedTrie(runtime_validation=runtime_validation)
        for key in test_keys:
            trie.update(key, None)

    return benchmark.run(n=len(test_keys), action=action_to_benchmark)


def benchmark_updating_trie(benchmark: BenchmarkRunner) -> BenchResults:
    '''Benchmark update() operations on keys in an existing trie.

    This is effectively a benchmark of code like this where all test keys
    are already in the trie and updated to the same value.

    ```
    for key in test_keys:
        trie.update(key, 1)
    ```
    Args:
        benchmark (BenchmarkRunner): The benchmark runner instance.

    Returns:
        A list of BenchResults containing the benchmark results.
    '''
    # Build the prefix tree - built here because we are modifying it
    # and don't want to modify the pre-generated test tries
    kwargs = benchmark.kwargs
    depth = kwargs['depth']
    test_keys = kwargs['test_data'][depth]
    test_args_data: list[tuple[GeneralizedKey, int]] = list([(key, 1) for key in test_keys])
    if len(test_keys) != len(test_args_data):
        raise ValueError("Test keys and args data length mismatch")
    trie = generate_test_trie_from_data(data=test_keys, value=None)
    trie.runtime_validation = kwargs['runtime_validation']

    def action_to_benchmark():
        for key in test_keys:
            trie.update(key, None)

    return benchmark.run(n=len(test_keys), action=action_to_benchmark)


def benchmark_remove_key_from_trie(benchmark: BenchmarkRunner) -> BenchResults:
    '''Benchmark remove() operations on keys in an existing trie.

    This is effectively a benchmark of code like this where all test keys
    are already in the trie and updated to the same value.

    ```
    for key in test_keys:
        trie.remove(key)
    ```
    Args:
        benchmark (BenchmarkRunner): The benchmark runner instance.

    Returns:
        A list of BenchResults containing the benchmark results.
    '''
    kwargs = benchmark.kwargs
    depth = kwargs['depth']
    test_keys = kwargs['test_data'][depth]
    runtime_validation = kwargs['runtime_validation']
    trie: GeneralizedTrie = GeneralizedTrie(runtime_validation=runtime_validation)

    def setup():
        "Setup the trie with test keys."
        for key in test_keys:
            trie.update(key, None)

    def action_to_benchmark():
        "Remove all test keys from the trie."
        for key in test_keys:
            trie.remove(key)

    def teardown():
        "Reset the trie after the benchmark iteration."
        trie.clear()

    return benchmark.run(n=len(test_keys), action=action_to_benchmark, setup=setup, teardown=teardown)


def benchmark_del_key_from_trie(benchmark: BenchmarkRunner) -> BenchResults:
    '''Benchmark "del trie[<key>] operations on keys in an existing trie.

    This is effectively a benchmark of code like this where all test keys
    are already in the trie.

    ```
    for key in test_keys:
        del trie[key]
    ```
    Args:
        benchmark (BenchmarkRunner): The benchmark runner instance.

    Returns:
        A list of BenchResults containing the benchmark results.
    '''
    kwargs = benchmark.kwargs
    depth = kwargs['depth']
    test_keys = kwargs['test_data'][depth]
    runtime_validation = kwargs['runtime_validation']
    trie: GeneralizedTrie = GeneralizedTrie(runtime_validation=runtime_validation)

    def setup():
        "Setup the trie with test keys."
        for key in test_keys:
            trie.update(key, None)

    def action_to_benchmark():
        "Remove all test keys from the trie using del operator"
        for key in test_keys:
            del trie[key]

    def teardown():
        "Clear the trie after the benchmark iteration."
        trie.clear()

    return benchmark.run(n=len(test_keys), action=action_to_benchmark, setup=setup, teardown=teardown)


def benchmark_del_id_from_trie(benchmark: BenchmarkRunner) -> BenchResults:
    '''Benchmark "del trie[<key>] operations on keys in an existing trie.

    This is effectively a benchmark of code like this where all test keys
    are already in the trie.

    ```
    for key in test_keys:
        del trie[key]
    ```
    Args:
        benchmark (BenchmarkRunner): The benchmark runner instance.

    Returns:
        A list of BenchResults containing the benchmark results.
    '''
    kwargs = benchmark.kwargs
    depth = kwargs['depth']
    test_keys: Sequence[GeneralizedKey] = kwargs['test_data'][depth]
    runtime_validation = kwargs['runtime_validation']
    test_ids: list[TrieId] = []
    trie: GeneralizedTrie = GeneralizedTrie(runtime_validation=runtime_validation)

    def setup():
        "Setup the trie with test keys and get the TrieIds for deletion."
        for key in test_keys:
            trie.update(key, None)
        test_ids.extend(trie.keys())

    def action_to_benchmark():
        "Remove all test keys from the trie."
        for trie_id in test_ids:
            del trie[trie_id]

    def teardown():
        "Reset the trie and test ids after the benchmark iteration."
        test_ids.clear()
        trie.clear()

    return benchmark.run(n=len(test_keys), action=action_to_benchmark, setup=setup, teardown=teardown)


def benchmark_key_in_trie(benchmark: BenchmarkRunner) -> BenchResults:
    '''Benchmark '<key> in <trie>' operations.

    Args:
        benchmark (BenchmarkRunner): The benchmark runner instance.

    Returns:
        A list of BenchResults containing the benchmark results.
    '''
    kwargs = benchmark.kwargs
    dataset = kwargs['dataset']
    test_keys: list[GeneralizedKey] = kwargs['test_data'][dataset]
    trie: GeneralizedTrie = kwargs['test_tries'][dataset]
    trie.runtime_validation = kwargs['runtime_validation']

    def action_to_benchmark():
        "Check if all test keys are in the trie."
        for key in test_keys:
            _ = key in trie

    return benchmark.run(n=len(test_keys), action=action_to_benchmark)


def benchmark_id_in_trie(benchmark: BenchmarkRunner) -> BenchResults:
    '''Benchmark '<TrieId> in trie' operations.

    Args:
        benchmark (BenchmarkRunner): The benchmark runner instance.

    Returns:
        A list of BenchResults containing the benchmark results.
    '''
    kwargs: dict[str, Any] = benchmark.kwargs
    dataset = kwargs['dataset']
    trie: GeneralizedTrie = kwargs['test_tries'][dataset]
    trie.runtime_validation = kwargs['runtime_validation']
    test_ids: list[TrieId] = list(trie.keys())

    def action_to_benchmark():
        for key in test_ids:
            _ = key in trie

    return benchmark.run(n=len(test_ids), action=action_to_benchmark)


def benchmark_trie_prefixes_key(benchmark: BenchmarkRunner) -> BenchResults:
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
        benchmark (BenchmarkRunner): The benchmark runner instance.

    Returns:
        A list of BenchResults containing the benchmark results.
    '''
    kwargs: dict[str, Any] = benchmark.kwargs
    depth = kwargs['depth']
    test_keys: list[str] = kwargs['test_keys'][depth]
    trie: GeneralizedTrie = kwargs['test_tries'][depth]
    trie.runtime_validation = kwargs['runtime_validation']

    def action_to_benchmark():
        for key in test_keys:
            _ = list(trie.prefixes(key))

    return benchmark.run(n=len(test_keys), action=action_to_benchmark)


def benchmark_trie_prefixed_by_key(benchmark: BenchmarkRunner) -> BenchResults:
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
        benchmark (BenchmarkRunner): The benchmark runner instance.

    Returns:
        A list of BenchResults containing the benchmark results.
    '''
    kwargs: dict[str, Any] = benchmark.kwargs
    search_depth = kwargs['search_depth']
    if not isinstance(search_depth, int):
        raise TypeError(f"Expected 'search_depth' to be int, got {type(search_depth).__name__}")
    trie: GeneralizedTrie = kwargs['test_trie']
    trie.runtime_validation = kwargs['runtime_validation']
    test_keys: list[GeneralizedKey] = kwargs['test_data']

    def action_to_benchmark():
        for key in test_keys:
            _ = list(trie.prefixed_by(key, search_depth))

    return benchmark.run(n=len(test_keys), action=action_to_benchmark)


@cache
def get_benchmark_cases() -> list[BenchCase]:
    """
    Define the benchmark cases to be run.
    """

    symbols: str = '0123'  # Define the symbols for the trie

    test_data: dict[int, list[str]] = {}
    test_depths: list[int] = [3, 4, 5, 6, 7, 8, 9]  # Depths to test - 1 and 2 are omitted due to low key counts
    for gen_depth in test_depths:
        max_keys_for_depth = len(symbols) ** gen_depth  # pylint: disable=invalid-name
        test_data[gen_depth] = generate_test_data(gen_depth, symbols, max_keys=max_keys_for_depth)

    # We generate the test_tries from the test_data for synchronization
    test_tries: dict[int, GeneralizedTrie] = {}
    for gen_depth in test_depths:
        test_tries[gen_depth] = generate_test_trie_from_data(test_data[gen_depth], None)

    # We generate the test_missing_key_tries from the test_data for synchronization
    test_missing_key_tries: dict[int, tuple[GeneralizedTrie, str]] = {}
    for gen_depth in test_depths:
        test_missing_key_tries[gen_depth] = generate_trie_with_missing_key_from_data(test_data[gen_depth], None)

    test_fully_populated_tries: dict[int, GeneralizedTrie] = {}
    for gen_depth in test_depths:
        test_fully_populated_tries[gen_depth] = generate_fully_populated_trie(
                                                    test_data=test_data,
                                                    symbols=symbols,
                                                    max_depth=gen_depth)

    english_words = load_english_words()
    test_organic_data: dict[str, list[str]] = {
        'english': english_words
    }
    test_organic_tries: dict[str, GeneralizedTrie] = {
        'english': generate_test_trie_from_data(english_words, None)
    }

    benchmark_cases_list: list[BenchCase] = [
        BenchCase(
            group='synthetic-id-in-trie',
            title='<TrieId> in trie (Synthetic)',
            description='Timing [yellow bold]<TrieId> in trie[/yellow bold] with synthetic data',
            action=benchmark_id_in_trie,
            iterations=DEFAULT_ITERATIONS,
            variation_cols={'dataset': 'Depth', 'runtime_validation': 'Runtime Validation'},
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_tries': [test_tries],
                'test_data': [test_data],
                'dataset': test_depths,
            }
        ),
        BenchCase(
            group='synthetic-key-in-trie',
            title='<key> in trie (Synthetic)',
            description='Timing [yellow bold]<key> in trie[/yellow bold] with synthetic data',
            action=benchmark_key_in_trie,
            iterations=DEFAULT_ITERATIONS,
            variation_cols={'dataset': 'Depth', 'runtime_validation': 'Runtime Validation'},
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_tries': [test_tries],
                'test_data': [test_data],
                'dataset': test_depths,
            }
        ),
        BenchCase(
            group='english-dictionary-id-in-trie',
            title='<TrieId> in trie (English)',
            description=(
                'Timing [yellow bold]<TrieId> in trie[/yellow bold] with words from the English dictionary'),
            action=benchmark_id_in_trie,
            iterations=DEFAULT_ITERATIONS,
            variation_cols={'dataset': 'Dataset', 'runtime_validation': 'Runtime Validation'},
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_tries': [test_organic_tries],
                'dataset': ['english'],
            }
        ),
        BenchCase(
            group='english-dictionary-key-in-trie',
            title='<key> in trie (English)',
            description=('Timing [yellow bold]<key> in trie[/yellow bold] with words from the English dictionary'),
            action=benchmark_key_in_trie,
            iterations=DEFAULT_ITERATIONS,
            variation_cols={'dataset': 'Dataset', 'runtime_validation': 'Runtime Validation'},
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_tries': [test_organic_tries],
                'test_data': [test_organic_data],
                'dataset': ['english'],
            }
        ),
        BenchCase(
            group='synthetic-building-trie-add',
            title='trie.add(<key>, <value>) (Synthetic)',
            description=('Timing [yellow bold]trie.add(<key>, <value>)[/yellow bold] '
                         'while building a new trie with synthetic data'),
            action=benchmark_build_with_add,
            iterations=DEFAULT_ITERATIONS,
            variation_cols={'depth': 'Depth', 'runtime_validation': 'Runtime Validation'},
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_data': [test_data],
                'depth': test_depths,
            }
        ),
        BenchCase(
            group='synthetic-building-trie-update',
            title='trie.update(<key>, <value>) (Synthetic)',
            description=('Timing [yellow bold]trie.update(<key>, <value>)[/yellow bold] '
                         'while building a new trie with synthetic data'),
            action=benchmark_build_with_update,
            iterations=DEFAULT_ITERATIONS,
            variation_cols={'depth': 'Depth', 'runtime_validation': 'Runtime Validation'},
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_data': [test_data],
                'depth': test_depths,
            }
        ),
        BenchCase(
            group='synthetic-building-trie-assign',
            title='trie[<key>] = <value> (Synthetic)',
            description=('Timing [yellow bold]trie[<key>] = <value>[/yellow bold] '
                         'while building a new trie with synthetic data'),
            action=benchmark_build_with_assign,
            iterations=DEFAULT_ITERATIONS,
            variation_cols={'depth': 'Depth', 'runtime_validation': 'Runtime Validation'},
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_data': [test_data],
                'depth': test_depths,
            }
        ),
        BenchCase(
            group='synthetic-updating-trie-update',
            title='trie.update(<key>, <value>) (Synthetic)',
            description=('Timing [yellow bold]trie.update(<key>, <value>)[/yellow bold] '
                         'while updating values for existing keys with synthetic data'),
            action=benchmark_updating_trie,
            iterations=DEFAULT_ITERATIONS,
            variation_cols={'depth': 'Depth', 'runtime_validation': 'Runtime Validation'},
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_data': [test_data],
                'depth': test_depths,
            }
        ),
        BenchCase(
            group='synthetic-updating-trie-remove',
            title='trie.remove(<key>) (Synthetic)',
            description=('Timing [yellow bold]trie.remove(<key>)[/yellow bold] '
                         'while removing keys from a trie with synthetic data'),
            action=benchmark_remove_key_from_trie,
            variation_cols={'depth': 'Depth', 'runtime_validation': 'Runtime Validation'},
            iterations=DEFAULT_ITERATIONS,
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_data': [test_data],
                'depth': test_depths,
            },
        ),
        BenchCase(
            group='synthetic-updating-trie-del-key',
            title='del trie[<key>] (Synthetic)',
            description=('Timing [yellow bold]del trie[<key>][/yellow bold] '
                         'while deleting keys from a trie with synthetic data'),
            action=benchmark_del_key_from_trie,
            variation_cols={'depth': 'Depth', 'runtime_validation': 'Runtime Validation'},
            iterations=DEFAULT_ITERATIONS,
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_data': [test_data],
                'depth': test_depths,
            },
        ),
        BenchCase(
            group='synthetic-updating-trie-del-id',
            title='del trie[<TrieId>] (Synthetic)',
            description=('Timing [yellow bold]del trie[<TrieId>][/yellow bold] '
                         'while deleting keys from a trie with synthetic data'),
            action=benchmark_del_id_from_trie,
            variation_cols={'depth': 'Depth', 'runtime_validation': 'Runtime Validation'},
            iterations=DEFAULT_ITERATIONS,
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_data': [test_data],
                'depth': test_depths,
            },
        ),
        BenchCase(
            group='synthetic-trie-prefixes',
            title='trie.prefixes(<key>) (Synthetic)',
            description=('Timing [yellow bold]trie.prefixes(<key>)[/yellow bold] '
                         'while finding keys matching a specific prefix in a trie with synthetic data'),
            action=benchmark_trie_prefixes_key,
            variation_cols={'depth': 'Depth', 'runtime_validation': 'Runtime Validation'},
            iterations=DEFAULT_ITERATIONS,
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_tries': [test_fully_populated_tries],
                'test_keys': [test_data],
                'depth': test_depths,
            },
        ),
        BenchCase(
            group='synthetic-trie-prefixed_by',
            title='trie.prefixed_by(<key>, <search_depth>) (Synthetic)',
            description=('Timing [yellow bold]trie.prefixed_by(<key>, <search_depth>)[/yellow bold] '
                         'in a fully populated trie'),
            action=benchmark_trie_prefixed_by_key,
            variation_cols={'search_depth': 'Search Depth', 'runtime_validation': 'Runtime Validation'},
            iterations=DEFAULT_ITERATIONS,
            kwargs_variations={
                'runtime_validation': [False, True],
                'test_trie': [test_fully_populated_tries[9]],
                'test_keys': [test_data[5]],
                'search_depth': [1, 2, 3],
            },
        ),
    ]
    return benchmark_cases_list


def run_benchmarks(args: Namespace):
    """Run the benchmark tests and print the results.
    """
    benchmark_cases: list[BenchCase] = get_benchmark_cases()
    for case in benchmark_cases:
        case.verbose = args.verbose
        case.progress = args.progress

    if args.progress:
        PROGRESS.start()
    try:
        cases_to_run: list[BenchCase] = []
        for case in benchmark_cases:
            if not (args.run == 'all' or case.group in args.run):
                continue
            cases_to_run.append(case)

        task_name: str = 'cases'
        if task_name not in TASKS and args.progress:
            TASKS[task_name] = PROGRESS.add_task(
                description='Running benchmark cases',
                total=len(cases_to_run))

        for case_counter, case in enumerate(cases_to_run):
            if task_name in TASKS:
                PROGRESS.update(
                    task_id=TASKS[task_name],
                    completed=case_counter,
                    description=f'Running benchmark cases (case {case_counter + 1:2d}/{len(cases_to_run)})')
            case.run()
            if case.results:
                if args.json:
                    if args.ops:
                        case.ops_results_as_json()
                    if args.timing:
                        case.timing_results_as_json()

                if args.tcsv:
                    if args.ops:
                        case.ops_results_as_tagged_csv()
                    if args.timing:
                        case.timing_results_as_tagged_csv()
                if args.console:
                    if args.ops:
                        case.ops_results_as_rich_table()
                    if args.timing:
                        case.timing_results_as_rich_table()
            else:
                PROGRESS.console.print('No results available')
        if task_name in TASKS:
            PROGRESS.update(
                task_id=TASKS[task_name],
                completed=len(cases_to_run),
                description=f'Running benchmark cases (case {case_counter + 1:2d}/{len(cases_to_run)})')
        TASKS.clear()
    except KeyboardInterrupt:
        PROGRESS.console.print('Benchmarking interrupted by keyboard interrupt')
    except Exception as exc:  # pylint: disable=broad-exception-caught
        PROGRESS.console.print(f'Error occurred while running benchmarks: {exc}')
    finally:
        if args.progress:
            PROGRESS.stop()


def main():
    """Main entry point for running benchmarks."""
    parser = ArgumentParser(description='Run GeneralizedTrie benchmarks.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--progress', action='store_true', help='Enable progress output')
    parser.add_argument('--list', action='store_true', help='List all available benchmarks')
    parser.add_argument('--run', nargs="+", default='all', metavar='<benchmark>', help='Run specific benchmarks')
    parser.add_argument('--console', action='store_true', help='Enable console output')
    parser.add_argument('--json', action='store_true', help='Enable JSON file output to files')
    parser.add_argument('--tcsv', action='store_true', help='Enable tagged CSV output to files')
    parser.add_argument('--output_dir', default='.', help='Output destination directory (default: .)')
    parser.add_argument('--ops', action='store_true', help='Enable operations per second output')
    parser.add_argument('--timing', action='store_true', help='Enable operations timing output')

    args: Namespace = parser.parse_args()
    if args.verbose:
        PROGRESS.console.print('Verbose output enabled')

    if args.list:
        PROGRESS.console.print('Available benchmarks:')
        for case in get_benchmark_cases():
            PROGRESS.console.print('  - ', f'[green]{case.group:<40s}[/green]', f'{case.title}')
        return

    if not (args.console or args.json or args.tcsv):
        PROGRESS.console.print('No output format(s) selected, using console output by default')
        args.console = True

    if not (args.ops or args.timing):
        PROGRESS.console.print('No benchmark result type selected: At least one of --ops or --timing must be enabled')
        parser.print_usage()
        return


    run_benchmarks(args=args)


if __name__ == '__main__':
    main()
