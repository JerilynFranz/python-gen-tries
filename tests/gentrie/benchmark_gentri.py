from random import randint
import time
from typing import Literal, NamedTuple, Sequence

from src.gentrie import GeneralizedTrie

SYMBOLS: str = '0123456789ABCDEFGHIJKLMNIOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'  # Define the symbols for the trie


class GentriBenchmark(NamedTuple):
    description: str
    nums: list[int]


if __name__ == '__main__':
    n: int = 1000000
   test_cases: Sequence[GentriBenchmark] = []




    solution: Solution = Solution()
    editorial_solution: EditorialSolution = EditorialSolution()

    iterations: int = 10

    counter: int = iterations
    null_timer: dict[Literal['start', 'end', 'elapsed'], int] = {}
    null_timer['start'] = time.process_time_ns()
    while counter > 0:
        counter -= 1
    null_timer['end'] = time.process_time_ns()
    null_timer['elapsed'] = null_timer['end'] - null_timer['start']
    print('Null loop timing: ', null_timer['elapsed'] / iterations, ' ns')

    for test in test_cases:
        solution_timer: dict[Literal['start', 'end', 'elapsed'], int] = {}
        editorial_solution_timer: dict[Literal['start', 'end', 'elapsed'], int] = {}

        counter = iterations
        solution_timer['start'] = time.process_time_ns()
        while counter > 0:
            counter -= 1
            _ = solution.longestMonotonicSubarray(test.nums)
        solution_timer['end'] = time.process_time_ns()
        solution_timer['elapsed'] = solution_timer['end'] - solution_timer['start']

        counter = iterations
        editorial_solution_timer['start'] = time.process_time_ns()
        while counter > 0:
            counter -= 1
            _ = editorial_solution.longestMonotonicSubarray(test.nums)
        editorial_solution_timer['end'] = time.process_time_ns()
        editorial_solution_timer['elapsed'] = editorial_solution_timer['end'] - editorial_solution_timer['start']

        print(f'Test Case: {test.description}, n = {n}, iterations = {iterations}')
        print('  solution timing:           ', solution_timer['elapsed'] / (iterations * 1000000000), ' seconds')
        print('  editorial solution timing: ', editorial_solution_timer['elapsed'] / (iterations * 1000000000),
              ' seconds')
        print('  Solution / Editorial Solution: ', solution_timer['elapsed'] / editorial_solution_timer['elapsed'])
