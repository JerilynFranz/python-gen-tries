import traceback
from typing import Any, Callable, Dict, List, NamedTuple, Optional
import unittest

from gentries.generalized import GeneralizedTrie, InvalidTokenError


class NoExpectedValue():
    """This is used to distinguish between having an expected return value
    of None and and not expecting a value."""
    pass


class TestConfig(NamedTuple):
    name: str
    action: Callable
    args: List[Any] = list()
    kwargs: Dict[str, Any] = dict()
    expected: Any = NoExpectedValue()
    obj: Optional[Any] = None
    validate_obj: Optional[Callable] = None
    validate_result: Optional[Callable] = None
    exception: Optional[Exception] = None
    exception_tag: Optional[str] = None


def run_tests_list(self, tests_list: List[TestConfig]) -> None:
    for test in tests_list:
        run_test(self, test)


def run_test(self, entry: TestConfig) -> None:
    with self.subTest(msg=entry.name):
        test_description: str = (
            f'{entry.name}')
        errors: List[str] = []
        try:
            found: Any = entry.action(*entry.args, **entry.kwargs)
            if entry.exception:
                errors.append('returned result instead of raising exception')

            else:
                if entry.validate_result and not entry.validate_result(found):
                    errors.append(f'failed result validation: found={found}')
                if entry.validate_obj and not entry.validate_obj(entry.obj):
                    errors.append(f'failed object validation: obj={entry.obj}')
                if (not isinstance(entry.expected, NoExpectedValue)
                        and entry.expected != found):
                    errors.append(f'expected={entry.expected}, found={found}')
        except Exception as err:  # pylint: disable=broad-exception-caught
            if entry.exception is None:
                errors.append(
                    f'Did not expect exception. Caught exception {repr(err)}')
                errors.append('stacktrace = ')
                errors.append('\n'.join(
                    traceback.format_tb(tb=err.__traceback__)))

            if not (entry.exception and isinstance(err, entry.exception)):
                errors.append(
                    f'Unexpected exception type: expected={entry.exception}, '
                    f'found = {repr(err)}')

            elif entry.exception_tag:
                if str(err).find(entry.exception_tag) == -1:
                    errors.append(
                        f'correct exception type, but tag '
                        f'{entry.exception_tag} not found: {repr(err)}')
        if errors:
            self.fail(msg=test_description + ': ' + '\n'.join(errors))


class TestGeneralizedTrie(unittest.TestCase):

    def test_create_trie(self) -> None:
        tests: List[TestConfig] = [
             TestConfig(
                name='[TCT001] create GeneralizedTrie()',
                action=GeneralizedTrie,
                validate_result=lambda found: isinstance(found, GeneralizedTrie)),
             TestConfig(
                name='[TCT002] create GeneralizedTrie(filter_id=1)',
                action=GeneralizedTrie,
                kwargs={'filter_id': 1},
                exception=TypeError)
        ]
        run_tests_list(self, tests)

    def test_add(self) -> None:
        trie: GeneralizedTrie = GeneralizedTrie()
        tests: List[TestConfig] = [
            TestConfig(
                name="[TA001] trie.add(['tree', 'value', 'ape'])",
                action=trie.add,
                args=[['tree', 'value', 'ape']],
                kwargs={},
                expected=1),
            TestConfig(
                name="[TA002] trie.add(['tree', 'value']",
                action=trie.add,
                args=[['tree', 'value']],
                expected=2),
            TestConfig(
                name="[TA003] trie.add('abcdef')",
                action=trie.add,
                args=['abcdef'],
                expected=3),
            TestConfig(
                name="[TA004] trie.add(1, 3, 4, 5])",
                action=trie.add,
                args=[[1, 3, 4, 5]],
                kwargs={},
                expected=4),
            TestConfig(
                name="[TA006] trie.add(frozenset([1]), 3, 4, 5])",
                action=trie.add,
                args=[[frozenset([1]), 3, 4, 5]],
                kwargs={},
                expected=5),
            TestConfig(
                name="[TA007] trie.add(frozenset([1]), 3, 4, 5])",
                action=trie.add,
                args=[[frozenset([1]), 3, 4, 6]],
                expected=6),
            TestConfig(
                name="[TA008] trie.add(1)",
                action=trie.add,
                args=[1],
                exception=TypeError,
                exception_tag='[GTAFBT001]'),
            TestConfig(
                name="[TA009] trie.add([])",
                action=trie.add,
                args=[[]],
                exception=ValueError,
                exception_tag='[GTAFBT002]'),
            TestConfig(
                name="[TA010] trie.add(set([1]), 3, 4, 5])",
                action=trie.add,
                args=[[set([1]), 3, 4, 5]],
                exception=InvalidTokenError,
                exception_tag='[GTAFBT003]'),
            TestConfig(
                name="[TA011] trie.add(key=[1, 3, 4, 7])",
                action=trie.add,
                kwargs={'key': [1, 3, 4, 7]},
                exception=TypeError),
            TestConfig(
                name="[TA012] trie.add()",
                action=trie.add,
                exception=TypeError),
            TestConfig(
                name="[TA013] trie.add(['a'], ['b'])",
                action=trie.add,
                args=[['a'], ['b']],
                exception=TypeError),
            TestConfig(
                name="[TA014] len(trie)",
                action=len,
                args=[trie],
                expected=6),
        ]
        run_tests_list(self, tests)

    def test_prefixes(self) -> None:
        trie: GeneralizedTrie = GeneralizedTrie()
        tests: List[TestConfig] = [
            TestConfig(
                name="[TTP001] trie.add(['tree', 'value', 'ape'])",
                action=trie.add,
                args=[['tree', 'value', 'ape']],
                expected=1),
            TestConfig(
                name="[TTP002] trie.add(['tree', 'value']",
                action=trie.add,
                args=[['tree', 'value']],
                expected=2),
            TestConfig(
                name="[TTP003] trie.add('abcdef')",
                action=trie.add,
                args=['abcdef'],
                kwargs={},
                expected=3),
            TestConfig(
                name="[TTP004] trie.add('abc')",
                action=trie.add,
                args=['abc'],
                expected=4),
            TestConfig(
                name="[TTP005] trie.prefixes(['tree', 'value', 'ape'])",
                action=trie.prefixes,
                args=[['tree', 'value', 'ape']],
                expected=set([1, 2])),
            TestConfig(
                name="[TTP006] trie.prefixes(['tree', 'value'])",
                action=trie.prefixes,
                args=[['tree', 'value']],
                expected=set([2])),
            TestConfig(
                name="[TTP007] trie.prefixes('a')",
                action=trie.prefixes,
                args=['a'],
                expected=set()),
            TestConfig(
                name="[TTP008] trie.prefixes('abc')",
                action=trie.prefixes,
                args=['abc'],
                expected=set([4])),
            TestConfig(
                name="[TTP009] trie.prefixes('abcd')",
                action=trie.prefixes,
                args=['abcd'],
                expected=set([4])),
            TestConfig(
                name="[TTP010] trie.prefixes(['abc'])",
                action=trie.prefixes,
                args=[['abc']],
                expected=set()),
            TestConfig(
                name="[TTP011] trie.add([1,3,4])",
                action=trie.add,
                args=[[1, 3, 4]],
                expected=5),
            TestConfig(
                name="[TTP012] trie.prefixes([1, 3, 4, 5, 6, ])",
                action=trie.prefixes,
                args=[[1, 3, 4, 5, 6]],
                expected=set([5])),
            TestConfig(
                name="[TTP013] trie.prefixes(['a', 3, 4, 5])",
                action=trie.prefixes,
                args=[['a', 3, 4, 5]],
                expected=set()),
            TestConfig(
                name="[TTP014] trie.add(frozenset([1]), 3, 4, 5])",
                action=trie.add,
                args=[[frozenset([1]), 3, 4, 5]],
                expected=6),
            TestConfig(
                name="[TTP015] trie.prefixes([frozenset([1]), 3, 4, 5])",
                action=trie.prefixes,
                args=[[frozenset([1]), 3, 4, 5]],
                expected=set([6])),
            TestConfig(
                name="[TTP017] trie.prefixes(tokens=[frozenset([1]), 3, 4, 5])",
                action=trie.prefixes,
                kwargs={'tokens': [frozenset([1]), 3, 4, 5]},
                expected=set([6])),
            TestConfig(
                name="[TTP018] trie.prefixes()",
                action=trie.prefixes,
                exception=TypeError),
            TestConfig(
                name="[TTP019] trie.prefixes(None)",
                action=trie.prefixes,
                args=[None],
                exception=TypeError,
                exception_tag='[GTM001]'),
        ]
        run_tests_list(self, tests)

    def test_remove(self) -> None:
        trie: GeneralizedTrie = GeneralizedTrie()
        tests: List[TestConfig] = [
            TestConfig(
                name="[TR001] trie.add('abc')",
                action=trie.add,
                args=['abc'],
                expected=1),
            TestConfig(
                name="[TR002] trie.tokens_prefix('abcde')",
                action=trie.prefixes,
                args=['abcde'],
                expected=set([1])),
            TestConfig(
                name="[TR003] len(trie)",
                action=len,
                args=[trie],
                expected=1),
            TestConfig(
                name="[TR004] trie.remove(2)",
                action=trie.remove,
                args=[2],
                exception=KeyError,
                exception_tag='[GTR003]'),
            TestConfig(
                name="[TR005] len(trie)",
                action=len,
                args=[trie],
                expected=1),
            TestConfig(
                name="[TR006] trie.remove(1)",
                action=trie.remove,
                args=[1],
                expected=None),
            TestConfig(
                name="[TR007] len(trie)",
                action=len,
                args=[trie],
                expected=0),
            TestConfig(
                name="[TR008] trie.remove('abc')",
                action=trie.remove,
                args=['abc'],
                exception=TypeError,
                exception_tag='[GTR001]'),
            TestConfig(
                name="[TR009] trie.remove(0)",
                action=trie.remove,
                args=[0],
                exception=KeyError,
                exception_tag='[GTR002]'),
            TestConfig(
                name="[TR010] trie.remove(1)",
                action=trie.remove,
                args=[1],
                exception=KeyError,
                exception_tag='[GTR003]'),
            TestConfig(
                name="[TR011] trie.tokens_prefix('abcde')",
                action=trie.prefixes,
                args=['abcde'],
                expected=set())
        ]
        run_tests_list(self, tests)


if __name__ == '__main__':
    unittest.main()
