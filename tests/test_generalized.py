import traceback
from typing import Any, Callable, Dict, List, NamedTuple, Optional
import unittest

from gentries.generalized import GeneralizedTrie


class Ignore():
    pass


class TestConfig(NamedTuple):
    name: str
    action: Callable
    args: List[Any]
    kwargs: Dict[str, Any]
    expected: Any = Ignore()
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
                if (not isinstance(entry.expected, Ignore)
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
                args=[],
                kwargs={},
                validate_result=lambda found: isinstance(found,
                                                         GeneralizedTrie),
                                                         ),
             TestConfig(
                name='[TCT002] create GeneralizedTrie(filter_id=1)',
                action=GeneralizedTrie,
                args=[],
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
                args=[],
                kwargs={'trie_key': ['tree', 'value', 'ape']},
                expected=1),
            TestConfig(
                name="[TA002] trie.add(['tree', 'value']",
                action=trie.add,
                args=[],
                kwargs={'trie_key': ['tree', 'value']},
                expected=2),
            TestConfig(
                name="[TA003] trie.add('abcdef')",
                action=trie.add,
                args=[],
                kwargs={'trie_key': 'abcdef'},
                expected=3),
            TestConfig(
                name="[TA004] trie.add(1, 3, 4, 5])",
                action=trie.add,
                args=[],
                kwargs={'trie_key': [1, 3, 4, 5]},
                expected=4),
            TestConfig(
                name="[TA004] trie.add(set([1]), 3, 4, 5])",
                action=trie.add,
                args=[],
                kwargs={'trie_key': [set([1]), 3, 4, 5]},
                exception=TypeError,
                exception_tag='[GTAFBT003]'),
            TestConfig(
                name="[TA004] trie.add(frozenset([1]), 3, 4, 5])",
                action=trie.add,
                args=[],
                kwargs={'trie_key': [frozenset([1]), 3, 4, 5]},
                expected=5),
        ]
        run_tests_list(self, tests)

    def test_token_prefixes(self) -> None:
        trie: GeneralizedTrie = GeneralizedTrie()
        tests: List[TestConfig] = [
            TestConfig(
                name="[TTP001] trie.add(['tree', 'value', 'ape'])",
                action=trie.add,
                args=[],
                kwargs={'trie_key': ['tree', 'value', 'ape']},
                expected=1),
            TestConfig(
                name="[TTP002] trie.add(['tree', 'value']",
                action=trie.add,
                args=[],
                kwargs={'trie_key': ['tree', 'value']},
                expected=2),
            TestConfig(
                name="[TTP003] trie.add('abcdef')",
                action=trie.add,
                args=[],
                kwargs={'trie_key': 'abcdef'},
                expected=3),
            TestConfig(
                name="[TTP004] trie.add('abc')",
                action=trie.add,
                args=[],
                kwargs={'trie_key': 'abc'},
                expected=4),
            TestConfig(
                name="[TTP005] trie.token_prefixes(['tree', 'value', 'ape'])",
                action=trie.token_prefixes,
                args=[],
                kwargs={'tokens': ['tree', 'value', 'ape']},
                expected=set([1, 2])),
            TestConfig(
                name="[TTP006] trie.token_prefixes(['tree', 'value'])",
                action=trie.token_prefixes,
                args=[],
                kwargs={'tokens': ['tree', 'value']},
                expected=set([2])),
            TestConfig(
                name="[TTP007] trie.token_prefixes('a')",
                action=trie.token_prefixes,
                args=[],
                kwargs={'tokens': 'a'},
                expected=set()),
            TestConfig(
                name="[TTP008] trie.token_prefixes('abc')",
                action=trie.token_prefixes,
                args=[],
                kwargs={'tokens': 'abc'},
                expected=set([4])),
            TestConfig(
                name="[TTP009] trie.token_prefixes('abcd')",
                action=trie.token_prefixes,
                args=[],
                kwargs={'tokens': 'abcd'},
                expected=set([4])),
            TestConfig(
                name="[TTP010] trie.token_prefixes(['abc'])",
                action=trie.token_prefixes,
                args=[],
                kwargs={'tokens': ['abc']},
                expected=set()),
            TestConfig(
                name="[TTP011] trie.add([1,3,4])",
                action=trie.add,
                args=[],
                kwargs={'trie_key': [1, 3, 4]},
                expected=5),
            TestConfig(
                name="[TTP012] trie.token_prefixes([1, 3, 4, 5, 6, ])",
                action=trie.token_prefixes,
                args=[],
                kwargs={'tokens': [1, 3, 4, 5, 6]},
                expected=set([5])),
            TestConfig(
                name="[TTP012] trie.token_prefixes(['a', 3, 4, 5])",
                action=trie.token_prefixes,
                args=[],
                kwargs={'tokens': ['a', 3, 4, 5]},
                expected=set()),
            TestConfig(
                name="[TTP013] trie.add(frozenset([1]), 3, 4, 5])",
                action=trie.add,
                args=[],
                kwargs={'trie_key': [frozenset([1]), 3, 4, 5]},
                expected=6),
            TestConfig(
                name="[TTP014] trie.token_prefixes([frozenset([1]), 3, 4, 5])",
                action=trie.token_prefixes,
                args=[],
                kwargs={'tokens': [frozenset([1]), 3, 4, 5]},
                expected=set([6])),
        ]
        run_tests_list(self, tests)


if __name__ == '__main__':
    unittest.main()
