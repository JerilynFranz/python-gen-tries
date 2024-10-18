"""Tests for the gentries.generalized module."""

from textwrap import dedent
import traceback
from typing import Any, Callable, Dict, List, NamedTuple, Optional
import unittest

from gentries.generalized import GeneralizedTrie


class NoExpectedValue:  # pylint: disable=too-few-public-methods
    """This is used to distinguish between having an expected return value
    of None and and not expecting a value."""


class TestConfig(NamedTuple):
    name: str
    action: Callable  # type: ignore
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}
    expected: Any = NoExpectedValue()
    obj: Optional[Any] = None
    validate_obj: Optional[Callable] = None  # type: ignore
    validate_result: Optional[Callable] = None  # type: ignore
    exception: Optional[type[Exception]] = None
    exception_tag: Optional[str] = None


def run_tests_list(self, tests_list: List[TestConfig]) -> None:  # type: ignore
    for test in tests_list:
        run_test(self, test)  # type: ignore


def run_test(self, entry: TestConfig) -> None:  # type: ignore
    with self.subTest(msg=entry.name):  # type: ignore
        test_description: str = f"{entry.name}"
        errors: List[str] = []
        try:
            found: Any = entry.action(*entry.args, **entry.kwargs)  # type: ignore
            if entry.exception:
                errors.append("returned result instead of raising exception")

            else:
                if entry.validate_result and not entry.validate_result(found):  # type: ignore
                    errors.append(f"failed result validation: found={found}")
                if entry.validate_obj and not entry.validate_obj(entry.obj):  # type: ignore
                    errors.append(f"failed object validation: obj={entry.obj}")
                if (
                    not isinstance(entry.expected, NoExpectedValue)
                    and entry.expected != found
                ):
                    errors.append(f"expected={entry.expected}, found={found}")
        except Exception as err:  # pylint: disable=broad-exception-caught
            if entry.exception is None:
                errors.append(f"Did not expect exception. Caught exception {repr(err)}")
                errors.append("stacktrace = ")
                errors.append("\n".join(traceback.format_tb(tb=err.__traceback__)))

            if not (entry.exception and isinstance(err, entry.exception)):  # type: ignore
                errors.append(
                    f"Unexpected exception type: expected={entry.exception}, "
                    f"found = {repr(err)}"
                )

            elif entry.exception_tag:
                if str(err).find(entry.exception_tag) == -1:
                    errors.append(
                        f"correct exception type, but tag "
                        f"{entry.exception_tag} not found: {repr(err)}"
                    )
        if errors:
            self.fail(msg=test_description + ": " + "\n".join(errors))  # type: ignore


class TestGeneralizedTrie(unittest.TestCase):

    def test_create_trie(self) -> None:
        tests: List[TestConfig] = [
            TestConfig(
                name="[TCT001] create GeneralizedTrie()",
                action=GeneralizedTrie,
                validate_result=lambda found: isinstance(found, GeneralizedTrie),  # type: ignore
            ),  # type: ignore
            TestConfig(
                name="[TCT002] create GeneralizedTrie(filter_id=1)",
                action=GeneralizedTrie,
                kwargs={"filter_id": 1},
                exception=TypeError,
            ),
        ]
        run_tests_list(self, tests)

    def test_add(self) -> None:
        trie: GeneralizedTrie = GeneralizedTrie()
        tests: List[TestConfig] = [
            TestConfig(
                name="[TA001] trie.add(['tree', 'value', 'ape'])",
                action=trie.add,
                args=[["tree", "value", "ape"]],
                kwargs={},
                expected=1,
            ),
            TestConfig(
                name="[TA002] trie.add(['tree', 'value']",
                action=trie.add,
                args=[["tree", "value"]],
                expected=2,
            ),
            TestConfig(
                name="[TA003] trie.add('abcdef')",
                action=trie.add,
                args=["abcdef"],
                expected=3,
            ),
            TestConfig(
                name="[TA004] trie.add(1, 3, 4, 5])",
                action=trie.add,
                args=[[1, 3, 4, 5]],
                kwargs={},
                expected=4,
            ),
            TestConfig(
                name="[TA006] trie.add(frozenset([1]), 3, 4, 5])",
                action=trie.add,
                args=[[frozenset([1]), 3, 4, 5]],
                kwargs={},
                expected=5,
            ),
            TestConfig(
                name="[TA007] trie.add(frozenset([1]), 3, 4, 5])",
                action=trie.add,
                args=[[frozenset([1]), 3, 4, 6]],
                expected=6,
            ),
            TestConfig(
                name="[TA008] trie.add(1)",
                action=trie.add,
                args=[1],
                exception=TypeError,
                exception_tag="[GTAFBT001]",
            ),
            TestConfig(
                name="[TA009] trie.add([])",
                action=trie.add,
                args=[[]],
                exception=ValueError,
                exception_tag="[GTAFBT002]",
            ),
            TestConfig(
                name="[TA010] trie.add(set([1]), 3, 4, 5])",
                action=trie.add,
                args=[[set([1]), 3, 4, 5]],
                exception=TypeError,
                exception_tag="[GTAFBT003]",
            ),
            TestConfig(
                name="[TA011] trie.add(key=[1, 3, 4, 7])",
                action=trie.add,
                kwargs={"key": [1, 3, 4, 7]},
                exception=TypeError,
            ),
            TestConfig(name="[TA012] trie.add()", action=trie.add, exception=TypeError),
            TestConfig(
                name="[TA013] trie.add(['a'], ['b'])",
                action=trie.add,
                args=[["a"], ["b"]],
                exception=TypeError,
            ),
            TestConfig(name="[TA014] len(trie)", action=len, args=[trie], expected=6),
            TestConfig(
                name="[TA015] trie.add(['tree', 'value', 'ape'])",
                action=trie.add,
                args=[["tree", "value", "ape"]],
                kwargs={},
                expected=1,
            ),
            TestConfig(name="[TA016] len(trie)", action=len, args=[trie], expected=6),
            TestConfig(
                name="[TA017] trie.add(['apple', 'value', 'ape'])",
                action=trie.add,
                args=[["apple", "value", "ape"]],
                kwargs={},
                expected=7,
            ),
        ]
        run_tests_list(self, tests)

    def test_prefixes(self) -> None:
        trie: GeneralizedTrie = GeneralizedTrie()
        tests: List[TestConfig] = [
            TestConfig(
                name="[TP001] trie.add(['tree', 'value', 'ape'])",
                action=trie.add,
                args=[["tree", "value", "ape"]],
                expected=1,
            ),
            TestConfig(
                name="[TP002] trie.add(['tree', 'value']",
                action=trie.add,
                args=[["tree", "value"]],
                expected=2,
            ),
            TestConfig(
                name="[TP003] trie.add('abcdef')",
                action=trie.add,
                args=["abcdef"],
                kwargs={},
                expected=3,
            ),
            TestConfig(
                name="[TP004] trie.add('abc')",
                action=trie.add,
                args=["abc"],
                expected=4,
            ),
            TestConfig(
                name="[TP005] trie.prefixes(['tree', 'value', 'ape'])",
                action=trie.prefixes,
                args=[["tree", "value", "ape"]],
                expected=set([1, 2]),
            ),
            TestConfig(
                name="[TP006] trie.prefixes(['tree', 'value'])",
                action=trie.prefixes,
                args=[["tree", "value"]],
                expected=set([2]),
            ),
            TestConfig(
                name="[TP007] trie.prefixes('a')",
                action=trie.prefixes,
                args=["a"],
                expected=set(),
            ),
            TestConfig(
                name="[TP008] trie.prefixes('abc')",
                action=trie.prefixes,
                args=["abc"],
                expected=set([4]),
            ),
            TestConfig(
                name="[TP009] trie.prefixes('abcd')",
                action=trie.prefixes,
                args=["abcd"],
                expected=set([4]),
            ),
            TestConfig(
                name="[TP010] trie.prefixes(['abc'])",
                action=trie.prefixes,
                args=[["abc"]],
                expected=set(),
            ),
            TestConfig(
                name="[TP011] trie.add([1,3,4])",
                action=trie.add,
                args=[[1, 3, 4]],
                expected=5,
            ),
            TestConfig(
                name="[TP012] trie.prefixes([1, 3, 4, 5, 6, ])",
                action=trie.prefixes,
                args=[[1, 3, 4, 5, 6]],
                expected=set([5]),
            ),
            TestConfig(
                name="[TP013] trie.prefixes(['a', 3, 4, 5])",
                action=trie.prefixes,
                args=[["a", 3, 4, 5]],
                expected=set(),
            ),
            TestConfig(
                name="[TP014] trie.add(frozenset([1]), 3, 4, 5])",
                action=trie.add,
                args=[[frozenset([1]), 3, 4, 5]],
                expected=6,
            ),
            TestConfig(
                name="[TP015] trie.prefixes([frozenset([1]), 3, 4, 5])",
                action=trie.prefixes,
                args=[[frozenset([1]), 3, 4, 5]],
                expected=set([6]),
            ),
            TestConfig(
                name="[TP016] trie.prefixes(trie_key=[frozenset([1]), 3, 4, 5])",
                action=trie.prefixes,
                kwargs={"trie_key": [frozenset([1]), 3, 4, 5]},
                expected=set([6]),
            ),
            TestConfig(
                name="[TP017] trie.prefixes(trie_key=[set([1]), 3, 4, 5])",
                action=trie.prefixes,
                kwargs={"trie_key": [set([1]), 3, 4, 5]},
                exception=TypeError,
            ),
            TestConfig(
                name="[TP018] trie.prefixes()",
                action=trie.prefixes,
                exception=TypeError,
            ),
            TestConfig(
                name="[TP019] trie.prefixes(None)",
                action=trie.prefixes,
                args=[None],
                exception=TypeError,
                exception_tag="[GTM001]",
            ),
        ]
        run_tests_list(self, tests)

    def test_suffixes(self) -> None:
        trie: GeneralizedTrie = GeneralizedTrie()
        tests: List[TestConfig] = [
            TestConfig(
                name="[TS001] trie.add(['tree', 'value', 'ape'])",
                action=trie.add,
                args=[["tree", "value", "ape"]],
                expected=1,
            ),
            TestConfig(
                name="[TS002] trie.add(['tree', 'value']",
                action=trie.add,
                args=[["tree", "value"]],
                expected=2,
            ),
            TestConfig(
                name="[TS003] trie.add('abcdef')",
                action=trie.add,
                args=["abcdef"],
                kwargs={},
                expected=3,
            ),
            TestConfig(
                name="[TS004] trie.add('abc')",
                action=trie.add,
                args=["abc"],
                expected=4,
            ),
            TestConfig(
                name="[TS005] trie.suffixes(['tree', 'value', 'ape'])",
                action=trie.suffixes,
                args=[["tree", "value", "ape"]],
                expected=set([1]),
            ),
            TestConfig(
                name="[TS006] trie.suffixes(['tree', 'value'])",
                action=trie.suffixes,
                args=[["tree", "value"]],
                expected=set([1, 2]),
            ),
            TestConfig(
                name="[TS007] trie.suffixes('a')",
                action=trie.suffixes,
                args=["a"],
                expected=set([3, 4]),
            ),
            TestConfig(
                name="[TS008] trie.suffixes('abc')",
                action=trie.suffixes,
                args=["abc"],
                expected=set([3, 4]),
            ),
            TestConfig(
                name="[TS009] trie.suffixes('abcd')",
                action=trie.suffixes,
                args=["abcd"],
                expected=set([3]),
            ),
            TestConfig(
                name="[TS010] trie.suffixes(['abc'])",
                action=trie.suffixes,
                args=[["abc"]],
                expected=set(),
            ),
            TestConfig(
                name="[TS011] trie.add([1,3,4])",
                action=trie.add,
                args=[[1, 3, 4]],
                expected=5,
            ),
            TestConfig(
                name="[TS012] trie.suffixes([1, 3, 4, 5, 6])",
                action=trie.suffixes,
                args=[[1, 3, 4, 5, 6]],
                expected=set(),
            ),
            TestConfig(
                name="[TS013] trie.suffixes(['a', 3, 4, 5])",
                action=trie.suffixes,
                args=[["a", 3, 4, 5]],
                expected=set(),
            ),
            TestConfig(
                name="[TS014] trie.add(frozenset([1]), 3, 4, 5])",
                action=trie.add,
                args=[[frozenset([1]), 3, 4, 5]],
                expected=6,
            ),
            TestConfig(
                name="[TS015] trie.suffixes([frozenset([1]), 3, 4, 5])",
                action=trie.suffixes,
                args=[[frozenset([1]), 3, 4, 5]],
                expected=set([6]),
            ),
            TestConfig(
                name="[TS017] trie.suffixes(trie_key=[frozenset([1]), 3, 4, 5])",
                action=trie.suffixes,
                kwargs={"trie_key": [frozenset([1]), 3, 4, 5]},
                expected=set([6]),
            ),
            TestConfig(
                name="[TS018] trie.suffixes(trie_key=[set([1]), 3, 4, 5])",
                action=trie.suffixes,
                kwargs={"trie_key": [set([1]), 3, 4, 5]},
                exception=TypeError,
            ),
            TestConfig(
                name="[TS019] trie.suffixes()",
                action=trie.suffixes,
                exception=TypeError,
            ),
            TestConfig(
                name="[TS020] trie.suffixes(None)",
                action=trie.suffixes,
                args=[None],
                exception=TypeError,
                exception_tag="[GTS001]",
            ),
            TestConfig(
                name="[TS021] trie.suffixes(depth=1)",
                action=trie.suffixes,
                kwargs={"depth": 1},
                exception=TypeError,
            ),
            TestConfig(
                name="[TS022] trie.suffixes(trie_key='a', depth='b')",
                action=trie.suffixes,
                kwargs={"trie_key": "a", "depth": "b"},
                exception=TypeError,
                exception_tag="[GTS002]",
            ),
            TestConfig(
                name="[TS023] trie.suffixes(trie_key='a', depth=-2)",
                action=trie.suffixes,
                kwargs={"trie_key": "a", "depth": -2},
                exception=ValueError,
                exception_tag="[GTS003]",
            ),
            TestConfig(
                name="[TS023] trie.suffixes(trie_key=[set(['a'], 'b']))",
                action=trie.suffixes,
                kwargs={"trie_key": [set("a"), "b"]},
                exception=TypeError,
                exception_tag="[GTS004]",
            ),
        ]
        run_tests_list(self, tests)

    def test_remove(self) -> None:
        trie: GeneralizedTrie = GeneralizedTrie()
        tests: List[TestConfig] = [
            TestConfig(
                name="[TR001] trie.add('a')", action=trie.add, args=["a"], expected=1
            ),
            TestConfig(
                name="[TR002] trie.add('ab')", action=trie.add, args=["ab"], expected=2
            ),
            TestConfig(
                name="[TR003] trie.add('abc')",
                action=trie.add,
                args=["abc"],
                expected=3,
            ),
            TestConfig(
                name="[TR004] trie.add('abe')",
                action=trie.add,
                args=["abe"],
                expected=4,
            ),
            TestConfig(
                name="[TR005] trie.add('abef')",
                action=trie.add,
                args=["abef"],
                expected=5,
            ),
            TestConfig(
                name="[TR006] trie.add('abcd')",
                action=trie.add,
                args=["abcd"],
                expected=6,
            ),
            TestConfig(
                name="[TR007] trie.add('abcde')",
                action=trie.add,
                args=["abcde"],
                expected=7,
            ),
            TestConfig(
                name="[TR008] trie.add('abcdf')",
                action=trie.add,
                args=["abcdef"],
                expected=8,
            ),
            TestConfig(
                name="[TR009] trie.add('abcdefg')",
                action=trie.add,
                args=["abcdefg"],
                expected=9,
            ),
            TestConfig(
                name="[TR010] trie.remove(9)",
                action=trie.remove,
                args=[9],
                expected=None,
            ),
            TestConfig(name="[TR011] len(trie)", action=len, args=[trie], expected=8),
            TestConfig(
                name="[TR012] trie.remove(9)",
                action=trie.remove,
                args=[9],
                exception=KeyError,
                exception_tag="[GTR003]",
            ),
            TestConfig(name="[TR013] len(trie)", action=len, args=[trie], expected=8),
            TestConfig(
                name="[TR014] trie.remove(1)",
                action=trie.remove,
                args=[1],
                expected=None,
            ),
            TestConfig(name="[TR015] len(trie)", action=len, args=[trie], expected=7),
            TestConfig(
                name="[TR016] trie.remove(2)",
                action=trie.remove,
                args=[2],
                expected=None,
            ),
            TestConfig(name="[TR017] len(trie)", action=len, args=[trie], expected=6),
            TestConfig(
                name="[TR018] trie.remove('abc')",
                action=trie.remove,
                args=["abc"],
                exception=TypeError,
                exception_tag="[GTR001]",
            ),
            TestConfig(
                name="[TR019] trie.remove(0)",
                action=trie.remove,
                args=[0],
                exception=KeyError,
                exception_tag="[GTR002]",
            ),
            TestConfig(
                name="[TR020] trie.add('qrstuv')",
                action=trie.add,
                args=['qrstuv'],
                expected=10,
            ),
            TestConfig(
                name="[TR021] trie.remove(10)",
                action=trie.remove,
                args=[10],
                expected=None,
            ),
            TestConfig(
                name="[TR022] len(trie)",
                action=len,
                args=[trie],
                expected=6,
            ),
        ]
        run_tests_list(self, tests)

    def test_str(self) -> None:
        trie: GeneralizedTrie = GeneralizedTrie()
        trie.add('ab')
        found: str = dedent(str(trie))
        expected: str = dedent("""\
        {
          trie number = 1
          node token = None
          children = {
            a = {
              parent = root node
              node token = a
              children = {
                b = {
                  parent = a
                  node token = b
                  trie id = 1
                }
              }
            }
          }
          trie index = dict_keys([1])
        }""")
        self.assertEqual(found, expected, msg='[TSTR001] str(trie)')

    def test_contains(self) -> None:
        trie: GeneralizedTrie = GeneralizedTrie()
        tests: List[TestConfig] = [
            TestConfig(
                name="[TC001] trie.__contains__('a')",
                action=trie.__contains__,
                args=['a'],
                expected=False
            ),
            TestConfig(
                name="[TC002] trie.add('a')", action=trie.add, args=["a"], expected=1
            ),
            TestConfig(
                name="[TC003] trie.__contains__('a')",
                action=trie.__contains__,
                args=['a'],
                expected=True
            ),
            TestConfig(
                name="[TC004] trie.remove(1)", action=trie.remove, args=[1], expected=None
            ),
            TestConfig(
                name="[TC006] trie.__contains__('a')",
                action=trie.__contains__,
                args=['a'],
                expected=False
            ),
        ]
        run_tests_list(self, tests)

    def test_bool(self) -> None:
        trie: GeneralizedTrie = GeneralizedTrie()
        tests: List[TestConfig] = [
            TestConfig(
                name="[TB001] bool(trie)", action=bool, args=[trie], expected=False
            ),
            TestConfig(
                name="[TB002] trie.add('a')", action=trie.add, args=["a"], expected=1
            ),
            TestConfig(
                name="[TB003] bool(trie)", action=bool, args=[trie], expected=True
            ),
            TestConfig(
                name="[TB004] trie.remove(1)", action=trie.remove, args=[1], expected=None
            ),
            TestConfig(
                name="[TB003] bool(trie)", action=bool, args=[trie], expected=False
            ),
        ]
        run_tests_list(self, tests)


if __name__ == "__main__":
    unittest.main()
