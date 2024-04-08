from typing import runtime_checkable, Protocol


@runtime_checkable
class GeneralizedToken(Protocol):
    """GeneralizedToken defines tokens that are usable with a GeneralizedTrie.

    The protocol requires that a token object implements both an __eq__()
    method and a __hash__() method. This generally means that immutable types
    are suitable for use as tokens.

    Some examples of types usable as a token:
        str  bytes  int  float  complex  frozenset  tuple

    Usage:
        from gentries import GeneralizedToken
        if isinstance(token, GeneralizedToken):
            print("token supports the GeneralizedToken protocol")
        else:
            print("token does not support the GeneralizedToken protocol")
    """
    def __eq__(self) -> bool:
        ...

    def __hash__(self) -> int:
        ...


class InvalidTokenError(Exception):
    """Raised when a token does not conform to the GeneralizedToken protocol"""
    ...
