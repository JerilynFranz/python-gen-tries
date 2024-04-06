from typing import Protocol


class GeneralizedToken(Protocol):
    """GeneralizedToken defines tokens that are usable with a GeneralizedTrie.

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
