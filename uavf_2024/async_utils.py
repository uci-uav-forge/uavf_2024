from typing import Any, Callable


class OnceCallable():
    """
    Runs the given function only the first time it's called and do nothing for subsequent invocations.
    """
    def __init__(self, func: Callable):
        self._func = func
        self._called = False
        
    def __call__(self, *args, **kwargs) -> Any | None:
        if self._called:
            return 
        
        self._called = True
        return self._func(*args, **kwargs)
    