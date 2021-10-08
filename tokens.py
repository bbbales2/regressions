from dataclasses import dataclass
from typing import List, Tuple

class Expr:
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

@dataclass(frozen = True)
class Data(Expr):
    name : str

    def get_key(self):
        return self.name

@dataclass(frozen = True)
class Index(Expr):
    names : Tuple[str]
    
    def get_key(self):
        return self.names

@dataclass(frozen = True)
class Param(Expr):
    name : str
    index : Index = None
    lower : float = float("-inf")
    upper : float = float("inf")

    def __iter__(self):
        if self.index is not None:
            return iter([self.index])
        else:
            return iter([])

    def get_key(self):
        return self.name

class Distr(Expr):
    pass

@dataclass(frozen = True)
class Normal(Distr):
    variate : Expr
    mean : Expr
    std : Expr

    def __iter__(self):
        return iter([self.lhs, self.mean, self.std])

@dataclass(frozen = True)
class Constant(Expr):
    def __init__(self, value):
        self.value = value

    def code(self):
        return f"{self.value}"

@dataclass(frozen = True)
class Diff(Expr):
    lhs : Expr
    rhs : Expr

    def __iter__(self):
        return iter([self.lhs, self.rhs])

def search_tree(type, expr):
    if isinstance(expr, type):
        yield expr
    else:
        for child in expr:
            for child_expr in search_tree(type, child):
                yield child_expr
