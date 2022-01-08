from dataclasses import dataclass, field
from typing import List, Tuple, Union

from . import variables


@dataclass
class Expr:
    line_index: int = field(default=-1, kw_only=True)  # line index of the original model code
    column_index: int = field(default=-1, kw_only=True)  # column index of the original model code

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def code(self):
        pass

    def __str__(self):
        return "Expr()"


@dataclass
class RealConstant(Expr):
    value: float

    def code(self):
        if self.value == float("inf"):
            return "float('inf')"
        elif self.value == float("-inf"):
            return "float('-inf')"
        return f"{self.value}"

    def __str__(self):
        return f"RealConstant({self.value})"


@dataclass
class IntegerConstant(Expr):
    value: int

    def code(self):
        return f"{self.value}"

    def __str__(self):
        return f"IntegerConstant({self.value})"


@dataclass
class Subscript(Expr):
    names: Tuple[str]
    shifts: Tuple[Union[str, None]] = (None,)
    variable: variables.SubscriptUse = None

    def get_key(self):
        return self.names

    def code(self):
        return self.variable.code()

    def __str__(self):
        return f"Index(names=({', '.join(x.__str__() for x in self.names)}), shift=({', '.join(x.__str__() for x in self.shifts)}))"


@dataclass
class Data(Expr):
    name: str
    subscript: Subscript = None
    variable: variables.Data = None

    def get_key(self):
        return self.name

    def code(self):
        variable_code = self.variable.code()
        if self.subscript is not None:
            return variable_code + f"[{self.subscript.code()}]"
        else:
            return variable_code

    def __str__(self):
        if self.subscript is None:
            return f"Data({self.name})"
        else:
            return f"Data({self.name}, {self.subscript})"
        # return f"Placeholder({self.name}, {self.subscript.__str__()}) = {{{self.value.__str__()}}}"


@dataclass
class Param(Expr):
    name: str
    subscript: Subscript = None
    lower: Expr = RealConstant(float("-inf"))
    upper: Expr = RealConstant(float("inf"))
    variable: variables.Param = None

    def __iter__(self):
        if self.subscript is not None:
            return iter([self.subscript])
        else:
            return iter([])

    def scalar(self):
        return self.subscript is None

    def get_key(self):
        return self.name

    def code(self):
        variable_code = self.variable.code()
        if self.subscript:
            return variable_code + f"[{self.subscript.code()}]"
        else:
            return variable_code

    def __str__(self):
        return f"Param({self.name}, {self.subscript.__str__()}, lower={self.lower}, upper={self.upper})"


@dataclass
class Distr(Expr):
    variate: Expr


@dataclass
class Normal(Distr):
    variate: Expr
    mean: Expr
    std: Expr

    def __iter__(self):
        return iter([self.variate, self.mean, self.std])

    def code(self):
        return f"jax.scipy.stats.norm.logpdf({self.variate.code()}, {self.mean.code()}, {self.std.code()})"

    def __str__(self):
        return f"Normal({self.variate.__str__()}, {self.mean.__str__()}, {self.std.__str__()})"


@dataclass
class BernoulliLogit(Distr):
    variate: Expr
    logit_p: Expr

    def __iter__(self):
        return iter([self.variate, self.logit_p])

    def code(self):
        return f"bernoulli_logit({self.variate.code()}, {self.logit_p.code()})"

    def __str__(self):
        return f"BernoulliLogit({self.variate.__str__()}, {self.logit_p.__str__()})"


@dataclass
class LogNormal(Distr):
    variate: Expr
    mean: Expr
    std: Expr

    def __iter__(self):
        return iter([self.variate, self.mean, self.std])

    def code(self):
        return f"log_normal({self.variate.code()}, {self.mean.code()}, {self.std.code()})"

    def __str__(self):
        return f"LogNormal({self.variate.__str__()}, {self.mean.__str__()}, {self.std.__str__()})"


@dataclass
class Cauchy(Distr):
    variate: Expr
    location: Expr
    scale: Expr

    def __iter__(self):
        return iter([self.variate, self.location, self.scale])

    def code(self):
        return f"jax.scipy.stats.cauchy.logpdf({self.variate.code()}, {self.location.code()}, {self.scale.code()})"

    def __str__(self):
        return f"Cauchy({self.variate.code()}, {self.location.code()}, {self.scale.code()})"


@dataclass
class Exponential(Distr):
    variate: Expr
    scale: Expr

    def __iter__(self):
        return iter([self.variate, self.scale])

    def code(self):
        return f"jax.scipy.stats.expon.logpdf({self.variate.code()}, loc=0, scale={self.scale.code()})"

    def __str__(self):
        return f"Exponential({self.variate.code()}, {self.scale.code()})"


@dataclass
class Diff(Expr):
    lhs: Expr
    rhs: Expr

    def __iter__(self):
        return iter([self.lhs, self.rhs])

    def code(self):
        return f"({self.lhs.code()} - {self.rhs.code()})"

    def __str__(self):
        return f"Diff({self.lhs.__str__()}, {self.rhs.__str__()})"


@dataclass
class Sum(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} + {self.right.code()}"

    def __str__(self):
        return f"Sum({self.left.__str__()}, {self.right.__str__()})"


@dataclass
class Mul(Expr):
    lbp = 30
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} * {self.right.code()}"

    def __str__(self):
        return f"Mul({self.left.__str__()}, {self.right.__str__()})"


@dataclass
class Pow(Expr):
    lbp = 40
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} ^ {self.right.code()}"

    def __str__(self):
        return f"Pow({self.left.__str__()}, {self.right.__str__()})"


@dataclass
class Div(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} / {self.right.code()}"

    def __str__(self):
        return f"Div({self.left.__str__(), self.right.__str__()})"


@dataclass
class Mod(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} % {self.right.code()}"

    def __str__(self):
        return f"Mod({self.left.__str__(), self.right.__str__()})"


@dataclass
class LogicalOR(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} || {self.right.code()}"

    def __str__(self):
        return f"LogicalOR({self.left.__str__(), self.right.__str__()})"


@dataclass
class LogicalAND(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} && {self.right.code()}"

    def __str__(self):
        return f"LogicalAND({self.left.__str__(), self.right.__str__()})"


@dataclass
class Equality(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} == {self.right.code()}"

    def __str__(self):
        return f"Equality({self.left.__str__(), self.right.__str__()})"


@dataclass
class Inequality(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} != {self.right.code()}"

    def __str__(self):
        return f"Inequality({self.left.__str__(), self.right.__str__()})"


@dataclass
class LessThan(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} < {self.right.code()}"

    def __str__(self):
        return f"LessThan({self.left.__str__(), self.right.__str__()})"


@dataclass
class LessThanOrEqual(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} <= {self.right.code()}"

    def __str__(self):
        return f"LessThanOrEqual({self.left.__str__(), self.right.__str__()})"


@dataclass
class GreaterThan(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} > {self.right.code()}"

    def __str__(self):
        return f"GreaterThan({self.left.__str__(), self.right.__str__()})"


@dataclass
class GreaterThanOrEqual(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} >= {self.right.code()}"

    def __str__(self):
        return f"GreaterThanOrEqual({self.left.__str__(), self.right.__str__()})"


@dataclass
class PrefixNegation(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"-{self.subexpr.code()}"

    def __str__(self):
        return f"PrefixNegation({self.subexpr.__str__()})"


@dataclass
class PrefixLogicalNegation(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"!{self.subexpr.code()}"

    def __str__(self):
        return f"PrefixLogicalNegation({self.subexpr.__str__()})"


@dataclass
class Assignment(Expr):
    lhs: Expr
    rhs: Expr

    def __iter__(self):
        return iter([self.lhs, self.rhs])

    def code(self):
        return f"{self.lhs.code()} = {self.rhs.code()}"

    def __str__(self):
        return f"Assignment({self.lhs.__str__()}, {self.rhs.__str__()})"


@dataclass
class Log(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.log({self.subexpr.code()})"

    def __str__(self):
        return f"Log({self.subexpr.__str__()})"


@dataclass
class Exp(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.exp({self.subexpr.code()})"

    def __str__(self):
        return f"Exp({self.subexpr.__str__()})"


@dataclass
class Abs(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.abs({self.subexpr.code()})"

    def __str__(self):
        return f"Abs({self.subexpr.__str__()})"


@dataclass
class Floor(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.floor({self.subexpr.code()})"

    def __str__(self):
        return f"Floor({self.subexpr.__str__()})"


@dataclass
class Ceil(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.ceil({self.subexpr.code()})"

    def __str__(self):
        return f"Ceil({self.subexpr.__str__()})"


@dataclass
class Round(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.round({self.subexpr.code()})"

    def __str__(self):
        return f"Round({self.subexpr.__str__()})"


@dataclass
class Sin(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.sin({self.subexpr.code()})"

    def __str__(self):
        return f"Sin({self.subexpr.__str__()})"


@dataclass
class Cos(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.cos({self.subexpr.code()})"

    def __str__(self):
        return f"Cos({self.subexpr.__str__()})"


@dataclass
class Tan(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.tan({self.subexpr.code()})"

    def __str__(self):
        return f"Tan({self.subexpr.__str__()})"


@dataclass
class Arcsin(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.arcsin({self.subexpr.code()})"

    def __str__(self):
        return f"Arcsin({self.subexpr.__str__()})"


@dataclass
class Arccos(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.arccos({self.subexpr.code()})"

    def __str__(self):
        return f"Arccos({self.subexpr.__str__()})"


@dataclass
class Arctan(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.arctan({self.subexpr.code()})"

    def __str__(self):
        return f"Arctan({self.subexpr.__str__()})"


@dataclass
class Logit(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.scipy.special.logit({self.subexpr.code()})"

    def __str__(self):
        return f"Logit({self.subexpr.__str__()})"


@dataclass
class InverseLogit(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.scipy.special.expit({self.subexpr.code()})"

    def __str__(self):
        return f"InverseLogit({self.subexpr.__str__()})"


@dataclass
class Placeholder(Expr):
    name: str
    index: Union[Subscript, None]
    value: float = None

    def __iter__(self):
        if self.index is not None:
            return iter([self.index])
        else:
            return iter([])

    def code(self):
        if self.index is not None:
            return f"{self.name}[{self.index.code()}]"
        else:
            return self.name

    def __str__(self):
        return f"Placeholder({self.name}, {self.index.__str__()})"


def search_tree(expr: Expr, *types) -> Expr:
    for _type in types:
        if isinstance(expr, _type):
            yield expr
    else:
        for child in expr:
            for child_expr in search_tree(child, *types):
                yield child_expr
