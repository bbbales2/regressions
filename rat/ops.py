from dataclasses import dataclass
from typing import List, Tuple, Union

from . import variables


class Expr:
    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def code(self):
        pass

    def __str__(self):
        return "Expr()"


@dataclass(frozen=True)
class RealConstant(Expr):
    value: float

    def code(self):
        return f"{self.value}"

    def __str__(self):
        return f"RealConstant({self.value})"


@dataclass(frozen=True)
class IntegerConstant(Expr):
    value: int

    def code(self):
        return f"{self.value}"

    def __str__(self):
        return f"IntegerConstant({self.value})"


@dataclass
class Index(Expr):
    names: Tuple[str]
    shifts: Tuple[Union[str, None]] = (None,)
    variable: variables.Index = None

    def get_key(self):
        return self.names

    def code(self):
        return self.variable.code()

    def __str__(self):
        return f"Index(names=({', '.join(x.__str__() for x in self.names)}), shift=({', '.join(x.__str__() for x in self.shifts)}))"


@dataclass
class Data(Expr):
    name: str
    index: Index = None
    variable: variables.Data = None

    def get_key(self):
        return self.name

    def code(self):
        variable_code = self.variable.code()
        if self.index is not None:
            return variable_code + f"[{self.index.code()}]"
        else:
            return variable_code

    def __str__(self):
        if self.index is None:
            return f"Data({self.name})"
        else:
            return f"Data({self.name}, {self.index})"
        # return f"Placeholder({self.name}, {self.index.__str__()}) = {{{self.value.__str__()}}}"


@dataclass
class Param(Expr):
    name: str
    index: Index = None
    lower: RealConstant = RealConstant(float("-inf"))
    upper: RealConstant = RealConstant(float("inf"))
    variable: variables.Param = None

    def __iter__(self):
        if self.index is not None:
            return iter([self.index])
        else:
            return iter([])

    def scalar(self):
        return self.index is None

    def get_key(self):
        return self.name

    def code(self):
        variable_code = self.variable.code()
        if self.index is not None:
            return variable_code + f"[{self.index.code()}]"
        else:
            return variable_code

    def __str__(self):
        return f"Param({self.name}, {self.index.__str__()}, lower={self.lower}, upper={self.upper})"
        # return f"Placeholder({self.name}, {self.index.__str__()}) = {{{self.value.__str__()}}}"


class Distr(Expr):
    pass


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class BernoulliLogit(Distr):
    variate: Expr
    logit_p: Expr

    def __iter__(self):
        return iter([self.variate, self.logit_p])

    def code(self):
        return f"bernoulli_logit({self.variate.code()}, {self.logit_p.code()})"

    def __str__(self):
        return f"Normal({self.variate.__str__()}, {self.mean.__str__()}, {self.std.__str__()})"


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class Diff(Expr):
    lhs: Expr
    rhs: Expr

    def __iter__(self):
        return iter([self.lhs, self.rhs])

    def code(self):
        return f"({self.lhs.code()} - {self.rhs.code()})"

    def __str__(self):
        return f"Diff({self.lhs.__str__()}, {self.rhs.__str__()})"


@dataclass(frozen=True)
class Sum(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} + {self.right.code()}"

    def __str__(self):
        return f"Sum({self.left.__str__()}, {self.right.__str__()})"


@dataclass(frozen=True)
class Mul(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} * {self.right.code()}"

    def __str__(self):
        return f"Mul({self.left.__str__()}, {self.right.__str__()})"


@dataclass(frozen=True)
class Pow(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} ^ {self.right.code()}"

    def __str__(self):
        return f"Pow({self.left.__str__()}, {self.right.__str__()})"


@dataclass(frozen=True)
class Div(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} / {self.right.code()}"

    def __str__(self):
        return f"Div({self.left.__str__(), self.right.__str__()})"


@dataclass(frozen=True)
class Mod(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} % {self.right.code()}"

    def __str__(self):
        return f"Mod({self.left.__str__(), self.right.__str__()})"


@dataclass(frozen=True)
class LogicalOR(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} || {self.right.code()}"

    def __str__(self):
        return f"LogicalOR({self.left.__str__(), self.right.__str__()})"


@dataclass(frozen=True)
class LogicalAND(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} && {self.right.code()}"

    def __str__(self):
        return f"LogicalAND({self.left.__str__(), self.right.__str__()})"


@dataclass(frozen=True)
class Equality(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} == {self.right.code()}"

    def __str__(self):
        return f"Equality({self.left.__str__(), self.right.__str__()})"


@dataclass(frozen=True)
class Inequality(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} != {self.right.code()}"

    def __str__(self):
        return f"Inequality({self.left.__str__(), self.right.__str__()})"


@dataclass(frozen=True)
class LessThan(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} < {self.right.code()}"

    def __str__(self):
        return f"LessThan({self.left.__str__(), self.right.__str__()})"


@dataclass(frozen=True)
class LessThanOrEqual(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} <= {self.right.code()}"

    def __str__(self):
        return f"LessThanOrEqual({self.left.__str__(), self.right.__str__()})"


@dataclass(frozen=True)
class GreaterThan(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} > {self.right.code()}"

    def __str__(self):
        return f"GreaterThan({self.left.__str__(), self.right.__str__()})"


@dataclass(frozen=True)
class GreaterThanOrEqual(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} >= {self.right.code()}"

    def __str__(self):
        return f"GreaterThanOrEqual({self.left.__str__(), self.right.__str__()})"


@dataclass(frozen=True)
class PrefixNegation(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"-{self.subexpr.code()}"

    def __str__(self):
        return f"PrefixNegation({self.subexpr.__str__()})"


@dataclass(frozen=True)
class PrefixLogicalNegation(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"!{self.subexpr.code()}"

    def __str__(self):
        return f"PrefixLogicalNegation({self.subexpr.__str__()})"


@dataclass(frozen=True)
class Assignment(Expr):
    lhs: Expr
    rhs: Expr

    def __iter__(self):
        return iter([self.lhs, self.rhs])

    def code(self):
        return f"{self.lhs.code()} = {self.rhs.code()}"

    def __str__(self):
        return f"Assignment({self.lhs.__str__()}, {self.rhs.__str__()})"


@dataclass(frozen=True)
class AddAssignment(Expr):
    lhs: Expr
    rhs: Expr

    def __iter__(self):
        return iter([self.lhs, self.rhs])

    def code(self):
        return f"{self.lhs.code()} += {self.rhs.code()}"

    def __str__(self):
        return f"AddAssignment({self.lhs.__str__()}, {self.rhs.__str__()})"


@dataclass(frozen=True)
class DiffAssignment(Expr):
    lhs: Expr
    rhs: Expr

    def __iter__(self):
        return iter([self.lhs, self.rhs])

    def code(self):
        return f"{self.lhs.code()} -= {self.rhs.code()}"

    def __str__(self):
        return f"DiffAssignment({self.lhs.__str__()}, {self.rhs.__str__()})"


@dataclass(frozen=True)
class MulAssignment(Expr):
    lhs: Expr
    rhs: Expr

    def __iter__(self):
        return iter([self.lhs, self.rhs])

    def code(self):
        return f"{self.lhs.code()} *= {self.rhs.code()}"

    def __str__(self):
        return f"MulAssignment({self.lhs.__str__()}, {self.rhs.__str__()})"


@dataclass(frozen=True)
class DivAssignment(Expr):
    lhs: Expr
    rhs: Expr

    def __iter__(self):
        return iter([self.lhs, self.rhs])

    def code(self):
        return f"{self.lhs.code()} /= {self.rhs.code()}"

    def __str__(self):
        return f"DivAssignment({self.lhs.__str__()}, {self.rhs.__str__()})"


@dataclass(frozen=True)
class Log(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.log({self.subexpr.code()})"

    def __str__(self):
        return f"Log({self.subexpr.__str__()})"


@dataclass(frozen=True)
class Exp(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.exp({self.subexpr.code()})"

    def __str__(self):
        return f"Exp({self.subexpr.__str__()})"


@dataclass(frozen=True)
class Abs(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.abs({self.subexpr.code()})"

    def __str__(self):
        return f"Abs({self.subexpr.__str__()})"


@dataclass(frozen=True)
class Floor(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.floor({self.subexpr.code()})"

    def __str__(self):
        return f"Floor({self.subexpr.__str__()})"


@dataclass(frozen=True)
class Ceil(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.ceil({self.subexpr.code()})"

    def __str__(self):
        return f"Ceil({self.subexpr.__str__()})"


@dataclass(frozen=True)
class Round(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.round({self.subexpr.code()})"

    def __str__(self):
        return f"Round({self.subexpr.__str__()})"


@dataclass(frozen=True)
class Sin(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.sin({self.subexpr.code()})"

    def __str__(self):
        return f"Sin({self.subexpr.__str__()})"


@dataclass(frozen=True)
class Cos(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.cos({self.subexpr.code()})"

    def __str__(self):
        return f"Cos({self.subexpr.__str__()})"


@dataclass(frozen=True)
class Tan(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.tan({self.subexpr.code()})"

    def __str__(self):
        return f"Tan({self.subexpr.__str__()})"


@dataclass(frozen=True)
class Arcsin(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.arcsin({self.subexpr.code()})"

    def __str__(self):
        return f"Arcsin({self.subexpr.__str__()})"


@dataclass(frozen=True)
class Arccos(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.arccos({self.subexpr.code()})"

    def __str__(self):
        return f"Arccos({self.subexpr.__str__()})"


@dataclass(frozen=True)
class Arctan(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.arctan({self.subexpr.code()})"

    def __str__(self):
        return f"Arctan({self.subexpr.__str__()})"


@dataclass
class Placeholder(Expr):
    name: str
    index: Union[Index, None]
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
        # return f"Placeholder({self.name}, {self.index.__str__()}) = {{{self.value.__str__()}}}"


def search_tree(type, expr):
    if isinstance(expr, type):
        yield expr
    else:
        for child in expr:
            for child_expr in search_tree(type, child):
                yield child_expr
