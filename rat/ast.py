from dataclasses import dataclass, field
from distutils.errors import CompileError
import itertools
from types import NoneType
from typing import Tuple, Union, Type, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from . import ir


from . import types
from .scanner import Range


@dataclass
class ASTNode:
    range: Range = field(default=lambda: None, kw_only=True)

    def __iter__(self):
        yield self

    def __next__(self):
        raise StopIteration

    def __str__(self):
        return "ASTNode()"


@dataclass
class Expr(ASTNode):
    """
    out_type denotes the return type of the expression
    """

    out_type: Type[types.BaseType] = field(default=types.BaseType, kw_only=True)

    def __post_init__(self):
        """
        __post_init__ must resolve and set self.out_type
        """
        raise NotImplementedError()

    def __str__(self):
        return "Expr()"


@dataclass
class Statement(ASTNode):
    def __str__(self):
        return "Statement()"


@dataclass
class RealConstant(Expr):
    """
    Elementary value
    """

    value: float

    def __post_init__(self):
        self.out_type = types.RealType

    def __str__(self):
        return f"RealConstant({self.value})"


@dataclass
class IntegerConstant(Expr):
    """
    Elementary value
    """

    value: int

    def code(self, scalar=False):
        return f"{self.value}"

    def __post_init__(self):
        self.out_type = types.IntegerType

    def __str__(self):
        return f"IntegerConstant({self.value})"


@dataclass
class SubscriptColumn(Expr):
    """
    columns/names used as subscripts
    """

    name: str

    def __post_init__(self):
        self.out_type = types.SubscriptSetType

    def __str__(self):
        return f"SubscriptColumn({self.name})"


@dataclass
class Subscript(Expr):
    """
    Elementary value
    """

    names: Tuple[SubscriptColumn]
    shifts: Tuple[Expr]

    def get_key(self):
        return self.names

    def __post_init__(self):
        assert len(self.shifts) == len(
            self.names
        ), "Internal Error: length of types.Subscript.names must equal length of types.Subscript.shifts"
        signatures = {(types.SubscriptSetType,) * len(self.shifts) + (types.IntegerType,) * len(self.shifts): types.SubscriptSetType}
        self.out_type = types.get_output_type(signatures, tuple(expr.out_type for expr in (*self.names, *self.shifts)))

    def __str__(self):
        return f"Subscript(names=({', '.join(x.__str__() for x in self.names)}), shift=({', '.join(x.__str__() for x in self.shifts)}))"


@dataclass
class Shift(Expr):
    """
    This is a function with type signature (SubscriptSet, Integer) -> SubscriptSet
    """

    subscript_column: Expr
    shift_expr: Expr

    def __post_init__(self):
        signatures = {(types.SubscriptSetType, types.IntegerType): types.SubscriptSetType}
        self.out_type = types.get_output_type(signatures, (self.subscript_column.out_type, self.shift_expr.out_type))

    def __str__(self):
        return f"Shift(subscript={self.subscript}, amount={self.shift_expr})"


@dataclass
class PrimeableExpr(Expr):
    name: str
    prime: bool = False
    subscript: Subscript = None

    def __iter__(self):
        yield self
        if self.subscript is not None:
            yield self.subscript


@dataclass
class Data(PrimeableExpr):
    def get_key(self):
        return self.name

    def __post_init__(self):
        self.out_type = types.NumericType

    def __str__(self):
        return f"Data({self.name}, subscript={self.subscript}, prime={self.prime})"


@dataclass
class Param(PrimeableExpr):
    lower: Expr = RealConstant(float("-inf"))
    upper: Expr = RealConstant(float("inf"))
    assigned_by_scan: bool = False

    def get_key(self):
        return self.name

    def __post_init__(self):
        self.out_type = types.NumericType

    def __str__(self):
        return f"Param({self.name}, subscript={self.subscript}, prime={self.prime}, lower={self.lower}, upper={self.upper}, assigned={self.assigned_by_scan})"


@dataclass
class Distr(Statement):
    """
    Distributions are statements in Rat, semantically equal to incrementing the log density given a distribution and
    a variate.
    variate_type is used to denote whether draws from the distributions are real or integer valued.
    """

    variate: Expr
    variate_type: Type[types.BaseType] = field(default=types.BaseType, kw_only=True)


@dataclass
class Normal(Distr):
    mean: Expr
    std: Expr

    def __iter__(self):
        yield self
        for expr in itertools.chain(self.variate, self.mean, self.std):
            yield expr

    def __post_init__(self):
        signatures = {
            (types.NumericType, types.NumericType): types.RealType,
        }
        self.variate_type = types.get_output_type(signatures, (self.mean.out_type, self.std.out_type))

    def __str__(self):
        return f"Normal({self.variate}, {self.mean}, {self.std})"


@dataclass
class BernoulliLogit(Distr):
    logit_p: Expr

    def __iter__(self):
        yield self
        for expr in itertools.chain(self.variate, self.logit_p):
            yield expr

    def __post_init__(self):
        signatures = {(types.NumericType,): types.IntegerType}
        self.variate_type = types.get_output_type(signatures, (self.logit_p.out_type,))

    def __str__(self):
        return f"BernoulliLogit({self.variate}, {self.logit_p})"


@dataclass
class LogNormal(Distr):
    mean: Expr
    std: Expr

    def __iter__(self):
        yield self
        for expr in itertools.chain(self.variate, self.mean, self.std):
            yield expr

    def __post_init__(self):
        signatures = {(types.NumericType, types.NumericType): types.RealType}
        self.variate_type = types.get_output_type(signatures, (self.mean.out_type, self.std.out_type))

    def __str__(self):
        return f"LogNormal({self.variate}, {self.mean}, {self.std})"


@dataclass
class Cauchy(Distr):
    location: Expr
    scale: Expr

    def __iter__(self):
        yield self
        for expr in itertools.chain(self.variate, self.location, self.scale):
            yield expr

    def __post_init__(self):
        signatures = {(types.NumericType, types.NumericType): types.RealType}
        self.variate_type = types.get_output_type(signatures, (self.location.out_type, self.scale.out_type))

    def __str__(self):
        return f"Cauchy({self.variate}, {self.location}, {self.scale})"


@dataclass
class Exponential(Distr):
    scale: Expr

    def __iter__(self):
        yield self
        for expr in itertools.chain(self.variate, self.scale):
            yield expr

    def __post_init__(self):
        signatures = {(types.NumericType,): types.RealType}
        self.variate_type = types.get_output_type(signatures, (self.scale.out_type,))

    def __str__(self):
        return f"Exponential({self.variate}, {self.scale})"


@dataclass
class UnaryExpr(Expr):
    subexpr: Expr

    def __iter__(self):
        yield self
        for expr in self.subexpr:
            yield expr


@dataclass
class BinaryExpr(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        yield self
        for expr in itertools.chain(self.left, self.right):
            yield expr


@dataclass
class Diff(BinaryExpr):
    def __post_init__(self):
        signatures = {
            (types.IntegerType, types.IntegerType): types.IntegerType,
            (types.NumericType, types.NumericType): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.left.out_type, self.right.out_type))

    def __str__(self):
        return f"Diff({self.left}, {self.right})"


@dataclass
class Sum(BinaryExpr):
    def __post_init__(self):
        signatures = {
            (types.IntegerType, types.IntegerType): types.IntegerType,
            (types.NumericType, types.NumericType): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.left.out_type, self.right.out_type))

    def __str__(self):
        return f"Sum({self.left}, {self.right})"


@dataclass
class Mul(BinaryExpr):
    def __post_init__(self):
        signatures = {
            (types.IntegerType, types.IntegerType): types.IntegerType,
            (types.NumericType, types.NumericType): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.left.out_type, self.right.out_type))

    def __str__(self):
        return f"Mul({self.left}, {self.right})"


@dataclass
class Pow(BinaryExpr):
    """
    left: base
    right: exponent
    """

    def __post_init__(self):
        signatures = {
            (types.IntegerType, types.IntegerType): types.IntegerType,
            (types.NumericType, types.NumericType): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.left.out_type, self.right.out_type))

    def __str__(self):
        return f"Pow({self.left}, {self.right})"


@dataclass
class Div(BinaryExpr):
    def __post_init__(self):
        signatures = {
            (types.NumericType, types.NumericType): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.left.out_type, self.right.out_type))

    def __str__(self):
        return f"Div({self.left}, {self.right})"


@dataclass
class Mod(BinaryExpr):
    def __post_init__(self):
        signatures = {
            (types.IntegerType, types.IntegerType): types.IntegerType,
            (types.NumericType, types.NumericType): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.left.out_type, self.right.out_type))

    def __str__(self):
        return f"Mod({self.left, self.right})"


@dataclass
class LessThan(BinaryExpr):
    def __post_init__(self):
        signatures = {
            (types.NumericType, types.NumericType): types.BooleanType,
        }
        self.out_type = types.get_output_type(signatures, (self.left.out_type, self.right.out_type))

    def __str__(self):
        return f"LessThan({self.left}, {self.right})"


@dataclass
class GreaterThan(BinaryExpr):
    def __post_init__(self):
        signatures = {
            (types.NumericType, types.NumericType): types.BooleanType,
        }
        self.out_type = types.get_output_type(signatures, (self.left.out_type, self.right.out_type))

    def __str__(self):
        return f"GreaterThan({self.left}, {self.right})"


@dataclass
class GreaterThanOrEq(BinaryExpr):
    def __post_init__(self):
        signatures = {
            (types.NumericType, types.NumericType): types.BooleanType,
        }
        self.out_type = types.get_output_type(signatures, (self.left.out_type, self.right.out_type))

    def __str__(self):
        return f"GreaterThanOrEq({self.left}, {self.right})"


@dataclass
class LessThanOrEq(BinaryExpr):
    def __post_init__(self):
        signatures = {
            (types.NumericType, types.NumericType): types.BooleanType,
        }
        self.out_type = types.get_output_type(signatures, (self.left.out_type, self.right.out_type))

    def __str__(self):
        return f"LessThanOrEq({self.left}, {self.right})"


@dataclass
class EqualTo(BinaryExpr):
    def __post_init__(self):
        signatures = {
            (types.NumericType, types.NumericType): types.BooleanType,
        }
        self.out_type = types.get_output_type(signatures, (self.left.out_type, self.right.out_type))

    def __str__(self):
        return f"EqualTo({self.left}, {self.right})"


@dataclass
class NotEqualTo(BinaryExpr):
    def __post_init__(self):
        signatures = {
            (types.NumericType, types.NumericType): types.BooleanType,
        }
        self.out_type = types.get_output_type(signatures, (self.left.out_type, self.right.out_type))

    def __str__(self):
        return f"NotEqualTo({self.left}, {self.right})"


@dataclass
class Assignment(Statement):
    lhs: PrimeableExpr
    rhs: Expr

    def __iter__(self):
        yield self
        for expr in itertools.chain(self.lhs, self.rhs):
            yield expr

    # def __post_init__(self):
    #     signatures = {
    #         (types.IntegerType,): types.IntegerType,
    #         (types.RealType,): types.RealType,
    #         (types.NumericType,): types.NumericType,
    #     }
    #     self.out_type = types.get_output_type(signatures, (self.rhs.out_type,))

    def __str__(self):
        return f"Assignment({self.lhs}, {self.rhs})"


@dataclass
class PrefixNegation(UnaryExpr):
    def __post_init__(self):
        signatures = {
            (types.IntegerType,): types.IntegerType,
            (types.RealType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"PrefixNegation({self.subexpr})"


@dataclass
class Prime(UnaryExpr):
    def code(self, scalar=False):
        return f"{self.subexpr.code(scalar)}"

    def __str__(self):
        return f"Prime({self.subexpr})"


@dataclass
class Sqrt(UnaryExpr):
    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Sqrt({self.subexpr})"


@dataclass
class Log(UnaryExpr):
    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Log({self.subexpr})"


@dataclass
class Exp(UnaryExpr):
    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Exp({self.subexpr})"


@dataclass
class Abs(UnaryExpr):
    def __post_init__(self):
        signatures = {
            (types.IntegerType,): types.IntegerType,
            (types.RealType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Abs({self.subexpr})"


@dataclass
class Floor(UnaryExpr):
    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.IntegerType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Floor({self.subexpr})"


@dataclass
class Ceil(UnaryExpr):
    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.IntegerType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Ceil({self.subexpr})"


@dataclass
class Round(UnaryExpr):
    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.IntegerType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Round({self.subexpr})"


@dataclass
class Sin(UnaryExpr):
    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Sin({self.subexpr})"


@dataclass
class Cos(UnaryExpr):
    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Cos({self.subexpr})"


@dataclass
class Tan(UnaryExpr):
    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Tan({self.subexpr})"


@dataclass
class Arcsin(UnaryExpr):
    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Arcsin({self.subexpr})"


@dataclass
class Arccos(UnaryExpr):
    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Arccos({self.subexpr})"


@dataclass
class Arctan(UnaryExpr):
    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Arctan({self.subexpr})"


@dataclass
class Logit(UnaryExpr):
    def __post_init__(self):
        signatures = {
            (types.RealType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Logit({self.subexpr})"


@dataclass
class InverseLogit(UnaryExpr):
    def __post_init__(self):
        signatures = {
            (types.RealType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"InverseLogit({self.subexpr})"


@dataclass
class IfElse(Expr):
    condition: Expr
    true_expr: Expr
    false_expr: Expr

    def __iter__(self):
        yield self
        for expr in itertools.chain(self.condition, self.true_expr, self.false_expr):
            yield expr

    def __post_init__(self):
        signatures = {
            (types.BooleanType, types.IntegerType, types.IntegerType): types.IntegerType,
            (types.BooleanType, types.RealType, types.RealType): types.RealType,
            (types.BooleanType, types.NumericType, types.NumericType): types.NumericType,
        }
        self.out_type = types.get_output_type(signatures, (self.condition.out_type, self.true_expr.out_type, self.false_expr.out_type))

    def __str__(self):
        return f"IfElse({self.condition}, {self.true_expr}, {self.false_expr})"


def search_tree(expr: Expr, *types) -> Expr:
    for child in expr:
        for _type in types:
            if isinstance(child, _type):
                yield child
