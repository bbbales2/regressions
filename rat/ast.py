from dataclasses import dataclass, field
from distutils.errors import CompileError
from typing import Tuple, Union, Type, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from . import ir


from . import types


@dataclass
class Expr:
    line_index: int = field(default=-1, kw_only=True)  # line index of the original model code
    column_index: int = field(default=-1, kw_only=True)  # column index of the original model code
    out_type: Type[types.BaseType] = field(default=types.BaseType, kw_only=True)

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def __post_init__(self):
        """
        __post_init__ must resolve and set self.out_type
        """
        raise NotImplementedError()

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        """
        accept()
        """
        raise NotImplementedError()

    def __str__(self):
        return "Expr()"


@dataclass
class RealConstant(Expr):
    """
    Elementary value
    """

    value: float

    def __post_init__(self):
        self.out_type = types.RealType

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_RealConstant(self, *args, **kwargs)

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
class Subscript(Expr):
    """
    Elementary value
    """

    names: Tuple["SubscriptColumn"]
    shifts: Tuple[Expr]

    def get_key(self):
        return self.names

    def __post_init__(self):
        assert len(self.shifts) == len(
            self.names
        ), "Internal Error: length of types.Subscript.names must equal length of types.Subscript.shifts"
        signatures = {(types.SubscriptSetType,) * len(self.shifts) + (types.IntegerType,) * len(self.shifts): types.SubscriptSetType}
        self.out_type = types.get_output_type(signatures, tuple(expr.out_type for expr in (*self.names, *self.shifts)))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Subscript(self, *args, **kwargs)


    def __str__(self):
        return f"Subscript(names=({', '.join(x.__str__() for x in self.names)}), shift=({', '.join(x.__str__() for x in self.shifts)}))"


@dataclass
class SubscriptColumn(Expr):
    """
    columns/names used as subscripts
    """

    name: str

    def __post_init__(self):
        self.out_type = types.SubscriptSetType

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_SubscriptColumn(self, *args, **kwargs)

    def __str__(self):
        return f"SubscriptColumn({self.name})"


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


@dataclass
class Data(PrimeableExpr):
    def get_key(self):
        return self.name

    def __post_init__(self):
        self.out_type = types.NumericType

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Data(self, *args, **kwargs)

    def __str__(self):
        return f"Data({self.name}, subscript={self.subscript}, prime={self.prime})"


@dataclass
class Param(PrimeableExpr):
    lower: Expr = RealConstant(value=float("-inf"))
    upper: Expr = RealConstant(value=float("inf"))
    assigned_by_scan: bool = False

    def __iter__(self):
        if self.subscript is not None:
            return iter([self.subscript])
        else:
            return iter([])

    def get_key(self):
        return self.name

    def __post_init__(self):
        self.out_type = types.NumericType

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Param(self, *args, **kwargs)

    def __str__(self):
        return f"Param({self.name}, subscript={self.subscript}, prime={self.prime}, lower={self.lower}, upper={self.upper}, assigned={self.assigned_by_scan})"


@dataclass
class Distr(Expr):
    variate: Expr


@dataclass
class Normal(Distr):
    mean: Expr
    std: Expr

    def __iter__(self):
        return iter([self.variate, self.mean, self.std])

    def __post_init__(self):
        signatures = {
            (types.NumericType, types.NumericType): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.mean.out_type, self.std.out_type))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Normal(self, *args, **kwargs)

    def __str__(self):
        return f"Normal({self.variate}, {self.mean}, {self.std})"


@dataclass
class BernoulliLogit(Distr):
    logit_p: Expr

    def __iter__(self):
        return iter([self.variate, self.logit_p])

    def __post_init__(self):
        signatures = {(types.NumericType,): types.IntegerType}
        self.out_type = types.get_output_type(signatures, (self.logit_p.out_type,))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_BernoulliLogit(self, *args, **kwargs)

    def __str__(self):
        return f"BernoulliLogit({self.variate}, {self.logit_p})"


@dataclass
class LogNormal(Distr):
    mean: Expr
    std: Expr

    def __iter__(self):
        return iter([self.variate, self.mean, self.std])

    def __post_init__(self):
        signatures = {(types.NumericType, types.NumericType): types.RealType}
        self.out_type = types.get_output_type(signatures, (self.mean.out_type, self.std.out_type))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_LogNormal(self, *args, **kwargs)

    def __str__(self):
        return f"LogNormal({self.variate}, {self.mean}, {self.std})"


@dataclass
class Cauchy(Distr):
    location: Expr
    scale: Expr

    def __iter__(self):
        return iter([self.variate, self.location, self.scale])

    def __post_init__(self):
        signatures = {(types.NumericType, types.NumericType): types.RealType}
        self.out_type = types.get_output_type(signatures, (self.location.out_type, self.scale.out_type))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Cauchy(self, *args, **kwargs)

    def __str__(self):
        return f"Cauchy({self.variate}, {self.location}, {self.scale})"


@dataclass
class Exponential(Distr):
    scale: Expr

    def __iter__(self):
        return iter([self.variate, self.scale])

    def __post_init__(self):
        signatures = {(types.NumericType,): types.RealType}
        self.out_type = types.get_output_type(signatures, (self.scale.out_type,))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Exponential(self, *args, **kwargs)

    def __str__(self):
        return f"Exponential({self.variate}, {self.scale})"


@dataclass
class Diff(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def __post_init__(self):
        signatures = {
            (types.IntegerType, types.IntegerType): types.IntegerType,
            (types.NumericType, types.NumericType): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.left.out_type, self.right.out_type))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Diff(self, *args, **kwargs)

    def __str__(self):
        return f"Diff({self.left}, {self.right})"


@dataclass
class Sum(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def __post_init__(self):
        signatures = {
            (types.IntegerType, types.IntegerType): types.IntegerType,
            (types.NumericType, types.NumericType): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.left.out_type, self.right.out_type))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Sum(self, *args, **kwargs)

    def __str__(self):
        return f"Sum({self.left}, {self.right})"


@dataclass
class Mul(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def __post_init__(self):
        signatures = {
            (types.IntegerType, types.IntegerType): types.IntegerType,
            (types.NumericType, types.NumericType): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.left.out_type, self.right.out_type))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Mul(self, *args, **kwargs)

    def __str__(self):
        return f"Mul({self.left}, {self.right})"


@dataclass
class Pow(Expr):
    base: Expr
    exponent: Expr

    def __iter__(self):
        return iter([self.base, self.exponent])

    def __post_init__(self):
        signatures = {
            (types.IntegerType, types.IntegerType): types.IntegerType,
            (types.NumericType, types.NumericType): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.base.out_type, self.exponent.out_type))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Pow(self, *args, **kwargs)

    def __str__(self):
        return f"Pow({self.base}, {self.exponent})"


@dataclass
class Div(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def __post_init__(self):
        signatures = {
            (types.NumericType, types.NumericType): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.left.out_type, self.right.out_type))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Div(self, *args, **kwargs)

    def __str__(self):
        return f"Div({self.left}, {self.right})"


@dataclass
class Mod(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def __post_init__(self):
        signatures = {
            (types.IntegerType, types.IntegerType): types.IntegerType,
            (types.NumericType, types.NumericType): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.left.out_type, self.right.out_type))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Mod(self, *args, **kwargs)

    def __str__(self):
        return f"Mod({self.left, self.right})"


@dataclass
class PrefixNegation(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def __post_init__(self):
        signatures = {
            (types.IntegerType,): types.IntegerType,
            (types.RealType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_PrefixNegation(self, *args, **kwargs)

    def __str__(self):
        return f"PrefixNegation({self.subexpr})"


@dataclass
class Assignment(Expr):
    lhs: PrimeableExpr
    rhs: Expr

    def __iter__(self):
        return iter([self.lhs, self.rhs])

    def __post_init__(self):
        signatures = {
            (types.IntegerType,): types.IntegerType,
            (types.RealType,): types.RealType,
            (types.NumericType,): types.NumericType,
        }
        self.out_type = types.get_output_type(signatures, (self.rhs.out_type,))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Assignment(self, *args, **kwargs)

    def __str__(self):
        return f"Assignment({self.lhs}, {self.rhs})"


class Prime(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self, scalar=False):
        return f"{self.subexpr.code(scalar)}"

    def __str__(self):
        return f"Prime({self.subexpr})"


@dataclass
class Sqrt(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Sqrt(self, *args, **kwargs)

    def __str__(self):
        return f"Sqrt({self.subexpr})"


@dataclass
class Log(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Log(self, *args, **kwargs)

    def __str__(self):
        return f"Log({self.subexpr})"


@dataclass
class Exp(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Exp(self, *args, **kwargs)

    def __str__(self):
        return f"Exp({self.subexpr})"


@dataclass
class Abs(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def __post_init__(self):
        signatures = {
            (types.IntegerType,): types.IntegerType,
            (types.RealType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Abs(self, *args, **kwargs)

    def __str__(self):
        return f"Abs({self.subexpr})"


@dataclass
class Floor(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.IntegerType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Floor(self, *args, **kwargs)

    def __str__(self):
        return f"Floor({self.subexpr})"


@dataclass
class Ceil(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.IntegerType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Ceil(self, *args, **kwargs)

    def __str__(self):
        return f"Ceil({self.subexpr})"


@dataclass
class Round(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.IntegerType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Round(self, *args, **kwargs)

    def __str__(self):
        return f"Round({self.subexpr})"


@dataclass
class Sin(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Sin(self, *args, **kwargs)

    def __str__(self):
        return f"Sin({self.subexpr})"


@dataclass
class Cos(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Cos(self, *args, **kwargs)
    def __str__(self):
        return f"Cos({self.subexpr})"


@dataclass
class Tan(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Tan(self, *args, **kwargs)
    def __str__(self):
        return f"Tan({self.subexpr})"


@dataclass
class Arcsin(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Arcsin(self, *args, **kwargs)
    def __str__(self):
        return f"Arcsin({self.subexpr})"


@dataclass
class Arccos(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Arccos(self, *args, **kwargs)
    def __str__(self):
        return f"Arccos({self.subexpr})"


@dataclass
class Arctan(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Arctan(self, *args, **kwargs)
    def __str__(self):
        return f"Arctan({self.subexpr})"


@dataclass
class Logit(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def __post_init__(self):
        signatures = {
            (types.RealType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_Logit(self, *args, **kwargs)
    def __str__(self):
        return f"Logit({self.subexpr})"


@dataclass
class InverseLogit(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def __post_init__(self):
        signatures = {
            (types.RealType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def accept(self, visitor: "ir.BaseVisitor", *args, **kwargs):
        visitor.visit_InverseLogit(self, *args, **kwargs)
    def __str__(self):
        return f"InverseLogit({self.subexpr})"


def search_tree(expr: Expr, *types) -> Expr:
    for _type in types:
        if isinstance(expr, _type):
            yield expr
    else:
        for child in expr:
            for child_expr in search_tree(child, *types):
                yield child_expr
