from codeop import Compile
from dataclasses import dataclass, field
from distutils.errors import CompileError
from typing import Tuple, Union, Type, List, Dict
import jax


from . import variables
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
        pass

    def code(self, scalar=False):
        pass

    def __str__(self):
        return "Expr()"


@dataclass
class RealConstant(Expr):
    """
    Elementary value, cannot be folded anymore
    """

    value: float

    def code(self, scalar=False):
        if self.value == float("inf"):
            return "float('inf')"
        elif self.value == float("-inf"):
            return "float('-inf')"
        return f"{self.value}"

    def __post_init__(self):
        self.out_type = types.RealType

    def __str__(self):
        return f"RealConstant({self.value})"


@dataclass
class IntegerConstant(Expr):
    """
    Elementary value, cannot be folded anymore
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
    Elementary value, cannot be folded anymore
    """

    names: Tuple[str]
    shifts: Tuple[Union[int, None]] = (None,)
    variable: variables.SubscriptUse = None

    def get_key(self):
        return self.names

    def code(self, scalar=False):
        if scalar:
            return f"subscripts['{self.variable.code()}'][index]"
        else:
            return f"subscripts['{self.variable.code()}']"

    def __post_init__(self):
        self.out_type = types.SubscriptSetType

    def __str__(self):
        return f"Subscript(names=({', '.join(x.__str__() for x in self.names)}), shift=({', '.join(x.__str__() for x in self.shifts)}))"


@dataclass
class Shift(Expr):
    """
    This is a function with signature (Subscript, Integer) -> Subscript
    """

    subscript: Expr
    shift_expr: Expr

    def __post_init__(self):
        signatures = {(types.SubscriptSetType, types.IntegerType): types.SubscriptSetType}
        self.out_type = types.get_output_type(signatures, (self.subscript.out_type, self.shift_expr.out_type))

    def __str__(self):
        return f"Shift(subscript={self.subscript}, amount={self.shift_expr})"


@dataclass
class PrimeableExpr(Expr):
    name: str
    prime: bool = False
    subscript: Subscript = None


@dataclass
class Data(PrimeableExpr):
    variable: variables.Data = None

    def get_key(self):
        return self.name

    def code(self, scalar=False):
        variable_code = self.variable.code()
        if self.subscript is not None:
            return f"data['{variable_code}'][{self.subscript.code(scalar)}]"
        else:
            return f"data['{variable_code}']"

    def fold(self):
        if self.subscript:
            self.subscript = self.subscript.fold()
        return Data(name=self.name, prime=self.prime, subscript=self.subscript, line_index=self.line_index, column_index=self.column_index)

    def __post_init__(self):
        self.out_type = types.NumericType

    def __str__(self):
        if self.subscript is None:
            return f"Data({self.name}, {self.prime})"
        else:
            return f"Data({self.name}, {self.prime}, {self.subscript})"
        # return f"Placeholder({self.name}, {self.subscript.__str__()}) = {{{self.value.__str__()}}}"


@dataclass
class Param(PrimeableExpr):
    variable: variables.Param = None
    lower: Expr = None  # RealConstant(float("-inf"))
    upper: Expr = None  # RealConstant(float("inf"))
    assigned_by_scan: bool = False

    def __iter__(self):
        if self.subscript is not None:
            return iter([self.subscript])
        else:
            return iter([])

    def scalar(self):
        return self.subscript is None

    def get_key(self):
        return self.name

    def __post_init__(self):
        self.out_type = types.NumericType

    def code(self, scalar=False):
        variable_code = self.variable.code()
        if self.subscript:
            if self.assigned_by_scan:
                if scalar == False:
                    msg = "Internal error: Cannot do a vector evaluation of a recursively assigned variable"
                    raise CompileError(msg, self.line_index, self.column_index)

                integer_shifts = [shift for shift in self.subscript.shifts if shift is not None]

                if len(integer_shifts) != 1:
                    msg = "Internal error: There should be exactly one shift"
                    raise CompileError(msg, self.line_index, self.column_index)

                shift = integer_shifts[0]

                if shift <= 0:
                    msg = "Internal error: Shifts must be positive"
                    raise CompileError(msg, self.line_index, self.column_index)

                return f"carry{shift - 1}"
            else:
                return f"parameters['{variable_code}'][{self.subscript.code(scalar)}]"
        else:
            return f"parameters['{variable_code}']"

    def __str__(self):
        return f"Param({self.name}, subscript={self.subscript.__str__()}, lower={self.lower}, upper={self.upper}, assigned={self.assigned_by_scan})"


@dataclass
class Distr(Expr):
    variate: Expr


@dataclass
class Normal(Distr):
    mean: Expr
    std: Expr

    def __iter__(self):
        return iter([self.variate, self.mean, self.std])

    def code(self, scalar=False):
        return f"jax.scipy.stats.norm.logpdf({self.variate.code(scalar)}, {self.mean.code(scalar)}, {self.std.code(scalar)})"

    def __post_init__(self):
        signatures = {
            (types.NumericType, types.NumericType): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.mean.out_type, self.std.out_type))

    def __str__(self):
        return f"Normal({self.variate.__str__()}, {self.mean.__str__()}, {self.std.__str__()})"


@dataclass
class BernoulliLogit(Distr):
    logit_p: Expr

    def __iter__(self):
        return iter([self.variate, self.logit_p])

    def code(self, scalar=False):
        return f"rat.math.bernoulli_logit({self.variate.code(scalar)}, {self.logit_p.code(scalar)})"

    def __post_init__(self):
        signatures = {(types.NumericType,): types.IntegerType}
        self.out_type = types.get_output_type(signatures, (self.logit_p.out_type,))

    def __str__(self):
        return f"BernoulliLogit({self.variate.__str__()}, {self.logit_p.__str__()})"


@dataclass
class LogNormal(Distr):
    mean: Expr
    std: Expr

    def __iter__(self):
        return iter([self.variate, self.mean, self.std])

    def code(self, scalar=False):
        return f"rat.math.log_normal({self.variate.code(scalar)}, {self.mean.code(scalar)}, {self.std.code(scalar)})"

    def __post_init__(self):
        signatures = {(types.NumericType, types.NumericType): types.RealType}
        self.out_type = types.get_output_type(signatures, (self.mean.out_type, self.std.out_type))

    def __str__(self):
        return f"LogNormal({self.variate.__str__()}, {self.mean.__str__()}, {self.std.__str__()})"


@dataclass
class Cauchy(Distr):
    location: Expr
    scale: Expr

    def __iter__(self):
        return iter([self.variate, self.location, self.scale])

    def code(self, scalar=False):
        return f"jax.scipy.stats.cauchy.logpdf({self.variate.code(scalar)}, {self.location.code(scalar)}, {self.scale.code(scalar)})"

    def __post_init__(self):
        signatures = {(types.NumericType, types.NumericType): types.RealType}
        self.out_type = types.get_output_type(signatures, (self.location.out_type, self.scale.out_type))

    def __str__(self):
        return f"Cauchy({self.variate.__str__()}, {self.location.__str__()}, {self.scale.__str__()})"


@dataclass
class Exponential(Distr):
    scale: Expr

    def __iter__(self):
        return iter([self.variate, self.scale])

    def code(self, scalar=False):
        return f"jax.scipy.stats.expon.logpdf({self.variate.code(scalar)}, loc=0, scale={self.scale.code(scalar)})"

    def __post_init__(self):
        signatures = {(types.NumericType,): types.RealType}
        self.out_type = types.get_output_type(signatures, (self.scale.out_type,))

    def __str__(self):
        return f"Exponential({self.variate.__str__()}, {self.scale.__str__()})"


@dataclass
class Diff(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self, scalar=False):
        return f"({self.left.code(scalar)} - {self.right.code(scalar)})"

    def __post_init__(self):
        signatures = {
            (types.IntegerType, types.IntegerType): types.IntegerType,
            (types.NumericType, types.NumericType): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.left.out_type, self.right.out_type))

    def __str__(self):
        return f"Diff({self.left.__str__()}, {self.right.__str__()})"


@dataclass
class Sum(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self, scalar=False):
        return f"{self.left.code(scalar)} + {self.right.code(scalar)}"

    def __post_init__(self):
        signatures = {
            (types.IntegerType, types.IntegerType): types.IntegerType,
            (types.NumericType, types.NumericType): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.left.out_type, self.right.out_type))

    def __str__(self):
        return f"Sum({self.left.__str__()}, {self.right.__str__()})"


@dataclass
class Mul(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self, scalar=False):
        return f"{self.left.code(scalar)} * {self.right.code(scalar)}"

    def __post_init__(self):
        signatures = {
            (types.IntegerType, types.IntegerType): types.IntegerType,
            (types.NumericType, types.NumericType): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.left.out_type, self.right.out_type))

    def __str__(self):
        return f"Mul({self.left.__str__()}, {self.right.__str__()})"


@dataclass
class Pow(Expr):
    base: Expr
    exponent: Expr

    def __iter__(self):
        return iter([self.base, self.exponent])

    def code(self, scalar=False):
        return f"{self.base.code(scalar)} ** {self.exponent.code(scalar)}"

    def __post_init__(self):
        signatures = {
            (types.IntegerType, types.IntegerType): types.IntegerType,
            (types.NumericType, types.NumericType): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.base.out_type, self.exponent.out_type))

    def __str__(self):
        return f"Pow({self.base.__str__()}, {self.exponent.__str__()})"


@dataclass
class Div(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self, scalar=False):
        return f"{self.left.code(scalar)} / {self.right.code(scalar)}"

    def __post_init__(self):
        signatures = {
            (types.NumericType, types.NumericType): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.left.out_type, self.right.out_type))

    def __str__(self):
        return f"Div({self.left.__str__()}, {self.right.__str__()})"


@dataclass
class Mod(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self, scalar=False):
        return f"{self.left.code(scalar)} % {self.right.code(scalar)}"

    def __post_init__(self):
        signatures = {
            (types.IntegerType, types.IntegerType): types.IntegerType,
            (types.NumericType, types.NumericType): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.left.out_type, self.right.out_type))

    def __str__(self):
        return f"Mod({self.left.__str__(), self.right.__str__()})"


@dataclass
class PrefixNegation(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self, scalar=False):
        return f"-{self.subexpr.code(scalar)}"

    def __post_init__(self):
        signatures = {
            (types.IntegerType,): types.IntegerType,
            (types.RealType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"PrefixNegation({self.subexpr.__str__()})"


@dataclass
class Assignment(Expr):
    lhs: Expr
    rhs: Expr

    def __iter__(self):
        return iter([self.lhs, self.rhs])

    def code(self, scalar=False):
        return f"{self.lhs.code(scalar)} = {self.rhs.code(scalar)}"

    def __post_init__(self):
        signatures = {
            (types.IntegerType,): types.IntegerType,
            (types.RealType,): types.RealType,
            (types.NumericType,): types.NumericType,
        }
        self.out_type = types.get_output_type(signatures, (self.rhs.out_type,))

    def __str__(self):
        return f"Assignment({self.lhs.__str__()}, {self.rhs.__str__()})"


class Prime(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self, scalar=False):
        return f"{self.subexpr.code(scalar)}"

    def __str__(self):
        return f"Prime({self.subexpr.__str__()})"


@dataclass
class Sqrt(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self, scalar=False):
        return f"jax.numpy.sqrt({self.subexpr.code(scalar)})"

    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Sqrt({self.subexpr.__str__()})"


@dataclass
class Log(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self, scalar=False):
        return f"jax.numpy.log({self.subexpr.code(scalar)})"

    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Log({self.subexpr.__str__()})"


@dataclass
class Exp(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self, scalar=False):
        return f"jax.numpy.exp({self.subexpr.code(scalar)})"

    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Exp({self.subexpr.__str__()})"


@dataclass
class Abs(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self, scalar=False):
        return f"jax.numpy.abs({self.subexpr.code(scalar)})"

    def __post_init__(self):
        signatures = {
            (types.IntegerType,): types.IntegerType,
            (types.RealType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Abs({self.subexpr.__str__()})"


@dataclass
class Floor(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self, scalar=False):
        return f"jax.numpy.floor({self.subexpr.code(scalar)})"

    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.IntegerType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Floor({self.subexpr.__str__()})"


@dataclass
class Ceil(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self, scalar=False):
        return f"jax.numpy.ceil({self.subexpr.code(scalar)})"

    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.IntegerType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Ceil({self.subexpr.__str__()})"


@dataclass
class Round(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self, scalar=False):
        return f"jax.numpy.round({self.subexpr.code(scalar)})"

    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.IntegerType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Round({self.subexpr.__str__()})"


@dataclass
class Sin(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self, scalar=False):
        return f"jax.numpy.sin({self.subexpr.code(scalar)})"

    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Sin({self.subexpr.__str__()})"


@dataclass
class Cos(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self, scalar=False):
        return f"jax.numpy.cos({self.subexpr.code(scalar)})"

    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Cos({self.subexpr.__str__()})"


@dataclass
class Tan(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self, scalar=False):
        return f"jax.numpy.tan({self.subexpr.code(scalar)})"

    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Tan({self.subexpr.__str__()})"


@dataclass
class Arcsin(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self, scalar=False):
        return f"jax.numpy.arcsin({self.subexpr.code(scalar)})"

    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Arcsin({self.subexpr.__str__()})"


@dataclass
class Arccos(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self, scalar=False):
        return f"jax.numpy.arccos({self.subexpr.code(scalar)})"

    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Arccos({self.subexpr.__str__()})"


@dataclass
class Arctan(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self, scalar=False):
        return f"jax.numpy.arctan({self.subexpr.code(scalar)})"

    def __post_init__(self):
        signatures = {
            (types.NumericType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Arctan({self.subexpr.__str__()})"


@dataclass
class Logit(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self, scalar=False):
        return f"jax.scipy.special.logit({self.subexpr.code(scalar)})"

    def __post_init__(self):
        signatures = {
            (types.RealType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

    def __str__(self):
        return f"Logit({self.subexpr.__str__()})"


@dataclass
class InverseLogit(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self, scalar=False):
        return f"jax.scipy.special.expit({self.subexpr.code(scalar)})"

    def __post_init__(self):
        signatures = {
            (types.RealType,): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type,))

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

    def code(self, scalar=False):
        if self.index is not None:
            return f"{self.name}[{self.index.code(scalar)}]"
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
