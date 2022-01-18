from dataclasses import dataclass, field
from typing import Tuple, Union, Type, List

from . import variables
from . import types


class ConstantFoldError(Exception):
    def __init__(self, msg):
        super.__init__(msg)

@dataclass
class Expr:
    line_index: int = field(default=-1, kw_only=True)  # line index of the original model code
    column_index: int = field(default=-1, kw_only=True)  # column index of the original model code
    out_type: Type[types.BaseType] = field(default=types.BaseType)

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def __post_init__(self):
        """
        post_init must resolve and set self.out_type
        """
        pass

    def fold(self):
        """
        method for constant folding, if applicable. Should return self if not
        """
        pass

    def code(self):
        pass

    def __str__(self):
        return "Expr()"


@dataclass
class RealConstant(Expr):
    """
    Elementary value, cannot be folded anymore
    """
    def code(self):
        if self.value == float("inf"):
            return "float('inf')"
        elif self.value == float("-inf"):
            return "float('-inf')"
        return f"{self.value}"

    def __post_init__(self):
        self.out_type = types.RealType

    def fold(self):
        return self

    def __str__(self):
        return f"RealConstant({self.value})"


@dataclass
class IntegerConstant(Expr):
    """
    Elementary value, cannot be folded anymore
    """
    value: int

    def code(self):
        return f"{self.value}"

    def __post_init__(self):
        self.out_type = types.IntegerType

    def fold(self):
        return self

    def __str__(self):
        return f"IntegerConstant({self.value})"


@dataclass
class Subscript(Expr):
    """
    Elementary value, cannot be folded anymore
    """
    names: Tuple[str]
    shifts: Tuple[Union[str, None]] = (None,)
    variable: variables.SubscriptUse = None

    def get_key(self):
        return self.names

    def code(self):
        return f"subscripts['{self.variable.code()}']"

    def __post_init__(self):
        self.out_type = types.SubscriptType

    def fold(self):
        return self

    def __str__(self):
        return f"Index(names=({', '.join(x.__str__() for x in self.names)}), shift=({', '.join(x.__str__() for x in self.shifts)}))"


@dataclass
class SubscriptOp(Expr):
    """
    SubscriptOp = operator[] (Subscript, ..., Subscript)
    (Subscript, ..., Subscript) -> Subscript
    """
    subscripts = List[Expr]

    def __post_init__(self):
        signatures = {
            (types.SubscriptType, ) * len(self.subexprs): types.SubscriptType
        }
        self.out_type = types.get_output_type(signatures, tuple([x.out_types for x in self.subexprs]))

    def fold(self) -> Subscript:
        """
        Combines multiple subscripts into a single subscript
        """
        names = []
        shifts = []
        for subscript in self.subscripts:
            names.extend(subscript.name)
            shifts.extend(subscript.shifts)
        return Subscript(names=tuple(names), shifts=tuple(shifts), line_index=self.line_index, column_index=self.column_index)




@dataclass
class Shift(Expr):
    """
    This is a function with signature (Subscript, Integer) -> Subscript
    """
    subscript: Subscript
    shift_expr: Expr

    def __post_init__(self):
        signatures = {
            (types.SubscriptType, types.IntegerType): types.SubscriptType
        }
        self.out_type = types.get_output_type(signatures, (self.subscript.out_type, self.shift_expr.out_type))

    def fold(self) -> Subscript:
        folded_shifts = []
        for shift in self.shift_expr:
            if not shift:
                folded_shifts.append(None)
            else:
                folded_shifts.append(shift.fold())
        return Subscript(names=self.subscript.names, shifts=tuple(folded_shifts), line_index=self.subscript.line_index,
                         column_index=self.subscript.column_index)

    def __str__(self):
        return f"shift(subscript={self.name}, amount={self.shift_expr})"


@dataclass
class Data(Expr):
    name: str
    subscript: Union[Subscript, SubscriptOp] = None
    variable: variables.Data = None

    def get_key(self):
        return self.name

    def code(self):
        variable_code = self.variable.code()
        if self.subscript is not None:
            return f"data['{variable_code}'][{self.subscript.code()}]"
        else:
            return f"data['{variable_code}']"

    def fold(self):
        return self

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

    def fold(self):
        self.lower = self.lower.fold()
        self.upper = self.upper.fold()
        self.subscript = self.subscript.fold()
        if not issubclass(self.lower.out_type, types.NumericType):
            raise ConstantFoldError(f"Lower bound value must fold-able into a Numeric constant at compile time, but folded expression {self.lower} is not a constant!")
        if not issubclass(self.upper.out_type, types.NumericType):
            raise ConstantFoldError(f"Upper bound value must fold-able into a Numeric constant at compile time, but folded expression {self.upper} is not a constant!")
        return self

    def code(self):
        variable_code = self.variable.code()
        if self.subscript:
            return f"parameters['{variable_code}'][{self.subscript.code()}]"
        else:
            return f"parameters['{variable_code}']"

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

    def __post_init__(self):
        signatures = {
            (types.NumericType, types.NumericType): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.mean.out_type, self.std.out_type))

    def __str__(self):
        return f"Normal({self.variate.__str__()}, {self.mean.__str__()}, {self.std.__str__()})"


@dataclass
class BernoulliLogit(Distr):
    variate: Expr
    logit_p: Expr

    def __iter__(self):
        return iter([self.variate, self.logit_p])

    def code(self):
        return f"rat.math.bernoulli_logit({self.variate.code()}, {self.logit_p.code()})"

    def __post_init__(self):
        signatures = {
            (types.NumericType, ): types.IntegerType
        }
        self.out_type = types.get_output_type(signatures, (self.logit_p.out_type, ))

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
        return f"rat.math.log_normal({self.variate.code()}, {self.mean.code()}, {self.std.code()})"

    def __post_init__(self):
        signatures = {
            (types.NumericType, types.NumericType): types.RealType
        }
        self.out_type = types.get_output_type(signatures, (self.mean.out_type, self.std.out_type))

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

    def __post_init__(self):
        signatures = {
            (types.NumericType, types.NumericType): types.RealType
        }
        self.out_type = types.get_output_type(signatures, (self.location.out_type, self.scale.out_type))

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

    def __post_init__(self):
        signatures = {
            (types.NumericType, ): types.RealType
        }
        self.out_type = types.get_output_type(signatures, (self.scale.out_type, ))

    def __str__(self):
        return f"Exponential({self.variate.code()}, {self.scale.code()})"


@dataclass
class Diff(Expr):
    left: Expr
    right: Expr

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"({self.left.code()} - {self.right.code()})"

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

    def code(self):
        return f"{self.left.code()} + {self.right.code()}"

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

    def code(self):
        return f"{self.left.code()} * {self.right.code()}"

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

    def code(self):
        return f"{self.base.code()} ** {self.exponent.code()}"

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

    def code(self):
        return f"{self.left.code()} / {self.right.code()}"

    def __post_init__(self):
        signatures = {
            (types.NumericType, types.NumericType): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.left.out_type, self.right.out_type))

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

    def code(self):
        return f"-{self.subexpr.code()}"

    def __post_init__(self):
        signatures = {
            (types.IntegerType, ): types.IntegerType,
            (types.RealType, ): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type, ))

    def __str__(self):
        return f"PrefixNegation({self.subexpr.__str__()})"


@dataclass
class Assignment(Expr):
    lhs: Expr
    rhs: Expr

    def __iter__(self):
        return iter([self.lhs, self.rhs])

    def code(self):
        return f"{self.lhs.code()} = {self.rhs.code()}"

    def __post_init__(self):
        signatures = {
            (types.IntegerType, ): types.IntegerType,
            (types.RealType, ): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.rhs.out_type, ))

    def __str__(self):
        return f"Assignment({self.lhs.__str__()}, {self.rhs.__str__()})"


@dataclass
class Log(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.log({self.subexpr.code()})"

    def __post_init__(self):
        signatures = {
            (types.NumericType, ): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type, ))

    def __str__(self):
        return f"Log({self.subexpr.__str__()})"


@dataclass
class Exp(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.exp({self.subexpr.code()})"

    def __post_init__(self):
        signatures = {
            (types.NumericType, ): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type, ))

    def __str__(self):
        return f"Exp({self.subexpr.__str__()})"


@dataclass
class Abs(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.abs({self.subexpr.code()})"

    def __post_init__(self):
        signatures = {
            (types.IntegerType, ): types.IntegerType,
            (types.RealType, ): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type, ))

    def __str__(self):
        return f"Abs({self.subexpr.__str__()})"


@dataclass
class Floor(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.floor({self.subexpr.code()})"

    def __post_init__(self):
        signatures = {
            (types.NumericType, ): types.IntegerType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type, ))

    def __str__(self):
        return f"Floor({self.subexpr.__str__()})"


@dataclass
class Ceil(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.ceil({self.subexpr.code()})"

    def __post_init__(self):
        signatures = {
            (types.NumericType, ): types.IntegerType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type, ))

    def __str__(self):
        return f"Ceil({self.subexpr.__str__()})"


@dataclass
class Round(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.round({self.subexpr.code()})"

    def __post_init__(self):
        signatures = {
            (types.NumericType, ): types.IntegerType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type, ))

    def __str__(self):
        return f"Round({self.subexpr.__str__()})"


@dataclass
class Sin(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.sin({self.subexpr.code()})"

    def __post_init__(self):
        signatures = {
            (types.NumericType, ): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type, ))

    def __str__(self):
        return f"Sin({self.subexpr.__str__()})"


@dataclass
class Cos(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.cos({self.subexpr.code()})"

    def __post_init__(self):
        signatures = {
            (types.NumericType, ): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type, ))

    def __str__(self):
        return f"Cos({self.subexpr.__str__()})"


@dataclass
class Tan(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.tan({self.subexpr.code()})"

    def __post_init__(self):
        signatures = {
            (types.NumericType, ): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type, ))

    def __str__(self):
        return f"Tan({self.subexpr.__str__()})"


@dataclass
class Arcsin(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.arcsin({self.subexpr.code()})"

    def __post_init__(self):
        signatures = {
            (types.NumericType, ): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type, ))

    def __str__(self):
        return f"Arcsin({self.subexpr.__str__()})"


@dataclass
class Arccos(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.arccos({self.subexpr.code()})"

    def __post_init__(self):
        signatures = {
            (types.NumericType, ): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type, ))

    def __str__(self):
        return f"Arccos({self.subexpr.__str__()})"


@dataclass
class Arctan(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.numpy.arctan({self.subexpr.code()})"

    def __post_init__(self):
        signatures = {
            (types.NumericType, ): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type, ))

    def __str__(self):
        return f"Arctan({self.subexpr.__str__()})"


@dataclass
class Logit(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.scipy.special.logit({self.subexpr.code()})"

    def __post_init__(self):
        signatures = {
            (types.NumericType, ): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type, ))

    def __str__(self):
        return f"Logit({self.subexpr.__str__()})"


@dataclass
class InverseLogit(Expr):
    subexpr: Expr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"jax.scipy.special.expit({self.subexpr.code()})"

    def __post_init__(self):
        signatures = {
            (types.NumericType, ): types.RealType,
        }
        self.out_type = types.get_output_type(signatures, (self.subexpr.out_type, ))

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
