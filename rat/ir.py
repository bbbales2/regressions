from codeop import Compile
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from typing import Tuple, Union, Type, List, Dict, TYPE_CHECKING, Set
if TYPE_CHECKING:
    from . import ast

from . import types


class BaseVisitor:
    def __init__(self):
        self.expression_string = ""

    def visit_IntegerConstant(self, integer_node: ast.IntegerConstant, *args, **kwargs):
        self.expression_string += integer_node.value

    def visit_RealConstant(self, real_node: ast.RealConstant, *args, **kwargs):
        if real_node.value == float("inf"):
            self.expression_string += "float('inf')"
        elif real_node.value == float("-inf"):
            self.expression_string += "float('-inf')"
        else:
            self.expression_string += str(real_node.value)

    def visit_Subscript(self, subscript_node: ast.Subscript, *args, **kwargs):
        raise NotImplementedError()

    def visit_SubscriptColumn(self, column_node: ast.SubscriptColumn, *args, **kwargs):
        raise NotImplementedError()

    def visit_Shift(self, shift_node: ast.Shift, *args, **kwargs):
        raise NotImplementedError()

    def visit_Data(self, data_node: ast.Data, *args, **kwargs):
        raise NotImplementedError()

    def visit_Param(self, param_node: ast.Param, *args, **kwargs):
        raise NotImplementedError()

    def visit_Normal(self, normal_node: ast.Normal, *args, **kwargs):
        self.expression_string += "jax.scipy.stats.norm.logpdf("
        normal_node.variate.accept(self, *args, **kwargs)
        self.expression_string += ", "
        normal_node.mean.accept(self, *args, **kwargs)
        self.expression_string += ", "
        normal_node.std.accept(self, *args, **kwargs)
        self.expression_string += ")"

    def visit_BernoulliLogit(self, bernoulli_node: ast.BernoulliLogit, *args, **kwargs):
        self.expression_string += "rat.math.bernoulli_logit("
        bernoulli_node.variate.accept(self, *args, **kwargs)
        self.expression_string += ", "
        bernoulli_node.logit_p.accept(self, *args, **kwargs)
        self.expression_string += ")"

    def visit_LogNormal(self, lognormal_node: ast.LogNormal, *args, **kwargs):
        self.expression_string += "rat.math.log_normal("
        lognormal_node.variate.accept(self, *args, **kwargs)
        self.expression_string += ", "
        lognormal_node.mean.accept(self, *args, **kwargs)
        self.expression_string += ", "

    def visit_Cauchy(self, cauchy_node: ast.Cauchy, *args, **kwargs):
        self.expression_string += "jax.scipy.stats.cauchy.logpdf("
        cauchy_node.variate.accept(self, *args, **kwargs)
        self.expression_string += ", "
        cauchy_node.location.accept(self, *args, **kwargs)
        self.expression_string += ", "
        cauchy_node.scale.accept(self, *args, **kwargs)
        self.expression_string += ")"

    def visit_Exponential(self, exponential_node: ast.Exponential, *args, **kwargs):
        self.expression_string += "jax.scipy.stats.expon.logpdf("
        exponential_node.variate.accept(self, *args, **kwargs)
        self.expression_string += ", loc=0, scale="
        exponential_node.scale.accept(self, *args, **kwargs)
        self.expression_string += ")"

    def visit_Diff(self, diff_node: ast.Diff, *args, **kwargs):
        diff_node.left.accept(self, *args, **kwargs)
        self.expression_string += " - "
        diff_node.right.accept(self, *args, **kwargs)

    def visit_Sum(self, sum_node: ast.Sum, *args, **kwargs):
        sum_node.left.accept(self, *args, **kwargs)
        self.expression_string += " + "
        sum_node.right.accept(self, *args, **kwargs)

    def visit_Mul(self, mul_node: ast.Mul, *args, **kwargs):
        mul_node.left.accept(self, *args, **kwargs)
        self.expression_string += " * "
        mul_node.right.accept(self, *args, **kwargs)

    def visit_Pow(self, pow_node: ast.Pow, *args, **kwargs):
        pow_node.base.accept(self, *args, **kwargs)
        self.expression_string += " ** "
        pow_node.exponent.accept(self, *args, **kwargs)

    def visit_Div(self, div_node: ast.Div, *args, **kwargs):
        div_node.left.accept(self, *args, **kwargs)
        self.expression_string += " / "
        div_node.right.accept(self, *args, **kwargs)

    def visit_Mod(self, mod_node: ast.Mod, *args, **kwargs):
        mod_node.left.accept(self, *args, **kwargs)
        self.expression_string += " % "
        mod_node.right.accept(self, *args, **kwargs)

    def visit_PrefixNegation(self, pneg_node: ast.PrefixNegation, *args, **kwargs):
        self.expression_string += "-"
        pneg_node.accept(self, *args, **kwargs)

    def visit_Sqrt(self, sqrt_node: ast.Sqrt, *args, **kwargs):
        self.expression_string += "jax.numpy.sqrt("
        sqrt_node.subexpr.accept(self, *args, **kwargs)
        self.expression_string += ")"

    def visit_Log(self, log_node: ast.Log, *args, **kwargs):
        self.expression_string += "jax.numpy.log("
        log_node.subexpr.accept(self, *args, **kwargs)
        self.expression_string += ")"

    def visit_Exp(self, exp_node: ast.Exp, *args, **kwargs):
        self.expression_string += "jax.numpy.exp("
        exp_node.subexpr.accept(self, *args, **kwargs)
        self.expression_string += ")"

    def visit_Abs(self, abs_node: ast.Abs, *args, **kwargs):
        self.expression_string += "jax.numpy.abs("
        abs_node.subexpr.accept(self, *args, **kwargs)
        self.expression_string += ")"

    def visit_Floor(self, floor_node: ast.Floor, *args, **kwargs):
        self.expression_string += "jax.numpy.floor("
        floor_node.subexpr.accept(self, *args, **kwargs)
        self.expression_string += ")"

    def visit_Ceil(self, ceil_node: ast.Ceil, *args, **kwargs):
        self.expression_string += "jax.numpy.ceil("
        ceil_node.subexpr.accept(self, *args, **kwargs)
        self.expression_string += ")"

    def visit_Round(self, round_node: ast.Round, *args, **kwargs):
        self.expression_string += "jax.numpy.round("
        round_node.subexpr.accept(self, *args, **kwargs)
        self.expression_string += ")"

    def visit_Sin(self, sin_node: ast.Sin, *args, **kwargs):
        self.expression_string += "jax.numpy.sin("
        sin_node.subexpr.accept(self, *args, **kwargs)
        self.expression_string += ")"

    def visit_Cos(self, cos_node: ast.Cos, *args, **kwargs):
        self.expression_string += "jax.numpy.cos("
        cos_node.subexpr.accept(self, *args, **kwargs)
        self.expression_string += ")"

    def visit_Tan(self, tan_node: ast.Tan, *args, **kwargs):
        self.expression_string += "jax.numpy.tan("
        tan_node.subexpr.accept(self, *args, **kwargs)
        self.expression_string += ")"

    def visit_Arcsin(self, arcsin_node: ast.Arcsin, *args, **kwargs):
        self.expression_string += "jax.numpy.arcsin("
        arcsin_node.subexpr.accept(self, *args, **kwargs)
        self.expression_string += ")"

    def visit_Arccos(self, arccos_node: ast.Arccos, *args, **kwargs):
        self.expression_string += "jax.numpy.arccos("
        arccos_node.subexpr.accept(self, *args, **kwargs)
        self.expression_string += ")"

    def visit_Arctan(self, arctan_node: ast.Arctan, *args, **kwargs):
        self.expression_string += "jax.numpy.arctan("
        arctan_node.subexpr.accept(self, *args, **kwargs)
        self.expression_string += ")"

    def visit_Logit(self, logit_node: ast.Logit, *args, **kwargs):
        self.expression_string += "jax.scipy.special.logit("
        logit_node.subexpr.accept(self, *args, **kwargs)
        self.expression_string += ")"

    def visit_InverseLogit(self, invlogit_node: ast.InverseLogit, *args, **kwargs):
        self.expression_string += "jax.scipy.special.expit("
        invlogit_node.subexpr.accept(self, *args, **kwargs)
        self.expression_string += ")"


