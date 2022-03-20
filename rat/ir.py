from codeop import Compile
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from typing import Tuple, Union, Type, List, Dict, TYPE_CHECKING, Set

if TYPE_CHECKING:
    from . import ast
import jax


from . import variables
from . import types


class BaseVisitor:
    def __init__(self):
        self.expression_string = ""

    def visit_IntegerConstant(self, integer_node: ast.IntegerConstant):
        self.expression_string += integer_node.value

    def visit_RealConstant(self, real_node: ast.RealConstant):
        self.expression_string += real_node.value

    def visit_Subscript(self, subscript_node: ast.Subscript):
        pass

    def visit_SubscriptColumn(self, column_node: ast.SubscriptColumn):
        pass

    def visit_Shift(self, shift_node: ast.Shift):
        pass

    def visit_Data(self, data_node: ast.Data):
        pass

    def visit_Param(self, param_node: ast.Param):
        pass

    def visit_Normal(self, normal_node: ast.Normal):
        self.expression_string += "jax.scipy.stats.norm.logpdf("
        normal_node.variate.accept(self)
        self.expression_string += ", "
        normal_node.mean.accept(self)
        self.expression_string += ", "
        normal_node.std.accept(self)
        self.expression_string += ")"

    def visit_BernoulliLogit(self, bernoulli_node: ast.BernoulliLogit):
        self.expression_string += "rat.math.bernoulli_logit("
        bernoulli_node.variate.accept(self)
        self.expression_string += ", "
        bernoulli_node.logit_p.accept(self)
        self.expression_string += ")"

    def visit_LogNormal(self, lognormal_node: ast.LogNormal):
        self.expression_string += "rat.math.log_normal("
        lognormal_node.variate.accept(self)
        self.expression_string += ", "
        lognormal_node.mean.accept(self)
        self.expression_string += ", "

    def visit_Cauchy(self, cauchy_node: ast.Cauchy):
        self.expression_string += "jax.scipy.stats.cauchy.logpdf("
        cauchy_node.variate.accept(self)
        self.expression_string += ", "
        cauchy_node.location.accept(self)
        self.expression_string += ", "
        cauchy_node.scale.accept(self)
        self.expression_string += ")"

    def visit_Exponential(self, exponential_node: ast.Exponential):
        pass

    def visit_Diff(self, diff_node: ast.Diff):
        diff_node.left.accept(self)
        self.expression_string += " - "
        diff_node.right.accept(self)
