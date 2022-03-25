from codeop import Compile
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from typing import Tuple, Union, Type, List, Dict, TYPE_CHECKING, Set

if TYPE_CHECKING:
    from . import compiler_rewrite

from . import ast


class IndentedString:
    def __init__(self, indent_level = 0):
        self.prefix = " " * indent_level
        self.string = self.prefix

    def __iadd__(self, other: str):
        self.string += other.replace("\n", f"\n{self.prefix}")
        return self

    def __str__(self):
        return self.string


class BaseVisitor:
    def __init__(self, symbol_table: "compiler_rewrite.SymbolTable", primary_variable=None, indent=0):
        self.expression_string = IndentedString(indent_level=indent)
        self.symbol_table = symbol_table
        self.primary_variable = primary_variable
        self.indent = indent

    def get_expression_string(self):
        return self.expression_string.string

    def visit_IntegerConstant(self, integer_node: ast.IntegerConstant, *args, **kwargs):
        self.expression_string += str(integer_node.value)

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
        self.expression_string += f"data['{data_node.name}']"

    def visit_Param(self, param_node: ast.Param, *args, **kwargs):
        raise NotImplementedError()

    def visit_Assignment(self, assignment_node: ast.Assignment, *args, **kwargs):
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
        lognormal_node.std.accept(self, *args, **kwargs)
        self.expression_string += ")"

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


class EvaluateDensityVisitor(BaseVisitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.primary_variable_name = self.primary_variable.name
        if self.primary_variable.subscript:
            self.primary_variable_subscript_names = tuple(
                [subscript_column.name for subscript_column in self.primary_variable.subscript.names]
            )
        else:
            self.primary_variable_subscript_names = tuple()

    def visit_Data(self, data_node: ast.Data, *args, **kwargs):
        self.expression_string += f"data['{data_node.name}']"

    def visit_Param(self, param_node: ast.Param, *args, **kwargs):
        self.expression_string += f"parameters['{param_node.name}']"
        if param_node.subscript:
            self.expression_string += "["
            param_node.subscript.accept(self, variable_name=param_node.name, *args, **kwargs)
            self.expression_string += "]"

            # delete if not needed
            if self.symbol_table.lookup(param_node.name).pad_needed:
                # if padded: remove the zero-pad
                self.expression_string += "[:-1]"

    def visit_Subscript(self, subscript_node: ast.Subscript, *args, **kwargs):
        variable_name = kwargs.pop("variable_name")  # passed from EvaluateDensityVisitor.visit_Param
        subscript_names = tuple([x.name for x in subscript_node.names])
        subscript_shifts = tuple([x.value for x in subscript_node.shifts])
        subscript_key = self.symbol_table.get_subscript_key(
            self.primary_variable_name, self.primary_variable_subscript_names, variable_name, subscript_names, subscript_shifts
        )
        self.expression_string += f"subscripts['{subscript_key}']"


class TransformedParametersVisitor(EvaluateDensityVisitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lhs_used_in_rhs = False
        self.lhs_key = ""
        self.at_rhs = False

    def visit_Subscript(self, subscript_node: ast.Subscript, *args, **kwargs):
        super().visit_Subscript(subscript_node, *args, **kwargs)
        if self.at_rhs:
            self.expression_string += "[index]"

    def visit_Param(self, param_node: ast.Param, *args, **kwargs):
        param_key = param_node.get_key()

        if param_node.subscript:
            if param_key == self.lhs_key:
                if self.at_rhs:
                    # scan function
                    subscript_node = param_node.subscript
                    subscript_names = tuple([x.name for x in subscript_node.names])
                    subscript_shifts = tuple([x.value for x in subscript_node.shifts])
                    subscript_key = self.symbol_table.get_subscript_key(
                        self.primary_variable_name, self.primary_variable_subscript_names, param_key,
                        subscript_names, subscript_shifts
                    )

                    shift = subscript_shifts[0]
                    self.expression_string += f"carry{shift}"
                else:
                    self.expression_string += f"parameters['{param_key}']"
                    # self.expression_string += "["
                    # param_node.subscript.accept(self, variable_name=param_node.name, *args, **kwargs)
                    # self.expression_string += "]"
            elif param_key != self.lhs_key:
                self.expression_string += f"parameters['{param_key}']"
                self.expression_string += "["
                param_node.subscript.accept(self, variable_name=param_node.name, *args, **kwargs)
                self.expression_string += "]"

        else:
            self.expression_string += f"parameters['{param_key}']"

    def visit_Assignment(self, assignment_node: ast.Assignment, *args, **kwargs):
        lhs = assignment_node.lhs
        self.lhs_key = lhs.get_key()
        rhs = assignment_node.rhs

        carry_size = 0

        for symbol in ast.search_tree(rhs, ast.Param):
            if symbol.get_key() == self.lhs_key:
                self.lhs_used_in_rhs = True

                shifts = symbol.subscript.shifts
                carry_size = max(carry_size, *[shift.value for shift in shifts])

        self.expression_string += f"# Assigned param: {self.lhs_key}\n"
        if self.lhs_used_in_rhs:
            lhs_record = self.symbol_table.lookup(self.lhs_key)
            lhs_size = lhs_record.base_df.shape[0]

            zero_tuple_string = f"({','.join(['0.0'] * carry_size)})"
            carry_strings = [f"carry{n}" for n in range(carry_size)]
            carry_tuple_string = f"({','.join(carry_strings)})"  # (carry0, carry1, ...)
            next_carry_tuple_string = f"({','.join(['next_value'] + carry_strings[:-1])})"
            scan_function_name = f"scan_function_{self.symbol_table.get_unique_number()}"

            self.expression_string += f"def {scan_function_name}(carry, index):\n"
            if self.lhs_key in self.symbol_table.first_in_group_indicator:
                self.expression_string += f"    carry = jax.lax.cond(first_in_group_indicators['{self.lhs_key}'], lambda : {zero_tuple_string}, lambda : carry)\n"
            self.expression_string += f"    {carry_tuple_string} = carry\n"
            self.expression_string += f"    next_value = "
            self.at_rhs = True
            rhs.accept(self, *args, **kwargs)
            self.at_rhs = False
            self.expression_string += "\n"
            self.expression_string += f"    return {next_carry_tuple_string}, next_value\n\n"

            self.expression_string += "_, "
            lhs.accept(self, *args, **kwargs)
            self.expression_string += f" = jax.lax.scan({scan_function_name}, {zero_tuple_string}, jax.numpy.arange({lhs_size}))"
            self.expression_string += "\n"

        else:
            lhs.accept(self, *args, **kwargs)
            self.expression_string += " = "
            rhs.accept(self, *args, **kwargs)
            self.expression_string += "\n"
