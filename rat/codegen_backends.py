from typing import Tuple

from . import ast
from .exceptions import CompileError
from .symbol_table import SymbolTable


class IndentedString:
    def __init__(self, indent_level=0):
        self.prefix = " " * indent_level
        self.string = self.prefix

    def __iadd__(self, other: str):
        self.string += other.replace("\n", f"\n{self.prefix}")
        return self

    def __str__(self):
        return self.string


class BaseCodeGenerator:
    def __init__(self, symbol_table: SymbolTable, primary_variable=None, indent=0):
        self.expression_string = IndentedString(indent_level=indent)
        self.symbol_table = symbol_table
        self.primary_variable = primary_variable
        self.indent = indent

    def get_expression_string(self):
        return self.expression_string.string

    def generate(self, ast_node: ast.Expr):
        match ast_node:
            case ast.IntegerConstant():
                self.expression_string += str(ast_node.value)
            case ast.RealConstant():
                if ast_node.value == float("inf"):
                    self.expression_string += "float('inf')"
                elif ast_node.value == float("-inf"):
                    self.expression_string += "float('-inf')"
                else:
                    self.expression_string += str(ast_node.value)
            case ast.Data():
                self.expression_string += f"data['{ast_node.name}']"
            case ast.Normal():
                self.expression_string += "jax.scipy.stats.norm.logpdf("
                self.generate(ast_node.variate)
                self.expression_string += ", "
                self.generate(ast_node.mean)
                self.expression_string += ", "
                self.generate(ast_node.std)
                self.expression_string += ")"
            case ast.BernoulliLogit():
                self.expression_string += "rat.math.bernoulli_logit("
                self.generate(ast_node.variate)
                self.expression_string += ", "
                self.generate(ast_node.logit_p)
                self.expression_string += ")"
            case ast.LogNormal():
                self.expression_string += "rat.math.log_normal("
                self.generate(ast_node.variate)
                self.expression_string += ", "
                self.generate(ast_node.mean)
                self.expression_string += ", "
                self.generate(ast_node.std)
                self.expression_string += ")"
            case ast.Cauchy():
                self.expression_string += "jax.scipy.stats.cauchy.logpdf("
                self.generate(ast_node.variate)
                self.expression_string += ", "
                self.generate(ast_node.location)
                self.expression_string += ", "
                self.generate(ast_node.scale)
                self.expression_string += ")"
            case ast.Exponential():
                self.expression_string += "jax.scipy.stats.expon.logpdf("
                self.generate(ast_node.variate)
                self.expression_string += ", loc=0, scale="
                self.generate(ast_node.scale)
                self.expression_string += ")"

            case ast.Diff():
                self.generate(ast_node.left)
                self.expression_string += " - "
                self.generate(ast_node.right)
            case ast.Sum():
                self.generate(ast_node.left)
                self.expression_string += " + "
                self.generate(ast_node.right)
            case ast.Mul():
                self.generate(ast_node.left)
                self.expression_string += " * "
                self.generate(ast_node.right)
            case ast.Pow():
                self.generate(ast_node.left)
                self.expression_string += " ** "
                self.generate(ast_node.right)
            case ast.Div():
                self.generate(ast_node.left)
                self.expression_string += " / "
                self.generate(ast_node.right)
            case ast.Mod():
                self.generate(ast_node.left)
                self.expression_string += " % "
                self.generate(ast_node.right)
            case ast.PrefixNegation():
                self.expression_string += "-"
                self.generate(ast_node.subexpr)

            case ast.Sqrt():
                self.expression_string += "jax.numpy.sqrt("
                self.generate(ast.subexpr)
                self.expression_string += ")"
            case ast.Log():
                self.expression_string += "jax.numpy.log("
                self.generate(ast_node.subexpr)
                self.expression_string += ")"
            case ast.Exp():
                self.expression_string += "jax.numpy.exp("
                self.generate(ast_node.subexpr)
                self.expression_string += ")"
            case ast.Abs():
                self.expression_string += "jax.numpy.abs("
                self.generate(ast_node.subexpr)
                self.expression_string += ")"
            case ast.Floor():
                self.expression_string += "jx.numpy.floor("
                self.generate(ast_node.subexpr)
                self.expression_string += ")"
            case ast.Ceil():
                self.expression_string += "jax.numpy.ceil("
                self.generate(ast_node.subexpr)
                self.expression_string += ")"
            case ast.Round():
                self.expression_string += "jax.numpy.round("
                self.generate(ast_node.subexpr)
                self.expression_string += ")"
            case ast.Sin():
                self.expression_string += "jax.numpy.sin("
                self.generate(ast_node.subexpr)
                self.expression_string += ")"
            case ast.Cos():
                self.expression_string += "jax.numpy.cos("
                self.generate(ast_node.subexpr)
                self.expression_string += ")"
            case ast.Tan():
                self.expression_string += "jax.numpy.tan("
                self.generate(ast_node.subexpr)
                self.expression_string += ")"
            case ast.Arcsin():
                self.expression_string += "jax.numpy.arcsin("
                self.generate(ast_node.subexpr)
                self.expression_string += ")"
            case ast.Arccos():
                self.expression_string += "jax.numpy.arccos("
                self.generate(ast_node.subexpr)
                self.expression_string += ")"
            case ast.Arctan():
                self.expression_string += "jax.numpy.arctan("
                self.generate(ast_node.subexpr)
                self.expression_string += ")"
            case ast.Logit():
                self.expression_string += "jax.scipy.special.logit("
                self.generate(ast_node.subexpr)
                self.expression_string += ")"
            case ast.InverseLogit():
                self.expression_string += "jax.scipy.special.expit("
                self.generate(ast_node.subexpr)
                self.expression_string += ")"
            case _:
                raise NotImplementedError()


class EvaluateDensityCodeGenerator(BaseCodeGenerator):
    variable_name: str = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.primary_variable_name = self.primary_variable.name
        if self.primary_variable.subscript:
            self.primary_variable_subscript_names = tuple(
                [subscript_column.name for subscript_column in self.primary_variable.subscript.names]
            )
        else:
            self.primary_variable_subscript_names = tuple()

    def generate(self, ast_node: ast.Expr):
        match ast_node:
            case ast.Param():
                self.expression_string += f"parameters['{ast_node.name}']"
                if ast_node.subscript:
                    self.expression_string += "["
                    self.variable_name = ast_node.name
                    self.generate(ast_node.subscript)
                    self.variable_name = None
                    self.expression_string += "]"
            case ast.Subscript():
                if not self.variable_name:
                    raise Exception("Internal compiler error -- Variable name must be passed for subscript codegen!")
                subscript_shifts = tuple([x.value for x in ast_node.shifts])
                subscript_key = self.symbol_table.get_subscript_key(
                    self.primary_variable_name, self.variable_name, subscript_shifts
                )
                self.expression_string += f"subscripts['{subscript_key}']"
            case _:
                super().generate(ast_node)


class TransformedParametersCodeGenerator(EvaluateDensityCodeGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lhs_used_in_rhs = False
        self.lhs_key = ""
        self.at_rhs = False

    def generate(self, ast_node: ast.Expr):
        match ast_node:
            case ast.Subscript():
                super().generate(ast_node)
                if self.at_rhs:
                    self.expression_string += "[index]"
            case ast.Param():
                param_key = ast_node.get_key()

                if ast_node.subscript:
                    if param_key == self.lhs_key:
                        if self.at_rhs:
                            # scan function
                            subscript = ast_node.subscript                            
                            shift = next(x.value for x in subscript.shifts if x.value != 0)
                            self.expression_string += f"carry{shift}"
                        else:
                            self.expression_string += f"parameters['{param_key}']"
                    elif param_key != self.lhs_key:
                        self.expression_string += f"parameters['{param_key}']"
                        self.expression_string += "["
                        self.variable_name = param_key
                        self.generate(ast_node.subscript)
                        self.variable_name = None
                        self.expression_string += "]"
                else:
                    self.expression_string += f"parameters['{param_key}']"
            case ast.Assignment():
                lhs = ast_node.lhs
                self.lhs_key = lhs.get_key()
                rhs = ast_node.rhs

                carry_size = 0

                need_first_in_group_indicator = False
                for symbol in ast.search_tree(rhs, ast.Param):
                    if symbol.get_key() == self.lhs_key:
                        self.lhs_used_in_rhs = True

                        shifts = symbol.subscript.shifts

                        if len(shifts) > 1:
                            need_first_in_group_indicator = True
                        carry_size = max(carry_size, *[shift.value for shift in shifts])

                self.expression_string += f"# Assigned param: {self.lhs_key}\n"
                if self.lhs_used_in_rhs:
                    lhs_record = self.symbol_table.lookup(self.lhs_key)
                    lhs_size = lhs_record.base_df.shape[0]

                    zero_tuple_string = f"({','.join(['0.0'] * carry_size)})"
                    carry_strings = [f"carry{n}" for n in range(1, carry_size + 1)]
                    carry_tuple_string = f"({','.join(carry_strings)})"  # (carry0, carry1, ...)
                    next_carry_tuple_string = f"({','.join(['next_value'] + carry_strings[:-1])})"
                    scan_function_name = f"scan_function_{self.symbol_table.get_unique_number()}"

                    self.expression_string += f"def {scan_function_name}(carry, index):\n"
                    if need_first_in_group_indicator:
                        self.expression_string += f"    carry = jax.lax.cond(first_in_group_indicators['{self.lhs_key}'][index], lambda : {zero_tuple_string}, lambda : carry)\n"
                    self.expression_string += f"    {carry_tuple_string} = carry\n"
                    self.expression_string += f"    next_value = "
                    self.at_rhs = True
                    self.generate(rhs)
                    self.at_rhs = False
                    self.expression_string += "\n"
                    self.expression_string += f"    return {next_carry_tuple_string}, next_value\n\n"

                    self.expression_string += "_, "
                    self.generate(lhs)
                    self.expression_string += f" = jax.lax.scan({scan_function_name}, {zero_tuple_string}, jax.numpy.arange({lhs_size}))"
                    self.expression_string += "\n"

                else:
                    self.generate(lhs)
                    self.expression_string += " = "
                    self.generate(rhs)
                    self.expression_string += "\n"

            case _:
                super().generate(ast_node)
