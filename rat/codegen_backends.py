from typing import Tuple

from . import ast
from .exceptions import CompileError
from .variable_table import VariableTable, VariableRecord


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
    expression_string : IndentedString
    indent : int

    def __init__(self, indent : int = 0):
        self.expression_string = IndentedString(indent_level=indent)
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
            case ast.LessThan():
                self.generate(ast_node.left)
                self.expression_string += " < "
                self.generate(ast_node.right)
            case ast.GreaterThan():
                self.generate(ast_node.left)
                self.expression_string += " > "
                self.generate(ast_node.right)
            case ast.LessThanOrEq():
                self.generate(ast_node.left)
                self.expression_string += " <= "
                self.generate(ast_node.right)
            case ast.GreaterThanOrEq():
                self.generate(ast_node.left)
                self.expression_string += " >= "
                self.generate(ast_node.right)
            case ast.EqualTo():
                self.generate(ast_node.left)
                self.expression_string += " == "
                self.generate(ast_node.right)
            case ast.NotEqualTo():
                self.generate(ast_node.left)
                self.expression_string += " != "
                self.generate(ast_node.right)

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
                self.expression_string += "jax.numpy.floor("
                self.generate(ast_node.subexpr)
                self.expression_string += ")"
            case ast.Ceil():
                self.expression_string += "jax.numpy.ceil("
                self.generate(ast_node.subexpr)
                self.expression_string += ")"
            case ast.Real():
                self.expression_string += "jax.numpy.array("
                self.generate(ast_node.subexpr)
                self.expression_string += ", dtype = 'float')"
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
    variable_name: str = None  # This is an internal attribute that's used to pass variable name when generating subscripts
    variable_table : VariableTable
    primary_symbol : ast.PrimeableExpr
    primary_variable : VariableRecord

    def __init__(self, variable_table : VariableTable, primary_symbol : ast.PrimeableExpr = None, indent : int = 0):
        super().__init__(indent)
        self.variable_table = variable_table
        self.primary_symbol = primary_symbol
        self.primary_variable = self.variable_table[primary_symbol.name]

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
            case ast.Data():
                if ast_node.name != self.primary_variable.name and ast_node.name in self.primary_variable.subscripts:
                    subscript_data_name = self.primary_variable.request_subscript_as_data(ast_node.name)
                    self.expression_string += f"data['{subscript_data_name}']"
                else:
                    self.expression_string += f"data['{ast_node.name}']"
                    if ast_node.name != self.primary_variable.name and ast_node.subscript:
                        self.expression_string += "["
                        self.variable_name = ast_node.name
                        self.generate(ast_node.subscript)
                        self.variable_name = None
                        self.expression_string += "]"
            case ast.Subscript():
                if not self.variable_name:
                    raise Exception("Internal compiler error -- Variable name must be passed for subscript codegen!")
                subscript_names = tuple(column.name for column in ast_node.names)
                subscript_shifts = tuple(x.value for x in ast_node.shifts)
                subscript_key = self.variable_table.get_subscript_key(
                    self.primary_variable.name, self.variable_name, subscript_names, subscript_shifts
                )
                self.expression_string += f"subscripts['{subscript_key}']"

            case ast.IfElse():
                primary = self.primary_variable.name
                self.expression_string += "rat.math.lax_select_scalar("
                self.generate(ast_node.condition)
                self.expression_string += ", "
                self.generate(ast_node.true_expr)
                self.expression_string += ", "
                self.generate(ast_node.false_expr)
                self.expression_string += ")"

            case _:
                super().generate(ast_node)


class TransformedParametersCodeGenerator(EvaluateDensityCodeGenerator):
    lhs_used_in_rhs : bool
    at_rhs_of_scan : bool
    def __init__(self, variable_table : VariableTable, primary_symbol : ast.PrimeableExpr = None, indent : int = 0):
        super().__init__(variable_table, primary_symbol, indent)
        self.lhs_used_in_rhs = False
        self.lhs_key = ""
        self.at_rhs_of_scan = False
        # at_rhs_of_scan is used to indicate whether we are generating code for the RHS of a jax.lax.scan().
        # scan() differ fron normal array operations because we're working elementwise on the parameter array.
        # this means we can't just do data["a"] + parameter["b"], but instead need to index them one by one, depending
        # on the LHS parameter's index. We do that by appending [index] to RHS variables, so that we're indexing an
        # array. (ex. parameters["b"][1], where index = 1)

    def generate(self, ast_node: ast.Expr):
        match ast_node:
            case ast.Subscript():
                super().generate(ast_node)
                if self.at_rhs_of_scan:
                    # This means parameters need to be treated elementwise
                    self.expression_string += "[index]"
            case ast.Data():
                super().generate(ast_node)
                if self.at_rhs_of_scan:
                    # This means parameters need to be treated elementwise
                    self.expression_string += "[index]"
            case ast.Param():
                param_key = ast_node.get_key()

                if ast_node.subscript:
                    if param_key == self.lhs_key:
                        if self.at_rhs_of_scan:
                            # scan function
                            subscript = ast_node.subscript

                            subscript_names = tuple(column.name for column in subscript.names)
                            subscript_shifts = tuple(x.value for x in subscript.shifts)
                            self.variable_table.get_subscript_key(param_key, param_key, subscript_names, subscript_shifts)

                            carry_shift = next(shift for shift in subscript_shifts if shift != 0)
                            self.expression_string += f"carry{carry_shift}"
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
                    lhs_record = self.variable_table[self.lhs_key]
                    lhs_size = lhs_record.base_df.shape[0]

                    zero_tuple_string = f"({','.join(['0.0'] * carry_size)})"
                    carry_strings = [f"carry{n}" for n in range(1, carry_size + 1)]
                    carry_tuple_string = f"({','.join(carry_strings)})"  # (carry0, carry1, ...)
                    next_carry_tuple_string = f"({','.join(['next_value'] + carry_strings[:-1])})"
                    scan_function_name = f"scan_function_{self.variable_table.get_unique_number()}"

                    self.expression_string += f"def {scan_function_name}(carry, index):\n"
                    if need_first_in_group_indicator:
                        self.expression_string += f"    carry = jax.lax.cond(first_in_group_indicators['{self.lhs_key}'][index], lambda : {zero_tuple_string}, lambda : carry)\n"
                    self.expression_string += f"    {carry_tuple_string} = carry\n"
                    self.expression_string += f"    next_value = "
                    self.at_rhs_of_scan = True
                    self.generate(rhs)
                    self.at_rhs_of_scan = False
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

class DiscoverVariablesCodeGenerator():
    name_arguments : bool
    expression_string : str

    def __init__(self):
        self.name_arguments = False
        self.expression_string = ""

    def generate(self, ast_node: ast.Expr):
        match ast_node:
            case ast.Param():
                self.expression_string += ast_node.name
                self.expression_string += "("
                if ast_node.subscript:
                    self.generate(ast_node.subscript)
                self.expression_string += ")"
            case ast.Data():
                if ast_node.subscript:
                    # This is a join
                    self.expression_string += ast_node.name
                    self.expression_string += f"('{ast_node.name}', "
                    self.name_arguments = True
                    self.generate(ast_node.subscript)
                    self.name_arguments = False
                    self.expression_string += ")"
                else:
                    # With no subscript the variable comes from primary dataframe
                    self.expression_string += ast_node.name
            case ast.Subscript():
                subscript_names = tuple(column.name for column in ast_node.names)
                subscript_shifts = tuple(x.value for x in ast_node.shifts)
                arguments = []
                for name, shift in zip(subscript_names, subscript_shifts):
                    argument = f"{name} = {name}" if self.name_arguments else name
                    if shift != 0:
                        argument += f" - {shift}"
                    arguments.append(argument)
                self.expression_string += ",".join(arguments)

            case ast.IfElse():
                self.expression_string += "(("
                self.generate(ast_node.true_expr)
                self.expression_string += ") if ("
                self.generate(ast_node.condition)
                self.expression_string += ") else ("
                self.generate(ast_node.false_expr)
                self.expression_string += "))"

            case ast.Assignment():
                self.expression_string += "("
                self.generate(ast_node.lhs)
                self.expression_string += ","
                self.generate(ast_node.rhs)
                self.expression_string += ")"

            case ast.IntegerConstant():
                self.expression_string += str(ast_node.value)
            case ast.RealConstant():
                if ast_node.value == float("inf"):
                    self.expression_string += "float('inf')"
                elif ast_node.value == float("-inf"):
                    self.expression_string += "float('-inf')"
                else:
                    self.expression_string += str(ast_node.value)
            case ast.Normal():
                self.expression_string += "("
                self.generate(ast_node.variate)
                self.expression_string += ", "
                self.generate(ast_node.mean)
                self.expression_string += ", "
                self.generate(ast_node.std)
                self.expression_string += ")"
            case ast.BernoulliLogit():
                self.expression_string += "("
                self.generate(ast_node.variate)
                self.expression_string += ", "
                self.generate(ast_node.logit_p)
                self.expression_string += ")"
            case ast.LogNormal():
                self.expression_string += "("
                self.generate(ast_node.variate)
                self.expression_string += ", "
                self.generate(ast_node.mean)
                self.expression_string += ", "
                self.generate(ast_node.std)
                self.expression_string += ")"
            case ast.Cauchy():
                self.expression_string += "("
                self.generate(ast_node.variate)
                self.expression_string += ", "
                self.generate(ast_node.location)
                self.expression_string += ", "
                self.generate(ast_node.scale)
                self.expression_string += ")"
            case ast.Exponential():
                self.expression_string += "("
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
            case ast.LessThan():
                self.generate(ast_node.left)
                self.expression_string += " < "
                self.generate(ast_node.right)
            case ast.GreaterThan():
                self.generate(ast_node.left)
                self.expression_string += " > "
                self.generate(ast_node.right)
            case ast.LessThanOrEq():
                self.generate(ast_node.left)
                self.expression_string += " <= "
                self.generate(ast_node.right)
            case ast.GreaterThanOrEq():
                self.generate(ast_node.left)
                self.expression_string += " >= "
                self.generate(ast_node.right)
            case ast.EqualTo():
                self.generate(ast_node.left)
                self.expression_string += " == "
                self.generate(ast_node.right)
            case ast.NotEqualTo():
                self.generate(ast_node.left)
                self.expression_string += " != "
                self.generate(ast_node.right)

            case (
                ast.Sqrt()
                | ast.Log()
                | ast.Exp()
                | ast.Floor()
                | ast.Ceil()
                | ast.Real()
                | ast.Round()
                | ast.Sin()
                | ast.Cos()
                | ast.Tan()
                | ast.Arcsin()
                | ast.Arccos()
                | ast.Arctan()
                | ast.Logit()
                | ast.InverseLogit()
            ):
                self.expression_string += "("
                self.generate(ast_node.subexpr)
                self.expression_string += ")"
            case ast.Abs():
                self.expression_string += "abs("
                self.generate(ast_node.subexpr)
                self.expression_string += ")"
            

            case _:
                raise NotImplementedError()
