from typing import *
from .scanner import (
    Token,
    Identifier,
    Operator,
    RealLiteral,
    IntLiteral,
    Special,
    NullToken,
)
from .ops import *
import warnings
import jax.numpy

# https://mc-stan.org/docs/2_18/reference-manual/bnf-grammars.html
# https://mc-stan.org/docs/2_28/reference-manual/arithmetic-expressions.html


# define group parsing rules for operators
class PrefixOps:
    """
    A utility class that's used to identify and build prefix-operation expressions.
    """

    ops = ["!", "-"]
    precedence = {"!": 50, "-": 50}

    @staticmethod
    def check(tok: Token):
        if isinstance(tok, Operator) and tok.value in PrefixOps.ops:
            return True
        return False

    @staticmethod
    def generate(expr: Expr, operator: Operator):
        if operator.value == "-":
            return PrefixNegation(subexpr=expr, line_index=operator.line_index, column_index=operator.column_index)


class InfixOps:
    """
    A utility class that's used to identify and build binary operation expressions.
    Currently supported operations are:
    `ops.Sum`, `ops.Diff`, `ops.Mul`, `ops.Pow`, `ops.Mod`, `ops.Div`
    """

    ops = ["+", "-", "*", "^", "/", "%", "<", ">"]
    precedence = {"+": 10, "-": 10, "*": 30, "/": 30, "^": 40, "%": 30, "<": 5, ">": 5}

    @staticmethod
    def check(tok: Type[Token]):
        if isinstance(tok, Operator) and tok.value in InfixOps.ops:
            return True
        return False

    @staticmethod
    def generate(left: Expr, right: Expr, operator: Type[Token]):
        if operator.value == "+":
            return Sum(left=left, right=right, line_index=operator.line_index, column_index=operator.column_index)
        elif operator.value == "-":
            return Diff(left=left, right=right, line_index=operator.line_index, column_index=operator.column_index)
        elif operator.value == "*":
            return Mul(left=left, right=right, line_index=operator.line_index, column_index=operator.column_index)
        elif operator.value == "/":
            return Div(left=left, right=right, line_index=operator.line_index, column_index=operator.column_index)
        elif operator.value == "^":
            return Pow(base=left, exponent=right, line_index=operator.line_index, column_index=operator.column_index)
        elif operator.value == "%":
            return Mod(left=left, right=right, line_index=operator.line_index, column_index=operator.column_index)
        else:
            raise Exception(f"InfixOps: Unknown operator type {operator.value}")


# group parsing rules for statements


class AssignmentOps:
    """
    A utility class that's used to identify and build assignments in statements.
    Currently supports the following assignment types:
    `ops.Assignment`
    """

    ops = ["="]

    @staticmethod
    def check(tok: Type[Token]):
        if isinstance(tok, Operator) and tok.value in AssignmentOps.ops:
            return True
        return False

    @staticmethod
    def generate(lhs: Expr, rhs: Expr, operator: Operator):
        # check() has been ran for operator
        if operator.value == "=":
            return Assignment(lhs=lhs, rhs=rhs, line_index=operator.line_index, column_index=operator.column_index)


class UnaryFunctions:
    """
    A utility class that's used to identify and build unary functions.
    """

    names = ["exp", "log", "abs", "floor", "ceil", "round", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "logit", "inverse_logit"]
    precedence = {
        "exp": 100,
        "log": 100,
        "abs": 100,
        "floor": 100,
        "ceil": 100,
        "round": 100,
        "sin": 100,
        "cos": 100,
        "tan": 100,
        "arcsin": 100,
        "arccos": 100,
        "arctan": 100,
        "logit": 100,
        "inverse_logit": 100,
    }

    @staticmethod
    def check(tok: Type[Token]):
        if isinstance(tok, Identifier) and tok.value in UnaryFunctions.names:
            return True
        return False

    @staticmethod
    def generate(subexpr: Expr, func_type: Identifier):
        if func_type.value == "exp":
            return Exp(subexpr=subexpr, line_index=func_type.line_index, column_index=func_type.column_index)
        elif func_type.value == "log":
            return Log(subexpr=subexpr, line_index=func_type.line_index, column_index=func_type.column_index)
        elif func_type.value == "abs":
            return Abs(subexpr=subexpr, line_index=func_type.line_index, column_index=func_type.column_index)
        elif func_type.value == "floor":
            return Floor(subexpr=subexpr, line_index=func_type.line_index, column_index=func_type.column_index)
        elif func_type.value == "ceil":
            return Ceil(subexpr=subexpr, line_index=func_type.line_index, column_index=func_type.column_index)
        elif func_type.value == "round":
            return Round(subexpr=subexpr, line_index=func_type.line_index, column_index=func_type.column_index)
        elif func_type.value == "sin":
            return Sin(subexpr=subexpr, line_index=func_type.line_index, column_index=func_type.column_index)
        elif func_type.value == "cos":
            return Cos(subexpr=subexpr, line_index=func_type.line_index, column_index=func_type.column_index)
        elif func_type.value == "tan":
            return Tan(subexpr=subexpr, line_index=func_type.line_index, column_index=func_type.column_index)
        elif func_type.value == "arcsin":
            return Arcsin(subexpr=subexpr, line_index=func_type.line_index, column_index=func_type.column_index)
        elif func_type.value == "arccos":
            return Arccos(subexpr=subexpr, line_index=func_type.line_index, column_index=func_type.column_index)
        elif func_type.value == "arctan":
            return Arctan(subexpr=subexpr, line_index=func_type.line_index, column_index=func_type.column_index)
        elif func_type.value == "logit":
            return Logit(subexpr=subexpr, line_index=func_type.line_index, column_index=func_type.column_index)
        elif func_type.value == "inverse_logit":
            return InverseLogit(subexpr=subexpr, line_index=func_type.line_index, column_index=func_type.column_index)


class BinaryFunctions:
    """
    A utility class that's used to identify and build binary functions.
    """

    names = ["shift"]
    precedence = {
        "shift": 100,
    }

    @staticmethod
    def check(tok: Type[Token]):
        if isinstance(tok, Identifier) and tok.value in BinaryFunctions.names:
            return True
        return False

    @staticmethod
    def generate(arg1: Expr, arg2: Expr, func_type: Identifier):
        if func_type.value == "shift":
            return Shift(subscript=arg1, shift_expr=arg2, line_index=func_type.line_index, column_index=func_type.column_index)


class Distributions:
    """
    A utility class that's used to identify and build distributions.
    Currently supported distributions are:
    `ops.Normal`, `ops.BernoulliLogit`, `ops.LogNormal`, `ops.Cauchy`, `ops.Exponential`
    """

    names = ["normal", "bernoulli_logit", "log_normal", "cauchy", "exponential"]

    @staticmethod
    def check(tok: Type[Token]):
        if isinstance(tok, Identifier) and tok.value in Distributions.names:
            return True
        return False

    @staticmethod
    def generate(lhs: Expr, expressions: List[Expr], dist_type: Identifier):
        if dist_type.value == "normal":
            if len(expressions) != 2:
                raise Exception(f"normal distribution needs 2 parameters, but got {len(expressions)}!")
            return Normal(variate=lhs, mean=expressions[0], std=expressions[1], line_index=dist_type.line_index, column_index=dist_type.column_index)
        elif dist_type.value == "bernoulli_logit":
            if len(expressions) != 1:
                raise Exception(f"bernoulli_logit distribution needs 1 parameter, but got {len(expressions)}!")
            return BernoulliLogit(variate=lhs, logit_p=expressions[0], line_index=dist_type.line_index, column_index=dist_type.column_index)
        elif dist_type.value == "log_normal":
            if len(expressions) != 2:
                raise Exception(f"log_normal distribution needs 2 parameters, but got {len(expressions)}!")
            return LogNormal(variate=lhs, mean=expressions[0], std=expressions[1], line_index=dist_type.line_index, column_index=dist_type.column_index)
        elif dist_type.value == "cauchy":
            if len(expressions) != 2:
                raise Exception(f"cauchy distribution needs 2 parameters, but got {len(expressions)}!")
            return Cauchy(variate=lhs, location=expressions[0], scale=expressions[1], line_index=dist_type.line_index, column_index=dist_type.column_index)
        elif dist_type.value == "exponential":
            if len(expressions) != 1:
                raise Exception(f"exponential distribution needs 1 parameter, but got {len(expressions)}!")
            return Exponential(variate=lhs, scale=expressions[0], line_index=dist_type.line_index, column_index=dist_type.column_index)


class ParseError(Exception):
    def __init__(self, message, code_string: str, line_num: int, column_num: int):
        code_string = code_string.split("\n")[line_num]
        exception_message = f"An error occured while parsing the following line({line_num}:{column_num}):\n{code_string}\n{' ' * column_num + '^'}\n{message}"
        super().__init__(exception_message)


class Parser:
    """
    The parser for rat is a modified Pratt parser.
    Since rat programs are defined within the context of data, the parser needs to know
    the column names of the data.
    """

    def __init__(self, tokens: List[Token], data_names: List[str], model_string: str = ""):
        """
        Initialize the parser
        :param tokens: A list of `scanner.Token`. This should be the output format of `scanner.scanner`
        :param data_names: A list of data column names
        :param model_string: Optional. The original model code string. If supplied, used to generate detailed errors.
        """
        self.out_tree = []
        self.tokens = tokens
        self.data_names = data_names
        self.model_string = model_string

    def check_bracket_stack(self):
        bracket_stack: List[Token] = []
        for tok in self.tokens:
            if isinstance(tok, Special) and tok.value in ("(", "{", "["):
                bracket_stack.append(tok)
            elif isinstance(tok, Special) and tok.value in (")", "}", "]"):
                if len(bracket_stack) == 0:
                    raise ParseError("Found unmatching brackets!!", self.model_string, tok.line_index, tok.column_index)
                top = bracket_stack.pop()
                if (
                    (top.value == "(" and tok.value == ")")
                    or (top.value == "{" and tok.value == "}")
                    or (top.value == "[" and tok.value == "]")
                ):
                    pass
                else:
                    raise ParseError("Found unmatching brackets!!", self.model_string, tok.line_index, tok.column_index)

        if len(bracket_stack) > 0:
            tok = bracket_stack[0]
            raise ParseError("Found unmatching brackets!!", self.model_string, tok.line_index, tok.column_index)

    def peek(self, k=0) -> Type[Token]:
        """
        k-token lookahead. Returns `scanner.NullToken` if there are no tokens in the token stack.
        """
        if k >= len(self.tokens):
            return NullToken()
        return self.tokens[k]

    def remove(self, index=0):
        self.tokens.pop(index)

    def expect_token(
        self,
        token_types: Union[Type[Token], List[Type[Token]]],
        token_value=None,
        remove=False,
        lookahead=0,
    ):
        """
        Checks if the next token in the token stack is of designated type and value. If not, raise an Exception.
        :param token_types: A list of `scanner.Token` types or a single `scanner.Token` type that's allowed.
        :param token_value: A single or a list of allowed token value strings
        :param remove: Boolean, whether to remove the token after checking or not. Defaults to False
        :param lookahead: lookahead. Defaults to 0(immediate token)
        :return: None
        """
        next_token = self.peek(lookahead)
        if not token_value:
            token_value = [next_token.value]

        if isinstance(token_value, str):
            token_value = [token_value]

        if not isinstance(token_types, tuple):
            token_types = (token_types,)

        for token_type in token_types:
            if isinstance(next_token, token_type) and next_token.value in token_value:
                if remove:
                    self.remove()
                return True

        raise ParseError(
            f"Expected token type(s) {[x.__name__ for x in token_types]} with value in {token_value}, but received {next_token.__class__.__name__} with value '{next_token.value}'!",
            self.model_string,
            next_token.line_index,
            next_token.column_index,
        )

    def expressions(self, entry_token_value) -> List[Expr]:
        """
        expressions are used to evaluate repeated, comma-separated expressions in the form "expr, expr, expr"
        It's primarily used to evaluate subscripts or function arguments. In the case it's evaluating subscripts, it
        will also return the shift amounts of each subscript.
        :param entry_token_value: A single character which denotes the boundary token that starts the expression
        sequence. For example, "myFunc(expr1, expr2, expr3)" would mean the 3-expression sequence is present between the
        parantheses. So the entry token would be "(" and exit token ")".
        For subscripts, it would be something like "my_variable[sub_1, shift(sub_2, 1)]. That would mean entry token
        "[" and exit token "]".
        :return: list of expressions
        """
        if entry_token_value == "[":
            exit_value = "]"
        elif entry_token_value == "(":
            exit_value = ")"
        else:
            raise Exception(f"expressions() received invalid entry token value with value {entry_token_value}, but expected '[' or ']'")
        expressions = []
        while True:
            token = self.peek()
            if isinstance(token, Special):
                self.expect_token(Special, (exit_value, ","))
                if token.value == exit_value:
                    break
                elif token.value == ",":
                    self.remove()  # character ,
                    continue
            # elif isinstance(token, Identifier) and token.value == "shift":
            #     # parse shift(subscript, integer)
            #     self.remove()  # identifier "shift"
            #     self.expect_token(Special, "(")
            #     self.remove()  # character )
            #     self.expect_token(Identifier)  # subscript name
            #     subscript_name = self.peek()
            #     if subscript_name.value not in self.data_names:
            #         raise ParseError(
            #             "Subscript specified with shift() must be in data columns.",
            #             self.model_string,
            #             subscript_name.line_index,
            #             subscript_name.column_index,
            #         )
            #     expression = Data(subscript_name.value, line_index=subscript_name.line_index, column_index=subscript_name.column_index)
            #     self.remove()  # subscript name
            #     self.expect_token(Special, ",")
            #     self.remove()  # character ,
            #     shift_amount = self.expression()
            #     shift_amount = int(eval(shift_amount.code()))
            #     self.expect_token(Special, ")")
            #     self.remove()  # character )
            else:
                expression = self.expression()
                expressions.append(expression)

        return expressions

    def parse_nud(self, is_lhs=False):
        token = self.peek()
        if isinstance(token, RealLiteral):  # if just a real number, return it
            exp = RealConstant(value=float(token.value))
            self.remove()  # real
            return exp
        elif isinstance(token, IntLiteral):  # if just an integer, return it
            exp = IntegerConstant(value=int(token.value))
            self.remove()  # integer
            return exp

        elif PrefixOps.check(token):  # prefixOp expression
            self.expect_token(Operator, PrefixOps.ops)  # operator
            self.remove()

            next_expression = self.expression(PrefixOps.precedence[token.value])
            exp = PrefixOps.generate(next_expression, token)
            return exp

        elif isinstance(token, Identifier):
            if UnaryFunctions.check(token):  # unaryFunction '(' expression ')'
                func_name = token
                self.remove()  # functionName

                self.expect_token(Special, "(")
                self.remove()  # (
                argument = self.expression()

                self.expect_token(Special, ")")
                self.remove()  # )
                exp = UnaryFunctions.generate(argument, func_name)
                return exp

            elif BinaryFunctions.check(token):  # binaryFunction '(' expression, expression ')'
                func_name = token
                self.remove()  # function name
                self.expect_token(Special, "(")
                self.remove()  # (
                arguments = self.expressions("(")
                self.expect_token(Special, ")")
                self.remove()  # )
                exp = BinaryFunctions.generate(arguments[0], arguments[1], func_name)

            elif token.value in self.data_names:
                exp = Data(name=token.value, line_index=token.line_index, column_index=token.column_index)
                self.remove()  # identifier
            elif token.value in Distributions.names:
                raise ParseError("A distribution has been found in an expressions", self.model_string, token.line_index, token.column_index)
            else:
                exp = self.parse_param(is_lhs=is_lhs)

            next_token = self.peek()
            if isinstance(next_token, Special) and next_token.value == "[":
                # identifier '[' subscript_expressions ']'
                self.remove()  # [
                subscript_expressions = self.expressions("[")  # list of expressions
                # The data types in the parsed expressions are being used as subscripts
                self.expect_token(Special, "]")
                self.remove()  # ]
                # Assume subscript is a single identifier - this is NOT GOOD
                exp.subscript = SubscriptOp(subscripts=subscript_expressions, line_index=token.line_index, column_index=token.column_index)

            return exp

        elif isinstance(token, Special) and token.value == "(":  # ( expression )
            self.remove()  # (
            exp = self.expression()
            self.expect_token(Special, ")")
            self.remove()  # )
            return exp
        else:
            raise ParseError(
                f"{token.value} can't be in the beginning of a construct!", self.model_string, token.line_index, token.column_index
            )

    def parse_param(self, is_lhs=False):
        self.expect_token(Identifier)
        token = self.peek()
        exp = Param(name=token.value, line_index=token.line_index, column_index=token.column_index)
        self.remove()  # identifier

        # check for constraints  param<lower = 0.0, upper = 1.0>
        # 3-token lookahead: "<" + "lower" or "upper"
        lookahead_1 = self.peek()  # <
        lookahead_2 = self.peek(1)  # lower, upper
        if lookahead_1.value == "<" and lookahead_2.value in (
            "lower",
            "upper",
        ):
            if not is_lhs:
                raise ParseError(
                    "Constraints for parameters/variables are only allowed on LHS.",
                    self.model_string,
                    lookahead_1.line_index,
                    lookahead_1.column_index,
                )
            self.remove()  # <
            # the problem is that ">" is considered as an operator, but in the case of constraints, it is
            # not an operator, but a delimeter denoting the end of the constraint region.
            # Therefore, we need to find the matching ">" and change it from operator type to special, so
            # the expression parser does not think of it as a "greater than" operator. This goes away from
            # the ll(k) approach and therefore is a very hacky way to fix the issue.
            n_openbrackets = 0
            for idx in range(len(self.tokens)):
                if self.peek(idx).value == "<":
                    n_openbrackets += 1
                if self.peek(idx).value == ">":
                    if n_openbrackets == 0:
                        # switch from Operator to Special
                        self.tokens[idx] = Special(">", self.peek(idx).line_index, self.peek(idx).column_index)
                        break
                    else:
                        n_openbrackets -= 1
            # now actually parse the constraints
            lower = RealConstant(value=float("-inf"))
            upper = RealConstant(value=float("inf"))
            for _ in range(2):
                # loop at max 2 times, once for lower, once for upper
                if lookahead_2.value == "lower":
                    self.remove()  # "lower"
                    self.expect_token(Operator, token_value="=")
                    self.remove()  # =
                    lower = self.expression()
                elif lookahead_2.value == "upper":
                    self.remove()  # "upper"
                    self.expect_token(Operator, token_value="=")
                    self.remove()  # =
                    upper = self.expression()

                lookahead_1 = self.peek()
                # can be either ",", which means loop again, or ">", which breaks
                lookahead_2 = self.peek(1)
                # either "lower", or "upper" if lookahead_1 == ","
                if lookahead_1.value == ",":
                    self.remove()  # ,
                elif lookahead_1.value == ">":
                    self.remove()  # >
                    break
                else:
                    raise ParseError(
                        f"Found unknown token with value {lookahead_1.value} when evaluating constraints",
                        self.model_string,
                        lookahead_1.line_index,
                        lookahead_1.column_index,
                    )

            # the for loop takes of the portion "<lower= ... >
            # this means the constraint part of been processed and
            # removed from the token queue at this point
            exp.lower = lower
            exp.upper = upper

        return exp

    def expression(self, min_precedence=0, is_lhs=False):
        """
        This function is used to evaluate an expression. Please refer to the BNF grammer to see what types of
        rules are being applied.
        :param min_precedence: Minimum precedence value to evaluate
        :return: An `ops.Expr` object.
        """
        left = self.parse_nud(is_lhs=is_lhs)
        token = self.peek()

        while True:
            if isinstance(token, Special) and token.value in (";", ",", ">", ")", "]"):
                break
            elif isinstance(token, NullToken):
                break
            elif isinstance(token, Special) and token.value == "(":  # '(' expression ')'
                self.remove()  # (
                next_expression = self.expression()
                self.expect_token(Special, ")")  # )
                self.remove()  # )
                exp = next_expression  # expression

            elif InfixOps.check(token):  # expression infixOps expression
                if InfixOps.precedence[token.value] <= min_precedence:
                    break
                self.expect_token(Operator, InfixOps.ops)
                self.remove()  # op
                rhs = self.expression(InfixOps.precedence[token.value])
                exp = InfixOps.generate(left, rhs, token)

            elif isinstance(token, Identifier):
                if UnaryFunctions.check(token):  # unaryFunction '(' expression ')'
                    if UnaryFunctions.precedence[token.value] <= min_precedence:
                        break
                    func_name = token
                    self.remove()  # functionName

                    self.expect_token(Special, "(")
                    self.remove()  # (
                    argument = self.expression()

                    self.expect_token(Special, ")")
                    self.remove()  # )
                    exp = UnaryFunctions.generate(argument, func_name)

                elif BinaryFunctions.check(token):
                    if BinaryFunctions.precedence[token.value] <= min_precedence:
                        break
                    self.remove()  # function name
                    self.expect_token(Special, "(")
                    self.remove()  # (
                    arguments = self.expressions("(")
                    self.expect_token(Special, ")")
                    self.remove()  # )
                    exp = BinaryFunctions.generate(arguments[0], arguments[1], token)

            else:
                raise ParseError(f"Unknown token '{token.value}'", self.model_string, token.line_index, token.column_index)

            token = self.peek()
            left = exp

        return left

    def statement(self):
        """
        Evaluates a single statement. Statements in rat are either assignments or sampling statements. They will get
        resolved into an `ops.Assignment` or an `ops.Distr` object.
        :return:
        """
        self.check_bracket_stack()
        token = self.peek()
        if Distributions.check(token):
            raise ParseError("Cannot assign to a distribution.", self.model_string, token.line_index, token.column_index)

        # Step 1. evaluate lhs, assume it's expression
        lhs = self.parse_nud(is_lhs=True)
        # if isinstance(lhs, Param) or isinstance(lhs, Data):
        if isinstance(lhs, Expr):
            op = self.peek()
            if AssignmentOps.check(op):
                self.remove()  # assignment operator
                rhs = self.expression()
                return AssignmentOps.generate(lhs, rhs, op)

            elif isinstance(op, Special) and op.value == "~":
                # distribution declaration
                self.expect_token(Special, "~")
                self.remove()  # ~
                distribution = self.peek()
                self.expect_token(Identifier, Distributions.names)
                self.remove()  # distribution

                self.expect_token(Special, "(")
                self.remove()  # (
                expressions = self.expressions("(")  # list of expression
                self.expect_token(Special, ")")
                self.remove()  # )
                return Distributions.generate(lhs, expressions, distribution)

            else:
                raise ParseError(f"Unknown operator '{op.value}' in statement", self.model_string, op.line_index, op.column_index)

        return Expr()
