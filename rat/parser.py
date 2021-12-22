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

# https://mc-stan.org/docs/2_18/reference-manual/bnf-grammars.html
# https://mc-stan.org/docs/2_28/reference-manual/arithmetic-expressions.html


# define group parsing rules for operators
class PrefixOps:
    """
    A utility class that's used to identify and build prefix-operation expressions.
    """

    ops = ["!", "-"]

    @staticmethod
    def check(tok: Token):
        if isinstance(tok, Operator) and tok.value in PrefixOps.ops:
            return True
        return False

    @staticmethod
    def generate(expr: Expr, tok: Operator):
        if tok.value == "!":
            return PrefixLogicalNegation(expr)
        elif tok.value == "-":
            return PrefixNegation(expr)


class PostfixOps:  # not used atm
    ops = ["'"]

    @staticmethod
    def check(tok: Token):
        if isinstance(tok, Operator) and tok.value in PostfixOps.ops:
            return True
        return False


class InfixOps:
    """
    A utility class that's used to indentify and build binary operation expressions.
    Currently supported operations are:
    `ops.Sum`, `ops.Diff`, `ops.Mul`, `ops.Pow`, `ops.Mod`
    """

    ops = ["+", "-", "*", "^", "/", "%", "<", ">"]

    @staticmethod
    def check(tok: Type[Token]):
        if isinstance(tok, Operator) and tok.value in InfixOps.ops:
            return True
        return False

    @staticmethod
    def generate(lhs: Expr, rhs: Expr, token: Type[Token]):
        if token.value == "+":
            return Sum(lhs, rhs)
        elif token.value == "-":
            return Diff(lhs, rhs)
        elif token.value == "*":
            return Mul(lhs, rhs)
        elif token.value == "/":
            return Div(lhs, rhs)
        elif token.value == "^":
            return Pow(lhs, rhs)
        elif token.value == "%":
            return Mod(lhs, rhs)
        elif token.value == "||":
            return LogicalOR(lhs, rhs)
        elif token.value == "&&":
            return LogicalAND(lhs, rhs)
        elif token.value == "==":
            return Equality(lhs, rhs)
        elif token.value == "!=":
            return Inequality(lhs, rhs)
        elif token.value == "<":
            return LessThan(lhs, rhs)
        elif token.value == "<=":
            return LessThanOrEqual(lhs, rhs)
        elif token.value == ">":
            return GreaterThan(lhs, rhs)
        elif token.value == ">=":
            return GreaterThanOrEqual(lhs, rhs)
        else:
            raise Exception(f"InfixOps: Unknown operator type {token.value}")


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
    def generate(lhs: Expr, operator: Operator, rhs: Expr):
        # check() has been ran for operator
        if operator.value == "=":
            return Assignment(lhs, rhs)


class UnaryFunctions:
    """
    A utility class that's used to identify and build unary functions.
    """

    names = ["exp", "log", "abs", "floor", "ceil", "round", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "logit", "inverse_logit"]

    @staticmethod
    def check(tok: Type[Token]):
        if isinstance(tok, Identifier) and tok.value in UnaryFunctions.names:
            return True
        return False

    @staticmethod
    def generate(subexpr: Expr, func_type: Identifier):
        if func_type.value == "exp":
            return Exp(subexpr)
        elif func_type.value == "log":
            return Log(subexpr)
        elif func_type.value == "abs":
            return Abs(subexpr)
        elif func_type.value == "floor":
            return Floor(subexpr)
        elif func_type.value == "ceil":
            return Ceil(subexpr)
        elif func_type.value == "round":
            return Round(subexpr)
        elif func_type.value == "sin":
            return Sin(subexpr)
        elif func_type.value == "cos":
            return Cos(subexpr)
        elif func_type.value == "tan":
            return Tan(subexpr)
        elif func_type.value == "arcsin":
            return Arcsin(subexpr)
        elif func_type.value == "arccos":
            return Arccos(subexpr)
        elif func_type.value == "arctan":
            return Arctan(subexpr)
        elif func_type.value == "logit":
            return Logit(subexpr)
        elif func_type.value == "inverse_logit":
            return InverseLogit(subexpr)


class Distributions:
    """
    A utility class that's used to identify and build distributions.
    Currently supported distributions are:
    `ops.Normal`, `ops.BernoulliLogit`, `ops.LogNormal`
    """

    names = ["normal", "bernoulli_logit", "log_normal", "cauchy"]

    @staticmethod
    def check(tok: Type[Token]):
        if isinstance(tok, Identifier) and tok.value in Distributions.names:
            return True
        return False

    @staticmethod
    def generate(dist_type: Identifier, lhs: Expr, expressions: List[Expr]):
        if dist_type.value == "normal":
            if len(expressions) != 2:
                raise Exception(f"normal distribution needs 2 parameters, but got {len(expressions)}!")
            return Normal(lhs, expressions[0], expressions[1])
        elif dist_type.value == "bernoulli_logit":
            if len(expressions) != 1:
                raise Exception(f"bernoulli_logit distribution needs 1 parameters, but got {len(expressions)}!")
            return BernoulliLogit(lhs, expressions[0])
        elif dist_type.value == "log_normal":
            if len(expressions) != 2:
                raise Exception(f"log_normal distribution needs 2 parameters, but got {len(expressions)}!")
            return LogNormal(lhs, expressions[0], expressions[1])
        elif dist_type.value == "cauchy":
            if len(expressions) != 2:
                raise Exception(f"cauchy distribution needs 2 parameters, but got {len(expressions)}!")
            return Cauchy(lhs, expressions[0], expressions[1])


class ParseError(Exception):
    def __init__(self, message, code_string: str, line_num: int, column_num: int):
        code_string = code_string.split("\n")[line_num]
        exception_message = f"An error occured while parsing the following line({line_num}:{column_num}):\n{code_string}\n{' ' * column_num + '^'}\n{message}"
        super().__init__(exception_message)


class Parser:
    """
    The parser for rat is a modified top-down, leftmost derivative LL parser.
    Since rat programs are defined within the context of data, the parser needs to know
    the column names of the data.
    """

    def __init__(self, tokens: List[Type[Token]], data_names: List[str], model_string: str = ""):
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

    def peek(self, k=0) -> Type[Token]:
        """
        k-token lookahead. Returns `scanner.NullToken` if there are no tokens in the token stack.
        :param k:
        :return:
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

    def expressions(self, entry_token_value, allow_shift=False) -> Tuple[List[Expr], Tuple[int]]:
        """
        expressions are used to evaluate repeated, comma-separeted expressions in the form "expr, expr, expr"
        It's primarily used to evaluate subscripts or function arguments. In the case it's evaluating subscripts, it
        will also return the shift amounts of each subscript.
        :param entry_token_value: A single character which denotes the boundary token that starts the expression
        sequence. For example, "myFunc(expr1, expr2, expr3)" would mean the 3-expression sequence is present between the
        parantheses. So the entry token would be "(" and exit token ")".
        For subscripts, it would be something like "my_variable[sub_1, shift(sub_2, 1)]. That would mean entry token
        "[" and exit token "]".
        :param allow_shift: This is for a quick sanity check that checks whether shift() is allowed to be used within
        the expression sequence.
        :return: A Tuple of length 2, with the first value being a list of expressions, and second value being a Tuple
        of integers denoting shift amounts, if any.
        """
        if entry_token_value == "[":
            exit_value = "]"
        elif entry_token_value == "(":
            exit_value = ")"
        else:
            raise Exception(f"expresions() received invalid entry token value with value {entry_token_value}, but expected '[' or ']'")
        expressions = []
        shift_amounts = []  # integer specifying the amount to shift for each index
        # while True:
        #     token = self.peek()
        #     if isinstance(token, Special):
        #         if token.value == "]" or token.value == ")":
        #             break
        #         elif token.value == ",":
        #             self.remove()  # character ,
        #             continue
        #         elif token.value == "[":
        #             self.remove()  # character [
        #             expression = self.expressions()  # nested expressions ex: a[b[1], 2]
        #         else:
        #             expression = self.expression()
        #     else:
        #         expression = self.expression()
        #     expressions.append(expression)
        while True:
            token = self.peek()
            shift_amount = None
            if isinstance(token, Special):
                self.expect_token(Special, (exit_value, ","))
                if token.value == exit_value:
                    break
                elif token.value == ",":
                    self.remove()  # character ,
                    continue
            elif isinstance(token, Identifier) and token.value == "shift":
                if not allow_shift:
                    raise ParseError("shift() has been used in a position that is not allowed.", self.model_string, token.line_index, token.column_index)
                # parse lag(index, integer)
                self.remove()  # identifier "lag"
                self.expect_token(Special, "(")
                self.remove()  # character )
                self.expect_token(Identifier)  # index name
                subscript_name = self.peek()
                if subscript_name.value not in self.data_names:
                    raise ParseError("Index specified with shift() must be in data columns.", self.model_string, subscript_name.line_index, subscript_name.column_index)
                expression = Data(subscript_name.value)
                self.remove()  # index name
                self.expect_token(Special, ",")
                self.remove()  # character ,
                self.expect_token(IntLiteral)  # shift amount
                shift_amount = int(self.peek().value)
                self.remove()  # shift integer
                self.expect_token(Special, ")")
                self.remove()  # character )
            else:
                expression = self.expression()
            expressions.append(expression)
            shift_amounts.append(shift_amount)

        return expressions, tuple(shift_amounts)

    def expression(self, allow_subscripts=False):
        """
        This function is used to evaluate an expression. Please refer to the BNF grammer to see what types of
        rules are being applied.
        :param allow_subscripts:
        :return: An `ops.Expr` object.
        """
        token = self.peek()

        exp = Expr()
        if isinstance(token, RealLiteral):  # if just a real number, return it
            exp = RealConstant(float(token.value))
            self.remove()  # real
        if isinstance(token, IntLiteral):  # if just an integer, return it
            exp = IntegerConstant(int(token.value))
            self.remove()  # integer

        if isinstance(token, Identifier):  # parameter/data/function
            if UnaryFunctions.check(token):  # unaryFunction '(' expression ')'
                func_name = token
                self.remove()  # functionName

                self.expect_token(Special, "(")
                self.remove()  # (
                argument = self.expression()

                self.expect_token(Special, ")")
                self.remove()  # )
                exp = UnaryFunctions.generate(argument, func_name)

            else:  # parameter/data
                if token.value in self.data_names:
                    exp = Data(token.value)
                    self.remove()  # identifier
                elif token.value in Distributions.names:
                    raise ParseError("A distribution has been found in an expressions", self.model_string, token.line_index, token.column_index)
                else:
                    exp = Param(token.value)
                    self.remove()  # identifier

                    # check for constraints  param<lower = 0.0, upper = 1.0>
                    # 3-token lookahead: "<" + "lower" or "upper"
                    lookahead_1 = self.peek()  # <
                    lookahead_2 = self.peek(1)  # lower, upper
                    if lookahead_1.value == "<" and lookahead_2.value in (
                        "lower",
                        "upper",
                    ):
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
                        lower = RealConstant(float("-inf"))
                        upper = RealConstant(float("inf"))
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

        if PrefixOps.check(token):  # prefixOp expression
            self.expect_token(Operator, PrefixOps.ops)  # operator
            self.remove()

            next_expression = self.expression()
            exp = PrefixOps.generate(next_expression, token)

        if isinstance(token, Special) and token.value == "(":  # '(' expression ')'
            self.remove()  # (
            next_expression = self.expression()
            self.expect_token(Special, ")")  # )
            self.remove()  # )
            exp = next_expression  # expression

        next_token = self.peek()
        # this is for the following 2 rules, which have conditions after expression
        if isinstance(next_token, Special) and next_token.value == "[":
            # identifier '[' subscript_expressions ']'
            self.remove()  # [
            warnings.warn("Parser: subscripts are assumed to be a single literal, not expression.")
            expressions, shift_amount = self.expressions("[", allow_shift=True)  # list of expression
            self.expect_token(Special, "]")
            self.remove()  # ]
            # Assume index is a single identifier - this is NOT GOOD
            exp.index = Index(
                names=tuple(expression.name for expression in expressions),
                shifts=shift_amount,
            )
            next_token = self.peek()  # Update token in case we need evaluate case 2

        if InfixOps.check(next_token):  # expression infixOps expression
            self.expect_token(Operator, InfixOps.ops)
            self.remove()  # op
            rhs = self.expression()
            exp = InfixOps.generate(exp, rhs, next_token)

        return exp

    def statement(self):
        """
        Evaluates a single statement. Statements in rat are either assignments or sampling statements. They will get
        resolved into an `ops.Assignment` or an `ops.Distr` object.
        :return:
        """
        token = self.peek()
        if Distributions.check(token):
            raise ParseError("Cannot assign to a distribution.", self.model_string, token.line_index, token.column_index)

        # Step 1. evaluate lhs, assume it's expression
        lhs = self.expression()
        # if isinstance(lhs, Param) or isinstance(lhs, Data):
        if isinstance(lhs, Expr):
            op = self.peek()
            if AssignmentOps.check(op):
                self.remove()  # assignment operator
                rhs = self.expression()
                return AssignmentOps.generate(lhs, op, rhs)

            elif isinstance(op, Special) and op.value == "~":
                # distribution declaration
                self.expect_token(Special, "~")
                self.remove()  # ~
                distribution = self.peek()
                self.expect_token(Identifier, Distributions.names)
                self.remove()  # distribution

                self.expect_token(Special, "(")
                self.remove()  # (
                expressions, _ = self.expressions("(")  # list of expression
                self.expect_token(Special, ")")
                self.remove()  # )
                return Distributions.generate(distribution, lhs, expressions)

            else:
                raise ParseError(f"Unknown operator '{op.value}' in statement", self.model_string, op.line_index, op.column_index)

        return Expr()
