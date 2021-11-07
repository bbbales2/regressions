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
    ops = ["+", "-", "*", "^", "/", "%", "||", "&&", "==", "!=", "<", "<=", ">", ">="]

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
    ops = ["=", "+=", "-=", "*=", "/="]

    @staticmethod
    def check(tok: Type[Token]):
        if isinstance(tok, Operator) and tok.value in AssignmentOps.ops:
            return True
        return False

    @staticmethod
    def generate(lhs: Expr, operator: Operator, rhs: Expr):
        if operator.value == "=":
            return Assignment(lhs, rhs)
        elif operator.value == "+=":
            return AddAssignment(lhs, rhs)
        elif operator.value == "-=":
            return DiffAssignment(lhs, rhs)
        elif operator.value == "*=":
            return MulAssignment(lhs, rhs)
        elif operator.value == "/=":
            return DivAssignment(lhs, rhs)


class UnaryFunctions:
    names = ["exp", "abs", "floor", "ceil", "round"]

    @staticmethod
    def check(tok: Type[Token]):
        if isinstance(tok, Identifier) and tok.value in UnaryFunctions.names:
            return True
        return False

    @staticmethod
    def generate(subexpr: Expr, func_type: Identifier):
        if func_type.value == "exp":
            return Exp(subexpr)
        elif func_type.value == "abs":
            return Abs(subexpr)
        elif func_type.value == "floor":
            return Floor(subexpr)
        elif func_type.value == "ceil":
            return Ceil(subexpr)
        elif func_type.value == "round":
            return Round(subexpr)


class Distributions:
    names = ["normal", "bernoulli_logit"]

    @staticmethod
    def check(tok: Type[Token]):
        if isinstance(tok, Identifier) and tok.value in Distributions.names:
            return True
        return False

    @staticmethod
    def generate(dist_type: Identifier, lhs: Expr, expressions: List[Expr]):
        if dist_type.value == "normal":
            if len(expressions) != 2:
                raise Exception(
                    f"normal distribution needs 2 parameters, but got {len(expressions)}!"
                )
            return Normal(lhs, expressions[0], expressions[1])
        elif dist_type.value == "bernoulli_logit":
            if len(expressions) != 1:
                raise Exception(
                    f"bernoulli_logit distribution needs 1 parameters, but got {len(expressions)}!"
                )
            return BernoulliLogit(lhs, expressions[0])


class Parser:
    def __init__(self, tokens: List[Type[Token]], data_names: List[str]):
        self.out_tree = []
        self.tokens = tokens
        self.data_names = data_names

    def peek(self, k=0) -> Type[Token]:
        if k >= len(self.tokens):
            return NullToken()
        return self.tokens[k]

    def remove(self, index=0):
        self.tokens.pop(index)

    def expect_token(
        self, token_type: Type[Token], token_value=None, remove=False, lookahead=0
    ):
        next_token = self.peek(lookahead)
        if not token_value:
            token_value = [next_token.value]

        if isinstance(token_value, str):
            token_value = [token_value]

        if isinstance(next_token, token_type) and next_token.value in token_value:
            if remove:
                self.remove()
            return True

        raise Exception(
            f"Expected token type {token_type.__name__} with value in {token_value}, but received {next_token.__class__.__name__} with value '{next_token.value}'!"
        )

    def expressions(self):
        expressions = []
        while True:
            token = self.peek()
            if isinstance(token, Special):
                if token.value == "]" or token.value == ")":
                    break
                elif token.value == ",":
                    self.remove()  # character ,
                    continue
                elif token.value == "[":
                    self.remove()  # character [
                    expression = self.expressions()  # nested expressions ex: a[b[1], 2]
                else:
                    expression = self.expression()
            else:
                expression = self.expression()
            expressions.append(expression)
        return expressions

    def expression(self):
        token = self.peek()

        exp = Expr()
        if isinstance(token, RealLiteral):  # if just a real number, return it
            exp = RealConstant(float(token.value))
            self.remove()  # real
        if isinstance(token, IntLiteral):  # if just a integer, return it
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
                                    self.tokens[idx] = Special(">")
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
                                raise Exception(
                                    f"Found unknown token with value {lookahead_1.value} when evaluating constraints"
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
            # identifier '[' expressions ']'
            self.remove()  # [
            warnings.warn(
                "Parser: Indices are assumed to be a single literal, not expression."
            )
            expressions = self.expressions()  # list of expression
            self.expect_token(Special, "]")
            self.remove()  # ]

            # Assume index is a single identifier - this is NOT GOOD

            exp.index = Index(tuple(expression.name for expression in expressions))
            next_token = self.peek()  # Update token in case we need evaluate case 2

        if InfixOps.check(next_token):  # expression infixOps expression
            self.expect_token(Operator, InfixOps.ops)
            self.remove()  # op
            rhs = self.expression()
            exp = InfixOps.generate(exp, rhs, next_token)

        return exp

    def statement(self):
        token = self.peek()
        if Distributions.check(token):
            raise Exception("Cannot assign to a distribution.")

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
                expressions = self.expressions()  # list of expression
                return Distributions.generate(distribution, lhs, expressions)

            else:
                raise Exception("Statement finished without assignment")

        return Expr()
