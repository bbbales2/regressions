from typing import *
from scanner import Token, Identifier, Operator, RealLiteral, IntLiteral, Special
from ops import *

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
    ops = ["+", "-", "*", "/", "%", "||", "&&", "==", "!=", "<", "<=", ">", ">="]

    @staticmethod
    def check(tok: Token):
        if isinstance(tok, Operator) and tok.value in InfixOps.ops:
            return True
        return False

    @staticmethod
    def generate(lhs: Expr, rhs: Expr, token: Operator):
        if token.value == "+":
            return Sum(lhs, rhs)
        elif token.value == "-":
            return Diff(lhs, rhs)
        elif token.value == "*":
            return Mul(lhs, rhs)
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


# group parsing rules for statements

class AssignmentOps:
    ops = ["=", "+=", "-=", "*=", "/="]

    @staticmethod
    def check(tok: Token):
        if isinstance(tok, Operator) and tok.value in AssignmentOps.ops:
            return True
        return False

    @staticmethod
    def generate(lhs: Identifier, operator: Operator, rhs: Expr):
        if operator.value == "=":
            return Assignment(lhs, rhs)
        elif operator.value == "+=":
            return AddAssignment(lhs, rhs)
        elif operator.value == "-=":
            return DiffAssignment(lhs, rhs)
        elif operator.value == "*=":
            return MulAssignment(lhs, rhs)
        elif operator.value == "/=":
            return DivAssignment(lhs, rhs);


class DefaultFunctions:
    names = ["exp", "abs", "floor", "ceil", "round"]

    @staticmethod
    def check(tok: Token):
        if isinstance(tok, Identifier) and tok.value in DefaultFunctions.names:
            return True
        return False

    @staticmethod
    def generate(subexpr: Expr, tok: Identifier):
        pass


class Distributions:
    names = ["normal"]

    @staticmethod
    def check(tok: Token):
        if isinstance(tok, Identifier) and tok.value in Distributions.names:
            return True
        return False

    @staticmethod
    def generate(dist_type: Identifier, lhs: Expr, expressions: List[Expr]):
        if dist_type.value == "normal":
            if len(expressions) != 2:
                raise Exception(f"normal distribution needs 2 parameters, but got {len(expressions)}!")
            return Normal(lhs, expressions[0], expressions[1])


class Parser:
    def __init__(self, tokens: List[Token]):
        self.out_tree = []
        self.tokens = tokens

    def peek(self, k=0) -> Token:
        if k >= len(self.tokens):
            return None
        return self.tokens[k]

    def remove(self, index=0):
        self.tokens.pop(index)

    def expressions(self):
        expressions = []
        while True:
            token = self.peek()
            if isinstance(token, Special):
                if token.value == "]" or token.value == ")":
                    break
                elif token.value == ",":
                    self.remove()  # ,
                    continue
                elif token.value == "[":
                    self.remove()  # [
                    expression = self.expressions()  # nested expressions ex: a[b[1], 2]
                else:
                    expression = self.expression()
            else:
                expression = self.expression()
            expressions.append(expression)
        return expressions

    def expression(self):
        token = self.peek()
        #print(token.token_type, token.value)
        exp = Expr()
        if isinstance(token, RealLiteral):  # if just a real number, return it
            exp = RealConstant(token.value)
            self.remove()  # real
        if isinstance(token, IntLiteral):  # if just a integer, return it
            exp = IntegerConstant(token.value)
            self.remove()  # integer

        if isinstance(token, Identifier):  # parameter/data
            exp = Placeholder(token.value, None)
            self.remove()  # identifier

        if PrefixOps.check(token):  # prefixOp expression
            self.remove()  # op
            next_expression = self.expression()
            exp = PrefixOps.generate(next_expression, token)

        if isinstance(token, Special) and token.value == "(":  # '(' expression ')'
            self.remove()  # (
            next_expression = self.expression()
            self.remove()  # )
            exp = next_expression  # expression

        next_token = self.peek()  # this is for the following 2 rules, which have conditions after expression
        if isinstance(next_token, Special) and next_token.value == "[":  # identifier '[' expressions ']'
            self.remove()  # ]
            expressions = self.expressions()  # list of expression
            self.remove()  # ]
            exp.index = Index(*expressions)
            next_token = self.peek()  # Update token in case we need evaluate case 2

        if InfixOps.check(next_token):  # expression infixOps expression
            self.remove()  # op
            rhs = self.expression()
            exp = InfixOps.generate(exp, rhs, next_token)

        return exp

    def statement(self):
        token = self.peek()
        if DefaultFunctions.check(token):
            raise Exception("Cannot assign to a function name.")
        if Distributions.check(token):
            raise Exception("Cannot assign to a distribution.")

        # Step 1. evaluate lhs, assume it's expression
        lhs = self.expression()

        if isinstance(lhs, Placeholder):
            op = self.peek()
            if AssignmentOps.check(op):
                self.remove()  # assignment operator
                rhs = self.expression()
                return AssignmentOps.generate(lhs, op, rhs)

            elif isinstance(op, Special) and op.value == "~":  # distribution declaration
                self.remove()  # ~
                distribution = self.peek()
                self.remove()  # distribution
                if not Distributions.check(distribution):
                    raise Exception(f"No valid distribution defined after ~. '{distribution.value}' is not a valid distribution.'")
                self.remove()  # (
                expressions = self.expressions()  # list of expression
                return Distributions.generate(distribution, lhs, expressions)

            else:
                raise Exception("Statement finished without assignment")

        return lhs


if __name__ == '__main__':
    from scanner import scanner

    teststr = """
score_diff~normal(skills[home_team, year]-skills[away_team, year],sigma);
skills[team, year] ~ normal(skills_mu[year], tau);
tau += -10;
rob = 500;
sigma ~ normal(0.0, -10.0);"""

    #teststr = "tau += -1"

    for line in teststr.split("\n"):
        if not line: continue
        print("-" * 10)
        print(line)
        print(list(x.value for x in scanner(line)))
        #print(Parser(scanner(line)).statement().code() + ";")
        print(Parser(scanner(line)).statement())
