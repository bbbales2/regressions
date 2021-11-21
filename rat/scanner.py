import typing
import re

realnum = re.compile("^[-]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?$")  # (negative or positive) + (integer, real, scientific)


class Token:
    def __init__(self, value):
        self.value = value
        self.token_type = self.__class__.__name__


class Identifier(Token):
    def __init__(self, value):
        super(Identifier, self).__init__(value)


class Special(Token):
    def __init__(self, value):
        super(Special, self).__init__(value)


class IntLiteral(Token):
    def __init__(self, value):
        super(IntLiteral, self).__init__(value)


class RealLiteral(Token):
    def __init__(self, value):
        super(RealLiteral, self).__init__(value)


class Operator(Token):
    def __init__(self, value):
        super(Operator, self).__init__(value)


class Terminate(Token):
    def __init__(self, value):
        super(Terminate, self).__init__(value)


class NullToken(Token):
    def __init__(self):
        super(NullToken, self).__init__(None)


special_characters = [
    "(",
    ")",
    ",",
    "[",
    "]",
    "~",
]  # symbols indicating change in semantics or control flow
operator_strings = [
    "=",
    "+=",
    "-=",
    "/=",
    "*=",
    "%=",
    "+",
    "-",
    "*",
    "/",
    "^",
    "%",
    "!",
    ">=",
    "<=",
    "==",
    "<",
    ">",
    "&&",
    "||",
]


def scanner(model_code):
    delimeters = [
        " ",
        ";",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        ",",
        "~",
    ]  # characters that ALWAYS demarcate tokens
    result = []
    charstack = ""
    model_code = " ".join(model_code.split("\n"))
    while model_code:
        char = model_code[0]
        model_code = model_code[1:]
        if char in delimeters:
            if charstack:
                resolved = resolve_token(charstack)
                if resolved:
                    result.append(resolved)
                charstack = ""
            resolved = resolve_token(char)
            if resolved:
                result.append(resolved)

        elif char in operator_strings:
            if charstack and charstack + char not in operator_strings:
                resolved = resolve_token(charstack)
                if resolved:
                    result.append(resolved)
                charstack = ""
            charstack += char

        elif charstack in operator_strings:
            if charstack == "-":
                resolved = resolve_token(charstack + char)
                if resolved and (isinstance(resolved, IntLiteral) or isinstance(resolved, RealLiteral)):
                    charstack += char
                    continue
            if charstack + char not in operator_strings:
                resolved = resolve_token(charstack)
                if resolved:
                    result.append(resolved)
                charstack = char
            else:
                charstack += char
        else:
            charstack += char

    if charstack:
        resolved = resolve_token(charstack)
        if resolved:
            result.append(resolved)

    return result


def casts_to_int(val):
    try:
        int(val)
    except ValueError:
        return False
    else:
        return True


def resolve_token(charstack):
    if charstack == " " or not charstack:
        return None
    elif charstack == ";":
        return Terminate(charstack)
    elif casts_to_int(charstack):
        return IntLiteral(charstack)
    elif realnum.match(charstack):
        return RealLiteral(charstack)
    elif charstack in special_characters:
        return Special(charstack)
    elif charstack in operator_strings:
        return Operator(charstack)
    else:
        return Identifier(charstack)


if __name__ == "__main__":
    test_code = """
score_diff ~ normal(skills[home_team, year] - skills[away_team, year], sigma);
skills[team, year] ~ normal(skills_mu[year], tau);
tau += -10;
sigma ~ normal(0.0, 10.0);"""

    for val in scanner(test_code):
        print(val.token_type, val.value)
