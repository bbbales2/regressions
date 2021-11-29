from typing import ClassVar, Dict, Tuple, Set, List
import re

realnum = re.compile("^[-]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?$")  # (negative or positive) + (integer, real, scientific)


class Token:
    """
    A token denotes a sequence of input characters, typically demarcated by other characters that represent
    semantics, most commonly whitespace. Please Refer to individual token types' docs for more information
    """

    value: str
    """A string containing the input characters"""
    token_type: str
    """A string representing the type of token. Set by self.__class__.__name"""

    def __init__(self, value):
        self.value: str = value
        self.token_type = self.__class__.__name__


class Identifier(Token):
    """
    The Identifier token represents any alphanumeric string that isn't used as a standalone number.
    This includes function names, variables, distributions, etc.
    """

    def __init__(self, value):
        super(Identifier, self).__init__(value)


class Special(Token):
    """
    The Special token represents any single characters that indicate change in semantics or control flow.
    Explicitly any single token that has the following values("(", ")", ",", "[", "]", "~") are set as Special
    """

    def __init__(self, value):
        super(Special, self).__init__(value)


class IntLiteral(Token):
    """
    The IntLiteral token represent integers.
    """

    def __init__(self, value):
        super(IntLiteral, self).__init__(value)


class RealLiteral(Token):
    """
    The RealLiteral token represent real numbers. Currently the regex expression
    ^[-]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?$ is being used to parse real numbers, which allows negative values as well as
    scientific notation.
    """

    def __init__(self, value):
        super(RealLiteral, self).__init__(value)


class Operator(Token):
    """
    The Operator token represent character(s) that represent operators. Explicitly, any single token that has the
    following values("=","+=","-=","/=","*=","%=","+","-","*","/","^","%","!",">=","<=","==","<",">","&&","||") are
    set as an Operator.
    """

    def __init__(self, value):
        super(Operator, self).__init__(value)


class Terminate(Token):
    """
    In rat, all statements are terminated using with ;. As of now, the scanner nor parser relies on the Terminate
    token when resolving. This is left in case the need for explicitly terminated statements arise in the future.
    """

    def __init__(self, value):
        super(Terminate, self).__init__(value)


class NullToken(Token):
    """
    NullToken doesn't represent anything. This is pretty much used in the same context as None.
    """

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


def scanner(model_code: str) -> List[Token]:
    """
    The scanner receives a string as an input and returns a list of `Token`s.
    :param model_code: A string of rat code
    :return: A list of `Token`s.
    """
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
