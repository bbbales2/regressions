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
    line_index: int
    """An integer denoting the line number of the Token in the original code string"""
    column_index: int
    """An integer denoting the start position of the Token in the code line"""
    token_type: str
    """A string representing the type of token. Set by self.__class__.__name"""

    def __init__(self, value, line_index, column_index):
        self.value: str = value
        self.line_index: int = line_index
        self.column_index: int = column_index
        self.token_type = self.__class__.__name__


class Identifier(Token):
    """
    The Identifier token represents any alphanumeric string that isn't used as a standalone number.
    This includes function names, variables, distributions, etc.
    """

    def __init__(self, value, line_index, column_index):
        super(Identifier, self).__init__(value, line_index, column_index)


class Special(Token):
    """
    The Special token represents any single characters that indicate change in semantics or control flow.
    Explicitly any single token that has the following values("(", ")", ",", "[", "]", "~") are set as Special
    """

    def __init__(self, value, line_index, column_index):
        super(Special, self).__init__(value, line_index, column_index)


class IntLiteral(Token):
    """
    The IntLiteral token represent integers.
    """

    def __init__(self, value, line_index, column_index):
        super(IntLiteral, self).__init__(value, line_index, column_index)


class RealLiteral(Token):
    """
    The RealLiteral token represent real numbers. Currently the regex expression
    ^[-]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?$ is being used to parse real numbers, which allows negative values as well as
    scientific notation.
    """

    def __init__(self, value, line_index, column_index):
        super(RealLiteral, self).__init__(value, line_index, column_index)


class Operator(Token):
    """
    The Operator token represent character(s) that represent operators. Explicitly, any single token that has the
    following values("=","+","-","*","/","^","%") are
    set as an Operator.
    """

    def __init__(self, value, line_index, column_index):
        super(Operator, self).__init__(value, line_index, column_index)


class Terminate(Token):
    """
    In rat, all statements are terminated using with ; and not newline.
    """

    def __init__(self, value, line_index, column_index):
        super(Terminate, self).__init__(value, line_index, column_index)


class NullToken(Token):
    """
    NullToken doesn't represent anything. This is pretty much used in the same context as None.
    """

    def __init__(self):
        super(NullToken, self).__init__(None, -1, -1)


special_characters = [
    "(",
    ")",
    ",",
    "[",
    "]",
    "~",
    "{",
    "}",
]  # symbols indicating change in semantics or control flow

operator_strings = [
    "=",
    "+",
    "-",
    "*",
    "/",
    "^",
    "%",
    "<",
    ">",
    "=",
]

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


class TokenizeError(Exception):
    def __init__(self, message: str, code_string: str, line_num: int, column_num: int):
        code_string = code_string.split("\n")[line_num]
        exception_message = f"An error occured while tokenizing the following line({line_num}:{column_num}):\n{code_string}\n{' ' * (column_num - 1) + '^'}\n{message}"
        super().__init__(exception_message)


class Scanner:
    def __init__(self, model_code):
        self.model_code: str = model_code
        self.code_length: int = len(model_code)
        self.scanned_lines: List[List[Token]] = []  # List of tokens which have been tokenized
        self.current_tokens = []  # Tokens which are on the current line
        self.current_char: str = ""  # head character of model string
        self.register: str = ""  # The register stores the string that's currently being digested

        self.index: int = 0  # This is the current model string subscript

        self.column_index: int = 0
        self.line_index: int = 0

        self.current_state = self.default_state

    def scan(self) -> List[List[Token]]:
        """
        Entry function of the Scanner class.
        """
        while self.index < self.code_length:
            self.current_state()

        if self.register == ";":
            self.current_state()
        elif self.current_tokens:
            self.raise_error("Unfinished statement. Missing termination character!")

        return self.scanned_lines

    def consume(self, ignore_newline=True):
        if self.index == self.code_length:
            self.current_char = ""
            return
        self.current_char = self.model_code[self.index]
        self.index += 1
        self.column_index += 1
        if self.current_char == "\n":
            self.current_state = self.default_state
            self.column_index = -1
            self.line_index += 1
            if ignore_newline:
                self.consume()

    def reduce_register(self):
        def casts_to_int(val):
            try:
                int(val)
            except ValueError:
                return False
            else:
                return True

        if self.register == " " or not self.register:
            token = NullToken()
        elif self.register == ";":
            if self.current_tokens:
                self.scanned_lines.append(self.current_tokens)
            self.current_tokens = []
            self.register = ""
            return
        elif casts_to_int(self.register):
            token = IntLiteral(self.register, self.line_index, self.column_index - len(self.register))
        elif realnum.match(self.register):
            token = RealLiteral(self.register, self.line_index, self.column_index - len(self.register))
        elif self.register in special_characters:
            token = Special(self.register, self.line_index, self.column_index - len(self.register) + 1)
        elif self.register in operator_strings:
            token = Operator(self.register, self.line_index, self.column_index - len(self.register) + 1)
        else:
            if self.register.isidentifier():
                token = Identifier(self.register, self.line_index, self.column_index - len(self.register))
            else:
                token = NullToken()

        self.register = ""
        if not isinstance(token, NullToken):
            self.current_tokens.append(token)

    def raise_error(self, msg):
        raise TokenizeError(msg, self.model_code, self.line_index, self.column_index)

    # The following functions are scanner states

    def delimeter_state(self):
        self.reduce_register()
        self.current_state = self.default_state

    def default_state(self):
        self.consume()
        if not self.current_char:
            return

        elif self.current_char == "#":
            # comment line. Read through end of line
            while self.current_char and self.current_char != "\n":
                self.consume(ignore_newline=False)

        elif self.current_char in delimeters:
            # If we find a terminate token, stay on current state
            self.register += self.current_char
            self.current_state = self.delimeter_state

        elif self.current_char in operator_strings:
            self.register += self.current_char
            self.current_state = self.operator_state

        elif self.current_char.isalpha():
            # transition 1: If we find an alphabet, change to identifier state.
            self.register += self.current_char
            self.current_state = self.identifier_state

        elif self.current_char.isnumeric():
            # transition 2: If we find an integer, change to Integer state
            self.register += self.current_char
            self.current_state = self.integer_state

        else:
            self.raise_error(f"Don't know how to resolve chracter '{self.current_char}'; No lexing rule to apply.")

    def identifier_state(self):
        self.consume()
        if not self.current_char:
            self.reduce_register()

        elif self.current_char.isalnum() or self.current_char == "_":
            # isidentifier() will not work, since we allow digits
            self.register += self.current_char

        elif self.current_char == "#":
            self.reduce_register()
            # comment line. Read through end of line
            while self.current_char and self.current_char != "\n":
                self.consume(ignore_newline=False)
            self.current_state = self.default_state

        elif self.current_char in delimeters:
            # transition 1: If we find a delimeter, change to delimeter state
            self.reduce_register()
            self.register += self.current_char
            self.current_state = self.delimeter_state

        elif self.current_char in operator_strings:
            # transition 2: If we find an operator, change to operator state
            self.reduce_register()
            self.register += self.current_char
            self.current_state = self.operator_state

        else:
            self.raise_error(f"Character '{self.current_char}' cannot be present within an Identifier!")

    def integer_state(self):
        self.consume()
        if not self.current_char:
            self.reduce_register()

        elif self.current_char == "#":
            self.reduce_register()
            # comment line. Read through end of line
            while self.current_char and self.current_char != "\n":
                self.consume(ignore_newline=False)
            self.current_state = self.default_state

        elif self.current_char in delimeters:
            self.reduce_register()
            self.register += self.current_char
            self.current_state = self.delimeter_state

        elif self.current_char.isnumeric():
            # If we find a numeric, stay on current state
            self.register += self.current_char

        elif self.current_char == ".":
            # transition 1: If we find a decimal point, change to real State
            self.register += self.current_char
            self.current_state = self.real_state

        elif self.current_char == "e":
            # transition 2: If we find character "e", change to real_exponent state
            self.register += self.current_char
            self.current_state = self.real_exponent_state

        elif self.current_char in operator_strings:
            # transition 3: If we find an operator, change to operator state
            self.reduce_register()
            self.register += self.current_char
            self.current_state = self.operator_state

        else:
            self.raise_error(f"Character '{self.current_char}' cannot be present within an Integer!")

    def real_state(self):
        self.consume()
        if not self.current_char:
            self.reduce_register()

        elif self.current_char == "#":
            self.reduce_register()
            # comment line. Read through end of line
            while self.current_char and self.current_char != "\n":
                self.consume(ignore_newline=False)
            self.current_state = self.default_state

        elif self.current_char in delimeters:
            self.reduce_register()
            self.register += self.current_char
            self.current_state = self.delimeter_state

        elif self.current_char.isnumeric():
            # If we find a numeric, stay on current state
            self.register += self.current_char

        elif self.current_char == "e":
            # transition 1: If we find character "e", change to real_exponent state
            self.register += self.current_char
            self.current_state = self.real_exponent_state

        elif self.current_char in operator_strings:
            # transition 2: If we find an operator, change to operator state
            self.reduce_register()
            self.register += self.current_char
            self.current_state = self.operator_state

        else:
            self.raise_error(f"Character '{self.current_char}' cannot be present within a Real!")

    def real_exponent_state(self):
        self.consume()
        if not self.current_char:
            self.reduce_register()

        elif self.current_char == "#":
            self.reduce_register()
            # comment line. Read through end of line
            while self.current_char and self.current_char != "\n":
                self.consume(ignore_newline=False)
            self.current_state = self.default_state

        elif self.current_char in delimeters:
            self.reduce_register()
            self.register += self.current_char
            self.current_state = self.delimeter_state

        elif self.current_char.isnumeric():
            # If we find a numeric, stay on current state
            self.register += self.current_char

        elif self.current_char == "-":
            # Allow negative powers i.e. 1e-10
            if self.register[-1] == "e":
                # - is allowed exponent after 'e'
                self.register += self.current_char
            else:
                # if we aleady have a integer, the negative sign is an operator
                self.reduce_register()
                self.register += self.current_char
                self.current_state = self.operator_state

        elif self.current_char in operator_strings and self.register[-1] != "e":
            # transition 1: If we find an operator, change to operator state
            # This in kind of cheating since we're doing a lookback, but I'm too lazy to create another state for just e
            self.reduce_register()
            self.register += self.current_char
            self.current_state = self.operator_state

        else:
            self.raise_error(f"Character '{self.current_char}' cannot be present within scientific notation!")

    def operator_state(self):
        # if self.model_code[self.subscript].isnumeric() and self.register == "-":
        #     # transition 1: if we find a numeric, change to integer state
        #     # lookahead used, needs to be improved
        #     #self.register += self.current_char
        #     self.current_state = self.integer_state
        #
        # else:
        #     self.reduce_register()
        #     self.current_state = self.default_state
        self.reduce_register()
        self.current_state = self.default_state
