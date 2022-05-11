from typing import ClassVar, Dict, Tuple, Set, List
import re

from .position_and_range import Position, Range
from .exceptions import TokenizeError


realnum = re.compile("^[-]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?$")  # (negative or positive) + (integer, real, scientific)


class Token:
    """
    A token denotes a sequence of input characters, typically demarcated by other characters that represent
    semantics, most commonly whitespace. Please Refer to individual token types' docs for more information
    """

    value: str
    """A string containing the input characters"""
    start: Position
    """Position in file of start of token"""
    token_type: str
    """A string representing the type of token. Set by self.__class__.__name"""

    def __init__(self, value, start):
        self.value = value
        self.start = start
        self.token_type = self.__class__.__name__

    @property
    def end(self) -> Position:
        """Position in file of end of token"""
        return Position(self.start.line, self.start.col + len(self.value), document=self.start.document)

    @property
    def range(self) -> Range:
        """Range of token in file"""
        return Range(self.start, self.end)


class Identifier(Token):
    """
    The Identifier token represents any alphanumeric string that isn't used as a standalone number.
    This includes function names, variables, distributions, etc.
    """

    pass


class Special(Token):
    """
    The Special token represents any single characters that indicate change in semantics or control flow.
    Explicitly any single token that has the following values("(", ")", ",", "[", "]", "~") are set as Special
    """

    pass


class IntLiteral(Token):
    """
    The IntLiteral token represent integers.
    """

    pass


class RealLiteral(Token):
    """
    The RealLiteral token represent real numbers. Currently the regex expression
    ^[-]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?$ is being used to parse real numbers, which allows negative values as well as
    scientific notation.
    """

    pass


class Operator(Token):
    """
    The Operator token represent character(s) that represent operators. Explicitly, any single token that has the
    following values("=","+","-","*","/","^","%") are
    set as an Operator.
    """

    pass


class Terminate(Token):
    """
    In rat, all statements are terminated using with ; and not newline.
    """

    pass


class NullToken(Token):
    """
    NullToken doesn't represent anything. This is pretty much used in the same context as None.
    """

    def __init__(self):
        pass


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

operator_strings = ["=", "+", "-", "*", "/", "^", "%", "<", ">", "=", ">=", "<=", "!=", "==", "'"]

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


class Scanner:
    def __init__(self, model_code):
        self.model_code: str = model_code
        self.code_length: int = len(model_code)
        self.scanned_lines: List[List[Token]] = []  # List of tokens which have been tokenized
        self.current_tokens = []  # Tokens which are on the current line
        self.current_char: str = ""  # head character of model string
        self.register: str = ""  # The register stores the string that's currently being digested

        self.bracket_stack: List[str] = []
        # this list is to check whether brackets are closed correctly, and determine the scope of braces
        # (, [, {

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
        """
        Get a token from the stream
        """
        if self.index == self.code_length:
            self.current_char = ""
            return
        self.current_char = self.model_code[self.index]
        self.index += 1
        self.column_index += 1
        if self.current_char == "\n":
            self.current_state = self.default_state
            self.column_index = 0
            self.line_index += 1
            if ignore_newline:
                self.consume()

    def raise_error(self, msg):
        position = Position(self.line_index, self.column_index, self.model_code)
        raise TokenizeError(msg, Range(position, length=len(self.register)))

    def reduce_register(self, offset: int = 1):
        """
        Resolve the current characters in the register into a token
        """

        def casts_to_int(val):
            try:
                int(val)
            except ValueError:
                return False
            else:
                return True

        position = Position(self.line_index, self.column_index - len(self.register) - offset, document=self.model_code)

        if self.register == " " or not self.register:
            token = NullToken()
        elif self.register == ";":
            token = Terminate(self.register, position)
            self.register = ""
            self.current_tokens.append(token)
            if self.current_tokens and len(self.bracket_stack) == 0:
                self.scanned_lines.append(self.current_tokens)
                self.current_tokens = []
            return

        elif casts_to_int(self.register):
            token = IntLiteral(self.register, position)
        elif realnum.match(self.register):
            token = RealLiteral(self.register, position)
        elif self.register in special_characters:
            token = Special(self.register, position)
        elif self.register in operator_strings:
            token = Operator(self.register, position)
        else:
            if self.register.isidentifier():
                token = Identifier(self.register, position)
            else:
                self.raise_error(f"Failed to handle reduce of {self.register}")
                token = NullToken()

        self.register = ""
        if not isinstance(token, NullToken):
            self.current_tokens.append(token)

    # The following functions are scanner states

    def delimeter_state(self):
        bracket_dict = {")": "(", "}": "{", "]": "["}
        if self.current_char in bracket_dict.values():
            self.bracket_stack.append(self.current_char)
        elif self.current_char in bracket_dict.keys():
            if len(self.bracket_stack) == 0:
                self.raise_error(f"Found unmatching brackets {self.current_char}")
            elif self.bracket_stack.pop() != bracket_dict[self.current_char]:
                self.raise_error(f"Found unmatching brackets {self.current_char}")
        self.reduce_register(offset=0)
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
                # - is allowed right after 'e'
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
        self.reduce_register(offset=0)
        self.current_state = self.default_state
