from .position_and_range import Range, Position
from . import ast2

# TODO: These four errors are basically all the same thing. Probably be possible to simplify them somehow

def get_range(node : ast2.ModelBase):
    return Range(
        Position(node.ast.parseinfo.line, node.ast.parseinfo.pos, node.ast.parseinfo.tokenizer.text),
        Position(node.ast.parseinfo.line, node.ast.parseinfo.endpos, node.ast.parseinfo.tokenizer.text)
    )

class TokenizeError(Exception):
    def __init__(self, message: str, node : ast2.ModelBase):
        range = get_range(node)
        code_string = range.document.split("\n")[range.start.line]
        pointer_string = " " * range.start.col + "^" + "~" * max(0, range.end.col - range.start.col - 1)
        exception_message = f"An error occured while tokenizing string at ({range.start.line}:{range.start.col}):\n{code_string}\n{pointer_string}\n{message}"
        super().__init__(exception_message)


class ParseError(Exception):
    def __init__(self, message, node : ast2.ModelBase):
        range = get_range(node)
        code_string = range.document.split("\n")[range.start.line]
        pointer_string = " " * range.start.col + "^" + "~" * max(0, range.end.col - range.start.col - 1)
        exception_message = f"An error occured while parsing the expression at ({range.start.line}:{range.start.col}):\n{code_string}\n{pointer_string}\n{message}"
        super().__init__(exception_message)


class CompileError(Exception):
    def __init__(self, message, node : ast2.ModelBase = None):
        if node is None:
            exception_message = message
        else:
            range = get_range(node)
            code_string = range.document.split("\n")[range.start.line]
            pointer_string = " " * range.start.col + "^" + "~" * max(0, range.end.col - range.start.col - 1)
            exception_message = f"An error occurred while compiling code at ({range.start.line}:{range.start.col}):\n{code_string}\n{pointer_string}\n{message}"
        super().__init__(exception_message)


class CompileWarning(UserWarning):
    def __init__(self, message, node : ast2.ModelBase):
        range = get_range(node)
        code_string = range.document.split("\n")[range.start.line]
        pointer_string = " " * range.start.col + "^" + "~" * max(0, range.end.col - range.start.col - 1)
        warning_message = f"Warning generated at ({range.start.line}:{range.start.col}):\n{code_string}\n{pointer_string}\n{message}"
        super().__init__(warning_message)


class MergeError(Exception):
    pass
