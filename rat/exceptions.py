from .position_and_range import Range, Position
from . import ast

# TODO: These four errors are basically all the same thing. Probably be possible to simplify them somehow


def get_range(node: ast.ModelBase):
    return Range(
        Position(node.ast.parseinfo.line, node.ast.parseinfo.pos, node.ast.parseinfo.tokenizer.text),
        Position(node.ast.parseinfo.line, node.ast.parseinfo.endpos, node.ast.parseinfo.tokenizer.text),
    )

class AstException(Exception):
    def __init__(self, operation_message: str, message: str, node: ast.ModelBase = None):
        if node is None:
            exception_message = f"An error occurred while {operation_message}\n{message}"
        else:
            range = get_range(node)
            code_string = range.document.split("\n")[range.start.line]
            pointer_string = " " * range.start.col + "^" + "~" * max(0, range.end.col - range.start.col - 1)
            exception_message = f"An error occured while {operation_message} at ({range.start.line}:{range.start.col}):\n{code_string}\n{pointer_string}\n{message}"
        super().__init__(exception_message)
class CompileError(AstException):
    def __init__(self, message: str, node: ast.ModelBase = None):
        super().__init__("compiling code", message, node)

