from .position_and_range import Range, Position
from . import ast


def find_line_and_relative_position(text : str, target_position : int) -> int:
    """
    Reinterpret target_position (given as an absolute offset in a block of text)
    as a line number and relative offset
    """
    if target_position > len(text):
        raise ValueError(f"Internal error: target_position {target_position} must be less than text length {len(text)}")

    line_start = 0
    for line_number, line in enumerate(text.split("\n")):
        line_end = line_start + len(line)
        if line_end >= target_position:
            return line_number, target_position - line_start
        line_start = line_end + 1
    else:
        raise ValueError("Internal error: Failed to find line number")


def get_range(node: ast.ModelBase):
    text = node.ast.parseinfo.tokenizer.text
    start_pos = node.ast.parseinfo.pos
    end_pos = node.ast.parseinfo.endpos

    start_line, relative_start_pos = find_line_and_relative_position(text, start_pos)
    end_line, relative_end_pos = find_line_and_relative_position(text, end_pos)

    return Range(Position(start_line, relative_start_pos, text), Position(end_line, relative_end_pos, text))


class AstException(Exception):
    def __init__(self, operation_message: str, message: str, node: ast.ModelBase = None):
        if node is None:
            exception_message = f"An error occurred while {operation_message}\n{message}"
        else:
            node_range = get_range(node)
            code_string = node_range.document.split("\n")[node_range.start.line]
            pointer_string = " " * node_range.start.col + "^" + "~" * max(0, node_range.end.col - node_range.start.col - 1)
            exception_message = (
                f"An error occurred while {operation_message} at"
                f" ({node_range.start.line}:{node_range.start.col}):\n{code_string}\n{pointer_string}\n{message}"
            )
        super().__init__(exception_message)


class CompileError(AstException):
    def __init__(self, message: str, node: ast.ModelBase = None):
        super().__init__("compiling code", message, node)
