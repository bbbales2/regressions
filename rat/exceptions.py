from .position_and_range import Range

# TODO: These four errors are basically all the same thing. Probably be possible to simplify them somehow


class TokenizeError(Exception):
    def __init__(self, message: str, range: Range):
        code_string = range.document.split("\n")[range.start.line]
        pointer_string = " " * range.start.col + "^" + "~" * max(0, range.end.col - range.start.col - 1)
        exception_message = f"An error occured while tokenizing string at ({range.start.line}:{range.start.col}):\n{code_string}\n{pointer_string}\n{message}"
        super().__init__(exception_message)


class ParseError(Exception):
    def __init__(self, message, range: Range):
        code_string = range.document.split("\n")[range.start.line]
        pointer_string = " " * range.start.col + "^" + "~" * max(0, range.end.col - range.start.col - 1)
        exception_message = f"An error occured while parsing the expression at ({range.start.line}:{range.start.col}):\n{code_string}\n{pointer_string}\n{message}"
        super().__init__(exception_message)


class CompileError(Exception):
    def __init__(self, message, range: Range = None):
        if range is None:
            exception_message = message
        else:
            code_string = range.document.split("\n")[range.start.line]
            pointer_string = " " * range.start.col + "^" + "~" * max(0, range.end.col - range.start.col - 1)
            exception_message = f"An error occurred while compiling code at ({range.start.line}:{range.start.col}):\n{code_string}\n{pointer_string}\n{message}"
        super().__init__(exception_message)


class CompileWarning(UserWarning):
    def __init__(self, message, range: Range):
        code_string = range.document.split("\n")[range.start.line]
        pointer_string = " " * range.start.col + "^" + "~" * max(0, range.end.col - range.start.col - 1)
        warning_message = f"Warning generated at ({range.start.line}:{range.start.col}):\n{code_string}\n{pointer_string}\n{message}"
        super().__init__(warning_message)


class MergeError(Exception):
    pass
