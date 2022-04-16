class CompileError(Exception):
    def __init__(self, message, code_string: str = "", line_num: int = -1, column_num: int = -1):
        code_string = code_string.split("\n")[line_num] if code_string else ""
        if code_string:
            exception_message = f"An error occurred while compiling the following line({line_num}:{column_num}):\n{code_string}\n{' ' * column_num + '^'}\n{message}"
        else:
            exception_message = f"An error occurred during compilation:\n{message}"
        super().__init__(exception_message)


class CompileWarning(UserWarning):
    def __init__(self, message, code_string: str = "", line_num: int = -1, column_num: int = -1):
        code_string = code_string.split("\n")[line_num] if code_string else ""
        if code_string:
            warning_message = f"Compilation warning({line_num}:{column_num}):\n{code_string}\n{' ' * column_num + '^'}\n{message}"
        else:
            warning_message = f"Compilation warning:\n{message}"
        super().__init__(warning_message)
