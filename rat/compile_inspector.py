from typing import Tuple, Iterable, Dict
import pandas
from . import ast
from .trace_table import TraceTable, TraceRecord
from .trace_table import TraceTable
from .variable_table import VariableTable, AssignedVariableRecord, SampledVariableRecord, ConstantVariableRecord, DynamicVariableRecord
from dataclasses import dataclass, field
import numpy
from .compiler import *
from .walker import RatWalker


class bcolors:
    BLACK = '\033[90m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


class VarTypeColor:
    DATA = bcolors.MAGENTA
    PARAM = bcolors.BLUE
    ASSIGNED_PARAM = bcolors.CYAN

    @staticmethod
    def data(string):
        return VarTypeColor.DATA + string + bcolors.RESET

    @staticmethod
    def param(string):
        return VarTypeColor.PARAM + string + bcolors.RESET

    @staticmethod
    def assigned_param(string):
        return VarTypeColor.ASSIGNED_PARAM + string + bcolors.RESET


@dataclass
class TraceInfoWalker(RatWalker):
    trace_table: TraceTable
    variable_table: VariableTable

    def walk_Variable(self, node: ast.Variable):
        if node in self.trace_table:
            record = self.trace_table[node]
        else:
            record = None

        match self.variable_table[node.name]:
            case AssignedVariableRecord():
                var_color = VarTypeColor.ASSIGNED_PARAM
            case SampledVariableRecord():
                var_color = VarTypeColor.PARAM
            case ConstantVariableRecord():
                var_color = VarTypeColor.DATA

        if record:
            print("\t", var_color, node.name, bcolors.RESET + " array dim: ", ",".join(str(x) for x in record.array.shape), " - ", record.array)
        else:
            print("\t", var_color, node.name, bcolors.RESET + " array dim: ", bcolors.RED, "Not present in trace table!")


class RatCompileInspector(RatCompiler):
    def __init__(self, data: Dict[str, pandas.DataFrame], program: ast.Program, max_trace_iterations: int):
        self.data = data
        self.program = program
        super().__init__(data, program, max_trace_iterations)

    def print_inspect_info(self):
        print(f"Variable Table Information({VarTypeColor.data('data')}, {VarTypeColor.param('param')}, {VarTypeColor.assigned_param('assigned param')}):")
        for variable_name, value in self.variable_table.variable_dict.items():
            match value:
                case AssignedVariableRecord():
                    print(VarTypeColor.ASSIGNED_PARAM, variable_name, bcolors.RESET + "-", value.subscripts)
                case SampledVariableRecord():
                    print(VarTypeColor.PARAM, variable_name, bcolors.RESET + "-", value.subscripts)
                case ConstantVariableRecord():
                    print(VarTypeColor.DATA, variable_name, bcolors.RESET + "-", value.subscripts)

        print("Per-statement Trace Information")
        for index, statement in enumerate(self.program.statements):
            # print(statement.parseinfo.tokenizer.line_info().text)
            # This won't work because tatsu reuses LineInfo for parsing the next statement
            print(f"Statement {index}")
            TraceInfoWalker(self.trace_table, self.variable_table).walk(statement)
