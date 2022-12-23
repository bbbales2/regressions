from dataclasses import dataclass
import numpy
from . import ast
from typing import Dict, Iterator, Tuple


@dataclass(frozen=True)
class TraceRecord:
    name: str
    array: numpy.ndarray


class TraceTable:
    subscript_dict: Dict[ast.ModelBase, TraceRecord]

    def __init__(self):
        self.subscript_dict = {}

    def __contains__(self, node: ast.ModelBase) -> bool:
        return node in self.subscript_dict

    def __getitem__(self, node: ast.ModelBase) -> TraceRecord:
        return self.subscript_dict[node]

    def insert(self, node: ast.ModelBase, array: numpy.ndarray):
        record = TraceRecord(f"subscript_{len(self.subscript_dict)}", array)
        self.subscript_dict[node] = record
        return record

    def __iter__(self) -> Iterator[ast.ModelBase]:
        for node in self.subscript_dict:
            yield node

    def items(self) -> Iterator[Tuple[ast.ModelBase, TraceRecord]]:
        for node, record in self.subscript_dict.items():
            yield node, record

    def values(self) -> Iterator[TraceRecord]:
        for record in self.subscript_dict.values():
            yield record
