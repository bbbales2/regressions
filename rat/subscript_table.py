from dataclasses import dataclass
import numpy
from . import ast
from typing import Dict, Iterator, Tuple


@dataclass(frozen=True)
class SubscriptRecord:
    name: str
    array: numpy.ndarray


class SubscriptTable:
    subscript_dict: Dict[ast.ModelBase, SubscriptRecord]
    record_count: int

    def __init__(self):
        self.subscript_dict = {}
        self.record_count = 0

    def __contains__(self, node: ast.ModelBase) -> bool:
        return node in self.subscript_dict

    def __getitem__(self, node: ast.ModelBase) -> SubscriptRecord:
        return self.subscript_dict[node]

    def insert(self, node: ast.ModelBase, array: numpy.ndarray):
        record = SubscriptRecord(f"subscript_{self.record_count}", array)
        self.subscript_dict[node] = record
        self.record_count += 1
        return record

    def __iter__(self) -> Iterator[ast.ModelBase]:
        for node in self.subscript_dict:
            yield node

    def items(self) -> Iterator[Tuple[ast.ModelBase, SubscriptRecord]]:
        for node, record in self.subscript_dict.items():
            yield node, record

    def values(self) -> Iterator[SubscriptRecord]:
        for record in self.subscript_dict.values():
            yield record
