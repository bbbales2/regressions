"""
This module defines the various types an operator returns.
RealType: real number set, primitive type
IntegerType: integer set, primitive type
NumericType: real and integer
SubscriptSetType: a set of values which construct a subscript. Passed through dataframe columns. Referred as a subscript
set.
SubscriptIndexType: For ordered subscript sets, "SubscriptIndex" denotes the index of a subscript value
within the SubscriptSet
"""
from typing import Dict, Tuple, Type


class BaseType:
    name = "BaseType"


class NumericType(BaseType):
    name = "Numeric"


class RealType(NumericType):
    name = "Real"


class IntegerType(NumericType):
    name = "Integer"


class SubscriptSetType(BaseType):
    name = "SubscriptSet"


class BooleanType(BaseType):
    name = "Boolean"


class TypeOr:
    def __init__(self, *args):
        self.types = args

    def __subclasscheck__(self, subclass):
        for _type in self.types:
            if issubclass(subclass, _type):
                return True
        return False


class TypeCheckError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


def get_output_type(signatures: Dict[Tuple[Type[BaseType], ...], Type[BaseType]], in_sigs: Tuple[Type[BaseType], ...]) -> Type[BaseType]:
    string_output = ""
    for in_sig, out_sig in signatures.items():
        string_output += f"{[x.name for x in in_sig]} -> '{out_sig.name}'\n"
        if all([issubclass(in_sigs[i], in_sig[i]) for i in range(len(in_sigs))]):
            return out_sig

    raise TypeCheckError(f"Input Type signature {[x.name for x in in_sigs]} is invalid.\nValid type signatures:\n\n" + string_output)
