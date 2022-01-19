"""
This module defines the various types an operator returns.
RealType: real number set
IntegerType: integer set
SubscriptType: Literal identifying subscript column name. They are distinct from strings.
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


class SubscriptType(BaseType):
    name = "Subscript"


class TypeCheckError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


def get_output_type(signatures: Dict[Tuple[Type[BaseType], ...], Type[BaseType]], in_sigs: Tuple[Type[BaseType], ...]) -> Type[BaseType]:
    for in_sig, out_sig in signatures.items():
        if all([issubclass(in_sigs[i], in_sig[i]) for i in range(len(in_sigs))]):
            return out_sig

    raise TypeCheckError(f"Input Type signature {[x.name for x in in_sigs]} is invalid.")
