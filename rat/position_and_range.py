
from dataclasses import dataclass


@dataclass
class Position:
    """Position describes a location in a document"""

    line : int
    "Line in document"
    col : int
    "Column of line"
    document : str = None
    "Document as string"


class Range:
    """Range describes a range of text in a document"""
    start : Position
    end : Position

    def __init__(self, start : Position, end : Position = None, length : int = None):
        """
        Start should be start document position and end should be the end document position.

        If end is not provided, length should be the length in characters of the Range.
        """
        self.start = start
        if end is None:
            if length is None:
                raise Exception("At least one of end or length must be provided")
            else:
                self.end = Position(start.line, start.col + length, document = self.start.document)
        else:
            if start.document is not end.document:
                raise Exception("Internal error. Range documents must match")
            self.end = end
    
    @property
    def document(self) -> str:
        return self.start.document
