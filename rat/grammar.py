import os
import pathlib

src_dir = pathlib.Path(__file__).parent


def grammar() -> str:
    with open(os.path.join(src_dir, "grammar.ebnf")) as f:
        return f.read()