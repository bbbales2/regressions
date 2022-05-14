from collections import namedtuple
from dataclasses import dataclass
import logging
import json
import jsonrpcserver
import jsonrpcclient
import re
import sys

from typing import TextIO, Dict, List

from rat import ast, scanner, compiler
from rat.scanner import Scanner
from rat.parser import Parser

# https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#baseProtocol
def read_json_rpc(file: TextIO):
    content_length = None
    content_type = None

    while True:
        line = file.readline().strip()
        if len(line) == 0:
            break
        cmd, string_value = line.split(":")

        if cmd == "Content-Length":
            content_length = int(string_value)
        elif cmd == "Content-Type":
            content_type = string_value
        else:
            raise Exception("Unrecognized command!")

    if content_length is None:
        raise Exception("Content length never set!")

    content = file.read(content_length)
    logging.debug(f"Reading in message: {content}")

    return content


# https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#baseProtocol
def write_json_rpc(string_obj, file: TextIO):
    output_string = f"Content-Length:{len(string_obj)}\r\n\r\n{string_obj}"
    logging.debug(f"Writing out message: {repr(output_string)}")
    file.write(output_string)
    file.flush()


# https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_publishDiagnostics
def publishDiagnostics(uri, diagnostics):
    message_out = jsonrpcclient.notification_json("textDocument/publishDiagnostics", params={"uri": uri, "diagnostics": diagnostics})
    write_json_rpc(message_out, sys.stdout)


@dataclass
class Document:
    uri: str
    version: int
    text: str


def validateDocument(document: Document):
    diagnostics = []
    # scanned_lines = Scanner(model_string).scan()
    # for scanned_line in scanned_lines:
    #     tokens = Parser(scanned_line, [], model_string).statement()

    #     for token in tokens:
    #         match token:
    #             case ast.Param():
    #                 diagnostics.append(
    #                     {
    #                         "range": {
    #                             "start": {"line": token.range.start.line, "character": token.range.start.col},
    #                             "end": {"line": token.range.end.line, "character": token.range.end.col},
    #                         },
    #                         "message": "It's a parameter!",
    #                     }
    #                 )
    publishDiagnostics(document.uri, diagnostics)


class LanguageServer:
    documents: Dict[str, Document]

    def __init__(self):
        self.documents = {}

    # https://microsoft.github.io//language-server-protocol/specifications/lsp/3.17/specification/#initialize
    def initialize(self, processId, rootUri, capabilities, **kwargs):
        return jsonrpcserver.Success(
            {
                "capabilities": {
                    # 1 means this server requires the full document to be synced -- no incremental updates
                    "textDocumentSync": 1,
                    # "documentSymbolProvider": True,
                    "semanticTokensProvider": {
                        "legend": {
                            "tokenTypes": [
                                "data",
                                "parameter",
                            ],
                            "tokenModifiers": [
                                "primary",
                                "secondary"
                            ]
                        },
                        "range": False,
                        "full": True,
                    },
                }
            }
        )

    # https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#semanticTokens_fullRequest
    def semantic_tokens(self, textDocument):
        uri: str = textDocument["uri"]

        document = self.documents[uri]

        model_string = document.text

        # Read in the data hints
        data_hints_string = ""
        for line in model_string.split("\n"):
            clean_line = line.strip()

            if clean_line == "":
                continue

            match = re.match("#!(.*)", clean_line)
            if match is None:
                break
            
            logging.debug(clean_line)
            logging.debug(match.group(1))
            data_hints_string += match.group(1)
        
        logging.debug(f"Parsing {data_hints_string}")
        
        try:
            data_hints = json.loads(data_hints_string)
        except Exception:
            raise Exception("Error parsing json")

        if not isinstance(data_hints, dict):
            if isinstance(data_hints, list):
                data_hints = { "default" : data_hints }
            else:
                raise Exception("Data hints must be a dict or array")

        # Figure out all data columns
        data_names = set()
        for columns in data_hints.values():
            data_names.update(columns)

        parsed_lines = []
        scanned_lines = Scanner(model_string).scan()
        for scanned_line in scanned_lines:
            parsed_lines.append(Parser(scanned_line, data_names, model_string).statement())
        
        expr_tree_list = compiler.add_primary_information_to_ast(parsed_lines)

        @dataclass
        class VariableReference:
            name : str
            is_data : bool
            associated_with_primary_dataframe : bool
            line : int
            col : int
            length : int

        # Find and classify all variable references in the program
        variable_references : List[VariableReference] = []
        for top_expr in expr_tree_list:
            primary_dataframe_name = None
            for primeable_symbol in ast.search_tree(top_expr, ast.PrimeableExpr):
                if primeable_symbol.prime == True:
                    primary_name = primeable_symbol.name
                    match primeable_symbol:
                        case ast.Data():
                            for dataframe_name, columns in data_hints.items():
                                if primary_name in columns:
                                    primary_dataframe_name = dataframe_name

            primary_columns = data_hints[primary_dataframe_name] if primary_dataframe_name else None
            logging.debug(f"primary variable: {primary_name}, primary_columns: {primary_columns}")
            
            for primeable_symbol in ast.search_tree(top_expr, ast.PrimeableExpr):
                primeable_name = primeable_symbol.name
                associated_with_primary_dataframe = primeable_name == primary_name
                is_data = False
                match primeable_symbol:
                    case ast.Data():
                        if primary_dataframe_name is not None:
                            if primeable_name in data_hints[primary_dataframe_name]:
                                associated_with_primary_dataframe = True
                        is_data = True
                variable_references.append(VariableReference(
                    name = primeable_name,
                    is_data = is_data,
                    associated_with_primary_dataframe = associated_with_primary_dataframe,
                    line = primeable_symbol.range.start.line,
                    col = primeable_symbol.range.start.col,
                    length = max(1, primeable_symbol.range.end.col - primeable_symbol.range.start.col)
                ))

        # Re-code the information for the LSP format
        variable_references = sorted(variable_references, key = lambda x : (x.line, x.col))
        logging.debug(variable_references)
        data = []
        previous_line = 0
        previous_char = 0
        for reference in variable_references:
            type_as_int = 0 if reference.is_data else 1
            modifier_as_int = 1 << (0 if reference.associated_with_primary_dataframe else 1)

            if reference.line != previous_line:
                delta_line = reference.line - previous_line
                delta_start_char = reference.col
            else:
                delta_line = 0
                delta_start_char = reference.col - previous_char

            length = reference.length
            data.extend([delta_line, delta_start_char, length, type_as_int, modifier_as_int])
            previous_line = reference.line
            previous_char = reference.col

        return jsonrpcserver.Success({"data": data})

    # https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_didOpen
    def did_open(self, textDocument):
        document = Document(uri=textDocument["uri"], version=textDocument["version"], text=textDocument["text"])

        self.documents[document.uri] = document

        validateDocument(document)

        return jsonrpcserver.Success()

    # https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_didChange
    def did_change(self, textDocument, contentChanges):
        if len(contentChanges) != 1:
            raise Exception("There should be exactly one change. textDocumentSync should be for full document only, not incremental")
        uri = textDocument["uri"]
        incoming_document = Document(uri=uri, version=textDocument["version"], text=contentChanges[0])

        if self.documents[uri] < incoming_document:
            validateDocument(incoming_document)
            self.documents[uri] = incoming_document

        return jsonrpcserver.Success()

    # https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_didClose
    def did_close(self, textDocument):
        uri: str = textDocument["uri"]
        del self.documents[uri]

        return jsonrpcserver.Success()
