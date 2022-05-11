from dataclasses import dataclass
import logging
import jsonrpcserver
import jsonrpcclient
import sys

from typing import TextIO, Dict

from rat import ast, scanner
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
                    # Does the server provide symbols
                    # "documentSymbolProvider": True,
                    # The semantic tokens will be used to communicate with the editor what
                    #
                    "semanticTokensProvider": {
                        "legend": {"tokenTypes": ["type", "variable", "operator", "enum", "number"], "tokenModifiers": []},
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

        data = []
        scanned_lines = Scanner(model_string).scan()
        previous_line = 0
        previous_char = 0
        for scanned_line in scanned_lines:
            for token in scanned_line:
                type_as_int = None
                match token:
                    case scanner.Identifier():
                        type_as_int = 1
                    case scanner.IntLiteral() | scanner.RealLiteral():
                        type_as_int = 4
                    case _:
                        type_as_int = 2

                if type_as_int is not None:
                    if token.range.start.line != previous_line:
                        delta_line = token.range.start.line - previous_line
                        delta_start_char = token.range.start.col
                    else:
                        delta_line = 0
                        delta_start_char = token.range.start.col - previous_char
                    length = token.range.end.col - token.range.start.col
                    data.extend([delta_line, delta_start_char, length, type_as_int, 0])
                    previous_line = token.range.start.line
                    previous_char = token.range.start.col

        return jsonrpcserver.Success({"data": data})

    # https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_didOpen
    def did_open(self, textDocument):
        document = Document(uri=textDocument["uri"], version=textDocument["version"], text=textDocument["text"])

        self.documents[document.uri] = document

        validateDocument(document)

        return jsonrpcserver.Success()

    # https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_didChange
    def did_change(self, textDocument, contentChanges):
        incoming_document = Document(uri=textDocument["uri"], version=textDocument["version"], text=textDocument["text"])
        uri = incoming_document.uri
        if self.documents[uri] < incoming_document:
            if len(contentChanges) != 1:
                raise Exception("There should be exactly one change. textDocumentSync should be for full document only, not incremental")
            self.documents[uri] = incoming_document
            validateDocument(incoming_document)

        return jsonrpcserver.Success()

    # https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_didClose
    def did_close(self, textDocument):
        uri: str = textDocument["uri"]
        del self.documents[uri]

        return jsonrpcserver.Success()
