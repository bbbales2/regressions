#!/usr/bin/env python

# CAVEAT UTILITOR
#
# This file was automatically generated by TatSu.
#
#    https://pypi.python.org/pypi/tatsu/
#
# Any changes you make to it will be overwritten the next time
# the file is generated.

from __future__ import annotations

import sys

from tatsu.buffering import Buffer
from tatsu.parsing import Parser
from tatsu.parsing import tatsumasu
from tatsu.parsing import leftrec, nomemo, isname # noqa
from tatsu.infos import ParserConfig
from tatsu.util import re, generic_main  # noqa


KEYWORDS = {
    'let',
    'in',
    'end',
    'fun',
}  # type: ignore


class RatBuffer(Buffer):
    def __init__(self, text, /, config: ParserConfig = None, **settings):
        config = ParserConfig.new(
            config,
            owner=self,
            whitespace=None,
            nameguard=None,
            comments_re=None,
            eol_comments_re='#.*?$',
            ignorecase=False,
            namechars='',
            parseinfo=True,
        )
        config = config.replace(**settings)
        super().__init__(text, config=config)


class RatParser(Parser):
    def __init__(self, /, config: ParserConfig = None, **settings):
        config = ParserConfig.new(
            config,
            owner=self,
            whitespace=None,
            nameguard=None,
            comments_re=None,
            eol_comments_re='#.*?$',
            ignorecase=False,
            namechars='',
            parseinfo=True,
            keywords=KEYWORDS,
            start='start',
        )
        config = config.replace(**settings)
        super().__init__(config=config)

    @tatsumasu('Program')
    def _start_(self):  # noqa

        def block0():
            self._statement_()
            self.add_last_node_to_name('statements')
        self._positive_closure(block0)
        self._check_eof()

        self._define(
            [],
            ['statements']
        )

    @tatsumasu('Statement')
    def _statement_(self):  # noqa
        self._factor_()
        self.name_last_node('left')
        self._cut()
        with self._group():
            with self._choice():
                with self._option():
                    self._token('~')
                with self._option():
                    self._token('=')
                self._error(
                    'expecting one of: '
                    "'=' '~'"
                )
        self.name_last_node('op')
        self._cut()
        self._expression_()
        self.name_last_node('right')
        self._token(';')

        self._define(
            ['left', 'op', 'right'],
            []
        )

    @tatsumasu()
    def _expression_(self):  # noqa
        with self._choice():
            with self._option():
                self._logical_()
            with self._option():
                self._addition_expression_()
            self._error(
                'expecting one of: '
                '<addition> <addition_expression>'
                '<logical> <multiplication_expression>'
            )

    @tatsumasu('Binary')
    def _logical_(self):  # noqa
        self._addition_expression_()
        self.name_last_node('left')
        with self._group():
            with self._choice():
                with self._option():
                    self._token('==')
                with self._option():
                    self._token('<=')
                with self._option():
                    self._token('>=')
                with self._option():
                    self._token('<')
                with self._option():
                    self._token('>')
                self._error(
                    'expecting one of: '
                    "'<' '<=' '==' '>' '>='"
                )
        self.name_last_node('op')
        self._cut()
        self._expression_()
        self.name_last_node('right')

        self._define(
            ['left', 'op', 'right'],
            []
        )

    @tatsumasu()
    def _addition_expression_(self):  # noqa
        with self._choice():
            with self._option():
                self._addition_()
            with self._option():
                self._multiplication_expression_()
            self._error(
                'expecting one of: '
                '<addition> <factor> <multiplication>'
                '<multiplication_expression>'
            )

    @tatsumasu('Binary')
    def _addition_(self):  # noqa
        self._multiplication_expression_()
        self.name_last_node('left')
        with self._group():
            with self._choice():
                with self._option():
                    self._token('+')
                with self._option():
                    self._token('-')
                self._error(
                    'expecting one of: '
                    "'+' '-'"
                )
        self.name_last_node('op')
        self._cut()
        self._addition_expression_()
        self.name_last_node('right')

        self._define(
            ['left', 'op', 'right'],
            []
        )

    @tatsumasu()
    def _multiplication_expression_(self):  # noqa
        with self._choice():
            with self._option():
                self._multiplication_()
            with self._option():
                self._factor_()
            self._error(
                'expecting one of: '
                '<factor> <function_call> <ifelse>'
                '<literal> <multiplication>'
                '<subexpression> <variable>'
            )

    @tatsumasu('Binary')
    def _multiplication_(self):  # noqa
        self._factor_()
        self.name_last_node('left')
        with self._group():
            with self._choice():
                with self._option():
                    self._token('*')
                with self._option():
                    self._token('/')
                self._error(
                    'expecting one of: '
                    "'*' '/'"
                )
        self.name_last_node('op')
        self._cut()
        self._multiplication_expression_()
        self.name_last_node('right')

        self._define(
            ['left', 'op', 'right'],
            []
        )

    @tatsumasu()
    def _factor_(self):  # noqa
        with self._choice():
            with self._option():
                self._ifelse_()
            with self._option():
                self._function_call_()
            with self._option():
                self._subexpression_()
            with self._option():
                self._variable_()
            with self._option():
                self._literal_()
            self._error(
                'expecting one of: '
                "'(' 'ifelse' <function_call>"
                '<identifier> <integer> <literal> <real>'
                '<subexpression> <variable>'
            )

    @tatsumasu('IfElse')
    def _ifelse_(self):  # noqa
        self._token('ifelse')
        self._token('(')
        self._cut()
        self._expression_()
        self.name_last_node('predicate')
        self._token(',')
        self._expression_()
        self.name_last_node('left')
        self._token(',')
        self._expression_()
        self.name_last_node('right')
        self._token(')')

        self._define(
            ['left', 'predicate', 'right'],
            []
        )

    @tatsumasu('FunctionCall')
    def _function_call_(self):  # noqa
        self._identifier_()
        self.name_last_node('name')
        self._token('(')
        with self._optional():
            self._arglist_()
            self.name_last_node('arglist')
        self._token(')')

        self._define(
            ['arglist', 'name'],
            []
        )

    @tatsumasu('Variable')
    def _variable_(self):  # noqa
        self._identifier_()
        self.name_last_node('name')
        with self._optional():
            self._constraints_()
            self.name_last_node('constraints')
        with self._optional():
            self._token('[')
            self._arglist_()
            self.name_last_node('arglist')
            self._token(']')

            self._define(
                ['arglist'],
                []
            )
        with self._optional():
            self._token("'")
            self.name_last_node('prime')

        self._define(
            ['arglist', 'constraints', 'name', 'prime'],
            []
        )

    @tatsumasu('Constraints')
    def _constraints_(self):  # noqa
        self._token('<')
        self._constraint_()
        self.name_last_node('left')
        self._cut()
        with self._optional():
            self._token(',')
            self._constraint_()
            self.name_last_node('right')

            self._define(
                ['right'],
                []
            )
        self._token('>')

        self._define(
            ['left', 'right'],
            []
        )

    @tatsumasu()
    def _constraint_(self):  # noqa
        with self._group():
            with self._choice():
                with self._option():
                    self._token('lower')
                with self._option():
                    self._token('upper')
                self._error(
                    'expecting one of: '
                    "'lower' 'upper'"
                )
        self.name_last_node('name')
        self._cut()
        self._token('=')
        self._cut()
        self._literal_()
        self.name_last_node('value')

        self._define(
            ['name', 'value'],
            []
        )

    @tatsumasu()
    def _arglist_(self):  # noqa
        self._expression_()
        self.add_last_node_to_name('@')

        def block1():
            self._token(',')
            self._cut()
            self._expression_()
            self.add_last_node_to_name('@')
        self._closure(block1)

    @tatsumasu()
    def _subexpression_(self):  # noqa
        self._token('(')
        self._cut()
        self._expression_()
        self.name_last_node('@')
        self._token(')')

    @tatsumasu('Literal')
    def _literal_(self):  # noqa
        with self._group():
            with self._choice():
                with self._option():
                    self._real_()
                with self._option():
                    self._integer_()
                self._error(
                    'expecting one of: '
                    '<integer> <real> [-]?[0-9]+\\.[0-9]*'
                    '[-]?[0-9]+\\.[0-9]*e[-+]?[0-9]+ [-]?\\d+'
                )
        self.name_last_node('value')

    @tatsumasu('int')
    def _integer_(self):  # noqa
        self._pattern('[-]?\\d+')

    @tatsumasu('float')
    def _real_(self):  # noqa
        with self._choice():
            with self._option():
                self._pattern('[-]?[0-9]+\\.[0-9]*e[-+]?[0-9]+')
            with self._option():
                self._pattern('[-]?[0-9]+\\.[0-9]*')
            self._error(
                'expecting one of: '
                '[-]?[0-9]+\\.[0-9]*'
                '[-]?[0-9]+\\.[0-9]*e[-+]?[0-9]+'
            )

    @tatsumasu('str')
    @isname
    def _identifier_(self):  # noqa
        self._pattern('[a-zA-Z_][a-zA-Z0-9_]*')


class RatSemantics:
    def start(self, ast):  # noqa
        return ast

    def statement(self, ast):  # noqa
        return ast

    def expression(self, ast):  # noqa
        return ast

    def logical(self, ast):  # noqa
        return ast

    def addition_expression(self, ast):  # noqa
        return ast

    def addition(self, ast):  # noqa
        return ast

    def multiplication_expression(self, ast):  # noqa
        return ast

    def multiplication(self, ast):  # noqa
        return ast

    def factor(self, ast):  # noqa
        return ast

    def ifelse(self, ast):  # noqa
        return ast

    def function_call(self, ast):  # noqa
        return ast

    def variable(self, ast):  # noqa
        return ast

    def constraints(self, ast):  # noqa
        return ast

    def constraint(self, ast):  # noqa
        return ast

    def arglist(self, ast):  # noqa
        return ast

    def subexpression(self, ast):  # noqa
        return ast

    def literal(self, ast):  # noqa
        return ast

    def integer(self, ast):  # noqa
        return ast

    def real(self, ast):  # noqa
        return ast

    def identifier(self, ast):  # noqa
        return ast


def main(filename, **kwargs):
    if not filename or filename == '-':
        text = sys.stdin.read()
    else:
        with open(filename) as f:
            text = f.read()
    parser = RatParser()
    return parser.parse(
        text,
        filename=filename,
        **kwargs
    )


if __name__ == '__main__':
    import json
    from tatsu.util import asjson

    ast = generic_main(main, RatParser, name='Rat')
    data = asjson(ast)
    print(json.dumps(data, indent=2))
