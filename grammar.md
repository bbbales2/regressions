### My lil draft of the formal syntax for *regressions*

We are going to follow stan's grammar, but alot of stuff stripped out

#### Tokens

1. `Identifier` Any variable/function/distribution/data
2. `RealLiteral` Real constant
3. `IntLiteral` Integer constant
4. `Operator`  Any operator
5. `Terminate`  `;` end of statement
6. `Special` Special semantic characters `["(", ")", ",", "[", "]", "~"]`
    - Sequencing control and order of operations `"(", ")"`
    - expression lists(called `expressions`) `","`
    - Indexing `"[", "]"`
    - Sampling `"~"`
#### BNF

```

expressions ::= expression % ','
expression ::= expression infixOps expression
             | prefixOp expression
             | identifier '[' expressions ']'
             | real_literal
             | integer_literal
             | '(' expression ')'

; binary operations, just arithmetic and logical
infixOps ::= ("+", "-", "*", "/", "%",
              "||", "&&", "==", "!=", "<", "<=", ">", ">=")

prefixOps ::= ("!", "-")

; didn't implement control flow yet
statement ::= atomic_statement

Distribution ::= ("normal")

lhs ::= identifier ('[' expressions ']')*
assignmentOp ::= ("=", "+=", "-=", "*=", "/=")

; statements, assume assignments are allowed
atomic_statement ::=  lhs assignmentOp expression ';'
   | expression '~' Distribution '(' expressions ')' ';'
   | ';'
```