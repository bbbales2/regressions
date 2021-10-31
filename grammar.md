### My lil draft of the formal syntax for *rat*

We are going to follow stan's [grammar](https://mc-stan.org/docs/2_18/reference-manual/bnf-grammars.html), but alot of stuff stripped out 

1. Scanner
    - Input: a string
    - Output: a list of Tokens
2. Parser
    - Input: a list of Tokens
    - Output: an expression tree
3. Compiler
    - Input: a list of expression trees
    - Output: a python function representing the model
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

##### Note on `<` and `>`
 You can see that angled brackets can mean `greater than` or `less than` when interpreted as operators,
but a complete set of them(`<lower=...>`) also represent the constraint region when in the left hand side of a
statement. In the scanner, which does not take into account semantics, they are just classified as an `Operator`
token. 

This means the parser will fail when evaluating constraints in a standard ll(k) approach because it will attempt 
to try and evaluate a non-existant rhs expression for the `>` operator. Therefore, the one and only artificial modification 
I have made to the parser when evaluating constraints is to identify the angled brackets, and convert then from 
`Operator` to `Special` tokens, which makes the parser recognize them as delimiters instead of operators.   

#### BNF

```

expressions ::= expression % ','
             | lag(expression, expression) % ','
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

unaryFunction ::= ("exp", "abs", "floor", "ceil", "round")

; didn't implement control flow yet
statement ::= atomic_statement

; constraints
constraint ::= ?('<' constraint_range '>')

constraint_range ::= 'lower' '=' expressioni ',' 'upper' = expressoin
        | 'lower' '=' expression
        | 'upper' '=' expression

Distribution ::= ("normal")

lhs ::= identifier constraint ('[' expressions ']')*
assignmentOp ::= ("=", "+=", "-=", "*=", "/=")

; statements, assume assignments are allowed
atomic_statement ::=  lhs assignmentOp expression ';'
   | expression '~' Distribution '(' expressions ')' ';'
   | ';'
```