# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\5962.309fbfe15ce0bdf1.js`

```py
"use strict";
// 使用严格模式，强制执行更严格的JavaScript语法和错误检查

(self.webpackChunk_N_E = self.webpackChunk_N_E || []).push([[5962], {
    85962: function(e, s, t) {
        t.r(s), t.d(s, {
            conf: function() {
                return n
            },
            language: function() {
                return r
            }
        });
        /*!-----------------------------------------------------------------------------
         * 版权所有（c）Microsoft Corporation。保留所有权利。
         * 版本：0.34.1(547870b6881302c5b4ff32173c16d06009e3588f)
         * 以 MIT 许可证发布
         * https://github.com/microsoft/monaco-editor/blob/main/LICENSE.txt
         *-----------------------------------------------------------------------------*/
        
        var n = {
            // 定义注释的格式，这里使用的是以井号(#)开头的单行注释
            comments: {
                lineComment: "#"
            },
            // 定义不同类型的括号及其配对关系
            brackets: [
                ["{", "}"],
                ["[", "]"],
                ["(", ")"]
            ],
            // 定义自动补全括号的规则，包括单引号和双引号的自动补全
            autoClosingPairs: [{
                    open: "'",
                    close: "'",
                    notIn: ["string"]
                },
                {
                    open: '"',
                    close: '"',
                    notIn: ["string"]
                },
                {
                    open: "{",
                    close: "}"
                },
                {
                    open: "[",
                    close: "]"
                },
                {
                    open: "(",
                    close: ")"
                }
            ]
        };

        var r = {
            defaultToken: "",
            tokenPostfix: ".rq",
            // 定义不同类型括号的token类型及其配对
            brackets: [{
                    token: "delimiter.curly",
                    open: "{",
                    close: "}"
                },
                {
                    token: "delimiter.parenthesis",
                    open: "(",
                    close: ")"
                },
                {
                    token: "delimiter.square",
                    open: "[",
                    close: "]"
                },
                {
                    token: "delimiter.angle",
                    open: "<",
                    close: ">"
                }
            ],
            // 定义关键字和内置函数
            keywords: [
                "add", "as", "asc", "ask", "base", "by", "clear", "construct", "copy", "create", "data",
                "delete", "desc", "describe", "distinct", "drop", "false", "filter", "from", "graph", "group",
                "having", "in", "insert", "limit", "load", "minus", "move", "named", "not", "offset", "optional",
                "order", "prefix", "reduced", "select", "service", "silent", "to", "true", "undef", "union",
                "using", "values", "where", "with"
            ],
            // 定义内置函数
            builtinFunctions: [
                "a", "abs", "avg", "bind", "bnode", "bound", "ceil", "coalesce", "concat", "contains", "count",
                "datatype", "day", "encode_for_uri", "exists", "floor", "group_concat", "hours", "if", "iri",
                "isblank", "isiri", "isliteral", "isnumeric", "isuri", "lang", "langmatches", "lcase", "max",
                "md5", "min", "minutes", "month", "now", "rand", "regex", "replace", "round", "sameterm", "sample",
                "seconds", "sha1", "sha256", "sha384", "sha512", "str", "strafter", "strbefore", "strdt", "strends",
                "strlang", "strlen", "strstarts", "struuid", "substr", "sum", "timezone", "tz", "ucase", "uri", "uuid",
                "year"
            ],
            ignoreCase: true,
            // 定义代码的词法分析规则
            tokenizer: {
                root: [
                    [/<[^\s\u00a0>]*>?/, "tag"], // 匹配HTML标签
                    { include: "@strings" }, // 引入字符串的词法规则
                    [/#.*/, "comment"], // 匹配以#开头的注释
                    [/[{}()\[\]]/, "@brackets"], // 匹配各种括号
                    [/[;,.]/, "delimiter"], // 匹配分号、逗号和句号
                    [/[$?]?[_\w\d]+/, {
                        // 匹配变量名和关键字
                        cases: {
                            "@keywords": { token: "keyword" },
                            "@builtinFunctions": { token: "predefined.sql" },
                            "@default": "identifier"
                        }
                    }],
                    [/\^\^/, "operator.sql"], // 匹配^^运算符
                    [/\^[*+\-<>=&|^\/!?]*/, "operator.sql"], // 匹配其他运算符
                    [/[*+\-<>=&|\/!?]/, "operator.sql"], // 匹配其他运算符
                    [/@[a-z\d\-]*/, "metatag.html"], // 匹配HTML元标签
                    [/\s+/, "white"] // 匹配空白字符
                ],
                // 定义字符串的词法分析规则
                strings: [
                    [/'([^'\\]|\\.)*$/, "string.invalid"],
                    [/'$/, "string.sql", "@pop"],
                    [/'/, "string.sql", "@stringBody"],
                    [/"/, "string.invalid"],
                    [/"/, "string.sql", "@pop"],
                    [/"/, "string.sql", "@dblStringBody"]
                ],
                // 定义单引号字符串的内容分析规则
                stringBody: [
                    [/[^\']*+/, "string.sql"],
                    [/'/, "string.sql", "@pop"]
                ],
                // 定义双引号字符串的内容分析规则
                dblStringBody: [
                    [/[^\"]*+/, "string.sql"],
                    [/"/, "string.sql", "@pop"]
                ]
            }
        };
    }
}]);
```