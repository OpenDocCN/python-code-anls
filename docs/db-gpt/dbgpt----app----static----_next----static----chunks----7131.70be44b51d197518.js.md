# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\7131.70be44b51d197518.js`

```py
"use strict";
// 使用严格模式，确保代码在更严格的条件下执行，提高代码质量和安全性

(self.webpackChunk_N_E = self.webpackChunk_N_E || []).push([[7131], {
    47131: function(e, n, t) {
        t.r(n), t.d(n, {
            conf: function() {
                return r
            },
            language: function() {
                return i
            }
        });
        // 定义两个导出函数：conf 和 language

        var o = `[ \\t\\r\\n]`;
        // 定义正则表达式 o 匹配空格、制表符、回车和换行符

        var r = {
            comments: {
                lineComment: "//",
                blockComment: ["/*", "*/"]
            },
            brackets: [
                ["{", "}"],
                ["[", "]"],
                ["(", ")"]
            ],
            surroundingPairs: [{
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
                },
                {
                    open: "'",
                    close: "'"
                },
                {
                    open: "'''",
                    close: "'''"
                }
            ],
            autoClosingPairs: [{
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
                },
                {
                    open: "'",
                    close: "'",
                    notIn: ["string", "comment"]
                },
                {
                    open: "'''",
                    close: "'''",
                    notIn: ["string", "comment"]
                }
            ],
            autoCloseBefore: ":.,=}"
        };
        // 定义语言配置对象 r，包括注释类型、括号匹配、自动闭合和自动关闭前字符

        var i = {
            defaultToken: "",
            tokenPostfix: ".bicep",
            brackets: [{
                    open: "{",
                    close: "}",
                    token: "delimiter.curly"
                },
                {
                    open: "[",
                    close: "]",
                    token: "delimiter.square"
                },
                {
                    open: "(",
                    close: ")",
                    token: "delimiter.parenthesis"
                }
            ],
            symbols: /[=><!~?:&|+\-*/^%]+/,
            keywords: [
                "targetScope",
                "resource",
                "module",
                "param",
                "var",
                "output",
                "for",
                "in",
                "if",
                "existing"
            ],
            namedLiterals: ["true", "false", "null"],
            escapes: "\\\\(u{[0-9A-Fa-f]+}|n|r|t|\\\\|'|\\${)",
            tokenizer: {
                root: [{
                        include: "@expression"
                    },
                    {
                        include: "@whitespace"
                    }
                ],
                stringVerbatim: [{
                        regex: "('|'')[^']",
                        action: {
                            token: "string"
                        }
                    },
                    {
                        regex: "'''",
                        action: {
                            token: "string.quote",
                            next: "@pop"
                        }
                    }
                ],
                stringLiteral: [{
                        regex: "\\${",
                        action: {
                            token: "delimiter.bracket",
                            next: "@bracketCounting"
                        }
                    },
                    {
                        regex: "[^\\\\'$]+",
                        action: {
                            token: "string"
                        }
                    },
                    {
                        regex: "@escapes",
                        action: {
                            token: "string.escape"
                        }
                    },
                    {
                        regex: "\\\\.",
                        action: {
                            token: "string.escape.invalid"
                        }
                    },
                    {
                        regex: "'",
                        action: {
                            token: "string",
                            next: "@pop"
                        }
                    }
                ],
                bracketCounting: [{
                        regex: "{",
                        action: {
                            token: "delimiter.bracket",
                            next: "@bracketCounting"
                        }
                    },
                    {
                        regex: "}",
                        action: {
                            token: "delimiter.bracket",
                            next: "@pop"
                        }
                    },
                    {
                        include: "expression"
                    }
                ],
                comment: [{
                        regex: "[^*]+",
                        action: {
                            token: "comment"
                        }
                    },
                    {
                        regex: "\\*\\/",
                        action: {
                            token: "comment",
                            next: "@pop"
                        }
                    },
                    {
                        regex: "[\\/*]",
                        action: {
                            token: "comment"
                        }
                    }
                ],
                whitespace: [{
                        regex: o
                    },
                    {
                        regex: "\\/\\*",
                        action: {
                            token: "comment",
                            next: "@comment"
                        }
                    },
                    {
                        regex: "\\/\\/.*$",
                        action: {
                            token: "comment"
                        }
                    }
                ],
                expression: [{
                        regex: "'''",
                        action: {
                            token: "string.quote",
                            next: "@stringVerbatim"
                        }
                    },
                    {
                        regex: "'",
                        action: {
                            token: "string.quote",
                            next: "@stringLiteral"
                        }
                    },
                    {
                        regex: "[0-9]+",
                        action: {
                            token: "number"
                        }
                    },
                    {
                        regex: "\\b[_a-zA-Z][_a-zA-Z0-9]*\\b",
                        action: {
                            cases: {
                                "@keywords": {
                                    token: "keyword"
                                },
                                "@namedLiterals": {
                                    token: "keyword"
                                },
                                "@default": {
                                    token: "identifier"
                                }
                            }
                        }
                    }
                ]
            }
        };
        // 定义语言配置对象 i，包括默认标记、括号、符号、关键字、转义字符和分词器规则
    }
}]);
```