# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\6489.24995c12818670f9.js`

```py
"use strict";
// 使用严格模式，确保代码执行在更严格的语法和错误检查下

(self.webpackChunk_N_E = self.webpackChunk_N_E || []).push([[6489], {
    66489: function(e, n, t) {
        // 定义模块函数，接受 e, n, t 作为参数
        t.r(n), t.d(n, {
            conf: function() {
                return o
            },
            language: function() {
                return s
            }
        });
        /*!-----------------------------------------------------------------------------
         * 版权所有 (c) Microsoft Corporation. 保留所有权利。
         * 版本: 0.34.1(547870b6881302c5b4ff32173c16d06009e3588f)
         * 在 MIT 许可下发布
         * https://github.com/microsoft/monaco-editor/blob/main/LICENSE.txt
         *-----------------------------------------------------------------------------*/
        var o = {
            // 定义 Monaco 编辑器的配置对象
            comments: {
                lineComment: "#"
            },
            brackets: [
                ["{", "}"],
                ["[", "]"],
                ["(", ")"]
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
                    open: '"""',
                    close: '"""',
                    notIn: ["string", "comment"]
                },
                {
                    open: '"',
                    close: '"',
                    notIn: ["string", "comment"]
                }
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
                    open: '"""',
                    close: '"""'
                },
                {
                    open: '"',
                    close: '"'
                }
            ],
            folding: {
                offSide: !0
            }
        };
        var s = {
            // 定义 Monaco 编辑器的语言定义对象
            defaultToken: "invalid",
            tokenPostfix: ".gql",
            keywords: [
                "null", "true", "false", "query", "mutation", "subscription",
                "extend", "schema", "directive", "scalar", "type", "interface",
                "union", "enum", "input", "implements", "fragment", "on"
            ],
            typeKeywords: [
                "Int", "Float", "String", "Boolean", "ID"
            ],
            directiveLocations: [
                "SCHEMA", "SCALAR", "OBJECT", "FIELD_DEFINITION", "ARGUMENT_DEFINITION",
                "INTERFACE", "UNION", "ENUM", "ENUM_VALUE", "INPUT_OBJECT", "INPUT_FIELD_DEFINITION",
                "QUERY", "MUTATION", "SUBSCRIPTION", "FIELD", "FRAGMENT_DEFINITION",
                "FRAGMENT_SPREAD", "INLINE_FRAGMENT", "VARIABLE_DEFINITION"
            ],
            operators: [
                "=", "!", "?", ":", "&", "|"
            ],
            symbols: /[=!?:&|]+/,
            escapes: /\\(?:["\\\/bfnrt]|u[0-9A-Fa-f]{4})/,
            tokenizer: {
                root: [
                    [/[a-z_][\w$]*/, {
                        cases: {
                            "@keywords": "keyword",
                            "@default": "key.identifier"
                        }
                    }],
                    [/\$[\w$]*/, {
                        cases: {
                            "@keywords": "keyword",
                            "@default": "argument.identifier"
                        }
                    }],
                    [/[A-Z][\w\$]*/, {
                        cases: {
                            "@typeKeywords": "keyword",
                            "@default": "type.identifier"
                        }
                    }],
                    { include: "@whitespace" },
                    [/[{}()\[\]]/, "@brackets"],
                    [/@symbols/, {
                        cases: {
                            "@operators": "operator",
                            "@default": ""
                        }
                    }],
                    [/@\s*[a-zA-Z_\$][\w\$]*/, {
                        token: "annotation",
                        log: "annotation token: $0"
                    }],
                    [/\d*\.\d+([eE][\-+]?\d+)?/, "number.float"],
                    [/0[xX][0-9a-fA-F]+/, "number.hex"],
                    [/\d+/, "number"],
                    [/[;,.]/, "delimiter"],
                    [/"""/, {
                        token: "string",
                        next: "@mlstring",
                        nextEmbedded: "markdown"
                    }],
                    [/"([^"\\]|\\.)*$/, "string.invalid"],
                    [/"/, {
                        token: "string.quote",
                        bracket: "@open",
                        next: "@string"
                    }]
                ],
                mlstring: [
                    [/[^"]+/, "string"],
                    ['"""', {
                        token: "string",
                        next: "@pop",
                        nextEmbedded: "@pop"
                    }]
                ],
                string: [
                    [/[^\\"]+/, "string"],
                    [/@escapes/, "string.escape"],
                    [/.\\/, "string.escape.invalid"],
                    [/"/, {
                        token: "string.quote",
                        bracket: "@close",
                        next: "@pop"
                    }]
                ],
                whitespace: [
                    [/[ \t\r\n]+/, ""],
                    [/#.*$/, "comment"]
                ]
            }
        }
    }
}]);
```