# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\8084.9c5525713f6a5357.js`

```py
"use strict";
// 使用严格模式，确保代码执行在严格的语法和错误检查环境中

(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[8084], {
    98084: function(e,o,n) {
        n.r(o),
        n.d(o, {
            conf: function() {
                return t
            },
            language: function() {
                return s
            }
        });
        /*!-----------------------------------------------------------------------------
         * 版权所有 (c) Microsoft Corporation. 保留所有权利.
         * 版本: 0.34.1(547870b6881302c5b4ff32173c16d06009e3588f)
         * 在 MIT 许可下发布
         * https://github.com/microsoft/monaco-editor/blob/main/LICENSE.txt
         *-----------------------------------------------------------------------------*/
        var t = {
            comments: {
                lineComment: "//",
                blockComment: ["(*", "*)"]
            },
            brackets: [
                ["{", "}"],
                ["[", "]"],
                ["(", ")"],
                ["<", ">"]
            ],
            autoClosingPairs: [
                { open: "{", close: "}" },
                { open: "[", close: "]" },
                { open: "(", close: ")" },
                { open: "<", close: ">" },
                { open: "'", close: "'" }
            ],
            surroundingPairs: [
                { open: "{", close: "}" },
                { open: "[", close: "]" },
                { open: "(", close: ")" },
                { open: "<", close: ">" },
                { open: "'", close: "'" }
            ]
        };
        var s = {
            defaultToken: "",
            tokenPostfix: ".pascaligo",
            ignoreCase: !0,
            brackets: [
                { open: "{", close: "}", token: "delimiter.curly" },
                { open: "[", close: "]", token: "delimiter.square" },
                { open: "(", close: ")", token: "delimiter.parenthesis" },
                { open: "<", close: ">", token: "delimiter.angle" }
            ],
            keywords: [
                "begin", "block", "case", "const", "else", "end", "fail", "for", "from",
                "function", "if", "is", "nil", "of", "remove", "return", "skip", "then",
                "type", "var", "while", "with", "option", "None", "transaction"
            ],
            typeKeywords: [
                "bool", "int", "list", "map", "nat", "record", "string", "unit",
                "address", "map", "mtz", "xtz"
            ],
            operators: [
                "=", ">", "<", "<=", ">=", "<>", ":", ":=", "and", "mod", "or", "+", "-",
                "*", "/", "@", "&", "^", "%"
            ],
            symbols: /[=><:@\^&|+\-*\/\^%]+/,
            tokenizer: {
                root: [
                    [/[a-zA-Z_][\w]*/, {
                        cases: {
                            "@keywords": { token: "keyword.$0" },
                            "@default": "identifier"
                        }
                    }],
                    { include: "@whitespace" },
                    [/[{}()\[\]]/, "@brackets"],
                    [/[<>](?!@symbols)/, "@brackets"],
                    [/@symbols/, {
                        cases: {
                            "@operators": "delimiter",
                            "@default": ""
                        }
                    }],
                    [/\d*\.\d+([eE][\-+]?\d+)?/, "number.float"],
                    [/\$[0-9a-fA-F]{1,16}/, "number.hex"],
                    [/\d+/, "number"],
                    [/[;,.]/, "delimiter"],
                    [/'([^'\\]|\\.)*$/, "string.invalid"],
                    [/'/, "string", "@string"],
                    [/'[^\\']'/, "string"],
                    [/'/, "string.invalid"],
                    [/#\d+/, "string"]
                ],
                comment: [
                    [/[^\(\*]+/, "comment"],
                    [/\*\)/, "comment", "@pop"],
                    [/\(\*/, "comment"]
                ],
                string: [
                    [/[^\\']+/, "string"],
                    [/@./, "string.escape.invalid"],
                    [/'/, { token: "string.quote", bracket: "@close", next: "@pop" }]
                ],
                whitespace: [
                    /[ \t\r\n]+/, "white"],
                    [/\(\*/, "comment", "@comment"],
                    [/\s+/, "comment"]
                ]
            }
        }
    }
}]);
```