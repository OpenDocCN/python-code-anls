# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\4912.402608f60657cdc1.js`

```py
"use strict";
// 使用严格模式，确保代码在更严格的语法和错误检查条件下执行
(self.webpackChunk_N_E = self.webpackChunk_N_E || []).push([[4912], {
    // 定义模块 ID 为 4912，并添加模块定义函数
    64912: function(e, t, i) {
        // 导出模块函数，包括 conf() 和 language() 方法
        i.r(t), i.d(t, {
            conf: function() {
                return o
            },
            language: function() {
                return n
            }
        });
        /*!-----------------------------------------------------------------------------
         * 版权所有 (c) Microsoft Corporation. 保留所有权利.
         * 版本: 0.34.1(547870b6881302c5b4ff32173c16d06009e3588f)
         * 根据 MIT 许可证发布
         * https://github.com/microsoft/monaco-editor/blob/main/LICENSE.txt
         *-----------------------------------------------------------------------------*/
        var o = {
            // 定义注释配置，设置单行注释标记为 "COMMENT"
            comments: {
                lineComment: "COMMENT"
            },
            // 定义括号匹配对，包括 ( )
            brackets: [
                ["(", ")"]
            ],
            // 定义自动闭合字符对，如 { }, [ ], ( ), " "
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
                    open: '"',
                    close: '"'
                },
                {
                    open: ":",
                    close: "."
                }
            ],
            // 定义周围匹配字符对，如 { }, [ ], ( ), ` `, " ", ' ', : .
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
                    open: "`",
                    close: "`"
                },
                {
                    open: '"',
                    close: '"'
                },
                {
                    open: "'",
                    close: "'"
                },
                {
                    open: ":",
                    close: "."
                }
            ],
            // 定义折叠标记，以正则表达式匹配折叠的开始和结束
            folding: {
                markers: {
                    start: RegExp("^\\s*(::\\s*|COMMENT\\s+)#region"),
                    end: RegExp("^\\s*(::\\s*|COMMENT\\s+)#endregion")
                }
            }
        };
        var n = {
            // 定义语言后缀为 ".lexon"，忽略大小写
            tokenPostfix: ".lexon",
            ignoreCase: !0,
            // 定义关键字列表
            keywords: ["lexon", "lex", "clause", "terms", "contracts", "may", "pay", "pays", "appoints", "into", "to"],
            // 定义类型关键字列表
            typeKeywords: ["amount", "person", "key", "time", "date", "asset", "text"],
            // 定义操作符列表
            operators: ["less", "greater", "equal", "le", "gt", "or", "and", "add", "added", "subtract", "subtracted", "multiply", "multiplied", "times", "divide", "divided", "is", "be", "certified"],
            // 定义符号正则表达式，匹配各种符号
            symbols: /[=><!~?:&|+\-*\/\^%]+/,
            // 定义语法解析器，根据正则表达式匹配不同的词法单元
            tokenizer: {
                root: [
                    // 匹配注释行，以及可能的后续内容
                    [/^(\s*)(comment:?(?:\s.*|))$/, ["", "comment"]],
                    // 匹配双引号字符串
                    [/"/, {
                        token: "identifier.quote",
                        bracket: "@open",
                        next: "@quoted_identifier"
                    }],
                    // 匹配以 "LEX" 结尾的关键字
                    ["LEX$", {
                        token: "keyword",
                        bracket: "@open",
                        next: "@identifier_until_period"
                    }],
                    // 匹配以 "LEXON" 开头的关键字
                    ["LEXON", {
                        token: "keyword",
                        bracket: "@open",
                        next: "@semver"
                    }],
                    // 匹配冒号作为分隔符
                    [":", {
                        token: "delimiter",
                        bracket: "@open",
                        next: "@identifier_until_period"
                    }],
                    // 匹配标识符，包括操作符、类型关键字、普通关键字和默认情况下的标识符
                    [/^[a-z_$][\w$]*/, {
                        cases: {
                            "@operators": "operator",
                            "@typeKeywords": "keyword.type",
                            "@keywords": "keyword",
                            "@default": "identifier"
                        }
                    }],
                    // 包括空白字符的匹配
                    {
                        include: "@whitespace"
                    },
                    // 匹配括号字符
                    [/[{}()\[\]]/, "@brackets"],
                    // 匹配尖括号
                    [/[<>](?!@symbols)/, "@brackets"],
                    // 匹配各种符号作为分隔符
                    [/@symbols/, "delimiter"],
                    // 匹配语义版本号，形如 x.y.z
                    [/^\d*\.\d*\.\d*/, "number.semver"],
                    // 匹配浮点数
                    [/^\d*\.\d+([eE][\-+]?\d+)?/, "number.float"],
                    // 匹配十六进制数
                    [/^0[xX][0-9a-fA-F]+/, "number.hex"],
                    // 匹配普通数字
                    [/^\d+/, "number"],
                    // 匹配分隔符如 ; , .
                    [/[;,.]/, "delimiter"]
                ],
                // 匹配带引号的标识符
                quoted_identifier: [
                    // 匹配非转义的双引号内容
                    [/[^\\"]+/, "identifier"],
                    // 匹配双引号结束
                    [/" /, {
                        token: "identifier.quote",
                        bracket: "@close",
                        next: "@pop"
                    }]
                ],
                // 匹配标识符直到句号
                space_identifier_until_period: [
                    [":", "delimiter"],
                    [" ", {
                        token: "white",
                        next: "@identifier_rest"
                    }]
                ],
                // 匹配标识符直到句号，包括各种分隔符
                identifier_until_period: [{
                        include: "@whitespace"
                    },
                    [":", {
                        token: "delimiter",
                        next: "@identifier_rest"
                    }],
                    // 匹配除句号以外的标识符内容
                    [/[^\\.]+/, "identifier"],
                    // 匹配句号作为结束符
                    [/\./, {
                        token: "delimiter",
                        bracket: "@close",
                        next: "@pop"
                    }]
                ],
                // 匹配标识符的剩余部分，直到句号
                identifier_rest: [
                    // 匹配除句号以外的标识符内容
                    [/[^\\.]+/, "identifier"],
                    // 匹配句号作为结束符
                    [/\./, {
                        token: "delimiter",
                        bracket: "@close",
                        next: "@pop"
                    }]
                ],
                // 匹配语义版本号，形如 x.y.z
                semver: [{
                        include: "@whitespace"
                    },
                    [":", "delimiter"],
                    [/^\d*\.\d*\.\d*/, {
                        token: "number.semver",
                        bracket: "@close",
                        next: "@pop"
                    }]
                ],
                // 匹配空白字符
                whitespace: [
                    [/[ \t\r\n]+/, "white"]
                ]
            }
        }
    }
}]);
```