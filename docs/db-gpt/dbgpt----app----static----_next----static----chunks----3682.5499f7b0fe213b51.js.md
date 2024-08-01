# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\3682.5499f7b0fe213b51.js`

```py
"use strict";
// 使用严格模式，确保代码执行在严格的语义规则下

(self.webpackChunk_N_E = self.webpackChunk_N_E || []).push([[3682], {
    // 将模块添加到 webpackChunk_N_E 数组中，确保模块被加载和执行
    23682: function(e, t, o) {
        o.r(t),
        // 标记模块为已导出
        o.d(t, {
            // 定义导出的内容：conf 和 language 函数
            conf: function() {
                return n
            },
            language: function() {
                return s
            }
        });
        /*!-----------------------------------------------------------------------------
         * Copyright (c) Microsoft Corporation. All rights reserved.
         * Version: 0.34.1(547870b6881302c5b4ff32173c16d06009e3588f)
         * Released under the MIT license
         * https://github.com/microsoft/monaco-editor/blob/main/LICENSE.txt
         *-----------------------------------------------------------------------------*/
        // 定义 Monaco Editor 的配置对象 n
        var n = {
            comments: {
                lineComment: "#"
            },
            // 定义支持的括号类型和自动闭合对
            brackets: [
                ["[", "]"],
                ["<", ">"],
                ["(", ")"]
            ],
            autoClosingPairs: [{
                    open: "[",
                    close: "]"
                },
                {
                    open: "<",
                    close: ">"
                },
                {
                    open: "(",
                    close: ")"
                }
            ],
            surroundingPairs: [{
                    open: "[",
                    close: "]"
                },
                {
                    open: "<",
                    close: ">"
                },
                {
                    open: "(",
                    close: ")"
                }
            ]
        };
        // 定义 Monaco Editor 的语言配置对象 s
        var s = {
            defaultToken: "",
            tokenPostfix: ".pla",
            brackets: [{
                    open: "[",
                    close: "]",
                    token: "delimiter.square"
                },
                {
                    open: "<",
                    close: ">",
                    token: "delimiter.angle"
                },
                {
                    open: "(",
                    close: ")",
                    token: "delimiter.parenthesis"
                }
            ],
            // 定义关键字列表
            keywords: [
                ".i", ".o", ".mv", ".ilb", ".ob", ".label", ".type", ".phase", ".pair",
                ".symbolic", ".symbolic-output", ".kiss", ".p", ".e", ".end"
            ],
            // 定义注释的正则表达式
            comment: /#.*$/,
            // 定义标识符的正则表达式
            identifier: /[a-zA-Z]+[a-zA-Z0-9_\-]*/,
            // 定义 PLA 文件内容的正则表达式
            plaContent: /[01\-~\|]+/,
            // 定义语法分析规则
            tokenizer: {
                root: [{
                        include: "@whitespace"
                    },
                    [/ @comment /, "comment"],
                    [/\.([a-zA-Z_\-]+)/, {
                        cases: {
                            "@eos": {
                                token: "keyword.$1"
                            },
                            "@keywords": {
                                cases: {
                                    ".type": {
                                        token: "keyword.$1",
                                        next: "@type"
                                    },
                                    "@default": {
                                        token: "keyword.$1",
                                        next: "@keywordArg"
                                    }
                                }
                            },
                            "@default": {
                                token: "keyword.$1"
                            }
                        }
                    }],
                    [/ @identifier /, "identifier"],
                    [/ @plaContent /, "string"]
                ],
                whitespace: [
                    [/ [ \t\r\n]+ /, ""]
                ],
                type: [{
                        include: "@whitespace"
                    },
                    [/\w+/, {
                        token: "type",
                        next: "@pop"
                    }]
                ],
                keywordArg: [
                    [/ [ \t\r\n]+ /, {
                        cases: {
                            "@eos": {
                                token: "",
                                next: "@pop"
                            },
                            "@default": ""
                        }
                    }],
                    [/ @comment /, "comment", "@pop"],
                    [/[<>()\[\]]/, {
                        cases: {
                            "@eos": {
                                token: "@brackets",
                                next: "@pop"
                            },
                            "@default": "@brackets"
                        }
                    }],
                    [/-?\d+/, {
                        cases: {
                            "@eos": {
                                token: "number",
                                next: "@pop"
                            },
                            "@default": "number"
                        }
                    }],
                    [/ @identifier /, {
                        cases: {
                            "@eos": {
                                token: "identifier",
                                next: "@pop"
                            },
                            "@default": "identifier"
                        }
                    }],
                    [/[;=]/, {
                        cases: {
                            "@eos": {
                                token: "delimiter",
                                next: "@pop"
                            },
                            "@default": "delimiter"
                        }
                    }]
                ]
            }
        }
    }
}]);
```