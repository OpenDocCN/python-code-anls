# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\5849.1a5b2ead388fa43d.js`

```py
"use strict";
// 使用严格模式，确保代码执行在更严格的语义下

(self.webpackChunk_N_E = self.webpackChunk_N_E || []).push([[5849], {
    25849: function (e, n, s) {
        s.r(n), s.d(n, {
            // 导出配置函数
            conf: function () {
                return o
            },
            // 导出语言定义函数
            language: function () {
                return t
            }
        });
        /*!-----------------------------------------------------------------------------
         * 版权所有 (c) Microsoft Corporation. 保留所有权利.
         * 版本: 0.34.1(547870b6881302c5b4ff32173c16d06009e3588f)
         * 根据 MIT 许可证发布
         * https://github.com/microsoft/monaco-editor/blob/main/LICENSE.txt
         *-----------------------------------------------------------------------------*/
        var o = {
            // 定义括号的自动闭合对
            brackets: [
                ["{", "}"],
                ["[", "]"],
                ["(", ")"]
            ],
            // 定义自动闭合对
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
                    open: "'",
                    close: "'"
                }
            ],
            // 定义周围环绕的对
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
                    open: '"',
                    close: '"'
                },
                {
                    open: "'",
                    close: "'"
                }
            ]
        };

        var t = {
            // 默认 token
            defaultToken: "",
            // token 后缀
            tokenPostfix: ".dockerfile",
            // 变量定义的正则表达式
            variable: /\${?[\w]+}?/,
            // 定义词法解析器
            tokenizer: {
                // 根规则
                root: [{
                        // 包含空白字符
                        include: "@whitespace"
                    },
                    {
                        // 包含注释
                        include: "@comment"
                    },
                    // ONBUILD 关键字
                    [/(ONBUILD)(\s+)/, ["keyword", ""]],
                    // ENV 关键字
                    [/(ENV)(\s+)([\w]+)/, ["keyword", "", {
                        // 下一个状态为参数
                        token: "variable",
                        next: "@arguments"
                    }]],
                    // 其他命令关键字
                    [/(FROM|MAINTAINER|RUN|EXPOSE|ENV|ADD|ARG|VOLUME|LABEL|USER|WORKDIR|COPY|CMD|STOPSIGNAL|SHELL|HEALTHCHECK|ENTRYPOINT)/, {
                        // token 为关键字，下一个状态为参数
                        token: "keyword",
                        next: "@arguments"
                    }]
                ],
                // 参数解析
                arguments: [{
                        // 包含空白字符
                        include: "@whitespace"
                    },
                    {
                        // 包含字符串
                        include: "@strings"
                    },
                    // 变量
                    [/@variable/, {
                        cases: {
                            // 如果遇到末尾符号，token 为变量，弹出所有状态
                            "@eos": {
                                token: "variable",
                                next: "@popall"
                            },
                            "@default": "variable"
                        }
                    }],
                    // 反斜杠处理
                    [/\\/, {
                        cases: {
                            // 如果遇到末尾符号，保持不变
                            "@eos": "",
                            "@default": ""
                        }
                    }],
                    // 其他字符处理
                    [/./, {
                        cases: {
                            // 如果遇到末尾符号，token 为空，弹出所有状态
                            "@eos": {
                                token: "",
                                next: "@popall"
                            },
                            "@default": ""
                        }
                    }]
                ],
                // 空白字符处理
                whitespace: [
                    [/\s+/, {
                        cases: {
                            // 如果遇到末尾符号，token 为空，弹出所有状态
                            "@eos": {
                                token: "",
                                next: "@popall"
                            },
                            "@default": ""
                        }
                    }]
                ],
                // 注释处理
                comment: [
                    [/(^#.*$)/, "comment", "@popall"]
                ],
                // 字符串处理
                strings: [
                    [/'$/, "", "@popall"],
                    [/\\'/, ""],
                    [/'$/, "string", "@popall"],
                    [/'/, "string", "@stringBody"],
                    [/"$/, "string", "@popall"],
                    [/"/, "string", "@dblStringBody"]
                ],
                // 单引号字符串体处理
                stringBody: [
                    [/[^\$']/, {
                        cases: {
                            // 如果遇到末尾符号，token 为字符串，弹出所有状态
                            "@eos": {
                                token: "string",
                                next: "@popall"
                            },
                            "@default": "string"
                        }
                    }],
                    // 转义字符处理
                    [/@escape/, "string.escape"],
                    // 单引号结束处理
                    [/'$/, "string", "@popall"],
                    [/'/, "string", "@pop"],
                    // 变量处理
                    [/@variable/, "variable"],
                    // 反斜杠处理
                    [/\\$/, "string"],
                    // 结束处理
                    [/$/, "string", "@popall"]
                ],
                // 双引号字符串体处理
                dblStringBody: [
                    [/[^\$"]/, {
                        cases: {
                            // 如果遇到末尾符号，token 为字符串，弹出所有状态
                            "@eos": {
                                token: "string",
                                next: "@popall"
                            },
                            "@default": "string"
                        }
                    }],
                    // 转义字符处理
                    [/@escape/, "string.escape"],
                    // 双引号结束处理
                    [/"$/, "string", "@popall"],
                    [/"/, "string", "@pop"],
                    // 变量处理
                    [/@variable/, "variable"],
                    // 反斜杠处理
                    [/\\$/, "string"],
                    // 结束处理
                    [/$/, "string", "@popall"]
                ]
            }
        }
    }
}]);
```