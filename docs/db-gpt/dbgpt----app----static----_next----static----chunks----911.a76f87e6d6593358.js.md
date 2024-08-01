# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\911.a76f87e6d6593358.js`

```py
"use strict";
// 使用严格模式，确保代码执行在严格的语法和错误检查环境中

(self.webpackChunk_N_E = self.webpackChunk_N_E || []).push([[911], {
    20911: function (e, n, t) {
        // 定义模块 20911
        t.r(n), // 导出模块接口
        t.d(n, {
            conf: function () {
                return o
            }, // 导出配置对象 conf
            language: function () {
                return s
            } // 导出语言对象 language
        });
        /*!-----------------------------------------------------------------------------
         * 版权所有 © Microsoft Corporation. 保留所有权利.
         * 版本: 0.34.1(547870b6881302c5b4ff32173c16d06009e3588f)
         * 根据 MIT 许可发布
         * https://github.com/microsoft/monaco-editor/blob/main/LICENSE.txt
         *-----------------------------------------------------------------------------*/
        var o = {
            // 定义配置对象 o
            comments: {
                lineComment: "--", // 单行注释符号
                blockComment: ["--[[", "]]"] // 块注释的起始和结束符号
            },
            brackets: [
                ["{", "}"], // 大括号的配对
                ["[", "]"], // 中括号的配对
                ["(", ")"]  // 小括号的配对
            ],
            autoClosingPairs: [
                { open: "{", close: "}" },   // 自动补全的大括号配对
                { open: "[", close: "]" },   // 自动补全的中括号配对
                { open: "(", close: ")" },   // 自动补全的小括号配对
                { open: '"', close: '"' },   // 自动补全的双引号配对
                { open: "'", close: "'" }    // 自动补全的单引号配对
            ],
            surroundingPairs: [
                { open: "{", close: "}" },   // 包围选中文本的大括号配对
                { open: "[", close: "]" },   // 包围选中文本的中括号配对
                { open: "(", close: ")" },   // 包围选中文本的小括号配对
                { open: '"', close: '"' },   // 包围选中文本的双引号配对
                { open: "'", close: "'" }    // 包围选中文本的单引号配对
            ]
        };
        var s = {
            // 定义语言对象 s
            defaultToken: "",               // 默认的 token 类型
            tokenPostfix: ".lua",           // token 的后缀名
            keywords: [
                // Lua 关键字列表
                "and", "break", "do", "else", "elseif", "end", "false", "for", "function",
                "goto", "if", "in", "local", "nil", "not", "or", "repeat", "return", "then",
                "true", "until", "while"
            ],
            brackets: [
                // 不同类型括号的配置
                { token: "delimiter.bracket", open: "{", close: "}" },
                { token: "delimiter.array", open: "[", close: "]" },
                { token: "delimiter.parenthesis", open: "(", close: ")" }
            ],
            operators: [
                // Lua 中的运算符列表
                "+", "-", "*", "/", "%", "^", "#", "==", "~=", "<=", ">=", "<", ">", "=", ";", ":", ",", ".", "..", "..."
            ],
            symbols: /[=><!~?:&|+\-*\/\^%]+/,    // Lua 中的符号
            escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,   // 转义字符
            tokenizer: {
                root: [
                    // 根部的 token 规则
                    [/[a-zA-Z_]\w*/, {
                        cases: {
                            "@keywords": { token: "keyword.$0" },    // 根据关键字设置 token 类型
                            "@default": "identifier"                // 默认为标识符
                        }
                    }],
                    { include: "@whitespace" },                     // 包含空白符处理规则
                    [
                        /(,)(\s*)([a-zA-Z_]\w*)(\s*)(:)(?!:)/,
                        ["delimiter", "", "key", "", "delimiter"]
                    ],                                              // 处理逗号后面跟着标识符和冒号的情况
                    [
                        /({)(\s*)([a-zA-Z_]\w*)(\s*)(:)(?!:)/,
                        ["@brackets", "", "key", "", "delimiter"]
                    ],                                              // 处理大括号后面跟着标识符和冒号的情况
                    [/[{}()\[\]]/, "@brackets"],                    // 处理各种括号
                    [/@symbols/, {
                        cases: {
                            "@operators": "delimiter",                // 根据运算符设置为分隔符
                            "@default": ""                            // 默认情况为空
                        }
                    }],
                    [/\d*\.\d+([eE][\-+]?\d+)?/, "number.float"],    // 处理浮点数
                    [/0[xX][0-9a-fA-F_]*[0-9a-fA-F]/, "number.hex"], // 处理十六进制数
                    [/\d+?/, "number"],                             // 处理普通数字
                    /[;,.]/, "delimiter",                           // 处理分号、逗号、点号
                    [
                        /"([^"\\]|\\.)*$/,
                        "string.invalid"
                    ],                                              // 处理无效的双引号字符串
                    [
                        /'([^'\\]|\\.)*$/,
                        "string.invalid"
                    ],                                              // 处理无效的单引号字符串
                    [/"/, "string", '@string."'],                    // 处理双引号字符串
                    [/'/, "string", "@string.'"]                     // 处理单引号字符串
                ],
                whitespace: [
                    // 空白符处理规则
                    [/[ \t\r\n]+/, ""],                            // 匹配空格、制表符、回车符、换行符
                    [/--\[\([=]*)\[/, "comment", "@comment.$1"],   // 处理块注释起始符号
                    [/--.*$/, "comment"]                           // 处理单行注释
                ],
                comment: [
                    // 注释处理规则
                    [/[^\]]+/, "comment"],                         // 处理注释内容
                    [
                        /\]([=]*)\]/,
                        {
                            cases: {
                                "$1==$S2": { token: "comment", next: "@pop" }, // 处理块注释结束符号
                                "@default": "comment"
                            }
                        }
                    ],
                    [/\./, "comment"]                               // 处理块注释的点号
                ],
                string: [
                    // 字符串处理规则
                    [/[^\\"']+/, "string"],                        // 处理字符串内容
                    [/@escapes/, "string.escape"],                 // 处理转义字符
                    [/\\./, "string.escape.invalid"],              // 处理无效的转义字符
                    [
                        /["']/,
                        {
                            cases: {
                                "$#==$S2": { token: "string", next: "@pop" }, // 处理字符串结束符号
                                "@default": "string"
                            }
                        }
                    ]
                ]
            }
        };
    }
}]);
```