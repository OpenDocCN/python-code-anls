# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\2060.b7145aec88c35449.js`

```py
"use strict";
// 使用严格模式，确保代码执行在严格的语法和错误检查环境中

(self.webpackChunk_N_E = self.webpackChunk_N_E || []).push([[2060], {
    // 将当前模块推入 webpack 模块加载队列中

    32060: function(e, n, o) {
        // 定义模块 32060
        o.r(n), o.d(n, {
            conf: function() {
                return t
            },
            // 导出函数 conf 返回配置对象 t

            language: function() {
                return s
            }
            // 导出函数 language 返回语言配置对象 s
        });
        /*!-----------------------------------------------------------------------------
         * 版权所有 © Microsoft Corporation
         * 版本号: 0.34.1(547870b6881302c5b4ff32173c16d06009e3588f)
         * 使用 MIT 许可发布
         * https://github.com/microsoft/monaco-editor/blob/main/LICENSE.txt
         *-----------------------------------------------------------------------------*/

        var t = {
            // 定义配置对象 t
            comments: {
                lineComment: ";",
                // 单行注释符号为 ;
                blockComment: ["#|", "|#"]
                // 多行注释起始和结束符号为 #| 和 |#
            },
            brackets: [
                ["(", ")"],
                // 定义括号配对 ( 和 )
                ["{", "}"],
                // 定义括号配对 { 和 }
                ["[", "]"]
                // 定义括号配对 [ 和 ]
            ],
            autoClosingPairs: [
                { open: "{", close: "}" },
                // 定义自动闭合配对 { 和 }
                { open: "[", close: "]" },
                // 定义自动闭合配对 [ 和 ]
                { open: "(", close: ")" },
                // 定义自动闭合配对 ( 和 )
                { open: '"', close: '"' }
                // 定义自动闭合配对 " 和 "
            ],
            surroundingPairs: [
                { open: "{", close: "}" },
                // 定义包围配对 { 和 }
                { open: "[", close: "]" },
                // 定义包围配对 [ 和 ]
                { open: "(", close: ")" },
                // 定义包围配对 ( 和 )
                { open: '"', close: '"' }
                // 定义包围配对 " 和 "
            ]
        };

        var s = {
            // 定义语言配置对象 s
            defaultToken: "",
            // 默认 token 为空字符串
            ignoreCase: true,
            // 忽略大小写

            tokenPostfix: ".scheme",
            // token 后缀为 .scheme

            brackets: [
                { open: "(", close: ")", token: "delimiter.parenthesis" },
                // 定义括号配对 ( 和 )，使用 token delimiter.parenthesis
                { open: "{", close: "}", token: "delimiter.curly" },
                // 定义括号配对 { 和 }，使用 token delimiter.curly
                { open: "[", close: "]", token: "delimiter.square" }
                // 定义括号配对 [ 和 ]，使用 token delimiter.square
            ],

            keywords: [
                "case", "do", "let", "loop", "if", "else", "when", "cons", "car", "cdr",
                "cond", "lambda", "lambda*", "syntax-rules", "format", "set!", "quote",
                "eval", "append", "list", "list?", "member?", "load"
            ],
            // 定义关键字数组

            constants: ["#t", "#f"],
            // 定义常量数组

            operators: ["eq?", "eqv?", "equal?", "and", "or", "not", "null?"],
            // 定义操作符数组

            tokenizer: {
                root: [
                    [/#[xXoObB][0-9a-fA-F]+/, "number.hex"],
                    // 匹配十六进制数字
                    [/[+-]?\d+(?:(?:\.\d*)?(?:[eE][+-]?\d+)?)?/, "number.float"],
                    // 匹配浮点数
                    [
                        /(?:\b(?:(define|define-syntax|define-macro))\b)(\s+)((?:\w|\-|\!|\?)*)/,
                        ["keyword", "white", "variable"]
                        // 匹配 define 系列关键字
                    ],
                    { include: "@whitespace" },
                    // 引用 @whitespace 规则
                    { include: "@strings" },
                    // 引用 @strings 规则
                    [
                        /[a-zA-Z_#][a-zA-Z0-9_\-\?\!\*]*/,
                        {
                            cases: {
                                "@keywords": "keyword",
                                "@constants": "constant",
                                "@operators": "operators",
                                "@default": "identifier"
                            }
                        }
                        // 匹配标识符，并根据类型赋予不同的 token 类型
                    ]
                ],
                comment: [
                    [/[^\|#]+/, "comment"],
                    // 匹配注释内容
                    [/#\|/, "comment", "@push"],
                    // 匹配多行注释起始，推入状态栈
                    [/\|#/, "comment", "@pop"],
                    // 匹配多行注释结束，弹出状态栈
                    [/[\\|#]/, "comment"]
                    // 匹配多行注释内的其他内容
                ],
                whitespace: [
                    [/[ \t\r\n]+/, "white"],
                    // 匹配空白字符
                    [/#\|/, "comment", "@comment"],
                    // 匹配多行注释起始
                    [/;.*$/, "comment"]
                    // 匹配单行注释
                ]
            },

            strings: [
                [/"$/, "string", "@popall"],
                // 匹配多行字符串结束，弹出状态栈
                [/"(?=.)/, "string", "@multiLineString"]
                // 匹配多行字符串起始
            ],

            multiLineString: [
                [/[^\\"']+$/, "string", "@popall"],
                // 匹配多行字符串内容结尾，弹出状态栈
                [/[^\\"']+/,"string"],
                // 匹配多行字符串内容
                [/[\\]/, "string.escape"],
                // 匹配多行字符串转义字符
                [/"(?=.)/, "string", "@popall"],
                // 匹配多行字符串结束，弹出状态栈
                [/[\\]$/, "string"]
                // 匹配多行字符串转义字符结尾
            ]
        };
    }
}]);
```