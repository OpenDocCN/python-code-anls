# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\2798.c706cee7f9b852d2.js`

```py
"use strict";
// 使用严格模式，确保代码在更严格的条件下执行

(self.webpackChunk_N_E = self.webpackChunk_N_E || []).push([[2798], {
    59607: function(e, n, s) {
        // 定义模块 59607，包含了 monaco-editor 的配置和语言定义
        s.r(n), s.d(n, {
            conf: function() {
                return t
            },
            language: function() {
                return o
            }
        });
        /*!-----------------------------------------------------------------------------
         * 版权所有 (c) Microsoft Corporation. 保留所有权利.
         * 版本: 0.34.1(547870b6881302c5b4ff32173c16d06009e3588f)
         * 使用 MIT 许可发布
         * https://github.com/microsoft/monaco-editor/blob/main/LICENSE.txt
         *-----------------------------------------------------------------------------*/
        
        var t = {
            comments: {
                lineComment: "#" // 定义单行注释符号为 #
            },
            brackets: [
                ["{", "}"],
                ["[", "]"],
                ["(", ")"]
            ],
            autoClosingPairs: [
                { open: "{", close: "}" },
                { open: "[", close: "]" },
                { open: "(", close: ")" },
                { open: '"', close: '"' },
                { open: "'", close: "'" }
            ],
            surroundingPairs: [
                { open: "{", close: "}" },
                { open: "[", close: "]" },
                { open: "(", close: ")" },
                { open: '"', close: '"' },
                { open: "'", close: "'" }
            ]
        };
        
        var o = {
            defaultToken: "", // 定义默认的 token 类型为空字符串
            tokenPostfix: ".ini", // 定义 token 的后缀为 .ini
            escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,
            tokenizer: {
                root: [
                    [/^\[[^\]]*\]/, "metatag"], // 匹配 [标签] 形式的元标签
                    [/(^\w+)(\s*)(\=)/, ["key", "", "delimiter"]], // 匹配键值对的格式
                    { include: "@whitespace" }, // 包含空白字符的处理规则
                    [/d+/, "number"], // 匹配数字
                    [/"([^"\\]|\\.)*$/, "string.invalid"], // 匹配不完整的双引号字符串
                    [/'([^'\\]|\\.)*$/, "string.invalid"], // 匹配不完整的单引号字符串
                    [/"/, "string", "@string.\""], // 匹配双引号字符串的开始
                    [/'/, "string", "@string.'"] // 匹配单引号字符串的开始
                ],
                whitespace: [
                    [/[ \t\r\n]+/, ""], // 匹配空白字符
                    [/^\s*[#;].*$/, "comment"] // 匹配以 # 或 ; 开头的注释
                ],
                string: [
                    [/[^\\"']+/, "string"], // 匹配普通字符串内容
                    [/ @escapes /, "string.escape"], // 匹配转义字符
                    [ /\\./, "string.escape.invalid"], // 匹配无效的转义字符
                    [/["']/, {
                        cases: {
                            "$#==$S2": {
                                token: "string",
                                next: "@pop"
                            },
                            "@default": "string"
                        }
                    }] // 匹配字符串的结束
                ]
            }
        }
    }
}]);
```