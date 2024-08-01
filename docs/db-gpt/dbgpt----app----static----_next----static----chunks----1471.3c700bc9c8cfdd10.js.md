# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\1471.3c700bc9c8cfdd10.js`

```py
"use strict";
// 使用严格模式，确保代码执行时采用严格的语法和错误处理

(self.webpackChunk_N_E = self.webpackChunk_N_E || []).push([[1471], {
    31471: function(e, t, n) {
        // 定义模块的入口函数，包含主要的配置对象和语言定义
        n.r(t),
        n.d(t, {
            conf: function() {
                return s
            },
            language: function() {
                return o
            }
        });
        /*!-----------------------------------------------------------------------------
         * 版权所有 (c) Microsoft Corporation. 保留所有权利。
         * 版本: 0.34.1(547870b6881302c5b4ff32173c16d06009e3588f)
         * 根据 MIT 许可发布
         * https://github.com/microsoft/monaco-editor/blob/main/LICENSE.txt
         *-----------------------------------------------------------------------------*/
        
        var s = {
            // 配置对象，定义了语言的注释格式，以及其他设置
            comments: {
                lineComment: "#" // 定义单行注释的起始符号为 #
            }
        };

        var o = {
            // 语言定义对象，定义了默认的标记和词法分析器
            defaultToken: "keyword", // 默认的标记类型为 keyword
            ignoreCase: true, // 忽略大小写
            tokenPostfix: ".azcli", // 标记后缀为 .azcli
            str: /[^\s#]/, // 字符串定义，不包括空白字符和注释符号
            tokenizer: {
                root: [
                    { include: "@comment" }, // 包含注释的处理规则
                    [/\s-+@str*\s*/, {
                        cases: {
                            "@eos": { token: "key.identifier", next: "@popall" }, // 匹配行末尾，结束当前处理
                            "@default": { token: "key.identifier", next: "@type" } // 默认情况，进入 @type 处理状态
                        }
                    }],
                    [/^-+@str*\s*/, {
                        cases: {
                            "@eos": { token: "key.identifier", next: "@popall" }, // 匹配行末尾，结束当前处理
                            "@default": { token: "key.identifier", next: "@type" } // 默认情况，进入 @type 处理状态
                        }
                    }]
                ],
                type: [
                    { include: "@comment" }, // 包含注释的处理规则
                    [/-+@str*\s*/, {
                        cases: {
                            "@eos": { token: "key.identifier", next: "@popall" }, // 匹配行末尾，结束当前处理
                            "@default": "key.identifier" // 默认情况，标记为 key.identifier 类型
                        }
                    }],
                    [/@str+\s*/, {
                        cases: {
                            "@eos": { token: "string", next: "@popall" }, // 匹配行末尾，结束当前处理
                            "@default": "string" // 默认情况，标记为 string 类型
                        }
                    }]
                ],
                comment: [
                    [/#.*$/, {
                        cases: {
                            "@eos": { token: "comment", next: "@popall" } // 匹配行末尾，结束当前处理
                        }
                    }]
                ]
            }
        };
    }
}]);
```