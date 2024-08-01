# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\4129.d8b70a3f799af47e.js`

```py
"use strict";
// 使用严格模式，确保代码执行在严格的语义下

(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[4129], {
    // 将模块推送到 webpack 模块加载器的队列中，编号为 4129

    // 定义模块
    84129: function(e, s, o) {
        // o 是模块对象，e 和 s 分别是导出对象和导入对象

        // 声明模块的导出和依赖
        o.r(s),
        o.d(s, {
            conf: function() {
                return t
            },
            language: function() {
                return n
            }
        });

        /*!-----------------------------------------------------------------------------
         * 版权所有 (c) Microsoft Corporation. 保留所有权利。
         * 版本: 0.34.1(547870b6881302c5b4ff32173c16d06009e3588f)
         * 以 MIT 许可证发布
         * https://github.com/microsoft/monaco-editor/blob/main/LICENSE.txt
         *-----------------------------------------------------------------------------*/

        var t = {
            // 定义语言特定的配置
            comments: {
                lineComment: "REM"  // 设置行注释符为 "REM"
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
                { open: '"', close: '"' }
            ],
            surroundingPairs: [
                { open: "[", close: "]" },
                { open: "(", close: ")" },
                { open: '"', close: '"' }
            ],
            folding: {
                markers: {
                    start: RegExp("^\\s*(::\\s*|REM\\s+)#region"),  // 设置折叠区域的起始标记
                    end: RegExp("^\\s*(::\\s*|REM\\s+)#endregion")    // 设置折叠区域的结束标记
                }
            }
        };

        var n = {
            defaultToken: "",  // 默认 token 为空
            ignoreCase: true,  // 忽略大小写
            tokenPostfix: ".bat",  // 设置 token 后缀为 ".bat"

            brackets: [
                { token: "delimiter.bracket", open: "{", close: "}" },
                { token: "delimiter.parenthesis", open: "(", close: ")" },
                { token: "delimiter.square", open: "[", close: "]" }
            ],

            keywords: /call|defined|echo|errorlevel|exist|for|goto|if|pause|set|shift|start|title|not|pushd|popd/,
            // 定义关键字集合

            symbols: /[=><!~?&|+\-*\/\^;\.,]+/,
            // 定义符号集合

            escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,
            // 定义转义字符集合

            tokenizer: {
                root: [
                    [/^(\s*)(rem(?:\s.*|))$/, ["", "comment"]],
                    // 匹配 REM 命令的行注释

                    [/(?:@?)(@keywords)(?!\w)/, [{ token: "keyword" }, { token: "keyword.$2" }]],
                    // 匹配关键字

                    [/[ \t\r\n]+/, ""],
                    // 匹配空白字符

                    [/setlocal(?!\w)/, "keyword.tag-setlocal"],
                    // 匹配 setlocal 命令

                    [/endlocal(?!\w)/, "keyword.tag-setlocal"],
                    // 匹配 endlocal 命令

                    [/[a-zA-Z_]\w*/, ""],
                    // 匹配变量名

                    [/:\w*/, "metatag"],
                    // 匹配标签

                    [/%[^%]+%/, "variable"],
                    // 匹配环境变量

                    [/%%[\w]+(?!\w)/, "variable"],
                    // 匹配循环变量

                    [/[{}()\[\]]/, "@brackets"],
                    // 匹配括号和花括号

                    [/@symbols/, "delimiter"],
                    // 匹配符号

                    [/\d*\.\d+([eE][\-+]?\d+)?/, "number.float"],
                    // 匹配浮点数

                    [/0[xX][0-9a-fA-F_]*[0-9a-fA-F]/, "number.hex"],
                    // 匹配十六进制数

                    [/\d+/, "number"],
                    // 匹配数字

                    [/[;,.]/, "delimiter"],
                    // 匹配分隔符

                    [/"/, "string", "@string.\""],
                    // 匹配双引号字符串

                    [/'/, "string", "@string.'"]
                    // 匹配单引号字符串
                ],

                string: [
                    [/[^\\"'%]+/, { cases: { "@eos": { token: "string", next: "@popall" }, "@default": "string" } }],
                    // 匹配普通字符串内容

                    [/@escapes/, "string.escape"],
                    // 匹配转义字符

                    [/\\./, "string.escape.invalid"],
                    // 匹配无效的转义字符

                    [/%[\w ]+%/, "variable"],
                    // 匹配环境变量

                    [/%%[\w]+(?!\w)/, "variable"],
                    // 匹配循环变量

                    [/[\"']/, { cases: { "$#==$S2": { token: "string", next: "@pop" }, "@default": "string" } }],
                    // 匹配字符串的引号
                ]
            }
        };
    }
}]);
```