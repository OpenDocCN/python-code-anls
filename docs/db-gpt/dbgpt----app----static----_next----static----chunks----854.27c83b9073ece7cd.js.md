# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\854.27c83b9073ece7cd.js`

```py
"use strict";
// 使用严格模式，确保代码执行在严格的语义下

(self.webpackChunk_N_E = self.webpackChunk_N_E || []).push([[854], {
    60854: function (e, t, r) {
        // 定义模块 60854
        r.r(t), 
        // 导出模块配置函数 conf 和语言配置函数 language
        r.d(t, {
            conf: function () { return n },
            language: function () { return s }
        });
        /*!-----------------------------------------------------------------------------
         * 版权所有 (c) Microsoft Corporation. 保留所有权利。
         * 版本: 0.34.1(547870b6881302c5b4ff32173c16d06009e3588f)
         * 遵循 MIT 许可证发布
         * https://github.com/microsoft/monaco-editor/blob/main/LICENSE.txt
         *-----------------------------------------------------------------------------*/
        
        // 定义语言配置对象 n
        var n = {
            // 单词模式匹配，匹配数字、标识符等
            wordPattern: /(-?\d*\.\d\w*)|([^\`\~\!\@\#%\^\&\*\(\)\=\$\-\+\[\{\]\}\\\|\;\:\'\"\,\.\<\>\/\?\s]+)/g,
            // 注释配置，包括块注释和行注释
            comments: {
                blockComment: ["###", "###"],  // 块注释标记为 ###
                lineComment: "#"  // 行注释以 # 开头
            },
            // 折叠代码配置
            folding: {
                markers: {
                    // 折叠开始标记，以 #region 开头
                    start: RegExp("^\\s*#region\\b"),
                    // 折叠结束标记，以 #endregion 开头
                    end: RegExp("^\\s*#endregion\\b")
                }
            }
        };

        // 定义语言配置对象 s
        var s = {
            defaultToken: "",  // 默认的语法单元为空字符串
            ignoreCase: false,  // 不忽略大小写
            tokenPostfix: ".mips",  // token 后缀为 .mips

            // 正则表达式配置，用于匹配特定语法元素
            regEx: /\/(?!\/\/)(?:[^\/\\]|\\.)*\/[igm]*/,

            // 关键字列表
            keywords: [
                ".data", ".text", "syscall", "trap", "add", "addu", "addi", "addiu", "and", "andi",
                "div", "divu", "mult", "multu", "nor", "or", "ori", "sll", "slv", "sra", "srav",
                "srl", "srlv", "sub", "subu", "xor", "xori", "lhi", "lho", "lhi", "llo", "slt",
                "slti", "sltu", "sltiu", "beq", "bgtz", "blez", "bne", "j", "jal", "jalr", "jr",
                "lb", "lbu", "lh", "lhu", "lw", "li", "la", "sb", "sh", "sw", "mfhi", "mflo",
                "mthi", "mtlo", "move"
            ],

            // 符号表达式
            symbols: /[\. ,\:]+/,

            // 转义字符表达式
            escapes: /\\(?:[abfnrtv\\"'$]|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

            // Tokenizer 定义不同语法单元的识别规则
            tokenizer: {
                root: [
                    [/\$[a-zA-Z_]\w*/, "variable.predefined"],  // 变量名规则
                    [/[.a-zA-Z_]\w*/, {  // 标识符规则
                        cases: {
                            this: "variable.predefined",
                            "@keywords": { token: "keyword.$0" },  // 匹配关键字
                            "@default": ""
                        }
                    }],
                    [/[ \t\r\n]+/, ""],  // 空白字符
                    [/#.*$/, "comment"],  // 行注释
                    ["///", { token: "regexp", next: "@hereregexp" }],  // 正则表达式文本
                    [/^(\s*)(@regEx)/, ["", "regexp"]],  // 正则表达式匹配
                    [/\,(\s*)(@regEx)/, ["delimiter", "", "regexp"]],  // 分隔符
                    [/\:(\s*)(@regEx)/, ["delimiter", "", "regexp"]],  // 冒号
                    [/@symbols, "delimiter"],  // 符号
                    [/\d+[eE]([\-+]?\d+)?/, "number.float"],  // 浮点数
                    [/\d+\.\d+([eE][\-+]?\d+)?/, "number.float"],  // 浮点数
                    [/0[xX][0-9a-fA-F]+/, "number.hex"],  // 十六进制数
                    [/0[0-7]+(?!\d)/, "number.octal"],  // 八进制数
                    [/\d+/, "number"],  // 数字
                    [/[,.]/, "delimiter"],  // 分隔符
                    [/"""/, "string", '@herestring."""'],  // 多行双引号字符串
                    [/'''/, "string", "@herestring.'''"],  // 多行单引号字符串
                    [/"(?=#)/, {  // 双引号字符串
                        cases: {
                            "@eos": "string",
                            "@default": { token: "string", next: '@string."' }
                        }
                    }],
                    [/'(?=#)/, {  // 单引号字符串
                        cases: {
                            "@eos": "string",
                            "@default": { token: "string", next: "@string.'" }
                        }
                    }]
                ],

                // 字符串识别规则
                string: [
                    [/[^"'\#\\]+/, "string"],
                    [/@escapes/, "string.escape"],
                    [/\./, "string.escape.invalid"],
                    [/#{/, { cases: { '$S2=="': { token: "string", next: "root.interpolatedstring" }, "@default": "string" } }],
                    [/"|'/, {
                        cases: {
                            "$#==$S2": { token: "string", next: "@pop" },
                            "@default": "string"
                        }
                    }],
                    [/#/, "string"]
                ],

                // 多行字符串识别规则
                herestring: [
                    [/("""|''')/, {
                        cases: {
                            "$1==$S2": { token: "string", next: "@pop" },
                            "@default": "string"
                        }
                    }],
                    [/[^\#\\'"]+/, "string"],
                    [/'|"/, "string"],
                    [/@escapes/, "string.escape"],
                    [/\./, "string.escape.invalid"],
                    [/#{/, { token: "string.quote", next: "root.interpolatedstring" }],
                    [/#/, "string"]
                ],

                // 注释识别规则
                comment: [
                    [/[^\#]+/, "comment"],
                    [/#/, "comment"]
                ],

                // 正则表达式文本识别规则
                hereregexp: [
                    [/[^\\\\/#]+/, "regexp"],
                    [/\./, "regexp.escape.invalid"],
                    [/#.*$/, "comment"],
                    ["///[igm]*", { token: "regexp", next: "@pop" }],
                    [/\//, "regexp"]
                ]
            }
        }
    }
}]);
```