# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\1448.afef71544b4c931b.js`

```py
"use strict";
// 使用严格模式

(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[1448],{
    11448:function(e,o,t){ // 定义一个匿名函数，参数分别为 e, o, t
        t.r(o), // 导出模块对象 o
        t.d(o,{ // 定义 o 的属性
            conf:function(){return n}, // 属性 conf 返回 n
            language:function(){return s} // 属性 language 返回 s
        });
        /*!-----------------------------------------------------------------------------
         * 版权所有 © Microsoft Corporation. 保留所有权利。
         * 版本: 0.34.1(547870b6881302c5b4ff32173c16d06009e3588f)
         * 根据 MIT 许可发布
         * https://github.com/microsoft/monaco-editor/blob/main/LICENSE.txt
         *-----------------------------------------------------------------------------*/
        
        var n={ // 定义变量 n
            comments:{ // 对象 comments
                lineComment:"//", // 单行注释符号
                blockComment:["(*","*)"] // 块注释符号对
            },
            brackets:[ // 括号对
                ["{","}"],
                ["[","]"],
                ["(",")"],
                ["<",">"]
            ],
            autoClosingPairs:[ // 自动补全的括号对
                {open:"{",close:"}"},
                {open:"[",close:"]"},
                {open:"(",close:")"},
                {open:"<",close:">"},
                {open:"'",close:"'"},
                {open:'"',close:'"'},
                {open:"(*",close:"*)"} // 特殊的块注释括号对
            ],
            surroundingPairs:[ // 包围匹配的括号对
                {open:"{",close:"}"},
                {open:"[",close:"]"},
                {open:"(",close:")"},
                {open:"<",close:">"},
                {open:"'",close:"'"},
                {open:'"',close:'"'},
                {open:"(*",close:"*)"} // 特殊的块注释括号对
            ]
        };

        var s={ // 定义变量 s
            defaultToken:"", // 默认 token
            tokenPostfix:".cameligo", // token 后缀
            ignoreCase:!0, // 忽略大小写
            brackets:[ // 括号对
                {open:"{",close:"}",token:"delimiter.curly"},
                {open:"[",close:"]",token:"delimiter.square"},
                {open:"(",close:")",token:"delimiter.parenthesis"},
                {open:"<",close:">",token:"delimiter.angle"}
            ],
            keywords:[ // 关键字列表
                "abs","assert","block","Bytes","case","Crypto","Current","else","failwith",
                "false","for","fun","if","in","let","let%entry","let%init","List","list",
                "Map","map","match","match%nat","mod","not","operation","Operation","of",
                "record","Set","set","sender","skip","source","String","then","to","true",
                "type","with"
            ],
            typeKeywords:[ // 类型关键字
                "int","unit","string","tz","nat","bool"
            ],
            operators:[ // 运算符
                "=","<",">","<=",">=","<>",":",":=","and","mod","or","+","-","*","/",
                "@","&","^","%","->","<-","&&","||"
            ],
            symbols:/[=><:@\^&|+\-*\/\^%]+/, // 符号正则表达式
            tokenizer:{ // 分词器定义
                root:[
                    [/[\w]+/, { // 匹配字母数字字符序列
                        cases: {
                            "@keywords": {token: "keyword.$0"}, // 如果是关键字，则使用对应的 token
                            "@default": "identifier" // 否则默认为标识符
                        }
                    }],
                    {include: "@whitespace"}, // 包含空白字符的处理规则
                    [/[\{\}\(\)\[\]]/, "@brackets"], // 匹配各种括号
                    [/\<\>$/, "@brackets"], // 匹配尖括号
                    [/@symbols/, {cases: {"@operators": "delimiter", "@default": ""}}],
                    [/\d*\.\d+([eE][\-+]?\d+)?/, "number.float"], // 匹配浮点数
                    [/\d+/, "number"], // 匹配数字
                    [/[;,.]/, "delimiter"], // 匹配分号、逗号、句号
                    [/'([^'\\]|\\.)*$/, "string.invalid"], // 不完整的字符串
                    [/'/, "string", "@string"], // 匹配单引号字符串
                    [/'[^\\']'/, "string"], // 匹配完整的单引号字符串
                    [/'/, "string.invalid"], // 不完整的单引号字符串
                    [/\#\d+/, "string"] // 匹配哈希字符串
                ],
                comment:[
                    [/[^(\*)]+/, "comment"], // 匹配非块注释内容
                    [/\*\)/, "comment", "@pop"], // 匹配块注释结束
                    [/\(\*/, "comment"] // 匹配块注释开始
                ],
                string:[
                    [/[^\\"']+/,"string"], // 匹配非转义的双引号字符串
                    [/\\./,"string.escape.invalid"], // 无效的转义字符
                    [/'/, { // 匹配单引号字符串
                        token: "string.quote",
                        bracket: "@close",
                        next: "@pop"
                    }]
                ],
                whitespace:[
                    [/[ \t\r\n]+/, "white"], // 空白字符
                    [/\(\*/, "comment", "@comment"], // 块注释开始
                    [/\/\/.*$/, "comment"] // 单行注释
                ]
            }
        }
    }
}]);
```