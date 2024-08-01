# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\4188.9e6be9855d350383.js`

```py
"use strict";
// 使用严格模式，确保代码执行在严格的语义规则下

(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[4188],{
    14188:function(e,n,o){
        o.r(n),
        o.d(n,{conf:function(){return t},language:function(){return s}});
        /*!-----------------------------------------------------------------------------
         * 版权所有（c）Microsoft Corporation。保留所有权利。
         * 版本：0.34.1(547870b6881302c5b4ff32173c16d06009e3588f)
         * 根据 MIT 许可发布
         * https://github.com/microsoft/monaco-editor/blob/main/LICENSE.txt
         *-----------------------------------------------------------------------------*/
        var t={
            // 定义注释的语法：块注释和行注释
            comments:{blockComment:["/*","*/"],lineComment:"//"},
            // 定义匹配的括号对
            brackets:[["{","}"],["[","]"],["(",")"]],
            // 定义自动闭合的括号对，排除在字符串内部的情况
            autoClosingPairs:[
                {open:"{",close:"}",notIn:["string"]},
                {open:"[",close:"]",notIn:["string"]},
                {open:"(",close:")",notIn:["string"]},
                {open:'"',close:'"',notIn:["string"]},
                {open:"'",close:"'",notIn:["string"]}
            ],
            // 定义环绕选项的括号对
            surroundingPairs:[
                {open:"{",close:"}"},
                {open:"[",close:"]"},
                {open:"(",close:")"},
                {open:'"',close:'"'},
                {open:"'",close:"'"},
                {open:"<",close:">"}
            ]
        },
        s={defaultToken:"",tokenPostfix:".flow",
            // 定义关键字列表
            keywords:["import","require","export","forbid","native","if","else","cast","unsafe","switch","default"],
            // 定义类型列表
            types:["io","mutable","bool","int","double","string","flow","void","ref","true","false","with"],
            // 定义运算符列表
            operators:["=",">","<","<=",">=","==","!","!=","::=","&&","||","+","-","*","/","@","&","%",":","->","\\","$","??","^"],
            // 定义符号匹配正则表达式
            symbols:/[@$=><!~?:&|+\-*\\\/\^%]+/,
            // 定义转义字符匹配正则表达式
            escapes:/\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,
            // 定义词法分析器
            tokenizer:{
                root:[
                    // 匹配标识符（关键字、类型、默认）
                    [/[a-zA-Z_]\w*/,{
                        cases:{"@keywords":"keyword","@types":"type","@default":"identifier"}
                    }],
                    // 包含空白字符
                    {include:"@whitespace"},
                    // 匹配括号
                    [/[{}()\[\]]/,"delimiter"],
                    // 匹配尖括号，但排除符号
                    [/[<>](?!@symbols)/,"delimiter"],
                    // 匹配运算符或符号
                    [/[@symbols/,{
                        cases:{"@operators":"delimiter","@default":""}
                    }],
                    // 匹配数字
                    [/((0(x|X)[0-9a-fA-F]*)|(([0-9]+\.?[0-9]*)|(\.[0-9]+))((e|E)(\+|-)?[0-9]+)?)/,"number"],
                    // 匹配分隔符
                    [/[;,.]/,"delimiter"],
                    // 匹配无效字符串
                    [/"([^"\\]|\\.)*$/,"string.invalid"],
                    // 匹配双引号开头的字符串
                    [/"/,"string","@string"]
                ],
                // 匹配空白字符
                whitespace:[
                    [/[ \t\r\n]+/,""],
                    // 匹配块注释
                    [/\/*,"comment","@comment"],
                    // 匹配行注释
                    [/\/\/.*$/,"comment"]
                ],
                // 匹配注释
                comment:[
                    // 匹配非注释内容
                    [/[^\/*]+/,"comment"],
                    // 匹配注释结束标志
                    [/\*\//,"comment","@pop"],
                    // 匹配注释符号
                    [/[\/\*]/,"comment"]
                ],
                // 匹配字符串
                string:[
                    // 匹配非转义字符的字符串内容
                    [/[^\\""]+/,"string"],
                    // 匹配转义字符
                    [/@escapes/,"string.escape"],
                    // 匹配无效的转义字符
                    [/\\./,"string.escape.invalid"],
                    // 匹配字符串结束标志
                    [/"/,"string","@pop"]
                ]
            }
        }
    }
}]);
```