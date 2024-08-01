# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\5062.16bfe8fe60a42845.js`

```py
"use strict";
// 使用严格模式，确保代码执行在严格的语义和错误检查下

(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[5062],{
    95062:function(e,n,o){
        o.r(n),o.d(n,{conf:function(){return t},language:function(){return r}});
        /*!-----------------------------------------------------------------------------
         * 版权所有 (c) Microsoft Corporation. 保留所有权利。
         * 版本：0.34.1(547870b6881302c5b4ff32173c16d06009e3588f)
         * 根据 MIT 许可发布
         * https://github.com/microsoft/monaco-editor/blob/main/LICENSE.txt
         *-----------------------------------------------------------------------------*/
        var t={
            comments:{lineComment:"'"},
            brackets:[
                ["(",")"],
                ["[","]"],
                ["If","EndIf"],
                ["While","EndWhile"],
                ["For","EndFor"],
                ["Sub","EndSub"]
            ],
            autoClosingPairs:[
                {open:'"',close:'"',notIn:["string","comment"]},
                {open:"(",close:")",notIn:["string","comment"]},
                {open:"[",close:"]",notIn:["string","comment"]}
            ]
        },
        r={
            defaultToken:"",
            tokenPostfix:".sb",
            ignoreCase:!0,
            brackets:[
                {token:"delimiter.array",open:"[",close:"]"},
                {token:"delimiter.parenthesis",open:"(",close:")"},
                {token:"keyword.tag-if",open:"If",close:"EndIf"},
                {token:"keyword.tag-while",open:"While",close:"EndWhile"},
                {token:"keyword.tag-for",open:"For",close:"EndFor"},
                {token:"keyword.tag-sub",open:"Sub",close:"EndSub"}
            ],
            keywords:[
                "Else","ElseIf","EndFor","EndIf","EndSub","EndWhile","For","Goto","If","Step","Sub","Then","To","While"
            ],
            tagwords:["If","Sub","While","For"],
            operators:[
                ">","<","<>","<=",">=","And","Or","+","-","*","/","="
            ],
            identifier:/[a-zA-Z_][\w]*/,
            symbols:/[=><:+\-*\/%\.,]+/,
            escapes:/\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,
            tokenizer:{
                root:[
                    {include:"@whitespace"},
                    [/(@identifier)(?=[.])/,"type"],
                    [/@identifier/,{
                        cases:{
                            "@keywords":{token:"keyword.$0"},
                            "@operators":"operator",
                            "@default":"variable.name"
                        }
                    }],
                    [/([.])(@identifier)/,{
                        cases:{
                            $2:["delimiter","type.member"],
                            "@default":""
                        }
                    }],
                    [/\d*\.\d+/,"number.float"],
                    [/\d+/,"number"],
                    [/[()\[\]]/,"@brackets"],
                    [/@symbols/,{
                        cases:{
                            "@operators":"operator",
                            "@default":"delimiter"
                        }
                    }],
                    [/"/,"string","@string"]
                ],
                whitespace:[
                    [/[\s\t\r\n]+/,""],
                    [/'(.)*$/,"comment"]
                ],
                string:[
                    [/[^\\"]+/,"string"],
                    [/@escapes/,"string.escape"],
                    [/@./,"string.escape.invalid"],
                    [/"/,"string","@pop"]
                ]
            }
        }
    }
}]);
```