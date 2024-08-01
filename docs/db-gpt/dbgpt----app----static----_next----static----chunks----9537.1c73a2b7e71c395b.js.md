# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\9537.1c73a2b7e71c395b.js`

```py
"use strict";
// 使用严格模式，确保代码执行在严格的语法和错误检查环境中

(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[9537],{
    79537:function(e,n,o){
        o.r(n),o.d(n,{conf:function(){return t},language:function(){return s}});
        /*!-----------------------------------------------------------------------------
         * 版权所有 (c) Microsoft Corporation. 保留所有权利。
         * 版本: 0.34.1(547870b6881302c5b4ff32173c16d06009e3588f)
         * 根据 MIT 许可发布
         * https://github.com/microsoft/monaco-editor/blob/main/LICENSE.txt
         *-----------------------------------------------------------------------------*/
        var t={
            // 定义语言的注释配置
            comments:{lineComment:"//",blockComment:["/*","*/"]},
            // 定义语言的括号配对
            brackets:[["{","}"],["[","]"],["(",")"]],
            // 定义自动闭合的括号对
            autoClosingPairs:[
                {open:"{",close:"}"},
                {open:"[",close:"]"},
                {open:"(",close:")"},
                {open:'"',close:'"'},
                {open:"'",close:"'"}
            ],
            // 定义包围选择的括号对
            surroundingPairs:[
                {open:"{",close:"}"},
                {open:"[",close:"]"},
                {open:"(",close:")"},
                {open:'"',close:'"'},
                {open:"'",close:"'"}
            ]
        },
        s={defaultToken:"",tokenPostfix:".objective-c",
            keywords:["#import","#include","#define","#else","#endif","#if","#ifdef","#ifndef","#ident","#undef","@class","@defs","@dynamic","@encode","@end","@implementation","@interface","@package","@private","@protected","@property","@protocol","@public","@selector","@synthesize","__declspec","assign","auto","BOOL","break","bycopy","byref","case","char","Class","const","copy","continue","default","do","double","else","enum","extern","FALSE","false","float","for","goto","if","in","int","id","inout","IMP","long","nil","nonatomic","NULL","oneway","out","private","public","protected","readwrite","readonly","register","return","SEL","self","short","signed","sizeof","static","struct","super","switch","typedef","TRUE","true","union","unsigned","volatile","void","while"],
            decpart:/\d(_?\d)*/,
            decimal:/0|@decpart/,
            tokenizer:{
                root:[
                    {include:"@comments"},
                    {include:"@whitespace"},
                    {include:"@numbers"},
                    {include:"@strings"},
                    [/[,;]/,"delimiter"],
                    [/[\{\[\(\)<>]/,"@brackets"],
                    [/^[a-zA-Z@#]\w*/,{cases:{"@keywords":"keyword","@default":"identifier"}}],
                    [/[\+\\-\\*\\/\\^\\|\\~,]|and\\b|or\\b|not\\b/,"operator"]
                ],
                whitespace:[[ /\s+/,"white" ]],
                comments:[
                    ["\\/\\*", "comment", "@comment"],
                    ["\\/\\/+.*","comment"]
                ],
                comment:[
                    ["\\*\\/", "comment", "@pop"],
                    [".","comment"]
                ],
                numbers:[
                    [/0[xX][0-9a-fA-F]*(_?[0-9a-fA-F])*/,"number.hex"],
                    [/0|@decimal((\.@decpart)?([eE][\-+]?@decpart)?)[fF]*/,{
                        cases:{
                            "(\\d)*":"number",
                            "$0":"number.float"
                        }
                    }]
                ],
                strings:[
                    [/'$/,"string.escape","@popall"],
                    [/'/,"string.escape","@stringBody"],
                    [/"$/,"string.escape","@popall"],
                    [/"/,"string.escape","@dblStringBody"]
                ],
                stringBody:[
                    [/[^\']+$/,"string","@popall"],
                    [/[^\']+/,"string"],
                    [/\\./,"string"],
                    [/'/,"string.escape","@popall"],
                    [/\\$/,"string"]
                ],
                dblStringBody:[
                    [/[^\"]+$/,"string","@popall"],
                    [/[^\"]+/,"string"],
                    [/\\./,"string"],
                    [/"/,"string.escape","@popall"],
                    [/\\$/,"string"]
                ]
            }
        }
    }
]);
```