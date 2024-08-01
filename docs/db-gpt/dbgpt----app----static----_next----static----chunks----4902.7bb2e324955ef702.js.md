# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\4902.7bb2e324955ef702.js`

```py
"use strict";
// 使用严格模式

(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[4902],{
// 将代码块推送到 webpack 模块中

    4902:function(e,t,n){
    // 定义模块函数，接受 e, t, n 三个参数

        n.r(t),
        n.d(t,{conf:function(){return d},language:function(){return s}});
    // 导出模块的部分内容：conf 和 language 函数

        var o,a=n(9869),
        // 导入模块 9869 并赋值给 a
            i=Object.defineProperty,
            r=Object.getOwnPropertyDescriptor,
            c=Object.getOwnPropertyNames,
            l=Object.prototype.hasOwnProperty,
        // 导入多个对象和函数并赋值给变量

            m=(e,t,n,o)=>{
            // 定义函数 m，接受四个参数 e, t, n, o

                if(t&&"object"==typeof t||"function"==typeof t)
                // 如果 t 是对象或函数

                    for(let a of c(t))
                    // 遍历 t 的所有属性名

                        l.call(e,a)||a===n||i(e,a,{get:()=>t[a],enumerable:!(o=r(t,a))||o.enumerable});
                // 如果 e 没有该属性，定义该属性的 getter 方法

                return e
                // 返回对象 e
            },

            u={};
        // 创建空对象 u

        m(u,a,"default"),
        o&&m(o,a,"default");
        // 将 a 对象的属性赋值给对象 u 和 o（如果存在）

        var d={
            comments:{blockComment:["<!--","-->"]},
            brackets:[["<",">"]],
            autoClosingPairs:[
                {open:"<",close:">"},
                {open:"'",close:"'"},
                {open:'"',close:'"'}
            ],
            surroundingPairs:[
                {open:"<",close:">"},
                {open:"'",close:"'"},
                {open:'"',close:'"'}
            ],
            onEnterRules:[
                {
                    beforeText:RegExp("<([_:\\w][_:\\w-.\\d]*)([^/>]*(?!/)>)[^<]*$","i"),
                    afterText:/^<\/([_:\w][_:\w-.\d]*)\s*>$/i,
                    action:{indentAction:u.languages.IndentAction.IndentOutdent}
                },
                {
                    beforeText:RegExp("<(\\w[\\w\\d]*)([^/>]*(?!/)>)[^<]*$","i"),
                    action:{indentAction:u.languages.IndentAction.Indent}
                }
            ]
        },
        s={defaultToken:"",tokenPostfix:".xml",ignoreCase:!0,qualifiedName:/(?:[\w\.\-]+:)?[\w\.\-]+/,
        // 定义对象 s 的属性

        tokenizer:{
            root:[
                [/[^<&]+/,""],
                {include:"@whitespace"},
                [
                    /(<)(@qualifiedName)/,
                    [{token:"delimiter"},{token:"tag",next:"@tag"}]
                ],
                [
                    /(<\/)(@qualifiedName)(\s*)(>)/,
                    [{token:"delimiter"},{token:"tag"},"",{token:"delimiter"}]
                ],
                [
                    /(<\?)(@qualifiedName)/,
                    [{token:"delimiter"},{token:"metatag",next:"@tag"}]
                ],
                [
                    /(<\!)(@qualifiedName)/,
                    [{token:"delimiter"},{token:"metatag",next:"@tag"}]
                ],
                [
                    /<\!\[CDATA\[/,
                    {token:"delimiter.cdata",next:"@cdata"}
                ],
                [
                    /&\w+;/,
                    "string.escape"
                ]
            ],
            cdata:[
                [/[^<\]]+/,""],
                [
                    /\]\]>/,
                    {token:"delimiter.cdata",next:"@pop"}
                ],
                [
                    /\]/,
                    ""
                ]
            ],
            tag:[
                [
                    /[ \t\r\n]+/,
                    ""
                ],
                [
                    /(@qualifiedName)(\s*=\s*)("[^"]*"|'[^']*')/,
                    ["attribute.name","","attribute.value"]
                ],
                [
                    /(@qualifiedName)(\s*=\s*)("[^">?\/]*|'[^'>?\/]*)/,
                    ["attribute.name","","attribute.value"]
                ],
                [
                    /(@qualifiedName)(\s*=\s*)("[^">]*|'[^'>]*)/,
                    ["attribute.name","","attribute.value"]
                ],
                [
                    /@qualifiedName/,
                    "attribute.name"
                ],
                [
                    /\?>/,
                    {token:"delimiter",next:"@pop"}
                ],
                [
                    /(\/)(>)/,
                    [{token:"tag"},{token:"delimiter",next:"@pop"}]
                ],
                [
                    />/,
                    {token:"delimiter",next:"@pop"}
                ]
            ],
            whitespace:[
                [
                    /[ \t\r\n]+/,
                    ""
                ],
                [
                    /<!--/,
                    {token:"comment",next:"@comment"}
                ]
            ],
            comment:[
                [
                    /[^<\-]+/,
                    "comment.content"
                ],
                [
                    /-->/,
                    {token:"comment",next:"@pop"}
                ],
                [
                    /<!--/,
                    "comment.content.invalid"
                ],
                [
                    /[<\-]/,
                    "comment.content"
                ]
            ]
        }
        }
        // 定义 XML 文件的语法高亮规则

    }
// 结束模块函数定义
]);
// 结束 webpack 模块代码块
```