# `ZeroNet\plugins\UiFileManager\media\codemirror\mode\rust.js`

```
// CodeMirror, copyright (c) by Marijn Haverbeke and others
// Distributed under an MIT license: https://codemirror.net/LICENSE

// 匿名函数，传入 mod 参数
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror"), require("../../addon/mode/simple"));
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror", "../../addon/mode/simple"], mod);
  // 如果是普通浏览器环境
  else
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义简单模式 "rust"
  CodeMirror.defineSimpleMode("rust",{
    start: [
      // 字符串和字节字符串
      {regex: /b?"/, token: "string", next: "string"},
      // 原始字符串和原始字节字符串
      {regex: /b?r"/, token: "string", next: "string_raw"},
      {regex: /b?r#+"/, token: "string", next: "string_raw_hash"},
      // 字符
      {regex: /'(?:[^'\\]|\\(?:[nrt0'"]|x[\da-fA-F]{2}|u\{[\da-fA-F]{6}\}))'/, token: "string-2"},
      // 字节
      {regex: /b'(?:[^']|\\(?:['\\nrt0]|x[\da-fA-F]{2}))'/, token: "string-2"},

      // 数字
      {regex: /(?:(?:[0-9][0-9_]*)(?:(?:[Ee][+-]?[0-9_]+)|\.[0-9_]+(?:[Ee][+-]?[0-9_]+)?)(?:f32|f64)?)|(?:0(?:b[01_]+|(?:o[0-7_]+)|(?:x[0-9a-fA-F_]+))|(?:[0-9][0-9_]*))(?:u8|u16|u32|u64|i8|i16|i32|i64|isize|usize)?/,
       token: "number"},
      // 关键字
      {regex: /(let(?:\s+mut)?|fn|enum|mod|struct|type|union)(\s+)([a-zA-Z_][a-zA-Z0-9_]*)/, token: ["keyword", null, "def"]},
      {regex: /(?:abstract|alignof|as|async|await|box|break|continue|const|crate|do|dyn|else|enum|extern|fn|for|final|if|impl|in|loop|macro|match|mod|move|offsetof|override|priv|proc|pub|pure|ref|return|self|sizeof|static|struct|super|trait|type|typeof|union|unsafe|unsized|use|virtual|where|while|yield)\b/, token: "keyword"},
      {regex: /\b(?:Self|isize|usize|char|bool|u8|u16|u32|u64|f16|f32|f64|i8|i16|i32|i64|str|Option)\b/, token: "atom"},
      {regex: /\b(?:true|false|Some|None|Ok|Err)\b/, token: "builtin"},
      {regex: /\b(fn)(\s+)([a-zA-Z_][a-zA-Z0-9_]*)/,
       token: ["keyword", null ,"def"]},
    # 定义正则表达式匹配注释标记的模式，并设置对应的 token 类型
    {regex: /#!?\[.*\]/, token: "meta"},
    # 定义正则表达式匹配单行注释的模式，并设置对应的 token 类型
    {regex: /\/\/.*/, token: "comment"},
    # 定义正则表达式匹配多行注释的开始标记的模式，并设置对应的 token 类型，并指定下一个状态为 "comment"
    {regex: /\/\*/, token: "comment", next: "comment"},
    # 定义正则表达式匹配运算符的模式，并设置对应的 token 类型
    {regex: /[-+\/*=<>!]+/, token: "operator"},
    # 定义正则表达式匹配变量的模式，并设置对应的 token 类型
    {regex: /[a-zA-Z_]\w*!/,token: "variable-3"},
    # 定义正则表达式匹配变量的模式，并设置对应的 token 类型
    {regex: /[a-zA-Z_]\w*/, token: "variable"},
    # 定义正则表达式匹配代码块开始标记的模式，并设置缩进
    {regex: /[\{\[\(]/, indent: true},
    # 定义正则表达式匹配代码块结束标记的模式，并设置取消缩进
    {regex: /[\}\]\)]/, dedent: true}
  ],
  string: [
    # 定义正则表达式匹配双引号字符串的开始标记的模式，并设置下一个状态为 "start"
    {regex: /"/, token: "string", next: "start"},
    # 定义正则表达式匹配双引号字符串内容的模式，并设置对应的 token 类型
    {regex: /(?:[^\\"]|\\(?:.|$))*/, token: "string"}
  ],
  string_raw: [
    # 定义正则表达式匹配原始字符串的开始标记的模式，并设置下一个状态为 "start"
    {regex: /"/, token: "string", next: "start"},
    # 定义正则表达式匹配原始字符串内容的模式，并设置对应的 token 类型
    {regex: /[^"]*/, token: "string"}
  ],
  string_raw_hash: [
    # 定义正则表达式匹配带有 # 号的原始字符串的开始标记的模式，并设置下一个状态为 "start"
    {regex: /"#+/, token: "string", next: "start"},
    # 定义正则表达式匹配带有 # 号的原始字符串内容的模式，并设置对应的 token 类型
    {regex: /(?:[^"]|"(?!#))*/, token: "string"}
  ],
  comment: [
    # 定义正则表达式匹配多行注释结束标记的模式，并设置下一个状态为 "start"
    {regex: /.*?\*\//, token: "comment", next: "start"},
    # 定义正则表达式匹配多行注释内容的模式，并设置对应的 token 类型
    {regex: /.*/, token: "comment"}
  ],
  meta: {
    # 设置不缩进的状态为 "comment"
    dontIndentStates: ["comment"],
    # 设置电动输入的正则表达式
    electricInput: /^\s*\}$/,
    # 设置多行注释的开始标记
    blockCommentStart: "/*",
    # 设置多行注释的结束标记
    blockCommentEnd: "*/",
    # 设置单行注释的标记
    lineComment: "//",
    # 设置折叠的标记
    fold: "brace"
  }
# 结束定义Rust源代码的MIME类型
});

# 定义MIME类型"text/x-rustsrc"为"rust"
CodeMirror.defineMIME("text/x-rustsrc", "rust");

# 定义MIME类型"text/rust"为"rust"
CodeMirror.defineMIME("text/rust", "rust");
});
```