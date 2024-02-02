# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\lint\json-lint.js`

```py
// CodeMirror, copyright (c) by Marijn Haverbeke and others
// Distributed under an MIT license: https://codemirror.net/LICENSE

// Depends on jsonlint.js from https://github.com/zaach/jsonlint

// declare global: jsonlint

// 使用立即执行函数表达式（IIFE）将模块包装起来，避免变量污染全局作用域
(function(mod) {
  // 如果是 CommonJS 环境，使用 require 导入 CodeMirror 模块
  if (typeof exports == "object" && typeof module == "object") // CommonJS
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境，使用 define 导入 CodeMirror 模块
  else if (typeof define == "function" && define.amd) // AMD
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境，直接使用 CodeMirror
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 注册 JSON 的 lint 方法
  CodeMirror.registerHelper("lint", "json", function(text) {
    var found = [];
    // 如果 window.jsonlint 未定义，输出错误信息并返回空数组
    if (!window.jsonlint) {
      if (window.console) {
        window.console.error("Error: window.jsonlint not defined, CodeMirror JSON linting cannot run.");
      }
      return found;
    }
    // 对于 jsonlint 的 web dist，jsonlint 被导出为一个对象，其中包含一个名为 parser 的属性，parseError 是其子属性
    var jsonlint = window.jsonlint.parser || window.jsonlint
    // 定义 jsonlint 的 parseError 方法，将错误信息添加到 found 数组中
    jsonlint.parseError = function(str, hash) {
      var loc = hash.loc;
      found.push({from: CodeMirror.Pos(loc.first_line - 1, loc.first_column),
                  to: CodeMirror.Pos(loc.last_line - 1, loc.last_column),
                  message: str});
    };
    // 尝试解析 JSON 文本，捕获可能的异常
    try { jsonlint.parse(text); }
    catch(e) {}
    // 返回找到的错误信息数组
    return found;
  });

});
```