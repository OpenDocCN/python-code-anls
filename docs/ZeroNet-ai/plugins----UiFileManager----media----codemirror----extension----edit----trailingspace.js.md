# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\edit\trailingspace.js`

```py
// 使用立即执行函数表达式（IIFE）将代码封装在闭包中，避免全局作用域污染
(function(mod) {
  // 如果是 CommonJS 环境，使用 require 引入 CodeMirror 模块
  if (typeof exports == "object" && typeof module == "object") 
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境，使用 define 引入 CodeMirror 模块
  else if (typeof define == "function" && define.amd) 
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境，直接引入 CodeMirror 模块
  else 
    mod(CodeMirror);
})(function(CodeMirror) {
  // 定义 CodeMirror 的 showTrailingSpace 选项
  CodeMirror.defineOption("showTrailingSpace", false, function(cm, val, prev) {
    // 如果之前的值是初始化状态，则设置为 false
    if (prev == CodeMirror.Init) prev = false;
    // 如果之前是 true，但现在是 false，则移除尾随空格的覆盖层
    if (prev && !val)
      cm.removeOverlay("trailingspace");
    // 如果之前是 false，但现在是 true，则添加尾随空格的覆盖层
    else if (!prev && val)
      cm.addOverlay({
        // 定义覆盖层的 token 方法，用于标记尾随空格
        token: function(stream) {
          for (var l = stream.string.length, i = l; i && /\s/.test(stream.string.charAt(i - 1)); --i) {}
          if (i > stream.pos) { stream.pos = i; return null; }
          stream.pos = l;
          return "trailingspace";
        },
        name: "trailingspace"
      });
  });
});
```