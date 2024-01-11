# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\scroll\scrollpastend.js`

```
// 使用立即执行函数表达式（IIFE）将代码封装在闭包中，避免全局作用域污染
(function(mod) {
  // 如果是 CommonJS 环境，使用 require 导入 CodeMirror 模块
  if (typeof exports == "object" && typeof module == "object") 
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境，使用 define 导入 CodeMirror 模块
  else if (typeof define == "function" && define.amd) 
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境，直接使用全局对象中的 CodeMirror
  else 
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义 CodeMirror 的 scrollPastEnd 选项
  CodeMirror.defineOption("scrollPastEnd", false, function(cm, val, old) {
    // 如果旧值存在且不是初始值，则移除事件监听和底部填充
    if (old && old != CodeMirror.Init) {
      cm.off("change", onChange);
      cm.off("refresh", updateBottomMargin);
      cm.display.lineSpace.parentNode.style.paddingBottom = "";
      cm.state.scrollPastEndPadding = null;
    }
    // 如果新值为真，则添加事件监听和底部填充
    if (val) {
      cm.on("change", onChange);
      cm.on("refresh", updateBottomMargin);
      updateBottomMargin(cm);
    }
  });

  // 当编辑器内容改变时触发的事件处理函数
  function onChange(cm, change) {
    if (CodeMirror.changeEnd(change).line == cm.lastLine())
      updateBottomMargin(cm);
  }

  // 更新底部填充的函数
  function updateBottomMargin(cm) {
    var padding = "";
    // 如果编辑器行数大于1，则计算底部填充的高度
    if (cm.lineCount() > 1) {
      var totalH = cm.display.scroller.clientHeight - 30,
          lastLineH = cm.getLineHandle(cm.lastLine()).height;
      padding = (totalH - lastLineH) + "px";
    }
    // 如果当前填充值与之前的填充值不同，则更新填充值并重新设置编辑器大小
    if (cm.state.scrollPastEndPadding != padding) {
      cm.state.scrollPastEndPadding = padding;
      cm.display.lineSpace.parentNode.style.paddingBottom = padding;
      cm.off("refresh", updateBottomMargin);
      cm.setSize();
      cm.on("refresh", updateBottomMargin);
    }
  }
});
```