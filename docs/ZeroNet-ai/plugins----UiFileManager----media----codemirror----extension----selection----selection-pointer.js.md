# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\selection\selection-pointer.js`

```py
// 使用立即执行函数表达式（IIFE）将模块化代码封装起来，避免变量污染全局作用域
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
  "use strict";

  // 定义 CodeMirror 的 selectionPointer 选项
  CodeMirror.defineOption("selectionPointer", false, function(cm, val) {
    var data = cm.state.selectionPointer;
    // 如果已经存在 selectionPointer 数据，则移除相关事件监听和状态
    if (data) {
      CodeMirror.off(cm.getWrapperElement(), "mousemove", data.mousemove);
      CodeMirror.off(cm.getWrapperElement(), "mouseout", data.mouseout);
      CodeMirror.off(window, "scroll", data.windowScroll);
      cm.off("cursorActivity", reset);
      cm.off("scroll", reset);
      cm.state.selectionPointer = null;
      cm.display.lineDiv.style.cursor = "";
    }
    // 如果传入了新的 selectionPointer 值，则设置相关事件监听和状态
    if (val) {
      data = cm.state.selectionPointer = {
        value: typeof val == "string" ? val : "default",
        mousemove: function(event) { mousemove(cm, event); },
        mouseout: function(event) { mouseout(cm, event); },
        windowScroll: function() { reset(cm); },
        rects: null,
        mouseX: null, mouseY: null,
        willUpdate: false
      };
      CodeMirror.on(cm.getWrapperElement(), "mousemove", data.mousemove);
      CodeMirror.on(cm.getWrapperElement(), "mouseout", data.mouseout);
      CodeMirror.on(window, "scroll", data.windowScroll);
      cm.on("cursorActivity", reset);
      cm.on("scroll", reset);
    }
  });

  // 鼠标移动事件处理函数
  function mousemove(cm, event) {
    var data = cm.state.selectionPointer;
    // 如果鼠标按下，则重置鼠标坐标
    if (event.buttons == null ? event.which : event.buttons) {
      data.mouseX = data.mouseY = null;
    } else {
      data.mouseX = event.clientX;
      data.mouseY = event.clientY;
    }
    // 调度更新
    scheduleUpdate(cm);
  }

  // 鼠标移出事件处理函数
  function mouseout(cm, event) {
    // ...
  }
  // 如果鼠标移出编辑器区域，则执行以下操作
  if (!cm.getWrapperElement().contains(event.relatedTarget)) {
    // 获取当前光标位置的数据
    var data = cm.state.selectionPointer;
    // 重置鼠标位置
    data.mouseX = data.mouseY = null;
    // 调度更新操作
    scheduleUpdate(cm);
  }

  // 重置编辑器状态
  function reset(cm) {
    // 清空选择指针的矩形区域
    cm.state.selectionPointer.rects = null;
    // 调度更新操作
    scheduleUpdate(cm);
  }

  // 调度更新操作
  function scheduleUpdate(cm) {
    // 如果当前没有调度更新操作
    if (!cm.state.selectionPointer.willUpdate) {
      // 设置将要更新状态为 true
      cm.state.selectionPointer.willUpdate = true;
      // 延迟 50 毫秒后执行更新操作
      setTimeout(function() {
        update(cm);
        // 更新完成后将将要更新状态设置为 false
        cm.state.selectionPointer.willUpdate = false;
      }, 50);
    }
  }

  // 更新操作
  function update(cm) {
    // 获取当前光标位置的数据
    var data = cm.state.selectionPointer;
    // 如果没有数据，则返回
    if (!data) return;
    // 如果矩形区域为空且鼠标位置不为空
    if (data.rects == null && data.mouseX != null) {
      // 初始化矩形区域数组
      data.rects = [];
      // 如果有选中内容，则遍历选中内容的矩形区域并添加到矩形区域数组中
      if (cm.somethingSelected()) {
        for (var sel = cm.display.selectionDiv.firstChild; sel; sel = sel.nextSibling)
          data.rects.push(sel.getBoundingClientRect());
      }
    }
    // 初始化内部状态为 false
    var inside = false;
    // 如果鼠标位置不为空，则遍历矩形区域数组
    if (data.mouseX != null) for (var i = 0; i < data.rects.length; i++) {
      var rect = data.rects[i];
      // 如果鼠标在矩形区域内，则将内部状态设置为 true
      if (rect.left <= data.mouseX && rect.right >= data.mouseX &&
          rect.top <= data.mouseY && rect.bottom >= data.mouseY)
        inside = true;
    }
    // 根据内部状态设置光标样式
    var cursor = inside ? data.value : "";
    if (cm.display.lineDiv.style.cursor != cursor)
      cm.display.lineDiv.style.cursor = cursor;
  }
# 闭合了一个代码块或者函数的结束
```