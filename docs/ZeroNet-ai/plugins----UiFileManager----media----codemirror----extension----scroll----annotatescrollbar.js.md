# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\scroll\annotatescrollbar.js`

```py
// 将代码包装在自执行函数中，传入一个 mod 参数
(function(mod) {
  // 如果是 CommonJS 环境，使用 mod 函数引入 CodeMirror
  if (typeof exports == "object" && typeof module == "object") 
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境，使用 define 函数引入 CodeMirror
  else if (typeof define == "function" && define.amd) 
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境，直接调用 mod 函数
  else 
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义 CodeMirror 的扩展方法 annotateScrollbar
  CodeMirror.defineExtension("annotateScrollbar", function(options) {
    // 如果 options 是字符串，转换为对象
    if (typeof options == "string") options = {className: options};
    // 返回一个新的 Annotation 实例
    return new Annotation(this, options);
  });

  // 定义 CodeMirror 的配置选项 scrollButtonHeight，默认值为 0
  CodeMirror.defineOption("scrollButtonHeight", 0);

  // 定义 Annotation 类
  function Annotation(cm, options) {
    this.cm = cm;
    this.options = options;
    // 获取滚动条按钮的高度
    this.buttonHeight = options.scrollButtonHeight || cm.getOption("scrollButtonHeight");
    this.annotations = [];
    this.doRedraw = this.doUpdate = null;
    // 在 CodeMirror 的包裹元素中创建一个 div 元素，用于显示注解
    this.div = cm.getWrapperElement().appendChild(document.createElement("div"));
    this.div.style.cssText = "position: absolute; right: 0; top: 0; z-index: 7; pointer-events: none";
    // 计算滚动条的比例
    this.computeScale();

    // 定义 scheduleRedraw 函数，用于延迟重绘注解
    function scheduleRedraw(delay) {
      clearTimeout(self.doRedraw);
      self.doRedraw = setTimeout(function() { self.redraw(); }, delay);
    }

    var self = this;
    // 监听 CodeMirror 的 refresh 事件，调整注解的大小和位置
    cm.on("refresh", this.resizeHandler = function() {
      clearTimeout(self.doUpdate);
      self.doUpdate = setTimeout(function() {
        if (self.computeScale()) scheduleRedraw(20);
      }, 100);
    });
    // 监听 markerAdded 和 markerCleared 事件，调整注解的大小和位置
    cm.on("markerAdded", this.resizeHandler);
    cm.on("markerCleared", this.resizeHandler);
    // 如果 options 中 listenForChanges 不为 false，则监听 changes 事件，调整注解的大小和位置
    if (options.listenForChanges !== false)
      cm.on("changes", this.changeHandler = function() {
        scheduleRedraw(250);
      });
  }

  // 计算滚动条的比例
  Annotation.prototype.computeScale = function() {
    var cm = this.cm;
    // 计算水平滚动条的比例
    var hScale = (cm.getWrapperElement().clientHeight - cm.display.barHeight - this.buttonHeight * 2) /
      cm.getScrollerElement().scrollHeight
    # 如果传入的水平比例与当前比例不相等
    if (hScale != this.hScale) {
      # 更新当前比例
      this.hScale = hScale;
      # 返回 true
      return true;
    }
  };

  # 更新注释
  Annotation.prototype.update = function(annotations) {
    # 更新注释内容
    this.annotations = annotations;
    # 重新绘制
    this.redraw();
  };

  # 重新绘制注释
  Annotation.prototype.redraw = function(compute) {
    # 如果需要计算比例
    if (compute !== false) this.computeScale();
    # 获取当前编辑器和水平比例
    var cm = this.cm, hScale = this.hScale;

    # 创建文档片段和注释
    var frag = document.createDocumentFragment(), anns = this.annotations;

    # 获取文本是否换行和单行高度
    var wrapping = cm.getOption("lineWrapping");
    var singleLineH = wrapping && cm.defaultTextHeight() * 1.5;
    var curLine = null, curLineObj = null;
    # 获取位置的 Y 坐标
    function getY(pos, top) {
      if (curLine != pos.line) {
        curLine = pos.line;
        curLineObj = cm.getLineHandle(curLine);
      }
      if ((curLineObj.widgets && curLineObj.widgets.length) ||
          (wrapping && curLineObj.height > singleLineH))
        return cm.charCoords(pos, "local")[top ? "top" : "bottom"];
      var topY = cm.heightAtLine(curLineObj, "local");
      return topY + (top ? 0 : curLineObj.height);
    }

    # 获取最后一行的行号
    var lastLine = cm.lastLine()
    # 如果显示条宽度存在，则遍历注释数组
    if (cm.display.barWidth) for (var i = 0, nextTop; i < anns.length; i++) {
      # 获取当前注释对象
      var ann = anns[i];
      # 如果注释的结束行大于最后一行，则继续下一次循环
      if (ann.to.line > lastLine) continue;
      # 获取注释起始行的纵坐标
      var top = nextTop || getY(ann.from, true) * hScale;
      # 获取注释结束行的纵坐标
      var bottom = getY(ann.to, false) * hScale;
      # 循环直到找到下一个不相交的注释
      while (i < anns.length - 1) {
        if (anns[i + 1].to.line > lastLine) break;
        nextTop = getY(anns[i + 1].from, true) * hScale;
        if (nextTop > bottom + .9) break;
        ann = anns[++i];
        bottom = getY(ann.to, false) * hScale;
      }
      # 如果注释的起始和结束纵坐标相等，则继续下一次循环
      if (bottom == top) continue;
      # 计算注释的高度
      var height = Math.max(bottom - top, 3);

      # 创建一个 div 元素作为注释的容器
      var elt = frag.appendChild(document.createElement("div"));
      # 设置 div 元素的样式
      elt.style.cssText = "position: absolute; right: 0px; width: " + Math.max(cm.display.barWidth - 1, 2) + "px; top: "
        + (top + this.buttonHeight) + "px; height: " + height + "px";
      # 设置 div 元素的类名
      elt.className = this.options.className;
      # 如果注释有 id，则设置注释的属性
      if (ann.id) {
        elt.setAttribute("annotation-id", ann.id);
      }
    }
    # 清空注释容器的文本内容
    this.div.textContent = "";
    # 将创建的 div 元素添加到注释容器中
    this.div.appendChild(frag);
  };

  # 清除注释
  Annotation.prototype.clear = function() {
    # 移除事件监听器
    this.cm.off("refresh", this.resizeHandler);
    this.cm.off("markerAdded", this.resizeHandler);
    this.cm.off("markerCleared", this.resizeHandler);
    if (this.changeHandler) this.cm.off("changes", this.changeHandler);
    # 移除注释容器
    this.div.parentNode.removeChild(this.div);
  };
# 闭合了一个代码块或者函数的结束
```