# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\scroll\simplescrollbars.js`

```
// 定义一个立即执行函数，传入一个 mod 参数
(function(mod) {
  // 如果是 CommonJS 环境，使用 mod 函数引入 CodeMirror
  if (typeof exports == "object" && typeof module == "object") 
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境，使用 define 函数引入 CodeMirror
  else if (typeof define == "function" && define.amd) 
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境，直接使用 mod 函数引入 CodeMirror
  else 
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义一个 Bar 类
  function Bar(cls, orientation, scroll) {
    // 初始化 Bar 对象的属性
    this.orientation = orientation;
    this.scroll = scroll;
    this.screen = this.total = this.size = 1;
    this.pos = 0;

    // 创建一个 div 元素作为 Bar 对象的节点
    this.node = document.createElement("div");
    this.node.className = cls + "-" + orientation;
    this.inner = this.node.appendChild(document.createElement("div"));

    // 绑定鼠标按下事件，实现拖动滚动条的功能
    var self = this;
    CodeMirror.on(this.inner, "mousedown", function(e) {
      if (e.which != 1) return;
      CodeMirror.e_preventDefault(e);
      var axis = self.orientation == "horizontal" ? "pageX" : "pageY";
      var start = e[axis], startpos = self.pos;
      function done() {
        CodeMirror.off(document, "mousemove", move);
        CodeMirror.off(document, "mouseup", done);
      }
      function move(e) {
        if (e.which != 1) return done();
        self.moveTo(startpos + (e[axis] - start) * (self.total / self.size));
      }
      CodeMirror.on(document, "mousemove", move);
      CodeMirror.on(document, "mouseup", done);
    });

    // 绑定点击事件，实现点击滚动条空白区域滚动的功能
    CodeMirror.on(this.node, "click", function(e) {
      CodeMirror.e_preventDefault(e);
      var innerBox = self.inner.getBoundingClientRect(), where;
      if (self.orientation == "horizontal")
        where = e.clientX < innerBox.left ? -1 : e.clientX > innerBox.right ? 1 : 0;
      else
        where = e.clientY < innerBox.top ? -1 : e.clientY > innerBox.bottom ? 1 : 0;
      self.moveTo(self.pos + where * self.screen);
    });
  // 定义鼠标滚轮事件处理函数
  function onWheel(e) {
    // 获取鼠标滚轮滚动的距离
    var moved = CodeMirror.wheelEventPixels(e)[self.orientation == "horizontal" ? "x" : "y"];
    // 保存当前位置
    var oldPos = self.pos;
    // 移动滚动条到新位置
    self.moveTo(self.pos + moved);
    // 如果位置发生变化，则阻止默认事件
    if (self.pos != oldPos) CodeMirror.e_preventDefault(e);
  }
  // 绑定鼠标滚轮事件处理函数
  CodeMirror.on(this.node, "mousewheel", onWheel);
  CodeMirror.on(this.node, "DOMMouseScroll", onWheel);
}

// 设置滚动条位置的方法
Bar.prototype.setPos = function(pos, force) {
  // 如果位置小于0，则设置为0
  if (pos < 0) pos = 0;
  // 如果位置大于总长度减去屏幕长度，则设置为总长度减去屏幕长度
  if (pos > this.total - this.screen) pos = this.total - this.screen;
  // 如果不是强制设置且位置没有变化，则返回false
  if (!force && pos == this.pos) return false;
  // 设置新的位置
  this.pos = pos;
  // 设置滚动条内部的位置
  this.inner.style[this.orientation == "horizontal" ? "left" : "top"] =
    (pos * (this.size / this.total)) + "px";
  return true
};

// 移动滚动条到指定位置的方法
Bar.prototype.moveTo = function(pos) {
  // 如果设置位置成功，则滚动到指定位置
  if (this.setPos(pos)) this.scroll(pos, this.orientation);
}

// 定义最小按钮大小
var minButtonSize = 10;

// 更新滚动条的方法
Bar.prototype.update = function(scrollSize, clientSize, barSize) {
  // 判断滚动条的尺寸是否发生变化
  var sizeChanged = this.screen != clientSize || this.total != scrollSize || this.size != barSize
  // 如果尺寸发生变化，则更新相关属性
  if (sizeChanged) {
    this.screen = clientSize;
    this.total = scrollSize;
    this.size = barSize;
  }
  // 计算按钮的大小
  var buttonSize = this.screen * (this.size / this.total);
  // 如果按钮大小小于最小按钮大小，则调整滚动条大小
  if (buttonSize < minButtonSize) {
    this.size -= minButtonSize - buttonSize;
    buttonSize = minButtonSize;
  }
  // 设置滚动条内部的宽度或高度
  this.inner.style[this.orientation == "horizontal" ? "width" : "height"] =
    buttonSize + "px";
  // 设置滚动条位置
  this.setPos(this.pos, sizeChanged);
};

// 简单滚动条构造函数
function SimpleScrollbars(cls, place, scroll) {
  // 设置滚动条的样式类
  this.addClass = cls;
  // 创建水平滚动条
  this.horiz = new Bar(cls, "horizontal", scroll);
  // 将水平滚动条添加到指定位置
  place(this.horiz.node);
  // 创建垂直滚动条
  this.vert = new Bar(cls, "vertical", scroll);
  // 将垂直滚动条添加到指定位置
  place(this.vert.node);
  // 初始化宽度为null
  this.width = null;
}

// 更新滚动条的方法
SimpleScrollbars.prototype.update = function(measure) {
    # 如果宽度为空，则获取水平节点的计算样式，如果支持则使用 window.getComputedStyle，否则使用当前样式
    var style = window.getComputedStyle ? window.getComputedStyle(this.horiz.node) : this.horiz.node.currentStyle;
    # 如果样式存在，则将宽度设置为样式的高度的整数值
    if (style) this.width = parseInt(style.height);
    
    # 将宽度设置为当前宽度或者0
    var width = this.width || 0;

    # 判断是否需要水平滚动条
    var needsH = measure.scrollWidth > measure.clientWidth + 1;
    # 判断是否需要垂直滚动条
    var needsV = measure.scrollHeight > measure.clientHeight + 1;
    # 根据需要显示或隐藏垂直滚动条
    this.vert.node.style.display = needsV ? "block" : "none";
    # 根据需要显示或隐藏水平滚动条
    this.horiz.node.style.display = needsH ? "block" : "none";

    # 如果需要垂直滚动条
    if (needsV) {
      # 更新垂直滚动条
      this.vert.update(measure.scrollHeight, measure.clientHeight,
                       measure.viewHeight - (needsH ? width : 0));
      # 设置垂直滚动条的位置
      this.vert.node.style.bottom = needsH ? width + "px" : "0";
    }
    # 如果需要水平滚动条
    if (needsH) {
      # 更新水平滚动条
      this.horiz.update(measure.scrollWidth, measure.clientWidth,
                        measure.viewWidth - (needsV ? width : 0) - measure.barLeft);
      # 设置水平滚动条的位置
      this.horiz.node.style.right = needsV ? width + "px" : "0";
      this.horiz.node.style.left = measure.barLeft + "px";
    }

    # 返回滚动条的右侧和底部偏移量
    return {right: needsV ? width : 0, bottom: needsH ? width : 0};
  };

  # 设置垂直滚动条的滚动位置
  SimpleScrollbars.prototype.setScrollTop = function(pos) {
    this.vert.setPos(pos);
  };

  # 设置水平滚动条的滚动位置
  SimpleScrollbars.prototype.setScrollLeft = function(pos) {
    this.horiz.setPos(pos);
  };

  # 清除滚动条
  SimpleScrollbars.prototype.clear = function() {
    # 获取水平滚动条的父节点
    var parent = this.horiz.node.parentNode;
    # 移除水平滚动条节点
    parent.removeChild(this.horiz.node);
    # 移除垂直滚动条节点
    parent.removeChild(this.vert.node);
  };

  # 创建简单滚动条模型
  CodeMirror.scrollbarModel.simple = function(place, scroll) {
    return new SimpleScrollbars("CodeMirror-simplescroll", place, scroll);
  };
  # 创建覆盖滚动条模型
  CodeMirror.scrollbarModel.overlay = function(place, scroll) {
    return new SimpleScrollbars("CodeMirror-overlayscroll", place, scroll);
  };
# 闭合了一个代码块或者函数的结束
```