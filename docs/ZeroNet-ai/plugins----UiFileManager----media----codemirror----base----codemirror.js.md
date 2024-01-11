# `ZeroNet\plugins\UiFileManager\media\codemirror\base\codemirror.js`

```
// CodeMirror, 版权归 Marijn Haverbeke 和其他人所有
// 根据 MIT 许可证分发：https://codemirror.net/LICENSE

// 这是 CodeMirror（https://codemirror.net），一个基于浏览器 DOM 实现的 JavaScript 代码编辑器。

// 你可以在 http://marijnhaverbeke.nl/blog/#cm-internals 找到一些代码下面的技术背景。

// 使用立即执行函数表达式（IIFE）来创建一个作用域，防止变量污染全局作用域
(function (global, factory) {
  // 如果是在 Node.js 环境中，则将模块导出为 factory 的返回值
  typeof exports === 'object' && typeof module !== 'undefined' ? module.exports = factory() :
  // 如果是在 AMD 环境中，则使用 define 函数定义模块
  typeof define === 'function' && define.amd ? define(factory) :
  // 否则将 factory 的返回值挂载到全局对象上，即 window 对象
  (global = global || self, global.CodeMirror = factory());
// 严格模式，确保代码更安全
}(this, (function () { 'use strict';

  // 通过 userAgent 和 platform 进行浏览器特性检测
  var userAgent = navigator.userAgent;
  var platform = navigator.platform;

  // 检测是否为 gecko 内核浏览器
  var gecko = /gecko\/\d/i.test(userAgent);
  // 检测是否为 IE10 及以下版本
  var ie_upto10 = /MSIE \d/.test(userAgent);
  // 检测是否为 IE11 及以上版本
  var ie_11up = /Trident\/(?:[7-9]|\d{2,})\..*rv:(\d+)/.exec(userAgent);
  // 检测是否为 Edge 浏览器
  var edge = /Edge\/(\d+)/.exec(userAgent);
  // 检测是否为 IE 浏览器
  var ie = ie_upto10 || ie_11up || edge;
  // 获取 IE 浏览器的版本号
  var ie_version = ie && (ie_upto10 ? document.documentMode || 6 : +(edge || ie_11up)[1]);
  // 检测是否为 WebKit 内核浏览器
  var webkit = !edge && /WebKit\//.test(userAgent);
  // 检测是否为 QtWebKit 浏览器
  var qtwebkit = webkit && /Qt\/\d+\.\d+/.test(userAgent);
  // 检测是否为 Chrome 浏览器
  var chrome = !edge && /Chrome\//.test(userAgent);
  // 检测是否为 Opera 浏览器
  var presto = /Opera\//.test(userAgent);
  // 检测是否为 Safari 浏览器
  var safari = /Apple Computer/.test(navigator.vendor);
  // 检测是否为 Mac OS X Mountain Lion 及以上版本
  var mac_geMountainLion = /Mac OS X 1\d\D([8-9]|\d\d)\D/.test(userAgent);
  // 检测是否为 PhantomJS 浏览器
  var phantom = /PhantomJS/.test(userAgent);

  // 检测是否为 iOS 设备
  var ios = !edge && /AppleWebKit/.test(userAgent) && /Mobile\/\w+/.test(userAgent);
  // 检测是否为 Android 设备
  var android = /Android/.test(userAgent);
  // 检测是否为移动设备
  var mobile = ios || android || /webOS|BlackBerry|Opera Mini|Opera Mobi|IEMobile/i.test(userAgent);
  // 检测是否为 Mac 设备
  var mac = ios || /Mac/.test(platform);
  // 检测是否为 Chrome OS 设备
  var chromeOS = /\bCrOS\b/.test(userAgent);
  // 检测是否为 Windows 设备
  var windows = /win/i.test(platform);

  // 获取 Presto 引擎的版本号
  var presto_version = presto && userAgent.match(/Version\/(\d*\.\d*)/);
  if (presto_version) { presto_version = Number(presto_version[1]); }
  // 如果 Presto 引擎版本号大于等于 15，则将 presto 设置为 false，将 webkit 设置为 true
  if (presto_version && presto_version >= 15) { presto = false; webkit = true; }
  // 一些浏览器在 OS X 上使用错误的事件属性来表示 cmd/ctrl
  var flipCtrlCmd = mac && (qtwebkit || presto && (presto_version == null || presto_version < 12.11));
  // 某些浏览器使用错误的事件属性来表示右键单击
  var captureRightClick = gecko || (ie && ie_version >= 9);

  // 定义 classTest 函数，用于检测是否存在指定类名
  function classTest(cls) { return new RegExp("(^|\\s)" + cls + "(?:$|\\s)\\s*") }

  // 定义 rmClass 函数，用于移除指定节点的指定类名
  var rmClass = function(node, cls) {
  // 获取当前节点的类名
  var current = node.className;
  // 通过正则表达式匹配类名
  var match = classTest(cls).exec(current);
  // 如果匹配成功，则移除匹配到的类名
  if (match) {
    var after = current.slice(match.index + match[0].length);
    node.className = current.slice(0, match.index) + (after ? match[1] + after : "");
  }
};

// 移除节点的所有子节点
function removeChildren(e) {
  for (var count = e.childNodes.length; count > 0; --count)
    { e.removeChild(e.firstChild); }
  return e
}

// 移除节点的所有子节点并添加新节点
function removeChildrenAndAdd(parent, e) {
  return removeChildren(parent).appendChild(e)
}

// 创建元素节点
function elt(tag, content, className, style) {
  var e = document.createElement(tag);
  if (className) { e.className = className; }
  if (style) { e.style.cssText = style; }
  if (typeof content == "string") { e.appendChild(document.createTextNode(content)); }
  else if (content) { for (var i = 0; i < content.length; ++i) { e.appendChild(content[i]); } }
  return e
}
// 包装函数，用于创建元素节点并从可访问性树中移除该元素
function eltP(tag, content, className, style) {
  var e = elt(tag, content, className, style);
  e.setAttribute("role", "presentation");
  return e
}

var range;
// 根据浏览器支持情况选择创建范围的方法
if (document.createRange) { range = function(node, start, end, endNode) {
  var r = document.createRange();
  r.setEnd(endNode || node, end);
  r.setStart(node, start);
  return r
}; }
else { range = function(node, start, end) {
  var r = document.body.createTextRange();
  try { r.moveToElementText(node.parentNode); }
  catch(e) { return r }
  r.collapse(true);
  r.moveEnd("character", end);
  r.moveStart("character", start);
  return r
}; }

// 检查父节点是否包含子节点
function contains(parent, child) {
  if (child.nodeType == 3) // Android browser always returns false when child is a textnode
    { child = child.parentNode; }
  if (parent.contains)
    { return parent.contains(child) }
  do {
    if (child.nodeType == 11) { child = child.host; }
    if (child == parent) { return true }
  // 从当前节点开始，向上遍历其父节点，直到没有父节点为止
  } while (child = child.parentNode)
}

function activeElt() {
  // 在 IE 和 Edge 中，访问 document.activeElement 可能会抛出“未指定的错误”。
  // 在加载页面或在 iframe 中访问时，IE < 10 会抛出错误。
  // 在 iframe 中访问时，IE > 9 和 Edge 会抛出错误，如果 document.body 不可用。
  var activeElement;
  try {
    activeElement = document.activeElement;
  } catch(e) {
    activeElement = document.body || null;
  }
  while (activeElement && activeElement.shadowRoot && activeElement.shadowRoot.activeElement)
    { activeElement = activeElement.shadowRoot.activeElement; }
  return activeElement
}

function addClass(node, cls) {
  var current = node.className;
  // 如果节点的类名中不包含指定的类，则添加指定的类
  if (!classTest(cls).test(current)) { node.className += (current ? " " : "") + cls; }
}
function joinClasses(a, b) {
  var as = a.split(" ");
  for (var i = 0; i < as.length; i++)
    // 如果 b 中不包含 a 中的类，则将 a 中的类添加到 b 中
    { if (as[i] && !classTest(as[i]).test(b)) { b += " " + as[i]; } }
  return b
}

var selectInput = function(node) { node.select(); };
if (ios) // 移动 Safari 显然存在一个 bug，导致 select() 方法无效。
  { selectInput = function(node) { node.selectionStart = 0; node.selectionEnd = node.value.length; }; }
else if (ie) // 抑制神秘的 IE10 错误
  { selectInput = function(node) { try { node.select(); } catch(_e) {} }; }

function bind(f) {
  var args = Array.prototype.slice.call(arguments, 1);
  return function(){return f.apply(null, args)}
}

function copyObj(obj, target, overwrite) {
  if (!target) { target = {}; }
  for (var prop in obj)
    // 如果目标对象中不存在属性，或者允许覆盖已有属性，则将源对象的属性复制到目标对象
    { if (obj.hasOwnProperty(prop) && (overwrite !== false || !target.hasOwnProperty(prop)))
      { target[prop] = obj[prop]; } }
  return target
}

// 计算字符串中的列偏移量，考虑到制表符。
// 主要用于查找缩进。
function countColumn(string, end, tabSize, startIndex, startValue) {
    // 如果结束位置为null，则查找第一个非空白字符的位置，如果没有则结束位置为字符串长度
    if (end == null) {
      end = string.search(/[^\s\u00a0]/);
      if (end == -1) { end = string.length; }
    }
    // 从给定的起始位置开始遍历字符串，计算制表符占据的列数
    for (var i = startIndex || 0, n = startValue || 0;;) {
      // 查找下一个制表符的位置
      var nextTab = string.indexOf("\t", i);
      // 如果没有找到制表符或者制表符位置超过结束位置，则返回计算的列数
      if (nextTab < 0 || nextTab >= end)
        { return n + (end - i) }
      // 计算制表符之前的字符占据的列数，并加上制表符占据的列数
      n += nextTab - i;
      n += tabSize - (n % tabSize);
      i = nextTab + 1;
    }
  }

  // 延迟执行函数的构造函数
  var Delayed = function() {
    this.id = null;
    this.f = null;
    this.time = 0;
    this.handler = bind(this.onTimeout, this);
  };
  // 延迟执行函数的超时处理函数
  Delayed.prototype.onTimeout = function (self) {
    self.id = 0;
    // 如果当前时间超过设定的时间，则执行延迟函数，否则继续延迟
    if (self.time <= +new Date) {
      self.f();
    } else {
      setTimeout(self.handler, self.time - +new Date);
    }
  };
  // 设置延迟执行函数
  Delayed.prototype.set = function (ms, f) {
    this.f = f;
    var time = +new Date + ms;
    // 如果没有id或者新的时间小于之前设定的时间，则清除之前的延迟，重新设置延迟
    if (!this.id || time < this.time) {
      clearTimeout(this.id);
      this.id = setTimeout(this.handler, ms);
      this.time = time;
    }
  };

  // 查找元素在数组中的索引
  function indexOf(array, elt) {
    for (var i = 0; i < array.length; ++i)
      { if (array[i] == elt) { return i } }
    return -1
  }

  // 用于隐藏滚动条的滚动器和调整器添加的像素数
  var scrollerGap = 50;

  // 由各种协议返回或抛出以表示“我不处理这个”
  var Pass = {toString: function(){return "CodeMirror.Pass"}};

  // 用于setSelection等方法的重用选项对象
  var sel_dontScroll = {scroll: false}, sel_mouse = {origin: "*mouse"}, sel_move = {origin: "+move"};

  // countColumn的反函数，查找对应于特定列的偏移量
  function findColumn(string, goal, tabSize) {
    // 初始化变量 pos 为 0，col 为 0，进入无限循环
    for (var pos = 0, col = 0;;) {
      // 查找下一个制表符的位置
      var nextTab = string.indexOf("\t", pos);
      // 如果没有找到制表符，则将下一个制表符位置设置为字符串的长度
      if (nextTab == -1) { nextTab = string.length; }
      // 计算跳过的字符数
      var skipped = nextTab - pos;
      // 如果下一个制表符位置为字符串的长度，或者当前列加上跳过的字符数大于等于目标列数，则返回当前位置加上跳过的字符数和目标列数与当前列的差值的最小值
      if (nextTab == string.length || col + skipped >= goal)
        { return pos + Math.min(skipped, goal - col) }
      // 更新当前列数
      col += nextTab - pos;
      // 将当前列数调整为下一个制表符位置的列数
      col += tabSize - (col % tabSize);
      // 更新 pos 为下一个制表符位置加 1
      pos = nextTab + 1;
      // 如果当前列数大于等于目标列数，则返回当前位置
      if (col >= goal) { return pos }
    }
  }

  // 初始化空字符串数组
  var spaceStrs = [""];
  // 生成 n 个空格的字符串
  function spaceStr(n) {
    while (spaceStrs.length <= n)
      { spaceStrs.push(lst(spaceStrs) + " "); }
    return spaceStrs[n]
  }

  // 返回数组的最后一个元素
  function lst(arr) { return arr[arr.length-1] }

  // 对数组中的每个元素执行函数 f，并返回结果数组
  function map(array, f) {
    var out = [];
    for (var i = 0; i < array.length; i++) { out[i] = f(array[i], i); }
    return out
  }

  // 将 value 按照 score 的值插入到已排序的数组中
  function insertSorted(array, value, score) {
    var pos = 0, priority = score(value);
    while (pos < array.length && score(array[pos]) <= priority) { pos++; }
    array.splice(pos, 0, value);
  }

  // 空函数
  function nothing() {}

  // 创建一个对象，继承自 base，如果 props 存在，则将其属性复制到新对象中
  function createObj(base, props) {
    var inst;
    if (Object.create) {
      inst = Object.create(base);
    } else {
      nothing.prototype = base;
      inst = new nothing();
    }
    if (props) { copyObj(props, inst); }
    return inst
  }

  // 判断字符是否为单词字符的基本函数
  var nonASCIISingleCaseWordChar = /[\u00df\u0587\u0590-\u05f4\u0600-\u06ff\u3040-\u309f\u30a0-\u30ff\u3400-\u4db5\u4e00-\u9fcc\uac00-\ud7af]/;
  function isWordCharBasic(ch) {
    return /\w/.test(ch) || ch > "\x80" &&
      (ch.toUpperCase() != ch.toLowerCase() || nonASCIISingleCaseWordChar.test(ch))
  }
  // 判断字符是否为单词字符
  function isWordChar(ch, helper) {
    if (!helper) { return isWordCharBasic(ch) }
    if (helper.source.indexOf("\\w") > -1 && isWordCharBasic(ch)) { return true }
    return helper.test(ch)
  }

  // 判断对象是否为空
  function isEmpty(obj) {
    for (var n in obj) { if (obj.hasOwnProperty(n) && obj[n]) { return false } }
    while ((dir < 0 ? pos > 0 : pos < str.length) && isExtendingChar(str.charAt(pos))) { pos += dir; }
  // 返回位置
  return pos
}

// 返回在范围[`from`; `to`]内满足`pred`并且最接近`from`的值。假设至少`to`满足`pred`。支持`from`大于`to`。
function findFirst(pred, from, to) {
  // 在任何时刻，我们确定`to`满足`pred`，但不确定`from`是否满足。
  var dir = from > to ? -1 : 1;
  for (;;) {
    if (from == to) { return from }
    var midF = (from + to) / 2, mid = dir < 0 ? Math.ceil(midF) : Math.floor(midF);
    if (mid == from) { return pred(mid) ? from : to }
    if (pred(mid)) { to = mid; }
    else { from = mid + dir; }
  }
}

// BIDI HELPERS

// 迭代双向文本段落
function iterateBidiSections(order, from, to, f) {
  if (!order) { return f(from, to, "ltr", 0) }
  var found = false;
  for (var i = 0; i < order.length; ++i) {
    var part = order[i];
    if (part.from < to && part.to > from || from == to && part.to == from) {
      f(Math.max(part.from, from), Math.min(part.to, to), part.level == 1 ? "rtl" : "ltr", i);
      found = true;
    }
  }
  if (!found) { f(from, to, "ltr"); }
}

var bidiOther = null;
// 获取指定位置的双向文本段落
function getBidiPartAt(order, ch, sticky) {
  var found;
  bidiOther = null;
  for (var i = 0; i < order.length; ++i) {
    var cur = order[i];
    if (cur.from < ch && cur.to > ch) { return i }
    if (cur.to == ch) {
      if (cur.from != cur.to && sticky == "before") { found = i; }
      else { bidiOther = i; }
    }
    if (cur.from == ch) {
      if (cur.from != cur.to && sticky != "before") { found = i; }
      else { bidiOther = i; }
    }
  }
    return found != null ? found : bidiOther
  }
  
  // 如果找到的字符不为空，则返回找到的字符，否则返回另一个字符
  // 双向排序算法
  // 参见 http://unicode.org/reports/tr9/tr9-13.html 了解该算法的部分实现

  // 用于字符类型的单字符代码：
  // L (L):   从左到右
  // R (R):   从右到左
  // r (AL):  从右到左的阿拉伯语
  // 1 (EN):  欧洲数字
  // + (ES):  欧洲数字分隔符
  // % (ET):  欧洲数字终止符
  // n (AN):  阿拉伯数字
  // , (CS):  通用数字分隔符
  // m (NSM): 非间隔标记
  // b (BN):  边界中性
  // s (B):   段落分隔符
  // t (S):   段分隔符
  // w (WS):  空白字符
  // N (ON):  其他中性字符

  // 如果字符按照它们的视觉顺序排列（从左到右），则返回 null，否则返回一个包含部分实现的章节（{from, to, level} 对象）的数组，按照它们在视觉上出现的顺序排列
  var bidiOrdering = (function() {
    // 代码点从 0 到 0xff 的字符类型
    var lowTypes = "bbbbbbbbbtstwsbbbbbbbbbbbbbbssstwNN%%%NNNNNN,N,N1111111111NNNNNNNLLLLLLLLLLLLLLLLLLLLLLLLLLNNNNNNLLLLLLLLLLLLLLLLLLLLLLLLLLNNNNbbbbbbsbbbbbbbbbbbbbbbbbbbbbbbbbb,N%%%%NNNNLNNNNN%%11NLNNN1LNNNNNLLLLLLLLLLLLLLLLLLLLLLLNLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLN";
    // 代码点从 0x600 到 0x6f9 的字符类型
    var arabicTypes = "nnnnnnNNr%%r,rNNmmmmmmmmmmmrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrmmmmmmmmmmmmmmmmmmmmmnnnnnnnnnn%nnrrrmrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    // 定义正则表达式，用于匹配带有双向文本的字符
    var bidiRE = /[\u0590-\u05f4\u0600-\u06ff\u0700-\u08ac]/;
    // 定义用于判断文本方向的正则表达式
    var isNeutral = /[stwN]/, isStrong = /[LRr]/, countsAsLeft = /[Lb1n]/, countsAsNum = /[1n]/;

    // 定义 BidiSpan 对象的构造函数
    function BidiSpan(level, from, to) {
      this.level = level;
      this.from = from; this.to = to;
    }

    // 获取给定行的双向文本顺序（并缓存它）。对于完全从左到右的行，返回 false，否则返回 BidiSpan 对象的数组。
    function getOrder(line, direction) {
      var order = line.order;
      if (order == null) { order = line.order = bidiOrdering(line.text, direction); }
      return order
    }

    // 事件处理

    // 轻量级事件框架。on/off 也适用于 DOM 节点，注册原生 DOM 处理程序。

    // 空处理程序数组
    var noHandlers = [];

    // 注册事件处理程序
    var on = function(emitter, type, f) {
      if (emitter.addEventListener) {
        emitter.addEventListener(type, f, false);
      } else if (emitter.attachEvent) {
        emitter.attachEvent("on" + type, f);
      } else {
        var map = emitter._handlers || (emitter._handlers = {});
        map[type] = (map[type] || noHandlers).concat(f);
      }
    };

    // 获取事件处理程序
    function getHandlers(emitter, type) {
      return emitter._handlers && emitter._handlers[type] || noHandlers
    }

    // 注销事件处理程序
    function off(emitter, type, f) {
      if (emitter.removeEventListener) {
        emitter.removeEventListener(type, f, false);
      } else if (emitter.detachEvent) {
        emitter.detachEvent("on" + type, f);
      } else {
        var map = emitter._handlers, arr = map && map[type];
        if (arr) {
          var index = indexOf(arr, f);
          if (index > -1)
            { map[type] = arr.slice(0, index).concat(arr.slice(index + 1)); }
        }
      }
    }

    // 触发事件
    function signal(emitter, type /*, values...*/) {
      var handlers = getHandlers(emitter, type);
      if (!handlers.length) { return }
      var args = Array.prototype.slice.call(arguments, 2);
    }
  // 遍历 handlers 数组，依次调用每个元素的 apply 方法，传入 null 和 args 参数
  for (var i = 0; i < handlers.length; ++i) { handlers[i].apply(null, args); }
  }

  // 通过在编辑器上注册一个非 DOM 事件处理程序来覆盖 CodeMirror 处理的 DOM 事件，并在该处理程序中阻止默认行为
  function signalDOMEvent(cm, e, override) {
    // 如果 e 是字符串，则将其转换为包含 type 和 preventDefault 方法的对象
    if (typeof e == "string")
      { e = {type: e, preventDefault: function() { this.defaultPrevented = true; }}; }
    // 调用 signal 函数，传入 cm、override 或 e.type、cm 和 e 作为参数
    signal(cm, override || e.type, cm, e);
    // 返回事件是否被阻止默认行为或被忽略的布尔值
    return e_defaultPrevented(e) || e.codemirrorIgnore
  }

  // 触发光标活动事件
  function signalCursorActivity(cm) {
    // 获取 cm._handlers.cursorActivity 数组
    var arr = cm._handlers && cm._handlers.cursorActivity;
    // 如果 arr 不存在，则直接返回
    if (!arr) { return }
    // 获取 cm.curOp.cursorActivityHandlers，如果不存在则初始化为空数组
    var set = cm.curOp.cursorActivityHandlers || (cm.curOp.cursorActivityHandlers = []);
    // 遍历 arr 数组，将其中的元素添加到 set 数组中
    for (var i = 0; i < arr.length; ++i) { if (indexOf(set, arr[i]) == -1)
      { set.push(arr[i]); } }
  }

  // 检查 emitter 是否有指定类型的事件处理程序
  function hasHandler(emitter, type) {
    // 返回指定类型的事件处理程序数组的长度是否大于 0
    return getHandlers(emitter, type).length > 0
  }

  // 为构造函数的原型添加 on 和 off 方法，以便更方便地在对象上注册事件
  function eventMixin(ctor) {
    ctor.prototype.on = function(type, f) {on(this, type, f);};
    ctor.prototype.off = function(type, f) {off(this, type, f);};
  }

  // 由于我们仍然支持古老的 IE 版本，因此需要一些兼容性包装器

  // 阻止事件的默认行为
  function e_preventDefault(e) {
    if (e.preventDefault) { e.preventDefault(); }
    else { e.returnValue = false; }
  }
  // 停止事件的传播
  function e_stopPropagation(e) {
    if (e.stopPropagation) { e.stopPropagation(); }
    else { e.cancelBubble = true; }
  }
  // 返回事件是否被阻止默认行为
  function e_defaultPrevented(e) {
    return e.defaultPrevented != null ? e.defaultPrevented : e.returnValue == false
  }
  // 停止事件的默认行为和传播
  function e_stop(e) {e_preventDefault(e); e_stopPropagation(e);}

  // 获取事件的目标元素
  function e_target(e) {return e.target || e.srcElement}
  // 获取事件的按钮值
  function e_button(e) {
    var b = e.which;
    // 如果 b 为 null
    if (b == null) {
      // 如果鼠标左键被按下，设置 b 为 1
      if (e.button & 1) { b = 1; }
      // 如果鼠标右键被按下，设置 b 为 3
      else if (e.button & 2) { b = 3; }
      // 如果鼠标中键被按下，设置 b 为 2
      else if (e.button & 4) { b = 2; }
    }
    // 如果是 mac 并且按下了 ctrl 键并且 b 为 1，设置 b 为 3
    if (mac && e.ctrlKey && b == 1) { b = 3; }
    // 返回 b
    return b
  }

  // 检测是否支持拖放
  var dragAndDrop = function() {
    // 在 IE6-8 中有一些拖放支持，但我还没搞定它的工作方式
    if (ie && ie_version < 9) { return false }
    var div = elt('div');
    return "draggable" in div || "dragDrop" in div
  }();

  var zwspSupported;
  // 创建零宽度元素
  function zeroWidthElement(measure) {
    if (zwspSupported == null) {
      var test = elt("span", "\u200b");
      removeChildrenAndAdd(measure, elt("span", [test, document.createTextNode("x")]));
      if (measure.firstChild.offsetHeight != 0)
        { zwspSupported = test.offsetWidth <= 1 && test.offsetHeight > 2 && !(ie && ie_version < 8); }
    }
    var node = zwspSupported ? elt("span", "\u200b") :
      elt("span", "\u00a0", null, "display: inline-block; width: 1px; margin-right: -1px");
    node.setAttribute("cm-text", "");
    return node
  }

  // 检测 IE 对双向文本的糟糕客户端矩形报告特性
  var badBidiRects;
  function hasBadBidiRects(measure) {
    if (badBidiRects != null) { return badBidiRects }
    var txt = removeChildrenAndAdd(measure, document.createTextNode("A\u062eA"));
    var r0 = range(txt, 0, 1).getBoundingClientRect();
    var r1 = range(txt, 1, 2).getBoundingClientRect();
    removeChildren(measure);
    if (!r0 || r0.left == r0.right) { return false } // Safari 在某些情况下返回 null (#2780)
    return badBidiRects = (r1.right - r0.right < 3)
  }

  // 检测 "".split 是否是破损的 IE 版本，如果是，提供替代的分割行的方法
  var splitLinesAuto = "\n\nb".split(/\n/).length != 3 ? function (string) {
    var pos = 0, result = [], l = string.length;
    // 当位置小于等于字符串长度时，执行循环
    while (pos <= l) {
      // 查找换行符的位置
      var nl = string.indexOf("\n", pos);
      // 如果没有找到换行符，则将位置设置为字符串长度
      if (nl == -1) { nl = string.length; }
      // 截取从当前位置到换行符之间的字符串
      var line = string.slice(pos, string.charAt(nl - 1) == "\r" ? nl - 1 : nl);
      // 查找回车符的位置
      var rt = line.indexOf("\r");
      // 如果找到回车符
      if (rt != -1) {
        // 将回车符之前的部分添加到结果数组中
        result.push(line.slice(0, rt));
        // 更新位置
        pos += rt + 1;
      } else {
        // 如果没有回车符，则将整行添加到结果数组中
        result.push(line);
        // 更新位置
        pos = nl + 1;
      }
    }
    // 返回结果数组
    return result
  } : function (string) { return string.split(/\r\n?|\n/); };

  // 检查是否存在选中文本
  var hasSelection = window.getSelection ? function (te) {
    try { return te.selectionStart != te.selectionEnd }
    catch(e) { return false }
  } : function (te) {
    var range;
    try {range = te.ownerDocument.selection.createRange();}
    catch(e) {}
    if (!range || range.parentElement() != te) { return false }
    return range.compareEndPoints("StartToEnd", range) != 0
  };

  // 检查是否支持复制事件
  var hasCopyEvent = (function () {
    var e = elt("div");
    // 检查是否支持 oncopy 事件
    if ("oncopy" in e) { return true }
    // 设置 oncopy 属性并检查是否为函数
    e.setAttribute("oncopy", "return;");
    return typeof e.oncopy == "function"
  })();

  // 初始化变量 badZoomedRects
  var badZoomedRects = null;
  // 检查是否存在错误的缩放矩形
  function hasBadZoomedRects(measure) {
    // 如果已经有缓存的值，则直接返回
    if (badZoomedRects != null) { return badZoomedRects }
    // 创建一个 span 元素，并添加到 measure 元素中
    var node = removeChildrenAndAdd(measure, elt("span", "x"));
    // 获取正常状态下的矩形
    var normal = node.getBoundingClientRect();
    // 获取选中文本的矩形
    var fromRange = range(node, 0, 1).getBoundingClientRect();
    // 比较两个矩形的左边距，判断是否存在错误的缩放矩形
    return badZoomedRects = Math.abs(normal.left - fromRange.left) > 1
  }

  // 已知的模式，按名称和 MIME 类型存储
  var modes = {}, mimeModes = {};

  // 定义模式
  // 额外的参数存储为模式的依赖项，用于自动加载模式
  function defineMode(name, mode) {
    if (arguments.length > 2)
      { mode.dependencies = Array.prototype.slice.call(arguments, 2); }
    modes[name] = mode;
  }

  // 定义 MIME 类型
  function defineMIME(mime, spec) {
    // 将给定的 MIME 类型和对应的模式配置对象存储到 mimeModes 对象中
    mimeModes[mime] = spec;
  }

  // 根据给定的 MIME 类型、{name, ...options} 配置对象或者名称字符串，返回一个模式配置对象
  function resolveMode(spec) {
    // 如果 spec 是字符串并且 mimeModes 对象中存在对应的配置对象，则将 spec 替换为对应的配置对象
    if (typeof spec == "string" && mimeModes.hasOwnProperty(spec)) {
      spec = mimeModes[spec];
    } 
    // 如果 spec 存在并且具有 name 字符串属性，并且 mimeModes 对象中存在对应的配置对象，则根据找到的配置对象和 spec 创建新的配置对象
    else if (spec && typeof spec.name == "string" && mimeModes.hasOwnProperty(spec.name)) {
      var found = mimeModes[spec.name];
      if (typeof found == "string") { found = {name: found}; }
      spec = createObj(found, spec);
      spec.name = found.name;
    } 
    // 如果 spec 是字符串并且符合特定的正则表达式，则返回对应的模式配置对象
    else if (typeof spec == "string" && /^[\w\-]+\/[\w\-]+\+xml$/.test(spec)) {
      return resolveMode("application/xml")
    } 
    // 如果 spec 是字符串并且符合特定的正则表达式，则返回对应的模式配置对象
    else if (typeof spec == "string" && /^[\w\-]+\/[\w\-]+\+json$/.test(spec)) {
      return resolveMode("application/json")
    }
    // 如果 spec 是字符串，则返回一个具有 name 属性的对象
    if (typeof spec == "string") { return {name: spec} }
    // 否则返回 spec 或者具有 name 属性为 "null" 的对象
    else { return spec || {name: "null"} }
  }

  // 根据模式配置对象，找到并初始化一个实际的模式对象
  function getMode(options, spec) {
    // 根据 resolveMode 返回的配置对象找到对应的模式工厂函数
    spec = resolveMode(spec);
    var mfactory = modes[spec.name];
    // 如果找不到对应的模式工厂函数，则返回 "text/plain" 的模式对象
    if (!mfactory) { return getMode(options, "text/plain") }
    var modeObj = mfactory(options, spec);
    // 如果 modeExtensions 对象中存在对应模式的扩展属性，则将其添加到模式对象中
    if (modeExtensions.hasOwnProperty(spec.name)) {
      var exts = modeExtensions[spec.name];
      for (var prop in exts) {
        if (!exts.hasOwnProperty(prop)) { continue }
        if (modeObj.hasOwnProperty(prop)) { modeObj["_" + prop] = modeObj[prop]; }
        modeObj[prop] = exts[prop];
      }
    }
    modeObj.name = spec.name;
    // 如果 spec 中存在 helperType 属性，则将其添加到模式对象中
    if (spec.helperType) { modeObj.helperType = spec.helperType; }
    // 如果 spec 中存在 modeProps 属性，则将其添加到模式对象中
    if (spec.modeProps) { for (var prop$1 in spec.modeProps)
      { modeObj[prop$1] = spec.modeProps[prop$1]; } }

    return modeObj
  }

  // 可以用于从模式定义之外向模式对象附加属性
  var modeExtensions = {};
  function extendMode(mode, properties) {
  // 如果给定的模式在 modeExtensions 中存在，则使用该模式的扩展，否则创建一个新的扩展对象
  var exts = modeExtensions.hasOwnProperty(mode) ? modeExtensions[mode] : (modeExtensions[mode] = {});
  // 将 properties 对象的属性复制到 exts 对象中
  copyObj(properties, exts);
}

// 复制给定模式的状态
function copyState(mode, state) {
  // 如果状态为 true，则直接返回状态
  if (state === true) { return state }
  // 如果模式有 copyState 方法，则调用该方法复制状态
  if (mode.copyState) { return mode.copyState(state) }
  // 否则，创建一个新的状态对象，并复制原状态对象的属性
  var nstate = {};
  for (var n in state) {
    var val = state[n];
    if (val instanceof Array) { val = val.concat([]); }
    nstate[n] = val;
  }
  return nstate
}

// 查找给定位置的内部模式和状态
function innerMode(mode, state) {
  var info;
  while (mode.innerMode) {
    // 调用 innerMode 方法获取内部模式和状态信息
    info = mode.innerMode(state);
    // 如果没有信息或者内部模式和当前模式相同，则退出循环
    if (!info || info.mode == mode) { break }
    state = info.state;
    mode = info.mode;
  }
  return info || {mode: mode, state: state}
}

// 获取给定模式的起始状态
function startState(mode, a1, a2) {
  // 如果模式有 startState 方法，则调用该方法获取起始状态
  return mode.startState ? mode.startState(a1, a2) : true
}

// 字符串流

// 用于模式解析器，提供辅助函数以使解析器更加简洁
var StringStream = function(string, tabSize, lineOracle) {
  this.pos = this.start = 0;
  this.string = string;
  this.tabSize = tabSize || 8;
  this.lastColumnPos = this.lastColumnValue = 0;
  this.lineStart = 0;
  this.lineOracle = lineOracle;
};

// 判断当前位置是否在行尾
StringStream.prototype.eol = function () {return this.pos >= this.string.length};
// 判断当前位置是否在行首
StringStream.prototype.sol = function () {return this.pos == this.lineStart};
// 返回当前位置的下一个字符
StringStream.prototype.peek = function () {return this.string.charAt(this.pos) || undefined};
// 将位置移动到下一个字符，并返回该字符
StringStream.prototype.next = function () {
  if (this.pos < this.string.length)
    { return this.string.charAt(this.pos++) }
};
// 如果当前字符匹配给定的字符或正则表达式，则将位置移动到下一个字符
StringStream.prototype.eat = function (match) {
  var ch = this.string.charAt(this.pos);
  var ok;
  if (typeof match == "string") { ok = ch == match; }
  else { ok = ch && (match.test ? match.test(ch) : match(ch)); }
    # 如果条件为真，则增加位置计数并返回字符
    if (ok) {++this.pos; return ch}
  };
  # 从当前位置开始，吃掉匹配的字符
  StringStream.prototype.eatWhile = function (match) {
    var start = this.pos;
    while (this.eat(match)){}
    return this.pos > start
  };
  # 从当前位置开始，吃掉空白字符
  StringStream.prototype.eatSpace = function () {
    var start = this.pos;
    while (/[\s\u00a0]/.test(this.string.charAt(this.pos))) { ++this.pos; }
    return this.pos > start
  };
  # 跳过当前位置到字符串末尾的字符
  StringStream.prototype.skipToEnd = function () {this.pos = this.string.length;};
  # 跳到指定字符的位置
  StringStream.prototype.skipTo = function (ch) {
    var found = this.string.indexOf(ch, this.pos);
    if (found > -1) {this.pos = found; return true}
  };
  # 回退指定数量的位置
  StringStream.prototype.backUp = function (n) {this.pos -= n;};
  # 返回当前位置的列数
  StringStream.prototype.column = function () {
    if (this.lastColumnPos < this.start) {
      this.lastColumnValue = countColumn(this.string, this.start, this.tabSize, this.lastColumnPos, this.lastColumnValue);
      this.lastColumnPos = this.start;
    }
    return this.lastColumnValue - (this.lineStart ? countColumn(this.string, this.lineStart, this.tabSize) : 0)
  };
  # 返回当前位置的缩进量
  StringStream.prototype.indentation = function () {
    return countColumn(this.string, null, this.tabSize) -
      (this.lineStart ? countColumn(this.string, this.lineStart, this.tabSize) : 0)
  };
  # 匹配指定的模式
  StringStream.prototype.match = function (pattern, consume, caseInsensitive) {
    if (typeof pattern == "string") {
      var cased = function (str) { return caseInsensitive ? str.toLowerCase() : str; };
      var substr = this.string.substr(this.pos, pattern.length);
      if (cased(substr) == cased(pattern)) {
        if (consume !== false) { this.pos += pattern.length; }
        return true
      }
    } else {
      var match = this.string.slice(this.pos).match(pattern);
      if (match && match.index > 0) { return null }
      if (match && consume !== false) { this.pos += match[0].length; }
      return match
  // 结束 StringStream 对象的原型定义
  }
  // 结束 StringStream 对象的定义
  };
  // 定义 StringStream 对象的 current 方法，返回当前位置到结束位置的字符串
  StringStream.prototype.current = function (){return this.string.slice(this.start, this.pos)};
  // 定义 StringStream 对象的 hideFirstChars 方法，用于隐藏前面的字符
  StringStream.prototype.hideFirstChars = function (n, inner) {
    this.lineStart += n;
    try { return inner() }
    finally { this.lineStart -= n; }
  };
  // 定义 StringStream 对象的 lookAhead 方法，用于预览后面的字符
  StringStream.prototype.lookAhead = function (n) {
    var oracle = this.lineOracle;
    return oracle && oracle.lookAhead(n)
  };
  // 定义 StringStream 对象的 baseToken 方法，返回基本的标记
  StringStream.prototype.baseToken = function () {
    var oracle = this.lineOracle;
    return oracle && oracle.baseToken(this.pos)
  };

  // 查找给定行号对应的行对象
  function getLine(doc, n) {
    n -= doc.first;
    if (n < 0 || n >= doc.size) { throw new Error("There is no line " + (n + doc.first) + " in the document.") }
    var chunk = doc;
    while (!chunk.lines) {
      for (var i = 0;; ++i) {
        var child = chunk.children[i], sz = child.chunkSize();
        if (n < sz) { chunk = child; break }
        n -= sz;
      }
    }
    return chunk.lines[n]
  }

  // 获取文档中两个位置之间的部分，作为字符串数组返回
  function getBetween(doc, start, end) {
    var out = [], n = start.line;
    doc.iter(start.line, end.line + 1, function (line) {
      var text = line.text;
      if (n == end.line) { text = text.slice(0, end.ch); }
      if (n == start.line) { text = text.slice(start.ch); }
      out.push(text);
      ++n;
    });
    return out
  }
  // 获取从 from 到 to 之间的行，作为字符串数组返回
  function getLines(doc, from, to) {
    var out = [];
    doc.iter(from, to, function (line) { out.push(line.text); }); // iter aborts when callback returns truthy value
    return out
  }

  // 更新行的高度，向上传播高度变化到父节点
  function updateLineHeight(line, height) {
    var diff = height - line.height;
  // 如果存在高度差，则更新父级链中所有节点的高度
  if (diff) { for (var n = line; n; n = n.parent) { n.height += diff; } }
  }

  // 给定一个行对象，通过遍历其父链接来找到其行号
  function lineNo(line) {
    if (line.parent == null) { return null }
    var cur = line.parent, no = indexOf(cur.lines, line);
    for (var chunk = cur.parent; chunk; cur = chunk, chunk = chunk.parent) {
      for (var i = 0;; ++i) {
        if (chunk.children[i] == cur) { break }
        no += chunk.children[i].chunkSize();
      }
    }
    return no + cur.first
  }

  // 根据文档树中的高度信息，找到给定垂直位置的行
  function lineAtHeight(chunk, h) {
    var n = chunk.first;
    outer: do {
      for (var i$1 = 0; i$1 < chunk.children.length; ++i$1) {
        var child = chunk.children[i$1], ch = child.height;
        if (h < ch) { chunk = child; continue outer }
        h -= ch;
        n += child.chunkSize();
      }
      return n
    } while (!chunk.lines)
    var i = 0;
    for (; i < chunk.lines.length; ++i) {
      var line = chunk.lines[i], lh = line.height;
      if (h < lh) { break }
      h -= lh;
    }
    return n + i
  }

  // 检查给定的行号是否在文档范围内
  function isLine(doc, l) {return l >= doc.first && l < doc.first + doc.size}

  // 为给定的行号返回格式化后的行号
  function lineNumberFor(options, i) {
    return String(options.lineNumberFormatter(i + options.firstLineNumber))
  }

  // Pos 实例表示文本中的位置
  function Pos(line, ch, sticky) {
    if ( sticky === void 0 ) sticky = null;

    if (!(this instanceof Pos)) { return new Pos(line, ch, sticky) }
    this.line = line;
    this.ch = ch;
  }
  // 设置当前位置的粘性属性
  this.sticky = sticky;
}

// 比较两个位置，如果它们相同则返回0，如果a小于b则返回负数，否则返回正数
function cmp(a, b) { return a.line - b.line || a.ch - b.ch }

// 比较两个位置是否相等
function equalCursorPos(a, b) { return a.sticky == b.sticky && cmp(a, b) == 0 }

// 复制位置对象
function copyPos(x) {return Pos(x.line, x.ch)}
// 返回两个位置中的较大值
function maxPos(a, b) { return cmp(a, b) < 0 ? b : a }
// 返回两个位置中的较小值
function minPos(a, b) { return cmp(a, b) < 0 ? a : b }

// 大多数外部API会裁剪给定的位置，以确保它们实际存在于文档中
function clipLine(doc, n) {return Math.max(doc.first, Math.min(n, doc.first + doc.size - 1))}
function clipPos(doc, pos) {
  if (pos.line < doc.first) { return Pos(doc.first, 0) }
  var last = doc.first + doc.size - 1;
  if (pos.line > last) { return Pos(last, getLine(doc, last).text.length) }
  return clipToLen(pos, getLine(doc, pos.line).text.length)
}
function clipToLen(pos, linelen) {
  var ch = pos.ch;
  if (ch == null || ch > linelen) { return Pos(pos.line, linelen) }
  else if (ch < 0) { return Pos(pos.line, 0) }
  else { return pos }
}
function clipPosArray(doc, array) {
  var out = [];
  for (var i = 0; i < array.length; i++) { out[i] = clipPos(doc, array[i]); }
  return out
}

// 保存上下文状态
var SavedContext = function(state, lookAhead) {
  this.state = state;
  this.lookAhead = lookAhead;
};

// 上下文对象
var Context = function(doc, state, line, lookAhead) {
  this.state = state;
  this.doc = doc;
  this.line = line;
  this.maxLookAhead = lookAhead || 0;
  this.baseTokens = null;
  this.baseTokenPos = 1;
};

// 查看上下文中的下一行
Context.prototype.lookAhead = function (n) {
  var line = this.doc.getLine(this.line + n);
  if (line != null && n > this.maxLookAhead) { this.maxLookAhead = n; }
  return line
};

// 获取基本标记
Context.prototype.baseToken = function (n) {
  if (!this.baseTokens) { return null }
}
    // 当基础令牌数组中的值小于等于 n 时，循环执行
    while (this.baseTokens[this.baseTokenPos] <= n)
      { this.baseTokenPos += 2; }
    // 获取当前位置的令牌类型
    var type = this.baseTokens[this.baseTokenPos + 1];
    // 返回一个对象，包含类型和大小信息
    return {type: type && type.replace(/( |^)overlay .*/, ""),
            size: this.baseTokens[this.baseTokenPos] - n}
  };

  // 下一行的处理
  Context.prototype.nextLine = function () {
    // 行数加一
    this.line++;
    // 如果最大向前查看数大于 0，则减一
    if (this.maxLookAhead > 0) { this.maxLookAhead--; }
  };

  // 从保存的上下文中创建新的上下文
  Context.fromSaved = function (doc, saved, line) {
    // 如果保存的上下文是 SavedContext 类型，则创建新的上下文
    if (saved instanceof SavedContext)
      { return new Context(doc, copyState(doc.mode, saved.state), line, saved.lookAhead) }
    // 否则，从保存的状态创建新的上下文
    else
      { return new Context(doc, copyState(doc.mode, saved), line) }
  };

  // 保存当前上下文
  Context.prototype.save = function (copy) {
    // 如果需要复制状态，则复制当前模式的状态
    var state = copy !== false ? copyState(this.doc.mode, this.state) : this.state;
    // 如果最大向前查看数大于 0，则返回 SavedContext 对象，否则返回状态
    return this.maxLookAhead > 0 ? new SavedContext(state, this.maxLookAhead) : state
  };


  // 计算样式数组（以模式生成号开头，后面是结束位置和样式字符串的成对数组），用于对行上的标记进行高亮显示
  function highlightLine(cm, line, context, forceToEnd) {
    // 样式数组始终以标识其基础模式/叠加模式的数字开头（便于无效化）
    var st = [cm.state.modeGen], lineClasses = {};
    // 计算基础样式数组
    runMode(cm, line.text, cm.doc.mode, context, function (end, style) { return st.push(end, style); },
            lineClasses, forceToEnd);
    var state = context.state;

    // 运行叠加模式，调整样式数组
    // 定义循环函数，参数为 o
    var loop = function ( o ) {
      // 将 context.baseTokens 设置为 st
      context.baseTokens = st;
      // 获取当前 overlay
      var overlay = cm.state.overlays[o], i = 1, at = 0;
      // 设置 context.state 为 true
      context.state = true;
      // 运行代码模式，获取每个 token 的样式
      runMode(cm, line.text, overlay.mode, context, function (end, style) {
        // 记录当前位置的 token 开始位置
        var start = i;
        // 确保当前位置有一个 token 结束，并且 i 指向它
        while (at < end) {
          var i_end = st[i];
          if (i_end > end)
            { st.splice(i, 1, end, st[i+1], i_end); }
          i += 2;
          at = Math.min(end, i_end);
        }
        // 如果没有样式，直接返回
        if (!style) { return }
        // 如果 overlay 是不透明的
        if (overlay.opaque) {
          st.splice(start, i - start, end, "overlay " + style);
          i = start + 2;
        } else {
          // 否则，为每个 token 添加样式
          for (; start < i; start += 2) {
            var cur = st[start+1];
            st[start+1] = (cur ? cur + " " : "") + "overlay " + style;
          }
        }
      }, lineClasses);
      // 恢复 context.state
      context.state = state;
      // 将 context.baseTokens 设置为 null
      context.baseTokens = null;
      // 将 context.baseTokenPos 设置为 1
      context.baseTokenPos = 1;
    };

    // 遍历所有 overlay，并执行 loop 函数
    for (var o = 0; o < cm.state.overlays.length; ++o) loop( o );

    // 返回样式和类
    return {styles: st, classes: lineClasses.bgClass || lineClasses.textClass ? lineClasses : null}
  }

  // 获取行的样式
  function getLineStyles(cm, line, updateFrontier) {
    // 如果行的样式不存在或者与当前模式不匹配
    if (!line.styles || line.styles[0] != cm.state.modeGen) {
      // 获取行之前的上下文
      var context = getContextBefore(cm, lineNo(line));
      // 如果行的长度超过最大高亮长度，复制当前状态
      var resetState = line.text.length > cm.options.maxHighlightLength && copyState(cm.doc.mode, context.state);
      // 高亮当前行
      var result = highlightLine(cm, line, context);
      // 如果需要重置状态，恢复状态
      if (resetState) { context.state = resetState; }
      // 保存行的状态
      line.stateAfter = context.save(!resetState);
      // 保存行的样式
      line.styles = result.styles;
      // 如果有类，保存类
      if (result.classes) { line.styleClasses = result.classes; }
      // 否则，清空类
      else if (line.styleClasses) { line.styleClasses = null; }
      // 如果需要更新 frontier，更新 frontier
      if (updateFrontier === cm.doc.highlightFrontier)
        { cm.doc.modeFrontier = Math.max(cm.doc.modeFrontier, ++cm.doc.highlightFrontier); }
    }
  // 返回行的样式
  return line.styles
}

// 获取指定行号之前的上下文
function getContextBefore(cm, n, precise) {
  var doc = cm.doc, display = cm.display;
  // 如果文档没有模式的起始状态，则返回一个新的上下文对象
  if (!doc.mode.startState) { return new Context(doc, true, n) }
  // 查找起始行
  var start = findStartLine(cm, n, precise);
  // 如果起始行大于文档的第一行，并且起始行的前一行有保存的状态，则使用保存的状态创建上下文对象，否则创建一个新的上下文对象
  var saved = start > doc.first && getLine(doc, start - 1).stateAfter;
  var context = saved ? Context.fromSaved(doc, saved, start) : new Context(doc, startState(doc.mode), start);

  // 遍历起始行到指定行之间的每一行
  doc.iter(start, n, function (line) {
    // 处理每一行的文本，更新上下文状态
    processLine(cm, line.text, context);
    var pos = context.line;
    // 如果当前行是最后一行，或者是5的倍数，或者在显示区域内，则保存当前上下文状态，否则置为null
    line.stateAfter = pos == n - 1 || pos % 5 == 0 || pos >= display.viewFrom && pos < display.viewTo ? context.save() : null;
    // 移动到下一行
    context.nextLine();
  });
  // 如果需要精确模式，则更新文档的模式边界
  if (precise) { doc.modeFrontier = context.line; }
  // 返回上下文对象
  return context
}

// 轻量级的高亮处理，处理当前行的文本，但不保存样式数组。用于当前不可见的行
function processLine(cm, text, context, startAt) {
  var mode = cm.doc.mode;
  var stream = new StringStream(text, cm.options.tabSize, context);
  stream.start = stream.pos = startAt || 0;
  // 如果文本为空，则调用空行处理函数
  if (text == "") { callBlankLine(mode, context.state); }
  // 循环处理文本中的每个标记
  while (!stream.eol()) {
    readToken(mode, stream, context.state);
    stream.start = stream.pos;
  }
}

// 调用空行处理函数
function callBlankLine(mode, state) {
  if (mode.blankLine) { return mode.blankLine(state) }
  if (!mode.innerMode) { return }
  var inner = innerMode(mode, state);
  if (inner.mode.blankLine) { return inner.mode.blankLine(inner.state) }
}

// 读取标记
function readToken(mode, stream, state, inner) {
  for (var i = 0; i < 10; i++) {
    if (inner) { inner[0] = innerMode(mode, state).mode; }
    var style = mode.token(stream, state);
    if (stream.pos > stream.start) { return style }
  }
  throw new Error("Mode " + mode.name + " failed to advance stream.")
}

// 标记对象构造函数
var Token = function(stream, type, state) {
  this.start = stream.start; this.end = stream.pos;
}
    // 将当前流的内容赋值给 this.string
    this.string = stream.current();
    // 如果提供了 type 参数，则赋给 this.type，否则为 null
    this.type = type || null;
    // 将 state 赋给 this.state
    this.state = state;
  };

  // 用于 getTokenAt 和 getLineTokens 的实用函数
  function takeToken(cm, pos, precise, asArray) {
    // 获取文档和模式
    var doc = cm.doc, mode = doc.mode, style;
    // 对位置进行裁剪
    pos = clipPos(doc, pos);
    // 获取行内容和上下文
    var line = getLine(doc, pos.line), context = getContextBefore(cm, pos.line, precise);
    // 创建一个新的字符串流
    var stream = new StringStream(line.text, cm.options.tabSize, context), tokens;
    // 如果 asArray 为真，则创建一个空数组
    if (asArray) { tokens = []; }
    // 当 asArray 为真或者流的位置小于指定位置并且没有到行尾时循环
    while ((asArray || stream.pos < pos.ch) && !stream.eol()) {
      // 设置流的起始位置
      stream.start = stream.pos;
      // 读取 token 的样式
      style = readToken(mode, stream, context.state);
      // 如果 asArray 为真，则将 token 添加到 tokens 数组中
      if (asArray) { tokens.push(new Token(stream, style, copyState(doc.mode, context.state))); }
    }
    // 如果 asArray 为真，则返回 tokens 数组，否则返回一个 Token 对象
    return asArray ? tokens : new Token(stream, style, context.state)
  }

  // 提取行的类
  function extractLineClasses(type, output) {
    // 如果 type 存在，则循环执行以下操作
    if (type) { for (;;) {
      // 匹配行的类
      var lineClass = type.match(/(?:^|\s+)line-(background-)?(\S+)/);
      // 如果没有匹配到行的类，则跳出循环
      if (!lineClass) { break }
      // 从 type 中移除匹配到的行的类
      type = type.slice(0, lineClass.index) + type.slice(lineClass.index + lineClass[0].length);
      // 根据匹配到的行的类设置背景类或文本类
      var prop = lineClass[1] ? "bgClass" : "textClass";
      if (output[prop] == null)
        { output[prop] = lineClass[2]; }
      else if (!(new RegExp("(?:^|\\s)" + lineClass[2] + "(?:$|\\s)")).test(output[prop]))
        { output[prop] += " " + lineClass[2]; }
    } }
    // 返回处理后的 type
    return type
  }

  // 运行给定模式的解析器，对每个 token 调用 f
  function runMode(cm, text, mode, context, f, lineClasses, forceToEnd) {
    // 获取 flattenSpans 或者使用默认值
    var flattenSpans = mode.flattenSpans;
    if (flattenSpans == null) { flattenSpans = cm.options.flattenSpans; }
    // 初始化当前起始位置和样式
    var curStart = 0, curStyle = null;
    // 创建一个新的字符串流
    var stream = new StringStream(text, cm.options.tabSize, context), style;
    // 如果 cm.options.addModeClass 为真，则创建一个内部数组
    var inner = cm.options.addModeClass && [null];
    // 如果文本为空，则提取行的类
    if (text == "") { extractLineClasses(callBlankLine(mode, context.state), lineClasses); }
    // 当流未到达行尾时执行循环
    while (!stream.eol()) {
      // 如果流的位置超过了最大高亮长度
      if (stream.pos > cm.options.maxHighlightLength) {
        // 设置扁平化样式为false
        flattenSpans = false;
        // 如果强制结束标志为true，则处理当前行的文本
        if (forceToEnd) { processLine(cm, text, context, stream.pos); }
        // 将流的位置设置为文本的长度
        stream.pos = text.length;
        // 样式设置为null
        style = null;
      } else {
        // 从流中读取一个标记，并提取行类
        style = extractLineClasses(readToken(mode, stream, context.state, inner), lineClasses);
      }
      // 如果inner存在
      if (inner) {
        // 获取inner数组的第一个元素的name属性
        var mName = inner[0].name;
        // 如果mName存在，则将样式设置为"m-" + mName + " " + style或者mName
        if (mName) { style = "m-" + (style ? mName + " " + style : mName); }
      }
      // 如果不需要扁平化样式或者当前样式不等于style
      if (!flattenSpans || curStyle != style) {
        // 当前起始位置小于流的起始位置时执行循环
        while (curStart < stream.start) {
          // 将当前起始位置设置为流的起始位置和当前起始位置加5000的最小值
          curStart = Math.min(stream.start, curStart + 5000);
          // 调用f函数，传入当前起始位置和当前样式
          f(curStart, curStyle);
        }
        // 将当前样式设置为style
        curStyle = style;
      }
      // 将流的起始位置设置为流的位置
      stream.start = stream.pos;
    }
    // 当前起始位置小于流的位置时执行循环
    while (curStart < stream.pos) {
      // Webkit似乎拒绝渲染超过57444个字符的文本节点，并且在大约5000个字符的节点中返回不准确的测量值
      var pos = Math.min(stream.pos, curStart + 5000);
      // 调用f函数，传入pos和当前样式
      f(pos, curStyle);
      // 将当前起始位置设置为pos
      curStart = pos;
    }
  }

  // 查找解析的起始行
  // 尝试找到具有stateAfter的行，以便可以从有效状态开始
  // 如果失败，则返回缩进最小的行，这样需要最少的上下文来正确解析
  function findStartLine(cm, n, precise) {
    var minindent, minline, doc = cm.doc;
    // 如果precise为true，则lim为-1，否则lim为n - (cm.doc.mode.innerMode ? 1000 : 100)
    var lim = precise ? -1 : n - (cm.doc.mode.innerMode ? 1000 : 100);
    // 从当前行号 n 开始向前搜索，直到搜索到 lim 为止
    for (var search = n; search > lim; --search) {
      // 如果搜索到的行号小于等于文档的第一行号，则返回文档的第一行号
      if (search <= doc.first) { return doc.first }
      // 获取当前搜索行的状态
      var line = getLine(doc, search - 1), after = line.stateAfter;
      // 如果存在状态并且满足条件，则返回当前搜索行号
      if (after && (!precise || search + (after instanceof SavedContext ? after.lookAhead : 0) <= doc.modeFrontier))
        { return search }
      // 计算当前行的缩进
      var indented = countColumn(line.text, null, cm.options.tabSize);
      // 如果 minline 为空或者当前行的缩进小于 minindent，则更新 minline 和 minindent
      if (minline == null || minindent > indented) {
        minline = search - 1;
        minindent = indented;
      }
    }
    // 返回最小缩进的行号
    return minline
  }

  // 将文档的 modeFrontier 属性设置为不大于 n 的最小值
  function retreatFrontier(doc, n) {
    doc.modeFrontier = Math.min(doc.modeFrontier, n);
    // 如果 highlightFrontier 小于 n - 10，则返回
    if (doc.highlightFrontier < n - 10) { return }
    var start = doc.first;
    // 从当前行号 n - 1 开始向前搜索
    for (var line = n - 1; line > start; line--) {
      var saved = getLine(doc, line).stateAfter;
      // 如果存在状态并且满足条件，则更新 start 并跳出循环
      if (saved && (!(saved instanceof SavedContext) || line + saved.lookAhead < n)) {
        start = line + 1;
        break
      }
    }
    // 将 highlightFrontier 属性设置为不大于 start 的最小值
    doc.highlightFrontier = Math.min(doc.highlightFrontier, start);
  }

  // 当未使用这些特性时，优化一些代码
  var sawReadOnlySpans = false, sawCollapsedSpans = false;

  // 将 sawReadOnlySpans 设置为 true
  function seeReadOnlySpans() {
    sawReadOnlySpans = true;
  }

  // 将 sawCollapsedSpans 设置为 true
  function seeCollapsedSpans() {
    sawCollapsedSpans = true;
  }

  // TEXTMARKER SPANS

  // 定义 MarkedSpan 类
  function MarkedSpan(marker, from, to) {
    this.marker = marker;
    this.from = from; this.to = to;
  }

  // 在 spans 数组中搜索与给定 marker 匹配的 span
  function getMarkedSpanFor(spans, marker) {
    if (spans) { for (var i = 0; i < spans.length; ++i) {
      var span = spans[i];
      if (span.marker == marker) { return span }
    } }
  }
  // 从数组中移除一个 span，如果没有剩余的 span，则返回 undefined（对于没有 span 的行，我们不存储数组）
  function removeMarkedSpan(spans, span) {
    var r;
  // 遍历 spans 数组
  for (var i = 0; i < spans.length; ++i)
    { 
      // 如果当前遍历到的 span 不等于给定的 span，则将其添加到结果数组 r 中
      if (spans[i] != span) { (r || (r = [])).push(spans[i]); } 
    }
  // 返回结果数组 r
  return r
}

// 为一行添加标记的 span
function addMarkedSpan(line, span) {
  // 如果该行已经有 markedSpans，则将新的 span 添加到其后面，否则创建一个新的 markedSpans 数组
  line.markedSpans = line.markedSpans ? line.markedSpans.concat([span]) : [span];
  // 将 span 的 marker 与该行关联起来
  span.marker.attachLine(line);
}

// 用于调整文档中标记的算法。这些函数在给定字符位置处切割 span 数组，返回剩余的部分数组（如果没有剩余则返回 undefined）
function markedSpansBefore(old, startCh, isInsert) {
  var nw;
  // 如果 old 存在，则遍历 old 数组
  if (old) { 
    for (var i = 0; i < old.length; ++i) {
      var span = old[i], marker = span.marker;
      // 判断 span 的起始位置是否在给定的 startCh 之前
      var startsBefore = span.from == null || (marker.inclusiveLeft ? span.from <= startCh : span.from < startCh);
      // 如果起始位置在 startCh 之前，或者起始位置就是 startCh 且 marker 类型为 "bookmark"，并且不是插入操作或者不是插入到左侧
      if (startsBefore || span.from == startCh && marker.type == "bookmark" && (!isInsert || !span.marker.insertLeft)) {
        // 如果 nw 不存在，则创建一个新数组
        var endsAfter = span.to == null || (marker.inclusiveRight ? span.to >= startCh : span.to > startCh)
        ;(nw || (nw = [])).push(new MarkedSpan(marker, span.from, endsAfter ? null : span.to));
      }
    } 
  }
  // 返回新数组 nw
  return nw
}
function markedSpansAfter(old, endCh, isInsert) {
  var nw;
  // 如果 old 存在，则遍历 old 数组
  if (old) { 
    for (var i = 0; i < old.length; ++i) {
      var span = old[i], marker = span.marker;
      // 判断 span 的结束位置是否在给定的 endCh 之后
      var endsAfter = span.to == null || (marker.inclusiveRight ? span.to >= endCh : span.to > endCh);
      // 如果结束位置在 endCh 之后，或者结束位置就是 endCh 且 marker 类型为 "bookmark"，并且是插入操作或者插入到左侧
      if (endsAfter || span.from == endCh && marker.type == "bookmark" && (!isInsert || span.marker.insertLeft)) {
        var startsBefore = span.from == null || (marker.inclusiveLeft ? span.from <= endCh : span.from < endCh)
        ;(nw || (nw = [])).push(new MarkedSpan(marker, startsBefore ? null : span.from - endCh,
                                              span.to == null ? null : span.to - endCh));
      }
    } 
  }
  // 返回新数组 nw
  return nw
}
  // 返回新的文本行，覆盖发生变化的行。删除变化范围内的标记，重新连接出现在变化两侧的同一标记的标记，截断部分在变化范围内的标记。返回一个包含每行（变化后）的标记数组的数组。
  function stretchSpansOverChange(doc, change) {
    if (change.full) { return null } // 如果变化是全行替换，则返回空
    var oldFirst = isLine(doc, change.from.line) && getLine(doc, change.from.line).markedSpans; // 获取变化起始行的标记数组
    var oldLast = isLine(doc, change.to.line) && getLine(doc, change.to.line).markedSpans; // 获取变化结束行的标记数组
    if (!oldFirst && !oldLast) { return null } // 如果起始行和结束行都没有标记，则返回空

    var startCh = change.from.ch, endCh = change.to.ch, isInsert = cmp(change.from, change.to) == 0; // 获取变化的起始字符位置、结束字符位置，以及是否是插入操作
    // 获取两侧“突出”的标记
    var first = markedSpansBefore(oldFirst, startCh, isInsert); // 获取起始行变化前的标记
    var last = markedSpansAfter(oldLast, endCh, isInsert); // 获取结束行变化后的标记

    // 接下来，合并这两端
    var sameLine = change.text.length == 1, offset = lst(change.text).length + (sameLine ? startCh : 0); // 判断是否是同一行变化，计算偏移量
    if (first) {
      // 修正 first 的 .to 属性
      for (var i = 0; i < first.length; ++i) {
        var span = first[i];
        if (span.to == null) {
          var found = getMarkedSpanFor(last, span.marker); // 获取与当前标记相同的结束标记
          if (!found) { span.to = startCh; } // 如果没有找到相同的结束标记，则设置为起始字符位置
          else if (sameLine) { span.to = found.to == null ? null : found.to + offset; } // 如果是同一行变化，则设置为找到的结束标记位置加上偏移量
        }
      }
    }
  }
    if (last) {
      // 如果存在 last，则修正 last 中的 .from 属性（或者在 sameLine 的情况下将它们移动到 first 中）
      for (var i$1 = 0; i$1 < last.length; ++i$1) {
        var span$1 = last[i$1];
        // 如果 span$1 的 .to 属性不为空，则将其加上偏移量
        if (span$1.to != null) { span$1.to += offset; }
        // 如果 span$1 的 .from 属性为空
        if (span$1.from == null) {
          // 查找 first 中与 span$1.marker 相同的标记，如果找不到，则将 span$1 添加到 first 中，并将其 .from 属性设置为偏移量
          var found$1 = getMarkedSpanFor(first, span$1.marker);
          if (!found$1) {
            span$1.from = offset;
            if (sameLine) { (first || (first = [])).push(span$1); }
          }
        } else {
          // 否则将 span$1 的 .from 属性加上偏移量，并将其添加到 first 中
          span$1.from += offset;
          if (sameLine) { (first || (first = [])).push(span$1); }
        }
      }
    }
    // 确保没有创建任何长度为零的 spans
    if (first) { first = clearEmptySpans(first); }
    // 如果 last 存在且不等于 first，则清除其中的空 spans
    if (last && last != first) { last = clearEmptySpans(last); }

    // 创建一个包含 first 的新标记数组
    var newMarkers = [first];
    if (!sameLine) {
      // 用整行 spans 填充间隙
      var gap = change.text.length - 2, gapMarkers;
      if (gap > 0 && first)
        { for (var i$2 = 0; i$2 < first.length; ++i$2)
          { if (first[i$2].to == null)
            { (gapMarkers || (gapMarkers = [])).push(new MarkedSpan(first[i$2].marker, null, null)); } } }
      for (var i$3 = 0; i$3 < gap; ++i$3)
        { newMarkers.push(gapMarkers); }
      newMarkers.push(last);
    }
    return newMarkers
  }

  // 移除空的 spans，并且没有 clearWhenEmpty 选项为 false 的 spans
  function clearEmptySpans(spans) {
    for (var i = 0; i < spans.length; ++i) {
      var span = spans[i];
      // 如果 span 的 .from 不为空且等于 .to，并且其 marker 的 clearWhenEmpty 选项不为 false，则将其从 spans 中删除
      if (span.from != null && span.from == span.to && span.marker.clearWhenEmpty !== false)
        { spans.splice(i--, 1); }
    }
    // 如果 spans 为空，则返回 null，否则返回 spans
    if (!spans.length) { return null }
    return spans
  }

  // 用于在进行更改时“剪切”掉只读范围
  function removeReadOnlyRanges(doc, from, to) {
    var markers = null;
    // 从指定起始行到结束行 + 1的范围内遍历文档的每一行
    doc.iter(from.line, to.line + 1, function (line) {
      // 如果当前行有标记范围
      if (line.markedSpans) { 
        // 遍历当前行的所有标记范围
        for (var i = 0; i < line.markedSpans.length; ++i) {
          var mark = line.markedSpans[i].marker;
          // 如果标记是只读的，并且不在markers数组中
          if (mark.readOnly && (!markers || indexOf(markers, mark) == -1))
            { (markers || (markers = [])).push(mark); }
        } 
      }
    });
    // 如果markers数组为空，则返回null
    if (!markers) { return null }
    // 初始化parts数组，包含起始和结束位置
    var parts = [{from: from, to: to}];
    // 遍历所有标记范围
    for (var i = 0; i < markers.length; ++i) {
      var mk = markers[i], m = mk.find(0);
      // 遍历parts数组
      for (var j = 0; j < parts.length; ++j) {
        var p = parts[j];
        // 如果p的结束位置小于m的起始位置，或者p的起始位置大于m的结束位置，则继续下一次循环
        if (cmp(p.to, m.from) < 0 || cmp(p.from, m.to) > 0) { continue }
        // 初始化newParts数组，包含j和1
        var newParts = [j, 1], dfrom = cmp(p.from, m.from), dto = cmp(p.to, m.to);
        // 如果dfrom小于0或者mk的inclusiveLeft为false并且dfrom为0
        if (dfrom < 0 || !mk.inclusiveLeft && !dfrom)
          { newParts.push({from: p.from, to: m.from}); }
        // 如果dto大于0或者mk的inclusiveRight为false并且dto为0
        if (dto > 0 || !mk.inclusiveRight && !dto)
          { newParts.push({from: m.to, to: p.to}); }
        // 将newParts数组中的元素插入到parts数组中
        parts.splice.apply(parts, newParts);
        // 更新j的值
        j += newParts.length - 3;
      }
    }
    // 返回parts数组
    return parts
  }

  // 从行中分离标记范围
  function detachMarkedSpans(line) {
    var spans = line.markedSpans;
    // 如果没有标记范围，则返回
    if (!spans) { return }
    // 遍历所有标记范围，分离标记范围
    for (var i = 0; i < spans.length; ++i)
      { spans[i].marker.detachLine(line); }
    // 将行的标记范围设置为null
    line.markedSpans = null;
  }
  // 将标记范围附加到行上
  function attachMarkedSpans(line, spans) {
    // 如果没有标记范围，则返回
    if (!spans) { return }
    // 遍历所有标记范围，将标记范围附加到行上
    for (var i = 0; i < spans.length; ++i)
      { spans[i].marker.attachLine(line); }
    // 将行的标记范围设置为spans
    line.markedSpans = spans;
  }

  // 在计算重叠折叠标记范围时使用的辅助函数
  function extraLeft(marker) { return marker.inclusiveLeft ? -1 : 0 }
  function extraRight(marker) { return marker.inclusiveRight ? 1 : 0 }

  // 返回一个数字，指示两个重叠的折叠标记范围中哪一个更大（因此包含另一个）。当范围完全相同时，回退到比较id
  function compareCollapsedMarkers(a, b) {
    # 计算两个文本行的长度差
    var lenDiff = a.lines.length - b.lines.length;
    # 如果长度差不为零，则返回长度差
    if (lenDiff != 0) { return lenDiff }
    # 查找文本行中的位置信息
    var aPos = a.find(), bPos = b.find();
    # 比较起始位置，如果不同则返回负值
    var fromCmp = cmp(aPos.from, bPos.from) || extraLeft(a) - extraLeft(b);
    # 如果起始位置比较结果不为零，则返回负值
    if (fromCmp) { return -fromCmp }
    # 比较结束位置，如果不同则返回正值
    var toCmp = cmp(aPos.to, bPos.to) || extraRight(a) - extraRight(b);
    # 如果结束位置比较结果不为零，则返回比较结果
    if (toCmp) { return toCmp }
    # 返回 b.id - a.id 的结果
    return b.id - a.id
  }

  # 查找文本行中是否存在折叠的区间，如果存在则返回该区间的标记
  function collapsedSpanAtSide(line, start) {
    # 获取文本行中的标记区间
    var sps = sawCollapsedSpans && line.markedSpans, found;
    # 遍历标记区间，查找折叠的区间
    if (sps) { for (var sp = (void 0), i = 0; i < sps.length; ++i) {
      sp = sps[i];
      # 如果标记区间是折叠的，并且起始或结束位置为空，并且比已找到的标记更小，则更新找到的标记
      if (sp.marker.collapsed && (start ? sp.from : sp.to) == null &&
          (!found || compareCollapsedMarkers(found, sp.marker) < 0))
        { found = sp.marker; }
    } }
    # 返回找到的标记
    return found
  }
  # 查找文本行中是否存在折叠的区间，如果存在则返回该区间的标记
  function collapsedSpanAtStart(line) { return collapsedSpanAtSide(line, true) }
  # 查找文本行中是否存在折叠的区间，如果存在则返回该区间的标记
  function collapsedSpanAtEnd(line) { return collapsedSpanAtSide(line, false) }

  # 查找文本行中是否存在包含指定位置的折叠的区间，如果存在则返回该区间的标记
  function collapsedSpanAround(line, ch) {
    # 获取文本行中的标记区间
    var sps = sawCollapsedSpans && line.markedSpans, found;
    # 遍历标记区间，查找包含指定位置的折叠的区间
    if (sps) { for (var i = 0; i < sps.length; ++i) {
      var sp = sps[i];
      # 如果标记区间是折叠的，并且包含指定位置，并且比已找到的标记更小，则更新找到的标记
      if (sp.marker.collapsed && (sp.from == null || sp.from < ch) && (sp.to == null || sp.to > ch) &&
          (!found || compareCollapsedMarkers(found, sp.marker) < 0)) { found = sp.marker; }
    } }
    # 返回找到的标记
    return found
  }

  # 测试是否存在折叠的区间与新区间部分重叠（覆盖起始或结束位置，但不是同时覆盖），这种重叠是不允许的
  function conflictingCollapsedRange(doc, lineNo, from, to, marker) {
    # 获取指定行的文本行
    var line = getLine(doc, lineNo);
    # 获取文本行中的标记区间
    var sps = sawCollapsedSpans && line.markedSpans;
    // 如果存在折叠的逻辑行，则遍历处理每一个折叠的逻辑行
    if (sps) { for (var i = 0; i < sps.length; ++i) {
      // 获取当前折叠的逻辑行
      var sp = sps[i];
      // 如果当前逻辑行没有折叠，则继续下一个逻辑行
      if (!sp.marker.collapsed) { continue }
      // 在当前折叠的逻辑行中查找指定位置的标记
      var found = sp.marker.find(0);
      // 比较找到的标记位置和给定的位置，判断是否在同一行
      var fromCmp = cmp(found.from, from) || extraLeft(sp.marker) - extraLeft(marker);
      var toCmp = cmp(found.to, to) || extraRight(sp.marker) - extraRight(marker);
      // 如果找到的标记位置和给定的位置在同一行，则继续下一个逻辑行
      if (fromCmp >= 0 && toCmp <= 0 || fromCmp <= 0 && toCmp >= 0) { continue }
      // 如果找到的标记位置和给定的位置不在同一行，则根据标记的inclusiveLeft和inclusiveRight属性判断是否在同一行
      if (fromCmp <= 0 && (sp.marker.inclusiveRight && marker.inclusiveLeft ? cmp(found.to, from) >= 0 : cmp(found.to, from) > 0) ||
          fromCmp >= 0 && (sp.marker.inclusiveRight && marker.inclusiveLeft ? cmp(found.from, to) <= 0 : cmp(found.from, to) < 0))
        { return true }
    } }
  }

  // 一个视觉行是屏幕上绘制的一行。折叠等操作可能导致多个逻辑行出现在同一视觉行上。此函数找到给定行所在的视觉行的起始位置（通常是该行本身）。
  function visualLine(line) {
    var merged;
    // 如果存在折叠的逻辑行，则遍历处理每一个折叠的逻辑行
    while (merged = collapsedSpanAtStart(line))
      { line = merged.find(-1, true).line; }
    return line
  }

  // 获取给定行所在的视觉行的结束位置
  function visualLineEnd(line) {
    var merged;
    // 如果存在折叠的逻辑行，则遍历处理每一个折叠的逻辑行
    while (merged = collapsedSpanAtEnd(line))
      { line = merged.find(1, true).line; }
    return line
  }

  // 返回延续给定行开始的视觉行的逻辑行数组，如果没有这样的行，则返回undefined
  function visualLineContinued(line) {
    var merged, lines;
    // 如果存在折叠的逻辑行，则遍历处理每一个折叠的逻辑行
    while (merged = collapsedSpanAtEnd(line)) {
      line = merged.find(1, true).line
      ;(lines || (lines = [])).push(line);
    }
    return lines
  }

  // 获取给定行号所在的视觉行的起始行号
  function visualLineNo(doc, lineN) {
    var line = getLine(doc, lineN), vis = visualLine(line);
    // 如果给定行号所在的逻辑行就是起始行号，则直接返回该行号
    if (line == vis) { return lineN }
  // 返回给定可视行的行号
  return lineNo(vis)
}

// 获取给定行后下一个可视行的起始行号
function visualLineEndNo(doc, lineN) {
  if (lineN > doc.lastLine()) { return lineN }
  var line = getLine(doc, lineN), merged;
  if (!lineIsHidden(doc, line)) { return lineN }
  while (merged = collapsedSpanAtEnd(line))
    { line = merged.find(1, true).line; }
  return lineNo(line) + 1
}

// 计算一行是否被隐藏。当一行是另一行开始的可视行的一部分，或者完全被折叠的非小部件跨度覆盖时，行被视为隐藏
function lineIsHidden(doc, line) {
  var sps = sawCollapsedSpans && line.markedSpans;
  if (sps) { for (var sp = (void 0), i = 0; i < sps.length; ++i) {
    sp = sps[i];
    if (!sp.marker.collapsed) { continue }
    if (sp.from == null) { return true }
    if (sp.marker.widgetNode) { continue }
    if (sp.from == 0 && sp.marker.inclusiveLeft && lineIsHiddenInner(doc, line, sp))
      { return true }
  } }
}
function lineIsHiddenInner(doc, line, span) {
  if (span.to == null) {
    var end = span.marker.find(1, true);
    return lineIsHiddenInner(doc, end.line, getMarkedSpanFor(end.line.markedSpans, span.marker))
  }
  if (span.marker.inclusiveRight && span.to == line.text.length)
    { return true }
  for (var sp = (void 0), i = 0; i < line.markedSpans.length; ++i) {
    sp = line.markedSpans[i];
    if (sp.marker.collapsed && !sp.marker.widgetNode && sp.from == span.to &&
        (sp.to == null || sp.to != span.from) &&
        (sp.marker.inclusiveLeft || span.marker.inclusiveRight) &&
        lineIsHiddenInner(doc, line, sp)) { return true }
  }
}

// 查找给定行上方的高度
function heightAtLine(lineObj) {
  lineObj = visualLine(lineObj);

  var h = 0, chunk = lineObj.parent;
    // 遍历 chunk.lines 数组，计算行高之和
    for (var i = 0; i < chunk.lines.length; ++i) {
      var line = chunk.lines[i];
      // 如果当前行等于指定的 lineObj，则跳出循环
      if (line == lineObj) { break }
      // 否则累加行高到 h 变量
      else { h += line.height; }
    }
    // 从当前 chunk 开始，向上遍历父节点，计算每个节点的高度之和
    for (var p = chunk.parent; p; chunk = p, p = chunk.parent) {
      for (var i$1 = 0; i$1 < p.children.length; ++i$1) {
        var cur = p.children[i$1];
        // 如果当前节点等于 chunk，则跳出循环
        if (cur == chunk) { break }
        // 否则累加当前节点的高度到 h 变量
        else { h += cur.height; }
      }
    }
    // 返回计算得到的高度 h
    return h
  }

  // 计算一行的字符长度，考虑到可能隐藏部分的折叠范围（参见 markText）和其他行连接到它上面的情况
  function lineLength(line) {
    // 如果行高为 0，则返回 0
    if (line.height == 0) { return 0 }
    var len = line.text.length, merged, cur = line;
    // 循环处理折叠范围在行首的情况
    while (merged = collapsedSpanAtStart(cur)) {
      var found = merged.find(0, true);
      cur = found.from.line;
      len += found.from.ch - found.to.ch;
    }
    cur = line;
    // 循环处理折叠范围在行尾的情况
    while (merged = collapsedSpanAtEnd(cur)) {
      var found$1 = merged.find(0, true);
      len -= cur.text.length - found$1.from.ch;
      cur = found$1.to.line;
      len += cur.text.length - found$1.to.ch;
    }
    // 返回计算得到的字符长度 len
    return len
  }

  // 查找文档中最长的行
  function findMaxLine(cm) {
    var d = cm.display, doc = cm.doc;
    d.maxLine = getLine(doc, doc.first);
    d.maxLineLength = lineLength(d.maxLine);
    d.maxLineChanged = true;
    // 遍历文档中的每一行，找到最长的行
    doc.iter(function (line) {
      var len = lineLength(line);
      if (len > d.maxLineLength) {
        d.maxLineLength = len;
        d.maxLine = line;
      }
    });
  }

  // 行数据结构

  // 行对象。这些对象保存与行相关的状态，包括高亮信息（styles 数组）
  var Line = function(text, markedSpans, estimateHeight) {
    this.text = text;
    attachMarkedSpans(this, markedSpans);
    // ...
  }
  // 如果 estimateHeight 存在，则使用 estimateHeight 函数估算行高，否则默认为 1
  this.height = estimateHeight ? estimateHeight(this) : 1;
};

// 返回行号
Line.prototype.lineNo = function () { return lineNo(this) };
// 给 Line 对象添加 eventMixin 方法
eventMixin(Line);

// 更新行的内容（文本，标记），自动使缓存信息无效，并尝试重新估算行高
function updateLine(line, text, markedSpans, estimateHeight) {
  // 更新行的文本内容
  line.text = text;
  // 如果 line.stateAfter 存在，则置为 null
  if (line.stateAfter) { line.stateAfter = null; }
  // 如果 line.styles 存在，则置为 null
  if (line.styles) { line.styles = null; }
  // 如果 line.order 不为 null，则置为 null
  if (line.order != null) { line.order = null; }
  // 分离行的标记范围
  detachMarkedSpans(line);
  // 附加标记范围到行
  attachMarkedSpans(line, markedSpans);
  // 如果 estimateHeight 存在，则使用 estimateHeight 函数估算行高，否则默认为 1
  var estHeight = estimateHeight ? estimateHeight(line) : 1;
  // 如果估算的行高不等于当前行的高度，则更新行高
  if (estHeight != line.height) { updateLineHeight(line, estHeight); }
}

// 从文档树和其标记中分离一行
function cleanUpLine(line) {
  // 将行的父节点置为 null
  line.parent = null;
  // 分离行的标记范围
  detachMarkedSpans(line);
}

// 将模式返回的样式（可以是 null，也可以是包含一个或多个样式的字符串）转换为 CSS 样式。这是缓存的，并且还查找了行级样式。
var styleToClassCache = {}, styleToClassCacheWithMode = {};
function interpretTokenStyle(style, options) {
  // 如果样式不存在或全为空白字符，则返回 null
  if (!style || /^\s*$/.test(style)) { return null }
  // 根据 options.addModeClass 的值选择缓存对象，将样式转换为 CSS 样式
  var cache = options.addModeClass ? styleToClassCacheWithMode : styleToClassCache;
  return cache[style] ||
    (cache[style] = style.replace(/\S+/g, "cm-$&"))
}

// 渲染行的文本的 DOM 表示。还构建了一个“行映射”，指向表示特定文本段的 DOM 节点，并被测量代码使用。返回的对象包含 DOM 节点、行映射以及模式设置的行级样式信息。
function buildLineContent(cm, lineView) {
  // padding-right 强制元素具有“边框”，这在 Webkit 中是必需的，以便能够获取其行级边界矩形（在 measureChar 中）
}
    // 创建一个包含内容的 span 元素，设置样式为 webkit 时的特定样式
    var content = eltP("span", null, null, webkit ? "padding-right: .1px" : null);
    // 创建一个包含 content 的 pre 元素，设置类名为 "CodeMirror-line"
    // 初始化 builder 对象，包括 pre 元素、content 元素、列数、位置、CodeMirror 对象、尾随空格标志、分割空格标志
    var builder = {pre: eltP("pre", [content], "CodeMirror-line"), content: content,
                   col: 0, pos: 0, cm: cm,
                   trailingSpace: false,
                   splitSpaces: cm.getOption("lineWrapping")};
    // 初始化 lineView.measure 为空对象
    lineView.measure = {};

    // 遍历构成该可视行的逻辑行
    for (var i = 0; i <= (lineView.rest ? lineView.rest.length : 0); i++) {
      // 获取当前逻辑行
      var line = i ? lineView.rest[i - 1] : lineView.line, order = (void 0);
      // 重置位置和添加标记函数
      builder.pos = 0;
      builder.addToken = buildToken;
      // 如果浏览器存在双向文本渲染问题，并且存在文本方向，则进行特定处理
      if (hasBadBidiRects(cm.display.measure) && (order = getOrder(line, cm.doc.direction)))
        { builder.addToken = buildTokenBadBidi(builder.addToken, order); }
      // 初始化映射数组
      builder.map = [];
      // 获取逻辑行的样式，并插入行内容
      var allowFrontierUpdate = lineView != cm.display.externalMeasured && lineNo(line);
      insertLineContent(line, builder, getLineStyles(cm, line, allowFrontierUpdate));
      // 如果逻辑行存在样式类，则合并样式类
      if (line.styleClasses) {
        if (line.styleClasses.bgClass)
          { builder.bgClass = joinClasses(line.styleClasses.bgClass, builder.bgClass || ""); }
        if (line.styleClasses.textClass)
          { builder.textClass = joinClasses(line.styleClasses.textClass, builder.textClass || ""); }
      }

      // 确保至少有一个节点存在，用于测量
      if (builder.map.length == 0)
        { builder.map.push(0, 0, builder.content.appendChild(zeroWidthElement(cm.display.measure))); }

      // 存储映射和当前逻辑行的缓存对象
      if (i == 0) {
        lineView.measure.map = builder.map;
        lineView.measure.cache = {};
      } else {
  (lineView.measure.maps || (lineView.measure.maps = [])).push(builder.map)
        ;(lineView.measure.caches || (lineView.measure.caches = [])).push({});
      }
    }
    // 如果是 webkit 浏览器
    if (webkit) {
      // 获取 builder.content 的最后一个子节点
      var last = builder.content.lastChild;
      // 如果最后一个子节点的类名包含 "cm-tab" 或者包含 ".cm-tab" 的子节点
      if (/\bcm-tab\b/.test(last.className) || (last.querySelector && last.querySelector(".cm-tab")))
        { builder.content.className = "cm-tab-wrap-hack"; }
    }

    // 触发 "renderLine" 事件，传入 cm、lineView.line、builder.pre 作为参数
    signal(cm, "renderLine", cm, lineView.line, builder.pre);
    // 如果 builder.pre 的类名存在
    if (builder.pre.className)
      { builder.textClass = joinClasses(builder.pre.className, builder.textClass || ""); }

    // 返回 builder 对象
    return builder
  }

  // 默认的特殊字符占位符
  function defaultSpecialCharPlaceholder(ch) {
    // 创建一个包含特殊字符的 span 元素
    var token = elt("span", "\u2022", "cm-invalidchar");
    // 设置特殊字符的标题为 Unicode 编码
    token.title = "\\u" + ch.charCodeAt(0).toString(16);
    // 设置 aria-label 属性为标题
    token.setAttribute("aria-label", token.title);
    // 返回特殊字符的 span 元素
    return token
  }

  // 构建单个标记的 DOM 表示，并将其添加到行映射中。注意单独渲染特殊字符。
  function buildToken(builder, text, style, startStyle, endStyle, css, attributes) {
    // 如果文本为空，则返回
    if (!text) { return }
    // 如果需要拆分空格，则将文本拆分为显示文本
    var displayText = builder.splitSpaces ? splitSpaces(text, builder.trailingSpace) : text;
    // 获取特殊字符正则表达式和是否必须包装的标志
    var special = builder.cm.state.specialChars, mustWrap = false;
    var content;
    // 如果文本不包含特殊字符
    if (!special.test(text)) {
      // 增加列数
      builder.col += text.length;
      // 创建文本节点
      content = document.createTextNode(displayText);
      // 将文本节点添加到行映射中
      builder.map.push(builder.pos, builder.pos + text.length, content);
      // 如果是 IE 并且版本小于 9，则必须包装
      if (ie && ie_version < 9) { mustWrap = true; }
      // 增加位置
      builder.pos += text.length;
    } else {
      # 创建一个空的文档片段
      content = document.createDocumentFragment();
      # 初始化位置变量
      var pos = 0;
      # 进入循环，处理特殊字符
      while (true) {
        # 设置特殊字符的匹配位置
        special.lastIndex = pos;
        # 在文本中查找下一个特殊字符
        var m = special.exec(text);
        # 计算跳过的字符数
        var skipped = m ? m.index - pos : text.length - pos;
        # 如果有跳过的字符
        if (skipped) {
          # 创建文本节点
          var txt = document.createTextNode(displayText.slice(pos, pos + skipped));
          # 如果是 IE 并且版本小于 9，创建包含文本节点的 span 元素
          if (ie && ie_version < 9) { content.appendChild(elt("span", [txt])); }
          # 否则，直接添加文本节点
          else { content.appendChild(txt); }
          # 更新映射关系
          builder.map.push(builder.pos, builder.pos + skipped, txt);
          # 更新列数和位置
          builder.col += skipped;
          builder.pos += skipped;
        }
        # 如果没有找到特殊字符，退出循环
        if (!m) { break }
        # 更新位置
        pos += skipped + 1;
        # 初始化文本节点
        var txt$1 = (void 0);
        # 如果是制表符
        if (m[0] == "\t") {
          # 计算制表符的宽度
          var tabSize = builder.cm.options.tabSize, tabWidth = tabSize - builder.col % tabSize;
          # 创建包含制表符的 span 元素
          txt$1 = content.appendChild(elt("span", spaceStr(tabWidth), "cm-tab"));
          txt$1.setAttribute("role", "presentation");
          txt$1.setAttribute("cm-text", "\t");
          # 更新列数
          builder.col += tabWidth;
        } 
        # 如果是回车或换行符
        else if (m[0] == "\r" || m[0] == "\n") {
          # 创建包含回车或换行符的 span 元素
          txt$1 = content.appendChild(elt("span", m[0] == "\r" ? "\u240d" : "\u2424", "cm-invalidchar"));
          txt$1.setAttribute("cm-text", m[0]);
          # 更新列数
          builder.col += 1;
        } 
        # 如果是其他特殊字符
        else {
          # 根据特殊字符创建文本节点
          txt$1 = builder.cm.options.specialCharPlaceholder(m[0]);
          txt$1.setAttribute("cm-text", m[0]);
          # 如果是 IE 并且版本小于 9，创建包含文本节点的 span 元素
          if (ie && ie_version < 9) { content.appendChild(elt("span", [txt$1])); }
          # 否则，直接添加文本节点
          else { content.appendChild(txt$1); }
          # 更新列数
          builder.col += 1;
        }
        # 更新映射关系
        builder.map.push(builder.pos, builder.pos + 1, txt$1);
        # 更新位置
        builder.pos++;
      }
    }
    # 检查最后一个字符是否为空格
    builder.trailingSpace = displayText.charCodeAt(text.length - 1) == 32;
    // 如果存在样式、起始样式、结束样式、必须包装或者 CSS，则执行以下操作
    if (style || startStyle || endStyle || mustWrap || css) {
      // 如果存在样式，则将其赋给 fullStyle，否则为空字符串
      var fullStyle = style || "";
      // 如果存在起始样式，则将其添加到 fullStyle 中
      if (startStyle) { fullStyle += startStyle; }
      // 如果存在结束样式，则将其添加到 fullStyle 中
      if (endStyle) { fullStyle += endStyle; }
      // 使用 elt 函数创建一个 span 元素，包含 content 内容和 fullStyle 样式
      var token = elt("span", [content], fullStyle, css);
      // 如果存在 attributes，则遍历其中的属性，将其添加到 token 元素中
      if (attributes) {
        for (var attr in attributes) { 
          if (attributes.hasOwnProperty(attr) && attr != "style" && attr != "class") {
            token.setAttribute(attr, attributes[attr]); 
          } 
        }
      }
      // 将 token 元素添加到 builder.content 中
      return builder.content.appendChild(token)
    }
    // 将 content 直接添加到 builder.content 中
    builder.content.appendChild(content);
  }

  // 将一些空格替换为 NBSP，以防止浏览器在渲染文本时将行末空格合并在一起（问题＃1362）
  function splitSpaces(text, trailingBefore) {
    // 如果文本长度大于1且不包含连续两个空格，则直接返回文本
    if (text.length > 1 && !/  /.test(text)) { return text }
    var spaceBefore = trailingBefore, result = "";
    for (var i = 0; i < text.length; i++) {
      var ch = text.charAt(i);
      // 如果当前字符是空格且前一个字符也是空格，并且下一个字符也是空格或者是文本末尾，则将当前空格替换为 NBSP
      if (ch == " " && spaceBefore && (i == text.length - 1 || text.charCodeAt(i + 1) == 32)) {
        ch = "\u00a0";
      }
      // 将处理后的字符添加到结果中
      result += ch;
      // 更新 spaceBefore 变量
      spaceBefore = ch == " ";
    }
    return result
  }

  // 解决对右到左文本报告的无意义尺寸问题
  function buildTokenBadBidi(inner, order) {
    // 定义一个函数，接受多个参数，包括构建器、文本、样式、起始样式、结束样式、CSS 和属性
    return function (builder, text, style, startStyle, endStyle, css, attributes) {
      // 如果样式存在，则在原样式后添加" cm-force-border"，否则直接使用"cm-force-border"
      style = style ? style + " cm-force-border" : "cm-force-border";
      // 定义起始位置和结束位置
      var start = builder.pos, end = start + text.length;
      // 无限循环
      for (;;) {
        // 查找与文本起始位置重叠的部分
        var part = (void 0);
        for (var i = 0; i < order.length; i++) {
          part = order[i];
          // 如果部分的结束位置大于等于起始位置并且起始位置小于等于起始位置，则跳出循环
          if (part.to > start && part.from <= start) { break }
        }
        // 如果部分的结束位置大于等于结束位置，则调用inner函数，传入参数，并返回结果
        if (part.to >= end) { return inner(builder, text, style, startStyle, endStyle, css, attributes) }
        // 调用inner函数，传入参数，并返回结果
        inner(builder, text.slice(0, part.to - start), style, startStyle, null, css, attributes);
        // 重置起始样式为null
        startStyle = null;
        // 截取文本，更新起始位置
        text = text.slice(part.to - start);
        start = part.to;
      }
    }
  }

  // 定义一个函数，接受构建器、大小、标记和忽略小部件作为参数
  function buildCollapsedSpan(builder, size, marker, ignoreWidget) {
    // 判断是否忽略小部件，并获取小部件节点
    var widget = !ignoreWidget && marker.widgetNode;
    // 如果小部件存在，则将其位置和大小添加到构建器的映射中
    if (widget) { builder.map.push(builder.pos, builder.pos + size, widget); }
    // 如果不忽略小部件并且需要内容属性，则设置小部件的属性
    if (!ignoreWidget && builder.cm.display.input.needsContentAttribute) {
      if (!widget)
        { widget = builder.content.appendChild(document.createElement("span")); }
      widget.setAttribute("cm-marker", marker.id);
    }
    // 如果小部件存在，则设置其不可编辑，并将其添加到构建器的内容中
    if (widget) {
      builder.cm.display.input.setUneditable(widget);
      builder.content.appendChild(widget);
    }
    // 更新构建器的位置和尾随空格状态
    builder.pos += size;
    builder.trailingSpace = false;
  }

  // 输出多个span以构成一行，考虑到高亮和标记文本
  function insertLineContent(line, builder, styles) {
    // 获取行的标记span和全部文本
    var spans = line.markedSpans, allText = line.text, at = 0;
    // 如果没有标记span，则根据样式添加token
    if (!spans) {
      for (var i$1 = 1; i$1 < styles.length; i$1+=2)
        { builder.addToken(builder, allText.slice(at, at = styles[i$1]), interpretTokenStyle(styles[i$1+1], builder.cm.options)); }
      return
    }

    // 定义长度、位置、索引、文本、样式和CSS
    var len = allText.length, pos = 0, i = 1, text = "", style, css;
    # 定义变量 nextChange，用于存储下一个改变的位置
    var nextChange = 0, spanStyle, spanEndStyle, spanStartStyle, collapsed, attributes;
    # 结束函数定义
    }
  }

  # LineView 对象用于表示文档的可见部分，一个 LineView 可能对应多个逻辑行，如果它们被折叠范围连接
  function LineView(doc, line, lineN) {
    # 赋值起始行
    this.line = line;
    # 继续的行，如果有的话
    this.rest = visualLineContinued(line);
    # 这个可视行中的逻辑行数
    this.size = this.rest ? lineNo(lst(this.rest)) - lineN + 1 : 1;
    this.node = this.text = null;
    # 判断该行是否被隐藏
    this.hidden = lineIsHidden(doc, line);
  }

  # 为给定的行创建 LineView 对象的范围
  function buildViewArray(cm, from, to) {
    var array = [], nextPos;
    for (var pos = from; pos < to; pos = nextPos) {
      var view = new LineView(cm.doc, getLine(cm.doc, pos), pos);
      nextPos = pos + view.size;
      array.push(view);
    }
    return array
  }

  # 定义操作组变量
  var operationGroup = null;

  # 将操作推入操作组
  function pushOperation(op) {
    if (operationGroup) {
      operationGroup.ops.push(op);
    } else {
      op.ownsGroup = operationGroup = {
        ops: [op],
        delayedCallbacks: []
      };
    }
  }

  # 为操作组的操作调用延迟回调
  function fireCallbacksForOps(group) {
    # 调用延迟回调和光标活动处理程序，直到没有新的出现
    var callbacks = group.delayedCallbacks, i = 0;
    do {
      for (; i < callbacks.length; i++)
        { callbacks[i].call(null); }
      for (var j = 0; j < group.ops.length; j++) {
        var op = group.ops[j];
        if (op.cursorActivityHandlers)
          { while (op.cursorActivityCalled < op.cursorActivityHandlers.length)
            { op.cursorActivityHandlers[op.cursorActivityCalled++].call(null, op.cm); } }
      }
    } while (i < callbacks.length)
  }

  # 完成操作组的操作
  function finishOperation(op, endCb) {
    var group = op.ownsGroup;
    if (!group) { return }
    try { fireCallbacksForOps(group); }
  // 在 finally 块中，清空 operationGroup，并调用 endCb 回调函数
  finally {
    operationGroup = null;
    endCb(group);
  }

  var orphanDelayedCallbacks = null;

  // 通常情况下，我们希望在某个工作进行到一半时发出事件信号，但不希望处理程序开始调用编辑器的其他方法，
  // 这可能导致编辑器处于不一致的状态，或者根本不希望发生其他事件。
  // signalLater 检查是否有任何处理程序，并安排它们在最后一个操作结束时执行，或者如果没有活动操作，则在超时触发时执行。
  function signalLater(emitter, type /*, values...*/) {
    var arr = getHandlers(emitter, type);
    if (!arr.length) { return }
    var args = Array.prototype.slice.call(arguments, 2), list;
    if (operationGroup) {
      list = operationGroup.delayedCallbacks;
    } else if (orphanDelayedCallbacks) {
      list = orphanDelayedCallbacks;
    } else {
      list = orphanDelayedCallbacks = [];
      setTimeout(fireOrphanDelayed, 0);
    }
    var loop = function ( i ) {
      list.push(function () { return arr[i].apply(null, args); });
    };

    for (var i = 0; i < arr.length; ++i)
      loop( i );
  }

  // 当一个行的某个方面发生变化时，会向 lineView.changes 添加一个字符串。这个函数更新行的 DOM 结构的相关部分。
  function updateLineForChanges(cm, lineView, lineN, dims) {
    for (var j = 0; j < lineView.changes.length; j++) {
      var type = lineView.changes[j];
      if (type == "text") { updateLineText(cm, lineView); }
      else if (type == "gutter") { updateLineGutter(cm, lineView, lineN, dims); }
      else if (type == "class") { updateLineClasses(cm, lineView); }
      else if (type == "widget") { updateLineWidgets(cm, lineView, dims); }
    }
  }
  // 将 lineView 的 changes 属性设置为 null
  lineView.changes = null;
}

// 确保具有 gutter 元素、小部件或背景类的行被包装，并且额外的元素被添加到包装的 div 中
function ensureLineWrapped(lineView) {
  if (lineView.node == lineView.text) {
    // 如果 lineView 的 node 等于 text，则创建一个 div 元素作为 node，并将 text 移动到该 div 中
    lineView.node = elt("div", null, null, "position: relative");
    if (lineView.text.parentNode)
      { lineView.text.parentNode.replaceChild(lineView.node, lineView.text); }
    lineView.node.appendChild(lineView.text);
    // 如果是 IE 并且版本小于 8，则设置 node 的 z-index 为 2
    if (ie && ie_version < 8) { lineView.node.style.zIndex = 2; }
  }
  return lineView.node
}

// 更新行的背景
function updateLineBackground(cm, lineView) {
  // 获取行的背景类
  var cls = lineView.bgClass ? lineView.bgClass + " " + (lineView.line.bgClass || "") : lineView.line.bgClass;
  if (cls) { cls += " CodeMirror-linebackground"; }
  if (lineView.background) {
    if (cls) { lineView.background.className = cls; }
    else { lineView.background.parentNode.removeChild(lineView.background); lineView.background = null; }
  } else if (cls) {
    // 确保行被包装，并在其内部添加背景元素
    var wrap = ensureLineWrapped(lineView);
    lineView.background = wrap.insertBefore(elt("div", null, cls), wrap.firstChild);
    cm.display.input.setUneditable(lineView.background);
  }
}

// 包装 buildLineContent 方法，如果可能的话，将重用 display.externalMeasured 中的结构
function getLineContent(cm, lineView) {
  var ext = cm.display.externalMeasured;
  if (ext && ext.line == lineView.line) {
    cm.display.externalMeasured = null;
    lineView.measure = ext.measure;
    return ext.built
  }
  return buildLineContent(cm, lineView)
}

// 重新绘制行的文本。与背景和文本类交互，因为模式可能会输出影响这些类的标记
function updateLineText(cm, lineView) {
  // 获取 lineView 的 text 的类名
  var cls = lineView.text.className;
  // 获取行的内容
  var built = getLineContent(cm, lineView);
  if (lineView.text == lineView.node) { lineView.node = built.pre; }
}
  // 用构建好的 <pre> 元素替换原来的文本节点
  lineView.text.parentNode.replaceChild(built.pre, lineView.text);
  // 更新 lineView.text 为构建好的 <pre> 元素
  lineView.text = built.pre;
  // 如果构建好的背景类或文本类与 lineView 中的不同，则更新 lineView 中的类，并更新行的样式
  if (built.bgClass != lineView.bgClass || built.textClass != lineView.textClass) {
    lineView.bgClass = built.bgClass;
    lineView.textClass = built.textClass;
    updateLineClasses(cm, lineView);
  } else if (cls) {
    // 否则，如果有类名，则更新 lineView.text 的类名
    lineView.text.className = cls;
  }
}

// 更新行的类
function updateLineClasses(cm, lineView) {
  // 更新行的背景
  updateLineBackground(cm, lineView);
  // 如果行有 wrapClass，则更新行的包装元素的类名
  if (lineView.line.wrapClass)
    { ensureLineWrapped(lineView).className = lineView.line.wrapClass; }
  // 否则，如果行的节点不是 lineView.text，则清空行的节点的类名
  else if (lineView.node != lineView.text)
    { lineView.node.className = ""; }
  // 更新行的文本类
  var textClass = lineView.textClass ? lineView.textClass + " " + (lineView.line.textClass || "") : lineView.line.textClass;
  lineView.text.className = textClass || "";
}

// 更新行的行号区域
function updateLineGutter(cm, lineView, lineN, dims) {
  // 如果 lineView 有 gutter，则移除它
  if (lineView.gutter) {
    lineView.node.removeChild(lineView.gutter);
    lineView.gutter = null;
  }
  // 如果 lineView 有 gutterBackground，则移除它
  if (lineView.gutterBackground) {
    lineView.node.removeChild(lineView.gutterBackground);
    lineView.gutterBackground = null;
  }
  // 如果行有 gutterClass，则创建并插入行的 gutterBackground 元素
  if (lineView.line.gutterClass) {
    var wrap = ensureLineWrapped(lineView);
    lineView.gutterBackground = elt("div", null, "CodeMirror-gutter-background " + lineView.line.gutterClass,
                                    ("left: " + (cm.options.fixedGutter ? dims.fixedPos : -dims.gutterTotalWidth) + "px; width: " + (dims.gutterTotalWidth) + "px"));
    cm.display.input.setUneditable(lineView.gutterBackground);
    wrap.insertBefore(lineView.gutterBackground, lineView.text);
  }
  // 获取行的 gutterMarkers
  var markers = lineView.line.gutterMarkers;
    // 如果设置了显示行号或者有标记
    if (cm.options.lineNumbers || markers) {
      // 确保行视图被包裹
      var wrap$1 = ensureLineWrapped(lineView);
      // 创建装载行号的容器
      var gutterWrap = lineView.gutter = elt("div", null, "CodeMirror-gutter-wrapper", ("left: " + (cm.options.fixedGutter ? dims.fixedPos : -dims.gutterTotalWidth) + "px"));
      // 设置装载行号的容器为不可编辑
      cm.display.input.setUneditable(gutterWrap);
      // 将装载行号的容器插入到行视图中
      wrap$1.insertBefore(gutterWrap, lineView.text);
      // 如果行有特定的行号样式类，则添加到装载行号的容器上
      if (lineView.line.gutterClass)
        { gutterWrap.className += " " + lineView.line.gutterClass; }
      // 如果设置了显示行号并且没有标记或者没有"CodeMirror-linenumbers"标记
      if (cm.options.lineNumbers && (!markers || !markers["CodeMirror-linenumbers"]))
        { lineView.lineNumber = gutterWrap.appendChild(
          elt("div", lineNumberFor(cm.options, lineN),
              "CodeMirror-linenumber CodeMirror-gutter-elt",
              ("left: " + (dims.gutterLeft["CodeMirror-linenumbers"]) + "px; width: " + (cm.display.lineNumInnerWidth) + "px"))); }
      // 如果有标记
      if (markers) { for (var k = 0; k < cm.display.gutterSpecs.length; ++k) {
        var id = cm.display.gutterSpecs[k].className, found = markers.hasOwnProperty(id) && markers[id];
        // 如果找到了对应的标记，则将其添加到装载行号的容器中
        if (found)
          { gutterWrap.appendChild(elt("div", [found], "CodeMirror-gutter-elt",
                                     ("left: " + (dims.gutterLeft[id]) + "px; width: " + (dims.gutterWidth[id]) + "px"))); }
      } }
    }
  }

  // 更新行部件
  function updateLineWidgets(cm, lineView, dims) {
    // 如果行视图可对齐，则将其设置为不可对齐
    if (lineView.alignable) { lineView.alignable = null; }
    // 创建一个测试函数，用于检测是否为行部件
    var isWidget = classTest("CodeMirror-linewidget");
    // 遍历行视图的子节点
    for (var node = lineView.node.firstChild, next = (void 0); node; node = next) {
      next = node.nextSibling;
      // 如果是行部件，则将其从行视图中移除
      if (isWidget.test(node.className)) { lineView.node.removeChild(node); }
    }
    // 插入行部件
    insertLineWidgets(cm, lineView, dims);
  }

  // 从头开始构建行的 DOM 表示
  function buildLineElement(cm, lineView, lineN, dims) {
    // 获取构建的行内容
    var built = getLineContent(cm, lineView);
    // 将行内容设置为行视图的文本和节点
    lineView.text = lineView.node = built.pre;
    // 如果有背景样式类，则将其设置为行视图的背景样式类
    if (built.bgClass) { lineView.bgClass = built.bgClass; }
    # 如果存在文本类别，则将 lineView 的文本类别设置为已构建的文本类别
    if (built.textClass) { lineView.textClass = built.textClass; }

    # 更新行的类别
    updateLineClasses(cm, lineView);
    # 更新行号的类别
    updateLineGutter(cm, lineView, lineN, dims);
    # 插入行部件
    insertLineWidgets(cm, lineView, dims);
    # 返回行视图的节点
    return lineView.node
  }

  # 一个 lineView 可能包含多个逻辑行（通过合并的跨度合并）。所有这些行的部件都需要被绘制。
  function insertLineWidgets(cm, lineView, dims) {
    insertLineWidgetsFor(cm, lineView.line, lineView, dims, true);
    # 如果 lineView.rest 存在，则为其所有行插入部件
    if (lineView.rest) { for (var i = 0; i < lineView.rest.length; i++)
      { insertLineWidgetsFor(cm, lineView.rest[i], lineView, dims, false); } }
  }

  # 为指定行插入部件
  function insertLineWidgetsFor(cm, line, lineView, dims, allowAbove) {
    # 如果行没有部件，则返回
    if (!line.widgets) { return }
    # 确保行被包裹
    var wrap = ensureLineWrapped(lineView);
    # 遍历行的部件
    for (var i = 0, ws = line.widgets; i < ws.length; ++i) {
      var widget = ws[i], node = elt("div", [widget.node], "CodeMirror-linewidget" + (widget.className ? " " + widget.className : ""));
      # 如果部件不处理鼠标事件，则设置属性 cm-ignore-events 为 true
      if (!widget.handleMouseEvents) { node.setAttribute("cm-ignore-events", "true"); }
      # 定位行部件
      positionLineWidget(widget, node, lineView, dims);
      cm.display.input.setUneditable(node);
      # 如果允许在上方插入部件，并且部件在上方
      if (allowAbove && widget.above)
        { wrap.insertBefore(node, lineView.gutter || lineView.text); }
      else
        { wrap.appendChild(node); }
      signalLater(widget, "redraw");
    }
  }

  # 定位行部件
  function positionLineWidget(widget, node, lineView, dims) {
    # 如果部件不水平滚动，则将其添加到 alignable 数组中
    if (widget.noHScroll) {
  (lineView.alignable || (lineView.alignable = [])).push(node);
      var width = dims.wrapperWidth;
      node.style.left = dims.fixedPos + "px";
      # 如果部件不覆盖行号，则调整宽度和左边距
      if (!widget.coverGutter) {
        width -= dims.gutterTotalWidth;
        node.style.paddingLeft = dims.gutterTotalWidth + "px";
      }
      node.style.width = width + "px";
    }
    # 如果部件覆盖行号，则设置样式属性
    if (widget.coverGutter) {
      node.style.zIndex = 5;
      node.style.position = "relative";
      # 如果部件不水平滚动，则设置左边距
      if (!widget.noHScroll) { node.style.marginLeft = -dims.gutterTotalWidth + "px"; }
  // 结束函数定义
    }
  }

  // 计算小部件的高度
  function widgetHeight(widget) {
    // 如果小部件的高度不为空，则返回该高度
    if (widget.height != null) { return widget.height }
    // 获取小部件所在的 CodeMirror 编辑器
    var cm = widget.doc.cm;
    // 如果编辑器不存在，则返回 0
    if (!cm) { return 0 }
    // 如果小部件不在文档中
    if (!contains(document.body, widget.node)) {
      // 设置父元素的样式
      var parentStyle = "position: relative;";
      // 如果小部件覆盖了行号区域，则设置左边距
      if (widget.coverGutter)
        { parentStyle += "margin-left: -" + cm.display.gutters.offsetWidth + "px;"; }
      // 如果小部件不允许水平滚动，则设置宽度
      if (widget.noHScroll)
        { parentStyle += "width: " + cm.display.wrapper.clientWidth + "px;"; }
      // 移除原有的子元素，并添加小部件的节点
      removeChildrenAndAdd(cm.display.measure, elt("div", [widget.node], null, parentStyle));
    }
    // 返回小部件的高度，并将其赋值给 widget.height
    return widget.height = widget.node.parentNode.offsetHeight
  }

  // 判断鼠标事件是否发生在小部件内部
  function eventInWidget(display, e) {
    // 遍历事件的目标元素的父元素，判断是否在编辑器的 wrapper 内
    for (var n = e_target(e); n != display.wrapper; n = n.parentNode) {
      if (!n || (n.nodeType == 1 && n.getAttribute("cm-ignore-events") == "true") ||
          (n.parentNode == display.sizer && n != display.mover))
        { return true }
    }
  }

  // 位置测量

  // 获取行间距的上边距
  function paddingTop(display) {return display.lineSpace.offsetTop}
  // 获取行间距的垂直内边距
  function paddingVert(display) {return display.mover.offsetHeight - display.lineSpace.offsetHeight}
  // 获取行间距的水平内边距
  function paddingH(display) {
    // 如果已经缓存了水平内边距，则直接返回缓存值
    if (display.cachedPaddingH) { return display.cachedPaddingH }
    // 创建一个包含单个字符的 pre 元素，并获取其样式
    var e = removeChildrenAndAdd(display.measure, elt("pre", "x", "CodeMirror-line-like"));
    var style = window.getComputedStyle ? window.getComputedStyle(e) : e.currentStyle;
    // 解析样式中的左右内边距值
    var data = {left: parseInt(style.paddingLeft), right: parseInt(style.paddingRight)};
    // 如果内边距值合法，则缓存并返回
    if (!isNaN(data.left) && !isNaN(data.right)) { display.cachedPaddingH = data; }
    return data
  }

  // 获取滚动条的间隙
  function scrollGap(cm) { return scrollerGap - cm.display.nativeBarWidth }
  // 获取编辑器的显示宽度
  function displayWidth(cm) {
    return cm.display.scroller.clientWidth - scrollGap(cm) - cm.display.barWidth
  }
  // 获取编辑器的显示高度
  function displayHeight(cm) {
  // 计算并返回滚动条的高度
  return cm.display.scroller.clientHeight - scrollGap(cm) - cm.display.barHeight
}

// 确保lineView.wrapping.heights数组被填充。这是一个由绘制行组成的底部偏移量数组。当lineWrapping打开时，可能会有多个高度。
function ensureLineHeights(cm, lineView, rect) {
  var wrapping = cm.options.lineWrapping;
  var curWidth = wrapping && displayWidth(cm);
  if (!lineView.measure.heights || wrapping && lineView.measure.width != curWidth) {
    var heights = lineView.measure.heights = [];
    if (wrapping) {
      lineView.measure.width = curWidth;
      var rects = lineView.text.firstChild.getClientRects();
      for (var i = 0; i < rects.length - 1; i++) {
        var cur = rects[i], next = rects[i + 1];
        if (Math.abs(cur.bottom - next.bottom) > 2)
          { heights.push((cur.bottom + next.top) / 2 - rect.top); }
      }
    }
    heights.push(rect.bottom - rect.top);
  }
}

// 找到给定行号的行映射（将字符偏移映射到文本节点）和测量缓存。（当存在折叠范围时，行视图可能包含多行。）
function mapFromLineView(lineView, line, lineN) {
  if (lineView.line == line)
    { return {map: lineView.measure.map, cache: lineView.measure.cache} }
  for (var i = 0; i < lineView.rest.length; i++)
    { if (lineView.rest[i] == line)
      { return {map: lineView.measure.maps[i], cache: lineView.measure.caches[i]} } }
  for (var i$1 = 0; i$1 < lineView.rest.length; i$1++)
    { if (lineNo(lineView.rest[i$1]) > lineN)
      { return {map: lineView.measure.maps[i$1], cache: lineView.measure.caches[i$1], before: true} } }
}

// 将一行渲染到隐藏节点display.externalMeasured中。在需要测量不在视口中的行时使用。
function updateExternalMeasurement(cm, line) {
  line = visualLine(line);
}
  // 获取给定行的行号
  var lineN = lineNo(line);
  // 创建一个新的 LineView 对象，用于显示给定行的内容
  var view = cm.display.externalMeasured = new LineView(cm.doc, line, lineN);
  // 将行号赋值给 LineView 对象的属性
  view.lineN = lineN;
  // 构建 LineView 对象的内容
  var built = view.built = buildLineContent(cm, view);
  // 将构建好的内容赋值给 LineView 对象的文本属性
  view.text = built.pre;
  // 移除之前的子元素，并添加新的内容到行测量容器中
  removeChildrenAndAdd(cm.display.lineMeasure, built.pre);
  // 返回 LineView 对象
  return view
}

// 获取给定字符的 {top, bottom, left, right} 盒子（以行本地坐标表示）
function measureChar(cm, line, ch, bias) {
  return measureCharPrepared(cm, prepareMeasureForLine(cm, line), ch, bias)
}

// 查找与给定行号对应的 LineView 对象
function findViewForLine(cm, lineN) {
  if (lineN >= cm.display.viewFrom && lineN < cm.display.viewTo)
    { return cm.display.view[findViewIndex(cm, lineN)] }
  var ext = cm.display.externalMeasured;
  if (ext && lineN >= ext.lineN && lineN < ext.lineN + ext.size)
    { return ext }
}

// 测量可以分为两个步骤，适用于整行的设置工作和实际字符的测量。
// 因此，需要连续进行大量测量的函数（如 coordsChar）可以确保设置工作只执行一次。
function prepareMeasureForLine(cm, line) {
  // 获取行号
  var lineN = lineNo(line);
  // 查找与给定行号对应的 LineView 对象
  var view = findViewForLine(cm, lineN);
  // 如果存在对应的 LineView 对象且没有文本内容，则将 view 设置为 null
  if (view && !view.text) {
    view = null;
  } else if (view && view.changes) {
    // 更新 LineView 对象以反映文本变化
    updateLineForChanges(cm, view, lineN, getDimensions(cm));
    cm.curOp.forceUpdate = true;
  }
  // 如果不存在对应的 LineView 对象，则更新外部测量
  if (!view)
    { view = updateExternalMeasurement(cm, line); }

  // 将信息从 LineView 对象映射到行上，并返回准备好的测量对象
  var info = mapFromLineView(view, line, lineN);
  return {
    line: line, view: view, rect: null,
    map: info.map, cache: info.cache, before: info.before,
    hasHeights: false
  }
}

// 给定准备好的测量对象，测量实际字符的位置（或从缓存中获取）
function measureCharPrepared(cm, prepared, ch, bias, varHeight) {
  // 如果存在 before 属性，则将字符位置设置为 -1
  if (prepared.before) { ch = -1; }
}
    # 根据字符和偏移量生成缓存键值
    var key = ch + (bias || ""), found;
    # 检查缓存中是否存在对应键值的数据
    if (prepared.cache.hasOwnProperty(key)) {
      found = prepared.cache[key];
    } else {
      # 如果缓存中不存在对应键值的数据，则进行以下操作
      if (!prepared.rect)
        { prepared.rect = prepared.view.text.getBoundingClientRect(); }
      # 如果没有高度信息，则获取文本的行高信息
      if (!prepared.hasHeights) {
        ensureLineHeights(cm, prepared.view, prepared.rect);
        prepared.hasHeights = true;
      }
      # 获取字符的测量数据
      found = measureCharInner(cm, prepared, ch, bias);
      # 如果测量数据有效，则将其存入缓存
      if (!found.bogus) { prepared.cache[key] = found; }
    }
    # 返回测量数据
    return {left: found.left, right: found.right,
            top: varHeight ? found.rtop : found.top,
            bottom: varHeight ? found.rbottom : found.bottom}
  }

  # 定义一个空的矩形对象
  var nullRect = {left: 0, right: 0, top: 0, bottom: 0};

  # 根据字符和偏移量在行映射中查找节点和偏移量
  function nodeAndOffsetInLineMap(map, ch, bias) {
    var node, start, end, collapse, mStart, mEnd;
    # 首先，在行映射中搜索与目标字符对应或最接近的文本节点
    for (var i = 0; i < map.length; i += 3) {
      mStart = map[i];
      mEnd = map[i + 1];
      if (ch < mStart) {
        start = 0; end = 1;
        collapse = "left";
      } else if (ch < mEnd) {
        start = ch - mStart;
        end = start + 1;
      } else if (i == map.length - 3 || ch == mEnd && map[i + 3] > ch) {
        end = mEnd - mStart;
        start = end - 1;
        if (ch >= mEnd) { collapse = "right"; }
      }
      if (start != null) {
        node = map[i + 2];
        if (mStart == mEnd && bias == (node.insertLeft ? "left" : "right"))
          { collapse = bias; }
        if (bias == "left" && start == 0)
          { while (i && map[i - 2] == map[i - 3] && map[i - 1].insertLeft) {
            node = map[(i -= 3) + 2];
            collapse = "left";
          } }
        if (bias == "right" && start == mEnd - mStart)
          { while (i < map.length - 3 && map[i + 3] == map[i + 4] && !map[i + 5].insertLeft) {
            node = map[(i += 3) + 2];
            collapse = "right";
          } }
        break
      }
    }
  // 返回一个包含节点、起始位置、结束位置、折叠状态、覆盖起始位置和覆盖结束位置的对象
  return {node: node, start: start, end: end, collapse: collapse, coverStart: mStart, coverEnd: mEnd}
}

// 获取有用的矩形
function getUsefulRect(rects, bias) {
  var rect = nullRect;
  // 如果偏向左侧
  if (bias == "left") {
    // 遍历矩形数组
    for (var i = 0; i < rects.length; i++) {
      // 如果左边不等于右边
      if ((rect = rects[i]).left != rect.right) {
        // 跳出循环
        break
      }
    }
  } else {
    // 如果偏向右侧
    for (var i$1 = rects.length - 1; i$1 >= 0; i$1--) {
      if ((rect = rects[i$1]).left != rect.right) {
        break
      }
    }
  }
  // 返回矩形
  return rect
}

// 测量字符的内部方法
function measureCharInner(cm, prepared, ch, bias) {
  // 获取字符在行映射中的节点和偏移量
  var place = nodeAndOffsetInLineMap(prepared.map, ch, bias);
  var node = place.node, start = place.start, end = place.end, collapse = place.collapse;

  var rect;
  // 如果节点类型为文本节点
  if (node.nodeType == 3) {
    // 最多重试4次，当返回无意义的矩形时
    for (var i$1 = 0; i$1 < 4; i$1++) {
      // 当起始位置大于0且当前字符是扩展字符时，减小起始位置
      while (start && isExtendingChar(prepared.line.text.charAt(place.coverStart + start))) { --start; }
      // 当覆盖起始位置加上结束位置小于覆盖结束位置且当前字符是扩展字符时，增加结束位置
      while (place.coverStart + end < place.coverEnd && isExtendingChar(prepared.line.text.charAt(place.coverStart + end))) { ++end; }
      // 如果是IE并且版本小于9并且起始位置为0且结束位置为覆盖结束位置减去覆盖起始位置
      if (ie && ie_version < 9 && start == 0 && end == place.coverEnd - place.coverStart) {
        // 获取父节点的矩形
        rect = node.parentNode.getBoundingClientRect();
      } else {
        // 获取有用的矩形
        rect = getUsefulRect(range(node, start, end).getClientRects(), bias);
      }
      // 如果矩形左边、右边或者起始位置为0，则跳出循环
      if (rect.left || rect.right || start == 0) { break }
      // 调整起始位置和结束位置
      end = start;
      start = start - 1;
      collapse = "right";
    }
    // 如果是IE并且版本小于11
    if (ie && ie_version < 11) { rect = maybeUpdateRectForZooming(cm.display.measure, rect); }
  } else {
    // 如果是小部件，直接获取整个小部件的框
    if (start > 0) { collapse = bias = "right"; }
    var rects;
    // 如果代码编辑器选项中包含换行并且矩形数组的长度大于1
    if (cm.options.lineWrapping && (rects = node.getClientRects()).length > 1) {
      // 获取最后一个矩形或者第一个矩形
      rect = rects[bias == "right" ? rects.length - 1 : 0];
    } else {
      // 获取节点的矩形
      rect = node.getBoundingClientRect();
    }
  }
}
    // 如果满足条件：1. 存在ie；2. ie版本小于9；3. start为假；4. rect不存在或者left和right都为假
    if (ie && ie_version < 9 && !start && (!rect || !rect.left && !rect.right)) {
      // 获取父节点的第一个客户端矩形
      var rSpan = node.parentNode.getClientRects()[0];
      // 如果rSpan存在，则将其left、right、top、bottom值赋给rect，否则将rect赋值为nullRect
      if (rSpan)
        { rect = {left: rSpan.left, right: rSpan.left + charWidth(cm.display), top: rSpan.top, bottom: rSpan.bottom}; }
      else
        { rect = nullRect; }
    }

    // 计算rtop和rbot
    var rtop = rect.top - prepared.rect.top, rbot = rect.bottom - prepared.rect.top;
    // 计算mid
    var mid = (rtop + rbot) / 2;
    // 获取prepared.view.measure.heights
    var heights = prepared.view.measure.heights;
    var i = 0;
    // 遍历heights数组
    for (; i < heights.length - 1; i++)
      { if (mid < heights[i]) { break } }
    // 计算top和bot
    var top = i ? heights[i - 1] : 0, bot = heights[i];
    // 计算result对象
    var result = {left: (collapse == "right" ? rect.right : rect.left) - prepared.rect.left,
                  right: (collapse == "left" ? rect.left : rect.right) - prepared.rect.left,
                  top: top, bottom: bot};
    // 如果rect的left和right都为假，则将result对象的bogus属性设置为true
    if (!rect.left && !rect.right) { result.bogus = true; }
    // 如果cm.options.singleCursorHeightPerLine为假，则将result对象的rtop和rbottom属性设置为rtop和rbot
    if (!cm.options.singleCursorHeightPerLine) { result.rtop = rtop; result.rbottom = rbot; }

    // 返回result对象
    return result
  }

  // 解决IE10及以下缩放时边界客户端矩形返回不正确的问题
  function maybeUpdateRectForZooming(measure, rect) {
    // 如果不满足条件：1. window.screen不存在；2. logicalXDPI或者deviceXDPI为null；3. logicalXDPI等于deviceXDPI；4. 没有错误的缩放矩形
    if (!window.screen || screen.logicalXDPI == null ||
        screen.logicalXDPI == screen.deviceXDPI || !hasBadZoomedRects(measure))
      { return rect }
    // 计算scaleX和scaleY
    var scaleX = screen.logicalXDPI / screen.deviceXDPI;
    var scaleY = screen.logicalYDPI / screen.deviceYDPI;
    // 返回经过缩放处理后的rect对象
    return {left: rect.left * scaleX, right: rect.right * scaleX,
            top: rect.top * scaleY, bottom: rect.bottom * scaleY}
  }

  // 清除lineView的测量缓存
  function clearLineMeasurementCacheFor(lineView) {
    // 如果lineView.measure存在，则清空cache和heights，如果lineView.rest存在，则遍历清空caches
    if (lineView.measure) {
      lineView.measure.cache = {};
      lineView.measure.heights = null;
      if (lineView.rest) { for (var i = 0; i < lineView.rest.length; i++)
        { lineView.measure.caches[i] = {}; } }
    }
  }

  // 清除cm的行测量缓存
  function clearLineMeasurementCache(cm) {
    # 将外部测量值设置为null
    cm.display.externalMeasure = null;
    # 移除cm.display.lineMeasure中的所有子元素
    removeChildren(cm.display.lineMeasure);
    # 遍历cm.display.view数组
    for (var i = 0; i < cm.display.view.length; i++)
      { 
        # 清除cm.display.view[i]的行测量缓存
        clearLineMeasurementCacheFor(cm.display.view[i]); 
      }
  }

  # 清除缓存
  function clearCaches(cm) {
    # 清除行测量缓存
    clearLineMeasurementCache(cm);
    # 将cm.display.cachedCharWidth、cm.display.cachedTextHeight、cm.display.cachedPaddingH设置为null
    cm.display.cachedCharWidth = cm.display.cachedTextHeight = cm.display.cachedPaddingH = null;
    # 如果不是行包裹模式，则将cm.display.maxLineChanged设置为true
    if (!cm.options.lineWrapping) { cm.display.maxLineChanged = true; }
    # 将cm.display.lineNumChars设置为null
    cm.display.lineNumChars = null;
  }

  # 获取页面水平滚动距离
  function pageScrollX() {
    # 解决Chrome和Android的bug，返回页面水平滚动距离
    if (chrome && android) { return -(document.body.getBoundingClientRect().left - parseInt(getComputedStyle(document.body).marginLeft)) }
    return window.pageXOffset || (document.documentElement || document.body).scrollLeft
  }
  # 获取页面垂直滚动距离
  function pageScrollY() {
    # 如果是Chrome和Android，返回页面垂直滚动距离
    if (chrome && android) { return -(document.body.getBoundingClientRect().top - parseInt(getComputedStyle(document.body).marginTop)) }
    return window.pageYOffset || (document.documentElement || document.body).scrollTop
  }

  # 获取小部件顶部高度
  function widgetTopHeight(lineObj) {
    var height = 0;
    # 如果lineObj有小部件，则遍历小部件数组
    if (lineObj.widgets) { for (var i = 0; i < lineObj.widgets.length; ++i) { if (lineObj.widgets[i].above)
      { 
        # 如果小部件在行上方，则累加其高度
        height += widgetHeight(lineObj.widgets[i]); 
      } } }
    return height
  }

  # 将行本地坐标系中的{top, bottom, left, right}框转换为另一个坐标系
  # 上下文可以是"line"、"div"（display.lineDiv）、"local"/null（编辑器）、"window"或"page"
  function intoCoordSystem(cm, lineObj, rect, context, includeWidgets) {
    # 如果不包括小部件
    if (!includeWidgets) {
      # 获取小部件顶部高度
      var height = widgetTopHeight(lineObj);
      rect.top += height; rect.bottom += height;
    }
    # 如果上下文是"line"，则直接返回rect
    if (context == "line") { return rect }
    # 如果上下文为空，则默认为"local"
    if (!context) { context = "local"; }
    # 获取行的高度
    var yOff = heightAtLine(lineObj);
    // 如果上下文是“local”，则将yOff增加cm.display的paddingTop
    if (context == "local") { yOff += paddingTop(cm.display); }
    // 否则，将yOff减去cm.display的viewOffset
    else { yOff -= cm.display.viewOffset; }
    // 如果上下文是“page”或“window”
    if (context == "page" || context == "window") {
      // 获取cm.display.lineSpace的边界矩形
      var lOff = cm.display.lineSpace.getBoundingClientRect();
      // 将yOff增加lOff的top值，并根据上下文是否为“window”来决定是否加上pageScrollY()
      yOff += lOff.top + (context == "window" ? 0 : pageScrollY());
      // 获取lOff的left值，并根据上下文是否为“window”来决定是否加上pageScrollX()
      var xOff = lOff.left + (context == "window" ? 0 : pageScrollX());
      // 将rect的left和right值分别加上xOff
      rect.left += xOff; rect.right += xOff;
    }
    // 将rect的top和bottom值分别加上yOff
    rect.top += yOff; rect.bottom += yOff;
    // 返回rect
    return rect
  }

  // 将一个框从“div”坐标系转换为另一个坐标系
  // 上下文可以是“window”、“page”、“div”或“local”/null
  function fromCoordSystem(cm, coords, context) {
    // 如果上下文是“div”，则直接返回coords
    if (context == "div") { return coords }
    var left = coords.left, top = coords.top;
    // 首先将坐标转换为“page”坐标系
    if (context == "page") {
      left -= pageScrollX();
      top -= pageScrollY();
    } else if (context == "local" || !context) {
      // 获取cm.display.sizer的边界矩形
      var localBox = cm.display.sizer.getBoundingClientRect();
      left += localBox.left;
      top += localBox.top;
    }
    // 获取cm.display.lineSpace的边界矩形
    var lineSpaceBox = cm.display.lineSpace.getBoundingClientRect();
    // 返回转换后的坐标
    return {left: left - lineSpaceBox.left, top: top - lineSpaceBox.top}
  }

  // 获取字符的坐标
  function charCoords(cm, pos, context, lineObj, bias) {
    // 如果lineObj不存在，则获取pos所在行的line对象
    if (!lineObj) { lineObj = getLine(cm.doc, pos.line); }
  # 返回给定坐标系中的坐标位置
  return intoCoordSystem(cm, lineObj, measureChar(cm, lineObj, pos.ch, bias), context)
}

// 返回给定光标位置的框，可能包含一个'other'属性，其中包含次要光标在双向文本边界上的位置
// 光标 Pos(line, char, "before") 位于与 `char - 1` 相同的可视行上，并且在 `char - 1` 的书写顺序之后
// 光标 Pos(line, char, "after") 位于与 `char` 相同的可视行上，并且在 `char` 的书写顺序之前
// 例子（大写字母是RTL，小写字母是LTR）：
//     Pos(0, 1, ...)
//     before   after
// ab     a|b     a|b
// aB     a|B     aB|
// Ab     |Ab     A|b
// AB     B|A     B|A
// 每个位置在行上最后一个字符之后被认为是粘在行上的最后一个字符上
function cursorCoords(cm, pos, context, lineObj, preparedMeasure, varHeight) {
  lineObj = lineObj || getLine(cm.doc, pos.line);
  if (!preparedMeasure) { preparedMeasure = prepareMeasureForLine(cm, lineObj); }
  function get(ch, right) {
    var m = measureCharPrepared(cm, preparedMeasure, ch, right ? "right" : "left", varHeight);
    if (right) { m.left = m.right; } else { m.right = m.left; }
    return intoCoordSystem(cm, lineObj, m, context)
  }
  var order = getOrder(lineObj, cm.doc.direction), ch = pos.ch, sticky = pos.sticky;
  if (ch >= lineObj.text.length) {
    ch = lineObj.text.length;
    sticky = "before";
  } else if (ch <= 0) {
    ch = 0;
    sticky = "after";
  }
  if (!order) { return get(sticky == "before" ? ch - 1 : ch, sticky == "before") }

  function getBidi(ch, partPos, invert) {
    var part = order[partPos], right = part.level == 1;
    return get(invert ? ch - 1 : ch, right != invert)
  }
  var partPos = getBidiPartAt(order, ch, sticky);
  var other = bidiOther;
  var val = getBidi(ch, partPos, sticky == "before");
}
    # 如果 other 不为空，则根据条件获取双向绑定的值
    if (other != null) { val.other = getBidi(ch, other, sticky != "before"); }
    # 返回值
    return val
  }

  // 用于快速估算位置的坐标。用于中间滚动更新。
  function estimateCoords(cm, pos) {
    var left = 0;
    # 对位置进行裁剪
    pos = clipPos(cm.doc, pos);
    # 如果不换行，则计算左侧位置
    if (!cm.options.lineWrapping) { left = charWidth(cm.display) * pos.ch; }
    # 获取行对象
    var lineObj = getLine(cm.doc, pos.line);
    # 计算顶部位置
    var top = heightAtLine(lineObj) + paddingTop(cm.display);
    # 返回坐标对象
    return {left: left, right: left, top: top, bottom: top + lineObj.height}
  }

  // coordsChar 返回的位置包含一些额外信息。
  // xRel 是输入坐标相对于找到的位置的相对 x 位置（因此 xRel > 0 表示坐标在字符位置的右侧，例如）。
  // 当 outside 为 true 时，表示坐标位于行的垂直范围之外。
  function PosWithInfo(line, ch, sticky, outside, xRel) {
    var pos = Pos(line, ch, sticky);
    pos.xRel = xRel;
    if (outside) { pos.outside = outside; }
    return pos
  }

  // 计算最接近给定坐标的字符位置。
  // 输入必须是 lineSpace-local（“div”坐标系）。
  function coordsChar(cm, x, y) {
    var doc = cm.doc;
    # 将 y 坐标加上视图偏移量
    y += cm.display.viewOffset;
    # 如果 y 坐标小于 0，则返回第一个位置
    if (y < 0) { return PosWithInfo(doc.first, 0, null, -1, -1) }
    # 获取高度对应的行号
    var lineN = lineAtHeight(doc, y), last = doc.first + doc.size - 1;
    # 如果行号大于最后一行的行号，则返回最后一个位置
    if (lineN > last)
      { return PosWithInfo(doc.first + doc.size - 1, getLine(doc, last).text.length, null, 1, 1) }
    # 如果 x 坐标小于 0，则将其设为 0
    if (x < 0) { x = 0; }
    # 获取行对象
    var lineObj = getLine(doc, lineN);
    # 无限循环，直到条件不满足
    for (;;) {
      # 在给定坐标处找到对应的字符位置
      var found = coordsCharInner(cm, lineObj, lineN, x, y);
      # 找到坍缩的范围
      var collapsed = collapsedSpanAround(lineObj, found.ch + (found.xRel > 0 || found.outside > 0 ? 1 : 0));
      # 如果没有坍缩，则返回找到的字符位置
      if (!collapsed) { return found }
      # 找到坍缩范围的结束位置
      var rangeEnd = collapsed.find(1);
      # 如果结束位置在同一行，则返回结束位置
      if (rangeEnd.line == lineN) { return rangeEnd }
      # 获取结束位置所在的行对象
      lineObj = getLine(doc, lineN = rangeEnd.line);
    }
  }

  # 获取包裹行的范围
  function wrappedLineExtent(cm, lineObj, preparedMeasure, y) {
    # 减去小部件的高度
    y -= widgetTopHeight(lineObj);
    # 获取行文本的长度
    var end = lineObj.text.length;
    # 找到第一个满足条件的字符位置
    var begin = findFirst(function (ch) { return measureCharPrepared(cm, preparedMeasure, ch - 1).bottom <= y; }, end, 0);
    # 找到最后一个满足条件的字符位置
    end = findFirst(function (ch) { return measureCharPrepared(cm, preparedMeasure, ch).top > y; }, begin, end);
    # 返回包裹行的范围
    return {begin: begin, end: end}
  }

  # 获取包裹行的范围（根据字符位置）
  function wrappedLineExtentChar(cm, lineObj, preparedMeasure, target) {
    # 如果没有准备好的测量数据，则准备测量数据
    if (!preparedMeasure) { preparedMeasure = prepareMeasureForLine(cm, lineObj); }
    # 将目标字符位置转换为行坐标系中的位置
    var targetTop = intoCoordSystem(cm, lineObj, measureCharPrepared(cm, preparedMeasure, target), "line").top;
    # 返回包裹行的范围
    return wrappedLineExtent(cm, lineObj, preparedMeasure, targetTop)
  }

  # 如果盒子的给定边在给定坐标之后，则返回 true
  function boxIsAfter(box, x, y, left) {
    return box.bottom <= y ? false : box.top > y ? true : (left ? box.left : box.right) > x
  }

  # 获取给定坐标处的字符位置
  function coordsCharInner(cm, lineObj, lineNo, x, y) {
    # 将 y 坐标转换为行局部坐标空间
    y -= heightAtLine(lineObj);
    # 准备测量数据
    var preparedMeasure = prepareMeasureForLine(cm, lineObj);
    # 当直接调用 `measureCharPrepared` 时，必须调整小部件的高度
    var widgetHeight = widgetTopHeight(lineObj);
    var begin = 0, end = lineObj.text.length, ltr = true;

    # 获取文本的排列顺序
    var order = getOrder(lineObj, cm.doc.direction);
    # 如果行不是纯左到右的文本，则首先确定坐标落入哪个双向文本段落中
    # 如果存在 order 参数
    if (order) {
      # 根据代码编辑器的设置，计算行的坐标信息
      var part = (cm.options.lineWrapping ? coordsBidiPartWrapped : coordsBidiPart)
                   (cm, lineObj, lineNo, preparedMeasure, order, x, y);
      # 判断文本方向是否为从左到右
      ltr = part.level != 1;
      # 根据文本方向确定起始和结束位置
      begin = ltr ? part.from : part.to - 1;
      end = ltr ? part.to : part.from - 1;
    }

    # 用二分查找找到第一个边界框起始位置在坐标之后的字符
    var chAround = null, boxAround = null;
    var ch = findFirst(function (ch) {
      # 计算字符的边界框
      var box = measureCharPrepared(cm, preparedMeasure, ch);
      # 调整边界框的位置
      box.top += widgetHeight; box.bottom += widgetHeight;
      # 判断边界框是否在坐标之后
      if (!boxIsAfter(box, x, y, false)) { return false }
      # 如果边界框包裹坐标，则存储相关信息
      if (box.top <= y && box.left <= x) {
        chAround = ch;
        boxAround = box;
      }
      return true
    }, begin, end);

    var baseX, sticky, outside = false;
    # 如果找到了包裹坐标的边界框
    if (boxAround) {
      # 区分坐标靠近边界框左侧还是右侧
      var atLeft = x - boxAround.left < boxAround.right - x, atStart = atLeft == ltr;
      # 根据坐标位置确定字符位置和粘性
      ch = chAround + (atStart ? 0 : 1);
      sticky = atStart ? "after" : "before";
      baseX = atLeft ? boxAround.left : boxAround.right;
    } else {
      // 如果不是从左到右的文本，且字符等于结束或开始，则调整为扩展边界
      if (!ltr && (ch == end || ch == begin)) { ch++; }
      // 确定要关联的哪一侧，获取字符左侧的框，并将其垂直位置与坐标进行比较
      sticky = ch == 0 ? "after" : ch == lineObj.text.length ? "before" :
        (measureCharPrepared(cm, preparedMeasure, ch - (ltr ? 1 : 0)).bottom + widgetHeight <= y) == ltr ?
        "after" : "before";
      // 获取此位置的准确坐标，以获取基本 X 位置
      var coords = cursorCoords(cm, Pos(lineNo, ch, sticky), "line", lineObj, preparedMeasure);
      baseX = coords.left;
      outside = y < coords.top ? -1 : y >= coords.bottom ? 1 : 0;
    }

    ch = skipExtendingChars(lineObj.text, ch, 1);
    return PosWithInfo(lineNo, ch, sticky, outside, x - baseX)
  }

  function coordsBidiPart(cm, lineObj, lineNo, preparedMeasure, order, x, y) {
    // Bidi 部分按从左到右排序，在非换行情况下，可以将此排序视为视觉排序。找到第一个结束坐标在给定坐标之后的部分
    var index = findFirst(function (i) {
      var part = order[i], ltr = part.level != 1;
      return boxIsAfter(cursorCoords(cm, Pos(lineNo, ltr ? part.to : part.from, ltr ? "before" : "after"),
                                     "line", lineObj, preparedMeasure), x, y, true)
    }, 0, order.length - 1);
    var part = order[index];
    // 如果这不是第一个部分，部分的开始也在坐标之后，并且坐标不在与该开始相同的行上，则向后移动一个部分
    // 如果索引大于0
    if (index > 0) {
      // 判断当前部分是否为一级，确定起始位置
      var ltr = part.level != 1;
      var start = cursorCoords(cm, Pos(lineNo, ltr ? part.from : part.to, ltr ? "after" : "before"),
                               "line", lineObj, preparedMeasure);
      // 判断起始位置是否在指定坐标之后，并且在指定坐标之上
      if (boxIsAfter(start, x, y, true) && start.top > y)
        { part = order[index - 1]; }
    }
    // 返回部分
    return part
  }

  // 在包裹的行中查找双向文本部分的坐标
  function coordsBidiPartWrapped(cm, lineObj, _lineNo, preparedMeasure, order, x, y) {
    // 在包裹的行中，rtl文本在包裹边界上可能会做一些与我们“order”数组中的顺序不符的事情，因此二分搜索不起作用，我们希望返回一个仅跨越一行的部分，以便在coordsCharInner中的二分搜索是安全的。因此，我们首先找到包裹行的范围，然后进行一个扁平搜索，在这个搜索中，我们丢弃任何不在行上的跨度。
    var ref = wrappedLineExtent(cm, lineObj, preparedMeasure, y);
    var begin = ref.begin;
    var end = ref.end;
    // 如果行尾有空白字符，则结束位置减一
    if (/\s/.test(lineObj.text.charAt(end - 1))) { end--; }
    var part = null, closestDist = null;
    // 遍历顺序数组
    for (var i = 0; i < order.length; i++) {
      var p = order[i];
      // 如果部分的结束位置在范围之外，则继续下一次循环
      if (p.from >= end || p.to <= begin) { continue }
      var ltr = p.level != 1;
      var endX = measureCharPrepared(cm, preparedMeasure, ltr ? Math.min(end, p.to) - 1 : Math.max(begin, p.from)).right;
      // 根据距离确定最近的部分
      var dist = endX < x ? x - endX + 1e9 : endX - x;
      if (!part || closestDist > dist) {
        part = p;
        closestDist = dist;
      }
    }
    // 如果没有找到部分，则选择顺序数组中的最后一个部分
    if (!part) { part = order[order.length - 1]; }
    // 将部分裁剪到包裹的行范围内
    if (part.from < begin) { part = {from: begin, to: part.to, level: part.level}; }
    if (part.to > end) { part = {from: part.from, to: end, level: part.level}; }
  // 返回 part 变量
  return part
}

var measureText;
// 计算默认文本高度
function textHeight(display) {
  // 如果已经缓存了文本高度，则直接返回
  if (display.cachedTextHeight != null) { return display.cachedTextHeight }
  // 如果 measureText 为空，则创建一个 pre 元素，并添加文本内容
  if (measureText == null) {
    measureText = elt("pre", null, "CodeMirror-line-like");
    // 测量一系列行的高度，用于浏览器计算分数高度
    for (var i = 0; i < 49; ++i) {
      measureText.appendChild(document.createTextNode("x"));
      measureText.appendChild(elt("br"));
    }
    measureText.appendChild(document.createTextNode("x"));
  }
  // 移除 display.measure 下的所有子元素，并添加 measureText
  removeChildrenAndAdd(display.measure, measureText);
  // 计算文本高度
  var height = measureText.offsetHeight / 50;
  // 如果高度大于 3，则缓存文本高度
  if (height > 3) { display.cachedTextHeight = height; }
  // 移除 display.measure 下的所有子元素
  removeChildren(display.measure);
  return height || 1
}

// 计算默认字符宽度
function charWidth(display) {
  // 如果已经缓存了字符宽度，则直接返回
  if (display.cachedCharWidth != null) { return display.cachedCharWidth }
  // 创建一个 span 元素，并添加文本内容
  var anchor = elt("span", "xxxxxxxxxx");
  var pre = elt("pre", [anchor], "CodeMirror-line-like");
  // 移除 display.measure 下的所有子元素，并添加 pre
  removeChildrenAndAdd(display.measure, pre);
  // 获取字符宽度
  var rect = anchor.getBoundingClientRect(), width = (rect.right - rect.left) / 10;
  // 如果宽度大于 2，则缓存字符宽度
  if (width > 2) { display.cachedCharWidth = width; }
  return width || 10
}

// 执行 DOM 位置和大小的批量读取，以便在 DOM 读取和写入之间不会交错
function getDimensions(cm) {
  var d = cm.display, left = {}, width = {};
  var gutterLeft = d.gutters.clientLeft;
  for (var n = d.gutters.firstChild, i = 0; n; n = n.nextSibling, ++i) {
    var id = cm.display.gutterSpecs[i].className;
    left[id] = n.offsetLeft + n.clientLeft + gutterLeft;
    width[id] = n.clientWidth;
  }
}
  # 返回一个对象，包含固定位置、水平滚动的补偿、行号区域总宽度、行号区域左侧位置、行号区域宽度、编辑器宽度
  return {fixedPos: compensateForHScroll(d),
          gutterTotalWidth: d.gutters.offsetWidth,
          gutterLeft: left,
          gutterWidth: width,
          wrapperWidth: d.wrapper.clientWidth}
}

// 通过使用 getBoundingClientRect 获取子像素精确结果，计算 display.scroller.scrollLeft + display.gutters.offsetWidth
function compensateForHScroll(display) {
  return display.scroller.getBoundingClientRect().left - display.sizer.getBoundingClientRect().left
}

// 返回一个函数，用于估计行高，作为第一次近似值，直到行变得可见（因此可以准确测量）
function estimateHeight(cm) {
  var th = textHeight(cm.display), wrapping = cm.options.lineWrapping;
  var perLine = wrapping && Math.max(5, cm.display.scroller.clientWidth / charWidth(cm.display) - 3);
  return function (line) {
    if (lineIsHidden(cm.doc, line)) { return 0 }

    var widgetsHeight = 0;
    if (line.widgets) { for (var i = 0; i < line.widgets.length; i++) {
      if (line.widgets[i].height) { widgetsHeight += line.widgets[i].height; }
    } }

    if (wrapping)
      { return widgetsHeight + (Math.ceil(line.text.length / perLine) || 1) * th }
    else
      { return widgetsHeight + th }
  }
}

// 估算行高
function estimateLineHeights(cm) {
  var doc = cm.doc, est = estimateHeight(cm);
  doc.iter(function (line) {
    var estHeight = est(line);
    if (estHeight != line.height) { updateLineHeight(line, estHeight); }
  });
}

// 根据鼠标事件找到相应的位置。如果 liberal 为 false，则检查是否点击了行号区域或滚动条，如果是则返回 null。forRect 用于矩形选择，尝试估计字符位置，即使坐标超出文本右侧。
function posFromMouse(cm, e, liberal, forRect) {
  var display = cm.display;
    // 如果不是宽容模式并且目标元素的属性为"cm-not-content"，则返回null
    if (!liberal && e_target(e).getAttribute("cm-not-content") == "true") { return null }

    // 获取显示行间距的矩形区域的位置信息
    var x, y, space = display.lineSpace.getBoundingClientRect();
    // 在IE[67]上当鼠标快速拖动时，会不可预测地失败
    try { x = e.clientX - space.left; y = e.clientY - space.top; }
    catch (e$1) { return null }
    // 根据鼠标位置计算出对应的字符坐标
    var coords = coordsChar(cm, x, y), line;
    // 如果是矩形选择并且字符坐标的x相对位置大于0，并且当前行的文本长度等于字符坐标的列数
    if (forRect && coords.xRel > 0 && (line = getLine(cm.doc, coords.line).text).length == coords.ch) {
      // 计算列数的差值
      var colDiff = countColumn(line, line.length, cm.options.tabSize) - line.length;
      // 根据鼠标位置重新计算字符坐标
      coords = Pos(coords.line, Math.max(0, Math.round((x - paddingH(cm.display).left) / charWidth(cm.display)) - colDiff));
    }
    // 返回字符坐标
    return coords
  }

  // 查找给定行对应的视图元素。当行不可见时返回null
  function findViewIndex(cm, n) {
    if (n >= cm.display.viewTo) { return null }
    n -= cm.display.viewFrom;
    if (n < 0) { return null }
    var view = cm.display.view;
    for (var i = 0; i < view.length; i++) {
      n -= view[i].size;
      if (n < 0) { return i }
    }
  }

  // 根据文档的变化更新显示视图的数据结构。from和to是变化前的坐标。lendiff是变化的行数。用于跨多行的变化，或者改变行的可视行数。
  function regChange(cm, from, to, lendiff) {
    if (from == null) { from = cm.doc.first; }
    if (to == null) { to = cm.doc.first + cm.doc.size; }
    if (!lendiff) { lendiff = 0; }

    var display = cm.display;
    // 如果有行数变化并且to小于视图的结束位置，并且更新行号为null或者更新行号大于from
    if (lendiff && to < display.viewTo &&
        (display.updateLineNumbers == null || display.updateLineNumbers > from))
      { display.updateLineNumbers = from; }

    // 设置当前操作的视图已改变
    cm.curOp.viewChanged = true;
    # 如果改变发生在当前显示的视图之后
    if (from >= display.viewTo) { // Change after
      # 如果存在折叠的跨度并且当前行在显示视图之前
      if (sawCollapsedSpans && visualLineNo(cm.doc, from) < display.viewTo)
        { resetView(cm); }
    } else if (to <= display.viewFrom) { // Change before
      # 如果改变发生在当前显示的视图之前
      if (sawCollapsedSpans && visualLineEndNo(cm.doc, to + lendiff) > display.viewFrom) {
        resetView(cm);
      } else {
        display.viewFrom += lendiff;
        display.viewTo += lendiff;
      }
    } else if (from <= display.viewFrom && to >= display.viewTo) { // Full overlap
      # 如果改变完全覆盖当前显示的视图
      resetView(cm);
    } else if (from <= display.viewFrom) { // Top overlap
      # 如果改变与当前显示的视图有上部重叠
      var cut = viewCuttingPoint(cm, to, to + lendiff, 1);
      if (cut) {
        display.view = display.view.slice(cut.index);
        display.viewFrom = cut.lineN;
        display.viewTo += lendiff;
      } else {
        resetView(cm);
      }
    } else if (to >= display.viewTo) { // Bottom overlap
      # 如果改变与当前显示的视图有下部重叠
      var cut$1 = viewCuttingPoint(cm, from, from, -1);
      if (cut$1) {
        display.view = display.view.slice(0, cut$1.index);
        display.viewTo = cut$1.lineN;
      } else {
        resetView(cm);
      }
    } else { // Gap in the middle
      # 如果改变在中间有间隙
      var cutTop = viewCuttingPoint(cm, from, from, -1);
      var cutBot = viewCuttingPoint(cm, to, to + lendiff, 1);
      if (cutTop && cutBot) {
        display.view = display.view.slice(0, cutTop.index)
          .concat(buildViewArray(cm, cutTop.lineN, cutBot.lineN))
          .concat(display.view.slice(cutBot.index));
        display.viewTo += lendiff;
      } else {
        resetView(cm);
      }
    }

    # 更新外部测量的行数
    var ext = display.externalMeasured;
    if (ext) {
      if (to < ext.lineN)
        { ext.lineN += lendiff; }
      else if (from < ext.lineN + ext.size)
        { display.externalMeasured = null; }
    }
  }

  # 注册对单行的更改。类型必须是 "text", "gutter", "class", "widget"
  function regLineChange(cm, line, type) {
    cm.curOp.viewChanged = true;
    // 保存对 CodeMirror 显示的引用，以及外部测量的引用
    var display = cm.display, ext = cm.display.externalMeasured;
    // 如果外部测量存在，并且行数在外部测量的范围内，则将外部测量置空
    if (ext && line >= ext.lineN && line < ext.lineN + ext.size)
      { display.externalMeasured = null; }

    // 如果行数小于显示区域的起始行或者大于等于显示区域的结束行，则返回
    if (line < display.viewFrom || line >= display.viewTo) { return }
    // 获取指定行的视图
    var lineView = display.view[findViewIndex(cm, line)];
    // 如果视图中的节点为空，则返回
    if (lineView.node == null) { return }
    // 获取视图中的变化数组，如果不存在则创建一个空数组
    var arr = lineView.changes || (lineView.changes = []);
    // 如果指定类型不在变化数组中，则将其添加到数组中
    if (indexOf(arr, type) == -1) { arr.push(type); }
  }

  // 清空视图
  function resetView(cm) {
    cm.display.viewFrom = cm.display.viewTo = cm.doc.first;
    cm.display.view = [];
    cm.display.viewOffset = 0;
  }

  // 计算视图切割点
  function viewCuttingPoint(cm, oldN, newN, dir) {
    var index = findViewIndex(cm, oldN), diff, view = cm.display.view;
    // 如果没有折叠的跨度或者 newN 等于文档的第一行加上文档的大小，则返回新的切割点
    if (!sawCollapsedSpans || newN == cm.doc.first + cm.doc.size)
      { return {index: index, lineN: newN} }
    var n = cm.display.viewFrom;
    for (var i = 0; i < index; i++)
      { n += view[i].size; }
    if (n != oldN) {
      if (dir > 0) {
        if (index == view.length - 1) { return null }
        diff = (n + view[index].size) - oldN;
        index++;
      } else {
        diff = n - oldN;
      }
      oldN += diff; newN += diff;
    }
    while (visualLineNo(cm.doc, newN) != newN) {
      if (index == (dir < 0 ? 0 : view.length - 1)) { return null }
      newN += dir * view[index - (dir < 0 ? 1 : 0)].size;
      index += dir;
    }
    return {index: index, lineN: newN}
  }

  // 强制视图覆盖给定范围，根据需要添加空的视图元素或者裁剪现有的视图元素
  function adjustView(cm, from, to) {
    var display = cm.display, view = display.view;
    // 如果视图数组为空，或者 from 大于等于显示区域的结束行，或者 to 小于等于显示区域的起始行，则重新构建视图数组
    if (view.length == 0 || from >= display.viewTo || to <= display.viewFrom) {
      display.view = buildViewArray(cm, from, to);
      display.viewFrom = from;
    } else {
      // 如果视图的结束位置大于起始位置
      if (display.viewFrom > from)
        // 构建从起始位置到视图起始位置的视图数组，并与当前视图连接
        { display.view = buildViewArray(cm, from, display.viewFrom).concat(display.view); }
      // 如果视图的结束位置小于起始位置
      else if (display.viewFrom < from)
        // 从当前视图中截取从起始位置开始的部分
        { display.view = display.view.slice(findViewIndex(cm, from)); }
      // 更新视图的起始位置
      display.viewFrom = from;
      // 如果视图的结束位置小于指定位置
      if (display.viewTo < to)
        // 将从视图的结束位置到指定位置的视图数组连接到当前视图
        { display.view = display.view.concat(buildViewArray(cm, display.viewTo, to)); }
      // 如果视图的结束位置大于指定位置
      else if (display.viewTo > to)
        // 从当前视图中截取从开始到指定位置的部分
        { display.view = display.view.slice(0, findViewIndex(cm, to)); }
    }
    // 更新视图的结束位置
    display.viewTo = to;
  }

  // 计算视图中DOM表示过时（或不存在）的行数
  function countDirtyView(cm) {
    var view = cm.display.view, dirty = 0;
    for (var i = 0; i < view.length; i++) {
      var lineView = view[i];
      // 如果行不是隐藏的且DOM节点不存在或者有变化，则增加dirty计数
      if (!lineView.hidden && (!lineView.node || lineView.changes)) { ++dirty; }
    }
    return dirty
  }

  // 更新选区
  function updateSelection(cm) {
    // 显示选区
    cm.display.input.showSelection(cm.display.input.prepareSelection());
  }

  // 准备选区
  function prepareSelection(cm, primary) {
    // 如果primary未定义，则默认为true
    if ( primary === void 0 ) primary = true;

    var doc = cm.doc, result = {};
    var curFragment = result.cursors = document.createDocumentFragment();
    var selFragment = result.selection = document.createDocumentFragment();

    for (var i = 0; i < doc.sel.ranges.length; i++) {
      // 如果不是primary选区且索引等于主要选区的索引，则跳过
      if (!primary && i == doc.sel.primIndex) { continue }
      var range = doc.sel.ranges[i];
      // 如果选区的起始行大于视图的结束位置或者结束行小于视图的起始位置，则跳过
      if (range.from().line >= cm.display.viewTo || range.to().line < cm.display.viewFrom) { continue }
      var collapsed = range.empty();
      // 如果选区是折叠的或者选区在选择时显示光标，则绘制光标
      if (collapsed || cm.options.showCursorWhenSelecting)
        { drawSelectionCursor(cm, range.head, curFragment); }
      // 如果选区不是折叠的，则绘制选区范围
      if (!collapsed)
        { drawSelectionRange(cm, range, selFragment); }
    }
    return result
  }

  // 为给定范围绘制光标
  function drawSelectionCursor(cm, head, output) {
    # 获取光标位置的像素坐标
    var pos = cursorCoords(cm, head, "div", null, null, !cm.options.singleCursorHeightPerLine);

    # 创建光标元素并设置位置和高度
    var cursor = output.appendChild(elt("div", "\u00a0", "CodeMirror-cursor"));
    cursor.style.left = pos.left + "px";
    cursor.style.top = pos.top + "px";
    cursor.style.height = Math.max(0, pos.bottom - pos.top) * cm.options.cursorHeight + "px";

    # 如果存在另一个光标位置，则创建并设置位置和高度
    if (pos.other) {
      var otherCursor = output.appendChild(elt("div", "\u00a0", "CodeMirror-cursor CodeMirror-secondarycursor"));
      otherCursor.style.display = "";
      otherCursor.style.left = pos.other.left + "px";
      otherCursor.style.top = pos.other.top + "px";
      otherCursor.style.height = (pos.other.bottom - pos.other.top) * .85 + "px";
    }
  }

  # 比较两个坐标的大小
  function cmpCoords(a, b) { return a.top - b.top || a.left - b.left }

  # 将给定范围绘制为高亮选择
  function drawSelectionRange(cm, range, output) {
    var display = cm.display, doc = cm.doc;
    var fragment = document.createDocumentFragment();
    var padding = paddingH(cm.display), leftSide = padding.left;
    var rightSide = Math.max(display.sizerWidth, displayWidth(cm) - display.sizer.offsetLeft) - padding.right;
    var docLTR = doc.direction == "ltr";

    # 添加高亮选择的绘制
    function add(left, top, width, bottom) {
      if (top < 0) { top = 0; }
      top = Math.round(top);
      bottom = Math.round(bottom);
      fragment.appendChild(elt("div", null, "CodeMirror-selected", ("position: absolute; left: " + left + "px;\n                             top: " + top + "px; width: " + (width == null ? rightSide - left : width) + "px;\n                             height: " + (bottom - top) + "px")));
    }

    # 获取范围的起始和结束位置
    var sFrom = range.from(), sTo = range.to();
    # 如果起始和结束在同一行，则绘制该行的高亮选择
    if (sFrom.line == sTo.line) {
      drawForLine(sFrom.line, sFrom.ch, sTo.ch);
    } else {
      // 获取起始行和结束行的文本
      var fromLine = getLine(doc, sFrom.line), toLine = getLine(doc, sTo.line);
      // 判断是否为单行选中
      var singleVLine = visualLine(fromLine) == visualLine(toLine);
      // 获取左侧结束位置和右侧开始位置
      var leftEnd = drawForLine(sFrom.line, sFrom.ch, singleVLine ? fromLine.text.length + 1 : null).end;
      var rightStart = drawForLine(sTo.line, singleVLine ? 0 : null, sTo.ch).start;
      // 如果是单行选中
      if (singleVLine) {
        // 判断左侧结束位置和右侧开始位置的垂直位置关系
        if (leftEnd.top < rightStart.top - 2) {
          add(leftEnd.right, leftEnd.top, null, leftEnd.bottom);
          add(leftSide, rightStart.top, rightStart.left, rightStart.bottom);
        } else {
          add(leftEnd.right, leftEnd.top, rightStart.left - leftEnd.right, leftEnd.bottom);
        }
      }
      // 如果左侧结束位置的底部小于右侧开始位置的顶部
      if (leftEnd.bottom < rightStart.top)
        { add(leftSide, leftEnd.bottom, null, rightStart.top); }
    }

    // 将 fragment 添加到 output 中
    output.appendChild(fragment);
  }

  // 光标闪烁
  function restartBlink(cm) {
    // 如果编辑器失去焦点，则返回
    if (!cm.state.focused) { return }
    var display = cm.display;
    // 清除闪烁定时器
    clearInterval(display.blinker);
    var on = true;
    display.cursorDiv.style.visibility = "";
    // 如果光标闪烁速率大于 0
    if (cm.options.cursorBlinkRate > 0)
      { 
        // 设置光标闪烁定时器
        display.blinker = setInterval(function () { return display.cursorDiv.style.visibility = (on = !on) ? "" : "hidden"; },
        cm.options.cursorBlinkRate); 
      }
    // 如果光标闪烁速率小于 0
    else if (cm.options.cursorBlinkRate < 0)
      { display.cursorDiv.style.visibility = "hidden"; }
  }

  // 确保编辑器获得焦点
  function ensureFocus(cm) {
    if (!cm.state.focused) { cm.display.input.focus(); onFocus(cm); }
  }

  // 延迟失去焦点事件
  function delayBlurEvent(cm) {
    cm.state.delayingBlurEvent = true;
    setTimeout(function () { if (cm.state.delayingBlurEvent) {
      cm.state.delayingBlurEvent = false;
      onBlur(cm);
    } }, 100);
  }

  // 编辑器获得焦点时的处理
  function onFocus(cm, e) {
    if (cm.state.delayingBlurEvent) { cm.state.delayingBlurEvent = false; }

    // 如果编辑器为只读模式且没有光标，则返回
    if (cm.options.readOnly == "nocursor") { return }
    # 如果编辑器没有焦点
    if (!cm.state.focused) {
      # 触发焦点事件
      signal(cm, "focus", cm, e);
      # 设置编辑器为已聚焦状态
      cm.state.focused = true;
      # 给编辑器添加聚焦样式
      addClass(cm.display.wrapper, "CodeMirror-focused");
      # 防止在关闭上下文菜单时触发焦点事件
      if (!cm.curOp && cm.display.selForContextMenu != cm.doc.sel) {
        # 重置输入
        cm.display.input.reset();
        # 在 Webkit 浏览器中，延迟 20 毫秒后再次重置输入
        if (webkit) { setTimeout(function () { return cm.display.input.reset(true); }, 20); } // Issue #1730
      }
      # 接收焦点
      cm.display.input.receivedFocus();
    }
    # 重新启动光标闪烁
    restartBlink(cm);
  }
  # 失去焦点事件处理函数
  function onBlur(cm, e) {
    # 如果正在延迟失焦事件，则直接返回
    if (cm.state.delayingBlurEvent) { return }

    # 如果编辑器处于聚焦状态
    if (cm.state.focused) {
      # 触发失焦事件
      signal(cm, "blur", cm, e);
      # 设置编辑器为失焦状态
      cm.state.focused = false;
      # 移除编辑器的聚焦样式
      rmClass(cm.display.wrapper, "CodeMirror-focused");
    }
    # 清除光标闪烁定时器
    clearInterval(cm.display.blinker);
    # 延迟 150 毫秒后，如果编辑器仍然处于失焦状态，则将 shift 属性设置为 false
    setTimeout(function () { if (!cm.state.focused) { cm.display.shift = false; } }, 150);
  }

  # 读取渲染行的实际高度，并更新它们的存储高度以匹配
  function updateHeightsInViewport(cm) {
    # 获取显示对象
    var display = cm.display;
    # 获取上一个行的底部偏移量
    var prevBottom = display.lineDiv.offsetTop;
    // 遍历 display.view 数组中的每个元素
    for (var i = 0; i < display.view.length; i++) {
      // 获取当前元素和是否启用行包裹的配置
      var cur = display.view[i], wrapping = cm.options.lineWrapping;
      // 定义变量 height 和 width
      var height = (void 0), width = 0;
      // 如果当前元素是隐藏的，则跳过本次循环
      if (cur.hidden) { continue }
      // 如果是 IE 并且版本小于 8
      if (ie && ie_version < 8) {
        // 获取当前元素底部位置
        var bot = cur.node.offsetTop + cur.node.offsetHeight;
        // 计算高度差
        height = bot - prevBottom;
        prevBottom = bot;
      } else {
        // 获取当前元素的盒模型信息
        var box = cur.node.getBoundingClientRect();
        // 计算高度差
        height = box.bottom - box.top;
        // 如果不启用行包裹，并且当前元素有文本节点
        if (!wrapping && cur.text.firstChild)
          { width = cur.text.firstChild.getBoundingClientRect().right - box.left - 1; }
      }
      // 计算当前行高度与实际高度的差值
      var diff = cur.line.height - height;
      // 如果差值超过阈值
      if (diff > .005 || diff < -.005) {
        // 更新行高度
        updateLineHeight(cur.line, height);
        // 更新关联行部件的高度
        updateWidgetHeight(cur.line);
        // 如果当前元素有 rest 属性
        if (cur.rest) { for (var j = 0; j < cur.rest.length; j++)
          { updateWidgetHeight(cur.rest[j]); } }
      }
      // 如果当前行宽度超过编辑器的宽度
      if (width > cm.display.sizerWidth) {
        // 计算当前行的字符宽度
        var chWidth = Math.ceil(width / charWidth(cm.display));
        // 如果字符宽度超过最大行长度
        if (chWidth > cm.display.maxLineLength) {
          // 更新最大行长度和最大行
          cm.display.maxLineLength = chWidth;
          cm.display.maxLine = cur.line;
          cm.display.maxLineChanged = true;
        }
      }
    }
  }

  // 读取并存储与给定行关联的行部件的高度
  function updateWidgetHeight(line) {
    // 如果行有部件
    if (line.widgets) { for (var i = 0; i < line.widgets.length; ++i) {
      // 获取部件节点的父节点
      var w = line.widgets[i], parent = w.node.parentNode;
      // 如果父节点存在，则更新部件高度
      if (parent) { w.height = parent.offsetHeight; }
    } }
  }

  // 计算在给定视口中可见的行
  function visibleLines(display, doc, viewport) {
    // 获取视口顶部位置
    var top = viewport && viewport.top != null ? Math.max(0, viewport.top) : display.scroller.scrollTop;
    // 计算顶部位置，减去顶部内边距
    top = Math.floor(top - paddingTop(display));
    // 计算底部位置，如果视口存在且有底部位置，则使用底部位置，否则使用顶部位置加上显示区域的高度
    var bottom = viewport && viewport.bottom != null ? viewport.bottom : top + display.wrapper.clientHeight;

    // 根据顶部和底部位置计算起始行和结束行
    var from = lineAtHeight(doc, top), to = lineAtHeight(doc, bottom);
    // 确保返回一个包含起始和结束行的对象，并将这些行强制置于视口内（如果可能的话）
    if (viewport && viewport.ensure) {
      var ensureFrom = viewport.ensure.from.line, ensureTo = viewport.ensure.to.line;
      if (ensureFrom < from) {
        from = ensureFrom;
        to = lineAtHeight(doc, heightAtLine(getLine(doc, ensureFrom)) + display.wrapper.clientHeight);
      } else if (Math.min(ensureTo, doc.lastLine()) >= to) {
        from = lineAtHeight(doc, heightAtLine(getLine(doc, ensureTo)) - display.wrapper.clientHeight);
        to = ensureTo;
      }
    }
    // 返回包含起始和结束行的对象
    return {from: from, to: Math.max(to, from + 1)}
  }

  // 将内容滚动到视图中

  // 如果编辑器位于窗口顶部或底部，部分滚出视图，这将确保光标可见
  function maybeScrollWindow(cm, rect) {
    // 如果触发了滚动光标进入视图的 DOM 事件，则返回
    if (signalDOMEvent(cm, "scrollCursorIntoView")) { return }

    var display = cm.display, box = display.sizer.getBoundingClientRect(), doScroll = null;
    // 如果矩形的顶部加上显示区域的顶部小于 0，则需要滚动
    if (rect.top + box.top < 0) { doScroll = true; }
    // 如果矩形的底部加上显示区域的顶部大于窗口高度，则不需要滚动
    else if (rect.bottom + box.top > (window.innerHeight || document.documentElement.clientHeight)) { doScroll = false; }
    # 如果 doScroll 不为空且不是幻影模式
    if (doScroll != null && !phantom) {
      # 创建一个滚动节点，设置其样式和位置信息
      var scrollNode = elt("div", "\u200b", null, ("position: absolute;\n                         top: " + (rect.top - display.viewOffset - paddingTop(cm.display)) + "px;\n                         height: " + (rect.bottom - rect.top + scrollGap(cm) + display.barHeight) + "px;\n                         left: " + (rect.left) + "px; width: " + (Math.max(2, rect.right - rect.left)) + "px;"));
      # 将滚动节点添加到代码镜像的行空间中
      cm.display.lineSpace.appendChild(scrollNode);
      # 将滚动节点滚动到视图中
      scrollNode.scrollIntoView(doScroll);
      # 从代码镜像的行空间中移除滚动节点
      cm.display.lineSpace.removeChild(scrollNode);
    }
  }

  # 将给定位置滚动到视图中（立即），验证其是否实际可见（因为行高度被准确测量，某些东西的位置在绘制过程中可能会“漂移”）
  function scrollPosIntoView(cm, pos, end, margin) {
    # 如果 margin 为空，则设置为 0
    if (margin == null) { margin = 0; }
    # 定义一个矩形变量
    var rect;
    # 如果不使用行包裹且 pos 等于 end
    if (!cm.options.lineWrapping && pos == end) {
      # 设置 pos 和 end 为光标周围的位置
      # 如果 pos.sticky == "before"，则在 pos.ch - 1 周围，否则在 pos.ch 周围
      # 如果 pos == Pos(_, 0, "before")，则保持 pos 和 end 不变
      pos = pos.ch ? Pos(pos.line, pos.sticky == "before" ? pos.ch - 1 : pos.ch, "after") : pos;
      end = pos.sticky == "before" ? Pos(pos.line, pos.ch + 1, "before") : pos;
    }
    // 循环5次，每次处理一个坐标范围
    for (var limit = 0; limit < 5; limit++) {
      // 标记是否发生了滚动
      var changed = false;
      // 获取光标位置的坐标
      var coords = cursorCoords(cm, pos);
      // 获取结束位置的坐标
      var endCoords = !end || end == pos ? coords : cursorCoords(cm, end);
      // 计算矩形范围
      rect = {left: Math.min(coords.left, endCoords.left),
              top: Math.min(coords.top, endCoords.top) - margin,
              right: Math.max(coords.left, endCoords.left),
              bottom: Math.max(coords.bottom, endCoords.bottom) + margin};
      // 计算滚动位置
      var scrollPos = calculateScrollPos(cm, rect);
      // 记录滚动前的位置
      var startTop = cm.doc.scrollTop, startLeft = cm.doc.scrollLeft;
      // 如果有需要滚动的垂直位置
      if (scrollPos.scrollTop != null) {
        // 更新垂直滚动位置
        updateScrollTop(cm, scrollPos.scrollTop);
        // 如果滚动距离超过1像素，标记为发生了滚动
        if (Math.abs(cm.doc.scrollTop - startTop) > 1) { changed = true; }
      }
      // 如果有需要滚动的水平位置
      if (scrollPos.scrollLeft != null) {
        // 更新水平滚动位置
        setScrollLeft(cm, scrollPos.scrollLeft);
        // 如果滚动距离超过1像素，标记为发生了滚动
        if (Math.abs(cm.doc.scrollLeft - startLeft) > 1) { changed = true; }
      }
      // 如果没有发生滚动，跳出循环
      if (!changed) { break }
    }
    // 返回矩形范围
    return rect
  }

  // 立即将给定的坐标范围滚动到视图中
  function scrollIntoView(cm, rect) {
    // 计算滚动位置
    var scrollPos = calculateScrollPos(cm, rect);
    // 如果有需要滚动的垂直位置
    if (scrollPos.scrollTop != null) { updateScrollTop(cm, scrollPos.scrollTop); }
    // 如果有需要滚动的水平位置
    if (scrollPos.scrollLeft != null) { setScrollLeft(cm, scrollPos.scrollLeft); }
  }

  // 计算滚动到视图中给定矩形范围所需的新滚动位置。返回一个带有scrollTop和scrollLeft属性的对象。当它们为undefined时，垂直/水平位置不需要调整。
  function calculateScrollPos(cm, rect) {
    // 获取显示区域和文本高度
    var display = cm.display, snapMargin = textHeight(cm.display);
    // 如果矩形范围的顶部小于0，将其设置为0
    if (rect.top < 0) { rect.top = 0; }
    // 获取当前滚动位置
    var screentop = cm.curOp && cm.curOp.scrollTop != null ? cm.curOp.scrollTop : display.scroller.scrollTop;
    // 获取显示区域的高度
    var screen = displayHeight(cm), result = {};
    // 如果矩形范围的高度超过显示区域的高度，将其调整为显示区域的高度
    if (rect.bottom - rect.top > screen) { rect.bottom = rect.top + screen; }
    // 获取文档底部位置
    var docBottom = cm.doc.height + paddingVert(display);
    // 检查矩形是否在顶部或底部需要进行吸附
    var atTop = rect.top < snapMargin, atBottom = rect.bottom > docBottom - snapMargin;
    // 如果矩形在屏幕顶部之上
    if (rect.top < screentop) {
      // 如果在顶部需要吸附，则滚动到顶部；否则滚动到矩形顶部
      result.scrollTop = atTop ? 0 : rect.top;
    } else if (rect.bottom > screentop + screen) {
      // 如果矩形底部超出屏幕，计算新的顶部位置并滚动
      var newTop = Math.min(rect.top, (atBottom ? docBottom : rect.bottom) - screen);
      if (newTop != screentop) { result.scrollTop = newTop; }
    }

    // 获取屏幕左侧位置和宽度
    var screenleft = cm.curOp && cm.curOp.scrollLeft != null ? cm.curOp.scrollLeft : display.scroller.scrollLeft;
    var screenw = displayWidth(cm) - (cm.options.fixedGutter ? display.gutters.offsetWidth : 0);
    // 检查矩形是否太宽，如果是则调整右侧位置
    var tooWide = rect.right - rect.left > screenw;
    if (tooWide) { rect.right = rect.left + screenw; }
    // 根据矩形左侧位置进行滚动调整
    if (rect.left < 10)
      { result.scrollLeft = 0; }
    else if (rect.left < screenleft)
      { result.scrollLeft = Math.max(0, rect.left - (tooWide ? 0 : 10)); }
    else if (rect.right > screenw + screenleft - 3)
      { result.scrollLeft = rect.right + (tooWide ? 0 : 10) - screenw; }
    // 返回滚动结果
    return result
  }

  // 在当前操作中存储相对滚动位置的调整
  function addToScrollTop(cm, top) {
    // 如果top为null，则直接返回
    if (top == null) { return }
    resolveScrollToPos(cm);
    // 计算滚动后的顶部位置
    cm.curOp.scrollTop = (cm.curOp.scrollTop == null ? cm.doc.scrollTop : cm.curOp.scrollTop) + top;
  }

  // 确保在操作结束时当前光标可见
  function ensureCursorVisible(cm) {
    resolveScrollToPos(cm);
    var cur = cm.getCursor();
    // 设置滚动到光标位置的参数
    cm.curOp.scrollToPos = {from: cur, to: cur, margin: cm.options.cursorScrollMargin};
  }

  // 滚动到指定坐标
  function scrollToCoords(cm, x, y) {
    // 如果x或y不为null，则进行滚动位置的解析
    if (x != null || y != null) { resolveScrollToPos(cm); }
    // 如果x不为null，则设置水平滚动位置
    if (x != null) { cm.curOp.scrollLeft = x; }
    // 如果y不为null，则设置垂直滚动位置
    if (y != null) { cm.curOp.scrollTop = y; }
  }

  // 滚动到指定范围
  function scrollToRange(cm, range) {
    resolveScrollToPos(cm);
    # 设置当前操作的滚动位置为指定范围
    cm.curOp.scrollToPos = range;
  }

  // 当操作的 scrollToPos 属性被设置，并且在操作结束之前应用了另一个滚动操作时，
  // 这种方式“模拟”将该位置以一种廉价的方式滚动到视图中，以便中间滚动命令的效果不被忽略。
  function resolveScrollToPos(cm) {
    var range = cm.curOp.scrollToPos;
    if (range) {
      cm.curOp.scrollToPos = null;
      var from = estimateCoords(cm, range.from), to = estimateCoords(cm, range.to);
      scrollToCoordsRange(cm, from, to, range.margin);
    }
  }

  function scrollToCoordsRange(cm, from, to, margin) {
    var sPos = calculateScrollPos(cm, {
      left: Math.min(from.left, to.left),
      top: Math.min(from.top, to.top) - margin,
      right: Math.max(from.right, to.right),
      bottom: Math.max(from.bottom, to.bottom) + margin
    });
    scrollToCoords(cm, sPos.scrollLeft, sPos.scrollTop);
  }

  // 同步可滚动区域和滚动条，确保视口覆盖可见区域。
  function updateScrollTop(cm, val) {
    if (Math.abs(cm.doc.scrollTop - val) < 2) { return }
    if (!gecko) { updateDisplaySimple(cm, {top: val}); }
    setScrollTop(cm, val, true);
    if (gecko) { updateDisplaySimple(cm); }
    startWorker(cm, 100);
  }

  function setScrollTop(cm, val, forceScroll) {
    val = Math.max(0, Math.min(cm.display.scroller.scrollHeight - cm.display.scroller.clientHeight, val));
    if (cm.display.scroller.scrollTop == val && !forceScroll) { return }
    cm.doc.scrollTop = val;
    cm.display.scrollbars.setScrollTop(val);
    if (cm.display.scroller.scrollTop != val) { cm.display.scroller.scrollTop = val; }
  }

  // 同步滚动条和滚动条，确保 gutter 元素对齐。
  function setScrollLeft(cm, val, isScroller, forceScroll) {
    val = Math.max(0, Math.min(val, cm.display.scroller.scrollWidth - cm.display.scroller.clientWidth));
    // 如果是滚动条，且值等于文档的水平滚动位置，或者滚动位置变化小于2，并且不是强制滚动，则返回
    if ((isScroller ? val == cm.doc.scrollLeft : Math.abs(cm.doc.scrollLeft - val) < 2) && !forceScroll) { return }
    // 设置文档的水平滚动位置
    cm.doc.scrollLeft = val;
    // 水平对齐文本
    alignHorizontally(cm);
    // 如果显示区域的水平滚动位置不等于给定值，则设置显示区域的水平滚动位置
    if (cm.display.scroller.scrollLeft != val) { cm.display.scroller.scrollLeft = val; }
    // 设置滚动条的水平滚动位置
    cm.display.scrollbars.setScrollLeft(val);
  }

  // SCROLLBARS

  // 准备更新滚动条所需的 DOM 读取。一次性完成以最小化更新/测量的往返
  function measureForScrollbars(cm) {
    var d = cm.display, gutterW = d.gutters.offsetWidth;
    var docH = Math.round(cm.doc.height + paddingVert(cm.display));
    return {
      clientHeight: d.scroller.clientHeight,
      viewHeight: d.wrapper.clientHeight,
      scrollWidth: d.scroller.scrollWidth, clientWidth: d.scroller.clientWidth,
      viewWidth: d.wrapper.clientWidth,
      barLeft: cm.options.fixedGutter ? gutterW : 0,
      docHeight: docH,
      scrollHeight: docH + scrollGap(cm) + d.barHeight,
      nativeBarWidth: d.nativeBarWidth,
      gutterWidth: gutterW
    }
  }

  // 创建原生滚动条对象
  var NativeScrollbars = function(place, scroll, cm) {
    this.cm = cm;
    // 创建垂直滚动条元素
    var vert = this.vert = elt("div", [elt("div", null, null, "min-width: 1px")], "CodeMirror-vscrollbar");
    // 创建水平滚动条元素
    var horiz = this.horiz = elt("div", [elt("div", null, null, "height: 100%; min-height: 1px")], "CodeMirror-hscrollbar");
    // 设置滚动条元素的 tabIndex 属性
    vert.tabIndex = horiz.tabIndex = -1;
    // 将垂直滚动条元素和水平滚动条元素添加到指定位置
    place(vert); place(horiz);

    // 监听垂直滚动条的滚动事件
    on(vert, "scroll", function () {
      if (vert.clientHeight) { scroll(vert.scrollTop, "vertical"); }
    });
    // 监听水平滚动条的滚动事件
    on(horiz, "scroll", function () {
      if (horiz.clientWidth) { scroll(horiz.scrollLeft, "horizontal"); }
    });

    this.checkedZeroWidth = false;
    // 需要在 IE7 上设置最小宽度以显示滚动条（但不能在 IE8 上设置）
    if (ie && ie_version < 8) { this.horiz.style.minHeight = this.vert.style.minWidth = "18px"; }
  };

  // 更新滚动条
  NativeScrollbars.prototype.update = function (measure) {
    # 检查是否需要垂直滚动条
    var needsH = measure.scrollWidth > measure.clientWidth + 1;
    # 检查是否需要水平滚动条
    var needsV = measure.scrollHeight > measure.clientHeight + 1;
    # 获取滚动条的宽度
    var sWidth = measure.nativeBarWidth;

    # 如果需要垂直滚动条
    if (needsV) {
      # 显示垂直滚动条
      this.vert.style.display = "block";
      # 设置垂直滚动条的位置
      this.vert.style.bottom = needsH ? sWidth + "px" : "0";
      # 计算垂直滚动条的高度
      var totalHeight = measure.viewHeight - (needsH ? sWidth : 0);
      # 修复 IE8 中可能出现的负值 bug
      this.vert.firstChild.style.height =
        Math.max(0, measure.scrollHeight - measure.clientHeight + totalHeight) + "px";
    } else {
      # 隐藏垂直滚动条
      this.vert.style.display = "";
      this.vert.firstChild.style.height = "0";
    }

    # 如果需要水平滚动条
    if (needsH) {
      # 显示水平滚动条
      this.horiz.style.display = "block";
      # 设置水平滚动条的位置
      this.horiz.style.right = needsV ? sWidth + "px" : "0";
      this.horiz.style.left = measure.barLeft + "px";
      # 计算水平滚动条的宽度
      var totalWidth = measure.viewWidth - measure.barLeft - (needsV ? sWidth : 0);
      this.horiz.firstChild.style.width =
        Math.max(0, measure.scrollWidth - measure.clientWidth + totalWidth) + "px";
    } else {
      # 隐藏水平滚动条
      this.horiz.style.display = "";
      this.horiz.firstChild.style.width = "0";
    }

    # 如果未检查过滚动条宽度且可视区域高度大于0
    if (!this.checkedZeroWidth && measure.clientHeight > 0) {
      # 如果滚动条宽度为0，则执行 zeroWidthHack 方法
      if (sWidth == 0) { this.zeroWidthHack(); }
      this.checkedZeroWidth = true;
    }

    # 返回滚动条的右侧和底部偏移量
    return {right: needsV ? sWidth : 0, bottom: needsH ? sWidth : 0}
  };

  # 设置水平滚动条的位置
  NativeScrollbars.prototype.setScrollLeft = function (pos) {
    if (this.horiz.scrollLeft != pos) { this.horiz.scrollLeft = pos; }
    # 如果禁用水平滚动条，则启用零宽度滚动条
    if (this.disableHoriz) { this.enableZeroWidthBar(this.horiz, this.disableHoriz, "horiz"); }
  };

  # 设置垂直滚动条的位置
  NativeScrollbars.prototype.setScrollTop = function (pos) {
    if (this.vert.scrollTop != pos) { this.vert.scrollTop = pos; }
    # 如果禁用垂直滚动条，则启用零宽度滚动条
    if (this.disableVert) { this.enableZeroWidthBar(this.vert, this.disableVert, "vert"); }
  };

  # 修复零宽度滚动条的方法
  NativeScrollbars.prototype.zeroWidthHack = function () {
    # 根据浏览器类型设置滚动条宽度
    var w = mac && !mac_geMountainLion ? "12px" : "18px";
  // 设置水平滚动条的高度和垂直滚动条的宽度为指定的宽度
  this.horiz.style.height = this.vert.style.width = w;
  // 设置水平滚动条和垂直滚动条的指针事件为"none"
  this.horiz.style.pointerEvents = this.vert.style.pointerEvents = "none";
  // 创建延迟对象来禁用水平滚动条和垂直滚动条
  this.disableHoriz = new Delayed;
  this.disableVert = new Delayed;
};

// 启用宽度为零的滚动条
NativeScrollbars.prototype.enableZeroWidthBar = function (bar, delay, type) {
  // 设置滚动条的指针事件为"auto"
  bar.style.pointerEvents = "auto";
  // 定义函数来检查滚动条是否仍然可见，并根据情况禁用指针事件
  function maybeDisable() {
    var box = bar.getBoundingClientRect();
    var elt = type == "vert" ? document.elementFromPoint(box.right - 1, (box.top + box.bottom) / 2)
        : document.elementFromPoint((box.right + box.left) / 2, box.bottom - 1);
    if (elt != bar) { bar.style.pointerEvents = "none"; }
    else { delay.set(1000, maybeDisable); }
  }
  delay.set(1000, maybeDisable);
};

// 清除滚动条
NativeScrollbars.prototype.clear = function () {
  var parent = this.horiz.parentNode;
  parent.removeChild(this.horiz);
  parent.removeChild(this.vert);
};

// 定义空滚动条对象
var NullScrollbars = function () {};

// 更新空滚动条
NullScrollbars.prototype.update = function () { return {bottom: 0, right: 0} };
NullScrollbars.prototype.setScrollLeft = function () {};
NullScrollbars.prototype.setScrollTop = function () {};
NullScrollbars.prototype.clear = function () {};

// 更新滚动条
function updateScrollbars(cm, measure) {
  if (!measure) { measure = measureForScrollbars(cm); }
  var startWidth = cm.display.barWidth, startHeight = cm.display.barHeight;
  updateScrollbarsInner(cm, measure);
}
    # 循环执行以下操作，直到满足条件：i < 4 且 startWidth 不等于 cm.display.barWidth 或 startHeight 不等于 cm.display.barHeight
    for (var i = 0; i < 4 && startWidth != cm.display.barWidth || startHeight != cm.display.barHeight; i++) {
      # 如果 startWidth 不等于 cm.display.barWidth 并且 cm.options.lineWrapping 为真，则执行 updateHeightsInViewport 函数
      if (startWidth != cm.display.barWidth && cm.options.lineWrapping)
        { updateHeightsInViewport(cm); }
      # 调用 updateScrollbarsInner 函数，传入参数 cm 和 measureForScrollbars(cm)，更新滚动条
      updateScrollbarsInner(cm, measureForScrollbars(cm));
      # 更新 startWidth 和 startHeight 的值
      startWidth = cm.display.barWidth; startHeight = cm.display.barHeight;
    }
  }

  // 重新同步虚拟滚动条与内容的实际大小
  function updateScrollbarsInner(cm, measure) {
    # 获取 cm.display 对象
    var d = cm.display;
    # 调用 d.scrollbars.update 函数，传入参数 measure，更新滚动条的大小
    var sizes = d.scrollbars.update(measure);

    # 设置 d.sizer 的右内边距为滚动条的右侧宽度，并将其转换为像素单位
    d.sizer.style.paddingRight = (d.barWidth = sizes.right) + "px";
    # 设置 d.sizer 的底部内边距为滚动条的底部高度，并将其转换为像素单位
    d.sizer.style.paddingBottom = (d.barHeight = sizes.bottom) + "px";
    # 设置 d.heightForcer 的底部边框为透明，高度为滚动条的底部高度，并将其转换为像素单位
    d.heightForcer.style.borderBottom = sizes.bottom + "px solid transparent";

    # 如果右侧滚动条和底部滚动条都存在，则设置 d.scrollbarFiller 的显示方式为块级元素，并设置其高度和宽度
    if (sizes.right && sizes.bottom) {
      d.scrollbarFiller.style.display = "block";
      d.scrollbarFiller.style.height = sizes.bottom + "px";
      d.scrollbarFiller.style.width = sizes.right + "px";
    } else { d.scrollbarFiller.style.display = ""; }
    # 如果底部滚动条存在，并且 cm.options.coverGutterNextToScrollbar 为真且 cm.options.fixedGutter 为真，则设置 d.gutterFiller 的显示方式为块级元素，并设置其高度和宽度
    if (sizes.bottom && cm.options.coverGutterNextToScrollbar && cm.options.fixedGutter) {
      d.gutterFiller.style.display = "block";
      d.gutterFiller.style.height = sizes.bottom + "px";
      d.gutterFiller.style.width = measure.gutterWidth + "px";
    } else { d.gutterFiller.style.display = ""; }
  }

  # 定义滚动条模型对象，包含 "native" 和 "null" 两种类型的滚动条
  var scrollbarModel = {"native": NativeScrollbars, "null": NullScrollbars};

  # 初始化滚动条
  function initScrollbars(cm) {
    # 如果 cm.display.scrollbars 存在，则清除滚动条
    if (cm.display.scrollbars) {
      cm.display.scrollbars.clear();
      # 如果 cm.display.scrollbars.addClass 存在，则移除 cm.display.wrapper 的指定类名
      if (cm.display.scrollbars.addClass)
        { rmClass(cm.display.wrapper, cm.display.scrollbars.addClass); }
    }
    # 设置滚动条样式，并将滚动条插入到编辑器显示区域中
    cm.display.scrollbars = new scrollbarModel[cm.options.scrollbarStyle](function (node) {
      cm.display.wrapper.insertBefore(node, cm.display.scrollbarFiller);
      # 防止在滚动条上的点击事件导致失去焦点
      on(node, "mousedown", function () {
        if (cm.state.focused) { setTimeout(function () { return cm.display.input.focus(); }, 0); }
      });
      # 设置节点属性，表示不是编辑器内容
      node.setAttribute("cm-not-content", "true");
    }, function (pos, axis) {
      # 根据滚动条的方向设置水平或垂直滚动位置
      if (axis == "horizontal") { setScrollLeft(cm, pos); }
      else { updateScrollTop(cm, pos); }
    }, cm);
    # 如果滚动条有额外的类，则添加到编辑器显示区域
    if (cm.display.scrollbars.addClass)
      { addClass(cm.display.wrapper, cm.display.scrollbars.addClass); }
  }

  # 操作用于包装对编辑器状态的一系列更改，以便每个更改不必更新光标和显示（这将是笨拙、缓慢和容易出错的）。相反，显示更新被批处理，然后全部组合并执行。

  # 下一个操作的 ID
  var nextOpId = 0;
  # 开始一个新的操作
  function startOperation(cm) {
    // 设置当前操作的属性
    cm.curOp = {
      cm: cm,  // 当前编辑器实例的引用
      viewChanged: false,      // 标志，指示可能需要重新绘制行
      startHeight: cm.doc.height, // 用于检测是否需要更新滚动条
      forceUpdate: false,      // 用于强制重新绘制
      updateInput: 0,       // 是否重置输入文本区域
      typing: false,           // 是否小心地保留现有文本（用于组合）
      changeObjs: null,        // 累积的更改，用于触发更改事件
      cursorActivityHandlers: null, // 要在光标活动上触发的处理程序集
      cursorActivityCalled: 0, // 跟踪已经调用的光标活动处理程序
      selectionChanged: false, // 是否需要重新绘制选择内容
      updateMaxLine: false,    // 当需要重新确定最宽的行时设置
      scrollLeft: null, scrollTop: null, // 中间滚动位置，尚未推送到 DOM
      scrollToPos: null,       // 用于滚动到特定位置
      focus: false,            // 是否聚焦
      id: ++nextOpId           // 唯一 ID
    };
    // 将当前操作推入操作栈
    pushOperation(cm.curOp);
  }

  // 结束一个操作，更新显示并发出延迟事件
  function endOperation(cm) {
    var op = cm.curOp;
    if (op) { finishOperation(op, function (group) {
      for (var i = 0; i < group.ops.length; i++)
        { group.ops[i].cm.curOp = null; }  // 清空操作引用
      endOperations(group);
    }); }
  }

  // 操作完成时进行的 DOM 更新被批处理，以便需要重新布局的次数最少
  function endOperations(group) {
    var ops = group.ops;
    for (var i = 0; i < ops.length; i++) // 读取 DOM
      { endOperation_R1(ops[i]); }
    for (var i$1 = 0; i$1 < ops.length; i$1++) // 写入 DOM（可能）
      { endOperation_W1(ops[i$1]); }
    for (var i$2 = 0; i$2 < ops.length; i$2++) // 读取 DOM
      { endOperation_R2(ops[i$2]); }
    // 遍历操作数组，执行结束操作，可能会写入 DOM
    for (var i$3 = 0; i$3 < ops.length; i$3++) // Write DOM (maybe)
      { endOperation_W2(ops[i$3]); }
    // 再次遍历操作数组，执行结束操作，读取 DOM
    for (var i$4 = 0; i$4 < ops.length; i$4++) // Read DOM
      { endOperation_finish(ops[i$4]); }
  }

  // 结束读取操作，处理相关逻辑
  function endOperation_R1(op) {
    var cm = op.cm, display = cm.display;
    // 可能裁剪滚动条
    maybeClipScrollbars(cm);
    // 如果需要更新最大行数，则查找最大行
    if (op.updateMaxLine) { findMaxLine(cm); }

    // 判断是否需要更新
    op.mustUpdate = op.viewChanged || op.forceUpdate || op.scrollTop != null ||
      op.scrollToPos && (op.scrollToPos.from.line < display.viewFrom ||
                         op.scrollToPos.to.line >= display.viewTo) ||
      display.maxLineChanged && cm.options.lineWrapping;
    // 如果需要更新，则创建显示更新对象
    op.update = op.mustUpdate &&
      new DisplayUpdate(cm, op.mustUpdate && {top: op.scrollTop, ensure: op.scrollToPos}, op.forceUpdate);
  }

  // 结束写入操作，处理相关逻辑
  function endOperation_W1(op) {
    // 如果需要更新显示，则更新显示
    op.updatedDisplay = op.mustUpdate && updateDisplayIfNeeded(op.cm, op.update);
  }

  // 结束读取操作，处理相关逻辑
  function endOperation_R2(op) {
    var cm = op.cm, display = cm.display;
    // 如果更新了显示，则更新视口内的高度
    if (op.updatedDisplay) { updateHeightsInViewport(cm); }

    // 计算滚动条的测量值
    op.barMeasure = measureForScrollbars(cm);

    // 如果最大行数发生变化且不换行，则测量最大行数的宽度，并确保文档的宽度与之匹配
    if (display.maxLineChanged && !cm.options.lineWrapping) {
      op.adjustWidthTo = measureChar(cm, display.maxLine, display.maxLine.text.length).left + 3;
      cm.display.sizerWidth = op.adjustWidthTo;
      op.barMeasure.scrollWidth =
        Math.max(display.scroller.clientWidth, display.sizer.offsetLeft + op.adjustWidthTo + scrollGap(cm) + cm.display.barWidth);
      op.maxScrollLeft = Math.max(0, display.sizer.offsetLeft + op.adjustWidthTo - displayWidth(cm));
    }

    // 如果更新了显示或选择发生变化，则准备选择
    if (op.updatedDisplay || op.selectionChanged)
      { op.preparedSelection = display.input.prepareSelection(); }
  }

  // 结束写入操作，处理相关逻辑
  function endOperation_W2(op) {
    var cm = op.cm;
    // 如果需要调整宽度，则设置显示区域的最小宽度
    if (op.adjustWidthTo != null) {
      cm.display.sizer.style.minWidth = op.adjustWidthTo + "px";
      // 如果最大滚动左边小于文档的滚动左边，则设置滚动左边的位置
      if (op.maxScrollLeft < cm.doc.scrollLeft)
        { setScrollLeft(cm, Math.min(cm.display.scroller.scrollLeft, op.maxScrollLeft), true); }
      // 标记最大行改变为 false
      cm.display.maxLineChanged = false;
    }

    // 如果需要获取焦点，并且焦点是活动元素
    var takeFocus = op.focus && op.focus == activeElt();
    // 如果有准备好的选择，则显示选择
    if (op.preparedSelection)
      { cm.display.input.showSelection(op.preparedSelection, takeFocus); }
    // 如果更新了显示或者开始高度不等于文档高度，则更新滚动条
    if (op.updatedDisplay || op.startHeight != cm.doc.height)
      { updateScrollbars(cm, op.barMeasure); }
    // 如果更新了显示，则设置文档高度
    if (op.updatedDisplay)
      { setDocumentHeight(cm, op.barMeasure); }

    // 如果选择改变，则重新开始闪烁
    if (op.selectionChanged) { restartBlink(cm); }

    // 如果编辑器处于焦点状态，并且需要更新输入，则重置输入
    if (cm.state.focused && op.updateInput)
      { cm.display.input.reset(op.typing); }
    // 如果需要获取焦点，则确保焦点在编辑器上
    if (takeFocus) { ensureFocus(op.cm); }
  }

  // 结束操作后的处理
  function endOperation_finish(op) {
    var cm = op.cm, display = cm.display, doc = cm.doc;

    // 如果更新了显示，则在更新后处理显示
    if (op.updatedDisplay) { postUpdateDisplay(cm, op.update); }

    // 当显式滚动时，中止鼠标滚轮的 delta 测量
    if (display.wheelStartX != null && (op.scrollTop != null || op.scrollLeft != null || op.scrollToPos))
      { display.wheelStartX = display.wheelStartY = null; }

    // 将滚动位置传播到实际的 DOM 滚动器
    if (op.scrollTop != null) { setScrollTop(cm, op.scrollTop, op.forceScroll); }

    // 如果需要滚动左边，则设置滚动左边的位置
    if (op.scrollLeft != null) { setScrollLeft(cm, op.scrollLeft, true, true); }
    // 如果需要将特定位置滚动到视图中，则执行
    if (op.scrollToPos) {
      var rect = scrollPosIntoView(cm, clipPos(doc, op.scrollToPos.from),
                                   clipPos(doc, op.scrollToPos.to), op.scrollToPos.margin);
      maybeScrollWindow(cm, rect);
    }

    // 通过编辑或撤销隐藏/显示标记的事件
    var hidden = op.maybeHiddenMarkers, unhidden = op.maybeUnhiddenMarkers;
    // 如果存在隐藏的内容，遍历隐藏的内容
    if (hidden) { for (var i = 0; i < hidden.length; ++i)
      { if (!hidden[i].lines.length) { signal(hidden[i], "hide"); } } }
    // 如果存在未隐藏的内容，遍历未隐藏的内容
    if (unhidden) { for (var i$1 = 0; i$1 < unhidden.length; ++i$1)
      { if (unhidden[i$1].lines.length) { signal(unhidden[i$1], "unhide"); } } }

    // 如果显示区域的高度不为0，设置文档的滚动位置
    if (display.wrapper.offsetHeight)
      { doc.scrollTop = cm.display.scroller.scrollTop; }

    // 触发变化事件和延迟事件处理程序
    if (op.changeObjs)
      { signal(cm, "changes", cm, op.changeObjs); }
    if (op.update)
      { op.update.finish(); }
  }

  // 在操作中运行给定的函数
  function runInOp(cm, f) {
    if (cm.curOp) { return f() }
    startOperation(cm);
    try { return f() }
    finally { endOperation(cm); }
  }
  // 将函数包装在操作中。返回包装后的函数。
  function operation(cm, f) {
    return function() {
      if (cm.curOp) { return f.apply(cm, arguments) }
      startOperation(cm);
      try { return f.apply(cm, arguments) }
      finally { endOperation(cm); }
    }
  }
  // 用于向编辑器和文档实例添加方法，将它们包装在操作中。
  function methodOp(f) {
    return function() {
      if (this.curOp) { return f.apply(this, arguments) }
      startOperation(this);
      try { return f.apply(this, arguments) }
      finally { endOperation(this); }
    }
  }
  function docMethodOp(f) {
    return function() {
      var cm = this.cm;
      if (!cm || cm.curOp) { return f.apply(this, arguments) }
      startOperation(cm);
      try { return f.apply(this, arguments) }
      finally { endOperation(cm); }
    }
  }

  // 高亮工作器

  // 开始工作器
  function startWorker(cm, time) {
    if (cm.doc.highlightFrontier < cm.display.viewTo)
      { cm.state.highlight.set(time, bind(highlightWorker, cm)); }
  }

  // 高亮工作器
  function highlightWorker(cm) {
    var doc = cm.doc;
    if (doc.highlightFrontier >= cm.display.viewTo) { return }
    var end = +new Date + cm.options.workTime;
    // 获取当前上下文
    var context = getContextBefore(cm, doc.highlightFrontier);
    // 存储改变的行
    var changedLines = [];

    // 迭代文档中的行
    doc.iter(context.line, Math.min(doc.first + doc.size, cm.display.viewTo + 500), function (line) {
      // 如果当前行可见
      if (context.line >= cm.display.viewFrom) { // Visible
        // 存储旧的样式
        var oldStyles = line.styles;
        // 如果行文本长度超过最大高亮长度，则复制当前状态
        var resetState = line.text.length > cm.options.maxHighlightLength ? copyState(doc.mode, context.state) : null;
        // 高亮当前行
        var highlighted = highlightLine(cm, line, context, true);
        // 如果有复制的状态，则将当前状态重置为复制的状态
        if (resetState) { context.state = resetState; }
        // 更新行的样式
        line.styles = highlighted.styles;
        // 更新行的类
        var oldCls = line.styleClasses, newCls = highlighted.classes;
        if (newCls) { line.styleClasses = newCls; }
        else if (oldCls) { line.styleClasses = null; }
        // 检查行是否改变
        var ischange = !oldStyles || oldStyles.length != line.styles.length ||
          oldCls != newCls && (!oldCls || !newCls || oldCls.bgClass != newCls.bgClass || oldCls.textClass != newCls.textClass);
        for (var i = 0; !ischange && i < oldStyles.length; ++i) { ischange = oldStyles[i] != line.styles[i]; }
        // 如果行改变，则将其添加到改变的行列表中
        if (ischange) { changedLines.push(context.line); }
        // 保存当前行的状态
        line.stateAfter = context.save();
        // 移动到下一行
        context.nextLine();
      } else {
        // 如果行文本长度小于等于最大高亮长度，则处理当前行
        if (line.text.length <= cm.options.maxHighlightLength)
          { processLine(cm, line.text, context); }
        // 每隔5行保存当前行的状态
        line.stateAfter = context.line % 5 == 0 ? context.save() : null;
        // 移动到下一行
        context.nextLine();
      }
      // 如果当前时间大于结束时间，则启动工作线程并返回true
      if (+new Date > end) {
        startWorker(cm, cm.options.workDelay);
        return true
      }
    });
    // 更新文档的高亮边界
    doc.highlightFrontier = context.line;
    // 更新文档的模式边界
    doc.modeFrontier = Math.max(doc.modeFrontier, context.line);
    // 如果有改变的行，则在操作中运行
    if (changedLines.length) { runInOp(cm, function () {
      for (var i = 0; i < changedLines.length; i++)
        { regLineChange(cm, changedLines[i], "text"); }
    }); }
  }

  // 显示绘制

  // 定义DisplayUpdate构造函数
  var DisplayUpdate = function(cm, viewport, force) {
    var display = cm.display;

    this.viewport = viewport;
  // 存储一些稍后需要使用的值（但不想强制重新布局）
  this.visible = visibleLines(display, cm.doc, viewport);
  this.editorIsHidden = !display.wrapper.offsetWidth;
  this.wrapperHeight = display.wrapper.clientHeight;
  this.wrapperWidth = display.wrapper.clientWidth;
  this.oldDisplayWidth = displayWidth(cm);
  this.force = force;
  this.dims = getDimensions(cm);
  this.events = [];
};

// 发出信号
DisplayUpdate.prototype.signal = function (emitter, type) {
  if (hasHandler(emitter, type))
    { this.events.push(arguments); }
};
// 完成更新
DisplayUpdate.prototype.finish = function () {
  for (var i = 0; i < this.events.length; i++)
    { signal.apply(null, this.events[i]); }
};

// 可能裁剪滚动条
function maybeClipScrollbars(cm) {
  var display = cm.display;
  if (!display.scrollbarsClipped && display.scroller.offsetWidth) {
    display.nativeBarWidth = display.scroller.offsetWidth - display.scroller.clientWidth;
    display.heightForcer.style.height = scrollGap(cm) + "px";
    display.sizer.style.marginBottom = -display.nativeBarWidth + "px";
    display.sizer.style.borderRightWidth = scrollGap(cm) + "px";
    display.scrollbarsClipped = true;
  }
}

// 选择快照
function selectionSnapshot(cm) {
  if (cm.hasFocus()) { return null }
  var active = activeElt();
  if (!active || !contains(cm.display.lineDiv, active)) { return null }
  var result = {activeElt: active};
  if (window.getSelection) {
    var sel = window.getSelection();
    if (sel.anchorNode && sel.extend && contains(cm.display.lineDiv, sel.anchorNode)) {
      result.anchorNode = sel.anchorNode;
      result.anchorOffset = sel.anchorOffset;
      result.focusNode = sel.focusNode;
      result.focusOffset = sel.focusOffset;
    }
  }
  return result
}

// 恢复选择
function restoreSelection(snapshot) {
  if (!snapshot || !snapshot.activeElt || snapshot.activeElt == activeElt()) { return }
  snapshot.activeElt.focus();
    # 如果快照的活动元素不是输入框或文本区域，并且锚点和焦点节点都在文档主体内
    if (!/^(INPUT|TEXTAREA)$/.test(snapshot.activeElt.nodeName) &&
        snapshot.anchorNode && contains(document.body, snapshot.anchorNode) && contains(document.body, snapshot.focusNode)) {
      # 获取当前窗口的选择对象和文档范围
      var sel = window.getSelection(), range = document.createRange();
      # 设置范围的结束点为锚点节点和偏移量
      range.setEnd(snapshot.anchorNode, snapshot.anchorOffset);
      # 将范围折叠到结束点
      range.collapse(false);
      # 移除所有的范围并添加新的范围
      sel.removeAllRanges();
      sel.addRange(range);
      # 将选择对象扩展到焦点节点和偏移量
      sel.extend(snapshot.focusNode, snapshot.focusOffset);
    }
  }

  // 执行行显示的实际更新。当没有需要执行的更新且不是强制更新时，返回 false
  function updateDisplayIfNeeded(cm, update) {
    var display = cm.display, doc = cm.doc;

    if (update.editorIsHidden) {
      resetView(cm);
      return false
    }

    # 如果编辑器被隐藏，则重置视图并返回 false
    if (!update.force &&
        update.visible.from >= display.viewFrom && update.visible.to <= display.viewTo &&
        (display.updateLineNumbers == null || display.updateLineNumbers >= display.viewTo) &&
        display.renderedView == display.view && countDirtyView(cm) == 0)
      { return false }

    # 如果可见区域已经渲染并且没有变化，则返回 false
    if (maybeUpdateLineNumberWidth(cm)) {
      resetView(cm);
      update.dims = getDimensions(cm);
    }

    # 计算适当的新视口（from 和 to）
    var end = doc.first + doc.size;
    var from = Math.max(update.visible.from - cm.options.viewportMargin, doc.first);
    var to = Math.min(end, update.visible.to + cm.options.viewportMargin);
    if (display.viewFrom < from && from - display.viewFrom < 20) { from = Math.max(doc.first, display.viewFrom); }
    if (display.viewTo > to && display.viewTo - to < 20) { to = Math.min(end, display.viewTo); }
    if (sawCollapsedSpans) {
      from = visualLineNo(cm.doc, from);
      to = visualLineEndNo(cm.doc, to);
    }
    # 如果存在折叠的跨度，则调整 from 和 to 的值
    # 检查视图是否发生了变化
    var different = from != display.viewFrom || to != display.viewTo ||
      display.lastWrapHeight != update.wrapperHeight || display.lastWrapWidth != update.wrapperWidth;
    # 调整编辑器的视图范围
    adjustView(cm, from, to);

    # 设置视图偏移量，使移动的 div 与当前滚动位置对齐
    display.viewOffset = heightAtLine(getLine(cm.doc, display.viewFrom));
    cm.display.mover.style.top = display.viewOffset + "px";

    # 计算需要更新的视图数量
    var toUpdate = countDirtyView(cm);
    # 如果视图没有变化且无需强制更新，则直接返回
    if (!different && toUpdate == 0 && !update.force && display.renderedView == display.view &&
        (display.updateLineNumbers == null || display.updateLineNumbers >= display.viewTo))
      { return false }

    # 对于大的变化，隐藏包裹元素以加快操作速度
    var selSnapshot = selectionSnapshot(cm);
    if (toUpdate > 4) { display.lineDiv.style.display = "none"; }
    # 更新显示
    patchDisplay(cm, display.updateLineNumbers, update.dims);
    if (toUpdate > 4) { display.lineDiv.style.display = ""; }
    display.renderedView = display.view;
    # 如果有小部件包含焦点元素被隐藏或更新，重新聚焦它
    restoreSelection(selSnapshot);

    # 防止选择和光标干扰滚动宽度和高度
    removeChildren(display.cursorDiv);
    removeChildren(display.selectionDiv);
    display.gutters.style.height = display.sizer.style.minHeight = 0;

    # 如果视图发生变化，更新最后的包裹高度和宽度，并启动 worker
    if (different) {
      display.lastWrapHeight = update.wrapperHeight;
      display.lastWrapWidth = update.wrapperWidth;
      startWorker(cm, 400);
    }

    display.updateLineNumbers = null;

    return true
  }

  # 更新显示后的处理
  function postUpdateDisplay(cm, update) {
    var viewport = update.viewport;
    // 使用 for 循环，初始化 first 为 true，无限循环，每次循环结束后将 first 设置为 false
    for (var first = true;; first = false) {
      // 如果不是第一次循环，或者不是自动换行，或者更新前的显示宽度不等于当前显示宽度
      if (!first || !cm.options.lineWrapping || update.oldDisplayWidth == displayWidth(cm)) {
        // 将强制视口裁剪到实际可滚动区域
        if (viewport && viewport.top != null)
          { viewport = {top: Math.min(cm.doc.height + paddingVert(cm.display) - displayHeight(cm), viewport.top)}; }
        // 更新行高可能导致绘制区域实际上并未覆盖视口。保持循环直到覆盖为止。
        update.visible = visibleLines(cm.display, cm.doc, viewport);
        // 如果更新后的可见行范围从视口开始到结束都在显示范围内，则跳出循环
        if (update.visible.from >= cm.display.viewFrom && update.visible.to <= cm.display.viewTo)
          { break }
      } else if (first) {
        // 如果是第一次循环，则更新可见行范围
        update.visible = visibleLines(cm.display, cm.doc, viewport);
      }
      // 如果需要更新显示，则执行更新
      if (!updateDisplayIfNeeded(cm, update)) { break }
      // 更新视口内的行高
      updateHeightsInViewport(cm);
      // 测量滚动条
      var barMeasure = measureForScrollbars(cm);
      // 更新选择
      updateSelection(cm);
      // 更新滚动条
      updateScrollbars(cm, barMeasure);
      // 设置文档高度
      setDocumentHeight(cm, barMeasure);
      // 取消强制更新
      update.force = false;
    }

    // 发送更新信号
    update.signal(cm, "update", cm);
    // 如果显示范围发生变化，则发送视口变化信号
    if (cm.display.viewFrom != cm.display.reportedViewFrom || cm.display.viewTo != cm.display.reportedViewTo) {
      update.signal(cm, "viewportChange", cm, cm.display.viewFrom, cm.display.viewTo);
      cm.display.reportedViewFrom = cm.display.viewFrom; cm.display.reportedViewTo = cm.display.viewTo;
    }
  }

  // 简单更新显示函数
  function updateDisplaySimple(cm, viewport) {
    // 创建更新对象
    var update = new DisplayUpdate(cm, viewport);
    // 如果需要更新显示，则执行更新
    if (updateDisplayIfNeeded(cm, update)) {
      // 更新视口内的行高
      updateHeightsInViewport(cm);
      // 更新显示后的处理
      postUpdateDisplay(cm, update);
      // 测量滚动条
      var barMeasure = measureForScrollbars(cm);
      // 更新选择
      updateSelection(cm);
      // 更新滚动条
      updateScrollbars(cm, barMeasure);
      // 设置文档高度
      setDocumentHeight(cm, barMeasure);
      // 完成更新
      update.finish();
  }
}

// 同步实际显示的 DOM 结构与 display.view，移除不再在视图中的行的节点，创建尚未存在的节点，并更新过时的节点。
function patchDisplay(cm, updateNumbersFrom, dims) {
  var display = cm.display, lineNumbers = cm.options.lineNumbers;
  var container = display.lineDiv, cur = container.firstChild;

  function rm(node) {
    var next = node.nextSibling;
    // 解决 OS X Webkit 中的一个抛出滚动 bug
    if (webkit && mac && cm.display.currentWheelTarget == node)
      { node.style.display = "none"; }
    else
      { node.parentNode.removeChild(node); }
    return next
  }

  var view = display.view, lineN = display.viewFrom;
  // 遍历视图中的元素，同步 cur（display.lineDiv 中的 DOM 节点）与视图
  for (var i = 0; i < view.length; i++) {
    var lineView = view[i];
    if (lineView.hidden) ; else if (!lineView.node || lineView.node.parentNode != container) { // 尚未绘制
      var node = buildLineElement(cm, lineView, lineN, dims);
      container.insertBefore(node, cur);
    } else { // 已经绘制
      while (cur != lineView.node) { cur = rm(cur); }
      var updateNumber = lineNumbers && updateNumbersFrom != null &&
        updateNumbersFrom <= lineN && lineView.lineNumber;
      if (lineView.changes) {
        if (indexOf(lineView.changes, "gutter") > -1) { updateNumber = false; }
        updateLineForChanges(cm, lineView, lineN, dims);
      }
      if (updateNumber) {
        removeChildren(lineView.lineNumber);
        lineView.lineNumber.appendChild(document.createTextNode(lineNumberFor(cm.options, lineN)));
      }
      cur = lineView.node.nextSibling;
    }
    lineN += lineView.size;
  }
  while (cur) { cur = rm(cur); }
}

function updateGutterSpace(display) {
  var width = display.gutters.offsetWidth;
  // 设置显示区域的左边距为指定宽度
  display.sizer.style.marginLeft = width + "px";
}

// 设置文档的高度
function setDocumentHeight(cm, measure) {
  // 设置显示区域的最小高度
  cm.display.sizer.style.minHeight = measure.docHeight + "px";
  // 设置高度强制器的位置
  cm.display.heightForcer.style.top = measure.docHeight + "px";
  // 设置行号区域的高度
  cm.display.gutters.style.height = (measure.docHeight + cm.display.barHeight + scrollGap(cm)) + "px";
}

// 重新对齐行号和边线标记以补偿水平滚动
function alignHorizontally(cm) {
  var display = cm.display, view = display.view;
  if (!display.alignWidgets && (!display.gutters.firstChild || !cm.options.fixedGutter)) { return }
  var comp = compensateForHScroll(display) - display.scroller.scrollLeft + cm.doc.scrollLeft;
  var gutterW = display.gutters.offsetWidth, left = comp + "px";
  for (var i = 0; i < view.length; i++) { if (!view[i].hidden) {
    if (cm.options.fixedGutter) {
      if (view[i].gutter)
        { view[i].gutter.style.left = left; }
      if (view[i].gutterBackground)
        { view[i].gutterBackground.style.left = left; }
    }
    var align = view[i].alignable;
    if (align) { for (var j = 0; j < align.length; j++)
      { align[j].style.left = left; } }
  } }
  if (cm.options.fixedGutter)
    { display.gutters.style.left = (comp + gutterW) + "px"; }
}

// 用于确保行号区域的大小适合当前文档大小。当需要更新时返回 true。
function maybeUpdateLineNumberWidth(cm) {
  if (!cm.options.lineNumbers) { return false }
  var doc = cm.doc, last = lineNumberFor(cm.options, doc.first + doc.size - 1), display = cm.display;
    // 检查最后一行的长度是否与显示行号的长度相等
    if (last.length != display.lineNumChars) {
      // 创建一个测试元素，用于测量行号的宽度
      var test = display.measure.appendChild(elt("div", [elt("div", last)],
                                                 "CodeMirror-linenumber CodeMirror-gutter-elt"));
      // 获取内部宽度和内外宽度之间的差值，用于计算行号的宽度
      var innerW = test.firstChild.offsetWidth, padding = test.offsetWidth - innerW;
      // 重置行号容器的宽度
      display.lineGutter.style.width = "";
      // 计算行号的内部宽度
      display.lineNumInnerWidth = Math.max(innerW, display.lineGutter.offsetWidth - padding) + 1;
      // 计算行号的总宽度
      display.lineNumWidth = display.lineNumInnerWidth + padding;
      // 更新显示行号的字符数
      display.lineNumChars = display.lineNumInnerWidth ? last.length : -1;
      // 设置行号容器的宽度
      display.lineGutter.style.width = display.lineNumWidth + "px";
      // 更新编辑器的边距空间
      updateGutterSpace(cm.display);
      // 返回 true
      return true
    }
    // 返回 false
    return false
  }

  // 获取所有的 gutter 样式和类名
  function getGutters(gutters, lineNumbers) {
    var result = [], sawLineNumbers = false;
    for (var i = 0; i < gutters.length; i++) {
      var name = gutters[i], style = null;
      // 如果 gutter 不是字符串，则获取样式和类名
      if (typeof name != "string") { style = name.style; name = name.className; }
      // 如果是行号的 gutter，则根据 lineNumbers 决定是否添加到结果中
      if (name == "CodeMirror-linenumbers") {
        if (!lineNumbers) { continue }
        else { sawLineNumbers = true; }
      }
      // 将样式和类名添加到结果中
      result.push({className: name, style: style});
    }
    // 如果需要行号且没有发现行号的 gutter，则添加行号的 gutter 到结果中
    if (lineNumbers && !sawLineNumbers) { result.push({className: "CodeMirror-linenumbers", style: null}); }
    // 返回结果
    return result
  }

  // 重建 gutter 元素，确保代码左侧的边距与它们的宽度匹配
  function renderGutters(display) {
    var gutters = display.gutters, specs = display.gutterSpecs;
    // 移除所有 gutter 元素
    removeChildren(gutters);
    // 重置行号容器
    display.lineGutter = null;
    // 遍历 specs 数组，对每个元素执行以下操作
    for (var i = 0; i < specs.length; ++i) {
      // 获取当前元素的引用
      var ref = specs[i];
      // 获取当前元素的 className
      var className = ref.className;
      // 获取当前元素的 style
      var style = ref.style;
      // 在 gutters 中添加一个 div 元素，类名为 "CodeMirror-gutter " + className
      var gElt = gutters.appendChild(elt("div", null, "CodeMirror-gutter " + className));
      // 如果存在 style，则将 gElt 的样式设置为 style
      if (style) { gElt.style.cssText = style; }
      // 如果 className 为 "CodeMirror-linenumbers"
      if (className == "CodeMirror-linenumbers") {
        // 将 gElt 赋值给 display.lineGutter
        display.lineGutter = gElt;
        // 设置 gElt 的宽度为 display.lineNumWidth 或 1 像素
        gElt.style.width = (display.lineNumWidth || 1) + "px";
      }
    }
    // 如果 specs 数组长度大于 0，则设置 gutters 的显示为默认值，否则设置为 "none"
    gutters.style.display = specs.length ? "" : "none";
    // 更新 gutter 区域的空间
    updateGutterSpace(display);
  }

  // 更新 gutter 区域
  function updateGutters(cm) {
    // 渲染 gutter 区域
    renderGutters(cm.display);
    // 注册改变
    regChange(cm);
    // 水平对齐
    alignHorizontally(cm);
  }

  // Display 类处理 DOM 集成，包括输入读取和内容绘制。它保存了 DOM 节点和与显示相关的状态。
  function Display(place, doc, input, options) {
    var d = this;
    this.input = input;

    // 用于覆盖当两个滚动条同时存在时的底部右侧方块
    d.scrollbarFiller = elt("div", null, "CodeMirror-scrollbar-filler");
    d.scrollbarFiller.setAttribute("cm-not-content", "true");
    // 用于覆盖当 coverGutterNextToScrollbar 开启且水平滚动条存在时的 gutter 底部
    d.gutterFiller = elt("div", null, "CodeMirror-gutter-filler");
    d.gutterFiller.setAttribute("cm-not-content", "true");
    // 包含实际代码的元素，定位以覆盖视口
    d.lineDiv = eltP("div", null, "CodeMirror-code");
    // 用于添加表示选择和光标的元素
    d.selectionDiv = elt("div", null, null, "position: relative; z-index: 1");
    d.cursorDiv = elt("div", null, "CodeMirror-cursors");
    // 用于找到元素大小的 visibility: hidden 元素
    d.measure = elt("div", null, "CodeMirror-measure");
    // 当视口外的行被测量时，它们会在这里绘制
    d.lineMeasure = elt("div", null, "CodeMirror-measure");
    // 将需要存在于垂直填充坐标系统内的所有内容进行包装
    d.lineSpace = eltP("div", [d.measure, d.lineMeasure, d.selectionDiv, d.cursorDiv, d.lineDiv],
                      null, "position: relative; outline: none");
    // 移动到其父元素周围，以覆盖可见视图
    var lines = eltP("div", [d.lineSpace], "CodeMirror-lines");
    // 设置为文档的高度，允许滚动
    d.mover = elt("div", [lines], null, "position: relative");
    // 用于确保具有 overflow: auto 和 padding 的元素在各个浏览器中的行为一致
    d.sizer = elt("div", [d.mover], "CodeMirror-sizer");
    d.sizerWidth = null;
    // 将包含 gutter（如果有）的元素
    d.heightForcer = elt("div", null, null, "position: absolute; height: " + scrollerGap + "px; width: 1px;");
    // 实际可滚动的元素
    d.gutters = elt("div", null, "CodeMirror-gutters");
    d.lineGutter = null;
    d.scroller = elt("div", [d.sizer, d.heightForcer, d.gutters], "CodeMirror-scroll");
    d.scroller.setAttribute("tabIndex", "-1");
    // 编辑器所在的元素
    d.wrapper = elt("div", [d.scrollbarFiller, d.gutterFiller, d.scroller], "CodeMirror");

    // 解决 IE7 的 z-index bug（不完美，因此 IE7 实际上不受支持）
    if (ie && ie_version < 8) { d.gutters.style.zIndex = -1; d.scroller.style.paddingRight = 0; }
    if (!webkit && !(gecko && mobile)) { d.scroller.draggable = true; }

    if (place) {
      if (place.appendChild) { place.appendChild(d.wrapper); }
      else { place(d.wrapper); }
    }

    // 当前渲染的范围（可能比视图窗口更大）
    d.viewFrom = d.viewTo = doc.first;
    d.reportedViewFrom = d.reportedViewTo = doc.first;
    // 关于渲染行的信息
    d.view = [];
    d.renderedView = null;
    // 在渲染时保存有关单个渲染行的信息
    // 用于测量，不在视图中时
    d.externalMeasured = null;
    // 视图上方的空白空间（以像素为单位）
    d.viewOffset = 0;
    d.lastWrapHeight = d.lastWrapWidth = 0;
    d.updateLineNumbers = null;

    d.nativeBarWidth = d.barHeight = d.barWidth = 0;
    d.scrollbarsClipped = false;

    // 用于仅在必要时调整行号边栏的宽度（当行数跨越边界时，其宽度会改变）
    d.lineNumWidth = d.lineNumInnerWidth = d.lineNumChars = null;
    // 当添加非水平滚动的行部件时设置为true。作为优化，当此值为false时，跳过行部件对齐。
    d.alignWidgets = false;

    d.cachedCharWidth = d.cachedTextHeight = d.cachedPaddingH = null;

    // 跟踪最大行长度，以便在滚动时保持水平滚动条静态。
    d.maxLine = null;
    d.maxLineLength = 0;
    d.maxLineChanged = false;

    // 用于测量滚轮滚动的粒度
    d.wheelDX = d.wheelDY = d.wheelStartX = d.wheelStartY = null;

    // 当按住Shift键时为true。
    d.shift = false;

    // 用于跟踪自上次打开上下文菜单以来是否发生了任何事件。
    d.selForContextMenu = null;

    d.activeTouch = null;

    d.gutterSpecs = getGutters(options.gutters, options.lineNumbers);
    renderGutters(d);
  // 初始化输入对象，传入参数 d
  input.init(d);
}

// 由于鼠标滚轮事件中的增量值在不同浏览器和版本之间不一致，并且通常难以预测，因此此代码首先测量前几个鼠标滚轮事件的滚动效果，并从中检测出如何将增量转换为像素偏移量。
//
// 我们想要知道滚轮事件将滚动的量的原因是，它给了我们在实际滚动发生之前更新显示的机会，从而减少闪烁。

var wheelSamples = 0, wheelPixelsPerUnit = null;
// 在我们知道的浏览器中填入一个浏览器检测到的起始值。这些值不必准确 -- 如果它们错误，结果只会是第一次滚轮滚动时轻微的闪烁（如果足够大）。
if (ie) { wheelPixelsPerUnit = -.53; }
else if (gecko) { wheelPixelsPerUnit = 15; }
else if (chrome) { wheelPixelsPerUnit = -.7; }
else if (safari) { wheelPixelsPerUnit = -1/3; }

function wheelEventDelta(e) {
  var dx = e.wheelDeltaX, dy = e.wheelDeltaY;
  if (dx == null && e.detail && e.axis == e.HORIZONTAL_AXIS) { dx = e.detail; }
  if (dy == null && e.detail && e.axis == e.VERTICAL_AXIS) { dy = e.detail; }
  else if (dy == null) { dy = e.wheelDelta; }
  return {x: dx, y: dy}
}
function wheelEventPixels(e) {
  var delta = wheelEventDelta(e);
  delta.x *= wheelPixelsPerUnit;
  delta.y *= wheelPixelsPerUnit;
  return delta
}

function onScrollWheel(cm, e) {
  var delta = wheelEventDelta(e), dx = delta.x, dy = delta.y;

  var display = cm.display, scroll = display.scroller;
  // 如果这里没有需要滚动的内容，则退出
  var canScrollX = scroll.scrollWidth > scroll.clientWidth;
  var canScrollY = scroll.scrollHeight > scroll.clientHeight;
  if (!(dx && canScrollX || dy && canScrollY)) { return }

  // 在 OS X 上，Webkit 浏览器在动量滚动时会中止滚动
    // 如果发生垂直滚动并且是在 Mac 平台上使用 Webkit 浏览器
    // 则执行以下操作
    if (dy && mac && webkit) {
      // 遍历滚动事件的目标元素及其父元素
      outer: for (var cur = e.target, view = display.view; cur != scroll; cur = cur.parentNode) {
        // 遍历显示视图中的节点，如果找到目标元素，则设置当前滚轮目标为该元素
        for (var i = 0; i < view.length; i++) {
          if (view[i].node == cur) {
            cm.display.currentWheelTarget = cur;
            break outer
          }
        }
      }
    }

    // 在某些浏览器上，水平滚动会导致重新绘制发生在 gutter 重新对齐之前，导致 gutter 在屏幕上闪烁。
    // 当我们有一个估计的像素/滚动单位值时，我们在这里完全处理水平滚动。
    // 这可能会略微偏离原生滚动，但比出现故障要好。
    if (dx && !gecko && !presto && wheelPixelsPerUnit != null) {
      // 如果发生垂直滚动并且可以滚动，则更新滚动条的位置
      if (dy && canScrollY)
        { updateScrollTop(cm, Math.max(0, scroll.scrollTop + dy * wheelPixelsPerUnit)); }
      // 更新水平滚动条的位置
      setScrollLeft(cm, Math.max(0, scroll.scrollLeft + dx * wheelPixelsPerUnit));
      // 只有在垂直滚动实际上是可能的情况下才阻止默认滚动。
      // 否则，当 deltaX 很小而 deltaY 很大时，会在 OSX 触摸板上引起垂直滚动抖动（问题＃3579）
      if (!dy || (dy && canScrollY))
        { e_preventDefault(e); }
      // 中止测量，如果正在进行中
      display.wheelStartX = null;
      return
    }

    // 如果发生垂直滚动并且有估计的像素/滚动单位值
    // 则执行以下操作
    if (dy && wheelPixelsPerUnit != null) {
      // 计算滚动的像素值
      var pixels = dy * wheelPixelsPerUnit;
      var top = cm.doc.scrollTop, bot = top + display.wrapper.clientHeight;
      // 如果像素值小于0，则调整顶部位置
      if (pixels < 0) { top = Math.max(0, top + pixels - 50); }
      // 否则，调整底部位置
      else { bot = Math.min(cm.doc.height, bot + pixels + 50); }
      // 更新显示视图
      updateDisplaySimple(cm, {top: top, bottom: bot});
    }
    // 如果滚轮样本小于20
    if (wheelSamples < 20) {
      // 如果滚轮开始X坐标为空
      if (display.wheelStartX == null) {
        // 设置滚轮开始X和Y坐标为当前滚动条的左偏移和上偏移
        display.wheelStartX = scroll.scrollLeft; display.wheelStartY = scroll.scrollTop;
        // 设置滚轮X和Y方向的偏移量
        display.wheelDX = dx; display.wheelDY = dy;
        // 设置一个延迟函数，用于计算滚轮的移动距离
        setTimeout(function () {
          // 如果滚轮开始X坐标为空，则返回
          if (display.wheelStartX == null) { return }
          // 计算滚动条的X和Y方向的移动距离
          var movedX = scroll.scrollLeft - display.wheelStartX;
          var movedY = scroll.scrollTop - display.wheelStartY;
          // 计算滚轮的样本值
          var sample = (movedY && display.wheelDY && movedY / display.wheelDY) ||
            (movedX && display.wheelDX && movedX / display.wheelDX);
          // 重置滚轮开始X和Y坐标
          display.wheelStartX = display.wheelStartY = null;
          // 如果样本值不存在，则返回
          if (!sample) { return }
          // 更新滚轮每单位像素的值和样本数量
          wheelPixelsPerUnit = (wheelPixelsPerUnit * wheelSamples + sample) / (wheelSamples + 1);
          ++wheelSamples;
        }, 200);
      } else {
        // 如果滚轮开始X坐标不为空，则更新滚轮X和Y方向的偏移量
        display.wheelDX += dx; display.wheelDY += dy;
      }
    }
  }

  // 选择对象是不可变的。每次选择更改时都会创建一个新的选择对象。
  // 选择是一个或多个不重叠（且不接触）的范围，排序，并且一个整数表示哪一个是主要选择（滚动到视图中，getCursor返回等）。
  var Selection = function(ranges, primIndex) {
    this.ranges = ranges;
    this.primIndex = primIndex;
  };

  // 返回主要选择的范围
  Selection.prototype.primary = function () { return this.ranges[this.primIndex] };

  // 比较两个选择对象是否相等
  Selection.prototype.equals = function (other) {
    // 如果other等于this，则返回true
    if (other == this) { return true }
    // 如果other的主要索引不等于this的主要索引，或者other的范围长度不等于this的范围长度，则返回false
    if (other.primIndex != this.primIndex || other.ranges.length != this.ranges.length) { return false }
    // 遍历范围数组，比较锚点和头部是否相等
    for (var i = 0; i < this.ranges.length; i++) {
      var here = this.ranges[i], there = other.ranges[i];
      if (!equalCursorPos(here.anchor, there.anchor) || !equalCursorPos(here.head, there.head)) { return false }
    }
    return true
  };

  // 深拷贝选择对象
  Selection.prototype.deepCopy = function () {
    var out = [];
  // 遍历this.ranges数组，复制每个范围的锚点和头部，存入out数组
  for (var i = 0; i < this.ranges.length; i++)
    { out[i] = new Range(copyPos(this.ranges[i].anchor), copyPos(this.ranges[i].head)); }
  // 返回一个新的Selection对象，包含复制后的范围数组和primIndex属性
  return new Selection(out, this.primIndex)
};

// 检查是否有选中内容
Selection.prototype.somethingSelected = function () {
  // 遍历this.ranges数组，如果有非空范围则返回true，否则返回false
  for (var i = 0; i < this.ranges.length; i++)
    { if (!this.ranges[i].empty()) { return true } }
  return false
};

// 检查给定的位置范围是否在选中内容中
Selection.prototype.contains = function (pos, end) {
  // 如果end未定义，则将其设置为pos
  if (!end) { end = pos; }
  // 遍历this.ranges数组，检查给定的位置范围是否在范围内，返回范围的索引，如果不在范围内则返回-1
  for (var i = 0; i < this.ranges.length; i++) {
    var range = this.ranges[i];
    if (cmp(end, range.from()) >= 0 && cmp(pos, range.to()) <= 0)
      { return i }
  }
  return -1
};

// 定义Range对象，包含锚点和头部属性
var Range = function(anchor, head) {
  this.anchor = anchor; this.head = head;
};

// 返回范围的起始位置
Range.prototype.from = function () { return minPos(this.anchor, this.head) };
// 返回范围的结束位置
Range.prototype.to = function () { return maxPos(this.anchor, this.head) };
// 检查范围是否为空
Range.prototype.empty = function () { return this.head.line == this.anchor.line && this.head.ch == this.anchor.ch };

// 标准化选择范围，处理重叠和排序
function normalizeSelection(cm, ranges, primIndex) {
  // 获取cm.options.selectionsMayTouch属性
  var mayTouch = cm && cm.options.selectionsMayTouch;
  // 获取主要范围
  var prim = ranges[primIndex];
  // 对范围数组进行排序
  ranges.sort(function (a, b) { return cmp(a.from(), b.from()); });
  // 获取主要范围在排序后的索引
  primIndex = indexOf(ranges, prim);
  // 遍历范围数组，处理重叠的范围
  for (var i = 1; i < ranges.length; i++) {
    var cur = ranges[i], prev = ranges[i - 1];
    var diff = cmp(prev.to(), cur.from());
    if (mayTouch && !cur.empty() ? diff > 0 : diff >= 0) {
      var from = minPos(prev.from(), cur.from()), to = maxPos(prev.to(), cur.to());
      var inv = prev.empty() ? cur.from() == cur.head : prev.from() == prev.head;
      if (i <= primIndex) { --primIndex; }
      ranges.splice(--i, 2, new Range(inv ? to : from, inv ? from : to));
    }
  }
}
  // 返回一个新的 Selection 对象，包含给定的 ranges 和 primIndex
  return new Selection(ranges, primIndex)
}

// 创建一个简单的 Selection 对象，包含一个 Range 对象，起始位置为 anchor，结束位置为 head 或者 anchor
function simpleSelection(anchor, head) {
  return new Selection([new Range(anchor, head || anchor)], 0)
}

// 计算一个 change 的结束位置（'to' 属性指向变化前的结束位置）
function changeEnd(change) {
  if (!change.text) { return change.to }
  return Pos(change.from.line + change.text.length - 1,
             lst(change.text).length + (change.text.length == 1 ? change.from.ch : 0))
}

// 调整一个位置，使其指向变化后的位置，如果变化覆盖了该位置，则指向变化的结束位置
function adjustForChange(pos, change) {
  if (cmp(pos, change.from) < 0) { return pos }
  if (cmp(pos, change.to) <= 0) { return changeEnd(change) }

  var line = pos.line + change.text.length - (change.to.line - change.from.line) - 1, ch = pos.ch;
  if (pos.line == change.to.line) { ch += changeEnd(change).ch - change.to.ch; }
  return Pos(line, ch)
}

// 根据变化计算出变化后的 Selection 对象
function computeSelAfterChange(doc, change) {
  var out = [];
  for (var i = 0; i < doc.sel.ranges.length; i++) {
    var range = doc.sel.ranges[i];
    out.push(new Range(adjustForChange(range.anchor, change),
                       adjustForChange(range.head, change)));
  }
  return normalizeSelection(doc.cm, out, doc.sel.primIndex)
}

// 根据旧的位置和新的位置，计算出偏移后的位置
function offsetPos(pos, old, nw) {
  if (pos.line == old.line)
    { return Pos(nw.line, pos.ch - old.ch + nw.ch) }
  else
    { return Pos(nw.line + (pos.line - old.line), pos.ch) }
}

// 用于 replaceSelections，允许将选择移动到替换文本的开头或周围。提示可以是 "start" 或 "around"
function computeReplacedSel(doc, changes, hint) {
  var out = [];
  var oldPrev = Pos(doc.first, 0), newPrev = oldPrev;
}
    # 遍历 changes 数组，对每个 change 进行处理
    for (var i = 0; i < changes.length; i++) {
      # 获取当前 change
      var change = changes[i];
      # 计算 from 坐标
      var from = offsetPos(change.from, oldPrev, newPrev);
      # 计算 to 坐标
      var to = offsetPos(changeEnd(change), oldPrev, newPrev);
      # 更新 oldPrev 和 newPrev
      oldPrev = change.to;
      newPrev = to;
      # 根据 hint 的值进行不同的处理
      if (hint == "around") {
        # 获取当前 range 和其方向
        var range = doc.sel.ranges[i], inv = cmp(range.head, range.anchor) < 0;
        # 根据 inv 的值创建 Range 对象
        out[i] = new Range(inv ? to : from, inv ? from : to);
      } else {
        # 根据 from 创建 Range 对象
        out[i] = new Range(from, from);
      }
    }
    # 返回新的 Selection 对象
    return new Selection(out, doc.sel.primIndex)
  }

  # 用于在选项更改时将编辑器恢复到一致的状态
  function loadMode(cm) {
    # 设置编辑器的模式
    cm.doc.mode = getMode(cm.options, cm.doc.modeOption);
    # 重置模式状态
    resetModeState(cm);
  }

  # 重置模式状态
  function resetModeState(cm) {
    # 遍历文档的每一行，重置状态
    cm.doc.iter(function (line) {
      if (line.stateAfter) { line.stateAfter = null; }
      if (line.styles) { line.styles = null; }
    });
    # 重置模式的前沿位置
    cm.doc.modeFrontier = cm.doc.highlightFrontier = cm.doc.first;
    # 启动 worker
    startWorker(cm, 100);
    # 增加模式的生成次数
    cm.state.modeGen++;
    if (cm.curOp) { regChange(cm); }
  }

  # 默认情况下，处理从行首开始到行尾的更新
  function isWholeLineUpdate(doc, change) {
    return change.from.ch == 0 && change.to.ch == 0 && lst(change.text) == "" &&
      (!doc.cm || doc.cm.options.wholeLineUpdateBefore)
  }

  # 对文档数据结构执行更改
  function updateDoc(doc, change, markedSpans, estimateHeight) {
    # 返回指定行的标记范围
    function spansFor(n) {return markedSpans ? markedSpans[n] : null}
    # 更新行的内容和标记范围
    function update(line, text, spans) {
      updateLine(line, text, spans, estimateHeight);
      signalLater(line, "change", line, change);
    }
    // 根据给定的起始行和结束行，生成包含这些行内容的数组
    function linesFor(start, end) {
      var result = [];
      for (var i = start; i < end; ++i)
        { result.push(new Line(text[i], spansFor(i), estimateHeight)); }
      return result
    }
    
    var from = change.from, to = change.to, text = change.text;
    var firstLine = getLine(doc, from.line), lastLine = getLine(doc, to.line);
    var lastText = lst(text), lastSpans = spansFor(text.length - 1), nlines = to.line - from.line;
    
    // 调整行结构
    if (change.full) {
      doc.insert(0, linesFor(0, text.length));
      doc.remove(text.length, doc.size - text.length);
    } else if (isWholeLineUpdate(doc, change)) {
      // 这是一个整行替换。特殊处理以确保行对象移动到它们应该去的地方。
      var added = linesFor(0, text.length - 1);
      update(lastLine, lastLine.text, lastSpans);
      if (nlines) { doc.remove(from.line, nlines); }
      if (added.length) { doc.insert(from.line, added); }
    } else if (firstLine == lastLine) {
      if (text.length == 1) {
        update(firstLine, firstLine.text.slice(0, from.ch) + lastText + firstLine.text.slice(to.ch), lastSpans);
      } else {
        var added$1 = linesFor(1, text.length - 1);
        added$1.push(new Line(lastText + firstLine.text.slice(to.ch), lastSpans, estimateHeight));
        update(firstLine, firstLine.text.slice(0, from.ch) + text[0], spansFor(0));
        doc.insert(from.line + 1, added$1);
      }
    } else if (text.length == 1) {
      update(firstLine, firstLine.text.slice(0, from.ch) + text[0] + lastLine.text.slice(to.ch), spansFor(0));
      doc.remove(from.line + 1, nlines);
    } else {
      update(firstLine, firstLine.text.slice(0, from.ch) + text[0], spansFor(0));
      update(lastLine, lastText + lastLine.text.slice(to.ch), lastSpans);
      var added$2 = linesFor(1, text.length - 1);
      if (nlines > 1) { doc.remove(from.line + 1, nlines - 1); }
      doc.insert(from.line + 1, added$2);
    }
  // 为文档添加一个“change”事件的监听器，当文档发生变化时调用给定的函数
  signalLater(doc, "change", doc, change);
}

// 对所有链接的文档调用函数f
function linkedDocs(doc, f, sharedHistOnly) {
  function propagate(doc, skip, sharedHist) {
    if (doc.linked) { for (var i = 0; i < doc.linked.length; ++i) {
      var rel = doc.linked[i];
      if (rel.doc == skip) { continue }
      var shared = sharedHist && rel.sharedHist;
      if (sharedHistOnly && !shared) { continue }
      f(rel.doc, shared);
      propagate(rel.doc, doc, shared);
    } }
  }
  propagate(doc, null, true);
}

// 将文档附加到编辑器
function attachDoc(cm, doc) {
  if (doc.cm) { throw new Error("This document is already in use.") }
  cm.doc = doc;
  doc.cm = cm;
  estimateLineHeights(cm);
  loadMode(cm);
  setDirectionClass(cm);
  if (!cm.options.lineWrapping) { findMaxLine(cm); }
  cm.options.mode = doc.modeOption;
  regChange(cm);
}

// 设置文本方向的类
function setDirectionClass(cm) {
  (cm.doc.direction == "rtl" ? addClass : rmClass)(cm.display.lineDiv, "CodeMirror-rtl");
}

// 方向改变时调用
function directionChanged(cm) {
  runInOp(cm, function () {
    setDirectionClass(cm);
    regChange(cm);
  });
}

// 历史记录对象
function History(startGen) {
  // 变更事件和选择的数组。执行操作会将事件添加到done数组并清除undone数组。撤销操作会将事件从done移动到undone，重做操作则相反。
  this.done = []; this.undone = [];
  this.undoDepth = Infinity;
  // 用于跟踪何时可以将变更合并为单个撤销事件
  this.lastModTime = this.lastSelTime = 0;
  this.lastOp = this.lastSelOp = null;
  this.lastOrigin = this.lastSelOrigin = null;
  // 由isClean()方法使用
  this.generation = this.maxGeneration = startGen || 1;
}

// 从updateDoc-style变更对象创建历史变更事件
function historyChangeFromChange(doc, change) {
  // 创建一个包含历史变化信息的对象，包括变化的起始位置、结束位置和文本内容
  var histChange = {from: copyPos(change.from), to: changeEnd(change), text: getBetween(doc, change.from, change.to)};
  // 在文档中附加本地跨度
  attachLocalSpans(doc, histChange, change.from.line, change.to.line + 1);
  // 在文档中附加链接文档的本地跨度
  linkedDocs(doc, function (doc) { return attachLocalSpans(doc, histChange, change.from.line, change.to.line + 1); }, true);
  // 返回历史变化对象
  return histChange
}

// 从历史数组中弹出所有选择事件，直到遇到一个变化事件
function clearSelectionEvents(array) {
  while (array.length) {
    var last = lst(array);
    // 如果最后一个元素包含范围信息，则弹出
    if (last.ranges) { array.pop(); }
    else { break }
  }
}

// 查找历史中最近的变化事件，弹出在其之后的选择事件
function lastChangeEvent(hist, force) {
  if (force) {
    // 弹出所有选择事件
    clearSelectionEvents(hist.done);
    // 返回最后一个变化事件
    return lst(hist.done)
  } else if (hist.done.length && !lst(hist.done).ranges) {
    // 如果历史中存在变化事件，则返回最后一个变化事件
    return lst(hist.done)
  } else if (hist.done.length > 1 && !hist.done[hist.done.length - 2].ranges) {
    // 如果历史中存在多个事件，并且倒数第二个事件不包含范围信息，则弹出最后一个事件并返回
    hist.done.pop();
    return lst(hist.done)
  }
}

// 将变化注册到历史中，合并在单个操作中或相邻的具有允许合并的起源（以“+”开头）的变化为单个事件
function addChangeToHistory(doc, change, selAfter, opId) {
  var hist = doc.history;
  // 清空未完成的历史记录
  hist.undone.length = 0;
  var time = +new Date, cur;
  var last;
}
    # 检查是否可以将当前变化合并到上一个事件中
    if ((hist.lastOp == opId ||  # 如果上一个操作和当前操作相同
         hist.lastOrigin == change.origin && change.origin &&  # 或者上一个操作的来源和当前操作的来源相同且不为空
         ((change.origin.charAt(0) == "+" && hist.lastModTime > time - (doc.cm ? doc.cm.options.historyEventDelay : 500)) ||  # 且当前操作是插入操作且上一个操作距离当前时间不超过指定延迟
          change.origin.charAt(0) == "*")) &&  # 或者当前操作是特殊操作
        (cur = lastChangeEvent(hist, hist.lastOp == opId))) {  # 并且当前事件是上一个操作的最后一个事件
      # 将当前变化合并到上一个事件中
      last = lst(cur.changes);
      if (cmp(change.from, change.to) == 0 && cmp(change.from, last.to) == 0) {  # 如果变化的起始和结束位置与上一个事件相同
        # 对于简单的插入情况进行优化处理，不需要为每个字符输入添加新的变化集
        last.to = changeEnd(change);
      } else {
        # 添加新的子事件
        cur.changes.push(historyChangeFromChange(doc, change));
      }
    } else {
      # 无法合并，开始一个新的事件
      var before = lst(hist.done);
      if (!before || !before.ranges)
        { pushSelectionToHistory(doc.sel, hist.done); }  # 如果之前没有事件或者没有范围，则将当前选择推入历史记录
      cur = {changes: [historyChangeFromChange(doc, change)],  # 创建新的事件对象
             generation: hist.generation};
      hist.done.push(cur);  # 将新事件推入历史记录
      while (hist.done.length > hist.undoDepth) {  # 如果历史记录长度超过了指定的撤销深度
        hist.done.shift();  # 移除最早的事件
        if (!hist.done[0].ranges) { hist.done.shift(); }  # 如果移除的事件没有范围，则再次移除
      }
    }
    hist.done.push(selAfter);  # 将当前选择推入历史记录
    hist.generation = ++hist.maxGeneration;  # 更新历史记录的最大代数
    hist.lastModTime = hist.lastSelTime = time;  # 更新最后修改时间和最后选择时间
    hist.lastOp = hist.lastSelOp = opId;  # 更新最后操作和最后选择的操作
    hist.lastOrigin = hist.lastSelOrigin = change.origin;  # 更新最后操作的来源和最后选择的操作的来源

    if (!last) { signal(doc, "historyAdded"); }  # 如果没有上一个事件，则触发"historyAdded"信号
  }

  # 检查选择事件是否可以合并
  function selectionEventCanBeMerged(doc, origin, prev, sel) {
    var ch = origin.charAt(0);
    # 如果字符为 * 或者 +，并且前一个选择范围的长度等于当前选择范围的长度，并且前一个选择范围是否有选中内容与当前选择范围是否有选中内容相同，并且当前时间与上一次选择的时间间隔小于等于500毫秒（如果有 CodeMirror 实例，则使用其选项中的 historyEventDelay，否则默认为500毫秒）
    return ch == "*" ||
      ch == "+" &&
      prev.ranges.length == sel.ranges.length &&
      prev.somethingSelected() == sel.somethingSelected() &&
      new Date - doc.history.lastSelTime <= (doc.cm ? doc.cm.options.historyEventDelay : 500)
  }

  // 每当选择发生变化时调用，将新选择设置为历史记录中的待处理选择，并在选择范围数量、是否为空或时间上有显著差异时，将旧的待处理选择推入“done”数组中。
  function addSelectionToHistory(doc, sel, opId, options) {
    var hist = doc.history, origin = options && options.origin;

    // 当前操作与上一个选择操作相同时，或者当前操作的来源与上一个选择操作的来源相同时，并且上一个操作的修改时间与上一个选择的时间相同时，或者选择事件可以合并时，将最后一个已完成的选择替换为当前选择
    if (opId == hist.lastSelOp ||
        (origin && hist.lastSelOrigin == origin &&
         (hist.lastModTime == hist.lastSelTime && hist.lastOrigin == origin ||
          selectionEventCanBeMerged(doc, origin, lst(hist.done), sel))))
      { hist.done[hist.done.length - 1] = sel; }
    else
      { pushSelectionToHistory(sel, hist.done); }

    hist.lastSelTime = +new Date;
    hist.lastSelOrigin = origin;
    hist.lastSelOp = opId;
    if (options && options.clearRedo !== false)
      { clearSelectionEvents(hist.undone); }
  }

  // 将选择推入历史记录
  function pushSelectionToHistory(sel, dest) {
    var top = lst(dest);
    if (!(top && top.ranges && top.equals(sel)))
      { dest.push(sel); }
  }

  // 用于在历史记录中存储标记跨度信息
  function attachLocalSpans(doc, change, from, to) {
    var existing = change["spans_" + doc.id], n = 0;
    doc.iter(Math.max(doc.first, from), Math.min(doc.first + doc.size, to), function (line) {
      if (line.markedSpans)
        { (existing || (existing = change["spans_" + doc.id] = {}))[n] = line.markedSpans; }
      ++n;
  // 当撤销/重做操作恢复包含标记范围的文本时，那些已经明确清除的标记范围不应该被恢复。
  function removeClearedSpans(spans) {
    if (!spans) { return null }
    var out;
    for (var i = 0; i < spans.length; ++i) {
      if (spans[i].marker.explicitlyCleared) { if (!out) { out = spans.slice(0, i); } }
      else if (out) { out.push(spans[i]); }
    }
    return !out ? spans : out.length ? out : null
  }

  // 检索并过滤存储在更改事件中的旧标记范围。
  function getOldSpans(doc, change) {
    var found = change["spans_" + doc.id];
    if (!found) { return null }
    var nw = [];
    for (var i = 0; i < change.text.length; ++i)
      { nw.push(removeClearedSpans(found[i])); }
    return nw
  }

  // 用于撤销/重做历史中的更改。将现有标记范围的计算结果与历史中存在的标记范围集合合并（以便在删除周围的标记范围然后撤消时将标记范围带回来）。
  function mergeOldSpans(doc, change) {
    var old = getOldSpans(doc, change);
    var stretched = stretchSpansOverChange(doc, change);
    if (!old) { return stretched }
    if (!stretched) { return old }

    for (var i = 0; i < old.length; ++i) {
      var oldCur = old[i], stretchCur = stretched[i];
      if (oldCur && stretchCur) {
        spans: for (var j = 0; j < stretchCur.length; ++j) {
          var span = stretchCur[j];
          for (var k = 0; k < oldCur.length; ++k)
            { if (oldCur[k].marker == span.marker) { continue spans } }
          oldCur.push(span);
        }
      } else if (stretchCur) {
        old[i] = stretchCur;
      }
    }
    return old
  }

  // 用于在.getHistory中提供一个 JSON 安全对象，并在分离文档时将历史分为两部分
  function copyHistoryArray(events, newGroup, instantiateSel) {
    var copy = [];
  }
    // 遍历事件数组
    for (var i = 0; i < events.length; ++i) {
      // 获取当前事件
      var event = events[i];
      // 如果事件有范围
      if (event.ranges) {
        // 如果需要实例化选择，则将深度复制的事件添加到副本数组中，否则直接添加事件
        copy.push(instantiateSel ? Selection.prototype.deepCopy.call(event) : event);
        // 继续下一个事件
        continue
      }
      // 获取事件的改变和新改变数组
      var changes = event.changes, newChanges = [];
      // 将新改变数组添加到副本数组中
      copy.push({changes: newChanges});
      // 遍历改变数组
      for (var j = 0; j < changes.length; ++j) {
        // 获取当前改变
        var change = changes[j], m = (void 0);
        // 将改变的起始位置、结束位置和文本添加到新改变数组中
        newChanges.push({from: change.from, to: change.to, text: change.text});
        // 如果有新组，则将对应的属性添加到新改变数组中，并删除原改变数组中的对应属性
        if (newGroup) { for (var prop in change) { if (m = prop.match(/^spans_(\d+)$/)) {
          if (indexOf(newGroup, Number(m[1])) > -1) {
            lst(newChanges)[prop] = change[prop];
            delete change[prop];
          }
        } } }
      }
    }
    // 返回副本数组
    return copy
  }

  // 给定的“scroll”参数表示在修改选择后是否应将新的光标位置滚动到视图中

  // 如果按住shift键或设置了extend标志，则扩展范围以包括给定位置（和可选的第二个位置）。
  // 否则，只返回给定位置之间的范围。
  // 用于光标移动等操作。
  function extendRange(range, head, other, extend) {
    // 如果需要扩展范围
    if (extend) {
      // 获取锚点位置
      var anchor = range.anchor;
      // 如果有第二个位置
      if (other) {
        // 比较头部位置和锚点位置的大小
        var posBefore = cmp(head, anchor) < 0;
        // 如果头部位置和锚点位置的大小不同于头部位置和第二个位置的大小
        if (posBefore != (cmp(other, anchor) < 0)) {
          // 交换头部位置和第二个位置
          anchor = head;
          head = other;
        } else if (posBefore != (cmp(head, other) < 0)) {
          // 如果头部位置和第二个位置的大小不同于头部位置和锚点位置的大小，则将头部位置设置为第二个位置
          head = other;
        }
      }
      // 返回新的范围
      return new Range(anchor, head)
    } else {
      // 返回给定位置之间的范围
      return new Range(other || head, head)
    }
  }

  // 扩展主要选择范围，丢弃其余部分。
  function extendSelection(doc, head, other, options, extend) {
    // 如果extend为null，则将其设置为doc.cm.display.shift或doc.extend
    if (extend == null) { extend = doc.cm && (doc.cm.display.shift || doc.extend); }
  // 设置新的选择，包括扩展选择和选项
  function setSelection(doc, newSel, options) {
    // 设置选择
    doc.setSelection(newSel, options);
  }

  // 扩展所有选择（pos 是一个具有与选择数量相等的长度的选择数组）
  function extendSelections(doc, heads, options) {
    var out = [];
    var extend = doc.cm && (doc.cm.display.shift || doc.extend);
    for (var i = 0; i < doc.sel.ranges.length; i++)
      { out[i] = extendRange(doc.sel.ranges[i], heads[i], null, extend); }
    var newSel = normalizeSelection(doc.cm, out, doc.sel.primIndex);
    setSelection(doc, newSel, options);
  }

  // 更新选择中的单个范围
  function replaceOneSelection(doc, i, range, options) {
    var ranges = doc.sel.ranges.slice(0);
    ranges[i] = range;
    setSelection(doc, normalizeSelection(doc.cm, ranges, doc.sel.primIndex), options);
  }

  // 将选择重置为单个范围
  function setSimpleSelection(doc, anchor, head, options) {
    setSelection(doc, simpleSelection(anchor, head), options);
  }

  // 在选择更新之前，让 beforeSelectionChange 处理程序影响选择更新
  function filterSelectionChange(doc, sel, options) {
    var obj = {
      ranges: sel.ranges,
      update: function(ranges) {
        this.ranges = [];
        for (var i = 0; i < ranges.length; i++)
          { this.ranges[i] = new Range(clipPos(doc, ranges[i].anchor),
                                     clipPos(doc, ranges[i].head)); }
      },
      origin: options && options.origin
    };
    signal(doc, "beforeSelectionChange", doc, obj);
    if (doc.cm) { signal(doc.cm, "beforeSelectionChange", doc.cm, obj); }
    if (obj.ranges != sel.ranges) { return normalizeSelection(doc.cm, obj.ranges, obj.ranges.length - 1) }
    else { return sel }
  }

  // 设置选择并替换历史记录
  function setSelectionReplaceHistory(doc, sel, options) {
    var done = doc.history.done, last = lst(done);
    if (last && last.ranges) {
      done[done.length - 1] = sel;
      setSelectionNoUndo(doc, sel, options);
    }
  }
  // 如果条件为真，执行第一个代码块，否则执行第二个代码块
  } else {
    // 调用 setSelection 函数，设置新的选择
    setSelection(doc, sel, options);
  }

  // 设置新的选择
  function setSelection(doc, sel, options) {
    // 调用 setSelectionNoUndo 函数，设置新的选择，不记录撤销操作
    setSelectionNoUndo(doc, sel, options);
    // 将选择添加到历史记录中
    addSelectionToHistory(doc, doc.sel, doc.cm ? doc.cm.curOp.id : NaN, options);
  }

  // 设置新的选择，不记录撤销操作
  function setSelectionNoUndo(doc, sel, options) {
    // 如果文档有 "beforeSelectionChange" 事件处理程序，或者 CodeMirror 对象有 "beforeSelectionChange" 事件处理程序
    if (hasHandler(doc, "beforeSelectionChange") || doc.cm && hasHandler(doc.cm, "beforeSelectionChange"))
      // 调用 filterSelectionChange 函数，过滤选择变化
      { sel = filterSelectionChange(doc, sel, options); }

    // 设置偏移量
    var bias = options && options.bias ||
      (cmp(sel.primary().head, doc.sel.primary().head) < 0 ? -1 : 1);
    // 调用 skipAtomicInSelection 函数，跳过选择中的原子标记范围
    setSelectionInner(doc, skipAtomicInSelection(doc, sel, bias, true));

    // 如果选项中没有设置 scroll 为 false，并且文档有 CodeMirror 对象
    if (!(options && options.scroll === false) && doc.cm)
      // 确保光标可见
      { ensureCursorVisible(doc.cm); }
  }

  // 设置内部选择
  function setSelectionInner(doc, sel) {
    // 如果选择与文档的选择相等，直接返回
    if (sel.equals(doc.sel)) { return }

    // 设置文档的选择为新的选择
    doc.sel = sel;

    // 如果文档有 CodeMirror 对象
    if (doc.cm) {
      // 更新输入
      doc.cm.curOp.updateInput = 1;
      // 选择已更改
      doc.cm.curOp.selectionChanged = true;
      // 发出光标活动信号
      signalCursorActivity(doc.cm);
    }
    // 延迟发送 "cursorActivity" 信号
    signalLater(doc, "cursorActivity", doc);
  }

  // 验证选择是否没有部分选择任何原子标记范围
  function reCheckSelection(doc) {
    // 调用 setSelectionInner 函数，跳过选择中的原子标记范围
    setSelectionInner(doc, skipAtomicInSelection(doc, doc.sel, null, false));
  }

  // 返回一个选择，不会部分选择任何原子范围
  function skipAtomicInSelection(doc, sel, bias, mayClear) {
    var out;
    for (var i = 0; i < sel.ranges.length; i++) {
      var range = sel.ranges[i];
      var old = sel.ranges.length == doc.sel.ranges.length && doc.sel.ranges[i];
      var newAnchor = skipAtomic(doc, range.anchor, old && old.anchor, bias, mayClear);
      var newHead = skipAtomic(doc, range.head, old && old.head, bias, mayClear);
      if (out || newAnchor != range.anchor || newHead != range.head) {
        if (!out) { out = sel.ranges.slice(0, i); }
        out[i] = new Range(newAnchor, newHead);
      }
    }
  }
  # 如果out为真，则调用normalizeSelection函数，传入doc.cm, out, sel.primIndex参数，否则返回sel
  return out ? normalizeSelection(doc.cm, out, sel.primIndex) : sel
}

# 跳过原子范围内的位置
function skipAtomicInner(doc, pos, oldPos, dir, mayClear) {
  # 获取指定行的内容
  var line = getLine(doc, pos.line);
  # 如果该行有标记范围
  if (line.markedSpans) { for (var i = 0; i < line.markedSpans.length; ++i) {
    var sp = line.markedSpans[i], m = sp.marker;

    # 确定是否应防止将光标放置在原子标记的左侧/右侧
    # 历史上，这是使用inclusiveLeft/Right选项确定的，但现在控制它的新方法是使用selectLeft/Right
    var preventCursorLeft = ("selectLeft" in m) ? !m.selectLeft : m.inclusiveLeft;
    var preventCursorRight = ("selectRight" in m) ? !m.selectRight : m.inclusiveRight;

    if ((sp.from == null || (preventCursorLeft ? sp.from <= pos.ch : sp.from < pos.ch)) &&
        (sp.to == null || (preventCursorRight ? sp.to >= pos.ch : sp.to > pos.ch))) {
      if (mayClear) {
        signal(m, "beforeCursorEnter");
        if (m.explicitlyCleared) {
          if (!line.markedSpans) { break }
          else {--i; continue}
        }
      }
      if (!m.atomic) { continue }

      if (oldPos) {
        var near = m.find(dir < 0 ? 1 : -1), diff = (void 0);
        if (dir < 0 ? preventCursorRight : preventCursorLeft)
          { near = movePos(doc, near, -dir, near && near.line == pos.line ? line : null); }
        if (near && near.line == pos.line && (diff = cmp(near, oldPos)) && (dir < 0 ? diff < 0 : diff > 0))
          { return skipAtomicInner(doc, near, pos, dir, mayClear) }
      }

      var far = m.find(dir < 0 ? -1 : 1);
      if (dir < 0 ? preventCursorLeft : preventCursorRight)
        { far = movePos(doc, far, dir, far.line == pos.line ? line : null); }
      return far ? skipAtomicInner(doc, far, pos, dir, mayClear) : null
    }
  } }
  return pos
}

# 确保给定位置不在原子范围内
function skipAtomic(doc, pos, oldPos, bias, mayClear) {
    # 如果 bias 为真，则将 dir 设置为 1，否则保持原值
    var dir = bias || 1;
    # 在指定方向上跳过原子内部的位置，如果找到则返回，否则继续尝试其他情况
    var found = skipAtomicInner(doc, pos, oldPos, dir, mayClear) ||
        (!mayClear && skipAtomicInner(doc, pos, oldPos, dir, true)) ||
        skipAtomicInner(doc, pos, oldPos, -dir, mayClear) ||
        (!mayClear && skipAtomicInner(doc, pos, oldPos, -dir, true));
    # 如果未找到，则将文档标记为不可编辑，并返回第一个位置
    if (!found) {
      doc.cantEdit = true;
      return Pos(doc.first, 0)
    }
    # 返回找到的位置
    return found
  }

  # 移动位置
  function movePos(doc, pos, dir, line) {
    # 如果方向为负且位置在行首，则返回上一行的位置
    if (dir < 0 && pos.ch == 0) {
      if (pos.line > doc.first) { return clipPos(doc, Pos(pos.line - 1)) }
      else { return null }
    } 
    # 如果方向为正且位置在行尾，则返回下一行的位置
    else if (dir > 0 && pos.ch == (line || getLine(doc, pos.line)).text.length) {
      if (pos.line < doc.first + doc.size - 1) { return Pos(pos.line + 1, 0) }
      else { return null }
    } 
    # 否则根据方向移动位置
    else {
      return new Pos(pos.line, pos.ch + dir)
    }
  }

  # 选择全部文本
  function selectAll(cm) {
    cm.setSelection(Pos(cm.firstLine(), 0), Pos(cm.lastLine()), sel_dontScroll);
  }

  # 更新

  # 允许“beforeChange”事件处理程序影响更改
  function filterChange(doc, change, update) {
    # 创建一个对象，包含更改的相关信息，并提供取消更改的方法
    var obj = {
      canceled: false,
      from: change.from,
      to: change.to,
      text: change.text,
      origin: change.origin,
      cancel: function () { return obj.canceled = true; }
    };
    # 如果允许更新，则提供更新更改的方法
    if (update) { obj.update = function (from, to, text, origin) {
      if (from) { obj.from = clipPos(doc, from); }
      if (to) { obj.to = clipPos(doc, to); }
      if (text) { obj.text = text; }
      if (origin !== undefined) { obj.origin = origin; }
    }; }
    # 触发“beforeChange”事件，并传递相关信息
    signal(doc, "beforeChange", doc, obj);
    if (doc.cm) { signal(doc.cm, "beforeChange", doc.cm, obj); }

    # 如果取消了更改，则将编辑操作标记为 2
    if (obj.canceled) {
      if (doc.cm) { doc.cm.curOp.updateInput = 2; }
      return null
    }
  // 返回一个包含指定属性的对象
  return {from: obj.from, to: obj.to, text: obj.text, origin: obj.origin}
}

// 对文档进行更改，并将其添加到文档的历史记录中，并传播到所有链接的文档
function makeChange(doc, change, ignoreReadOnly) {
  if (doc.cm) {
    if (!doc.cm.curOp) { return operation(doc.cm, makeChange)(doc, change, ignoreReadOnly) }
    if (doc.cm.state.suppressEdits) { return }
  }

  if (hasHandler(doc, "beforeChange") || doc.cm && hasHandler(doc.cm, "beforeChange")) {
    change = filterChange(doc, change, true);
    if (!change) { return }
  }

  // 根据其范围内的只读跨度的存在可能拆分或抑制更新
  var split = sawReadOnlySpans && !ignoreReadOnly && removeReadOnlyRanges(doc, change.from, change.to);
  if (split) {
    for (var i = split.length - 1; i >= 0; --i)
      { makeChangeInner(doc, {from: split[i].from, to: split[i].to, text: i ? [""] : change.text, origin: change.origin}); }
  } else {
    makeChangeInner(doc, change);
  }
}

function makeChangeInner(doc, change) {
  if (change.text.length == 1 && change.text[0] == "" && cmp(change.from, change.to) == 0) { return }
  var selAfter = computeSelAfterChange(doc, change);
  addChangeToHistory(doc, change, selAfter, doc.cm ? doc.cm.curOp.id : NaN);

  makeChangeSingleDoc(doc, change, selAfter, stretchSpansOverChange(doc, change));
  var rebased = [];

  linkedDocs(doc, function (doc, sharedHist) {
    if (!sharedHist && indexOf(rebased, doc.history) == -1) {
      rebaseHist(doc.history, change);
      rebased.push(doc.history);
    }
    makeChangeSingleDoc(doc, change, null, stretchSpansOverChange(doc, change));
  });
}

// 从文档的历史记录中恢复存储的更改
function makeChangeFromHistory(doc, type, allowSelectionOnly) {
  var suppress = doc.cm && doc.cm.state.suppressEdits;
  if (suppress && !allowSelectionOnly) { return }
}
    // 获取文档的历史记录
    var hist = doc.history, event, selAfter = doc.sel;
    // 根据撤销/重做类型选择不同的历史记录作为源和目标
    var source = type == "undo" ? hist.done : hist.undone, dest = type == "undo" ? hist.undone : hist.done;

    // 验证是否存在可用的事件，以防止 ctrl-z 无谓地清除选择事件
    var i = 0;
    for (; i < source.length; i++) {
      event = source[i];
      // 如果允许仅选择事件，并且事件包含范围并且不等于当前选择，则跳出循环
      if (allowSelectionOnly ? event.ranges && !event.equals(doc.sel) : !event.ranges)
        { break }
    }
    // 如果遍历完所有事件都没有符合条件的，则直接返回
    if (i == source.length) { return }
    // 重置历史记录的最后操作来源和最后选择来源
    hist.lastOrigin = hist.lastSelOrigin = null;

    // 循环处理源历史记录中的事件
    for (;;) {
      event = source.pop();
      // 如果事件包含范围，则将其添加到目标历史记录中
      if (event.ranges) {
        pushSelectionToHistory(event, dest);
        // 如果仅允许选择事件，并且事件不等于当前选择，则设置文档的选择并返回
        if (allowSelectionOnly && !event.equals(doc.sel)) {
          setSelection(doc, event, {clearRedo: false});
          return
        }
        // 更新选择后的状态
        selAfter = event;
      } else if (suppress) {
        // 如果需要抑制事件，则将事件重新添加到源历史记录中并返回
        source.push(event);
        return
      } else { break }
    }

    // 构建一个反向的变更对象，添加到相反的历史记录堆栈中（撤销时添加到重做历史记录中，反之亦然）
    var antiChanges = [];
    pushSelectionToHistory(selAfter, dest);
    dest.push({changes: antiChanges, generation: hist.generation});
    // 更新历史记录的代数
    hist.generation = event.generation || ++hist.maxGeneration;

    // 检查是否存在 beforeChange 处理程序，或者文档编辑器是否存在 beforeChange 处理程序
    var filter = hasHandler(doc, "beforeChange") || doc.cm && hasHandler(doc.cm, "beforeChange");
    // 定义一个循环函数，参数为索引 i
    var loop = function ( i ) {
      // 获取事件中的变化对象
      var change = event.changes[i];
      // 将变化对象的 origin 属性设置为 type
      change.origin = type;
      // 如果存在过滤器并且过滤器不通过，则清空 source 数组并返回空对象
      if (filter && !filterChange(doc, change, false)) {
        source.length = 0;
        return {}
      }

      // 将变化对象转换为历史变化对象并添加到 antiChanges 数组中
      antiChanges.push(historyChangeFromChange(doc, change));

      // 计算变化后的位置
      var after = i ? computeSelAfterChange(doc, change) : lst(source);
      // 对单个文档进行变化操作
      makeChangeSingleDoc(doc, change, after, mergeOldSpans(doc, change));
      // 如果是第一个变化并且存在代码镜像，则滚动到变化的位置
      if (!i && doc.cm) { doc.cm.scrollIntoView({from: change.from, to: changeEnd(change)}); }
      // 重新基于变化对象进行操作
      var rebased = [];

      // 传播到链接的文档
      linkedDocs(doc, function (doc, sharedHist) {
        if (!sharedHist && indexOf(rebased, doc.history) == -1) {
          rebaseHist(doc.history, change);
          rebased.push(doc.history);
        }
        makeChangeSingleDoc(doc, change, null, mergeOldSpans(doc, change));
      });
    };

    // 从最后一个变化对象开始循环
    for (var i$1 = event.changes.length - 1; i$1 >= 0; --i$1) {
      // 调用循环函数，并将返回值赋给 returned
      var returned = loop( i$1 );

      // 如果返回值存在，则返回其值
      if ( returned ) return returned.v;
    }
  }

  // 子视图在父文档中添加或删除文本时，需要调整行号
  function shiftDoc(doc, distance) {
    // 如果距离为 0，则直接返回
    if (distance == 0) { return }
    // 调整文档的第一行号
    doc.first += distance;
    // 调整选择对象的位置
    doc.sel = new Selection(map(doc.sel.ranges, function (range) { return new Range(
      Pos(range.anchor.line + distance, range.anchor.ch),
      Pos(range.head.line + distance, range.head.ch)
    ); }), doc.sel.primIndex);
    // 如果存在代码镜像，则注册变化
    if (doc.cm) {
      regChange(doc.cm, doc.first, doc.first - distance, distance);
      for (var d = doc.cm.display, l = d.viewFrom; l < d.viewTo; l++)
        { regLineChange(doc.cm, l, "gutter"); }
    }
  }

  // 更低级别的变化函数，仅处理单个文档（不包括链接的文档）
  function makeChangeSingleDoc(doc, change, selAfter, spans) {
    // 如果存在代码镜像并且当前操作为空，则执行操作函数
    if (doc.cm && !doc.cm.curOp)
      { return operation(doc.cm, makeChangeSingleDoc)(doc, change, selAfter, spans) }
    # 如果变化的结束行小于文档的第一行，则移动文档
    if (change.to.line < doc.first) {
      shiftDoc(doc, change.text.length - 1 - (change.to.line - change.from.line));
      return
    }
    # 如果变化的起始行大于文档的最后一行，则返回
    if (change.from.line > doc.lastLine()) { return }

    // 将变化限制在文档的大小范围内
    if (change.from.line < doc.first) {
      var shift = change.text.length - 1 - (doc.first - change.from.line);
      shiftDoc(doc, shift);
      change = {from: Pos(doc.first, 0), to: Pos(change.to.line + shift, change.to.ch),
                text: [lst(change.text)], origin: change.origin};
    }
    var last = doc.lastLine();
    if (change.to.line > last) {
      change = {from: change.from, to: Pos(last, getLine(doc, last).text.length),
                text: [change.text[0]], origin: change.origin};
    }

    # 获取变化范围内的内容
    change.removed = getBetween(doc, change.from, change.to);

    # 如果没有选择内容，则计算变化后的选择内容
    if (!selAfter) { selAfter = computeSelAfterChange(doc, change); }
    # 如果文档有编辑器，则在编辑器中进行单个文档的变化
    if (doc.cm) { makeChangeSingleDocInEditor(doc.cm, change, spans); }
    else { updateDoc(doc, change, spans); }
    # 设置选择内容，不记录撤销操作
    setSelectionNoUndo(doc, selAfter, sel_dontScroll);

    # 如果文档不能编辑且跳过原子操作，则将文档的cantEdit属性设置为false
    if (doc.cantEdit && skipAtomic(doc, Pos(doc.firstLine(), 0)))
      { doc.cantEdit = false; }
  }

  // 处理文档对编辑器的变化交互
  function makeChangeSingleDocInEditor(cm, change, spans) {
    var doc = cm.doc, display = cm.display, from = change.from, to = change.to;

    var recomputeMaxLength = false, checkWidthStart = from.line;
    # 如果不换行，则检查宽度的起始行为可视行的行号
    if (!cm.options.lineWrapping) {
      checkWidthStart = lineNo(visualLine(getLine(doc, from.line)));
      doc.iter(checkWidthStart, to.line + 1, function (line) {
        if (line == display.maxLine) {
          recomputeMaxLength = true;
          return true
        }
      });
    }

    # 如果选择内容包含变化的起始和结束位置，则发出光标活动信号
    if (doc.sel.contains(change.from, change.to) > -1)
      { signalCursorActivity(cm); }

    # 更新文档，包括变化、标记和估计高度
    updateDoc(doc, change, spans, estimateHeight(cm));
    # 如果编辑器选项中没有启用行包裹
    if (!cm.options.lineWrapping) {
      # 从指定行开始迭代文档，检查每行的长度
      doc.iter(checkWidthStart, from.line + change.text.length, function (line) {
        # 获取当前行的长度
        var len = lineLength(line);
        # 如果当前行长度超过了显示区域的最大行长度
        if (len > display.maxLineLength) {
          # 更新最大行的信息
          display.maxLine = line;
          display.maxLineLength = len;
          display.maxLineChanged = true;
          recomputeMaxLength = false;
        }
      });
      # 如果需要重新计算最大行的长度
      if (recomputeMaxLength) { cm.curOp.updateMaxLine = true; }
    }

    # 退回到指定行的前沿
    retreatFrontier(doc, from.line);
    # 启动工作线程
    startWorker(cm, 400);

    # 计算变化的行数
    var lendiff = change.text.length - (to.line - from.line) - 1;
    # 记录这些行的变化，以便更新显示
    if (change.full)
      { regChange(cm); }
    else if (from.line == to.line && change.text.length == 1 && !isWholeLineUpdate(cm.doc, change))
      { regLineChange(cm, from.line, "text"); }
    else
      { regChange(cm, from.line, to.line + 1, lendiff); }

    # 检查是否有变化处理程序
    var changesHandler = hasHandler(cm, "changes"), changeHandler = hasHandler(cm, "change");
    if (changeHandler || changesHandler) {
      # 创建变化对象
      var obj = {
        from: from, to: to,
        text: change.text,
        removed: change.removed,
        origin: change.origin
      };
      # 如果有变化处理程序，则延迟发送变化事件
      if (changeHandler) { signalLater(cm, "change", cm, obj); }
      # 如果有变化处理程序，则将变化对象推入当前操作的变化对象数组中
      if (changesHandler) { (cm.curOp.changeObjs || (cm.curOp.changeObjs = [])).push(obj); }
    }
    # 清空上下文菜单的选择
    cm.display.selForContextMenu = null;
  }

  # 替换指定范围的文本
  function replaceRange(doc, code, from, to, origin) {
    var assign;

    # 如果没有指定结束位置，则结束位置为起始位置
    if (!to) { to = from; }
    # 如果结束位置在起始位置之前，则交换起始位置和结束位置
    if (cmp(to, from) < 0) { (assign = [to, from], from = assign[0], to = assign[1]); }
    # 如果传入的文本是字符串，则将其拆分为行数组
    if (typeof code == "string") { code = doc.splitLines(code); }
    # 进行文本替换
    makeChange(doc, {from: from, to: to, text: code, origin: origin});
  }

  # 重新基于/重置历史记录以处理外部变化
  function rebaseHistSelSingle(pos, from, to, diff) {
    # 如果结束位置在指定位置之前，则更新位置信息
    if (to < pos.line) {
      pos.line += diff;
    } else if (from < pos.line) {
      pos.line = from;
      pos.ch = 0;
  // 结束当前函数的定义
  }
}

// 尝试重新基于文档的变化来重新定位历史事件数组。如果变化涉及与事件相同的行，则丢弃事件及其后的所有内容。如果变化在事件之前，则更新事件的位置。使用写时复制方案来处理位置，以避免在每次重新定位时重新分配它们，但也避免共享位置对象被不安全地更新的问题。
function rebaseHistArray(array, from, to, diff) {
  // 遍历历史事件数组
  for (var i = 0; i < array.length; ++i) {
    // 获取当前历史事件
    var sub = array[i], ok = true;
    // 如果历史事件包含范围
    if (sub.ranges) {
      // 如果历史事件未被复制，则进行深拷贝
      if (!sub.copied) { sub = array[i] = sub.deepCopy(); sub.copied = true; }
      // 遍历范围数组，更新位置
      for (var j = 0; j < sub.ranges.length; j++) {
        rebaseHistSelSingle(sub.ranges[j].anchor, from, to, diff);
        rebaseHistSelSingle(sub.ranges[j].head, from, to, diff);
      }
      continue
    }
    // 如果历史事件包含变化
    for (var j$1 = 0; j$1 < sub.changes.length; ++j$1) {
      var cur = sub.changes[j$1];
      // 如果变化在事件之前，则更新位置
      if (to < cur.from.line) {
        cur.from = Pos(cur.from.line + diff, cur.from.ch);
        cur.to = Pos(cur.to.line + diff, cur.to.ch);
      } else if (from <= cur.to.line) {
        ok = false;
        break
      }
    }
    // 如果不满足条件，则删除历史事件及其后的所有内容
    if (!ok) {
      array.splice(0, i + 1);
      i = 0;
    }
  }
}

// 重新定位历史记录，根据文档的变化
function rebaseHist(hist, change) {
  var from = change.from.line, to = change.to.line, diff = change.text.length - (to - from) - 1;
  // 重新定位已完成的历史事件数组
  rebaseHistArray(hist.done, from, to, diff);
  // 重新定位未完成的历史事件数组
  rebaseHistArray(hist.undone, from, to, diff);
}

// 应用变化到行的实用程序，通过句柄或编号，返回编号并可选择注册已更改的行
function changeLine(doc, handle, changeType, op) {
  var no = handle, line = handle;
  // 如果句柄是数字，则获取对应行
  if (typeof handle == "number") { line = getLine(doc, clipLine(doc, handle)); }
  else { no = lineNo(handle); }
  // 如果行号为空，则返回空
  if (no == null) { return null }
    // 如果操作函数返回 true 并且文档有 CodeMirror 对象，则注册行变化
    if (op(line, no) && doc.cm) { regLineChange(doc.cm, no, changeType); }
    // 返回行对象
    return line
  }

  // 文档被表示为一个 B 树，由叶子节点和包含行块的分支节点组成，每个分支节点最多有十个叶子节点或其他分支节点。顶部节点始终是一个分支节点，并且是文档对象本身（意味着它有额外的方法和属性）。
  //
  // 所有节点都有父链接。该树用于从行号到行对象的转换，以及从对象到行号的转换。它还按高度进行索引，并用于在高度和行对象之间进行转换，以及查找文档的总高度。
  //
  // 参见 http://marijnhaverbeke.nl/blog/codemirror-line-tree.html

  function LeafChunk(lines) {
    this.lines = lines;
    this.parent = null;
    var height = 0;
    for (var i = 0; i < lines.length; ++i) {
      lines[i].parent = this;
      height += lines[i].height;
    }
    this.height = height;
  }

  LeafChunk.prototype = {
    chunkSize: function() { return this.lines.length },

    // 在指定偏移处删除 n 行
    removeInner: function(at, n) {
      for (var i = at, e = at + n; i < e; ++i) {
        var line = this.lines[i];
        this.height -= line.height;
        cleanUpLine(line);
        signalLater(line, "delete");
      }
      this.lines.splice(at, n);
    },

    // 用于将一个小分支合并为单个叶子节点的辅助函数
    collapse: function(lines) {
      lines.push.apply(lines, this.lines);
    },

    // 在指定偏移处插入给定的行数组，将它们的高度计为给定的高度
    insertInner: function(at, lines, height) {
      this.height += height;
      this.lines = this.lines.slice(0, at).concat(lines).concat(this.lines.slice(at));
      for (var i = 0; i < lines.length; ++i) { lines[i].parent = this; }
    },

    // 用于遍历树的一部分
  // 定义一个名为 iterN 的方法，接受起始位置 at、数量 n 和操作 op 作为参数
  iterN: function(at, n, op) {
    // 从起始位置开始循环 n 次
    for (var e = at + n; at < e; ++at)
      // 如果操作 op 返回 true，则返回 true
      { if (op(this.lines[at])) { return true } }
  }
};

// 定义一个名为 BranchChunk 的构造函数，接受子节点数组作为参数
function BranchChunk(children) {
  // 初始化子节点数组
  this.children = children;
  // 初始化大小和高度
  var size = 0, height = 0;
  // 遍历子节点数组
  for (var i = 0; i < children.length; ++i) {
    var ch = children[i];
    // 计算总大小和高度
    size += ch.chunkSize(); height += ch.height;
    // 设置父节点为当前节点
    ch.parent = this;
  }
  this.size = size;
  this.height = height;
  this.parent = null;
}

// 设置 BranchChunk 的原型
BranchChunk.prototype = {
  // 定义一个名为 chunkSize 的方法，返回当前节点的大小
  chunkSize: function() { return this.size },

  // 定义一个名为 removeInner 的方法，接受起始位置 at 和数量 n 作为参数
  removeInner: function(at, n) {
    // 减去移除的数量
    this.size -= n;
    // 遍历子节点数组
    for (var i = 0; i < this.children.length; ++i) {
      var child = this.children[i], sz = child.chunkSize();
      // 如果起始位置小于当前子节点的大小
      if (at < sz) {
        var rm = Math.min(n, sz - at), oldHeight = child.height;
        // 递归调用 removeInner 方法
        child.removeInner(at, rm);
        // 更新高度
        this.height -= oldHeight - child.height;
        // 如果移除后子节点大小为 0，则从子节点数组中移除该子节点
        if (sz == rm) { this.children.splice(i--, 1); child.parent = null; }
        // 如果移除的数量为 0，则跳出循环
        if ((n -= rm) == 0) { break }
        at = 0;
      } else { at -= sz; }
    }
    // 如果结果小于 25 行，并且子节点数组长度大于 1 或第一个子节点不是 LeafChunk 类型，则将结果合并为单个 LeafChunk 节点
    if (this.size - n < 25 &&
        (this.children.length > 1 || !(this.children[0] instanceof LeafChunk))) {
      var lines = [];
      this.collapse(lines);
      this.children = [new LeafChunk(lines)];
      this.children[0].parent = this;
    }
  },

  // 定义一个名为 collapse 的方法，接受一个行数组作为参数
  collapse: function(lines) {
    // 遍历子节点数组，将每个子节点的行合并到行数组中
    for (var i = 0; i < this.children.length; ++i) { this.children[i].collapse(lines); }
  },
    // 在指定位置插入新的行，更新节点的大小和高度
    insertInner: function(at, lines, height) {
      this.size += lines.length;  // 更新节点的大小
      this.height += height;  // 更新节点的高度
      for (var i = 0; i < this.children.length; ++i) {  // 遍历子节点
        var child = this.children[i], sz = child.chunkSize();  // 获取子节点的大小
        if (at <= sz) {  // 如果插入位置小于等于子节点的大小
          child.insertInner(at, lines, height);  // 在子节点中插入新的行
          if (child.lines && child.lines.length > 50) {  // 如果子节点的行数大于50
            // 为了避免内存抖动，当子节点的行数很大时（例如大文件的第一个视图），不会进行切片
            // 取小的切片，按顺序取，因为顺序内存访问最快
            var remaining = child.lines.length % 25 + 25;  // 计算剩余行数
            for (var pos = remaining; pos < child.lines.length;) {  // 循环取小的切片
              var leaf = new LeafChunk(child.lines.slice(pos, pos += 25));  // 创建新的叶子节点
              child.height -= leaf.height;  // 更新子节点的高度
              this.children.splice(++i, 0, leaf);  // 在当前节点的子节点列表中插入新的叶子节点
              leaf.parent = this;  // 设置新的叶子节点的父节点
            }
            child.lines = child.lines.slice(0, remaining);  // 更新子节点的行数
            this.maybeSpill();  // 检查是否需要分裂
          }
          break  // 跳出循环
        }
        at -= sz;  // 更新插入位置
      }
    },

    // 当节点增长时，检查是否需要分裂
    maybeSpill: function() {
      if (this.children.length <= 10) { return }  // 如果子节点数量小于等于10，直接返回
      var me = this;  // 保存当前节点
      do {
        var spilled = me.children.splice(me.children.length - 5, 5);  // 从当前节点中取出最后5个子节点
        var sibling = new BranchChunk(spilled);  // 创建新的分支节点
        if (!me.parent) {  // 如果当前节点没有父节点
          var copy = new BranchChunk(me.children);  // 复制当前节点的子节点
          copy.parent = me;  // 设置复制节点的父节点
          me.children = [copy, sibling];  // 更新当前节点的子节点列表
          me = copy;  // 更新当前节点为复制节点
       } else {
          me.size -= sibling.size;  // 更新当前节点的大小
          me.height -= sibling.height;  // 更新当前节点的高度
          var myIndex = indexOf(me.parent.children, me);  // 获取当前节点在父节点中的索引
          me.parent.children.splice(myIndex + 1, 0, sibling);  // 在父节点中插入新的分支节点
        }
        sibling.parent = me.parent;  // 设置新的分支节点的父节点
      } while (me.children.length > 10)  // 当子节点数量大于10时继续循环
      me.parent.maybeSpill();  // 检查父节点是否需要分裂
    },
    // 定义一个名为 iterN 的方法，接受参数 at、n、op
    iterN: function(at, n, op) {
      // 遍历子元素数组
      for (var i = 0; i < this.children.length; ++i) {
        // 获取当前子元素和其大小
        var child = this.children[i], sz = child.chunkSize();
        // 如果 at 小于当前子元素大小
        if (at < sz) {
          // 计算实际使用的大小
          var used = Math.min(n, sz - at);
          // 递归调用 iterN 方法
          if (child.iterN(at, used, op)) { return true }
          // 更新剩余大小
          if ((n -= used) == 0) { break }
          // 重置 at
          at = 0;
        } else { at -= sz; }
      }
    }
  };

  // 定义 LineWidget 类，接受参数 doc、node、options
  // Line widgets 是显示在行上方或下方的块元素
  var LineWidget = function(doc, node, options) {
    // 如果有 options 参数，则遍历设置实例属性
    if (options) { for (var opt in options) { if (options.hasOwnProperty(opt))
      { this[opt] = options[opt]; } } }
    this.doc = doc;
    this.node = node;
  };

  // 为 LineWidget 原型添加 clear 方法
  LineWidget.prototype.clear = function () {
    // 获取相关变量
    var cm = this.doc.cm, ws = this.line.widgets, line = this.line, no = lineNo(line);
    // 如果行号为空或者没有 widgets，则返回
    if (no == null || !ws) { return }
    // 遍历 widgets 数组，移除当前实例
    for (var i = 0; i < ws.length; ++i) { if (ws[i] == this) { ws.splice(i--, 1); } }
    // 如果没有 widgets，则将 line.widgets 设置为 null
    if (!ws.length) { line.widgets = null; }
    // 计算 widget 的高度
    var height = widgetHeight(this);
    // 更新行高
    updateLineHeight(line, Math.max(0, line.height - height));
    // 如果存在编辑器实例
    if (cm) {
      // 在操作中运行以下代码
      runInOp(cm, function () {
        // 调整滚动条
        adjustScrollWhenAboveVisible(cm, line, -height);
        // 注册行变化事件
        regLineChange(cm, no, "widget");
      });
      // 延迟触发 lineWidgetCleared 事件
      signalLater(cm, "lineWidgetCleared", cm, this, no);
    }
  };

  // 为 LineWidget 原型添加 changed 方法
  LineWidget.prototype.changed = function () {
      var this$1 = this;

    // 获取相关变量
    var oldH = this.height, cm = this.doc.cm, line = this.line;
    this.height = null;
    // 计算高度差
    var diff = widgetHeight(this) - oldH;
    // 如果没有差异，则返回
    if (!diff) { return }
    // 如果行没有隐藏，则更新行高
    if (!lineIsHidden(this.doc, line)) { updateLineHeight(line, line.height + diff); }
    // 如果存在编辑器实例
    if (cm) {
      // 在操作中运行以下代码
      runInOp(cm, function () {
        // 强制更新
        cm.curOp.forceUpdate = true;
        // 调整滚动条
        adjustScrollWhenAboveVisible(cm, line, diff);
        // 延迟触发 lineWidgetChanged 事件
        signalLater(cm, "lineWidgetChanged", cm, this$1, lineNo(line));
      });
    }
  };
  // 为 LineWidget 添加事件混合器
  eventMixin(LineWidget);

  // 定义 adjustScrollWhenAboveVisible 方法，接受参数 cm、line、diff
    // 如果当前行的高度小于编辑器的滚动条位置，则将滚动条向上滚动
    if (heightAtLine(line) < ((cm.curOp && cm.curOp.scrollTop) || cm.doc.scrollTop))
      { addToScrollTop(cm, diff); }
  }

  // 向文档中添加一个行部件
  function addLineWidget(doc, handle, node, options) {
    // 创建一个新的行部件对象
    var widget = new LineWidget(doc, node, options);
    var cm = doc.cm;
    // 如果编辑器存在并且部件不需要水平滚动，则设置 alignWidgets 为 true
    if (cm && widget.noHScroll) { cm.display.alignWidgets = true; }
    // 在指定行上添加部件
    changeLine(doc, handle, "widget", function (line) {
      var widgets = line.widgets || (line.widgets = []);
      // 如果未指定插入位置，则将部件添加到数组末尾
      if (widget.insertAt == null) { widgets.push(widget); }
      // 否则，在指定位置插入部件
      else { widgets.splice(Math.min(widgets.length - 1, Math.max(0, widget.insertAt)), 0, widget); }
      widget.line = line;
      // 如果编辑器存在并且行不是隐藏的，则更新行高度并根据部件高度调整滚动条位置
      if (cm && !lineIsHidden(doc, line)) {
        var aboveVisible = heightAtLine(line) < doc.scrollTop;
        updateLineHeight(line, line.height + widgetHeight(widget));
        if (aboveVisible) { addToScrollTop(cm, widget.height); }
        cm.curOp.forceUpdate = true;
      }
      return true
    });
    // 如果编辑器存在，则延迟触发 lineWidgetAdded 事件
    if (cm) { signalLater(cm, "lineWidgetAdded", cm, widget, typeof handle == "number" ? handle : lineNo(handle)); }
    // 返回添加的部件对象
    return widget
  }

  // 文本标记

  // 使用 markText 和 setBookmark 方法创建的文本标记。TextMarker 是一个可以用于清除或查找文档中标记位置的句柄。行对象包含包含 {from, to, marker} 对象的数组 (markedSpans)，指向这样的标记对象，并指示该行上存在这样的标记。当标记跨越多行时，多行可能指向同一个标记。当标记延伸到行的起始/结束位置之外时，它们的 from/to 属性将为 null。标记具有指向它们当前触及的行的链接。

  // 折叠的标记具有唯一的 id，以便能够对它们进行排序，这对于在它们重叠时唯一确定外部标记是必要的（它们可以嵌套，但不能部分重叠）。
  var nextMarkerId = 0;

  // 文本标记构造函数
  var TextMarker = function(doc, type) {
    // 初始化行数组
    this.lines = [];
    // 设置当前对象的类型
    this.type = type;
    // 设置当前对象的文档
    this.doc = doc;
    // 设置当前对象的 ID，并递增下一个标记的 ID
    this.id = ++nextMarkerId;
    };
    
    // 清除标记
    TextMarker.prototype.clear = function () {
      // 如果已经明确清除，则直接返回
      if (this.explicitlyCleared) { return }
      // 获取当前编辑器实例，如果存在并且没有当前操作，则开始新的操作
      var cm = this.doc.cm, withOp = cm && !cm.curOp;
      if (withOp) { startOperation(cm); }
      // 如果有清除事件的处理函数，则查找标记并发送清除事件
      if (hasHandler(this, "clear")) {
        var found = this.find();
        if (found) { signalLater(this, "clear", found.from, found.to); }
      }
      var min = null, max = null;
      // 遍历标记所在的行
      for (var i = 0; i < this.lines.length; ++i) {
        var line = this.lines[i];
        // 获取标记的跨度
        var span = getMarkedSpanFor(line.markedSpans, this);
        // 如果编辑器存在并且标记没有折叠，则注册行变化事件
        if (cm && !this.collapsed) { regLineChange(cm, lineNo(line), "text"); }
        // 如果编辑器存在并且标记折叠了
        else if (cm) {
          if (span.to != null) { max = lineNo(line); }
          if (span.from != null) { min = lineNo(line); }
        }
        // 移除标记的跨度
        line.markedSpans = removeMarkedSpan(line.markedSpans, span);
        // 如果标记没有起始位置并且折叠了并且行没有隐藏并且编辑器存在，则更新行高
        if (span.from == null && this.collapsed && !lineIsHidden(this.doc, line) && cm)
          { updateLineHeight(line, textHeight(cm.display)); }
      }
      // 如果编辑器存在并且标记折叠了并且编辑器的选项中没有启用行换行
      if (cm && this.collapsed && !cm.options.lineWrapping) { for (var i$1 = 0; i$1 < this.lines.length; ++i$1) {
        var visual = visualLine(this.lines[i$1]), len = lineLength(visual);
        if (len > cm.display.maxLineLength) {
          cm.display.maxLine = visual;
          cm.display.maxLineLength = len;
          cm.display.maxLineChanged = true;
        }
      } }
      // 如果最小行号不为空并且编辑器存在并且标记折叠了，则注册变化事件
      if (min != null && cm && this.collapsed) { regChange(cm, min, max + 1); }
      // 清空标记所在的行
      this.lines.length = 0;
      // 标记已经明确清除
      this.explicitlyCleared = true;
      // 如果标记是原子的并且文档不可编辑，则设置文档可编辑并重新检查选择
      if (this.atomic && this.doc.cantEdit) {
        this.doc.cantEdit = false;
        if (cm) { reCheckSelection(cm.doc); }
      }
      // 如果编辑器存在，则发送标记清除事件
      if (cm) { signalLater(cm, "markerCleared", cm, this, min, max); }
      // 如果存在操作，则结束操作
      if (withOp) { endOperation(cm); }
    // 如果存在父元素，则清除父元素
    if (this.parent) { this.parent.clear(); }
  };

  // 在文档中查找标记的位置。默认情况下返回一个 {from, to} 对象。可以传递 side 来获取特定的位置 -- 0 (两侧), -1 (左侧), 或 1 (右侧)。当 lineObj 为 true 时，返回的 Pos 对象包含一个行对象，而不是行号（用于防止两次查找相同的行）。
  TextMarker.prototype.find = function (side, lineObj) {
    // 如果 side 为 null 并且类型为 "bookmark"，则 side 为 1
    if (side == null && this.type == "bookmark") { side = 1; }
    var from, to;
    for (var i = 0; i < this.lines.length; ++i) {
      var line = this.lines[i];
      var span = getMarkedSpanFor(line.markedSpans, this);
      if (span.from != null) {
        from = Pos(lineObj ? line : lineNo(line), span.from);
        if (side == -1) { return from }
      }
      if (span.to != null) {
        to = Pos(lineObj ? line : lineNo(line), span.to);
        if (side == 1) { return to }
      }
    }
    return from && {from: from, to: to}
  };

  // 表示标记的小部件已更改，并且应重新计算周围的布局。
  TextMarker.prototype.changed = function () {
      var this$1 = this;

    var pos = this.find(-1, true), widget = this, cm = this.doc.cm;
    if (!pos || !cm) { return }
    runInOp(cm, function () {
      var line = pos.line, lineN = lineNo(pos.line);
      var view = findViewForLine(cm, lineN);
      if (view) {
        clearLineMeasurementCacheFor(view);
        cm.curOp.selectionChanged = cm.curOp.forceUpdate = true;
      }
      cm.curOp.updateMaxLine = true;
      if (!lineIsHidden(widget.doc, line) && widget.height != null) {
        var oldHeight = widget.height;
        widget.height = null;
        var dHeight = widgetHeight(widget) - oldHeight;
        if (dHeight)
          { updateLineHeight(line, line.height + dHeight); }
      }
      signalLater(cm, "markerChanged", cm, this$1);
    });
  };

  TextMarker.prototype.attachLine = function (line) {
    // 如果没有行并且存在文档编辑器
    if (!this.lines.length && this.doc.cm) {
      // 获取当前操作
      var op = this.doc.cm.curOp;
      // 如果没有隐藏的标记或者当前标记不在隐藏标记列表中
      if (!op.maybeHiddenMarkers || indexOf(op.maybeHiddenMarkers, this) == -1)
        // 将当前标记添加到可能未隐藏的标记列表中
        { (op.maybeUnhiddenMarkers || (op.maybeUnhiddenMarkers = [])).push(this); }
    }
    // 将行添加到标记的行列表中
    this.lines.push(line);
  };

  // 从标记的行列表中移除行
  TextMarker.prototype.detachLine = function (line) {
    // 从行列表中移除指定的行
    this.lines.splice(indexOf(this.lines, line), 1);
    // 如果没有行并且存在文档编辑器
    if (!this.lines.length && this.doc.cm) {
      // 获取当前操作
      var op = this.doc.cm.curOp
      // 将当前标记添加到可能隐藏的标记列表中
      ;(op.maybeHiddenMarkers || (op.maybeHiddenMarkers = [])).push(this);
    }
  };
  // 事件混合
  eventMixin(TextMarker);

  // 创建一个标记，将其连接到正确的行，并且
  function markText(doc, from, to, options, type) {
    // 共享标记（跨链接文档）单独处理
    // （markTextShared将再次调用此函数，每个文档一次）
    if (options && options.shared) { return markTextShared(doc, from, to, options, type) }
    // 确保我们在一个操作中
    if (doc.cm && !doc.cm.curOp) { return operation(doc.cm, markText)(doc, from, to, options, type) }

    // 创建一个新的文本标记
    var marker = new TextMarker(doc, type), diff = cmp(from, to);
    // 如果存在选项，则将选项复制到标记中
    if (options) { copyObj(options, marker, false); }
    // 如果差值大于0或者等于0并且clearWhenEmpty不为false，则返回标记
    if (diff > 0 || diff == 0 && marker.clearWhenEmpty !== false)
      { return marker }
    // 如果标记有替换内容
    if (marker.replacedWith) {
      // 显示为小部件意味着折叠（小部件替换文本）
      marker.collapsed = true;
      // 创建小部件节点
      marker.widgetNode = eltP("span", [marker.replacedWith], "CodeMirror-widget");
      // 如果不处理鼠标事件，则将小部件节点设置为忽略事件
      if (!options.handleMouseEvents) { marker.widgetNode.setAttribute("cm-ignore-events", "true"); }
      // 如果选项中存在insertLeft，则将小部件节点设置为插入到左侧
      if (options.insertLeft) { marker.widgetNode.insertLeft = true; }
    }
    # 如果标记已经折叠
    if (marker.collapsed) {
      # 如果存在与插入的折叠标记部分重叠的现有折叠范围，或者起始行与结束行不同且存在与插入的折叠标记部分重叠的现有折叠范围，则抛出错误
      if (conflictingCollapsedRange(doc, from.line, from, to, marker) ||
          from.line != to.line && conflictingCollapsedRange(doc, to.line, from, to, marker))
        { throw new Error("Inserting collapsed marker partially overlapping an existing one") }
      # 查看折叠范围
      seeCollapsedSpans();
    }

    # 如果需要将操作添加到历史记录中
    if (marker.addToHistory)
      { addChangeToHistory(doc, {from: from, to: to, origin: "markText"}, doc.sel, NaN); }

    # 获取当前行和编辑器对象
    var curLine = from.line, cm = doc.cm, updateMaxLine;
    # 遍历从起始行到结束行的每一行
    doc.iter(curLine, to.line + 1, function (line) {
      # 如果编辑器对象存在、标记已折叠且不支持换行，并且当前行是显示的最大行
      if (cm && marker.collapsed && !cm.options.lineWrapping && visualLine(line) == cm.display.maxLine)
        { updateMaxLine = true; }
      # 如果标记已折叠且当前行不是起始行，则更新行高
      if (marker.collapsed && curLine != from.line) { updateLineHeight(line, 0); }
      # 添加标记范围到行
      addMarkedSpan(line, new MarkedSpan(marker,
                                         curLine == from.line ? from.ch : null,
                                         curLine == to.line ? to.ch : null));
      # 递增当前行
      ++curLine;
    });
    # lineIsHidden 依赖于标记范围的存在，因此需要第二次遍历
    if (marker.collapsed) { doc.iter(from.line, to.line + 1, function (line) {
      # 如果行被隐藏，则更新行高
      if (lineIsHidden(doc, line)) { updateLineHeight(line, 0); }
    }); }

    # 如果在输入时清除标记
    if (marker.clearOnEnter) { on(marker, "beforeCursorEnter", function () { return marker.clear(); }); }

    # 如果标记为只读
    if (marker.readOnly) {
      # 查看只读范围
      seeReadOnlySpans();
      # 如果存在已完成或未完成的历史记录，则清除历史记录
      if (doc.history.done.length || doc.history.undone.length)
        { doc.clearHistory(); }
    }
    # 如果标记已折叠
    if (marker.collapsed) {
      # 分配标记ID，并设置为原子操作
      marker.id = ++nextMarkerId;
      marker.atomic = true;
    }
    if (cm) {
      // 如果存在 CodeMirror 对象
      // 同步编辑器状态
      if (updateMaxLine) { cm.curOp.updateMaxLine = true; }  // 如果需要更新最大行数，则设置更新标志
      if (marker.collapsed)
        { regChange(cm, from.line, to.line + 1); }  // 如果标记已折叠，则注册编辑器变化
      else if (marker.className || marker.startStyle || marker.endStyle || marker.css ||
               marker.attributes || marker.title)
        { for (var i = from.line; i <= to.line; i++) { regLineChange(cm, i, "text"); } }  // 如果标记有类名、起始样式、结束样式、CSS、属性或标题，则注册行变化
      if (marker.atomic) { reCheckSelection(cm.doc); }  // 如果标记是原子性的，则重新检查选择
      signalLater(cm, "markerAdded", cm, marker);  // 发送标记添加事件
    }
    return marker  // 返回标记对象
  }

  // SHARED TEXTMARKERS

  // 共享文本标记
  // 一个共享标记跨越多个链接的文档。它被实现为控制多个普通标记的元标记对象。
  var SharedTextMarker = function(markers, primary) {
    this.markers = markers;  // 标记数组
    this.primary = primary;  // 主要标记
    for (var i = 0; i < markers.length; ++i)
      { markers[i].parent = this; }  // 遍历标记数组，设置父级为当前共享标记对象
  };

  SharedTextMarker.prototype.clear = function () {
    if (this.explicitlyCleared) { return }  // 如果已经明确清除，则返回
    this.explicitlyCleared = true;  // 设置为已明确清除
    for (var i = 0; i < this.markers.length; ++i)
      { this.markers[i].clear(); }  // 遍历标记数组，清除标记
    signalLater(this, "clear");  // 发送清除事件
  };

  SharedTextMarker.prototype.find = function (side, lineObj) {
    return this.primary.find(side, lineObj)  // 查找主要标记
  };
  eventMixin(SharedTextMarker);  // 事件混合

  function markTextShared(doc, from, to, options, type) {
    options = copyObj(options);  // 复制选项对象
    options.shared = false;  // 设置为非共享
    var markers = [markText(doc, from, to, options, type)], primary = markers[0];  // 创建标记数组，设置主要标记
    var widget = options.widgetNode;  // 获取小部件节点
    linkedDocs(doc, function (doc) {
      if (widget) { options.widgetNode = widget.cloneNode(true); }  // 如果存在小部件节点，则克隆节点
      markers.push(markText(doc, clipPos(doc, from), clipPos(doc, to), options, type));  // 添加标记
      for (var i = 0; i < doc.linked.length; ++i)
        { if (doc.linked[i].isParent) { return } }  // 遍历链接的文档，如果是父级文档，则返回
      primary = lst(markers);  // 设置主要标记为最后一个标记
    });
    return new SharedTextMarker(markers, primary)  // 返回共享文本标记对象
  }

  function findSharedMarkers(doc) {
  // 返回文档中指定位置范围内的所有标记
  return doc.findMarks(Pos(doc.first, 0), doc.clipPos(Pos(doc.lastLine())), function (m) { return m.parent; })
}

// 复制共享标记
function copySharedMarkers(doc, markers) {
  for (var i = 0; i < markers.length; i++) {
    var marker = markers[i], pos = marker.find();
    var mFrom = doc.clipPos(pos.from), mTo = doc.clipPos(pos.to);
    // 如果起始位置和结束位置不同，则创建子标记
    if (cmp(mFrom, mTo)) {
      var subMark = markText(doc, mFrom, mTo, marker.primary, marker.primary.type);
      marker.markers.push(subMark);
      subMark.parent = marker;
    }
  }
}

// 分离共享标记
function detachSharedMarkers(markers) {
  // 遍历所有标记
  var loop = function ( i ) {
    var marker = markers[i], linked = [marker.primary.doc];
    // 获取所有关联的文档
    linkedDocs(marker.primary.doc, function (d) { return linked.push(d); });
    for (var j = 0; j < marker.markers.length; j++) {
      var subMarker = marker.markers[j];
      // 如果子标记的文档不在关联文档列表中，则将其父标记设为null，并从列表中移除
      if (indexOf(linked, subMarker.doc) == -1) {
        subMarker.parent = null;
        marker.markers.splice(j--, 1);
      }
    }
  };

  for (var i = 0; i < markers.length; i++) loop( i );
}

// 初始化文档对象
var nextDocId = 0;
var Doc = function(text, mode, firstLine, lineSep, direction) {
  if (!(this instanceof Doc)) { return new Doc(text, mode, firstLine, lineSep, direction) }
  if (firstLine == null) { firstLine = 0; }

  // 创建文档对象
  BranchChunk.call(this, [new LeafChunk([new Line("", null)])]);
  this.first = firstLine;
  this.scrollTop = this.scrollLeft = 0;
  this.cantEdit = false;
  this.cleanGeneration = 1;
  this.modeFrontier = this.highlightFrontier = firstLine;
  var start = Pos(firstLine, 0);
  this.sel = simpleSelection(start);
  this.history = new History(null);
  this.id = ++nextDocId;
  this.modeOption = mode;
  this.lineSep = lineSep;
  this.direction = (direction == "rtl") ? "rtl" : "ltr";
  this.extend = false;

  // 如果传入的文本是字符串，则将其拆分成行数组
  if (typeof text == "string") { text = this.splitLines(text); }
  // 更新文档内容
  updateDoc(this, {from: start, to: start, text: text});
}
    # 设置选择区域，不滚动
    setSelection(this, simpleSelection(start), sel_dontScroll);
  };

  # 将 Doc 对象的原型设置为 BranchChunk 对象
  Doc.prototype = createObj(BranchChunk.prototype, {
    constructor: Doc,
    # 迭代文档。支持两种形式 -- 只有一个参数时，对文档中的每一行调用该函数。有三个参数时，迭代给定范围（第二个参数不包括在内）。
    iter: function(from, to, op) {
      if (op) { this.iterN(from - this.first, to - from, op); }
      else { this.iterN(this.first, this.first + this.size, from); }
    },

    # 非公开接口，用于添加和删除行
    insert: function(at, lines) {
      var height = 0;
      for (var i = 0; i < lines.length; ++i) { height += lines[i].height; }
      this.insertInner(at - this.first, lines, height);
    },
    remove: function(at, n) { this.removeInner(at - this.first, n); },

    # 从这里开始，这些方法是公共接口的一部分。大多数也可以从 CodeMirror (editor) 实例中使用。
    
    # 获取文档的值
    getValue: function(lineSep) {
      var lines = getLines(this, this.first, this.first + this.size);
      if (lineSep === false) { return lines }
      return lines.join(lineSep || this.lineSeparator())
    },
    # 设置文档的值
    setValue: docMethodOp(function(code) {
      var top = Pos(this.first, 0), last = this.first + this.size - 1;
      makeChange(this, {from: top, to: Pos(last, getLine(this, last).text.length),
                        text: this.splitLines(code), origin: "setValue", full: true}, true);
      if (this.cm) { scrollToCoords(this.cm, 0, 0); }
      setSelection(this, simpleSelection(top), sel_dontScroll);
    }),
    # 替换范围内的文本
    replaceRange: function(code, from, to, origin) {
      from = clipPos(this, from);
      to = to ? clipPos(this, to) : from;
      replaceRange(this, code, from, to, origin);
    },
    # 获取指定范围内的文本行
    getRange: function(from, to, lineSep) {
      # 获取指定范围内的文本行
      var lines = getBetween(this, clipPos(this, from), clipPos(this, to));
      # 如果不需要添加行分隔符，则直接返回文本行数组
      if (lineSep === false) { return lines }
      # 否则将文本行数组用指定的行分隔符连接起来并返回
      return lines.join(lineSep || this.lineSeparator())
    },

    # 获取指定行的文本内容
    getLine: function(line) {var l = this.getLineHandle(line); return l && l.text},

    # 获取指定行的句柄
    getLineHandle: function(line) {if (isLine(this, line)) { return getLine(this, line) }},
    
    # 获取指定行的行号
    getLineNumber: function(line) {return lineNo(line)},

    # 获取指定行的可视起始位置
    getLineHandleVisualStart: function(line) {
      # 如果指定的行是数字，则获取该行的句柄
      if (typeof line == "number") { line = getLine(this, line); }
      # 返回该行的可视起始位置
      return visualLine(line)
    },

    # 获取文档的行数
    lineCount: function() {return this.size},
    
    # 获取文档的第一行行号
    firstLine: function() {return this.first},
    
    # 获取文档的最后一行行号
    lastLine: function() {return this.first + this.size - 1},

    # 裁剪指定位置
    clipPos: function(pos) {return clipPos(this, pos)},

    # 获取光标位置
    getCursor: function(start) {
      # 获取主要选择范围
      var range = this.sel.primary(), pos;
      # 根据参数确定返回的光标位置
      if (start == null || start == "head") { pos = range.head; }
      else if (start == "anchor") { pos = range.anchor; }
      else if (start == "end" || start == "to" || start === false) { pos = range.to(); }
      else { pos = range.from(); }
      return pos
    },
    
    # 返回所有选择范围
    listSelections: function() { return this.sel.ranges },
    
    # 检查是否有选中内容
    somethingSelected: function() {return this.sel.somethingSelected()},

    # 设置光标位置
    setCursor: docMethodOp(function(line, ch, options) {
      setSimpleSelection(this, clipPos(this, typeof line == "number" ? Pos(line, ch || 0) : line), null, options);
    }),
    
    # 设置选择范围
    setSelection: docMethodOp(function(anchor, head, options) {
      setSimpleSelection(this, clipPos(this, anchor), clipPos(this, head || anchor), options);
    }),
    
    # 扩展选择范围
    extendSelection: docMethodOp(function(head, other, options) {
      extendSelection(this, clipPos(this, head), other && clipPos(this, other), options);
    }),
    
    # 扩展多个选择范围
    extendSelections: docMethodOp(function(heads, options) {
      extendSelections(this, clipPosArray(this, heads), options);
    }),
    extendSelectionsBy: docMethodOp(function(f, options) {
      // 通过函数 f 对选择范围的头部进行映射，得到新的头部数组
      var heads = map(this.sel.ranges, f);
      // 根据新的头部数组扩展选择范围
      extendSelections(this, clipPosArray(this, heads), options);
    }),
    setSelections: docMethodOp(function(ranges, primary, options) {
      // 如果选择范围为空，则直接返回
      if (!ranges.length) { return }
      var out = [];
      // 遍历选择范围数组，将每个范围的锚点和头部进行裁剪，然后存入 out 数组
      for (var i = 0; i < ranges.length; i++)
        { out[i] = new Range(clipPos(this, ranges[i].anchor),
                           clipPos(this, ranges[i].head)); }
      // 如果未指定主要选择范围，则默认为当前选择范围的主要选择
      if (primary == null) { primary = Math.min(ranges.length - 1, this.sel.primIndex); }
      // 设置选择范围
      setSelection(this, normalizeSelection(this.cm, out, primary), options);
    }),
    addSelection: docMethodOp(function(anchor, head, options) {
      // 复制当前选择范围数组
      var ranges = this.sel.ranges.slice(0);
      // 将新的选择范围添加到复制的选择范围数组中
      ranges.push(new Range(clipPos(this, anchor), clipPos(this, head || anchor)));
      // 设置选择范围
      setSelection(this, normalizeSelection(this.cm, ranges, ranges.length - 1), options);
    }),

    getSelection: function(lineSep) {
      // 获取当前选择范围数组
      var ranges = this.sel.ranges, lines;
      // 遍历选择范围数组，获取每个选择范围的文本内容，并拼接成一个数组
      for (var i = 0; i < ranges.length; i++) {
        var sel = getBetween(this, ranges[i].from(), ranges[i].to());
        lines = lines ? lines.concat(sel) : sel;
      }
      // 如果 lineSep 为 false，则直接返回拼接后的数组，否则以指定的分隔符拼接成字符串返回
      if (lineSep === false) { return lines }
      else { return lines.join(lineSep || this.lineSeparator()) }
    },
    getSelections: function(lineSep) {
      // 初始化空数组 parts
      var parts = [], ranges = this.sel.ranges;
      // 遍历选择范围数组，获取每个选择范围的文本内容，并根据 lineSep 拼接成字符串
      for (var i = 0; i < ranges.length; i++) {
        var sel = getBetween(this, ranges[i].from(), ranges[i].to());
        if (lineSep !== false) { sel = sel.join(lineSep || this.lineSeparator()); }
        parts[i] = sel;
      }
      // 返回拼接后的字符串数组
      return parts
    },
    replaceSelection: function(code, collapse, origin) {
      // 初始化空数组 dup
      var dup = [];
      // 遍历选择范围数组，将 code 添加到 dup 数组中
      for (var i = 0; i < this.sel.ranges.length; i++)
        { dup[i] = code; }
      // 调用 replaceSelections 方法，将 dup 数组作为参数传入
      this.replaceSelections(dup, collapse, origin || "+input");
    },
    // 替换选定文本的方法，接受代码、折叠标志和原始信息作为参数
    replaceSelections: docMethodOp(function(code, collapse, origin) {
      var changes = [], sel = this.sel;
      for (var i = 0; i < sel.ranges.length; i++) {
        var range = sel.ranges[i];
        changes[i] = {from: range.from(), to: range.to(), text: this.splitLines(code[i]), origin: origin};
      }
      var newSel = collapse && collapse != "end" && computeReplacedSel(this, changes, collapse);
      for (var i$1 = changes.length - 1; i$1 >= 0; i$1--)
        { makeChange(this, changes[i$1]); }
      if (newSel) { setSelectionReplaceHistory(this, newSel); }
      else if (this.cm) { ensureCursorVisible(this.cm); }
    }),
    // 撤销操作的方法
    undo: docMethodOp(function() {makeChangeFromHistory(this, "undo");}),
    // 重做操作的方法
    redo: docMethodOp(function() {makeChangeFromHistory(this, "redo");}),
    // 撤销选定文本的方法
    undoSelection: docMethodOp(function() {makeChangeFromHistory(this, "undo", true);}),
    // 重做选定文本的方法
    redoSelection: docMethodOp(function() {makeChangeFromHistory(this, "redo", true);}),
    // 设置扩展标志的方法
    setExtending: function(val) {this.extend = val;},
    // 获取扩展标志的方法
    getExtending: function() {return this.extend},
    // 获取历史记录大小的方法
    historySize: function() {
      var hist = this.history, done = 0, undone = 0;
      for (var i = 0; i < hist.done.length; i++) { if (!hist.done[i].ranges) { ++done; } }
      for (var i$1 = 0; i$1 < hist.undone.length; i$1++) { if (!hist.undone[i$1].ranges) { ++undone; } }
      return {undo: done, redo: undone}
    },
    // 清除历史记录的方法
    clearHistory: function() {
      var this$1 = this;

      this.history = new History(this.history.maxGeneration);
      linkedDocs(this, function (doc) { return doc.history = this$1.history; }, true);
    },
    // 标记为干净状态的方法
    markClean: function() {
      this.cleanGeneration = this.changeGeneration(true);
    },
    // 改变历史记录的代数的方法
    changeGeneration: function(forceSplit) {
      if (forceSplit)
        { this.history.lastOp = this.history.lastSelOp = this.history.lastOrigin = null; }
      return this.history.generation
    },
    // 检查编辑器是否干净，即当前的 generation 是否等于指定的 generation 或者默认的 cleanGeneration
    isClean: function (gen) {
      return this.history.generation == (gen || this.cleanGeneration)
    },

    // 获取编辑器的历史记录，包括已完成和未完成的操作
    getHistory: function() {
      return {done: copyHistoryArray(this.history.done),
              undone: copyHistoryArray(this.history.undone)}
    },

    // 设置编辑器的历史记录
    setHistory: function(histData) {
      // 创建一个新的历史记录对象
      var hist = this.history = new History(this.history.maxGeneration);
      // 复制已完成和未完成的操作到新的历史记录对象中
      hist.done = copyHistoryArray(histData.done.slice(0), null, true);
      hist.undone = copyHistoryArray(histData.undone.slice(0), null, true);
    },

    // 设置指定行号的 gutter 标记
    setGutterMarker: docMethodOp(function(line, gutterID, value) {
      return changeLine(this, line, "gutter", function (line) {
        var markers = line.gutterMarkers || (line.gutterMarkers = {});
        markers[gutterID] = value;
        if (!value && isEmpty(markers)) { line.gutterMarkers = null; }
        return true
      })
    }),

    // 清除指定 gutter 标记
    clearGutter: docMethodOp(function(gutterID) {
      var this$1 = this;

      // 遍历每一行，清除指定的 gutter 标记
      this.iter(function (line) {
        if (line.gutterMarkers && line.gutterMarkers[gutterID]) {
          changeLine(this$1, line, "gutter", function () {
            line.gutterMarkers[gutterID] = null;
            if (isEmpty(line.gutterMarkers)) { line.gutterMarkers = null; }
            return true
          });
        }
      });
    }),

    // 获取指定行号的信息，包括行号、内容、标记等
    lineInfo: function(line) {
      var n;
      if (typeof line == "number") {
        if (!isLine(this, line)) { return null }
        n = line;
        line = getLine(this, line);
        if (!line) { return null }
      } else {
        n = lineNo(line);
        if (n == null) { return null }
      }
      return {line: n, handle: line, text: line.text, gutterMarkers: line.gutterMarkers,
              textClass: line.textClass, bgClass: line.bgClass, wrapClass: line.wrapClass,
              widgets: line.widgets}
    },
    # 添加行的类
    addLineClass: docMethodOp(function(handle, where, cls) {
      # 根据位置和类名改变行的样式
      return changeLine(this, handle, where == "gutter" ? "gutter" : "class", function (line) {
        # 根据位置确定要改变的属性
        var prop = where == "text" ? "textClass"
                 : where == "background" ? "bgClass"
                 : where == "gutter" ? "gutterClass" : "wrapClass";
        # 如果属性为空，则设置为给定的类名
        if (!line[prop]) { line[prop] = cls; }
        # 如果属性不为空，且已经包含给定的类名，则返回 false
        else if (classTest(cls).test(line[prop])) { return false }
        # 如果属性不为空，且不包含给定的类名，则添加给定的类名
        else { line[prop] += " " + cls; }
        # 返回 true
        return true
      })
    }),
    # 移除行的类
    removeLineClass: docMethodOp(function(handle, where, cls) {
      # 根据位置和类名改变行的样式
      return changeLine(this, handle, where == "gutter" ? "gutter" : "class", function (line) {
        # 根据位置确定要改变的属性
        var prop = where == "text" ? "textClass"
                 : where == "background" ? "bgClass"
                 : where == "gutter" ? "gutterClass" : "wrapClass";
        # 获取当前属性值
        var cur = line[prop];
        # 如果当前属性值为空，则返回 false
        if (!cur) { return false }
        # 如果给定的类名为空，则将属性值设置为空
        else if (cls == null) { line[prop] = null; }
        # 如果给定的类名不为空
        else {
          # 查找属性值中是否包含给定的类名
          var found = cur.match(classTest(cls));
          # 如果没有找到，则返回 false
          if (!found) { return false }
          # 找到给定的类名，将其从属性值中移除
          var end = found.index + found[0].length;
          line[prop] = cur.slice(0, found.index) + (!found.index || end == cur.length ? "" : " ") + cur.slice(end) || null;
        }
        # 返回 true
        return true
      })
    }),

    # 添加行小部件
    addLineWidget: docMethodOp(function(handle, node, options) {
      # 调用函数添加行小部件
      return addLineWidget(this, handle, node, options)
    }),
    # 移除行小部件
    removeLineWidget: function(widget) { widget.clear(); },

    # 标记文本
    markText: function(from, to, options) {
      # 调用函数标记文本
      return markText(this, clipPos(this, from), clipPos(this, to), options, options && options.type || "range")
    },
    # 设置书签，将选项转换为真实选项
    setBookmark: function(pos, options) {
      var realOpts = {replacedWith: options && (options.nodeType == null ? options.widget : options),
                      insertLeft: options && options.insertLeft,
                      clearWhenEmpty: false, shared: options && options.shared,
                      handleMouseEvents: options && options.handleMouseEvents};
      # 限制位置在编辑器内
      pos = clipPos(this, pos);
      # 在指定位置标记文本
      return markText(this, pos, pos, realOpts, "bookmark")
    },
    # 查找指定位置的标记
    findMarksAt: function(pos) {
      # 限制位置在编辑器内
      pos = clipPos(this, pos);
      var markers = [], spans = getLine(this, pos.line).markedSpans;
      if (spans) { for (var i = 0; i < spans.length; ++i) {
        var span = spans[i];
        if ((span.from == null || span.from <= pos.ch) &&
            (span.to == null || span.to >= pos.ch))
          { markers.push(span.marker.parent || span.marker); }
      } }
      return markers
    },
    # 查找指定范围内的标记
    findMarks: function(from, to, filter) {
      # 限制位置在编辑器内
      from = clipPos(this, from); to = clipPos(this, to);
      var found = [], lineNo = from.line;
      # 遍历指定范围内的每一行
      this.iter(from.line, to.line + 1, function (line) {
        var spans = line.markedSpans;
        if (spans) { for (var i = 0; i < spans.length; i++) {
          var span = spans[i];
          # 检查标记是否在指定范围内
          if (!(span.to != null && lineNo == from.line && from.ch >= span.to ||
                span.from == null && lineNo != from.line ||
                span.from != null && lineNo == to.line && span.from >= to.ch) &&
              (!filter || filter(span.marker)))
            { found.push(span.marker.parent || span.marker); }
        } }
        ++lineNo;
      });
      return found
    },
    # 获取所有标记
    getAllMarks: function() {
      var markers = [];
      # 遍历每一行
      this.iter(function (line) {
        var sps = line.markedSpans;
        if (sps) { for (var i = 0; i < sps.length; ++i)
          { if (sps[i].from != null) { markers.push(sps[i].marker); } } }
      });
      return markers
    },
    // 根据偏移量计算位置
    posFromIndex: function(off) {
      // 初始化行号为第一行
      var ch, lineNo = this.first, sepSize = this.lineSeparator().length;
      // 遍历每一行
      this.iter(function (line) {
        // 计算当前行的长度加上换行符的长度
        var sz = line.text.length + sepSize;
        // 如果偏移量小于当前行的长度，则找到对应的位置并返回
        if (sz > off) { ch = off; return true }
        // 否则减去当前行的长度和换行符的长度，继续遍历下一行
        off -= sz;
        ++lineNo;
      });
      // 返回计算出的位置
      return clipPos(this, Pos(lineNo, ch))
    },
    // 根据位置计算索引
    indexFromPos: function (coords) {
      // 对位置进行裁剪
      coords = clipPos(this, coords);
      var index = coords.ch;
      // 如果行号小于第一行或者字符位置小于0，则返回0
      if (coords.line < this.first || coords.ch < 0) { return 0 }
      // 获取换行符的长度
      var sepSize = this.lineSeparator().length;
      // 遍历每一行
      this.iter(this.first, coords.line, function (line) { // iter aborts when callback returns a truthy value
        // 累加每一行的长度和换行符的长度
        index += line.text.length + sepSize;
      });
      // 返回计算出的索引
      return index
    },

    // 复制文档
    copy: function(copyHistory) {
      // 创建一个新的文档对象
      var doc = new Doc(getLines(this, this.first, this.first + this.size),
                        this.modeOption, this.first, this.lineSep, this.direction);
      // 复制滚动条位置和选择区域
      doc.scrollTop = this.scrollTop; doc.scrollLeft = this.scrollLeft;
      doc.sel = this.sel;
      doc.extend = false;
      // 如果需要复制历史记录
      if (copyHistory) {
        // 复制撤销深度和历史记录
        doc.history.undoDepth = this.history.undoDepth;
        doc.setHistory(this.getHistory());
      }
      // 返回复制后的文档对象
      return doc
    },

    // 创建关联文档
    linkedDoc: function(options) {
      // 如果没有传入选项，则初始化为空对象
      if (!options) { options = {}; }
      // 初始化起始和结束行号
      var from = this.first, to = this.first + this.size;
      // 如果传入了起始位置，并且大于起始行号，则更新起始行号
      if (options.from != null && options.from > from) { from = options.from; }
      // 如果传入了结束位置，并且小于结束行号，则更新结束行号
      if (options.to != null && options.to < to) { to = options.to; }
      // 创建一个新的文档对象
      var copy = new Doc(getLines(this, from, to), options.mode || this.modeOption, from, this.lineSep, this.direction);
      // 如果需要共享历史记录
      if (options.sharedHist) { copy.history = this.history
      ; }(this.linked || (this.linked = [])).push({doc: copy, sharedHist: options.sharedHist});
      // 更新关联文档的信息
      copy.linked = [{doc: this, isParent: true, sharedHist: options.sharedHist}];
      copySharedMarkers(copy, findSharedMarkers(this));
      // 返回关联的文档对象
      return copy
    },
    // unlinkDoc 方法用于取消文档之间的链接关系
    unlinkDoc: function(other) {
      // 如果传入的参数是 CodeMirror 对象，则获取其 doc 属性
      if (other instanceof CodeMirror) { other = other.doc; }
      // 如果当前文档已经和其他文档链接
      if (this.linked) { for (var i = 0; i < this.linked.length; ++i) {
        var link = this.linked[i];
        // 如果找到了要取消链接的文档
        if (link.doc != other) { continue }
        // 从 linked 数组中移除当前文档和其他文档的链接
        this.linked.splice(i, 1);
        // 调用其他文档的 unlinkDoc 方法，取消与当前文档的链接
        other.unlinkDoc(this);
        // 移除共享标记
        detachSharedMarkers(findSharedMarkers(this));
        break
      } }
      // 如果当前文档和其他文档的历史记录是共享的，则分割它们
      if (other.history == this.history) {
        var splitIds = [other.id];
        // 遍历其他文档的链接文档，将其 id 添加到 splitIds 数组中
        linkedDocs(other, function (doc) { return splitIds.push(doc.id); }, true);
        // 创建新的历史记录对象，将当前文档的历史记录拆分后的记录赋值给其他文档
        other.history = new History(null);
        other.history.done = copyHistoryArray(this.history.done, splitIds);
        other.history.undone = copyHistoryArray(this.history.undone, splitIds);
      }
    },
    // 遍历当前文档的链接文档，并对每个文档执行指定的函数
    iterLinkedDocs: function(f) {linkedDocs(this, f);},

    // 获取当前文档的模式
    getMode: function() {return this.mode},
    // 获取当前文档的编辑器对象
    getEditor: function() {return this.cm},

    // 将字符串按行分割
    splitLines: function(str) {
      // 如果当前文档有指定的行分隔符，则使用指定的分隔符进行分割
      if (this.lineSep) { return str.split(this.lineSep) }
      // 否则自动识别行分隔符进行分割
      return splitLinesAuto(str)
    },
    // 获取当前文档的行分隔符
    lineSeparator: function() { return this.lineSep || "\n" },

    // 设置文档的文本方向
    setDirection: docMethodOp(function (dir) {
      // 如果指定的方向不是 "rtl"，则默认设置为 "ltr"
      if (dir != "rtl") { dir = "ltr"; }
      // 如果指定的方向和当前文档的方向相同，则直接返回
      if (dir == this.direction) { return }
      // 否则设置文档的方向为指定的方向
      this.direction = dir;
      // 遍历文档的每一行，重置其 order 属性
      this.iter(function (line) { return line.order = null; });
      // 如果当前文档关联了编辑器对象，则调用 directionChanged 方法
      if (this.cm) { directionChanged(this.cm); }
    })
  });

  // 公共别名
  Doc.prototype.eachLine = Doc.prototype.iter;

  // 解决 IE 中拖放事件的奇怪行为
  var lastDrop = 0;

  // 处理拖放事件
  function onDrop(e) {
    var cm = this;
    // 清除拖动光标
    clearDragCursor(cm);
    // 如果触发了 DOM 事件或者在小部件中触发了事件，则直接返回
    if (signalDOMEvent(cm, e) || eventInWidget(cm.display, e))
      { return }
    // 阻止默认的拖放行为
    e_preventDefault(e);
    // 如果是 IE 浏览器，则记录最后一次拖放的时间
    if (ie) { lastDrop = +new Date; }
    // 获取鼠标位置和拖放的文件
    var pos = posFromMouse(cm, e, true), files = e.dataTransfer.files;
    // 如果 pos 为假值或者编辑器为只读状态，则直接返回，不执行后续操作
    if (!pos || cm.isReadOnly()) { return }
    // 可能是文件拖放操作，此时我们简单提取文本并插入
    if (files && files.length && window.FileReader && window.File) {
      // 获取文件数量和文本数组，以及已读取文件数量
      var n = files.length, text = Array(n), read = 0;
      // 定义一个函数，用于标记已读取并在所有文件都读取完毕后执行粘贴操作
      var markAsReadAndPasteIfAllFilesAreRead = function () {
        if (++read == n) {
          // 执行编辑器操作
          operation(cm, function () {
            // 获取光标位置
            pos = clipPos(cm.doc, pos);
            // 构造文本变更对象
            var change = {from: pos, to: pos,
                          text: cm.doc.splitLines(
                              text.filter(function (t) { return t != null; }).join(cm.doc.lineSeparator())),
                          origin: "paste"};
            // 执行文本变更
            makeChange(cm.doc, change);
            // 设置光标位置并记录操作历史
            setSelectionReplaceHistory(cm.doc, simpleSelection(clipPos(cm.doc, pos), clipPos(cm.doc, changeEnd(change))));
          })();
        }
      };
      // 定义一个函数，用于读取文件内容并标记已读取
      var readTextFromFile = function (file, i) {
        // 如果编辑器允许拖放的文件类型，并且当前文件类型不在允许的类型列表中，则标记已读取并返回
        if (cm.options.allowDropFileTypes &&
            indexOf(cm.options.allowDropFileTypes, file.type) == -1) {
          markAsReadAndPasteIfAllFilesAreRead();
          return
        }
        // 创建一个新的文件阅读器对象
        var reader = new FileReader;
        // 处理读取错误
        reader.onerror = function () { return markAsReadAndPasteIfAllFilesAreRead(); };
        // 处理读取成功
        reader.onload = function () {
          var content = reader.result;
          // 如果内容中包含不可见字符，则标记已读取并返回
          if (/[\x00-\x08\x0e-\x1f]{2}/.test(content)) {
            markAsReadAndPasteIfAllFilesAreRead();
            return
          }
          // 将文件内容存入文本数组，并标记已读取
          text[i] = content;
          markAsReadAndPasteIfAllFilesAreRead();
        };
        // 以文本形式读取文件内容
        reader.readAsText(file);
      };
      // 遍历文件列表，依次读取文件内容并标记已读取
      for (var i = 0; i < files.length; i++) { readTextFromFile(files[i], i); }
    } else { // Normal drop
      // 如果是正常的拖放操作，不在选中文本内则执行替换操作
      if (cm.state.draggingText && cm.doc.sel.contains(pos) > -1) {
        // 如果正在拖动文本并且拖放发生在选中文本内，则执行拖动文本的操作
        cm.state.draggingText(e);
        // 确保编辑器重新获得焦点
        setTimeout(function () { return cm.display.input.focus(); }, 20);
        return
      }
      try {
        var text$1 = e.dataTransfer.getData("Text");
        if (text$1) {
          var selected;
          // 如果正在拖动文本并且不是拷贝操作，则保存当前选中状态
          if (cm.state.draggingText && !cm.state.draggingText.copy)
            { selected = cm.listSelections(); }
          // 设置新的选中文本，不记录撤销操作
          setSelectionNoUndo(cm.doc, simpleSelection(pos, pos));
          // 如果有保存的选中状态，则清空选中文本
          if (selected) { for (var i$1 = 0; i$1 < selected.length; ++i$1)
            { replaceRange(cm.doc, "", selected[i$1].anchor, selected[i$1].head, "drag"); } }
          // 替换选中文本为拖放的文本，记录为粘贴操作
          cm.replaceSelection(text$1, "around", "paste");
          // 重新获得焦点
          cm.display.input.focus();
        }
      }
      catch(e$1){}
    }
  }

  // 拖动开始事件处理函数
  function onDragStart(cm, e) {
    // 如果是 IE 并且不是拖动文本或者距离上次拖放不到 100 毫秒，则阻止默认行为
    if (ie && (!cm.state.draggingText || +new Date - lastDrop < 100)) { e_stop(e); return }
    // 如果信号是 DOM 事件或者在小部件内部，则返回
    if (signalDOMEvent(cm, e) || eventInWidget(cm.display, e)) { return }

    // 设置拖动的文本数据和效果
    e.dataTransfer.setData("Text", cm.getSelection());
    e.dataTransfer.effectAllowed = "copyMove";

    // 使用虚拟图像代替默认的浏览器图像
    // 最近的 Safari（~6.0.2）在这种情况下有时会崩溃，所以我们在那里不这样做
    if (e.dataTransfer.setDragImage && !safari) {
      var img = elt("img", null, null, "position: fixed; left: 0; top: 0;");
      img.src = "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==";
      if (presto) {
        img.width = img.height = 1;
        cm.display.wrapper.appendChild(img);
        // 强制重新布局，否则 Opera 会因为某种隐晦的原因而不使用我们的图像
        img._top = img.offsetTop;
      }
      e.dataTransfer.setDragImage(img, 0, 0);
      if (presto) { img.parentNode.removeChild(img); }
  }
}

function onDragOver(cm, e) {
  // 获取鼠标位置对应的编辑器光标位置
  var pos = posFromMouse(cm, e);
  if (!pos) { return }
  // 创建一个文档片段
  var frag = document.createDocumentFragment();
  // 在光标位置绘制选择光标
  drawSelectionCursor(cm, pos, frag);
  if (!cm.display.dragCursor) {
    // 如果不存在拖动光标，则创建一个新的拖动光标
    cm.display.dragCursor = elt("div", null, "CodeMirror-cursors CodeMirror-dragcursors");
    cm.display.lineSpace.insertBefore(cm.display.dragCursor, cm.display.cursorDiv);
  }
  // 移除原有的子元素并添加新的文档片段
  removeChildrenAndAdd(cm.display.dragCursor, frag);
}

function clearDragCursor(cm) {
  if (cm.display.dragCursor) {
    // 如果存在拖动光标，则移除它
    cm.display.lineSpace.removeChild(cm.display.dragCursor);
    cm.display.dragCursor = null;
  }
}

// 这些必须小心处理，因为简单地为每个编辑器注册处理程序将导致编辑器永远不会被垃圾回收。

function forEachCodeMirror(f) {
  if (!document.getElementsByClassName) { return }
  // 获取所有具有 "CodeMirror" 类名的元素
  var byClass = document.getElementsByClassName("CodeMirror"), editors = [];
  for (var i = 0; i < byClass.length; i++) {
    var cm = byClass[i].CodeMirror;
    if (cm) { editors.push(cm); }
  }
  if (editors.length) { editors[0].operation(function () {
    for (var i = 0; i < editors.length; i++) { f(editors[i]); }
  }); }
}

var globalsRegistered = false;
function ensureGlobalHandlers() {
  if (globalsRegistered) { return }
  // 确保全局处理程序已注册
  registerGlobalHandlers();
  globalsRegistered = true;
}
function registerGlobalHandlers() {
  // 当窗口大小改变时，需要刷新活动编辑器
  var resizeTimer;
  on(window, "resize", function () {
    if (resizeTimer == null) { resizeTimer = setTimeout(function () {
      resizeTimer = null;
      forEachCodeMirror(onResize);
    }, 100); }
  });
  // 当窗口失去焦点时，希望将编辑器显示为模糊状态
  on(window, "blur", function () { return forEachCodeMirror(onBlur); });
}
// 窗口大小改变时调用
function onResize(cm) {
  var d = cm.display;
    // 可能是文本缩放操作，清除大小缓存。
    d.cachedCharWidth = d.cachedTextHeight = d.cachedPaddingH = null;
    d.scrollbarsClipped = false;
    cm.setSize();
  }

  // 键盘按键名称映射表
  var keyNames = {
    3: "Pause", 8: "Backspace", 9: "Tab", 13: "Enter", 16: "Shift", 17: "Ctrl", 18: "Alt",
    19: "Pause", 20: "CapsLock", 27: "Esc", 32: "Space", 33: "PageUp", 34: "PageDown", 35: "End",
    36: "Home", 37: "Left", 38: "Up", 39: "Right", 40: "Down", 44: "PrintScrn", 45: "Insert",
    46: "Delete", 59: ";", 61: "=", 91: "Mod", 92: "Mod", 93: "Mod",
    106: "*", 107: "=", 109: "-", 110: ".", 111: "/", 145: "ScrollLock",
    173: "-", 186: ";", 187: "=", 188: ",", 189: "-", 190: ".", 191: "/", 192: "`", 219: "[", 220: "\\",
    221: "]", 222: "'", 63232: "Up", 63233: "Down", 63234: "Left", 63235: "Right", 63272: "Delete",
    63273: "Home", 63275: "End", 63276: "PageUp", 63277: "PageDown", 63302: "Insert"
  };

  // 数字键
  for (var i = 0; i < 10; i++) { keyNames[i + 48] = keyNames[i + 96] = String(i); }
  // 字母键
  for (var i$1 = 65; i$1 <= 90; i$1++) { keyNames[i$1] = String.fromCharCode(i$1); }
  // 功能键
  for (var i$2 = 1; i$2 <= 12; i$2++) { keyNames[i$2 + 111] = keyNames[i$2 + 63235] = "F" + i$2; }

  // 键盘按键映射表
  var keyMap = {};

  keyMap.basic = {
    "Left": "goCharLeft", "Right": "goCharRight", "Up": "goLineUp", "Down": "goLineDown",
    "End": "goLineEnd", "Home": "goLineStartSmart", "PageUp": "goPageUp", "PageDown": "goPageDown",
    "Delete": "delCharAfter", "Backspace": "delCharBefore", "Shift-Backspace": "delCharBefore",
    "Tab": "defaultTab", "Shift-Tab": "indentAuto",
    "Enter": "newlineAndIndent", "Insert": "toggleOverwrite",
    "Esc": "singleSelection"
  };
  // 注意，默认情况下，保存和查找相关的命令未被定义。
  // 用户代码或插件可以定义它们。未知命令将被简单忽略。
  keyMap.pcDefault = {
    // 定义键盘快捷键映射对象
    keyMap.default = {
        "Ctrl-A": "selectAll", // 选择全部
        "Ctrl-D": "deleteLine", // 删除当前行
        "Ctrl-Z": "undo", // 撤销
        "Shift-Ctrl-Z": "redo", // 重做
        "Ctrl-Y": "redo", // 重做
        "Ctrl-Home": "goDocStart", // 光标移动到文档开头
        "Ctrl-End": "goDocEnd", // 光标移动到文档结尾
        "Ctrl-Up": "goLineUp", // 光标向上移动一行
        "Ctrl-Down": "goLineDown", // 光标向下移动一行
        "Ctrl-Left": "goGroupLeft", // 光标向左移动一个单词
        "Ctrl-Right": "goGroupRight", // 光标向右移动一个单词
        "Alt-Left": "goLineStart", // 光标移动到行首
        "Alt-Right": "goLineEnd", // 光标移动到行尾
        "Ctrl-Backspace": "delGroupBefore", // 删除光标前的单词
        "Ctrl-Delete": "delGroupAfter", // 删除光标后的单词
        "Ctrl-S": "save", // 保存
        "Ctrl-F": "find", // 查找
        "Ctrl-G": "findNext", // 查找下一个
        "Shift-Ctrl-G": "findPrev", // 查找上一个
        "Shift-Ctrl-F": "replace", // 替换
        "Shift-Ctrl-R": "replaceAll", // 替换所有
        "Ctrl-[": "indentLess", // 减少缩进
        "Ctrl-]": "indentMore", // 增加缩进
        "Ctrl-U": "undoSelection", // 撤销选择
        "Shift-Ctrl-U": "redoSelection", // 重做选择
        "Alt-U": "redoSelection", // 重做选择
        "fallthrough": "basic" // 基本操作
    };
    
    // 定义 readline/emacs 风格的键盘快捷键映射对象
    keyMap.emacsy = {
        "Ctrl-F": "goCharRight", // 光标向右移动一个字符
        "Ctrl-B": "goCharLeft", // 光标向左移动一个字符
        "Ctrl-P": "goLineUp", // 光标向上移动一行
        "Ctrl-N": "goLineDown", // 光标向下移动一行
        "Alt-F": "goWordRight", // 光标向右移动一个单词
        "Alt-B": "goWordLeft", // 光标向左移动一个单词
        "Ctrl-A": "goLineStart", // 光标移动到行首
        "Ctrl-E": "goLineEnd", // 光标移动到行尾
        "Ctrl-V": "goPageDown", // 向下翻页
        "Shift-Ctrl-V": "goPageUp", // 向上翻页
        "Ctrl-D": "delCharAfter", // 删除光标后的字符
        "Ctrl-H": "delCharBefore", // 删除光标前的字符
        "Alt-D": "delWordAfter", // 删除光标后的单词
        "Alt-Backspace": "delWordBefore", // 删除光标前的单词
        "Ctrl-K": "killLine", // 删除当前行
        "Ctrl-T": "transposeChars", // 交换光标前后的字符
        "Ctrl-O": "openLine" // 在当前行下方插入新行
    };
    
    // 定义 Mac 默认风格的键盘快捷键映射对象
    keyMap.macDefault = {
        "Cmd-A": "selectAll", // 选择全部
        "Cmd-D": "deleteLine", // 删除当前行
        "Cmd-Z": "undo", // 撤销
        "Shift-Cmd-Z": "redo", // 重做
        "Cmd-Y": "redo", // 重做
        "Cmd-Home": "goDocStart", // 光标移动到文档开头
        "Cmd-Up": "goDocStart", // 光标移动到文档开头
        "Cmd-End": "goDocEnd", // 光标移动到文档结尾
        "Cmd-Down": "goDocEnd", // 光标移动到文档结尾
        "Alt-Left": "goGroupLeft", // 光标向左移动一个单词
        "Alt-Right": "goGroupRight", // 光标向右移动一个单词
        "Cmd-Left": "goLineLeft", // 光标向左移动一个字符
        "Cmd-Right": "goLineRight", // 光标向右移动一个字符
        "Alt-Backspace": "delGroupBefore", // 删除光标前的单词
        "Ctrl-Alt-Backspace": "delGroupAfter", // 删除光标后的单词
        "Alt-Delete": "delGroupAfter", // 删除光标后的单词
        "Cmd-S": "save", // 保存
        "Cmd-F": "find", // 查找
        "Cmd-G": "findNext", // 查找下一个
        "Shift-Cmd-G": "findPrev", // 查找上一个
        "Cmd-Alt-F": "replace", // 替换
        "Shift-Cmd-Alt-F": "replaceAll", // 替换所有
        "Cmd-[": "indentLess", // 减少缩进
        "Cmd-]": "indentMore", // 增加缩进
        "Cmd-Backspace": "delWrappedLineLeft", // 删除当前行
        "Cmd-Delete": "delWrappedLineRight", // 删除当前行
    // 定义键盘快捷键映射对象，包括撤销、重做、移动到文档开头、移动到文档结尾等操作
    "Cmd-U": "undoSelection", "Shift-Cmd-U": "redoSelection", "Ctrl-Up": "goDocStart", "Ctrl-Down": "goDocEnd",
    // 定义键盘快捷键映射对象的默认值
    "fallthrough": ["basic", "emacsy"]
  };
  // 根据操作系统类型选择默认的键盘快捷键映射对象
  keyMap["default"] = mac ? keyMap.macDefault : keyMap.pcDefault;

  // KEYMAP DISPATCH

  // 标准化键盘快捷键名称，将其转换为统一格式
  function normalizeKeyName(name) {
    var parts = name.split(/-(?!$)/);
    name = parts[parts.length - 1];
    var alt, ctrl, shift, cmd;
    // 遍历键盘快捷键名称的各个部分，识别并标记修饰键
    for (var i = 0; i < parts.length - 1; i++) {
      var mod = parts[i];
      if (/^(cmd|meta|m)$/i.test(mod)) { cmd = true; }
      else if (/^a(lt)?$/i.test(mod)) { alt = true; }
      else if (/^(c|ctrl|control)$/i.test(mod)) { ctrl = true; }
      else if (/^s(hift)?$/i.test(mod)) { shift = true; }
      else { throw new Error("Unrecognized modifier name: " + mod) }
    }
    // 根据修饰键标记，重新构建标准化的键盘快捷键名称
    if (alt) { name = "Alt-" + name; }
    if (ctrl) { name = "Ctrl-" + name; }
    if (cmd) { name = "Cmd-" + name; }
    if (shift) { name = "Shift-" + name; }
    return name
  }

  // This is a kludge to keep keymaps mostly working as raw objects
  // (backwards compatibility) while at the same time support features
  // like normalization and multi-stroke key bindings. It compiles a
  // new normalized keymap, and then updates the old object to reflect
  // this.
  // 标准化键盘快捷键映射对象，以支持统一格式和多键绑定等功能
  function normalizeKeyMap(keymap) {
    var copy = {};
    // 遍历 keymap 对象的属性
    for (var keyname in keymap) { if (keymap.hasOwnProperty(keyname)) {
      // 获取属性值
      var value = keymap[keyname];
      // 如果属性名为 name、fallthrough、detach，则跳过当前循环
      if (/^(name|fallthrough|(de|at)tach)$/.test(keyname)) { continue }
      // 如果属性值为 "..."，则删除该属性并跳过当前循环
      if (value == "...") { delete keymap[keyname]; continue }

      // 将属性名按空格分割并规范化
      var keys = map(keyname.split(" "), normalizeKeyName);
      // 遍历分割后的属性名
      for (var i = 0; i < keys.length; i++) {
        // 初始化变量 val 和 name
        var val = (void 0), name = (void 0);
        // 如果当前索引为最后一个，则将 name 设置为连接后的属性名，val 设置为属性值
        if (i == keys.length - 1) {
          name = keys.join(" ");
          val = value;
        } else {
          // 否则将 name 设置为部分连接后的属性名，val 设置为 "..."
          name = keys.slice(0, i + 1).join(" ");
          val = "...";
        }
        // 获取之前的属性值
        var prev = copy[name];
        // 如果之前的属性值不存在，则将当前属性值赋给 copy 对象
        if (!prev) { copy[name] = val; }
        // 如果之前的属性值存在且与当前属性值不一致，则抛出错误
        else if (prev != val) { throw new Error("Inconsistent bindings for " + name) }
      }
      // 删除 keymap 对象中的属性
      delete keymap[keyname];
    } }
    // 将 copy 对象中的属性复制到 keymap 对象中
    for (var prop in copy) { keymap[prop] = copy[prop]; }
    // 返回处理后的 keymap 对象
    return keymap
  }

  // 查找按键对应的处理函数
  function lookupKey(key, map, handle, context) {
    // 获取处理后的 map 对象
    map = getKeyMap(map);
    // 调用 map 对象的 call 方法或直接获取 map 对象中的属性值
    var found = map.call ? map.call(key, context) : map[key];
    // 如果属性值为 false，则返回 "nothing"
    if (found === false) { return "nothing" }
    // 如果属性值为 "..."，则返回 "multi"
    if (found === "...") { return "multi" }
    // 如果属性值不为 null 且能够处理，则返回 "handled"
    if (found != null && handle(found)) { return "handled" }

    // 如果 map 对象有 fallthrough 属性
    if (map.fallthrough) {
      // 如果 fallthrough 属性不是数组，则递归调用 lookupKey 函数
      if (Object.prototype.toString.call(map.fallthrough) != "[object Array]")
        { return lookupKey(key, map.fallthrough, handle, context) }
      // 遍历 fallthrough 属性的数组
      for (var i = 0; i < map.fallthrough.length; i++) {
        // 递归调用 lookupKey 函数
        var result = lookupKey(key, map.fallthrough[i], handle, context);
        // 如果有结果，则返回结果
        if (result) { return result }
      }
    }
  }

  // 判断是否为修饰键
  function isModifierKey(value) {
    // 获取键名
    var name = typeof value == "string" ? value : keyNames[value.keyCode];
    // 判断是否为修饰键
    return name == "Ctrl" || name == "Alt" || name == "Shift" || name == "Mod"
  }

  // 添加修饰键的名称
  function addModifierNames(name, event, noShift) {
    // 获取基础名称
    var base = name;
    // 如果按下了 Alt 键且基础名称不为 "Alt"，则添加 "Alt-" 前缀
    if (event.altKey && base != "Alt") { name = "Alt-" + name; }
    # 如果 flipCtrlCmd 为真，则判断 event.metaKey，否则判断 event.ctrlKey，如果满足条件且 base 不为 "Ctrl"，则在 name 前加上 "Ctrl-"
    if ((flipCtrlCmd ? event.metaKey : event.ctrlKey) && base != "Ctrl") { name = "Ctrl-" + name; }
    # 如果 flipCtrlCmd 为真，则判断 event.ctrlKey，否则判断 event.metaKey，如果满足条件且 base 不为 "Cmd"，则在 name 前加上 "Cmd-"
    if ((flipCtrlCmd ? event.ctrlKey : event.metaKey) && base != "Cmd") { name = "Cmd-" + name; }
    # 如果 noShift 为假且 event.shiftKey 为真且 base 不为 "Shift"，则在 name 前加上 "Shift-"
    if (!noShift && event.shiftKey && base != "Shift") { name = "Shift-" + name; }
    # 返回处理后的 name
    return name
  }

  // 根据事件对象查找键的名称
  function keyName(event, noShift) {
    # 如果是 Presto 浏览器且 keyCode 为 34 且有字符，则返回假
    if (presto && event.keyCode == 34 && event["char"]) { return false }
    # 根据 keyCode 查找键的名称
    var name = keyNames[event.keyCode];
    # 如果名称为空或者有 altGraphKey，则返回假
    if (name == null || event.altGraphKey) { return false }
    # 如果 keyCode 为 3 且有 code，则使用 event.code 作为名称
    if (event.keyCode == 3 && event.code) { name = event.code; }
    # 返回添加修饰键后的名称
    return addModifierNames(name, event, noShift)
  }

  # 获取键映射
  function getKeyMap(val) {
    # 如果 val 是字符串，则返回 keyMap[val]，否则返回 val
    return typeof val == "string" ? keyMap[val] : val
  }

  // 用于删除选择区域附近的文本，实现退格、删除等功能
  function deleteNearSelection(cm, compute) {
    var ranges = cm.doc.sel.ranges, kill = [];
    # 遍历选择区域，构建要删除的区域集合
    for (var i = 0; i < ranges.length; i++) {
      var toKill = compute(ranges[i]);
      while (kill.length && cmp(toKill.from, lst(kill).to) <= 0) {
        var replaced = kill.pop();
        if (cmp(replaced.from, toKill.from) < 0) {
          toKill.from = replaced.from;
          break
        }
      }
      kill.push(toKill);
    }
    # 删除实际区域
    runInOp(cm, function () {
      for (var i = kill.length - 1; i >= 0; i--)
        { replaceRange(cm.doc, "", kill[i].from, kill[i].to, "+delete"); }
      ensureCursorVisible(cm);
    });
  }

  # 在逻辑上移动字符
  function moveCharLogically(line, ch, dir) {
    # 跳过扩展字符，获取目标位置
    var target = skipExtendingChars(line.text, ch + dir, dir);
  // 如果目标位置小于0或大于行文本长度，返回null；否则返回目标位置
  return target < 0 || target > line.text.length ? null : target
}

// 在文本行内逻辑移动光标位置
function moveLogically(line, start, dir) {
  // 根据方向和字符位置移动光标
  var ch = moveCharLogically(line, start.ch, dir);
  // 如果移动后位置为null，返回null；否则返回新的位置对象
  return ch == null ? null : new Pos(start.line, ch, dir < 0 ? "after" : "before")
}

// 获取行的可视末尾位置
function endOfLine(visually, cm, lineObj, lineNo, dir) {
  // 如果是可视模式
  if (visually) {
    // 如果文档方向是rtl，则改变方向
    if (cm.doc.direction == "rtl") { dir = -dir; }
    // 获取行的文本顺序
    var order = getOrder(lineObj, cm.doc.direction);
    // 如果存在文本顺序
    if (order) {
      // 根据方向获取对应的部分
      var part = dir < 0 ? lst(order) : order[0];
      // 根据移动方向和部分级别确定sticky值
      var moveInStorageOrder = (dir < 0) == (part.level == 1);
      var sticky = moveInStorageOrder ? "after" : "before";
      var ch;
      // 如果部分级别大于0或文档方向是rtl
      if (part.level > 0 || cm.doc.direction == "rtl") {
        // 准备测量行
        var prep = prepareMeasureForLine(cm, lineObj);
        // 根据方向确定初始字符位置
        ch = dir < 0 ? lineObj.text.length - 1 : 0;
        // 获取目标字符位置
        var targetTop = measureCharPrepared(cm, prep, ch).top;
        ch = findFirst(function (ch) { return measureCharPrepared(cm, prep, ch).top == targetTop; }, (dir < 0) == (part.level == 1) ? part.from : part.to - 1, ch);
        // 如果sticky为"before"，根据字符位置和方向移动光标
        if (sticky == "before") { ch = moveCharLogically(lineObj, ch, 1); }
      } else { 
        // 根据方向确定字符位置
        ch = dir < 0 ? part.to : part.from; 
      }
      // 返回新的位置对象
      return new Pos(lineNo, ch, sticky)
    }
  }
  // 返回新的位置对象
  return new Pos(lineNo, dir < 0 ? lineObj.text.length : 0, dir < 0 ? "before" : "after")
}

// 在文本行内可视移动光标位置
function moveVisually(cm, line, start, dir) {
  // 获取文本行的文本顺序
  var bidi = getOrder(line, cm.doc.direction);
  // 如果不存在文本顺序，返回逻辑移动的结果
  if (!bidi) { return moveLogically(line, start, dir) }
}
    # 如果起始位置超出了文本长度，则将起始位置设置为文本长度，并设置sticky属性为"before"
    if (start.ch >= line.text.length) {
      start.ch = line.text.length;
      start.sticky = "before";
    } 
    # 如果起始位置小于等于0，则将起始位置设置为0，并设置sticky属性为"after"
    else if (start.ch <= 0) {
      start.ch = 0;
      start.sticky = "after";
    }
    # 获取起始位置所在的双向文本部分
    var partPos = getBidiPartAt(bidi, start.ch, start.sticky), part = bidi[partPos];
    # 如果编辑器的文本方向为"ltr"，且当前部分的级别为偶数，并且（dir > 0 ? part.to > start.ch : part.from < start.ch）条件成立
    if (cm.doc.direction == "ltr" && part.level % 2 == 0 && (dir > 0 ? part.to > start.ch : part.from < start.ch)) {
      # 情况1：在ltr编辑器中移动ltr部分内。即使在换行的情况下，也不会发生有趣的事情。
      return moveLogically(line, start, dir)
    }

    # 定义一个函数mv，用于逻辑移动字符
    var mv = function (pos, dir) { return moveCharLogically(line, pos instanceof Pos ? pos.ch : pos, dir); };
    var prep;
    # 定义一个函数getWrappedLineExtent，用于获取包裹行的范围
    var getWrappedLineExtent = function (ch) {
      if (!cm.options.lineWrapping) { return {begin: 0, end: line.text.length} }
      prep = prep || prepareMeasureForLine(cm, line);
      return wrappedLineExtentChar(cm, line, prep, ch)
    };
    # 获取起始位置所在的包裹行的范围
    var wrappedLineExtent = getWrappedLineExtent(start.sticky == "before" ? mv(start, -1) : start.ch);

    # 如果文档的文本方向为"rtl"，或者当前部分的级别为1
    if (cm.doc.direction == "rtl" || part.level == 1) {
      # 根据条件判断是否按存储顺序移动
      var moveInStorageOrder = (part.level == 1) == (dir < 0);
      # 根据移动方向和条件判断是否在同一可视行内移动
      var ch = mv(start, moveInStorageOrder ? 1 : -1);
      if (ch != null && (!moveInStorageOrder ? ch >= part.from && ch >= wrappedLineExtent.begin : ch <= part.to && ch <= wrappedLineExtent.end)) {
        # 情况2：在rtl部分内移动或在同一可视行上的rtl编辑器中移动
        var sticky = moveInStorageOrder ? "before" : "after";
        return new Pos(start.line, ch, sticky)
      }
    }

    # 情况3：无法在当前可视行内的双向文本部分内移动，因此离开当前双向文本部分
    // 在可视行中搜索指定位置的文本
    var searchInVisualLine = function (partPos, dir, wrappedLineExtent) {
      // 获取结果的函数
      var getRes = function (ch, moveInStorageOrder) { return moveInStorageOrder
        ? new Pos(start.line, mv(ch, 1), "before")
        : new Pos(start.line, ch, "after"); };

      // 遍历双向文本数组
      for (; partPos >= 0 && partPos < bidi.length; partPos += dir) {
        var part = bidi[partPos];
        var moveInStorageOrder = (dir > 0) == (part.level != 1);
        var ch = moveInStorageOrder ? wrappedLineExtent.begin : mv(wrappedLineExtent.end, -1);
        // 如果指定位置在当前部分内，则返回结果
        if (part.from <= ch && ch < part.to) { return getRes(ch, moveInStorageOrder) }
        ch = moveInStorageOrder ? part.from : mv(part.to, -1);
        // 如果指定位置在当前行内，则返回结果
        if (wrappedLineExtent.begin <= ch && ch < wrappedLineExtent.end) { return getRes(ch, moveInStorageOrder) }
      }
    };

    // Case 3a: 在同一可视行上查找其他双向文本部分
    var res = searchInVisualLine(partPos + dir, dir, wrappedLineExtent);
    if (res) { return res }

    // Case 3b: 在下一行上查找其他双向文本部分
    var nextCh = dir > 0 ? wrappedLineExtent.end : mv(wrappedLineExtent.begin, -1);
    if (nextCh != null && !(dir > 0 && nextCh == line.text.length)) {
      res = searchInVisualLine(dir > 0 ? 0 : bidi.length - 1, dir, getWrappedLineExtent(nextCh));
      if (res) { return res }
    }

    // Case 4: 无法移动到任何位置
    return null
  }

  // Commands are parameter-less actions that can be performed on an
  // editor, mostly used for keybindings.
  // 命令是在编辑器上执行的无参数操作，主要用于键绑定。
  var commands = {
    // 选择全部文本
    selectAll: selectAll,
    // 单选
    singleSelection: function (cm) { return cm.setSelection(cm.getCursor("anchor"), cm.getCursor("head"), sel_dontScroll); },
    // 删除当前行的内容
    killLine: function (cm) { return deleteNearSelection(cm, function (range) {
      // 如果选区为空
      if (range.empty()) {
        // 获取当前行的长度
        var len = getLine(cm.doc, range.head.line).text.length;
        // 如果光标在行末且不是最后一行
        if (range.head.ch == len && range.head.line < cm.lastLine())
          { return {from: range.head, to: Pos(range.head.line + 1, 0)} }
        else
          { return {from: range.head, to: Pos(range.head.line, len)} }
      } else {
        // 返回选区的起始和结束位置
        return {from: range.from(), to: range.to()}
      }
    }); },
    // 删除整行
    deleteLine: function (cm) { return deleteNearSelection(cm, function (range) { return ({
      from: Pos(range.from().line, 0),
      to: clipPos(cm.doc, Pos(range.to().line + 1, 0))
    }); }); },
    // 删除光标左侧的内容
    delLineLeft: function (cm) { return deleteNearSelection(cm, function (range) { return ({
      from: Pos(range.from().line, 0), to: range.from()
    }); }); },
    // 删除包裹的行的左侧内容
    delWrappedLineLeft: function (cm) { return deleteNearSelection(cm, function (range) {
      // 获取光标所在行的顶部位置
      var top = cm.charCoords(range.head, "div").top + 5;
      // 获取左侧位置的光标
      var leftPos = cm.coordsChar({left: 0, top: top}, "div");
      return {from: leftPos, to: range.from()}
    }); },
    // 删除包裹的行的右侧内容
    delWrappedLineRight: function (cm) { return deleteNearSelection(cm, function (range) {
      // 获取光标所在行的顶部位置
      var top = cm.charCoords(range.head, "div").top + 5;
      // 获取右侧位置的光标
      var rightPos = cm.coordsChar({left: cm.display.lineDiv.offsetWidth + 100, top: top}, "div");
      return {from: range.from(), to: rightPos }
    }); },
    // 撤销操作
    undo: function (cm) { return cm.undo(); },
    // 重做操作
    redo: function (cm) { return cm.redo(); },
    // 撤销选区操作
    undoSelection: function (cm) { return cm.undoSelection(); },
    // 重做选区操作
    redoSelection: function (cm) { return cm.redoSelection(); },
    // 移动光标到文档开头
    goDocStart: function (cm) { return cm.extendSelection(Pos(cm.firstLine(), 0)); },
    // 移动光标到文档末尾
    goDocEnd: function (cm) { return cm.extendSelection(Pos(cm.lastLine())); },
    // 移动光标到行开头
    goLineStart: function (cm) { return cm.extendSelectionsBy(function (range) { return lineStart(cm, range.head.line); },
      {origin: "+move", bias: 1}
    ); },
    goLineStartSmart: function (cm) { return cm.extendSelectionsBy(function (range) { return lineStartSmart(cm, range.head); },
      {origin: "+move", bias: 1}
    ); },  // 在编辑器中将光标移动到当前行的开头，根据当前光标位置智能选择
    goLineEnd: function (cm) { return cm.extendSelectionsBy(function (range) { return lineEnd(cm, range.head.line); },
      {origin: "+move", bias: -1}
    ); },  // 在编辑器中将光标移动到当前行的末尾
    goLineRight: function (cm) { return cm.extendSelectionsBy(function (range) {
      var top = cm.cursorCoords(range.head, "div").top + 5;
      return cm.coordsChar({left: cm.display.lineDiv.offsetWidth + 100, top: top}, "div")
    }, sel_move); },  // 在编辑器中将光标移动到当前行的右侧
    goLineLeft: function (cm) { return cm.extendSelectionsBy(function (range) {
      var top = cm.cursorCoords(range.head, "div").top + 5;
      return cm.coordsChar({left: 0, top: top}, "div")
    }, sel_move); },  // 在编辑器中将光标移动到当前行的左侧
    goLineLeftSmart: function (cm) { return cm.extendSelectionsBy(function (range) {
      var top = cm.cursorCoords(range.head, "div").top + 5;
      var pos = cm.coordsChar({left: 0, top: top}, "div");
      if (pos.ch < cm.getLine(pos.line).search(/\S/)) { return lineStartSmart(cm, range.head) }
      return pos
    }, sel_move); },  // 在编辑器中将光标移动到当前行的左侧，根据当前光标位置智能选择
    goLineUp: function (cm) { return cm.moveV(-1, "line"); },  // 在编辑器中将光标向上移动一行
    goLineDown: function (cm) { return cm.moveV(1, "line"); },  // 在编辑器中将光标向下移动一行
    goPageUp: function (cm) { return cm.moveV(-1, "page"); },  // 在编辑器中将光标向上移动一页
    goPageDown: function (cm) { return cm.moveV(1, "page"); },  // 在编辑器中将光标向下移动一页
    goCharLeft: function (cm) { return cm.moveH(-1, "char"); },  // 在编辑器中将光标向左移动一个字符
    goCharRight: function (cm) { return cm.moveH(1, "char"); },  // 在编辑器中将光标向右移动一个字符
    goColumnLeft: function (cm) { return cm.moveH(-1, "column"); },  // 在编辑器中将光标向左移动一列
    goColumnRight: function (cm) { return cm.moveH(1, "column"); },  // 在编辑器中将光标向右移动一列
    goWordLeft: function (cm) { return cm.moveH(-1, "word"); },  // 在编辑器中将光标向左移动一个单词
    goGroupRight: function (cm) { return cm.moveH(1, "group"); },  // 在编辑器中将光标向右移动一个组
    goGroupLeft: function (cm) { return cm.moveH(-1, "group"); },  // 在编辑器中将光标向左移动一个组
    goWordRight: function (cm) { return cm.moveH(1, "word"); },  // 在编辑器中将光标向右移动一个单词
    delCharBefore: function (cm) { return cm.deleteH(-1, "char"); },  // 在编辑器中删除光标前的一个字符
    # 删除光标后的字符
    delCharAfter: function (cm) { return cm.deleteH(1, "char"); },
    # 删除光标前的单词
    delWordBefore: function (cm) { return cm.deleteH(-1, "word"); },
    # 删除光标后的单词
    delWordAfter: function (cm) { return cm.deleteH(1, "word"); },
    # 删除光标前的一组字符（例如，一个单词）
    delGroupBefore: function (cm) { return cm.deleteH(-1, "group"); },
    # 删除光标后的一组字符（例如，一个单词）
    delGroupAfter: function (cm) { return cm.deleteH(1, "group"); },
    # 自动缩进选定的文本
    indentAuto: function (cm) { return cm.indentSelection("smart"); },
    # 增加选定文本的缩进
    indentMore: function (cm) { return cm.indentSelection("add"); },
    # 减少选定文本的缩进
    indentLess: function (cm) { return cm.indentSelection("subtract"); },
    # 插入制表符
    insertTab: function (cm) { return cm.replaceSelection("\t"); },
    # 插入软制表符
    insertSoftTab: function (cm) {
      # 计算需要插入的空格数，以保持制表符的对齐
      var spaces = [], ranges = cm.listSelections(), tabSize = cm.options.tabSize;
      for (var i = 0; i < ranges.length; i++) {
        var pos = ranges[i].from();
        var col = countColumn(cm.getLine(pos.line), pos.ch, tabSize);
        spaces.push(spaceStr(tabSize - col % tabSize));
      }
      # 替换选定文本为计算出的空格
      cm.replaceSelections(spaces);
    },
    # 默认的制表符行为，如果有选定文本则增加缩进，否则插入制表符
    defaultTab: function (cm) {
      if (cm.somethingSelected()) { cm.indentSelection("add"); }
      else { cm.execCommand("insertTab"); }
    },
    # 交换光标位置左右两个字符的内容，并将光标移动到交换后的字符后面
    #
    # 不考虑换行符
    # 不扫描光标上方超过一行的内容
    # 空行不做任何操作
    # 非空选定文本不做任何操作
    // 定义一个名为transposeChars的函数，参数为cm，返回一个操作函数
    transposeChars: function (cm) { return runInOp(cm, function () {
      // 获取当前光标所在的选区范围
      var ranges = cm.listSelections(), newSel = [];
      // 遍历选区范围
      for (var i = 0; i < ranges.length; i++) {
        // 如果选区不为空，则跳过
        if (!ranges[i].empty()) { continue }
        // 获取当前光标位置和所在行的文本内容
        var cur = ranges[i].head, line = getLine(cm.doc, cur.line).text;
        // 如果行内容不为空
        if (line) {
          // 如果光标在行末尾
          if (cur.ch == line.length) { cur = new Pos(cur.line, cur.ch - 1); }
          // 如果光标不在行首
          if (cur.ch > 0) {
            // 将光标位置向后移动一位，然后交换光标位置处的两个字符
            cur = new Pos(cur.line, cur.ch + 1);
            cm.replaceRange(line.charAt(cur.ch - 1) + line.charAt(cur.ch - 2),
                            Pos(cur.line, cur.ch - 2), cur, "+transpose");
          } else if (cur.line > cm.doc.first) {
            // 获取上一行的文本内容
            var prev = getLine(cm.doc, cur.line - 1).text;
            // 如果上一行内容不为空
            if (prev) {
              // 将光标移动到当前行的行首，然后交换当前行的第一个字符和上一行的最后一个字符
              cur = new Pos(cur.line, 1);
              cm.replaceRange(line.charAt(0) + cm.doc.lineSeparator() +
                              prev.charAt(prev.length - 1),
                              Pos(cur.line - 1, prev.length - 1), cur, "+transpose");
            }
          }
        }
        // 将新的光标位置添加到新选区数组中
        newSel.push(new Range(cur, cur));
      }
      // 设置编辑器的选区为新选区数组中的值
      cm.setSelections(newSel);
    }); },
    // 定义一个名为newlineAndIndent的函数，参数为cm，返回一个操作函数
    newlineAndIndent: function (cm) { return runInOp(cm, function () {
      // 获取当前光标的选区
      var sels = cm.listSelections();
      // 遍历选区
      for (var i = sels.length - 1; i >= 0; i--)
        { cm.replaceRange(cm.doc.lineSeparator(), sels[i].anchor, sels[i].head, "+input"); }
      // 重新获取当前光标的选区
      sels = cm.listSelections();
      // 遍历选区
      for (var i$1 = 0; i$1 < sels.length; i$1++)
        { cm.indentLine(sels[i$1].from().line, null, true); }
      // 确保光标可见
      ensureCursorVisible(cm);
    }); },
    // 定义一个名为openLine的函数，参数为cm，返回一个操作函数，用于在当前光标位置插入换行符
    openLine: function (cm) { return cm.replaceSelection("\n", "start"); },
    // 定义一个名为toggleOverwrite的函数，参数为cm，返回一个操作函数，用于切换覆盖模式
    toggleOverwrite: function (cm) { return cm.toggleOverwrite(); }
  };

  // 定义一个名为lineStart的函数，参数为cm和lineN，返回当前行的起始位置
  function lineStart(cm, lineN) {
    // 获取当前行的文本内容
    var line = getLine(cm.doc, lineN);
    // 获取当前行的可视行
    var visual = visualLine(line);
    // 如果可视行和当前行不一致，则更新行号
    if (visual != line) { lineN = lineNo(visual); }
    // 返回当前行的起始位置
    return endOfLine(true, cm, visual, lineN, 1)
  }
  // 定义一个名为lineEnd的函数，参数为cm和lineN，返回当前行的结束位置
  function lineEnd(cm, lineN) {
    // 获取指定行的文本内容
    var line = getLine(cm.doc, lineN);
    // 获取可视行的结束位置
    var visual = visualLineEnd(line);
    // 如果可视行和原始行不一致，则更新行号为可视行的行号
    if (visual != line) { lineN = lineNo(visual); }
    // 返回指定行的结束位置
    return endOfLine(true, cm, line, lineN, -1)
    }
    
    // 获取指定位置所在行的起始位置
    function lineStartSmart(cm, pos) {
      // 获取指定行的起始位置
      var start = lineStart(cm, pos.line);
      // 获取指定行的文本内容
      var line = getLine(cm.doc, start.line);
      // 获取文本内容的显示顺序
      var order = getOrder(line, cm.doc.direction);
      // 如果没有显示顺序或者显示顺序的级别为0
      if (!order || order[0].level == 0) {
        // 获取第一个非空白字符的位置
        var firstNonWS = Math.max(start.ch, line.text.search(/\S/));
        // 判断是否在空白字符中，如果是则返回行的起始位置，否则返回第一个非空白字符的位置
        var inWS = pos.line == start.line && pos.ch <= firstNonWS && pos.ch;
        return Pos(start.line, inWS ? 0 : firstNonWS, start.sticky)
      }
      // 返回行的起始位置
      return start
    }
    
    // 执行绑定到按键的处理程序
    function doHandleBinding(cm, bound, dropShift) {
      if (typeof bound == "string") {
        bound = commands[bound];
        if (!bound) { return false }
      }
      // 确保已经读取了先前的输入，以便处理程序看到文档的一致视图
      cm.display.input.ensurePolled();
      var prevShift = cm.display.shift, done = false;
      try {
        if (cm.isReadOnly()) { cm.state.suppressEdits = true; }
        if (dropShift) { cm.display.shift = false; }
        done = bound(cm) != Pass;
      } finally {
        cm.display.shift = prevShift;
        cm.state.suppressEdits = false;
      }
      return done
    }
    
    // 查找编辑器的按键绑定
    function lookupKeyForEditor(cm, name, handle) {
      for (var i = 0; i < cm.state.keyMaps.length; i++) {
        var result = lookupKey(name, cm.state.keyMaps[i], handle, cm);
        if (result) { return result }
      }
      return (cm.options.extraKeys && lookupKey(name, cm.options.extraKeys, handle, cm))
        || lookupKey(name, cm.options.keyMap, handle, cm)
    }
    
    // 注意，尽管名称是这样，但此函数也用于检查绑定的鼠标点击
    var stopSeq = new Delayed;
    
    // 分发按键事件
    function dispatchKey(cm, name, e, handle) {
      var seq = cm.state.keySeq;
    # 如果按键序列不为空
    if (seq) {
      # 如果按键是修饰键，则返回"handled"
      if (isModifierKey(name)) { return "handled" }
      # 如果按键以单引号结尾
      if (/\'$/.test(name))
        { cm.state.keySeq = null; }
      # 否则
      else
        { stopSeq.set(50, function () {
          # 如果按键序列与当前按键相同
          if (cm.state.keySeq == seq) {
            cm.state.keySeq = null;
            cm.display.input.reset();
          }
        }); }
      # 如果按键分发成功，则返回true
      if (dispatchKeyInner(cm, seq + " " + name, e, handle)) { return true }
    }
    # 返回按键分发结果
    return dispatchKeyInner(cm, name, e, handle)
  }

  # 处理按键分发
  function dispatchKeyInner(cm, name, e, handle) {
    # 查找编辑器中按键的结果
    var result = lookupKeyForEditor(cm, name, handle);

    # 如果结果为"multi"，则设置按键序列
    if (result == "multi")
      { cm.state.keySeq = name; }
    # 如果结果为"handled"，则触发"keyHandled"事件
    if (result == "handled")
      { signalLater(cm, "keyHandled", cm, name, e); }

    # 如果结果为"handled"或"multi"，则阻止默认事件并重启光标闪烁
    if (result == "handled" || result == "multi") {
      e_preventDefault(e);
      restartBlink(cm);
    }

    # 返回结果是否为真
    return !!result
  }

  # 处理来自keydown事件的按键
  function handleKeyBinding(cm, e) {
    # 获取按键名称
    var name = keyName(e, true);
    # 如果按键名称不存在，则返回false
    if (!name) { return false }

    # 如果按下Shift键且按键序列为空
    if (e.shiftKey && !cm.state.keySeq) {
      # 首先尝试解析完整名称（包括'Shift-'）。如果失败，则查看是否有绑定到不带'Shift-'的按键名称的光标移动命令（以'go'开头）
      return dispatchKey(cm, "Shift-" + name, e, function (b) { return doHandleBinding(cm, b, true); })
          || dispatchKey(cm, name, e, function (b) {
               if (typeof b == "string" ? /^go[A-Z]/.test(b) : b.motion)
                 { return doHandleBinding(cm, b) }
             })
    } else {
      # 否则，直接分发按键
      return dispatchKey(cm, name, e, function (b) { return doHandleBinding(cm, b); })
    }
  }

  # 处理来自keypress事件的按键
  function handleCharBinding(cm, e, ch) {
    # 分发按键
    return dispatchKey(cm, "'" + ch + "'", e, function (b) { return doHandleBinding(cm, b, true); })
  }

  # 上一个停止的按键
  var lastStoppedKey = null;
  # 键盘按下事件处理函数
  function onKeyDown(e) {
    var cm = this;
    # 如果事件目标不是输入框，则返回
    if (e.target && e.target != cm.display.input.getField()) { return }
    # 设置当前操作的焦点为活动元素
    cm.curOp.focus = activeElt();
    # 如果触发了 CodeMirror 的 DOM 事件，则返回
    if (signalDOMEvent(cm, e)) { return }
    # 处理 IE 对于 escape 键的特殊情况
    if (ie && ie_version < 11 && e.keyCode == 27) { e.returnValue = false; }
    # 获取键盘按键的代码
    var code = e.keyCode;
    # 设置是否按下了 Shift 键
    cm.display.shift = code == 16 || e.shiftKey;
    # 处理键盘绑定
    var handled = handleKeyBinding(cm, e);
    # 处理 Opera 浏览器的特殊情况
    if (presto) {
      lastStoppedKey = handled ? code : null;
      # Opera 没有 cut 事件... 尝试至少捕获键盘组合
      if (!handled && code == 88 && !hasCopyEvent && (mac ? e.metaKey : e.ctrlKey))
        { cm.replaceSelection("", null, "cut"); }
    }
    # 处理 Gecko 浏览器的特殊情况
    if (gecko && !mac && !handled && code == 46 && e.shiftKey && !e.ctrlKey && document.execCommand)
      { document.execCommand("cut"); }

    # 当按下 Alt 键时，将鼠标变成十字线（仅在 Mac 上）
    if (code == 18 && !/\bCodeMirror-crosshair\b/.test(cm.display.lineDiv.className))
      { showCrossHair(cm); }
  }

  # 显示十字线
  function showCrossHair(cm) {
    var lineDiv = cm.display.lineDiv;
    addClass(lineDiv, "CodeMirror-crosshair");

    function up(e) {
      if (e.keyCode == 18 || !e.altKey) {
        rmClass(lineDiv, "CodeMirror-crosshair");
        off(document, "keyup", up);
        off(document, "mouseover", up);
      }
    }
    on(document, "keyup", up);
    on(document, "mouseover", up);
  }

  # 处理键盘抬起事件
  function onKeyUp(e) {
    if (e.keyCode == 16) { this.doc.sel.shift = false; }
    signalDOMEvent(this, e);
  }

  # 处理键盘按下事件
  function onKeyPress(e) {
    var cm = this;
    # 如果事件的目标不是 CodeMirror 的输入框，则返回
    if (e.target && e.target != cm.display.input.getField()) { return }
    # 如果事件在小部件中或者触发了 CodeMirror 的 DOM 事件，或者按下了 Ctrl 键但没有按下 Alt 键，或者在 Mac 上按下了 Meta 键，则返回
    if (eventInWidget(cm.display, e) || signalDOMEvent(cm, e) || e.ctrlKey && !e.altKey || mac && e.metaKey) { return }
    # 获取键盘按键的代码和字符代码
    var keyCode = e.keyCode, charCode = e.charCode;
    # 处理 Presto 浏览器的特殊情况
    if (presto && keyCode == lastStoppedKey) {lastStoppedKey = null; e_preventDefault(e); return}
    # 处理键盘绑定
    if ((presto && (!e.which || e.which < 10)) && handleKeyBinding(cm, e)) { return }
    var ch = String.fromCharCode(charCode == null ? keyCode : charCode);
    // 某些浏览器会对退格键触发键盘按键事件
    if (ch == "\x08") { return }
    // 如果字符绑定处理函数返回 true，则返回
    if (handleCharBinding(cm, e, ch)) { return }
    // 触发输入框的键盘按键事件
    cm.display.input.onKeyPress(e);
  }

  // 双击延迟时间
  var DOUBLECLICK_DELAY = 400;

  // 保存上一次点击的时间、位置和按钮信息
  var PastClick = function(time, pos, button) {
    this.time = time;
    this.pos = pos;
    this.button = button;
  };

  // 比较当前点击事件和上一次点击事件，判断是否为双击
  PastClick.prototype.compare = function (time, pos, button) {
    return this.time + DOUBLECLICK_DELAY > time &&
      cmp(pos, this.pos) == 0 && button == this.button
  };

  // 上一次单击和双击的点击事件
  var lastClick, lastDoubleClick;
  function clickRepeat(pos, button) {
    var now = +new Date;
    // 如果上一次是双击事件且符合条件，则返回三连击
    if (lastDoubleClick && lastDoubleClick.compare(now, pos, button)) {
      lastClick = lastDoubleClick = null;
      return "triple"
    // 如果上一次是单击事件且符合条件，则返回双击
    } else if (lastClick && lastClick.compare(now, pos, button)) {
      lastDoubleClick = new PastClick(now, pos, button);
      lastClick = null;
      return "double"
    // 否则返回单击
    } else {
      lastClick = new PastClick(now, pos, button);
      lastDoubleClick = null;
      return "single"
    }
  }

  // 鼠标按下事件可能是单击、双击、三连击、开始选择拖拽、开始文本拖拽、新光标（ctrl+单击）、矩形拖拽（alt+拖拽）或者 xwin 中键粘贴。也可能是点击滚动条或小部件等不应干预的内容。
  function onMouseDown(e) {
    var cm = this, display = cm.display;
    // 如果触发了 DOM 事件或者触摸事件，则返回
    if (signalDOMEvent(cm, e) || display.activeTouch && display.input.supportsTouch()) { return }
    // 确保输入框已经轮询
    display.input.ensurePolled();
    // 设置 shift 键状态
    display.shift = e.shiftKey;

    // 如果点击在小部件上，则返回
    if (eventInWidget(display, e)) {
      if (!webkit) {
        // 短暂关闭可拖动性，以允许小部件进行正常拖动操作
        display.scroller.draggable = false;
        setTimeout(function () { return display.scroller.draggable = true; }, 100);
      }
      return
    }
    // 如果点击在行号区域，则返回
    if (clickInGutter(cm, e)) { return }
    # 获取鼠标位置
    var pos = posFromMouse(cm, e), button = e_button(e), repeat = pos ? clickRepeat(pos, button) : "single";
    # 窗口获取焦点
    window.focus();

    // #3261: 确保不会开始第二次选择
    if (button == 1 && cm.state.selectingText)
      { cm.state.selectingText(e); }

    # 处理映射的按钮
    if (pos && handleMappedButton(cm, button, pos, repeat, e)) { return }

    # 处理鼠标左键
    if (button == 1) {
      if (pos) { leftButtonDown(cm, pos, repeat, e); }
      else if (e_target(e) == display.scroller) { e_preventDefault(e); }
    } 
    # 处理鼠标中键
    else if (button == 2) {
      if (pos) { extendSelection(cm.doc, pos); }
      # 设置延迟聚焦到输入框
      setTimeout(function () { return display.input.focus(); }, 20);
    } 
    # 处理鼠标右键
    else if (button == 3) {
      if (captureRightClick) { cm.display.input.onContextMenu(e); }
      else { delayBlurEvent(cm); }
    }
  }

  # 处理映射的按钮
  function handleMappedButton(cm, button, pos, repeat, event) {
    var name = "Click";
    if (repeat == "double") { name = "Double" + name; }
    else if (repeat == "triple") { name = "Triple" + name; }
    name = (button == 1 ? "Left" : button == 2 ? "Middle" : "Right") + name;

    return dispatchKey(cm,  addModifierNames(name, event), event, function (bound) {
      if (typeof bound == "string") { bound = commands[bound]; }
      if (!bound) { return false }
      var done = false;
      try {
        if (cm.isReadOnly()) { cm.state.suppressEdits = true; }
        done = bound(cm, pos) != Pass;
      } finally {
        cm.state.suppressEdits = false;
      }
      return done
    })
  }

  # 配置鼠标
  function configureMouse(cm, repeat, event) {
    var option = cm.getOption("configureMouse");
    var value = option ? option(cm, repeat, event) : {};
    if (value.unit == null) {
      var rect = chromeOS ? event.shiftKey && event.metaKey : event.altKey;
      value.unit = rect ? "rectangle" : repeat == "single" ? "char" : repeat == "double" ? "word" : "line";
    }
    if (value.extend == null || cm.doc.extend) { value.extend = cm.doc.extend || event.shiftKey; }
    # 如果 value.addNew 为 null，则根据操作系统类型和事件按键状态设置其值
    if (value.addNew == null) { value.addNew = mac ? event.metaKey : event.ctrlKey; }
    # 如果 value.moveOnDrag 为 null，则根据操作系统类型和事件按键状态设置其值
    if (value.moveOnDrag == null) { value.moveOnDrag = !(mac ? event.altKey : event.ctrlKey); }
    # 返回更新后的 value 对象
    return value
  }

  # 处理鼠标左键按下事件
  function leftButtonDown(cm, pos, repeat, event) {
    # 如果是 IE 浏览器，则延迟执行 ensureFocus 函数
    if (ie) { setTimeout(bind(ensureFocus, cm), 0); }
    # 否则，设置当前操作的焦点为活动元素
    else { cm.curOp.focus = activeElt(); }

    # 配置鼠标行为
    var behavior = configureMouse(cm, repeat, event);

    # 获取当前选择的文本
    var sel = cm.doc.sel, contained;
    # 如果允许拖放并且不是只读模式，并且是单击事件，并且当前位置包含选择的文本
    # 并且当前位置在选择文本的起始位置之前或者在其后，并且鼠标位置在文本的左侧或右侧
    if (cm.options.dragDrop && dragAndDrop && !cm.isReadOnly() &&
        repeat == "single" && (contained = sel.contains(pos)) > -1 &&
        (cmp((contained = sel.ranges[contained]).from(), pos) < 0 || pos.xRel > 0) &&
        (cmp(contained.to(), pos) > 0 || pos.xRel < 0))
      { leftButtonStartDrag(cm, event, pos, behavior); }
    # 否则，执行文本选择操作
    else
      { leftButtonSelect(cm, event, pos, behavior); }
  }

  # 开始文本拖放操作
  function leftButtonStartDrag(cm, event, pos, behavior) {
    var display = cm.display, moved = false;
    # 定义拖放结束时的操作
    var dragEnd = operation(cm, function (e) {
      # 如果是 Webkit 浏览器，则禁止滚动
      if (webkit) { display.scroller.draggable = false; }
      # 设置拖放状态为 false
      cm.state.draggingText = false;
      # 移除拖放相关事件监听
      off(display.wrapper.ownerDocument, "mouseup", dragEnd);
      off(display.wrapper.ownerDocument, "mousemove", mouseMove);
      off(display.scroller, "dragstart", dragStart);
      off(display.scroller, "drop", dragEnd);
      # 如果没有移动过，则视为点击事件
      if (!moved) {
        e_preventDefault(e);
        # 如果不是添加新的选择，则扩展当前选择
        if (!behavior.addNew)
          { extendSelection(cm.doc, pos, null, null, behavior.extend); }
        # 解决 IE9 和 Chrome 中的焦点问题
        if ((webkit && !safari) || ie && ie_version == 9)
          { setTimeout(function () {display.wrapper.ownerDocument.body.focus({preventScroll: true}); display.input.focus();}, 20); }
        else
          { display.input.focus(); }
      }
    });
    // 定义鼠标移动事件处理函数，判断鼠标是否移动了一定距离
    var mouseMove = function(e2) {
      moved = moved || Math.abs(event.clientX - e2.clientX) + Math.abs(event.clientY - e2.clientY) >= 10;
    };
    // 定义拖拽开始事件处理函数，设置 moved 为 true
    var dragStart = function () { return moved = true; };
    // 如果是 webkit 浏览器，设置 display.scroller.draggable 为 true
    if (webkit) { display.scroller.draggable = true; }
    // 设置 cm.state.draggingText 为 dragEnd
    cm.state.draggingText = dragEnd;
    // 设置 dragEnd.copy 为 !behavior.moveOnDrag
    dragEnd.copy = !behavior.moveOnDrag;
    // 如果是 IE 浏览器，调用 display.scroller.dragDrop() 方法
    if (display.scroller.dragDrop) { display.scroller.dragDrop(); }
    // 监听鼠标抬起事件，触发 dragEnd 函数
    on(display.wrapper.ownerDocument, "mouseup", dragEnd);
    // 监听鼠标移动事件，触发 mouseMove 函数
    on(display.wrapper.ownerDocument, "mousemove", mouseMove);
    // 监听拖拽开始事件，触发 dragStart 函数
    on(display.scroller, "dragstart", dragStart);
    // 监听拖拽结束事件，触发 dragEnd 函数
    on(display.scroller, "drop", dragEnd);

    // 延迟触发模糊事件
    delayBlurEvent(cm);
    // 延迟 20 毫秒后，让 display.input 获取焦点
    setTimeout(function () { return display.input.focus(); }, 20);
  }

  // 根据单位返回对应的范围
  function rangeForUnit(cm, pos, unit) {
    if (unit == "char") { return new Range(pos, pos) }
    if (unit == "word") { return cm.findWordAt(pos) }
    if (unit == "line") { return new Range(Pos(pos.line, 0), clipPos(cm.doc, Pos(pos.line + 1, 0))) }
    var result = unit(cm, pos);
    return new Range(result.from, result.to)
  }

  // 左键选择文本
  function leftButtonSelect(cm, event, start, behavior) {
    var display = cm.display, doc = cm.doc;
    e_preventDefault(event);

    var ourRange, ourIndex, startSel = doc.sel, ranges = startSel.ranges;
    // 如果是添加新的选择范围且不是扩展选择
    if (behavior.addNew && !behavior.extend) {
      ourIndex = doc.sel.contains(start);
      if (ourIndex > -1)
        { ourRange = ranges[ourIndex]; }
      else
        { ourRange = new Range(start, start); }
    } else {
      ourRange = doc.sel.primary();
      ourIndex = doc.sel.primIndex;
    }

    // 如果是矩形选择
    if (behavior.unit == "rectangle") {
      if (!behavior.addNew) { ourRange = new Range(start, start); }
      start = posFromMouse(cm, event, true, true);
      ourIndex = -1;
    } else {
      // 如果不是添加新的选择范围，则根据行为单位和起始位置计算范围
      var range = rangeForUnit(cm, start, behavior.unit);
      // 如果是扩展选择，则根据当前范围的锚点和头部以及扩展行为来扩展范围
      if (behavior.extend)
        { ourRange = extendRange(ourRange, range.anchor, range.head, behavior.extend); }
      // 否则直接使用计算出的范围
      else
        { ourRange = range; }
    }

    // 如果不是添加新的选择范围
    if (!behavior.addNew) {
      // 重置选择范围的索引为0
      ourIndex = 0;
      // 设置文档的选择范围为新的选择范围，并指定选择的方式为鼠标
      setSelection(doc, new Selection([ourRange], 0), sel_mouse);
      // 记录起始选择范围
      startSel = doc.sel;
    } else if (ourIndex == -1) {
      // 如果选择范围的索引为-1，则将索引设置为当前选择范围数组的长度
      ourIndex = ranges.length;
      // 设置文档的选择范围为包含当前选择范围和新的选择范围的规范化选择范围，并指定滚动为false，来源为鼠标
      setSelection(doc, normalizeSelection(cm, ranges.concat([ourRange]), ourIndex),
                   {scroll: false, origin: "*mouse"});
    } else if (ranges.length > 1 && ranges[ourIndex].empty() && behavior.unit == "char" && !behavior.extend) {
      // 如果选择范围数组的长度大于1，并且当前选择范围为空，并且行为单位为字符，并且不是扩展选择
      // 则设置文档的选择范围为规范化选择范围，并指定滚动为false，来源为鼠标
      setSelection(doc, normalizeSelection(cm, ranges.slice(0, ourIndex).concat(ranges.slice(ourIndex + 1)), 0),
                   {scroll: false, origin: "*mouse"});
      // 记录起始选择范围
      startSel = doc.sel;
    } else {
      // 替换文档中的一个选择范围为新的选择范围，并指定选择的方式为鼠标
      replaceOneSelection(doc, ourIndex, ourRange, sel_mouse);
    }

    // 记录最后的位置
    var lastPos = start;
    // 定义一个名为 extendTo 的函数，参数为 pos
    function extendTo(pos) {
      // 如果上一次的位置和当前位置相同，则直接返回
      if (cmp(lastPos, pos) == 0) { return }
      // 更新上一次的位置为当前位置
      lastPos = pos;

      // 如果行为单位为 "rectangle"
      if (behavior.unit == "rectangle") {
        // 定义变量 ranges 和 tabSize
        var ranges = [], tabSize = cm.options.tabSize;
        // 计算起始列和当前位置的列数
        var startCol = countColumn(getLine(doc, start.line).text, start.ch, tabSize);
        var posCol = countColumn(getLine(doc, pos.line).text, pos.ch, tabSize);
        var left = Math.min(startCol, posCol), right = Math.max(startCol, posCol);
        // 遍历行数，计算选中的范围
        for (var line = Math.min(start.line, pos.line), end = Math.min(cm.lastLine(), Math.max(start.line, pos.line));
             line <= end; line++) {
          var text = getLine(doc, line).text, leftPos = findColumn(text, left, tabSize);
          if (left == right)
            { ranges.push(new Range(Pos(line, leftPos), Pos(line, leftPos))); }
          else if (text.length > leftPos)
            { ranges.push(new Range(Pos(line, leftPos), Pos(line, findColumn(text, right, tabSize))); }
        }
        // 如果没有选中范围，则默认选中起始位置
        if (!ranges.length) { ranges.push(new Range(start, start)); }
        // 设置选中范围并滚动到当前位置
        setSelection(doc, normalizeSelection(cm, startSel.ranges.slice(0, ourIndex).concat(ranges), ourIndex),
                     {origin: "*mouse", scroll: false});
        cm.scrollIntoView(pos);
      } else {
        // 如果行为单位不是 "rectangle"
        var oldRange = ourRange;
        // 获取当前位置的范围
        var range = rangeForUnit(cm, pos, behavior.unit);
        var anchor = oldRange.anchor, head;
        // 根据范围的 anchor 和 head 更新选中范围
        if (cmp(range.anchor, anchor) > 0) {
          head = range.head;
          anchor = minPos(oldRange.from(), range.anchor);
        } else {
          head = range.anchor;
          anchor = maxPos(oldRange.to(), range.head);
        }
        var ranges$1 = startSel.ranges.slice(0);
        ranges$1[ourIndex] = bidiSimplify(cm, new Range(clipPos(doc, anchor), head));
        // 设置选中范围
        setSelection(doc, normalizeSelection(cm, ranges$1, ourIndex), sel_mouse);
      }
    }

    // 获取编辑器的尺寸信息
    var editorSize = display.wrapper.getBoundingClientRect();
    // 用于确保超时重试不会在另一个 extend 时触发
    // 计数器，用于跟踪鼠标移动事件的次数
    var counter = 0;
    
    // 扩展选区的函数
    function extend(e) {
      // 当前计数
      var curCount = ++counter;
      // 获取鼠标位置对应的编辑器中的位置
      var cur = posFromMouse(cm, e, true, behavior.unit == "rectangle");
      // 如果位置不存在，则返回
      if (!cur) { return }
      // 如果当前位置与上一个位置不同
      if (cmp(cur, lastPos) != 0) {
        // 设置编辑器操作的焦点
        cm.curOp.focus = activeElt();
        // 扩展选区到当前位置
        extendTo(cur);
        // 获取可见的行数范围
        var visible = visibleLines(display, doc);
        // 如果当前行超出可见范围，则延迟执行扩展操作
        if (cur.line >= visible.to || cur.line < visible.from)
          { setTimeout(operation(cm, function () {if (counter == curCount) { extend(e); }}), 150); }
      } else {
        // 如果鼠标位置在编辑器上方或下方，则滚动编辑器
        var outside = e.clientY < editorSize.top ? -20 : e.clientY > editorSize.bottom ? 20 : 0;
        // 如果存在滚动操作，则延迟执行滚动操作
        if (outside) { setTimeout(operation(cm, function () {
          if (counter != curCount) { return }
          display.scroller.scrollTop += outside;
          extend(e);
        }), 50); }
      }
    }
    
    // 完成选区操作的函数
    function done(e) {
      // 取消选区状态
      cm.state.selectingText = false;
      // 重置计数器
      counter = Infinity;
      // 如果 e 为 null 或 undefined，则取消选区
      if (e) {
        e_preventDefault(e);
        display.input.focus();
      }
      // 移除鼠标移动和鼠标释放事件监听
      off(display.wrapper.ownerDocument, "mousemove", move);
      off(display.wrapper.ownerDocument, "mouseup", up);
      doc.history.lastSelOrigin = null;
    }
    
    // 鼠标移动事件监听
    var move = operation(cm, function (e) {
      // 如果鼠标按钮为0或者没有按下鼠标，则完成选区操作
      if (e.buttons === 0 || !e_button(e)) { done(e); }
      else { extend(e); }
    });
    
    // 鼠标释放事件监听
    var up = operation(cm, done);
    // 设置选区状态为释放状态
    cm.state.selectingText = up;
    // 添加鼠标移动事件监听
    on(display.wrapper.ownerDocument, "mousemove", move);
    # 在显示器上注册鼠标抬起事件的监听器，当事件发生时调用 up 函数
    on(display.wrapper.ownerDocument, "mouseup", up);
  }

  // 当鼠标选择文本时，根据头部的可视位置调整锚点到正确的双向跳转边界
  function bidiSimplify(cm, range) {
    // 获取锚点和头部的位置信息
    var anchor = range.anchor;
    var head = range.head;
    // 获取锚点所在行的文本
    var anchorLine = getLine(cm.doc, anchor.line);
    // 如果锚点和头部位置相同且粘性相同，则直接返回范围
    if (cmp(anchor, head) == 0 && anchor.sticky == head.sticky) { return range }
    // 获取锚点所在行的双向文本顺序信息
    var order = getOrder(anchorLine);
    // 如果没有双向文本顺序信息，则直接返回范围
    if (!order) { return range }
    // 获取锚点所在行的双向文本部分索引
    var index = getBidiPartAt(order, anchor.ch, anchor.sticky), part = order[index];
    // 如果锚点不在部分的起始或结束位置，则直接返回范围
    if (part.from != anchor.ch && part.to != anchor.ch) { return range }
    // 计算头部相对于锚点的可视位置（<0 表示在左侧，>0 表示在右侧）
    var leftSide;
    if (head.line != anchor.line) {
      leftSide = (head.line - anchor.line) * (cm.doc.direction == "ltr" ? 1 : -1) > 0;
    } else {
      var headIndex = getBidiPartAt(order, head.ch, head.sticky);
      var dir = headIndex - index || (head.ch - anchor.ch) * (part.level == 1 ? -1 : 1);
      if (headIndex == boundary - 1 || headIndex == boundary)
        { leftSide = dir < 0; }
      else
        { leftSide = dir > 0; }
    }
    // 根据头部位置确定使用的双向文本部分
    var usePart = order[boundary + (leftSide ? -1 : 0)];
    var from = leftSide == (usePart.level == 1);
    var ch = from ? usePart.from : usePart.to, sticky = from ? "after" : "before";
    // 如果锚点位置和粘性与计算得到的位置相同，则返回范围，否则返回新的范围
    return anchor.ch == ch && anchor.sticky == sticky ? range : new Range(new Pos(anchor.line, ch, sticky), head)
  }

  // 确定事件是否发生在行号区域，并触发相应的事件处理程序
  function gutterEvent(cm, e, type, prevent) {
    var mX, mY;
    // 如果是触摸事件，则获取触摸点的坐标
    if (e.touches) {
      mX = e.touches[0].clientX;
      mY = e.touches[0].clientY;
  } else {
    // 如果鼠标事件中包含客户端坐标，则将其赋值给 mX 和 mY
    try { mX = e.clientX; mY = e.clientY; }
    // 如果出现异常，则返回 false
    catch(e$1) { return false }
  }
  // 如果鼠标的横坐标大于编辑器的 gutter 右边界，则返回 false
  if (mX >= Math.floor(cm.display.gutters.getBoundingClientRect().right)) { return false }
  // 如果需要阻止默认行为，则调用 e_preventDefault 函数
  if (prevent) { e_preventDefault(e); }

  // 获取编辑器的 display 对象
  var display = cm.display;
  // 获取编辑器的行框的位置信息
  var lineBox = display.lineDiv.getBoundingClientRect();

  // 如果鼠标的纵坐标大于行框的底部，或者没有指定类型的事件处理器，则返回 e_defaultPrevented(e)
  if (mY > lineBox.bottom || !hasHandler(cm, type)) { return e_defaultPrevented(e) }
  // 将鼠标的纵坐标减去行框的顶部和 display 的视图偏移量
  mY -= lineBox.top - display.viewOffset;

  // 遍历编辑器的 gutterSpecs 数组
  for (var i = 0; i < cm.display.gutterSpecs.length; ++i) {
    // 获取指定索引的 gutter 元素
    var g = display.gutters.childNodes[i];
    // 如果 gutter 存在且其右边界大于等于 mX，则执行以下操作
    if (g && g.getBoundingClientRect().right >= mX) {
      // 获取鼠标所在行的行号
      var line = lineAtHeight(cm.doc, mY);
      // 获取指定索引的 gutter 规格
      var gutter = cm.display.gutterSpecs[i];
      // 触发指定类型的事件处理器，并返回 e_defaultPrevented(e)
      signal(cm, type, cm, line, gutter.className, e);
      return e_defaultPrevented(e)
    }
  }
}

// 处理在 gutter 区域的点击事件
function clickInGutter(cm, e) {
  return gutterEvent(cm, e, "gutterClick", true)
}

// 上下文菜单处理

// 为了使上下文菜单生效，我们需要暂时取消隐藏文本区域（尽可能不显眼），以便右键单击生效
function onContextMenu(cm, e) {
  // 如果事件发生在小部件内部或者在 gutter 区域，则返回
  if (eventInWidget(cm.display, e) || contextMenuInGutter(cm, e)) { return }
  // 如果触发了 contextmenu 事件，则返回
  if (signalDOMEvent(cm, e, "contextmenu")) { return }
  // 如果不捕获右键单击事件，则调用 cm.display.input.onContextMenu(e)
  if (!captureRightClick) { cm.display.input.onContextMenu(e); }
}

// 判断上下文菜单是否在 gutter 区域内
function contextMenuInGutter(cm, e) {
  // 如果没有指定类型的 gutterContextMenu 事件处理器，则返回 false
  if (!hasHandler(cm, "gutterContextMenu")) { return false }
  // 返回 gutterEvent 函数的执行结果
  return gutterEvent(cm, e, "gutterContextMenu", false)
}

// 主题更改处理
function themeChanged(cm) {
  // 替换编辑器 wrapper 的类名，以更新主题
  cm.display.wrapper.className = cm.display.wrapper.className.replace(/\s*cm-s-\S+/g, "") +
    cm.options.theme.replace(/(^|\s)\s*/g, " cm-s-");
  // 清除缓存
  clearCaches(cm);
}

// 初始化对象
var Init = {toString: function(){return "CodeMirror.Init"}};

// 默认选项
var defaults = {};
// 选项处理器
var optionHandlers = {};

// 定义选项
function defineOptions(CodeMirror) {
  // 获取 CodeMirror 的选项处理器
  var optionHandlers = CodeMirror.optionHandlers;
    # 定义一个函数，用于设置 CodeMirror 的默认选项值，并且可以在初始化时执行处理函数
    function option(name, deflt, handle, notOnInit) {
      # 将默认值设置到 CodeMirror 的默认选项中
      CodeMirror.defaults[name] = deflt;
      # 如果有处理函数，则将处理函数添加到选项处理函数列表中
      if (handle) { optionHandlers[name] =
        # 如果不是初始化时执行处理函数，则在旧值不是初始化值时执行处理函数
        notOnInit ? function (cm, val, old) {if (old != Init) { handle(cm, val, old); }} : handle; }
    }

    # 定义一个函数，用于设置 CodeMirror 的选项
    CodeMirror.defineOption = option;

    # 当没有旧值时传递给选项处理函数的值
    CodeMirror.Init = Init;

    # 这两个选项在初始化时从构造函数中调用，因为它们必须在编辑器启动之前初始化
    option("value", "", function (cm, val) { return cm.setValue(val); }, true);
    option("mode", null, function (cm, val) {
      # 设置文档的模式选项，并加载模式
      cm.doc.modeOption = val;
      loadMode(cm);
    }, true);

    option("indentUnit", 2, loadMode, true);
    option("indentWithTabs", false);
    option("smartIndent", true);
    option("tabSize", 4, function (cm) {
      # 重置模式状态、清除缓存并注册变化
      resetModeState(cm);
      clearCaches(cm);
      regChange(cm);
    }, true);

    option("lineSeparator", null, function (cm, val) {
      # 设置文档的行分隔符，并根据新的分隔符重新计算断点位置
      cm.doc.lineSep = val;
      if (!val) { return }
      var newBreaks = [], lineNo = cm.doc.first;
      cm.doc.iter(function (line) {
        for (var pos = 0;;) {
          var found = line.text.indexOf(val, pos);
          if (found == -1) { break }
          pos = found + val.length;
          newBreaks.push(Pos(lineNo, found));
        }
        lineNo++;
      });
      for (var i = newBreaks.length - 1; i >= 0; i--)
        { replaceRange(cm.doc, val, newBreaks[i], Pos(newBreaks[i].line, newBreaks[i].ch + val.length)); }
    });
    option("specialChars", /[\u0000-\u001f\u007f-\u009f\u00ad\u061c\u200b-\u200c\u200e\u200f\u2028\u2029\ufeff\ufff9-\ufffc]/g, function (cm, val, old) {
      # 设置特殊字符的正则表达式，并在旧值不是初始化值时刷新编辑器
      cm.state.specialChars = new RegExp(val.source + (val.test("\t") ? "" : "|\t"), "g");
      if (old != Init) { cm.refresh(); }
    });
    option("specialCharPlaceholder", defaultSpecialCharPlaceholder, function (cm) { return cm.refresh(); }, true);
    option("electricChars", true);
    // 设置输入框样式，如果是移动端则使用 contenteditable，否则使用 textarea
    option("inputStyle", mobile ? "contenteditable" : "textarea", function () {
      // 如果编辑器正在运行，则抛出错误
      throw new Error("inputStyle can not (yet) be changed in a running editor") // FIXME
    }, true);
    // 设置拼写检查为 false
    option("spellcheck", false, function (cm, val) { return cm.getInputField().spellcheck = val; }, true);
    // 设置自动更正为 false
    option("autocorrect", false, function (cm, val) { return cm.getInputField().autocorrect = val; }, true);
    // 设置自动大写为 false
    option("autocapitalize", false, function (cm, val) { return cm.getInputField().autocapitalize = val; }, true);
    // 如果不是 Windows 系统，则设置 rtlMoveVisually 为 true
    option("rtlMoveVisually", !windows);
    // 设置在整行更新之前先更新整个行
    option("wholeLineUpdateBefore", true);

    // 设置主题为默认主题
    option("theme", "default", function (cm) {
      // 主题改变时触发主题改变函数和更新 gutter 函数
      themeChanged(cm);
      updateGutters(cm);
    }, true);
    // 设置键盘映射为默认
    option("keyMap", "default", function (cm, val, old) {
      // 获取新旧键盘映射
      var next = getKeyMap(val);
      var prev = old != Init && getKeyMap(old);
      // 如果有旧的键盘映射并且有 detach 函数，则执行 detach 函数
      if (prev && prev.detach) { prev.detach(cm, next); }
      // 如果有新的键盘映射并且有 attach 函数，则执行 attach 函数
      if (next.attach) { next.attach(cm, prev || null); }
    });
    // 设置额外按键为 null
    option("extraKeys", null);
    // 设置鼠标配置为 null
    option("configureMouse", null);

    // 设置是否自动换行为 false，当改变时触发换行改变函数
    option("lineWrapping", false, wrappingChanged, true);
    // 设置 gutter 为一个空数组，当改变时触发更新 gutter 函数
    option("gutters", [], function (cm, val) {
      cm.display.gutterSpecs = getGutters(val, cm.options.lineNumbers);
      updateGutters(cm);
    }, true);
    // 设置固定 gutter 为 true，当改变时更新 gutter 样式并刷新编辑器
    option("fixedGutter", true, function (cm, val) {
      cm.display.gutters.style.left = val ? compensateForHScroll(cm.display) + "px" : "0";
      cm.refresh();
    }, true);
    // 设置 gutter 是否覆盖在滚动条旁边为 false，当改变时更新滚动条
    option("coverGutterNextToScrollbar", false, function (cm) { return updateScrollbars(cm); }, true);
    // 设置滚动条样式为原生样式，当改变时初始化滚动条并更新滚动条位置
    option("scrollbarStyle", "native", function (cm) {
      initScrollbars(cm);
      updateScrollbars(cm);
      cm.display.scrollbars.setScrollTop(cm.doc.scrollTop);
      cm.display.scrollbars.setScrollLeft(cm.doc.scrollLeft);
    }, true);
    // 设置是否显示行号为 false，当改变时更新 gutter
    option("lineNumbers", false, function (cm, val) {
      cm.display.gutterSpecs = getGutters(cm.options.gutters, val);
      updateGutters(cm);
    }, true);
    // 设置第一行行号为 1，当改变时更新 gutter
    option("firstLineNumber", 1, updateGutters, true);
    # 设置行号格式化函数，将整数转换为字符串
    option("lineNumberFormatter", function (integer) { return integer; }, updateGutters, true);
    # 设置在选择文本时是否显示光标
    option("showCursorWhenSelecting", false, updateSelection, true);

    # 在右键菜单中重置选择
    option("resetSelectionOnContextMenu", true);
    # 设置是否按行复制/剪切
    option("lineWiseCopyCut", true);
    # 设置每次选择可以粘贴的行数
    option("pasteLinesPerSelection", true);
    # 设置选择是否可以相邻
    option("selectionsMayTouch", false);

    # 设置是否只读，如果是"nocursor"则失去焦点
    option("readOnly", false, function (cm, val) {
      if (val == "nocursor") {
        onBlur(cm);
        cm.display.input.blur();
      }
      cm.display.input.readOnlyChanged(val);
    });

    # 设置屏幕阅读器标签
    option("screenReaderLabel", null, function (cm, val) {
      val = (val === '') ? null : val;
      cm.display.input.screenReaderLabelChanged(val);
    });

    # 设置是否禁用输入
    option("disableInput", false, function (cm, val) {if (!val) { cm.display.input.reset(); }}, true);
    # 设置是否启用拖放
    option("dragDrop", true, dragDropChanged);
    # 设置允许拖放的文件类型
    option("allowDropFileTypes", null);

    # 设置光标闪烁速率
    option("cursorBlinkRate", 530);
    # 设置光标滚动边距
    option("cursorScrollMargin", 0);
    # 设置光标高度
    option("cursorHeight", 1, updateSelection, true);
    # 设置每行的单个光标高度
    option("singleCursorHeightPerLine", true, updateSelection, true);
    # 设置工作时间
    option("workTime", 100);
    # 设置工作延迟
    option("workDelay", 100);
    # 设置是否展开跨度
    option("flattenSpans", true, resetModeState, true);
    # 设置是否添加模式类
    option("addModeClass", false, resetModeState, true);
    # 设置轮询间隔
    option("pollInterval", 100);
    # 设置撤销深度
    option("undoDepth", 200, function (cm, val) { return cm.doc.history.undoDepth = val; });
    # 设置历史事件延迟
    option("historyEventDelay", 1250);
    # 设置视口边距
    option("viewportMargin", 10, function (cm) { return cm.refresh(); }, true);
    # 设置最大高亮长度
    option("maxHighlightLength", 10000, resetModeState, true);
    # 设置输入是否随光标移动
    option("moveInputWithCursor", true, function (cm, val) {
      if (!val) { cm.display.input.resetPosition(); }
    });

    # 设置tab键索引
    option("tabindex", null, function (cm, val) { return cm.display.input.getField().tabIndex = val || ""; });
    # 设置自动聚焦
    option("autofocus", null);
    # 设置文本方向
    option("direction", "ltr", function (cm, val) { return cm.doc.setDirection(val); }, true);
    // 设置选项中的 "phrases" 为 null
    option("phrases", null);
  }

  // 当拖拽操作改变时触发的函数
  function dragDropChanged(cm, value, old) {
    // 判断拖拽操作是否从开启到关闭或者从关闭到开启
    var wasOn = old && old != Init;
    if (!value != !wasOn) {
      var funcs = cm.display.dragFunctions;
      var toggle = value ? on : off;
      // 根据拖拽操作的开启或关闭状态，切换拖拽事件的绑定
      toggle(cm.display.scroller, "dragstart", funcs.start);
      toggle(cm.display.scroller, "dragenter", funcs.enter);
      toggle(cm.display.scroller, "dragover", funcs.over);
      toggle(cm.display.scroller, "dragleave", funcs.leave);
      toggle(cm.display.scroller, "drop", funcs.drop);
    }
  }

  // 当换行设置改变时触发的函数
  function wrappingChanged(cm) {
    // 如果开启了换行
    if (cm.options.lineWrapping) {
      // 给编辑器添加换行样式
      addClass(cm.display.wrapper, "CodeMirror-wrap");
      cm.display.sizer.style.minWidth = "";
      cm.display.sizerWidth = null;
    } else {
      // 移除编辑器的换行样式
      rmClass(cm.display.wrapper, "CodeMirror-wrap");
      // 重新计算最大行宽
      findMaxLine(cm);
    }
    // 估算行高
    estimateLineHeights(cm);
    // 注册编辑器的改变
    regChange(cm);
    // 清除缓存
    clearCaches(cm);
    // 延迟 100 毫秒后更新滚动条
    setTimeout(function () { return updateScrollbars(cm); }, 100);
  }

  // CodeMirror 实例代表一个编辑器。这是用户代码通常要处理的对象。
  function CodeMirror(place, options) {
    var this$1 = this;

    // 如果不是通过 new 关键字调用，则返回一个新的 CodeMirror 实例
    if (!(this instanceof CodeMirror)) { return new CodeMirror(place, options) }

    // 设置选项
    this.options = options = options ? copyObj(options) : {};
    // 根据给定值和默认值确定有效选项
    copyObj(defaults, options, false);

    // 获取文档内容
    var doc = options.value;
    // 如果文档是字符串，则创建一个新的文档对象
    if (typeof doc == "string") { doc = new Doc(doc, options.mode, null, options.lineSeparator, options.direction); }
    // 如果设置了编辑器模式，则设置文档的模式选项
    else if (options.mode) { doc.modeOption = options.mode; }
    this.doc = doc;

    // 创建输入对象
    var input = new CodeMirror.inputStyles[options.inputStyle](this);
    // 创建显示对象
    var display = this.display = new Display(place, doc, input, options);
    display.wrapper.CodeMirror = this;
    // 主题改变时触发的函数
    themeChanged(this);
    // 如果开启了换行，则给编辑器添加换行样式
    if (options.lineWrapping)
      { this.display.wrapper.className += " CodeMirror-wrap"; }
    // 初始化滚动条
    initScrollbars(this);
    # 定义组件的状态对象
    this.state = {
      keyMaps: [],  // 通过 addKeyMap 添加的键映射
      overlays: [], // 通过 addOverlay 添加的高亮覆盖层
      modeGen: 0,   // 当模式/覆盖层更改时递增，用于使高亮信息无效
      overwrite: false,  // 是否覆盖模式
      delayingBlurEvent: false,  // 是否延迟模糊事件
      focused: false,  // 是否聚焦
      suppressEdits: false, // 用于在只读模式下禁用编辑
      pasteIncoming: -1, cutIncoming: -1, // 用于识别输入.poll中的粘贴/剪切编辑
      selectingText: false,  // 是否选择文本
      draggingText: false,  // 是否拖拽文本
      highlight: new Delayed(), // 存储高亮工作器超时
      keySeq: null,  // 未完成的键序列
      specialChars: null  // 特殊字符
    };

    # 如果选项中设置了自动聚焦并且不是移动设备，则让输入框获得焦点
    if (options.autofocus && !mobile) { display.input.focus(); }

    # 覆盖 IE 在重新加载时有时对我们的隐藏文本区域执行的神奇文本内容恢复
    if (ie && ie_version < 11) { setTimeout(function () { return this$1.display.input.reset(true); }, 20); }

    # 注册事件处理程序
    registerEventHandlers(this);
    # 确保全局处理程序
    ensureGlobalHandlers();

    # 开始操作
    startOperation(this);
    this.curOp.forceUpdate = true;
    # 附加文档
    attachDoc(this, doc);

    # 如果设置了自动聚焦并且不是移动设备，或者已经聚焦，则延迟20毫秒后执行聚焦函数，否则执行失焦函数
    if ((options.autofocus && !mobile) || this.hasFocus())
      { setTimeout(bind(onFocus, this), 20); }
    else
      { onBlur(this); }

    # 遍历选项处理程序，执行相应的处理函数
    for (var opt in optionHandlers) { if (optionHandlers.hasOwnProperty(opt))
      { optionHandlers[opt](this, options[opt], Init); } }
    # 可能更新行号宽度
    maybeUpdateLineNumberWidth(this);
    # 如果设置了完成初始化函数，则执行
    if (options.finishInit) { options.finishInit(this); }
    # 遍历初始化钩子数组，执行相应的钩子函数
    for (var i = 0; i < initHooks.length; ++i) { initHooks[i](this); }
    # 结束操作
    endOperation(this);
    # 在 Webkit 中禁用 optimizelegibility，因为它会破坏文本在换行边界的测量
    // 检查是否存在 webkit，并且选项中包含换行，并且显示行的文本渲染为 "optimizelegibility"
    if (webkit && options.lineWrapping &&
        getComputedStyle(display.lineDiv).textRendering == "optimizelegibility")
      { display.lineDiv.style.textRendering = "auto"; }
  }

  // 默认配置选项
  CodeMirror.defaults = defaults;
  // 当选项改变时运行的函数
  CodeMirror.optionHandlers = optionHandlers;

  // 初始化编辑器时附加必要的事件处理程序
  function registerEventHandlers(cm) {
    var d = cm.display;
    // 当鼠标按下时运行操作函数
    on(d.scroller, "mousedown", operation(cm, onMouseDown));
    // 旧版 IE 在双击时不会触发第二次 mousedown
    if (ie && ie_version < 11)
      { on(d.scroller, "dblclick", operation(cm, function (e) {
        if (signalDOMEvent(cm, e)) { return }
        var pos = posFromMouse(cm, e);
        if (!pos || clickInGutter(cm, e) || eventInWidget(cm.display, e)) { return }
        e_preventDefault(e);
        var word = cm.findWordAt(pos);
        extendSelection(cm.doc, word.anchor, word.head);
      })); }
    else
      { on(d.scroller, "dblclick", function (e) { return signalDOMEvent(cm, e) || e_preventDefault(e); }); }
    // 一些浏览器在打开菜单后才触发 contextmenu 事件，这时我们无法再进行处理。这些浏览器的右键菜单在 onMouseDown 中处理。
    on(d.scroller, "contextmenu", function (e) { return onContextMenu(cm, e); });
    on(d.input.getField(), "contextmenu", function (e) {
      if (!d.scroller.contains(e.target)) { onContextMenu(cm, e); }
    });

    // 用于在触摸事件发生时抑制鼠标事件处理
    var touchFinished, prevTouch = {end: 0};
    function finishTouch() {
      if (d.activeTouch) {
        touchFinished = setTimeout(function () { return d.activeTouch = null; }, 1000);
        prevTouch = d.activeTouch;
        prevTouch.end = +new Date;
      }
    }
    // 判断触摸事件是否类似鼠标事件
    function isMouseLikeTouchEvent(e) {
      // 如果触摸点数量不为1，则返回false
      if (e.touches.length != 1) { return false }
      // 获取第一个触摸点
      var touch = e.touches[0];
      // 判断触摸点的半径是否小于等于1
      return touch.radiusX <= 1 && touch.radiusY <= 1
    }
    // 判断两个触摸点是否距离较远
    function farAway(touch, other) {
      // 如果另一个触摸点的左边界为null，则返回true
      if (other.left == null) { return true }
      // 计算两个触摸点的水平和垂直距离
      var dx = other.left - touch.left, dy = other.top - touch.top;
      // 判断距离是否大于20的平方
      return dx * dx + dy * dy > 20 * 20
    }
    // 绑定触摸开始事件
    on(d.scroller, "touchstart", function (e) {
      // 如果不是DOM事件信号、不类似鼠标事件、不在行号区域点击
      if (!signalDOMEvent(cm, e) && !isMouseLikeTouchEvent(e) && !clickInGutter(cm, e)) {
        // 确保输入已轮询
        d.input.ensurePolled();
        // 清除触摸结束的定时器
        clearTimeout(touchFinished);
        // 获取当前时间
        var now = +new Date;
        // 设置活动触摸点的属性
        d.activeTouch = {start: now, moved: false,
                         prev: now - prevTouch.end <= 300 ? prevTouch : null};
        // 如果触摸点数量为1，则设置活动触摸点的左边界和上边界
        if (e.touches.length == 1) {
          d.activeTouch.left = e.touches[0].pageX;
          d.activeTouch.top = e.touches[0].pageY;
        }
      }
    });
    // 绑定触摸移动事件
    on(d.scroller, "touchmove", function () {
      // 如果存在活动触摸点，则设置moved属性为true
      if (d.activeTouch) { d.activeTouch.moved = true; }
    });
    // 绑定触摸结束事件
    on(d.scroller, "touchend", function (e) {
      // 获取活动触摸点
      var touch = d.activeTouch;
      // 如果存在活动触摸点且不在小部件内、左边界不为null、未移动且触摸时间小于300毫秒
      if (touch && !eventInWidget(d, e) && touch.left != null &&
          !touch.moved && new Date - touch.start < 300) {
        // 获取触摸点对应的位置
        var pos = cm.coordsChar(d.activeTouch, "page"), range;
        // 如果不存在前一个触摸点或者与前一个触摸点距离较远，则为单击
        if (!touch.prev || farAway(touch, touch.prev)) { range = new Range(pos, pos); }
        // 如果不存在前一个前一个触摸点或者与前一个前一个触摸点距离较远，则为双击
        else if (!touch.prev.prev || farAway(touch, touch.prev.prev)) { range = cm.findWordAt(pos); }
        // 否则为三击
        else { range = new Range(Pos(pos.line, 0), clipPos(cm.doc, Pos(pos.line + 1, 0))); }
        // 设置选区并聚焦
        cm.setSelection(range.anchor, range.head);
        cm.focus();
        // 阻止默认事件
        e_preventDefault(e);
      }
      // 结束触摸事件
      finishTouch();
    });
    // 绑定触摸取消事件
    on(d.scroller, "touchcancel", finishTouch);

    // 同步虚拟滚动条和真实可滚动区域的滚动，确保视口在滚动时更新
    // 监听滚动事件，更新滚动条位置并发送滚动信号
    on(d.scroller, "scroll", function () {
      // 如果滚动容器有高度
      if (d.scroller.clientHeight) {
        // 更新编辑器的垂直滚动位置
        updateScrollTop(cm, d.scroller.scrollTop);
        // 设置编辑器的水平滚动位置
        setScrollLeft(cm, d.scroller.scrollLeft, true);
        // 发送滚动信号
        signal(cm, "scroll", cm);
      }
    });

    // 监听鼠标滚轮事件，调用onScrollWheel函数
    on(d.scroller, "mousewheel", function (e) { return onScrollWheel(cm, e); });
    // 监听鼠标滚轮事件（兼容Firefox），调用onScrollWheel函数
    on(d.scroller, "DOMMouseScroll", function (e) { return onScrollWheel(cm, e); });

    // 阻止包装器滚动
    on(d.wrapper, "scroll", function () { return d.wrapper.scrollTop = d.wrapper.scrollLeft = 0; });

    // 定义拖拽事件处理函数
    d.dragFunctions = {
      // 鼠标进入事件处理函数
      enter: function (e) {if (!signalDOMEvent(cm, e)) { e_stop(e); }},
      // 鼠标悬停事件处理函数
      over: function (e) {if (!signalDOMEvent(cm, e)) { onDragOver(cm, e); e_stop(e); }},
      // 拖拽开始事件处理函数
      start: function (e) { return onDragStart(cm, e); },
      // 放置事件处理函数
      drop: operation(cm, onDrop),
      // 鼠标离开事件处理函数
      leave: function (e) {if (!signalDOMEvent(cm, e)) { clearDragCursor(cm); }}
    };

    // 获取输入框元素
    var inp = d.input.getField();
    // 监听键盘按键抬起事件
    on(inp, "keyup", function (e) { return onKeyUp.call(cm, e); });
    // 监听键盘按键按下事件
    on(inp, "keydown", operation(cm, onKeyDown));
    // 监听键盘按键按下事件
    on(inp, "keypress", operation(cm, onKeyPress));
    // 监听输入框获取焦点事件
    on(inp, "focus", function (e) { return onFocus(cm, e); });
    // 监听输入框失去焦点事件
    on(inp, "blur", function (e) { return onBlur(cm, e); });
  }

  // 初始化钩子函数数组
  var initHooks = [];
  // 定义初始化钩子函数
  CodeMirror.defineInitHook = function (f) { return initHooks.push(f); };

  // 缩进给定行的代码
  function indentLine(cm, n, how, aggressive) {
    // 获取文档对象和状态
    var doc = cm.doc, state;
    // 如果how参数为null，则默认为"add"
    if (how == null) { how = "add"; }
    if (how == "smart") {
      // 如果缩进方式为"smart"，则当模式没有缩进方法时，回退到"prev"
      if (!doc.mode.indent) { how = "prev"; }
      else { state = getContextBefore(cm, n).state; }
    }

    var tabSize = cm.options.tabSize;  // 获取编辑器的制表符大小
    var line = getLine(doc, n), curSpace = countColumn(line.text, null, tabSize);  // 获取当前行的内容和缩进空格数
    if (line.stateAfter) { line.stateAfter = null; }  // 清除当前行的状态信息
    var curSpaceString = line.text.match(/^\s*/)[0], indentation;  // 获取当前行的缩进空格字符串
    if (!aggressive && !/\S/.test(line.text)) {
      indentation = 0;  // 如果不是侵略性缩进且当前行为空白，则缩进为0
      how = "not";
    } else if (how == "smart") {
      indentation = doc.mode.indent(state, line.text.slice(curSpaceString.length), line.text);  // 根据缩进模式计算缩进
      if (indentation == Pass || indentation > 150) {
        if (!aggressive) { return }  // 如果不是侵略性缩进，则直接返回
        how = "prev";  // 否则回退到"prev"
      }
    }
    if (how == "prev") {
      if (n > doc.first) { indentation = countColumn(getLine(doc, n-1).text, null, tabSize); }  // 如果缩进方式为"prev"，则获取上一行的缩进空格数
      else { indentation = 0; }  // 否则缩进为0
    } else if (how == "add") {
      indentation = curSpace + cm.options.indentUnit;  // 如果缩进方式为"add"，则在当前缩进基础上增加一个缩进单位
    } else if (how == "subtract") {
      indentation = curSpace - cm.options.indentUnit;  // 如果缩进方式为"subtract"，则在当前缩进基础上减少一个缩进单位
    } else if (typeof how == "number") {
      indentation = curSpace + how;  // 如果缩进方式为数字，则在当前缩进基础上增加指定的缩进数
    }
    indentation = Math.max(0, indentation);  // 缩进数取最大值为0

    var indentString = "", pos = 0;
    if (cm.options.indentWithTabs)
      { for (var i = Math.floor(indentation / tabSize); i; --i) {pos += tabSize; indentString += "\t";} }  // 如果使用制表符缩进，则计算制表符和空格的组合
    if (pos < indentation) { indentString += spaceStr(indentation - pos); }  // 如果还有剩余的空格，则添加到缩进字符串中

    if (indentString != curSpaceString) {
      replaceRange(doc, indentString, Pos(n, 0), Pos(n, curSpaceString.length), "+input");  // 替换当前行的缩进字符串
      line.stateAfter = null;  // 清除当前行的状态信息
      return true  // 返回true表示缩进已经处理完成
    } else {
      // 如果光标位于行首的空白处，确保将其移动到该空白的末尾
      for (var i$1 = 0; i$1 < doc.sel.ranges.length; i$1++) {
        var range = doc.sel.ranges[i$1];
        if (range.head.line == n && range.head.ch < curSpaceString.length) {
          var pos$1 = Pos(n, curSpaceString.length);
          replaceOneSelection(doc, i$1, new Range(pos$1, pos$1));
          break
        }
      }
    }
  }

  // 这将被设置为一个 {lineWise: bool, text: [string]} 对象，因此在粘贴时，我们知道复制的文本是由什么类型的选择组成的。
  var lastCopied = null;

  function setLastCopied(newLastCopied) {
    lastCopied = newLastCopied;
  }

  function applyTextInput(cm, inserted, deleted, sel, origin) {
    var doc = cm.doc;
    cm.display.shift = false;
    if (!sel) { sel = doc.sel; }

    var recent = +new Date - 200;
    var paste = origin == "paste" || cm.state.pasteIncoming > recent;
    var textLines = splitLinesAuto(inserted), multiPaste = null;
    // 当将N行粘贴到N个选择中时，每个选择插入一行
    if (paste && sel.ranges.length > 1) {
      if (lastCopied && lastCopied.text.join("\n") == inserted) {
        if (sel.ranges.length % lastCopied.text.length == 0) {
          multiPaste = [];
          for (var i = 0; i < lastCopied.text.length; i++)
            { multiPaste.push(doc.splitLines(lastCopied.text[i])); }
        }
      } else if (textLines.length == sel.ranges.length && cm.options.pasteLinesPerSelection) {
        multiPaste = map(textLines, function (l) { return [l]; });
      }
    }

    var updateInput = cm.curOp.updateInput;
    // 正常行为是将新文本插入到每个选择中
    # 遍历选择范围数组，从最后一个范围开始
    for (var i$1 = sel.ranges.length - 1; i$1 >= 0; i$1--) {
      # 获取当前范围
      var range = sel.ranges[i$1];
      # 获取范围的起始位置和结束位置
      var from = range.from(), to = range.to();
      # 如果范围为空
      if (range.empty()) {
        # 处理删除操作
        if (deleted && deleted > 0) // Handle deletion
          { from = Pos(from.line, from.ch - deleted); }
        # 处理覆盖操作
        else if (cm.state.overwrite && !paste) // Handle overwrite
          { to = Pos(to.line, Math.min(getLine(doc, to.line).text.length, to.ch + lst(textLines).length)); }
        # 处理粘贴操作
        else if (paste && lastCopied && lastCopied.lineWise && lastCopied.text.join("\n") == textLines.join("\n"))
          { from = to = Pos(from.line, 0); }
      }
      # 创建变更事件对象
      var changeEvent = {from: from, to: to, text: multiPaste ? multiPaste[i$1 % multiPaste.length] : textLines,
                         origin: origin || (paste ? "paste" : cm.state.cutIncoming > recent ? "cut" : "+input")};
      # 进行变更操作
      makeChange(cm.doc, changeEvent);
      # 发送输入读取信号
      signalLater(cm, "inputRead", cm, changeEvent);
    }
    # 如果有插入并且不是粘贴操作，则触发电子字符
    if (inserted && !paste)
      { triggerElectric(cm, inserted); }

    # 确保光标可见
    ensureCursorVisible(cm);
    # 如果当前操作的更新输入小于2，则更新输入
    if (cm.curOp.updateInput < 2) { cm.curOp.updateInput = updateInput; }
    # 设置当前操作为输入
    cm.curOp.typing = true;
    # 重置粘贴和剪切操作的状态
    cm.state.pasteIncoming = cm.state.cutIncoming = -1;
  }

  # 处理粘贴事件
  function handlePaste(e, cm) {
    # 获取粘贴的文本
    var pasted = e.clipboardData && e.clipboardData.getData("Text");
    # 如果有粘贴的文本
    if (pasted) {
      # 阻止默认粘贴行为
      e.preventDefault();
      # 如果编辑器不是只读且输入未被禁用，则应用文本输入
      if (!cm.isReadOnly() && !cm.options.disableInput)
        { runInOp(cm, function () { return applyTextInput(cm, pasted, 0, null, "paste"); }); }
      return true
    }
  }

  # 触发电子字符
  function triggerElectric(cm, inserted) {
    # 当插入'电子'字符时，立即触发重新缩进
    if (!cm.options.electricChars || !cm.options.smartIndent) { return }
    # 获取选择范围
    var sel = cm.doc.sel;
    // 从最后一个选区开始循环到第一个选区
    for (var i = sel.ranges.length - 1; i >= 0; i--) {
      // 获取当前选区
      var range = sel.ranges[i];
      // 如果当前选区的光标位置大于100，或者不是第一个选区且和前一个选区在同一行，则跳过当前循环
      if (range.head.ch > 100 || (i && sel.ranges[i - 1].head.line == range.head.line)) { continue }
      // 获取当前选区的模式
      var mode = cm.getModeAt(range.head);
      // 初始化缩进标志
      var indented = false;
      // 如果当前模式有电气字符
      if (mode.electricChars) {
        // 遍历电气字符数组
        for (var j = 0; j < mode.electricChars.length; j++)
          { 
            // 如果插入的字符在电气字符数组中，则进行智能缩进
            if (inserted.indexOf(mode.electricChars.charAt(j)) > -1) {
              indented = indentLine(cm, range.head.line, "smart");
              break
            } 
          }
      } else if (mode.electricInput) {
        // 如果当前模式有电气输入测试，则进行智能缩进
        if (mode.electricInput.test(getLine(cm.doc, range.head.line).text.slice(0, range.head.ch)))
          { indented = indentLine(cm, range.head.line, "smart"); }
      }
      // 如果进行了缩进，则发送“electricInput”信号
      if (indented) { signalLater(cm, "electricInput", cm, range.head.line); }
    }
  }

  // 获取可复制的选区和文本
  function copyableRanges(cm) {
    var text = [], ranges = [];
    for (var i = 0; i < cm.doc.sel.ranges.length; i++) {
      var line = cm.doc.sel.ranges[i].head.line;
      var lineRange = {anchor: Pos(line, 0), head: Pos(line + 1, 0)};
      ranges.push(lineRange);
      text.push(cm.getRange(lineRange.anchor, lineRange.head));
    }
    return {text: text, ranges: ranges}
  }

  // 禁用浏览器的输入法自动修正和自动大写功能
  function disableBrowserMagic(field, spellcheck, autocorrect, autocapitalize) {
    field.setAttribute("autocorrect", autocorrect ? "" : "off");
    field.setAttribute("autocapitalize", autocapitalize ? "" : "off");
    field.setAttribute("spellcheck", !!spellcheck);
  }

  // 创建一个隐藏的文本输入框
  function hiddenTextarea() {
    var te = elt("textarea", null, null, "position: absolute; bottom: -1em; padding: 0; width: 1px; height: 1em; outline: none");
    var div = elt("div", [te], null, "overflow: hidden; position: relative; width: 3px; height: 0px;");
    // 保持文本输入框靠近光标位置，防止因为输入而滚动光标位置出视野
    // 在webkit中，当wrap=off时，粘贴操作会导致文本输入框滚动
  }
    // 如果是 webkit 浏览器，将编辑区域宽度设置为 1000px
    if (webkit) { te.style.width = "1000px"; }
    // 如果不是 webkit 浏览器，将编辑区域的 wrap 属性设置为 off
    else { te.setAttribute("wrap", "off"); }
    // 如果是 iOS 设备，将编辑区域的边框样式设置为 1px 实线黑色边框
    if (ios) { te.style.border = "1px solid black"; }
    // 禁用浏览器的默认行为
    disableBrowserMagic(te);
    // 返回包含编辑区域的 div 元素
    return div
  }

  // 公开可见的 API。注意，methodOp(f) 表示“将 f 包装在一个操作中，该操作在其 `this` 参数上执行”。
  // 这不是编辑器方法的完整集合。Doc 类型定义的大多数方法也注入到 CodeMirror.prototype 中，以实现向后兼容性和便利性。
  function addEditorMethods(CodeMirror) {
    // 获取 CodeMirror 的选项处理程序
    var optionHandlers = CodeMirror.optionHandlers;
    // 定义 CodeMirror 的辅助方法
    var helpers = CodeMirror.helpers = {};
    // 将事件混合到 CodeMirror 中
    eventMixin(CodeMirror);
    // 注册辅助方法
    CodeMirror.registerHelper = function(type, name, value) {
      if (!helpers.hasOwnProperty(type)) { helpers[type] = CodeMirror[type] = {_global: []}; }
      helpers[type][name] = value;
    };
    // 注册全局辅助方法
    CodeMirror.registerGlobalHelper = function(type, name, predicate, value) {
      CodeMirror.registerHelper(type, name, value);
      helpers[type]._global.push({pred: predicate, val: value});
    };
  }

  // 用于水平相对运动。Dir 是 -1 或 1（左或右），unit 可以是 "char"、"column"（类似 char，但不跨越行边界）、"word"（跨越下一个单词）、或 "group"（到下一个单词或非单词非空白字符组的开头）。visually 参数控制在从右到左的文本中，方向 1 是否表示向字符串中的下一个索引移动，还是向当前位置右侧的字符移动。结果位置将具有 hitSide=true 属性，如果它到达文档的末尾。
  function findPosH(doc, pos, dir, unit, visually) {
    // 保存旧的位置
    var oldPos = pos;
    // 保存原始的方向
    var origDir = dir;
    // 获取行对象
    var lineObj = getLine(doc, pos.line);
    # 根据可视化方向和文档方向确定行的方向
    var lineDir = visually && doc.direction == "rtl" ? -dir : dir;
    
    # 查找下一行的函数
    function findNextLine() {
      # 计算下一行的行号
      var l = pos.line + lineDir;
      # 如果行号超出文档范围，则返回 false
      if (l < doc.first || l >= doc.first + doc.size) { return false }
      # 更新位置对象的行号
      pos = new Pos(l, pos.ch, pos.sticky);
      # 获取下一行的行对象
      return lineObj = getLine(doc, l)
    }
    
    # 单次移动光标的函数
    function moveOnce(boundToLine) {
      var next;
      # 根据可视化或逻辑方向移动光标
      if (visually) {
        next = moveVisually(doc.cm, lineObj, pos, dir);
      } else {
        next = moveLogically(lineObj, pos, dir);
      }
      # 如果下一个位置为空，则根据条件继续移动光标
      if (next == null) {
        if (!boundToLine && findNextLine())
          { pos = endOfLine(visually, doc.cm, lineObj, pos.line, lineDir); }
        else
          { return false }
      } else {
        pos = next;
      }
      return true
    }

    # 根据单位类型移动光标
    if (unit == "char") {
      moveOnce();
    } else if (unit == "column") {
      moveOnce(true);
    } else if (unit == "word" || unit == "group") {
      var sawType = null, group = unit == "group";
      var helper = doc.cm && doc.cm.getHelper(pos, "wordChars");
      # 循环移动光标直到满足条件
      for (var first = true;; first = false) {
        if (dir < 0 && !moveOnce(!first)) { break }
        var cur = lineObj.text.charAt(pos.ch) || "\n";
        var type = isWordChar(cur, helper) ? "w"
          : group && cur == "\n" ? "n"
          : !group || /\s/.test(cur) ? null
          : "p";
        if (group && !first && !type) { type = "s"; }
        if (sawType && sawType != type) {
          if (dir < 0) {dir = 1; moveOnce(); pos.sticky = "after";}
          break
        }

        if (type) { sawType = type; }
        if (dir > 0 && !moveOnce(!first)) { break }
      }
    }
    
    # 跳过原子单位
    var result = skipAtomic(doc, pos, oldPos, origDir, true);
    # 如果新位置和旧位置相同，则设置标志
    if (equalCursorPos(oldPos, result)) { result.hitSide = true; }
  // 返回结果
  return result
}

// 用于垂直方向的相对移动。Dir 可以是 -1 或 1。Unit 可以是 "page" 或 "line"。如果到达文档末尾，结果位置将具有 hitSide=true 属性。
function findPosV(cm, pos, dir, unit) {
  var doc = cm.doc, x = pos.left, y;
  if (unit == "page") {
    // 计算页面大小和移动量
    var pageSize = Math.min(cm.display.wrapper.clientHeight, window.innerHeight || document.documentElement.clientHeight);
    var moveAmount = Math.max(pageSize - .5 * textHeight(cm.display), 3);
    y = (dir > 0 ? pos.bottom : pos.top) + dir * moveAmount;

  } else if (unit == "line") {
    // 根据单位确定垂直位置
    y = dir > 0 ? pos.bottom + 3 : pos.top - 3;
  }
  var target;
  // 循环直到找到合适的位置
  for (;;) {
    target = coordsChar(cm, x, y);
    if (!target.outside) { break }
    if (dir < 0 ? y <= 0 : y >= doc.height) { target.hitSide = true; break }
    y += dir * 5;
  }
  return target
}

// CONTENTEDITABLE INPUT STYLE

// ContentEditableInput 类
var ContentEditableInput = function(cm) {
  this.cm = cm;
  this.lastAnchorNode = this.lastAnchorOffset = this.lastFocusNode = this.lastFocusOffset = null;
  this.polling = new Delayed();
  this.composing = null;
  this.gracePeriod = false;
  this.readDOMTimeout = null;
};

// 初始化 ContentEditableInput
ContentEditableInput.prototype.init = function (display) {
  var this$1 = this;
  var input = this, cm = input.cm;
  var div = input.div = display.lineDiv;
  // 禁用浏览器的输入法和自动修正功能
  disableBrowserMagic(div, cm.options.spellcheck, cm.options.autocorrect, cm.options.autocapitalize);

  function belongsToInput(e) {
    for (var t = e.target; t; t = t.parentNode) {
      if (t == div) { return true }
      if (/\bCodeMirror-(?:line)?widget\b/.test(t.className)) { break }
    }
    return false
  }
}
    # 给 div 元素添加粘贴事件监听器
    on(div, "paste", function (e) {
      # 如果事件不属于输入元素，或者信号DOM事件，或者处理粘贴事件，则返回
      if (!belongsToInput(e) || signalDOMEvent(cm, e) || handlePaste(e, cm)) { return }
      # 如果是IE浏览器版本小于等于11，则延迟20毫秒后执行操作，更新DOM
      if (ie_version <= 11) { setTimeout(operation(cm, function () { return this$1.updateFromDOM(); }), 20); }
    });

    # 给 div 元素添加开始组合事件监听器
    on(div, "compositionstart", function (e) {
      # 设置正在组合的状态对象，包括数据和完成状态
      this$1.composing = {data: e.data, done: false};
    });

    # 给 div 元素添加组合更新事件监听器
    on(div, "compositionupdate", function (e) {
      # 如果没有正在组合的状态对象，则设置正在组合的状态对象，包括数据和完成状态
      if (!this$1.composing) { this$1.composing = {data: e.data, done: false}; }
    });

    # 给 div 元素添加组合结束事件监听器
    on(div, "compositionend", function (e) {
      # 如果存在正在组合的状态对象
      if (this$1.composing) {
        # 如果事件数据不等于正在组合的状态对象的数据，则延迟读取DOM内容
        if (e.data != this$1.composing.data) { this$1.readFromDOMSoon(); }
        # 设置正在组合的状态对象的完成状态为true
        this$1.composing.done = true;
      }
    });

    # 给 div 元素添加触摸开始事件监听器
    on(div, "touchstart", function () { return input.forceCompositionEnd(); });

    # 给 div 元素添加输入事件监听器
    on(div, "input", function () {
      # 如果没有正在组合的状态对象，则延迟读取DOM内容
      if (!this$1.composing) { this$1.readFromDOMSoon(); }
    });
    // 定义 onCopyCut 函数，处理复制和剪切事件
    function onCopyCut(e) {
      // 如果事件不属于输入框，或者被信号 DOM 事件捕获，则返回
      if (!belongsToInput(e) || signalDOMEvent(cm, e)) { return }
      // 如果有选中内容
      if (cm.somethingSelected()) {
        // 设置最后复制的内容为非行级别，获取选中的文本
        setLastCopied({lineWise: false, text: cm.getSelections()});
        // 如果是剪切操作，则替换选中内容为空
        if (e.type == "cut") { cm.replaceSelection("", null, "cut"); }
      } else if (!cm.options.lineWiseCopyCut) {
        return
      } else {
        // 获取可复制的范围
        var ranges = copyableRanges(cm);
        // 设置最后复制的内容为行级别，获取可复制的文本
        setLastCopied({lineWise: true, text: ranges.text});
        // 如果是剪切操作
        if (e.type == "cut") {
          // 执行操作，设置选中范围，替换选中内容为空
          cm.operation(function () {
            cm.setSelections(ranges.ranges, 0, sel_dontScroll);
            cm.replaceSelection("", null, "cut");
          });
        }
      }
      // 如果支持剪贴板数据
      if (e.clipboardData) {
        // 清空剪贴板数据
        e.clipboardData.clearData();
        // 将最后复制的文本内容以换行符连接，设置为剪贴板数据
        var content = lastCopied.text.join("\n");
        // iOS 暴露了剪贴板 API，但似乎会丢弃插入其中的内容
        e.clipboardData.setData("Text", content);
        // 如果剪贴板数据获取到的内容与设置的内容一致，则阻止默认行为
        if (e.clipboardData.getData("Text") == content) {
          e.preventDefault();
          return
        }
      }
      // 旧式的焦点短暂聚焦到文本区域的 hack
      var kludge = hiddenTextarea(), te = kludge.firstChild;
      cm.display.lineSpace.insertBefore(kludge, cm.display.lineSpace.firstChild);
      te.value = lastCopied.text.join("\n");
      var hadFocus = document.activeElement;
      selectInput(te);
      setTimeout(function () {
        cm.display.lineSpace.removeChild(kludge);
        hadFocus.focus();
        if (hadFocus == div) { input.showPrimarySelection(); }
      }, 50);
    }
    // 绑定复制事件到 onCopyCut 函数
    on(div, "copy", onCopyCut);
    // 绑定剪切事件到 onCopyCut 函数
    on(div, "cut", onCopyCut);
  };

  // 定义 screenReaderLabelChanged 函数，用于改变屏幕阅读器的标签
  ContentEditableInput.prototype.screenReaderLabelChanged = function (label) {
    // 如果有标签，则设置 div 的 aria-label 属性为标签内容，否则移除 aria-label 属性
    if(label) {
      this.div.setAttribute('aria-label', label);
    } else {
      this.div.removeAttribute('aria-label');
    }
  };

  // 定义 prepareSelection 函数，准备选择内容
  ContentEditableInput.prototype.prepareSelection = function () {
    // 调用 prepareSelection 函数，传入当前编辑器对象和 false 参数
    var result = prepareSelection(this.cm, false);
  // 检查当前焦点是否在结果元素上，将结果赋给 focus 变量
  result.focus = document.activeElement == this.div;
  // 返回结果对象
  return result
};

// 显示选择的内容
ContentEditableInput.prototype.showSelection = function (info, takeFocus) {
  // 如果 info 不存在或者编辑器没有内容，则返回
  if (!info || !this.cm.display.view.length) { return }
  // 如果 info 有焦点或者 takeFocus 为真，则显示主要选择
  if (info.focus || takeFocus) { this.showPrimarySelection(); }
  // 显示多个选择
  this.showMultipleSelections(info);
};

// 获取选择的内容
ContentEditableInput.prototype.getSelection = function () {
  return this.cm.display.wrapper.ownerDocument.getSelection()
};

// 显示主要选择
ContentEditableInput.prototype.showPrimarySelection = function () {
  // 获取选择对象和编辑器对象
  var sel = this.getSelection(), cm = this.cm, prim = cm.doc.sel.primary();
  var from = prim.from(), to = prim.to();

  // 如果编辑器没有内容或者选择的行超出了显示范围，则移除所有选择
  if (cm.display.viewTo == cm.display.viewFrom || from.line >= cm.display.viewTo || to.line < cm.display.viewFrom) {
    sel.removeAllRanges();
    return
  }

  // 获取当前锚点和焦点的位置
  var curAnchor = domToPos(cm, sel.anchorNode, sel.anchorOffset);
  var curFocus = domToPos(cm, sel.focusNode, sel.focusOffset);
  // 如果当前位置有效且和选择的位置相同，则返回
  if (curAnchor && !curAnchor.bad && curFocus && !curFocus.bad &&
      cmp(minPos(curAnchor, curFocus), from) == 0 &&
      cmp(maxPos(curAnchor, curFocus), to) == 0)
    { return }

  // 获取编辑器的显示视图和起始、结束位置的 DOM 元素
  var view = cm.display.view;
  var start = (from.line >= cm.display.viewFrom && posToDOM(cm, from)) ||
      {node: view[0].measure.map[2], offset: 0};
  var end = to.line < cm.display.viewTo && posToDOM(cm, to);
  if (!end) {
    var measure = view[view.length - 1].measure;
    var map = measure.maps ? measure.maps[measure.maps.length - 1] : measure.map;
    end = {node: map[map.length - 1], offset: map[map.length - 2] - map[map.length - 3]};
  }

  // 如果起始或结束位置不存在，则移除所有选择
  if (!start || !end) {
    sel.removeAllRanges();
    return
  }

  // 获取旧的选择范围对象，尝试创建新的选择范围对象
  var old = sel.rangeCount && sel.getRangeAt(0), rng;
  try { rng = range(start.node, start.offset, end.offset, end.node); }
  catch(e) {} // Our model of the DOM might be outdated, in which case the range we try to set can be impossible
}
    # 如果存在选区
    if (rng) {
      # 如果不是 gecko 浏览器，并且编辑器处于焦点状态
      if (!gecko && cm.state.focused) {
        # 折叠选区到指定的起始节点和偏移位置
        sel.collapse(start.node, start.offset);
        # 如果选区没有折叠
        if (!rng.collapsed) {
          # 移除所有选区范围
          sel.removeAllRanges();
          # 添加指定的选区范围
          sel.addRange(rng);
        }
      } else {
        # 移除所有选区范围
        sel.removeAllRanges();
        # 添加指定的选区范围
        sel.addRange(rng);
      }
      # 如果存在旧的选区，并且选择锚点节点为 null
      if (old && sel.anchorNode == null) { sel.addRange(old); }
      # 如果是 gecko 浏览器，启动优雅期
      else if (gecko) { this.startGracePeriod(); }
    }
    # 记住当前选区状态
    this.rememberSelection();
  };

  # 开始优雅期
  ContentEditableInput.prototype.startGracePeriod = function () {
      var this$1 = this;

    # 清除之前的优雅期定时器
    clearTimeout(this.gracePeriod);
    # 设置新的优雅期定时器
    this.gracePeriod = setTimeout(function () {
      this$1.gracePeriod = false;
      # 如果选区发生变化，则在编辑器操作中设置选区变化标志
      if (this$1.selectionChanged())
        { this$1.cm.operation(function () { return this$1.cm.curOp.selectionChanged = true; }); }
    }, 20);
  };

  # 显示多重选区
  ContentEditableInput.prototype.showMultipleSelections = function (info) {
    # 移除光标和选区的子元素，并添加新的光标和选区
    removeChildrenAndAdd(this.cm.display.cursorDiv, info.cursors);
    removeChildrenAndAdd(this.cm.display.selectionDiv, info.selection);
  };

  # 记住当前选区状态
  ContentEditableInput.prototype.rememberSelection = function () {
    var sel = this.getSelection();
    # 记录当前选区的锚点节点和偏移位置，以及焦点节点和偏移位置
    this.lastAnchorNode = sel.anchorNode; this.lastAnchorOffset = sel.anchorOffset;
    this.lastFocusNode = sel.focusNode; this.lastFocusOffset = sel.focusOffset;
  };

  # 判断选区是否在编辑器内
  ContentEditableInput.prototype.selectionInEditor = function () {
    var sel = this.getSelection();
    # 如果没有选区范围，则返回 false
    if (!sel.rangeCount) { return false }
    # 获取选区范围的公共祖先节点，并判断是否在编辑器内
    var node = sel.getRangeAt(0).commonAncestorContainer;
    return contains(this.div, node)
  };

  # 设置编辑器焦点
  ContentEditableInput.prototype.focus = function () {
    # 如果编辑器不是只读模式
    if (this.cm.options.readOnly != "nocursor") {
      # 如果选区不在编辑器内，或者当前活动元素不是编辑器
      if (!this.selectionInEditor() || document.activeElement != this.div)
        { this.showSelection(this.prepareSelection(), true); }
      # 设置编辑器焦点
      this.div.focus();
  // 定义 ContentEditableInput 对象的 blur 方法，使其调用 div 的 blur 方法
  ContentEditableInput.prototype.blur = function () { this.div.blur(); };
  // 定义 ContentEditableInput 对象的 getField 方法，返回 div 对象
  ContentEditableInput.prototype.getField = function () { return this.div };
  // 定义 ContentEditableInput 对象的 supportsTouch 方法，始终返回 true
  ContentEditableInput.prototype.supportsTouch = function () { return true };
  // 定义 ContentEditableInput 对象的 receivedFocus 方法，处理焦点接收时的逻辑
  ContentEditableInput.prototype.receivedFocus = function () {
    // 保存当前 input 对象
    var input = this;
    // 如果光标在编辑器中，则轮询选择
    if (this.selectionInEditor())
      { this.pollSelection(); }
    // 否则，运行操作，将 input.cm.curOp.selectionChanged 设置为 true
    else
      { runInOp(this.cm, function () { return input.cm.curOp.selectionChanged = true; }); }

    // 定义轮询函数
    function poll() {
      // 如果编辑器处于焦点状态，则轮询选择
      if (input.cm.state.focused) {
        input.pollSelection();
        input.polling.set(input.cm.options.pollInterval, poll);
      }
    }
    // 设置轮询间隔
    this.polling.set(this.cm.options.pollInterval, poll);
  };
  // 定义 ContentEditableInput 对象的 selectionChanged 方法，判断选择是否发生变化
  ContentEditableInput.prototype.selectionChanged = function () {
    var sel = this.getSelection();
    return sel.anchorNode != this.lastAnchorNode || sel.anchorOffset != this.lastAnchorOffset ||
      sel.focusNode != this.lastFocusNode || sel.focusOffset != this.lastFocusOffset
  };
  // 定义 ContentEditableInput 对象的 pollSelection 方法，轮询选择的变化
  ContentEditableInput.prototype.pollSelection = function () {
    // 如果正在读取 DOM 或者处于优雅期间或者选择未发生变化，则返回
    if (this.readDOMTimeout != null || this.gracePeriod || !this.selectionChanged()) { return }
    var sel = this.getSelection(), cm = this.cm;
    // 在 Android Chrome（至少版本 56）中，退格到不可编辑的块元素将把光标放在该元素中，
    // 然后，因为它不可编辑，隐藏虚拟键盘。因为 Android 不允许我们以合理的方式实际检测退格按键，
    // 此代码检查当发生这种情况时并在这种情况下模拟退格按键。
    if (android && chrome && this.cm.display.gutterSpecs.length && isInGutter(sel.anchorNode)) {
      this.cm.triggerOnKeyDown({type: "keydown", keyCode: 8, preventDefault: Math.abs});
      this.blur();
      this.focus();
      return
    }
    // 如果正在组合输入，则返回
    if (this.composing) { return }
    // 记住当前选择
    this.rememberSelection();
    // 将 DOM 元素转换为位置
    var anchor = domToPos(cm, sel.anchorNode, sel.anchorOffset);
    # 根据当前选择的焦点节点和偏移量获取光标位置
    var head = domToPos(cm, sel.focusNode, sel.focusOffset);
    # 如果锚点和光标位置都存在，则在操作中执行以下代码
    if (anchor && head) { runInOp(cm, function () {
      # 设置编辑器的选择区域为锚点和光标位置之间的简单选择
      setSelection(cm.doc, simpleSelection(anchor, head), sel_dontScroll);
      # 如果锚点或光标位置存在问题，则将当前操作标记为选择区域已更改
      if (anchor.bad || head.bad) { cm.curOp.selectionChanged = true; }
    }); }
  };

  # 定义方法用于轮询内容
  ContentEditableInput.prototype.pollContent = function () {
    # 如果读取 DOM 的超时计时器存在，则清除它
    if (this.readDOMTimeout != null) {
      clearTimeout(this.readDOMTimeout);
      this.readDOMTimeout = null;
    }

    # 获取编辑器、显示区域和选择区域的相关信息
    var cm = this.cm, display = cm.display, sel = cm.doc.sel.primary();
    var from = sel.from(), to = sel.to();
    # 如果选择区域的起始位置在行首，则将起始位置向上移动一行
    if (from.ch == 0 && from.line > cm.firstLine())
      { from = Pos(from.line - 1, getLine(cm.doc, from.line - 1).length); }
    # 如果选择区域的结束位置在行末，则将结束位置向下移动一行
    if (to.ch == getLine(cm.doc, to.line).text.length && to.line < cm.lastLine())
      { to = Pos(to.line + 1, 0); }
    # 如果选择区域的起始行小于显示区域的起始行，或者结束行大于显示区域的结束行，则返回 false
    if (from.line < display.viewFrom || to.line > display.viewTo - 1) { return false }

    # 定义变量用于存储起始位置的索引、行号和节点
    var fromIndex, fromLine, fromNode;
    # 如果起始行等于显示区域的起始行，或者找不到起始行的索引，则设置起始行号和节点
    if (from.line == display.viewFrom || (fromIndex = findViewIndex(cm, from.line)) == 0) {
      fromLine = lineNo(display.view[0].line);
      fromNode = display.view[0].node;
    } else {
      fromLine = lineNo(display.view[fromIndex].line);
      fromNode = display.view[fromIndex - 1].node.nextSibling;
    }
    # 获取结束行的索引、行号和节点
    var toIndex = findViewIndex(cm, to.line);
    var toLine, toNode;
    if (toIndex == display.view.length - 1) {
      toLine = display.viewTo - 1;
      toNode = display.lineDiv.lastChild;
    } else {
      toLine = lineNo(display.view[toIndex + 1].line) - 1;
      toNode = display.view[toIndex + 1].node.previousSibling;
    }

    # 如果起始节点不存在，则返回 false
    if (!fromNode) { return false }
    # 获取起始节点和结束节点之间的文本，并将其拆分为新的文本行
    var newText = cm.doc.splitLines(domTextBetween(cm, fromNode, toNode, fromLine, toLine));
    # 获取起始位置和结束位置之间的旧文本
    var oldText = getBetween(cm.doc, Pos(fromLine, 0), Pos(toLine, getLine(cm.doc, toLine).text.length));
    // 当新旧文本长度都大于1时，进行循环比较
    while (newText.length > 1 && oldText.length > 1) {
      // 如果新旧文本的末尾相同，则删除末尾字符，更新行数
      if (lst(newText) == lst(oldText)) { newText.pop(); oldText.pop(); toLine--; }
      // 如果新旧文本的开头相同，则删除开头字符，更新行数
      else if (newText[0] == oldText[0]) { newText.shift(); oldText.shift(); fromLine++; }
      // 如果新旧文本不同，则跳出循环
      else { break }
    }

    // 初始化变量
    var cutFront = 0, cutEnd = 0;
    var newTop = newText[0], oldTop = oldText[0], maxCutFront = Math.min(newTop.length, oldTop.length);
    // 比较新旧文本开头相同的字符数
    while (cutFront < maxCutFront && newTop.charCodeAt(cutFront) == oldTop.charCodeAt(cutFront))
      { ++cutFront; }
    var newBot = lst(newText), oldBot = lst(oldText);
    var maxCutEnd = Math.min(newBot.length - (newText.length == 1 ? cutFront : 0),
                             oldBot.length - (oldText.length == 1 ? cutFront : 0));
    // 比较新旧文本末尾相同的字符数
    while (cutEnd < maxCutEnd &&
           newBot.charCodeAt(newBot.length - cutEnd - 1) == oldBot.charCodeAt(oldBot.length - cutEnd - 1))
      { ++cutEnd; }
    // 如果新旧文本长度都为1且起始行相同，则调整cutFront和cutEnd
    if (newText.length == 1 && oldText.length == 1 && fromLine == from.line) {
      while (cutFront && cutFront > from.ch &&
             newBot.charCodeAt(newBot.length - cutEnd - 1) == oldBot.charCodeAt(oldBot.length - cutEnd - 1)) {
        cutFront--;
        cutEnd++;
      }
    }

    // 更新新文本的末尾和开头
    newText[newText.length - 1] = newBot.slice(0, newBot.length - cutEnd).replace(/^\u200b+/, "");
    newText[0] = newText[0].slice(cutFront).replace(/\u200b+$/, "");

    // 计算起始和结束位置
    var chFrom = Pos(fromLine, cutFront);
    var chTo = Pos(toLine, oldText.length ? lst(oldText).length - cutEnd : 0);
    // 如果新文本长度大于1，或者新文本开头不为空，或者起始和结束位置不相同，则替换文本
    if (newText.length > 1 || newText[0] || cmp(chFrom, chTo)) {
      replaceRange(cm.doc, newText, chFrom, chTo, "+input");
      return true
    }
  };

  // 确保已经结束输入
  ContentEditableInput.prototype.ensurePolled = function () {
    this.forceCompositionEnd();
  };
  // 重置输入
  ContentEditableInput.prototype.reset = function () {
    this.forceCompositionEnd();
  };
  // 强制结束输入
  ContentEditableInput.prototype.forceCompositionEnd = function () {
    // 如果没有正在输入，则返回
    if (!this.composing) { return }
    // 清除读取 DOM 的定时器
    clearTimeout(this.readDOMTimeout);
    // 清空正在输入的内容
    this.composing = null;
    // 从 DOM 更新输入内容
    this.updateFromDOM();
    // 失去焦点
    this.div.blur();
    // 获取焦点
    this.div.focus();
  };
  // 延迟读取 DOM 内容
  ContentEditableInput.prototype.readFromDOMSoon = function () {
      var this$1 = this;

    // 如果已经设置了读取 DOM 的定时器，则直接返回
    if (this.readDOMTimeout != null) { return }
    // 设置定时器，延迟 80 毫秒后执行读取 DOM 内容的操作
    this.readDOMTimeout = setTimeout(function () {
      this$1.readDOMTimeout = null;
      // 如果正在输入中，则不执行读取 DOM 内容的操作
      if (this$1.composing) {
        if (this$1.composing.done) { this$1.composing = null; }
        else { return }
      }
      // 更新输入内容
      this$1.updateFromDOM();
    }, 80);
  };

  // 从 DOM 更新输入内容
  ContentEditableInput.prototype.updateFromDOM = function () {
      var this$1 = this;

    // 如果编辑器是只读的，或者无法获取内容，则执行注册变化的操作
    if (this.cm.isReadOnly() || !this.pollContent())
      { runInOp(this.cm, function () { return regChange(this$1.cm); }); }
  };

  // 设置节点为不可编辑状态
  ContentEditableInput.prototype.setUneditable = function (node) {
    node.contentEditable = "false";
  };

  // 处理按键事件
  ContentEditableInput.prototype.onKeyPress = function (e) {
    // 如果按键码为 0 或者正在输入中，则直接返回
    if (e.charCode == 0 || this.composing) { return }
    // 阻止默认行为
    e.preventDefault();
    // 如果编辑器不是只读的，则执行输入文本的操作
    if (!this.cm.isReadOnly())
      { operation(this.cm, applyTextInput)(this.cm, String.fromCharCode(e.charCode == null ? e.keyCode : e.charCode), 0); }
  };

  // 处理只读状态的改变
  ContentEditableInput.prototype.readOnlyChanged = function (val) {
    this.div.contentEditable = String(val != "nocursor");
  };

  // 处理右键菜单事件
  ContentEditableInput.prototype.onContextMenu = function () {};
  // 重置位置
  ContentEditableInput.prototype.resetPosition = function () {};

  // 需要内容属性
  ContentEditableInput.prototype.needsContentAttribute = true;

  // 将位置转换为 DOM 元素
  function posToDOM(cm, pos) {
    var view = findViewForLine(cm, pos.line);
    // 如果找不到视图或者视图被隐藏，则返回 null
    if (!view || view.hidden) { return null }
    var line = getLine(cm.doc, pos.line);
    var info = mapFromLineView(view, line, pos.line);

    var order = getOrder(line, cm.doc.direction), side = "left";
    // 如果存在文本方向，则根据文本方向确定偏移方向
    if (order) {
      var partPos = getBidiPartAt(order, pos.ch);
      side = partPos % 2 ? "right" : "left";
    }
    // 获取节点和偏移量
    var result = nodeAndOffsetInLineMap(info.map, pos.ch, side);
    // 如果折叠方向为右，则偏移量为结束位置，否则为起始位置
    result.offset = result.collapse == "right" ? result.end : result.start;
    // 返回结果对象
    return result
  }

  // 判断节点是否在行号区域
  function isInGutter(node) {
    // 遍历节点及其父节点，查找是否包含 CodeMirror-gutter-wrapper 类名
    for (var scan = node; scan; scan = scan.parentNode)
      { if (/CodeMirror-gutter-wrapper/.test(scan.className)) { return true } }
    // 如果未找到则返回 false
    return false
  }

  // 标记位置是否有问题
  function badPos(pos, bad) { if (bad) { pos.bad = true; } return pos }

  // 获取两个位置之间的文本内容
  function domTextBetween(cm, from, to, fromLine, toLine) {
    var text = "", closing = false, lineSep = cm.doc.lineSeparator(), extraLinebreak = false;
    // 识别标记函数
    function recognizeMarker(id) { return function (marker) { return marker.id == id; } }
    // 关闭文本内容
    function close() {
      if (closing) {
        text += lineSep;
        if (extraLinebreak) { text += lineSep; }
        closing = extraLinebreak = false;
      }
    }
    // 添加文本内容
    function addText(str) {
      if (str) {
        close();
        text += str;
      }
    }
    function walk(node) {
      // 递归遍历 DOM 节点
      if (node.nodeType == 1) {
        // 如果节点类型为元素节点
        var cmText = node.getAttribute("cm-text");
        // 获取节点的 cm-text 属性值
        if (cmText) {
          // 如果存在 cm-text 属性
          addText(cmText);
          // 将 cm-text 属性值添加到文本中
          return
        }
        var markerID = node.getAttribute("cm-marker"), range;
        // 获取节点的 cm-marker 属性值
        if (markerID) {
          // 如果存在 cm-marker 属性
          var found = cm.findMarks(Pos(fromLine, 0), Pos(toLine + 1, 0), recognizeMarker(+markerID));
          // 在编辑器中查找标记
          if (found.length && (range = found[0].find(0)))
            { addText(getBetween(cm.doc, range.from, range.to).join(lineSep)); }
          // 如果找到标记，则将标记范围内的文本添加到文本中
          return
        }
        if (node.getAttribute("contenteditable") == "false") { return }
        // 如果节点的 contenteditable 属性为 false，则返回
        var isBlock = /^(pre|div|p|li|table|br)$/i.test(node.nodeName);
        // 判断节点是否为块级元素
        if (!/^br$/i.test(node.nodeName) && node.textContent.length == 0) { return }
        // 如果节点不是 br 元素且文本内容长度为 0，则返回
    
        if (isBlock) { close(); }
        // 如果是块级元素，则关闭
        for (var i = 0; i < node.childNodes.length; i++)
          { walk(node.childNodes[i]); }
        // 遍历子节点
    
        if (/^(pre|p)$/i.test(node.nodeName)) { extraLinebreak = true; }
        // 如果节点是 pre 或 p 元素，则设置额外的换行符为 true
        if (isBlock) { closing = true; }
        // 如果是块级元素，则设置关闭标志为 true
      } else if (node.nodeType == 3) {
        // 如果节点类型为文本节点
        addText(node.nodeValue.replace(/\u200b/g, "").replace(/\u00a0/g, " "));
        // 将文本节点的值添加到文本中，替换特殊字符
      }
    }
    for (;;) {
      walk(from);
      // 从起始节点开始遍历
      if (from == to) { break }
      // 如果到达终止节点，则结束遍历
      from = from.nextSibling;
      // 获取下一个兄弟节点
      extraLinebreak = false;
    }
    return text
    // 返回文本
    }
    
    function domToPos(cm, node, offset) {
      var lineNode;
      if (node == cm.display.lineDiv) {
        // 如果节点为编辑器的行 div 元素
        lineNode = cm.display.lineDiv.childNodes[offset];
        // 获取行 div 元素的子节点
        if (!lineNode) { return badPos(cm.clipPos(Pos(cm.display.viewTo - 1)), true) }
        // 如果不存在行节点，则返回错误位置
        node = null; offset = 0;
      } else {
        for (lineNode = node;; lineNode = lineNode.parentNode) {
          // 遍历节点的父节点
          if (!lineNode || lineNode == cm.display.lineDiv) { return null }
          // 如果不存在父节点或父节点为编辑器的行 div 元素，则返回 null
          if (lineNode.parentNode && lineNode.parentNode == cm.display.lineDiv) { break }
          // 如果父节点存在且父节点为编辑器的行 div 元素，则跳出循环
        }
      }
    // 遍历 cm.display.view 数组
    for (var i = 0; i < cm.display.view.length; i++) {
      // 获取当前循环的 lineView
      var lineView = cm.display.view[i];
      // 如果 lineView 的 node 等于 lineNode，则调用 locateNodeInLineView 函数并返回结果
      if (lineView.node == lineNode)
        { return locateNodeInLineView(lineView, node, offset) }
    }
  }

  // 定义 locateNodeInLineView 函数，用于定位节点在行视图中的位置
  function locateNodeInLineView(lineView, node, offset) {
    // 获取 lineView 的文本包裹节点和 bad 标志
    var wrapper = lineView.text.firstChild, bad = false;
    // 如果 node 不存在或不在 wrapper 中，则返回 badPos 函数的结果
    if (!node || !contains(wrapper, node)) { return badPos(Pos(lineNo(lineView.line), 0), true) }
    // 如果 node 等于 wrapper，则设置 bad 为 true，并更新 node 和 offset
    if (node == wrapper) {
      bad = true;
      node = wrapper.childNodes[offset];
      offset = 0;
      // 如果 node 不存在，则返回 badPos 函数的结果
      if (!node) {
        var line = lineView.rest ? lst(lineView.rest) : lineView.line;
        return badPos(Pos(lineNo(line), line.text.length), bad)
      }
    }

    // 获取 textNode 和 topNode
    var textNode = node.nodeType == 3 ? node : null, topNode = node;
    // 如果 textNode 不存在且 node 的子节点长度为 1 且第一个子节点是文本节点，则更新 textNode 和 offset
    if (!textNode && node.childNodes.length == 1 && node.firstChild.nodeType == 3) {
      textNode = node.firstChild;
      if (offset) { offset = textNode.nodeValue.length; }
    }
    // 循环直到 topNode 的父节点等于 wrapper
    while (topNode.parentNode != wrapper) { topNode = topNode.parentNode; }
    // 获取 measure 和 maps
    var measure = lineView.measure, maps = measure.maps;

    // 定义 find 函数，用于查找节点在行视图中的位置
    function find(textNode, topNode, offset) {
      for (var i = -1; i < (maps ? maps.length : 0); i++) {
        var map = i < 0 ? measure.map : maps[i];
        for (var j = 0; j < map.length; j += 3) {
          var curNode = map[j + 2];
          if (curNode == textNode || curNode == topNode) {
            var line = lineNo(i < 0 ? lineView.line : lineView.rest[i]);
            var ch = map[j] + offset;
            if (offset < 0 || curNode != textNode) { ch = map[j + (offset ? 1 : 0)]; }
            return Pos(line, ch)
          }
        }
      }
    }
    // 调用 find 函数并返回结果
    var found = find(textNode, topNode, offset);
    if (found) { return badPos(found, bad) }

    // FIXME this is all really shaky. might handle the few cases it needs to handle, but likely to cause problems
  }
    # 遍历 topNode 后面的兄弟节点，计算距离 dist
    for (var after = topNode.nextSibling, dist = textNode ? textNode.nodeValue.length - offset : 0; after; after = after.nextSibling) {
      # 在兄弟节点中查找目标位置
      found = find(after, after.firstChild, 0);
      # 如果找到目标位置，返回错误位置
      if (found)
        { return badPos(Pos(found.line, found.ch - dist), bad) }
      # 如果未找到目标位置，更新距离 dist
      else
        { dist += after.textContent.length; }
    }
    # 遍历 topNode 前面的兄弟节点，计算距离 dist$1
    for (var before = topNode.previousSibling, dist$1 = offset; before; before = before.previousSibling) {
      # 在兄弟节点中查找目标位置
      found = find(before, before.firstChild, -1);
      # 如果找到目标位置，返回错误位置
      if (found)
        { return badPos(Pos(found.line, found.ch + dist$1), bad) }
      # 如果未找到目标位置，更新距离 dist$1
      else
        { dist$1 += before.textContent.length; }
    }
  }

  // TEXTAREA INPUT STYLE

  # 定义 TextareaInput 类
  var TextareaInput = function(cm) {
    this.cm = cm;
    # 保存上一次输入的内容
    this.prevInput = "";

    # 标志，表示是否预期输入很快就会出现
    this.pollingFast = false;
    # 自重置的轮询器超时
    this.polling = new Delayed();
    # 用于解决 IE 在焦点从文本区域移开时忘记选择的问题
    this.hasSelection = false;
    this.composing = null;
  };

  # 初始化 TextareaInput 实例
  TextareaInput.prototype.init = function (display) {
      var this$1 = this;

    var input = this, cm = this.cm;
    # 创建文本区域
    this.createField(display);
    var te = this.textarea;

    # 将文本区域插入到显示区域中
    display.wrapper.insertBefore(this.wrapper, display.wrapper.firstChild);

    # 在输入时触发事件
    on(te, "input", function () {
      # 在 IE9 及以上版本且存在选择时，重置选择标志
      if (ie && ie_version >= 9 && this$1.hasSelection) { this$1.hasSelection = null; }
      # 轮询输入
      input.poll();
    });

    # 在粘贴时触发事件
    on(te, "paste", function (e) {
      # 如果信号 DOM 事件或处理粘贴事件，则返回
      if (signalDOMEvent(cm, e) || handlePaste(e, cm)) { return }
      # 记录粘贴时间
      cm.state.pasteIncoming = +new Date;
      # 快速轮询输入
      input.fastPoll();
    });
    // 准备复制或剪切操作的处理函数
    function prepareCopyCut(e) {
      // 如果是 DOM 事件，则返回
      if (signalDOMEvent(cm, e)) { return }
      // 如果有选中内容
      if (cm.somethingSelected()) {
        // 设置最后复制的内容为非行级别，获取选中的文本
        setLastCopied({lineWise: false, text: cm.getSelections()});
      } else if (!cm.options.lineWiseCopyCut) {
        // 如果不支持行级别复制剪切，则返回
        return
      } else {
        // 获取可复制的范围
        var ranges = copyableRanges(cm);
        // 设置最后复制的内容为行级别，获取可复制的文本
        setLastCopied({lineWise: true, text: ranges.text});
        // 如果是剪切操作
        if (e.type == "cut") {
          // 设置选中范围为可复制的范围
          cm.setSelections(ranges.ranges, null, sel_dontScroll);
        } else {
          // 清空输入框的上一个输入
          input.prevInput = "";
          // 将可复制的文本放入输入框
          te.value = ranges.text.join("\n");
          // 选中输入框
          selectInput(te);
        }
      }
      // 如果是剪切操作，记录剪切的时间
      if (e.type == "cut") { cm.state.cutIncoming = +new Date; }
    }
    // 绑定剪切事件到处理函数
    on(te, "cut", prepareCopyCut);
    // 绑定复制事件到处理函数
    on(te, "copy", prepareCopyCut);

    // 处理粘贴事件
    on(display.scroller, "paste", function (e) {
      // 如果在小部件内部或是 DOM 事件，则返回
      if (eventInWidget(display, e) || signalDOMEvent(cm, e)) { return }
      // 如果不支持 dispatchEvent
      if (!te.dispatchEvent) {
        // 记录粘贴的时间，聚焦输入框
        cm.state.pasteIncoming = +new Date;
        input.focus();
        return
      }

      // 将 `paste` 事件传递给文本输入框，由其事件监听器处理
      var event = new Event("paste");
      event.clipboardData = e.clipboardData;
      te.dispatchEvent(event);
    });

    // 阻止编辑器内的正常选择操作
    on(display.lineSpace, "selectstart", function (e) {
      // 如果不在小部件内部，则阻止默认事件
      if (!eventInWidget(display, e)) { e_preventDefault(e); }
    });

    // 处理输入法开始输入事件
    on(te, "compositionstart", function () {
      // 获取光标起始位置
      var start = cm.getCursor("from");
      // 如果正在输入中，则清除输入范围
      if (input.composing) { input.composing.range.clear(); }
      // 设置正在输入的信息
      input.composing = {
        start: start,
        range: cm.markText(start, cm.getCursor("to"), {className: "CodeMirror-composing"})
      };
    });
    // 处理输入法结束输入事件
    on(te, "compositionend", function () {
      // 如果正在输入中
      if (input.composing) {
        // 轮询输入
        input.poll();
        // 清除输入范围
        input.composing.range.clear();
        // 清空正在输入的信息
        input.composing = null;
      }
    });
  };

  // 创建输入框
  TextareaInput.prototype.createField = function (_display) {
    // 包装并隐藏输入文本框
    // 创建一个隐藏的文本域，用于接收输入
    this.wrapper = hiddenTextarea();
    // 获取隐藏的文本域元素
    this.textarea = this.wrapper.firstChild;
  };

  TextareaInput.prototype.screenReaderLabelChanged = function (label) {
    // 用于屏幕阅读器的标签，提高可访问性
    if(label) {
      // 设置文本域的 aria-label 属性
      this.textarea.setAttribute('aria-label', label);
    } else {
      // 移除文本域的 aria-label 属性
      this.textarea.removeAttribute('aria-label');
    }
  };

  TextareaInput.prototype.prepareSelection = function () {
    // 重新绘制选择和/或光标
    var cm = this.cm, display = cm.display, doc = cm.doc;
    // 准备选择
    var result = prepareSelection(cm);

    // 将隐藏的文本域移动到光标附近，以防止滚动产生视觉问题
    if (cm.options.moveInputWithCursor) {
      var headPos = cursorCoords(cm, doc.sel.primary().head, "div");
      var wrapOff = display.wrapper.getBoundingClientRect(), lineOff = display.lineDiv.getBoundingClientRect();
      result.teTop = Math.max(0, Math.min(display.wrapper.clientHeight - 10,
                                          headPos.top + lineOff.top - wrapOff.top));
      result.teLeft = Math.max(0, Math.min(display.wrapper.clientWidth - 10,
                                           headPos.left + lineOff.left - wrapOff.left));
    }

    return result
  };

  TextareaInput.prototype.showSelection = function (drawn) {
    var cm = this.cm, display = cm.display;
    // 移除并添加光标元素
    removeChildrenAndAdd(display.cursorDiv, drawn.cursors);
    // 移除并添加选择元素
    removeChildrenAndAdd(display.selectionDiv, drawn.selection);
    if (drawn.teTop != null) {
      // 设置隐藏文本域的位置
      this.wrapper.style.top = drawn.teTop + "px";
      this.wrapper.style.left = drawn.teLeft + "px";
    }
  };

  // 重置输入以对应选择（或为空，当未输入且未选择任何内容时）
  TextareaInput.prototype.reset = function (typing) {
    if (this.contextMenuPending || this.composing) { return }
    var cm = this.cm;
    # 如果编辑器中有选中内容
    if (cm.somethingSelected()) {
      # 保存之前的输入内容
      this.prevInput = "";
      # 获取选中的内容
      var content = cm.getSelection();
      # 将选中的内容设置为文本框的值
      this.textarea.value = content;
      # 如果编辑器处于焦点状态，则选中文本框中的内容
      if (cm.state.focused) { selectInput(this.textarea); }
      # 如果是 IE 并且版本大于等于 9，则记录已选中的内容
      if (ie && ie_version >= 9) { this.hasSelection = content; }
    } 
    # 如果没有选中内容并且不是在输入状态
    else if (!typing) {
      # 清空之前的输入内容和文本框的值
      this.prevInput = this.textarea.value = "";
      # 如果是 IE 并且版本大于等于 9，则清空已选中的内容
      if (ie && ie_version >= 9) { this.hasSelection = null; }
    }
  };

  # 获取文本框对象
  TextareaInput.prototype.getField = function () { return this.textarea };

  # 判断是否支持触摸操作，始终返回 false
  TextareaInput.prototype.supportsTouch = function () { return false };

  # 设置文本框获取焦点
  TextareaInput.prototype.focus = function () {
    # 如果编辑器不是只读模式并且不是在移动设备上或者当前活动元素不是文本框，则尝试让文本框获取焦点
    if (this.cm.options.readOnly != "nocursor" && (!mobile || activeElt() != this.textarea)) {
      try { this.textarea.focus(); }
      catch (e) {} # 如果文本框的样式是 display: none 或者不在 DOM 中，IE8 会抛出异常
    }
  };

  # 让文本框失去焦点
  TextareaInput.prototype.blur = function () { this.textarea.blur(); };

  # 重置文本框的位置
  TextareaInput.prototype.resetPosition = function () {
    this.wrapper.style.top = this.wrapper.style.left = 0;
  };

  # 接收到焦点时执行的操作
  TextareaInput.prototype.receivedFocus = function () { this.slowPoll(); };

  # 以正常的频率轮询输入变化，只要编辑器处于焦点状态就会一直执行
  TextareaInput.prototype.slowPoll = function () {
      var this$1 = this;

    # 如果正在快速轮询，则直接返回
    if (this.pollingFast) { return }
    # 设置轮询间隔，并在间隔结束后执行轮询操作，如果编辑器仍然处于焦点状态，则继续慢速轮询
    this.polling.set(this.cm.options.pollInterval, function () {
      this$1.poll();
      if (this$1.cm.state.focused) { this$1.slowPoll(); }
    });
  };

  # 当有可能会改变文本框内容的事件发生时，加快轮询频率以确保内容快速显示在屏幕上
  TextareaInput.prototype.fastPoll = function () {
    var missed = false, input = this;
    # 标记正在快速轮询
    input.pollingFast = true;
    # 定义轮询函数
    function p() {
      # 执行轮询操作，并记录是否有内容改变
      var changed = input.poll();
      # 如果没有内容改变且之前没有错过轮询，则继续快速轮询
      if (!changed && !missed) {missed = true; input.polling.set(60, p);}
      # 否则，标记快速轮询结束，并继续慢速轮询
      else {input.pollingFast = false; input.slowPoll();}
    }
  // 设置输入的轮询时间为20毫秒，并传入回调函数p
  input.polling.set(20, p);
  };

  // 从文本区域读取输入，并更新文档以匹配
  // 当有选中内容时，它会出现在文本区域中，并且被选中（除非它很大，在这种情况下会使用占位符）
  // 当没有选中内容时，光标位于先前看到的文本之后（可以为空），这些文本存储在prevInput中（我们在输入时不能重置文本区域，因为这会破坏IME）
  TextareaInput.prototype.poll = function () {
      var this$1 = this;

    var cm = this.cm, input = this.textarea, prevInput = this.prevInput;
    // 由于这个函数被频繁调用，当明显没有发生任何变化时，尽可能以最低的成本退出
    // 当文本区域中有大量文本时，hasSelection将为true，此时读取其值将会很昂贵
    if (this.contextMenuPending || !cm.state.focused ||
        (hasSelection(input) && !prevInput && !this.composing) ||
        cm.isReadOnly() || cm.options.disableInput || cm.state.keySeq)
      { return false }

    var text = input.value;
    // 如果没有变化，退出
    if (text == prevInput && !cm.somethingSelected()) { return false }
    // 解决IE9/10中无意义的选择重置问题，以及Mac上一些键组合出现私有区域Unicode字符的问题
    if (ie && ie_version >= 9 && this.hasSelection === text ||
        mac && /[\uf700-\uf7ff]/.test(text)) {
      cm.display.input.reset();
      return false
    }

    if (cm.doc.sel == cm.display.selForContextMenu) {
      var first = text.charCodeAt(0);
      if (first == 0x200b && !prevInput) { prevInput = "\u200b"; }
      if (first == 0x21da) { this.reset(); return this.cm.execCommand("undo") }
    }
    // 找到实际新输入的部分
    var same = 0, l = Math.min(prevInput.length, text.length);
    while (same < l && prevInput.charCodeAt(same) == text.charCodeAt(same)) { ++same; }
    // 在操作中运行，执行输入文本的处理
    runInOp(cm, function () {
      // 应用输入文本到 CodeMirror 编辑器中
      applyTextInput(cm, text.slice(same), prevInput.length - same,
                     null, this$1.composing ? "*compose" : null);

      // 如果文本长度超过1000或者包含换行符，则清空输入框
      if (text.length > 1000 || text.indexOf("\n") > -1) { input.value = this$1.prevInput = ""; }
      else { this$1.prevInput = text; }

      // 如果正在输入中，清除正在输入的标记，并标记新的正在输入的文本
      if (this$1.composing) {
        this$1.composing.range.clear();
        this$1.composing.range = cm.markText(this$1.composing.start, cm.getCursor("to"),
                                           {className: "CodeMirror-composing"});
      }
    });
    // 返回 true
    return true
  };

  // 确保快速轮询
  TextareaInput.prototype.ensurePolled = function () {
    if (this.pollingFast && this.poll()) { this.pollingFast = false; }
  };

  // 处理按键事件
  TextareaInput.prototype.onKeyPress = function () {
    if (ie && ie_version >= 9) { this.hasSelection = null; }
    this.fastPoll();
  };

  // 处理右键菜单事件
  TextareaInput.prototype.onContextMenu = function (e) {
    var input = this, cm = input.cm, display = cm.display, te = input.textarea;
    if (input.contextMenuPending) { input.contextMenuPending(); }
    var pos = posFromMouse(cm, e), scrollPos = display.scroller.scrollTop;
    if (!pos || presto) { return } // Opera is difficult.

    // 如果点击位置在选中文本之外且选项为重置右键菜单时重置选择，则重置当前文本选择
    var reset = cm.options.resetSelectionOnContextMenu;
    if (reset && cm.doc.sel.contains(pos) == -1)
      { operation(cm, setSelection)(cm.doc, simpleSelection(pos), sel_dontScroll); }

    var oldCSS = te.style.cssText, oldWrapperCSS = input.wrapper.style.cssText;
    var wrapperBox = input.wrapper.offsetParent.getBoundingClientRect();
    input.wrapper.style.cssText = "position: static";
    # 设置文本编辑器样式
    te.style.cssText = "position: absolute; width: 30px; height: 30px;\n      top: " + (e.clientY - wrapperBox.top - 5) + "px; left: " + (e.clientX - wrapperBox.left - 5) + "px;\n      z-index: 1000; background: " + (ie ? "rgba(255, 255, 255, .05)" : "transparent") + ";\n      outline: none; border-width: 0; outline: none; overflow: hidden; opacity: .05; filter: alpha(opacity=5);";
    # 保存旧的滚动位置，解决 Chrome 问题
    var oldScrollY;
    if (webkit) { oldScrollY = window.scrollY; } // Work around Chrome issue (#2712)
    # 让输入框获得焦点
    display.input.focus();
    # 在 Chrome 中恢复滚动位置
    if (webkit) { window.scrollTo(null, oldScrollY); }
    # 重置输入框
    display.input.reset();
    # 在 Firefox 中添加 "Select all" 到上下文菜单
    if (!cm.somethingSelected()) { te.value = input.prevInput = " "; }
    # 标记上下文菜单待隐藏
    input.contextMenuPending = rehide;
    # 保存当前选择的文本
    display.selForContextMenu = cm.doc.sel;
    # 清除检测全选的定时器
    clearTimeout(display.detectingSelectAll);

    # 准备全选的 hack
    function prepareSelectAllHack() {
      # 如果支持选择开始位置
      if (te.selectionStart != null) {
        # 检查是否有选中内容
        var selected = cm.somethingSelected();
        # 添加零宽空格以便后续检查是否被选中
        var extval = "\u200b" + (selected ? te.value : "");
        # 设置输入框的值
        te.value = "\u21da"; # 用于捕获上下文菜单的撤销操作
        te.value = extval;
        # 保存上一次的输入值
        input.prevInput = selected ? "" : "\u200b";
        # 设置选择的开始和结束位置
        te.selectionStart = 1; te.selectionEnd = extval.length;
        # 重新设置当前选择的文本
        display.selForContextMenu = cm.doc.sel;
      }
    }
    // 定义函数 rehide
    function rehide() {
      // 如果 input.contextMenuPending 不等于 rehide 函数，则返回
      if (input.contextMenuPending != rehide) { return }
      // 将 input.contextMenuPending 设置为 false
      input.contextMenuPending = false;
      // 恢复 input.wrapper 和 te 的 CSS 样式
      input.wrapper.style.cssText = oldWrapperCSS;
      te.style.cssText = oldCSS;
      // 如果是 IE 并且版本小于 9，则设置滚动条位置
      if (ie && ie_version < 9) { display.scrollbars.setScrollTop(display.scroller.scrollTop = scrollPos); }

      // 尝试检测用户是否选择了全选
      if (te.selectionStart != null) {
        // 如果不是 IE 或者是 IE 并且版本小于 9，则准备全选的 hack
        if (!ie || (ie && ie_version < 9)) { prepareSelectAllHack(); }
        var i = 0, poll = function () {
          if (display.selForContextMenu == cm.doc.sel && te.selectionStart == 0 &&
              te.selectionEnd > 0 && input.prevInput == "\u200b") {
            // 如果满足条件，则执行全选操作
            operation(cm, selectAll)(cm);
          } else if (i++ < 10) {
            // 否则继续轮询
            display.detectingSelectAll = setTimeout(poll, 500);
          } else {
            // 超过轮询次数，则重置状态
            display.selForContextMenu = null;
            display.input.reset();
          }
        };
        // 设置定时器进行轮询
        display.detectingSelectAll = setTimeout(poll, 200);
      }
    }

    // 如果是 IE 并且版本大于等于 9，则准备全选的 hack
    if (ie && ie_version >= 9) { prepareSelectAllHack(); }
    // 如果 captureRightClick 为真
    if (captureRightClick) {
      // 阻止默认事件
      e_stop(e);
      // 定义 mouseup 函数
      var mouseup = function () {
        off(window, "mouseup", mouseup);
        // 延迟 20 毫秒后执行 rehide 函数
        setTimeout(rehide, 20);
      };
      // 绑定 mouseup 事件
      on(window, "mouseup", mouseup);
    } else {
      // 否则延迟 50 毫秒后执行 rehide 函数
      setTimeout(rehide, 50);
    }
  };

  // 定义 TextareaInput 原型的 readOnlyChanged 方法
  TextareaInput.prototype.readOnlyChanged = function (val) {
    // 如果 val 为假，则重置输入框
    if (!val) { this.reset(); }
    // 如果 val 为 "nocursor"，则禁用输入框
    this.textarea.disabled = val == "nocursor";
  };

  // 定义 TextareaInput 原型的 setUneditable 方法
  TextareaInput.prototype.setUneditable = function () {};

  // 设置 TextareaInput 原型的 needsContentAttribute 属性为假
  TextareaInput.prototype.needsContentAttribute = false;

  // 定义 fromTextArea 函数，接受一个 textarea 和 options 参数
  function fromTextArea(textarea, options) {
    // 如果 options 存在，则复制一份
    options = options ? copyObj(options) : {};
    // 设置 options 的 value 为 textarea 的值
    options.value = textarea.value;
    // 如果 options 中没有设置 tabindex 且 textarea 有设置 tabindex，则将其设置为相同的值
    if (!options.tabindex && textarea.tabIndex)
      { options.tabindex = textarea.tabIndex; }
    // 如果 options 中没有设置 placeholder 且 textarea 有设置 placeholder，则将其设置为相同的值
    if (!options.placeholder && textarea.placeholder)
      { options.placeholder = textarea.placeholder; }
    // 设置 autofocus 为真，如果 textarea 被聚焦，或者它有
    // 如果 options.autofocus 为 null，则判断当前是否有焦点元素，如果没有则设置 options.autofocus 为 true
    if (options.autofocus == null) {
      var hasFocus = activeElt();
      options.autofocus = hasFocus == textarea ||
        textarea.getAttribute("autofocus") != null && hasFocus == document.body;
    }

    // 定义保存函数，将 CodeMirror 编辑器中的内容保存到 textarea 中
    function save() {textarea.value = cm.getValue();}

    // 如果 textarea 所在的表单存在，则在表单提交时执行保存函数，并进行一些提交方法的处理
    var realSubmit;
    if (textarea.form) {
      on(textarea.form, "submit", save);
      // 用于处理提交方法的 hack
      if (!options.leaveSubmitMethodAlone) {
        var form = textarea.form;
        realSubmit = form.submit;
        try {
          var wrappedSubmit = form.submit = function () {
            save();
            form.submit = realSubmit;
            form.submit();
            form.submit = wrappedSubmit;
          };
        } catch(e) {}
      }
    }

    // 完成初始化时执行的函数，设置 CodeMirror 编辑器的保存函数和获取 textarea 的方法
    options.finishInit = function (cm) {
      cm.save = save;
      cm.getTextArea = function () { return textarea; };
      cm.toTextArea = function () {
        cm.toTextArea = isNaN; // 防止重复执行
        save();
        textarea.parentNode.removeChild(cm.getWrapperElement());
        textarea.style.display = "";
        if (textarea.form) {
          off(textarea.form, "submit", save);
          if (!options.leaveSubmitMethodAlone && typeof textarea.form.submit == "function")
            { textarea.form.submit = realSubmit; }
        }
      };
    };

    // 隐藏原始的 textarea 元素，创建 CodeMirror 编辑器，并返回
    textarea.style.display = "none";
    var cm = CodeMirror(function (node) { return textarea.parentNode.insertBefore(node, textarea.nextSibling); },
      options);
    return cm
  }

  // 为 CodeMirror 添加一些兼容的属性和方法
  function addLegacyProps(CodeMirror) {
    CodeMirror.off = off;
    CodeMirror.on = on;
    CodeMirror.wheelEventPixels = wheelEventPixels;
    CodeMirror.Doc = Doc;
    CodeMirror.splitLines = splitLinesAuto;
    CodeMirror.countColumn = countColumn;
    CodeMirror.findColumn = findColumn;
    CodeMirror.isWordChar = isWordCharBasic;
    CodeMirror.Pass = Pass;
    # 将 signal 赋值给 CodeMirror.signal
    CodeMirror.signal = signal;
    # 将 Line 赋值给 CodeMirror.Line
    CodeMirror.Line = Line;
    # 将 changeEnd 赋值给 CodeMirror.changeEnd
    CodeMirror.changeEnd = changeEnd;
    # 将 scrollbarModel 赋值给 CodeMirror.scrollbarModel
    CodeMirror.scrollbarModel = scrollbarModel;
    # 将 Pos 赋值给 CodeMirror.Pos
    CodeMirror.Pos = Pos;
    # 将 cmp 赋值给 CodeMirror.cmpPos
    CodeMirror.cmpPos = cmp;
    # 将 modes 赋值给 CodeMirror.modes
    CodeMirror.modes = modes;
    # 将 mimeModes 赋值给 CodeMirror.mimeModes
    CodeMirror.mimeModes = mimeModes;
    # 将 resolveMode 赋值给 CodeMirror.resolveMode
    CodeMirror.resolveMode = resolveMode;
    # 将 getMode 赋值给 CodeMirror.getMode
    CodeMirror.getMode = getMode;
    # 将 modeExtensions 赋值给 CodeMirror.modeExtensions
    CodeMirror.modeExtensions = modeExtensions;
    # 将 extendMode 赋值给 CodeMirror.extendMode
    CodeMirror.extendMode = extendMode;
    # 将 copyState 赋值给 CodeMirror.copyState
    CodeMirror.copyState = copyState;
    # 将 startState 赋值给 CodeMirror.startState
    CodeMirror.startState = startState;
    # 将 innerMode 赋值给 CodeMirror.innerMode
    CodeMirror.innerMode = innerMode;
    # 将 commands 赋值给 CodeMirror.commands
    CodeMirror.commands = commands;
    # 将 keyMap 赋值给 CodeMirror.keyMap
    CodeMirror.keyMap = keyMap;
    # 将 keyName 赋值给 CodeMirror.keyName
    CodeMirror.keyName = keyName;
    # 将 isModifierKey 赋值给 CodeMirror.isModifierKey
    CodeMirror.isModifierKey = isModifierKey;
    # 将 lookupKey 赋值给 CodeMirror.lookupKey
    CodeMirror.lookupKey = lookupKey;
    # 将 normalizeKeyMap 赋值给 CodeMirror.normalizeKeyMap
    CodeMirror.normalizeKeyMap = normalizeKeyMap;
    # 将 StringStream 赋值给 CodeMirror.StringStream
    CodeMirror.StringStream = StringStream;
    # 将 SharedTextMarker 赋值给 CodeMirror.SharedTextMarker
    CodeMirror.SharedTextMarker = SharedTextMarker;
    # 将 TextMarker 赋值给 CodeMirror.TextMarker
    CodeMirror.TextMarker = TextMarker;
    # 将 LineWidget 赋值给 CodeMirror.LineWidget
    CodeMirror.LineWidget = LineWidget;
    # 将 e_preventDefault 赋值给 CodeMirror.e_preventDefault
    CodeMirror.e_preventDefault = e_preventDefault;
    # 将 e_stopPropagation 赋值给 CodeMirror.e_stopPropagation
    CodeMirror.e_stopPropagation = e_stopPropagation;
    # 将 e_stop 赋值给 CodeMirror.e_stop
    CodeMirror.e_stop = e_stop;
    # 将 addClass 赋值给 CodeMirror.addClass
    CodeMirror.addClass = addClass;
    # 将 contains 赋值给 CodeMirror.contains
    CodeMirror.contains = contains;
    # 将 rmClass 赋值给 CodeMirror.rmClass
    CodeMirror.rmClass = rmClass;
    # 将 keyNames 赋值给 CodeMirror.keyNames
    CodeMirror.keyNames = keyNames;
  }

  # 在 CodeMirror 上定义选项
  defineOptions(CodeMirror);

  # 在 CodeMirror 上添加编辑器方法
  addEditorMethods(CodeMirror);

  # 设置在 CodeMirror 的原型上的方法重定向到编辑器的文档
  var dontDelegate = "iter insert remove copy getEditor constructor".split(" ");
  for (var prop in Doc.prototype) { 
    if (Doc.prototype.hasOwnProperty(prop) && indexOf(dontDelegate, prop) < 0) {
      CodeMirror.prototype[prop] = (function(method) {
        return function() {return method.apply(this.doc, arguments)}
  })(Doc.prototype[prop]); } }
  // 为 Doc 对象添加事件相关的方法
  eventMixin(Doc);
  // 定义 CodeMirror 的输入样式
  CodeMirror.inputStyles = {"textarea": TextareaInput, "contenteditable": ContentEditableInput};

  // 定义代码模式
  // 额外的参数被存储为模式的依赖项，这被用于自动加载模式的机制
  CodeMirror.defineMode = function(name/*, mode, …*/) {
    // 如果默认模式不存在且名称不为 "null"，则将默认模式设置为当前名称
    if (!CodeMirror.defaults.mode && name != "null") { CodeMirror.defaults.mode = name; }
    // 应用 defineMode 函数
    defineMode.apply(this, arguments);
  };

  // 定义 MIME 类型
  CodeMirror.defineMIME = defineMIME;

  // 最小的默认模式
  CodeMirror.defineMode("null", function () { return ({token: function (stream) { return stream.skipToEnd(); }}); });
  CodeMirror.defineMIME("text/plain", "null");

  // 扩展

  // 定义扩展方法
  CodeMirror.defineExtension = function (name, func) {
    // 将方法添加到 CodeMirror 的原型上
    CodeMirror.prototype[name] = func;
  };
  // 定义文档扩展方法
  CodeMirror.defineDocExtension = function (name, func) {
    // 将方法添加到 Doc 的原型上
    Doc.prototype[name] = func;
  };

  // 从文本区域创建 CodeMirror 实例
  CodeMirror.fromTextArea = fromTextArea;

  // 添加旧版本的属性
  addLegacyProps(CodeMirror);

  // 设置 CodeMirror 的版本号
  CodeMirror.version = "5.56.0";

  // 返回 CodeMirror 对象
  return CodeMirror;
# 该代码片段缺少上下文，无法确定其作用，需要查看其周围的代码才能添加合适的注释
```