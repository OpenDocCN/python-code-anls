# `ZeroNet\plugins\UiFileManager\media\codemirror\all.js`

```
// 定义一个立即执行函数，传入全局对象和工厂函数
(function (global, factory) {
  // 如果是 CommonJS 环境，将工厂函数的返回值赋给 module.exports
  typeof exports === 'object' && typeof module !== 'undefined' ? module.exports = factory() :
  // 如果是 AMD 环境，使用 define 定义模块
  typeof define === 'function' && define.amd ? define(factory) :
  // 如果是浏览器环境，将工厂函数的返回值赋给全局对象的 CodeMirror 属性
  (global = global || self, global.CodeMirror = factory());
  // 严格模式，确保代码更加严谨
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
  // 检测是否为 QtWebKit
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
  // 某些浏览器使用错误的事件属性来表示右键点击
  var captureRightClick = gecko || (ie && ie_version >= 9);

  // 定义 classTest 函数，用于检测是否存在指定类名
  function classTest(cls) { return new RegExp("(^|\\s)" + cls + "(?:$|\\s)\\s*") }

  // 定义 rmClass 函数，用于移除指定节点的指定类名
  var rmClass = function(node, cls) {
  // 获取当前节点的类名
  var current = node.className;
  // 通过正则表达式匹配类名
  var match = classTest(cls).exec(current);
  // 如果匹配成功
  if (match) {
    // 获取匹配成功后的类名
    var after = current.slice(match.index + match[0].length);
    // 更新节点的类名，去掉匹配成功的部分
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
  // 设置类名
  if (className) { e.className = className; }
  // 设置样式
  if (style) { e.style.cssText = style; }
  // 如果内容是字符串，则创建文本节点添加到元素中
  if (typeof content == "string") { e.appendChild(document.createTextNode(content)); }
  // 如果内容是数组，则遍历添加到元素中
  else if (content) { for (var i = 0; i < content.length; ++i) { e.appendChild(content[i]); } }
  return e
}
// 包装函数，用于创建元素并从可访问性树中移除
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
  // 如果子节点是文本节点，则将其父节点作为子节点
  if (child.nodeType == 3) // Android browser always returns false when child is a textnode
    { child = child.parentNode; }
  // 使用 contains 方法检查父节点是否包含子节点
  if (parent.contains)
    { return parent.contains(child) }
  // 循环遍历检查父节点是否包含子节点
  do {
    if (child.nodeType == 11) { child = child.host; }
    if (child == parent) { return true }
  // 从当前节点开始，向上遍历其父节点，直到没有父节点为止
  } while (child = child.parentNode)
}

function activeElt() {
  // 在 IE 和 Edge 中，访问 document.activeElement 可能会抛出“未指定的错误”。
  // 在加载页面或在 iframe 中访问时，IE < 10 会抛出错误。
  // 在 iframe 中访问时，如果 document.body 不可用，IE > 9 和 Edge 也会抛出错误。
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
  // 如果节点的类名中不包含要添加的类名，则添加该类名
  if (!classTest(cls).test(current)) { node.className += (current ? " " : "") + cls; }
}
function joinClasses(a, b) {
  var as = a.split(" ");
  for (var i = 0; i < as.length; i++)
    // 如果类名 b 中不包含类名 a 中的某个类，则将该类添加到类名 b 中
    { if (as[i] && !classTest(as[i]).test(b)) { b += " " + as[i]; } }
  return b
}

// 定义一个函数 selectInput，用于选择输入框中的文本
var selectInput = function(node) { node.select(); };
// 如果是 iOS 设备，修复 Mobile Safari 中 select() 方法的 bug
if (ios) { selectInput = function(node) { node.selectionStart = 0; node.selectionEnd = node.value.length; }; }
// 如果是 IE 浏览器，忽略 IE10 中的神秘错误
else if (ie) { selectInput = function(node) { try { node.select(); } catch(_e) {} }; }

function bind(f) {
  // 将参数转换为数组
  var args = Array.prototype.slice.call(arguments, 1);
  // 返回一个函数，该函数将在指定的上下文中调用原始函数，并传入指定的参数
  return function(){return f.apply(null, args)}
}

function copyObj(obj, target, overwrite) {
  // 如果目标对象不存在，则创建一个空对象
  if (!target) { target = {}; }
  for (var prop in obj)
    // 如果属性是对象自身的属性，并且不覆盖已有属性，或者目标对象中不存在该属性，则将属性复制到目标对象中
    { if (obj.hasOwnProperty(prop) && (overwrite !== false || !target.hasOwnProperty(prop)))
      { target[prop] = obj[prop]; } }
  return target
}

// 计算字符串中的列偏移量，考虑到制表符
// 主要用于查找缩进
function countColumn(string, end, tabSize, startIndex, startValue) {
    // 如果结束位置为null，则查找第一个非空白字符的位置，如果没有则结束位置为字符串长度
    if (end == null) {
      end = string.search(/[^\s\u00a0]/);
      if (end == -1) { end = string.length; }
    }
    // 从给定的开始位置和值开始循环，查找下一个制表符的位置并计算偏移量
    for (var i = startIndex || 0, n = startValue || 0;;) {
      var nextTab = string.indexOf("\t", i);
      // 如果没有找到制表符或者制表符位置超过结束位置，则返回偏移量加上剩余字符的长度
      if (nextTab < 0 || nextTab >= end)
        { return n + (end - i) }
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
    // 如果当前时间超过设定的时间，则执行函数，否则延迟执行
    if (self.time <= +new Date) {
      self.f();
    } else {
      setTimeout(self.handler, self.time - +new Date);
    }
  };
  // 设置延迟执行函数的执行时间和函数
  Delayed.prototype.set = function (ms, f) {
    this.f = f;
    var time = +new Date + ms;
    if (!this.id || time < this.time) {
      clearTimeout(this.id);
      this.id = setTimeout(this.handler, ms);
      this.time = time;
    }
  };

  // 查找数组中指定元素的索引
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

  // countColumn的反函数，查找与特定列对应的偏移量
  function findColumn(string, goal, tabSize) {
    // 初始化变量 pos 和 col，用于记录当前位置和列数
    for (var pos = 0, col = 0;;) {
      // 查找下一个制表符的位置
      var nextTab = string.indexOf("\t", pos);
      // 如果没有找到制表符，则将下一个制表符位置设置为字符串的长度
      if (nextTab == -1) { nextTab = string.length; }
      // 计算跳过的字符数
      var skipped = nextTab - pos;
      // 如果下一个制表符位置等于字符串的长度，或者当前列数加上跳过的字符数大于等于目标列数
      if (nextTab == string.length || col + skipped >= goal)
        { return pos + Math.min(skipped, goal - col) }
      // 更新列数
      col += nextTab - pos;
      // 将列数调整为下一个制表符的位置
      col += tabSize - (col % tabSize);
      // 更新位置为下一个制表符的位置加一
      pos = nextTab + 1;
      // 如果当前列数大于等于目标列数，则返回当前位置
      if (col >= goal) { return pos }
    }
  }

  // 初始化空字符串数组
  var spaceStrs = [""];
  // 生成指定长度的空格字符串
  function spaceStr(n) {
    while (spaceStrs.length <= n)
      { spaceStrs.push(lst(spaceStrs) + " "); }
    return spaceStrs[n]
  }

  // 返回数组的最后一个元素
  function lst(arr) { return arr[arr.length-1] }

  // 对数组中的每个元素执行指定的函数
  function map(array, f) {
    var out = [];
    for (var i = 0; i < array.length; i++) { out[i] = f(array[i], i); }
    return out
  }

  // 将值按照指定的顺序插入到已排序的数组中
  function insertSorted(array, value, score) {
    var pos = 0, priority = score(value);
    while (pos < array.length && score(array[pos]) <= priority) { pos++; }
    array.splice(pos, 0, value);
  }

  // 空函数
  function nothing() {}

  // 创建一个对象，继承自指定的基础对象，并且拥有指定的属性
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

  // 匹配非ASCII字符的正则表达式
  var nonASCIISingleCaseWordChar = /[\u00df\u0587\u0590-\u05f4\u0600-\u06ff\u3040-\u309f\u30a0-\u30ff\u3400-\u4db5\u4e00-\u9fcc\uac00-\ud7af]/;
  // 判断字符是否为单词字符的基本函数
  function isWordCharBasic(ch) {
    return /\w/.test(ch) || ch > "\x80" &&
      (ch.toUpperCase() != ch.toLowerCase() || nonASCIISingleCaseWordChar.test(ch))
  }
  // 判断字符是否为单词字符的函数
  function isWordChar(ch, helper) {
    // 如果没有辅助函数，则使用基本的判断函数
    if (!helper) { return isWordCharBasic(ch) }
    // 如果辅助函数中包含 \w，并且字符是单词字符，则返回 true
    if (helper.source.indexOf("\\w") > -1 && isWordCharBasic(ch)) { return true }
    // 否则使用辅助函数进行判断
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
  // N (ON):  其他中性

  // 如果字符按照它们在视觉上出现的顺序排列（从左到右），则返回 null，否则返回一个包含部分实现的顺序的数组（{from, to, level} 对象）。
  var bidiOrdering = (function() {
    // 代码点 0 到 0xff 的字符类型
    var lowTypes = "bbbbbbbbbtstwsbbbbbbbbbbbbbbssstwNN%%%NNNNNN,N,N1111111111NNNNNNNLLLLLLLLLLLLLLLLLLLLLLLLLLNNNNNNLLLLLLLLLLLLLLLLLLLLLLLLLLNNNNbbbbbbsbbbbbbbbbbbbbbbbbbbbbbbbbb,N%%%%NNNNLNNNNN%%11NLNNN1LNNNNNLLLLLLLLLLLLLLLLLLLLLLLNLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLN";
    // 代码点 0x600 到 0x6f9 的字符类型
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
  // 遍历处理程序数组，依次调用每个处理程序并传入参数
  for (var i = 0; i < handlers.length; ++i) { handlers[i].apply(null, args); }
  }

  // 通过在编辑器上注册一个（非DOM）处理程序来覆盖CodeMirror处理的DOM事件，并在该处理程序中阻止默认行为
  function signalDOMEvent(cm, e, override) {
    // 如果事件是字符串类型，则转换为对象类型，并添加preventDefault方法
    if (typeof e == "string")
      { e = {type: e, preventDefault: function() { this.defaultPrevented = true; }}; }
    // 触发事件处理程序
    signal(cm, override || e.type, cm, e);
    // 返回事件是否被阻止默认行为或被CodeMirror忽略
    return e_defaultPrevented(e) || e.codemirrorIgnore
  }

  // 触发光标活动事件处理程序
  function signalCursorActivity(cm) {
    // 获取光标活动事件处理程序数组
    var arr = cm._handlers && cm._handlers.cursorActivity;
    // 如果数组不存在，则直接返回
    if (!arr) { return }
    // 获取当前操作的光标活动处理程序集合
    var set = cm.curOp.cursorActivityHandlers || (cm.curOp.cursorActivityHandlers = []);
    // 遍历光标活动事件处理程序数组，将不在当前操作集合中的处理程序添加进去
    for (var i = 0; i < arr.length; ++i) { if (indexOf(set, arr[i]) == -1)
      { set.push(arr[i]); } }
  }

  // 判断指定事件类型的处理程序是否存在
  function hasHandler(emitter, type) {
    return getHandlers(emitter, type).length > 0
  }

  // 为构造函数的原型添加on和off方法，以便更方便地在这些对象上注册事件
  function eventMixin(ctor) {
    ctor.prototype.on = function(type, f) {on(this, type, f);};
    ctor.prototype.off = function(type, f) {off(this, type, f);};
  }

  // 由于我们仍然支持古老的IE版本，因此需要一些兼容性包装器

  // 阻止事件的默认行为
  function e_preventDefault(e) {
    if (e.preventDefault) { e.preventDefault(); }
    else { e.returnValue = false; }
  }
  // 阻止事件的传播
  function e_stopPropagation(e) {
    if (e.stopPropagation) { e.stopPropagation(); }
    else { e.cancelBubble = true; }
  }
  // 判断事件是否被阻止了默认行为
  function e_defaultPrevented(e) {
    return e.defaultPrevented != null ? e.defaultPrevented : e.returnValue == false
  }
  // 同时阻止事件的默认行为和传播
  function e_stop(e) {e_preventDefault(e); e_stopPropagation(e);}

  // 获取事件的目标元素
  function e_target(e) {return e.target || e.srcElement}
  // 获取事件的按钮信息
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
  function zeroWidthElement(measure) {
    // 如果 zwspSupported 为 null
    if (zwspSupported == null) {
      // 创建一个测试用的 span 元素
      var test = elt("span", "\u200b");
      // 将测试元素添加到 measure 中，并检查其高度是否为 0
      removeChildrenAndAdd(measure, elt("span", [test, document.createTextNode("x")]));
      if (measure.firstChild.offsetHeight != 0)
        { zwspSupported = test.offsetWidth <= 1 && test.offsetHeight > 2 && !(ie && ie_version < 8); }
    }
    // 如果支持零宽空格，创建一个 span 元素并设置属性
    var node = zwspSupported ? elt("span", "\u200b") :
      elt("span", "\u00a0", null, "display: inline-block; width: 1px; margin-right: -1px");
    node.setAttribute("cm-text", "");
    return node
  }

  // 检测 IE 对双向文本的矩形报告是否有问题
  var badBidiRects;
  function hasBadBidiRects(measure) {
    // 如果 badBidiRects 不为 null，返回其值
    if (badBidiRects != null) { return badBidiRects }
    // 创建包含双向文本的文本节点
    var txt = removeChildrenAndAdd(measure, document.createTextNode("A\u062eA"));
    // 获取两个字符的边界矩形
    var r0 = range(txt, 0, 1).getBoundingClientRect();
    var r1 = range(txt, 1, 2).getBoundingClientRect();
    // 移除文本节点
    removeChildren(measure);
    // 如果 r0 为 null 或者左右边界相等，返回 false
    if (!r0 || r0.left == r0.right) { return false } // Safari 在某些情况下返回 null (#2780)
    // 返回是否存在双向文本矩形问题
    return badBidiRects = (r1.right - r0.right < 3)
  }

  // 检测 "".split 是否是 IE 的有问题版本，如果是，提供替代的分割行的方法
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

  // 检查是否存在复制事件
  var hasCopyEvent = (function () {
    var e = elt("div");
    // 检查是否支持oncopy事件
    if ("oncopy" in e) { return true }
    // 设置oncopy属性并检查是否为函数
    e.setAttribute("oncopy", "return;");
    return typeof e.oncopy == "function"
  })();

  // 初始化变量badZoomedRects
  var badZoomedRects = null;
  // 检查是否存在错误的缩放矩形
  function hasBadZoomedRects(measure) {
    // 如果已经有缓存的值，则直接返回
    if (badZoomedRects != null) { return badZoomedRects }
    // 创建一个span元素，并在其中添加一个字符
    var node = removeChildrenAndAdd(measure, elt("span", "x"));
    // 获取正常状态下的矩形
    var normal = node.getBoundingClientRect();
    // 获取选中状态下的矩形
    var fromRange = range(node, 0, 1).getBoundingClientRect();
    // 检查是否存在错误的缩放矩形
    return badZoomedRects = Math.abs(normal.left - fromRange.left) > 1
  }

  // 已知的模式，按名称和MIME类型存储
  var modes = {}, mimeModes = {};

  // 定义模式
  // 额外的参数存储为模式的依赖项，用于自动加载模式的机制
  function defineMode(name, mode) {
    if (arguments.length > 2)
      { mode.dependencies = Array.prototype.slice.call(arguments, 2); }
    modes[name] = mode;
  }

  // 定义MIME类型
  function defineMIME(mime, spec) {
    // 将给定的 MIME 类型和对应的模式配置对象存储到 mimeModes 对象中
    mimeModes[mime] = spec;
  }

  // 根据给定的 MIME 类型、{name, ...options} 配置对象或者名称字符串，返回一个模式配置对象
  function resolveMode(spec) {
    // 如果 spec 是字符串并且存在于 mimeModes 对象中，则将其替换为对应的模式配置对象
    if (typeof spec == "string" && mimeModes.hasOwnProperty(spec)) {
      spec = mimeModes[spec];
    } 
    // 如果 spec 存在并且具有名称属性，并且该名称存在于 mimeModes 对象中，则根据找到的配置对象和 spec 创建新的配置对象
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
    // 如果 spec 是字符串，则返回一个具有该字符串为名称的模式配置对象
    if (typeof spec == "string") { return {name: spec} }
    // 否则返回 spec 或者一个具有名称为 "null" 的模式配置对象
    else { return spec || {name: "null"} }
  }

  // 根据模式配置对象，找到并初始化一个实际的模式对象
  function getMode(options, spec) {
    // 根据 resolveMode 返回的配置对象找到对应的模式工厂函数
    spec = resolveMode(spec);
    var mfactory = modes[spec.name];
    // 如果找不到对应的模式工厂函数，则返回 "text/plain" 的模式对象
    if (!mfactory) { return getMode(options, "text/plain") }
    // 根据模式工厂函数和配置对象创建模式对象
    var modeObj = mfactory(options, spec);
    // 如果存在模式扩展属性，则将其添加到模式对象中
    if (modeExtensions.hasOwnProperty(spec.name)) {
      var exts = modeExtensions[spec.name];
      for (var prop in exts) {
        if (!exts.hasOwnProperty(prop)) { continue }
        if (modeObj.hasOwnProperty(prop)) { modeObj["_" + prop] = modeObj[prop]; }
        modeObj[prop] = exts[prop];
      }
    }
    modeObj.name = spec.name;
    // 如果配置对象具有 helperType 属性，则将其添加到模式对象中
    if (spec.helperType) { modeObj.helperType = spec.helperType; }
    // 如果配置对象具有 modeProps 属性，则将其添加到模式对象中
    if (spec.modeProps) { for (var prop$1 in spec.modeProps)
      { modeObj[prop$1] = spec.modeProps[prop$1]; } }

    return modeObj
  }

  // 可以用于从模式定义之外向模式对象附加属性
  var modeExtensions = {};
  function extendMode(mode, properties) {
  // 如果给定的 modeExtensions 对象中存在指定 mode 的扩展，则使用该扩展，否则创建一个新的扩展对象
  var exts = modeExtensions.hasOwnProperty(mode) ? modeExtensions[mode] : (modeExtensions[mode] = {});
  // 将 properties 对象的属性复制到 exts 对象中
  copyObj(properties, exts);
}

// 复制给定 mode 的状态对象
function copyState(mode, state) {
  // 如果 state 为 true，则直接返回 state
  if (state === true) { return state }
  // 如果 mode 中有 copyState 方法，则调用该方法复制状态
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

// 根据给定的 mode 和 state，找到指定位置的内部 mode 和 state
function innerMode(mode, state) {
  var info;
  while (mode.innerMode) {
    info = mode.innerMode(state);
    if (!info || info.mode == mode) { break }
    state = info.state;
    mode = info.mode;
  }
  return info || {mode: mode, state: state}
}

// 根据给定的 mode 和参数 a1、a2，返回起始状态
function startState(mode, a1, a2) {
  // 如果 mode 中有 startState 方法，则调用该方法返回起始状态，否则返回 true
  return mode.startState ? mode.startState(a1, a2) : true
}

// 字符串流对象，提供辅助函数以使解析器更加简洁
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
// 如果当前位置的字符匹配给定的参数 match，则将位置移动到下一个字符
StringStream.prototype.eat = function (match) {
  var ch = this.string.charAt(this.pos);
  var ok;
  if (typeof match == "string") { ok = ch == match; }
  else { ok = ch && (match.test ? match.test(ch) : match(ch)); }
  // ...
};
    # 如果条件为真，则增加位置计数并返回字符
    if (ok) {++this.pos; return ch}
  };
  # 从当前位置开始，一直吃掉匹配的字符
  StringStream.prototype.eatWhile = function (match) {
    var start = this.pos;
    while (this.eat(match)){}
    return this.pos > start
  };
  # 从当前位置开始吃掉空白字符
  StringStream.prototype.eatSpace = function () {
    var start = this.pos;
    while (/[\s\u00a0]/.test(this.string.charAt(this.pos))) { ++this.pos; }
    return this.pos > start
  };
  # 跳过当前位置到字符串末尾的所有字符
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
  // 定义 StringStream 对象的 hideFirstChars 方法，用于隐藏字符串的前 n 个字符
  StringStream.prototype.hideFirstChars = function (n, inner) {
    this.lineStart += n;
    try { return inner() }
    finally { this.lineStart -= n; }
  };
  // 定义 StringStream 对象的 lookAhead 方法，用于预览后面的 n 个字符
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
  // 如果存在高度差，则更新所有父级节点的高度
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

  // 为给定的行号添加行号格式化
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

// 保存上下文的构造函数
var SavedContext = function(state, lookAhead) {
  this.state = state;
  this.lookAhead = lookAhead;
};

// 上下文的构造函数
var Context = function(doc, state, line, lookAhead) {
  this.state = state;
  this.doc = doc;
  this.line = line;
  this.maxLookAhead = lookAhead || 0;
  this.baseTokens = null;
  this.baseTokenPos = 1;
};

// 上下文原型链上的方法，用于预览指定行数后的内容
Context.prototype.lookAhead = function (n) {
  var line = this.doc.getLine(this.line + n);
  if (line != null && n > this.maxLookAhead) { this.maxLookAhead = n; }
  return line
};

// 上下文原型链上的方法，用于获取基本令牌
Context.prototype.baseToken = function (n) {
  if (!this.baseTokens) { return null }
}
    // 当基础令牌数组中的值小于等于 n 时，循环执行以下操作
    while (this.baseTokens[this.baseTokenPos] <= n)
      { this.baseTokenPos += 2; }
    // 获取当前位置的类型
    var type = this.baseTokens[this.baseTokenPos + 1];
    // 返回一个对象，包含类型和大小信息
    return {type: type && type.replace(/( |^)overlay .*/, ""),
            size: this.baseTokens[this.baseTokenPos] - n}
  };

  // 增加行数
  Context.prototype.nextLine = function () {
    this.line++;
    // 如果最大向前查看数大于 0，则减少最大向前查看数
    if (this.maxLookAhead > 0) { this.maxLookAhead--; }
  };

  // 从保存的上下文中创建新的上下文
  Context.fromSaved = function (doc, saved, line) {
    // 如果保存的上下文是 SavedContext 类型，则返回新的上下文
    if (saved instanceof SavedContext)
      { return new Context(doc, copyState(doc.mode, saved.state), line, saved.lookAhead) }
    // 否则，返回新的上下文
    else
      { return new Context(doc, copyState(doc.mode, saved), line) }
  };

  // 保存当前上下文状态
  Context.prototype.save = function (copy) {
    // 如果需要复制状态，则复制当前模式的状态
    var state = copy !== false ? copyState(this.doc.mode, this.state) : this.state;
    // 如果最大向前查看数大于 0，则返回 SavedContext 对象，否则返回状态
    return this.maxLookAhead > 0 ? new SavedContext(state, this.maxLookAhead) : state
  };


  // 计算样式数组（以模式生成开始，后面是结束位置和样式字符串的成对数组），用于对行上的标记进行高亮显示
  function highlightLine(cm, line, context, forceToEnd) {
    // 样式数组始终以标识其基础模式/叠加模式的数字开始（用于简单的失效处理）
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
      // 获取当前覆盖模式
      var overlay = cm.state.overlays[o], i = 1, at = 0;
      // 设置状态为 true
      context.state = true;
      // 运行代码模式，传入参数 cm, line.text, overlay.mode, context, 回调函数
      runMode(cm, line.text, overlay.mode, context, function (end, style) {
        // 获取起始位置
        var start = i;
        // 确保当前位置有一个标记结束，并且 i 指向它
        while (at < end) {
          var i_end = st[i];
          if (i_end > end)
            { st.splice(i, 1, end, st[i+1], i_end); }
          i += 2;
          at = Math.min(end, i_end);
        }
        // 如果没有样式，直接返回
        if (!style) { return }
        // 如果覆盖模式是不透明的
        if (overlay.opaque) {
          st.splice(start, i - start, end, "overlay " + style);
          i = start + 2;
        } else {
          // 如果覆盖模式是透明的
          for (; start < i; start += 2) {
            var cur = st[start+1];
            st[start+1] = (cur ? cur + " " : "") + "overlay " + style;
          }
        }
      }, lineClasses);
      // 恢复状态
      context.state = state;
      // 将 context.baseTokens 设置为 null
      context.baseTokens = null;
      // 将 context.baseTokenPos 设置为 1
      context.baseTokenPos = 1;
    };

    // 循环遍历 cm.state.overlays 数组
    for (var o = 0; o < cm.state.overlays.length; ++o) loop( o );

    // 返回结果对象，包含样式和类
    return {styles: st, classes: lineClasses.bgClass || lineClasses.textClass ? lineClasses : null}
  }

  // 获取行的样式
  function getLineStyles(cm, line, updateFrontier) {
    // 如果行的样式不存在或者第一个样式不等于 cm.state.modeGen
    if (!line.styles || line.styles[0] != cm.state.modeGen) {
      // 获取上下文
      var context = getContextBefore(cm, lineNo(line));
      // 如果行文本长度大于最大高亮长度，并且复制状态
      var resetState = line.text.length > cm.options.maxHighlightLength && copyState(cm.doc.mode, context.state);
      // 高亮行
      var result = highlightLine(cm, line, context);
      // 如果有复制状态，将上下文状态设置为复制状态
      if (resetState) { context.state = resetState; }
      // 保存行的状态
      line.stateAfter = context.save(!resetState);
      // 设置行的样式为结果的样式
      line.styles = result.styles;
      // 如果有结果的类，将行的样式类设置为结果的类，否则设置为 null
      if (result.classes) { line.styleClasses = result.classes; }
      else if (line.styleClasses) { line.styleClasses = null; }
      // 如果更新边界等于文档的高亮边界，将文档的模式边界设置为最大值
      if (updateFrontier === cm.doc.highlightFrontier)
        { cm.doc.modeFrontier = Math.max(cm.doc.modeFrontier, ++cm.doc.highlightFrontier); }
    }
  // 返回行的样式
  return line.styles
}

// 获取指定行号之前的上下文
function getContextBefore(cm, n, precise) {
  var doc = cm.doc, display = cm.display;
  // 如果文档没有模式的起始状态，则返回一个新的上下文
  if (!doc.mode.startState) { return new Context(doc, true, n) }
  // 查找起始行
  var start = findStartLine(cm, n, precise);
  // 如果起始行大于文档的第一行，并且起始行的前一行有保存的状态，则使用保存的状态创建上下文，否则创建一个新的上下文
  var saved = start > doc.first && getLine(doc, start - 1).stateAfter;
  var context = saved ? Context.fromSaved(doc, saved, start) : new Context(doc, startState(doc.mode), start);

  // 遍历起始行到指定行之间的每一行
  doc.iter(start, n, function (line) {
    // 处理每一行的文本，更新上下文状态
    processLine(cm, line.text, context);
    var pos = context.line;
    // 如果当前行是最后一行，或者是 5 的倍数，或者在显示区域内，则保存当前上下文状态，否则置为 null
    line.stateAfter = pos == n - 1 || pos % 5 == 0 || pos >= display.viewFrom && pos < display.viewTo ? context.save() : null;
    // 移动到下一行
    context.nextLine();
  });
  // 如果需要精确模式，则更新文档的模式边界
  if (precise) { doc.modeFrontier = context.line; }
  return context
}

// 轻量级的高亮处理，处理当前行的文本，但不保存样式数组。用于当前不可见的行
function processLine(cm, text, context, startAt) {
  var mode = cm.doc.mode;
  var stream = new StringStream(text, cm.options.tabSize, context);
  stream.start = stream.pos = startAt || 0;
  // 如果文本为空，则调用空行处理函数
  if (text == "") { callBlankLine(mode, context.state); }
  // 循环处理文本中的每个 token
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

// 读取 token
function readToken(mode, stream, state, inner) {
  for (var i = 0; i < 10; i++) {
    if (inner) { inner[0] = innerMode(mode, state).mode; }
    var style = mode.token(stream, state);
    if (stream.pos > stream.start) { return style }
  }
  throw new Error("Mode " + mode.name + " failed to advance stream.")
}

// Token 对象
var Token = function(stream, type, state) {
  this.start = stream.start; this.end = stream.pos;
}
    # 将当前流的内容赋值给对象的字符串属性
    this.string = stream.current();
    # 如果提供了类型参数，则将其赋值给对象的类型属性，否则为null
    this.type = type || null;
    # 将提供的状态参数赋值给对象的状态属性
    this.state = state;
  };

  # 为getTokenAt和getLineTokens提供的实用工具函数
  function takeToken(cm, pos, precise, asArray) {
    # 获取编辑器文档对象
    var doc = cm.doc, mode = doc.mode, style;
    # 对位置进行裁剪，确保在文档范围内
    pos = clipPos(doc, pos);
    # 获取指定行的文本内容
    var line = getLine(doc, pos.line), context = getContextBefore(cm, pos.line, precise);
    # 创建一个新的字符串流对象
    var stream = new StringStream(line.text, cm.options.tabSize, context), tokens;
    # 如果asArray为true，则初始化tokens为一个空数组
    if (asArray) { tokens = []; }
    # 当asArray为true或者流的位置小于指定列位置并且没有到达行尾时，循环执行以下操作
    while ((asArray || stream.pos < pos.ch) && !stream.eol()) {
      # 设置流的起始位置
      stream.start = stream.pos;
      # 从指定模式中读取一个标记的样式
      style = readToken(mode, stream, context.state);
      # 如果asArray为true，则将新的标记对象添加到tokens数组中
      if (asArray) { tokens.push(new Token(stream, style, copyState(doc.mode, context.state))); }
    }
    # 如果asArray为true，则返回tokens数组，否则返回一个新的标记对象
    return asArray ? tokens : new Token(stream, style, context.state)
  }

  # 提取行的类
  function extractLineClasses(type, output) {
    # 如果提供了类型参数，则执行以下操作
    if (type) { for (;;) {
      # 从类型中匹配行类的正则表达式
      var lineClass = type.match(/(?:^|\s+)line-(background-)?(\S+)/);
      # 如果没有匹配到行类，则跳出循环
      if (!lineClass) { break }
      # 从类型中移除匹配到的行类
      type = type.slice(0, lineClass.index) + type.slice(lineClass.index + lineClass[0].length);
      # 根据行类的类型确定是背景类还是文本类，并将其添加到输出对象中
      var prop = lineClass[1] ? "bgClass" : "textClass";
      if (output[prop] == null)
        { output[prop] = lineClass[2]; }
      else if (!(new RegExp("(?:^|\\s)" + lineClass[2] + "(?:$|\\s)")).test(output[prop]))
        { output[prop] += " " + lineClass[2]; }
    } }
    # 返回处理后的类型
    return type
  }

  # 运行给定模式的解析器，对一行文本进行处理
  function runMode(cm, text, mode, context, f, lineClasses, forceToEnd) {
    # 获取模式的flattenSpans属性，如果未定义则使用编辑器选项中的flattenSpans属性
    var flattenSpans = mode.flattenSpans;
    if (flattenSpans == null) { flattenSpans = cm.options.flattenSpans; }
    # 初始化当前标记的起始位置和样式
    var curStart = 0, curStyle = null;
    # 创建一个新的字符串流对象
    var stream = new StringStream(text, cm.options.tabSize, context), style;
    # 如果编辑器文本为空，则提取行的类
    var inner = cm.options.addModeClass && [null];
    if (text == "") { extractLineClasses(callBlankLine(mode, context.state), lineClasses); }
    // 当流未到达行尾时执行循环
    while (!stream.eol()) {
      // 如果流的位置超过了最大高亮长度
      if (stream.pos > cm.options.maxHighlightLength) {
        // 设置扁平化标记为false
        flattenSpans = false;
        // 如果强制结束标记为true，则处理当前行的文本
        if (forceToEnd) { processLine(cm, text, context, stream.pos); }
        // 将流的位置设置为文本的长度
        stream.pos = text.length;
        // 样式设置为null
        style = null;
      } else {
        // 从流中读取标记，并提取行类
        style = extractLineClasses(readToken(mode, stream, context.state, inner), lineClasses);
      }
      // 如果inner存在
      if (inner) {
        // 获取inner数组的第一个元素的name属性
        var mName = inner[0].name;
        // 如果mName存在，则将样式设置为"m-" + mName + " " + style或者mName
        if (mName) { style = "m-" + (style ? mName + " " + style : mName); }
      }
      // 如果不需要扁平化或者当前样式不等于style
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
  function findStartLine(cm, n, precise) {
    var minindent, minline, doc = cm.doc;
    // 如果precise为true，则lim为-1，否则lim为n减去1000或100（取决于是否存在内部模式）
    var lim = precise ? -1 : n - (cm.doc.mode.innerMode ? 1000 : 100);
    // 从当前行号 n 开始向上搜索，直到搜索到 lim 为止
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
    // 初始化 start 为文档的第一行号
    var start = doc.first;
    // 从当前行号 n - 1 开始向上搜索
    for (var line = n - 1; line > start; line--) {
      // 获取当前行的状态
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

  // 当这些特性未被使用时，优化一些代码
  var sawReadOnlySpans = false, sawCollapsedSpans = false;

  // 设置 sawReadOnlySpans 为 true
  function seeReadOnlySpans() {
    sawReadOnlySpans = true;
  }

  // 设置 sawCollapsedSpans 为 true
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
  // 遍历 spans 数组，将不等于当前 span 的元素添加到 r 数组中
  for (var i = 0; i < spans.length; ++i)
    { if (spans[i] != span) { (r || (r = [])).push(spans[i]); } }
  // 返回 r 数组
  return r
}
// 为一行添加一个标记的 span
function addMarkedSpan(line, span) {
  // 如果 line.markedSpans 存在，则将 span 添加到数组末尾，否则创建一个只包含 span 的数组
  line.markedSpans = line.markedSpans ? line.markedSpans.concat([span]) : [span];
  // 将 span.marker 附加到 line 上
  span.marker.attachLine(line);
}

// 用于调整文档变化时标记的算法。这些函数在给定字符位置处切割 spans 数组，返回剩余的数组块（如果没有剩余则返回 undefined）
function markedSpansBefore(old, startCh, isInsert) {
  var nw;
  // 如果 old 存在，则遍历 old 数组
  if (old) { for (var i = 0; i < old.length; ++i) {
    var span = old[i], marker = span.marker;
    // 判断 span 的起始位置是否在 startCh 之前，如果是则将其添加到 nw 数组中
    var startsBefore = span.from == null || (marker.inclusiveLeft ? span.from <= startCh : span.from < startCh);
    if (startsBefore || span.from == startCh && marker.type == "bookmark" && (!isInsert || !span.marker.insertLeft)) {
      // 如果 nw 不存在，则创建一个空数组，然后将新的 MarkedSpan 对象添加到数组中
      var endsAfter = span.to == null || (marker.inclusiveRight ? span.to >= startCh : span.to > startCh)
      ;(nw || (nw = [])).push(new MarkedSpan(marker, span.from, endsAfter ? null : span.to));
    }
  } }
  // 返回 nw 数组
  return nw
}
// 与 markedSpansBefore 类似，用于处理 endCh 之后的 spans 数组
function markedSpansAfter(old, endCh, isInsert) {
  var nw;
  // 如果 old 存在，则遍历 old 数组
  if (old) { for (var i = 0; i < old.length; ++i) {
    var span = old[i], marker = span.marker;
    // 判断 span 的结束位置是否在 endCh 之后，如果是则将其添加到 nw 数组中
    var endsAfter = span.to == null || (marker.inclusiveRight ? span.to >= endCh : span.to > endCh);
    if (endsAfter || span.from == endCh && marker.type == "bookmark" && (!isInsert || span.marker.insertLeft)) {
      // 如果 nw 不存在，则创建一个空数组，然后将新的 MarkedSpan 对象添加到数组中
      var startsBefore = span.from == null || (marker.inclusiveLeft ? span.from <= endCh : span.from < endCh)
      ;(nw || (nw = [])).push(new MarkedSpan(marker, startsBefore ? null : span.from - endCh,
                                            span.to == null ? null : span.to - endCh));
    }
  } }
  // 返回新的文本行
  return nw
}

// 根据变化对象，计算覆盖发生变化的行的新标记范围集合
// 删除完全在变化内部的范围，重新连接出现在变化两侧的属于同一标记的范围，并截断部分在变化内部的范围
// 返回一个包含每行（变化后）的一个元素的标记范围数组的数组
function stretchSpansOverChange(doc, change) {
  if (change.full) { return null }
  var oldFirst = isLine(doc, change.from.line) && getLine(doc, change.from.line).markedSpans;
  var oldLast = isLine(doc, change.to.line) && getLine(doc, change.to.line).markedSpans;
  if (!oldFirst && !oldLast) { return null }

  var startCh = change.from.ch, endCh = change.to.ch, isInsert = cmp(change.from, change.to) == 0;
  // 获取在两侧“突出”的范围
  var first = markedSpansBefore(oldFirst, startCh, isInsert);
  var last = markedSpansAfter(oldLast, endCh, isInsert);

  // 接下来，合并这两端
  var sameLine = change.text.length == 1, offset = lst(change.text).length + (sameLine ? startCh : 0);
  if (first) {
    // 修复 first 的 .to 属性
    for (var i = 0; i < first.length; ++i) {
      var span = first[i];
      if (span.to == null) {
        var found = getMarkedSpanFor(last, span.marker);
        if (!found) { span.to = startCh; }
        else if (sameLine) { span.to = found.to == null ? null : found.to + offset; }
      }
    }
  }
}
    // 如果存在last，则对last中的每个span进行修正
    if (last) {
      for (var i$1 = 0; i$1 < last.length; ++i$1) {
        var span$1 = last[i$1];
        // 如果span的to属性不为空，则将其值增加offset
        if (span$1.to != null) { span$1.to += offset; }
        // 如果span的from属性为空
        if (span$1.from == null) {
          // 在first中查找与span的marker相同的标记，并将其from属性设置为offset
          var found$1 = getMarkedSpanFor(first, span$1.marker);
          if (!found$1) {
            span$1.from = offset;
            // 如果sameLine为true，则将span添加到first中
            if (sameLine) { (first || (first = [])).push(span$1); }
          }
        } else {
          // 否则将span的from属性增加offset
          span$1.from += offset;
          // 如果sameLine为true，则将span添加到first中
          if (sameLine) { (first || (first = [])).push(span$1); }
        }
      }
    }
    // 确保没有创建任何长度为零的span
    if (first) { first = clearEmptySpans(first); }
    // 如果last存在且不等于first，则清除其中的空span
    if (last && last != first) { last = clearEmptySpans(last); }

    // 创建一个包含first的新标记数组
    var newMarkers = [first];
    // 如果sameLine为false
    if (!sameLine) {
      // 使用整行span填充间隙
      var gap = change.text.length - 2, gapMarkers;
      if (gap > 0 && first)
        { for (var i$2 = 0; i$2 < first.length; ++i$2)
          { if (first[i$2].to == null)
            { (gapMarkers || (gapMarkers = [])).push(new MarkedSpan(first[i$2].marker, null, null)); } } }
      for (var i$3 = 0; i$3 < gap; ++i$3)
        { newMarkers.push(gapMarkers); }
      newMarkers.push(last);
    }
    // 返回新的标记数组
    return newMarkers
  }

  // 移除空的span并且没有clearWhenEmpty选项为false的span
  function clearEmptySpans(spans) {
    for (var i = 0; i < spans.length; ++i) {
      var span = spans[i];
      // 如果span的from属性不为空且等于to属性，并且clearWhenEmpty选项不为false，则将其从数组中删除
      if (span.from != null && span.from == span.to && span.marker.clearWhenEmpty !== false)
        { spans.splice(i--, 1); }
    }
    // 如果数组为空，则返回null，否则返回数组
    if (!spans.length) { return null }
    return spans
  }

  // 用于在进行更改时'剪切'掉只读范围
  function removeReadOnlyRanges(doc, from, to) {
    var markers = null;
    // 从指定起始行到结束行遍历文档，对每一行执行指定的函数
    doc.iter(from.line, to.line + 1, function (line) {
      // 如果当前行有标记范围
      if (line.markedSpans) { 
        // 遍历当前行的所有标记范围
        for (var i = 0; i < line.markedSpans.length; ++i) {
          var mark = line.markedSpans[i].marker;
          // 如果标记是只读的并且不在markers数组中
          if (mark.readOnly && (!markers || indexOf(markers, mark) == -1))
            { (markers || (markers = [])).push(mark); }
        } 
      }
    });
    // 如果markers数组为空，则返回null
    if (!markers) { return null }
    // 初始化parts数组，包含起始和结束位置对象
    var parts = [{from: from, to: to}];
    // 遍历markers数组
    for (var i = 0; i < markers.length; ++i) {
      var mk = markers[i], m = mk.find(0);
      // 遍历parts数组
      for (var j = 0; j < parts.length; ++j) {
        var p = parts[j];
        // 如果p的结束位置小于m的起始位置，或者p的起始位置大于m的结束位置，则继续下一次循环
        if (cmp(p.to, m.from) < 0 || cmp(p.from, m.to) > 0) { continue }
        // 初始化newParts数组，包含插入位置和删除数量
        var newParts = [j, 1], dfrom = cmp(p.from, m.from), dto = cmp(p.to, m.to);
        // 如果dfrom小于0或者mk的inclusiveLeft为false并且dfrom为0
        if (dfrom < 0 || !mk.inclusiveLeft && !dfrom)
          { newParts.push({from: p.from, to: m.from}); }
        // 如果dto大于0或者mk的inclusiveRight为false并且dto为0
        if (dto > 0 || !mk.inclusiveRight && !dto)
          { newParts.push({from: m.to, to: p.to}); }
        // 在parts数组中插入newParts数组的内容
        parts.splice.apply(parts, newParts);
        // 更新j的值
        j += newParts.length - 3;
      }
    }
    // 返回parts数组
    return parts
  }

  // 从行中断开标记范围
  function detachMarkedSpans(line) {
    var spans = line.markedSpans;
    // 如果没有标记范围，则返回
    if (!spans) { return }
    // 遍历标记范围数组，断开与行的关联
    for (var i = 0; i < spans.length; ++i)
      { spans[i].marker.detachLine(line); }
    // 将行的标记范围设为null
    line.markedSpans = null;
  }
  // 将标记范围连接到行
  function attachMarkedSpans(line, spans) {
    // 如果没有标记范围，则返回
    if (!spans) { return }
    // 遍历标记范围数组，连接到行
    for (var i = 0; i < spans.length; ++i)
      { spans[i].marker.attachLine(line); }
    // 将行的标记范围设为spans数组
    line.markedSpans = spans;
  }

  // 计算重叠折叠标记范围时使用的辅助函数，返回左侧额外偏移量
  function extraLeft(marker) { return marker.inclusiveLeft ? -1 : 0 }
  // 计算重叠折叠标记范围时使用的辅助函数，返回右侧额外偏移量
  function extraRight(marker) { return marker.inclusiveRight ? 1 : 0 }

  // 比较两个重叠折叠标记范围的大小，返回值表示哪个范围更大（包含另一个范围）
  // 当范围完全相同时，通过比较id来确定大小
  function compareCollapsedMarkers(a, b) {
    # 计算两个文本行的长度差
    var lenDiff = a.lines.length - b.lines.length;
    # 如果长度差不为零，则返回长度差
    if (lenDiff != 0) { return lenDiff }
    # 查找文本行中的位置信息
    var aPos = a.find(), bPos = b.find();
    # 比较起始位置，如果不同则返回负值
    var fromCmp = cmp(aPos.from, bPos.from) || extraLeft(a) - extraLeft(b);
    if (fromCmp) { return -fromCmp }
    # 比较结束位置，如果不同则返回正值
    var toCmp = cmp(aPos.to, bPos.to) || extraRight(a) - extraRight(b);
    if (toCmp) { return toCmp }
    # 返回 b.id - a.id 的结果
    return b.id - a.id
  }

  # 查找文本行中是否存在折叠的区间，如果存在则返回该区间的标记
  function collapsedSpanAtSide(line, start) {
    # 获取文本行中标记的折叠区间
    var sps = sawCollapsedSpans && line.markedSpans, found;
    if (sps) { for (var sp = (void 0), i = 0; i < sps.length; ++i) {
      sp = sps[i];
      # 判断标记是否折叠，并且起始或结束位置为空，如果是则返回该标记
      if (sp.marker.collapsed && (start ? sp.from : sp.to) == null &&
          (!found || compareCollapsedMarkers(found, sp.marker) < 0))
        { found = sp.marker; }
    } }
    return found
  }
  # 查找文本行中是否存在折叠的区间，如果存在则返回该区间的标记
  function collapsedSpanAtStart(line) { return collapsedSpanAtSide(line, true) }
  # 查找文本行中是否存在折叠的区间，如果存在则返回该区间的标记
  function collapsedSpanAtEnd(line) { return collapsedSpanAtSide(line, false) }

  # 查找文本行中是否存在包含指定位置的折叠区间，如果存在则返回该区间的标记
  function collapsedSpanAround(line, ch) {
    # 获取文本行中标记的折叠区间
    var sps = sawCollapsedSpans && line.markedSpans, found;
    if (sps) { for (var i = 0; i < sps.length; ++i) {
      var sp = sps[i];
      # 判断标记是否折叠，并且包含指定位置，如果是则返回该标记
      if (sp.marker.collapsed && (sp.from == null || sp.from < ch) && (sp.to == null || sp.to > ch) &&
          (!found || compareCollapsedMarkers(found, sp.marker) < 0)) { found = sp.marker; }
    } }
    return found
  }

  # 测试是否存在折叠区间与新区间部分重叠（覆盖起始或结束位置，但不是同时），如果是则返回 true
  function conflictingCollapsedRange(doc, lineNo, from, to, marker) {
    # 获取指定行的文本行
    var line = getLine(doc, lineNo);
    # 获取文本行中标记的折叠区间
    var sps = sawCollapsedSpans && line.markedSpans;
    // 如果存在折叠的逻辑行，则遍历处理
    if (sps) { for (var i = 0; i < sps.length; ++i) {
      // 获取当前折叠的逻辑行
      var sp = sps[i];
      // 如果折叠标记未折叠，则继续下一次循环
      if (!sp.marker.collapsed) { continue }
      // 在折叠的逻辑行中查找指定位置的标记
      var found = sp.marker.find(0);
      // 比较起始位置和结束位置的偏移量
      var fromCmp = cmp(found.from, from) || extraLeft(sp.marker) - extraLeft(marker);
      var toCmp = cmp(found.to, to) || extraRight(sp.marker) - extraRight(marker);
      // 判断指定位置是否在折叠的逻辑行内
      if (fromCmp >= 0 && toCmp <= 0 || fromCmp <= 0 && toCmp >= 0) { continue }
      // 判断指定位置是否在折叠的逻辑行的左侧或右侧
      if (fromCmp <= 0 && (sp.marker.inclusiveRight && marker.inclusiveLeft ? cmp(found.to, from) >= 0 : cmp(found.to, from) > 0) ||
          fromCmp >= 0 && (sp.marker.inclusiveRight && marker.inclusiveLeft ? cmp(found.from, to) <= 0 : cmp(found.from, to) < 0))
        { return true }
    } }
  }

  // 获取给定行所在的可视行的起始位置
  function visualLine(line) {
    var merged;
    // 循环处理折叠的逻辑行，直到找到可视行的起始位置
    while (merged = collapsedSpanAtStart(line))
      { line = merged.find(-1, true).line; }
    return line
  }

  // 获取给定行所在的可视行的结束位置
  function visualLineEnd(line) {
    var merged;
    // 循环处理折叠的逻辑行，直到找到可视行的结束位置
    while (merged = collapsedSpanAtEnd(line))
      { line = merged.find(1, true).line; }
    return line
  }

  // 返回延续给定行的可视行的逻辑行数组，如果没有则返回undefined
  function visualLineContinued(line) {
    var merged, lines;
    // 循环处理折叠的逻辑行，直到找到延续的逻辑行
    while (merged = collapsedSpanAtEnd(line)) {
      line = merged.find(1, true).line
      ;(lines || (lines = [])).push(line);
    }
    return lines
  }

  // 获取给定行号所在的可视行的起始行号
  function visualLineNo(doc, lineN) {
    var line = getLine(doc, lineN), vis = visualLine(line);
    // 如果给定行就是可视行的起始行，则返回行号
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

// 计算一行是否被隐藏。当一行是另一行的可视行的一部分，或者完全被折叠的非小部件跨度覆盖时，行被视为隐藏
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
      // 获取当前行
      var line = chunk.lines[i];
      // 如果当前行等于指定的行对象，则跳出循环
      if (line == lineObj) { break }
      // 否则累加行高
      else { h += line.height; }
    }
    // 从当前行向上遍历，计算父级块的高度之和
    for (var p = chunk.parent; p; chunk = p, p = chunk.parent) {
      for (var i$1 = 0; i$1 < p.children.length; ++i$1) {
        var cur = p.children[i$1];
        // 如果当前块等于指定的块，则跳出循环
        if (cur == chunk) { break }
        // 否则累加块的高度
        else { h += cur.height; }
      }
    }
    // 返回计算得到的高度
    return h
  }

  // 计算行的字符长度，考虑到可能隐藏部分内容和连接其他行
  function lineLength(line) {
    // 如果行高为0，则返回0
    if (line.height == 0) { return 0 }
    // 初始化字符长度为行文本的长度
    var len = line.text.length, merged, cur = line;
    // 处理起始处的折叠范围
    while (merged = collapsedSpanAtStart(cur)) {
      var found = merged.find(0, true);
      cur = found.from.line;
      len += found.from.ch - found.to.ch;
    }
    cur = line;
    // 处理结束处的折叠范围
    while (merged = collapsedSpanAtEnd(cur)) {
      var found$1 = merged.find(0, true);
      len -= cur.text.length - found$1.from.ch;
      cur = found$1.to.line;
      len += cur.text.length - found$1.to.ch;
    }
    // 返回计算得到的字符长度
    return len
  }

  // 查找文档中最长的行
  function findMaxLine(cm) {
    var d = cm.display, doc = cm.doc;
    // 初始化最长行为文档的第一行
    d.maxLine = getLine(doc, doc.first);
    d.maxLineLength = lineLength(d.maxLine);
    d.maxLineChanged = true;
    // 遍历文档的每一行，找到最长的行
    doc.iter(function (line) {
      var len = lineLength(line);
      if (len > d.maxLineLength) {
        d.maxLineLength = len;
        d.maxLine = line;
      }
    });
  }

  // 行数据结构

  // 行对象。这些对象保存与行相关的状态，包括高亮信息（styles 数组）。
  var Line = function(text, markedSpans, estimateHeight) {
    this.text = text;
    attachMarkedSpans(this, markedSpans);
  // 如果有估计的高度，则使用估计的高度，否则默认为1
  this.height = estimateHeight ? estimateHeight(this) : 1;
};

// 返回行号
Line.prototype.lineNo = function () { return lineNo(this) };
eventMixin(Line);

// 更改行的内容（文本，标记），自动使缓存信息无效，并尝试重新估计行的高度
function updateLine(line, text, markedSpans, estimateHeight) {
  // 更新行的文本内容
  line.text = text;
  // 如果存在状态信息，则将其置为null
  if (line.stateAfter) { line.stateAfter = null; }
  // 如果存在样式信息，则将其置为null
  if (line.styles) { line.styles = null; }
  // 如果存在顺序信息，则将其置为null
  if (line.order != null) { line.order = null; }
  // 分离标记的范围
  detachMarkedSpans(line);
  // 附加标记的范围
  attachMarkedSpans(line, markedSpans);
  // 如果有估计的高度，则使用估计的高度，否则默认为1
  var estHeight = estimateHeight ? estimateHeight(line) : 1;
  // 如果估计的高度不等于行的高度，则更新行的高度
  if (estHeight != line.height) { updateLineHeight(line, estHeight); }
}

// 从文档树和其标记中分离一行
function cleanUpLine(line) {
  // 将行的父节点置为null
  line.parent = null;
  // 分离标记的范围
  detachMarkedSpans(line);
}

// 将模式返回的样式（可以是null，也可以是包含一个或多个样式的字符串）转换为CSS样式。这是缓存的，并且还查找了行级样式。
var styleToClassCache = {}, styleToClassCacheWithMode = {};
function interpretTokenStyle(style, options) {
  // 如果样式为空或只包含空白字符，则返回null
  if (!style || /^\s*$/.test(style)) { return null }
  // 根据选项决定使用不同的缓存
  var cache = options.addModeClass ? styleToClassCacheWithMode : styleToClassCache;
  return cache[style] ||
    // 将样式替换为CSS类名，并缓存起来
    (cache[style] = style.replace(/\S+/g, "cm-$&"))
}

// 渲染行文本的DOM表示。还构建了一个'行映射'，指向表示特定文本段的DOM节点，并且被测量代码使用。返回的对象包含DOM节点、行映射以及模式设置的行级样式信息。
function buildLineContent(cm, lineView) {
  // padding-right强制元素具有'边框'，这在Webkit中是必需的，以便能够获取其行级边界矩形（在measureChar中使用）。
}
    // 创建一个包含内容的 span 元素，设置样式为 webkit 时的特殊样式
    var content = eltP("span", null, null, webkit ? "padding-right: .1px" : null);
    // 创建一个包含 content 的 pre 元素，设置类名为 "CodeMirror-line"
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
      // 重置 builder.pos 为 0，设置 builder.addToken 为 buildToken 函数
      builder.pos = 0;
      builder.addToken = buildToken;
      // 如果浏览器存在一些问题，将一些修复算法嵌入到标记渲染算法中
      if (hasBadBidiRects(cm.display.measure) && (order = getOrder(line, cm.doc.direction)))
        { builder.addToken = buildTokenBadBidi(builder.addToken, order); }
      // 初始化 builder.map 为空数组
      builder.map = [];
      // 获取逻辑行的样式，插入行内容
      var allowFrontierUpdate = lineView != cm.display.externalMeasured && lineNo(line);
      insertLineContent(line, builder, getLineStyles(cm, line, allowFrontierUpdate));
      // 如果逻辑行有样式类，添加到 builder 的样式类中
      if (line.styleClasses) {
        if (line.styleClasses.bgClass)
          { builder.bgClass = joinClasses(line.styleClasses.bgClass, builder.bgClass || ""); }
        if (line.styleClasses.textClass)
          { builder.textClass = joinClasses(line.styleClasses.textClass, builder.textClass || ""); }
      }

      // 确保至少有一个节点存在，用于测量
      if (builder.map.length == 0)
        { builder.map.push(0, 0, builder.content.appendChild(zeroWidthElement(cm.display.measure))); }

      // 存储当前逻辑行的 map 和缓存对象
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
        # 在文本中查找特殊字符的匹配项
        var m = special.exec(text);
        # 计算跳过的字符数
        var skipped = m ? m.index - pos : text.length - pos;
        # 如果有跳过的字符，则创建文本节点并添加到文档片段中
        if (skipped) {
          var txt = document.createTextNode(displayText.slice(pos, pos + skipped));
          # 根据浏览器类型和版本，添加文本节点到文档片段中
          if (ie && ie_version < 9) { content.appendChild(elt("span", [txt])); }
          else { content.appendChild(txt); }
          # 更新映射关系
          builder.map.push(builder.pos, builder.pos + skipped, txt);
          builder.col += skipped;
          builder.pos += skipped;
        }
        # 如果没有匹配项，则跳出循环
        if (!m) { break }
        # 更新位置变量
        pos += skipped + 1;
        var txt$1 = (void 0);
        # 处理制表符的情况
        if (m[0] == "\t") {
          # 计算制表符的宽度
          var tabSize = builder.cm.options.tabSize, tabWidth = tabSize - builder.col % tabSize;
          # 创建表示制表符的元素并添加到文档片段中
          txt$1 = content.appendChild(elt("span", spaceStr(tabWidth), "cm-tab"));
          txt$1.setAttribute("role", "presentation");
          txt$1.setAttribute("cm-text", "\t");
          builder.col += tabWidth;
        } else if (m[0] == "\r" || m[0] == "\n") {
          # 处理换行符的情况
          txt$1 = content.appendChild(elt("span", m[0] == "\r" ? "\u240d" : "\u2424", "cm-invalidchar"));
          txt$1.setAttribute("cm-text", m[0]);
          builder.col += 1;
        } else {
          # 处理其他特殊字符的情况
          txt$1 = builder.cm.options.specialCharPlaceholder(m[0]);
          txt$1.setAttribute("cm-text", m[0]);
          # 根据浏览器类型和版本，添加特殊字符元素到文档片段中
          if (ie && ie_version < 9) { content.appendChild(elt("span", [txt$1])); }
          else { content.appendChild(txt$1); }
          builder.col += 1;
        }
        # 更新映射关系
        builder.map.push(builder.pos, builder.pos + 1, txt$1);
        builder.pos++;
      }
    }
    # 检查文本末尾是否有空格
    builder.trailingSpace = displayText.charCodeAt(text.length - 1) == 32;
    // 如果存在样式、起始样式、结束样式、必须包装或者 CSS，则执行以下操作
    if (style || startStyle || endStyle || mustWrap || css) {
      // 创建一个完整的样式字符串，如果没有样式则为空字符串
      var fullStyle = style || "";
      // 如果存在起始样式，则添加到完整样式字符串中
      if (startStyle) { fullStyle += startStyle; }
      // 如果存在结束样式，则添加到完整样式字符串中
      if (endStyle) { fullStyle += endStyle; }
      // 使用完整样式字符串和 CSS 创建一个 span 元素
      var token = elt("span", [content], fullStyle, css);
      // 如果存在属性，则遍历属性并设置到 token 元素中
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
    // 如果不存在样式、起始样式、结束样式、必须包装或者 CSS，则将 content 直接添加到 builder.content 中
    builder.content.appendChild(content);
  }

  // 将一些空格转换为 NBSP，以防止浏览器在渲染文本时将行尾的空格合并在一起（问题＃1362）
  function splitSpaces(text, trailingBefore) {
    // 如果文本长度大于1且不包含连续两个空格，则直接返回文本
    if (text.length > 1 && !/  /.test(text)) { return text }
    var spaceBefore = trailingBefore, result = "";
    for (var i = 0; i < text.length; i++) {
      var ch = text.charAt(i);
      // 如果当前字符是空格且前一个字符也是空格，并且当前字符是文本的最后一个字符或者下一个字符也是空格，则将当前空格替换为 NBSP
      if (ch == " " && spaceBefore && (i == text.length - 1 || text.charCodeAt(i + 1) == 32)) {
        ch = "\u00a0";
      }
      // 将处理后的字符添加到结果字符串中
      result += ch;
      // 更新 spaceBefore 变量，记录当前字符是否是空格
      spaceBefore = ch == " ";
    }
    return result
  }

  // 解决浏览器对右到左文本报告的无意义尺寸问题
  function buildTokenBadBidi(inner, order) {
    // 定义一个函数，接受多个参数，包括构建器、文本、样式、起始样式、结束样式、CSS 和属性
    return function (builder, text, style, startStyle, endStyle, css, attributes) {
      // 如果样式存在，则将其与 "cm-force-border" 拼接，否则直接使用 "cm-force-border"
      style = style ? style + " cm-force-border" : "cm-force-border";
      // 定义起始位置为构建器的位置，结束位置为起始位置加上文本长度
      var start = builder.pos, end = start + text.length;
      // 无限循环
      for (;;) {
        // 定义变量 part，暂时赋值为 undefined
        var part = (void 0);
        // 遍历 order 数组
        for (var i = 0; i < order.length; i++) {
          // 将 part 赋值为 order[i]
          part = order[i];
          // 如果 part 的结束位置大于等于起始位置并且起始位置大于 part 的起始位置，则跳出循环
          if (part.to > start && part.from <= start) { break }
        }
        // 如果 part 的结束位置大于等于结束位置，则调用 inner 函数，传入参数并返回结果
        if (part.to >= end) { return inner(builder, text, style, startStyle, endStyle, css, attributes) }
        // 调用 inner 函数，传入参数并返回结果
        inner(builder, text.slice(0, part.to - start), style, startStyle, null, css, attributes);
        // 将起始样式置为 null
        startStyle = null;
        // 将文本截取为 part.to - start 的部分
        text = text.slice(part.to - start);
        // 更新起始位置
        start = part.to;
      }
    }
  }

  // 定义一个函数，接受构建器、大小、标记和是否忽略小部件作为参数
  function buildCollapsedSpan(builder, size, marker, ignoreWidget) {
    // 定义变量 widget，如果 ignoreWidget 为 false 并且 marker.widgetNode 存在，则赋值为 marker.widgetNode
    var widget = !ignoreWidget && marker.widgetNode;
    // 如果 widget 存在，则将构建器的 map 数组推入 builder.pos、builder.pos + size 和 widget
    if (widget) { builder.map.push(builder.pos, builder.pos + size, widget); }
    // 如果不忽略小部件并且构建器的 cm.display.input.needsContentAttribute 为真
    if (!ignoreWidget && builder.cm.display.input.needsContentAttribute) {
      // 如果 widget 不存在，则创建一个 span 元素并添加到构建器的 content 中
      if (!widget)
        { widget = builder.content.appendChild(document.createElement("span")); }
      // 设置 widget 的属性 "cm-marker" 为 marker.id
      widget.setAttribute("cm-marker", marker.id);
    }
    // 如果 widget 存在，则将构建器的 cm.display.input.setUneditable 方法应用于 widget
    if (widget) {
      builder.cm.display.input.setUneditable(widget);
      // 将 widget 添加到构建器的 content 中
      builder.content.appendChild(widget);
    }
    // 更新构建器的位置
    builder.pos += size;
    // 将构建器的 trailingSpace 置为 false
    builder.trailingSpace = false;
  }

  // 输出多个 span 元素以构成一行，考虑到高亮和标记文本
  function insertLineContent(line, builder, styles) {
    // 定义变量 spans 为 line 的 markedSpans，allText 为 line 的文本，at 为 0
    var spans = line.markedSpans, allText = line.text, at = 0;
    // 如果 spans 不存在
    if (!spans) {
      // 遍历 styles 数组
      for (var i$1 = 1; i$1 < styles.length; i$1+=2)
        // 调用 builder 的 addToken 方法，传入参数并返回结果
        { builder.addToken(builder, allText.slice(at, at = styles[i$1]), interpretTokenStyle(styles[i$1+1], builder.cm.options)); }
      return
    }

    // 定义变量 len 为 allText 的长度，pos 为 0，i 为 1，text 为 ""，style 和 css
    var len = allText.length, pos = 0, i = 1, text = "", style, css;
    // 定义变量 nextChange，用于存储下一个改变的位置
    var nextChange = 0, spanStyle, spanEndStyle, spanStartStyle, collapsed, attributes;
    // 结束函数定义

  // 这些对象用于表示文档的可见（当前绘制的）部分。如果这些部分由折叠范围连接，则 LineView 可能对应于多个逻辑行。
  function LineView(doc, line, lineN) {
    // 起始行
    this.line = line;
    // 继续的行，如果有的话
    this.rest = visualLineContinued(line);
    // 这个可视行中的逻辑行数
    this.size = this.rest ? lineNo(lst(this.rest)) - lineN + 1 : 1;
    this.node = this.text = null;
    this.hidden = lineIsHidden(doc, line);
  }

  // 为给定的行创建 LineView 对象的范围
  function buildViewArray(cm, from, to) {
    var array = [], nextPos;
    for (var pos = from; pos < to; pos = nextPos) {
      var view = new LineView(cm.doc, getLine(cm.doc, pos), pos);
      nextPos = pos + view.size;
      array.push(view);
    }
    return array
  }

  // 初始化操作组为 null
  var operationGroup = null;

  // 将操作推入操作组
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

  // 为操作组中的操作调用延迟回调函数
  function fireCallbacksForOps(group) {
    // 调用延迟回调函数和光标活动处理程序，直到没有新的出现
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

  // 完成操作
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
  // signalLater 查看是否有任何处理程序，并安排它们在最后一个操作结束时执行，或者如果没有活动操作，则在超时触发时执行。
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

  // 确保具有 gutter 元素、小部件或背景类的行被包装，并将额外的元素添加到包装的 div 中
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
    // 根据 lineView 的 bgClass 和 line 的 bgClass 设置 cls
    var cls = lineView.bgClass ? lineView.bgClass + " " + (lineView.line.bgClass || "") : lineView.line.bgClass;
    if (cls) { cls += " CodeMirror-linebackground"; }
    if (lineView.background) {
      if (cls) { lineView.background.className = cls; }
      else { lineView.background.parentNode.removeChild(lineView.background); lineView.background = null; }
    } else if (cls) {
      // 如果有 cls，则在 ensureLineWrapped 返回的包装中插入一个 div 元素作为背景
      var wrap = ensureLineWrapped(lineView);
      lineView.background = wrap.insertBefore(elt("div", null, cls), wrap.firstChild);
      cm.display.input.setUneditable(lineView.background);
    }
  }

  // 包装 buildLineContent，如果可能的话，将重用 display.externalMeasured 中的结构
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
    // 获取 lineView 的 text 的 className
    var cls = lineView.text.className;
    // 获取行内容
    var built = getLineContent(cm, lineView);
    // 如果 lineView 的 text 等于 node，则将 node 设置为 build 的 pre 元素
    if (lineView.text == lineView.node) { lineView.node = built.pre; }
    # 用构建好的 <pre> 元素替换原来的文本节点
    lineView.text.parentNode.replaceChild(built.pre, lineView.text);
    # 更新 lineView.text 为构建好的 <pre> 元素
    lineView.text = built.pre;
    # 如果构建好的背景类或文本类与 lineView 的不同，更新 lineView 的背景类和文本类，并更新行的类
    if (built.bgClass != lineView.bgClass || built.textClass != lineView.textClass) {
      lineView.bgClass = built.bgClass;
      lineView.textClass = built.textClass;
      updateLineClasses(cm, lineView);
    } else if (cls) {
      # 否则，如果有类名，更新 lineView.text 的类名
      lineView.text.className = cls;
    }
  }

  # 更新行的类
  function updateLineClasses(cm, lineView) {
    # 更新行的背景
    updateLineBackground(cm, lineView);
    # 如果行有 wrapClass，确保行被包裹，并更新类名
    if (lineView.line.wrapClass)
      { ensureLineWrapped(lineView).className = lineView.line.wrapClass; }
    # 否则，如果行的节点不等于文本节点，清空类名
    else if (lineView.node != lineView.text)
      { lineView.node.className = ""; }
    # 构建文本类，更新 lineView.text 的类名
    var textClass = lineView.textClass ? lineView.textClass + " " + (lineView.line.textClass || "") : lineView.line.textClass;
    lineView.text.className = textClass || "";
  }

  # 更新行的行号区域
  function updateLineGutter(cm, lineView, lineN, dims) {
    # 如果行有 gutter，移除它
    if (lineView.gutter) {
      lineView.node.removeChild(lineView.gutter);
      lineView.gutter = null;
    }
    # 如果行有 gutterBackground，移除它
    if (lineView.gutterBackground) {
      lineView.node.removeChild(lineView.gutterBackground);
      lineView.gutterBackground = null;
    }
    # 如果行有 gutterClass，创建并插入 gutterBackground 元素
    if (lineView.line.gutterClass) {
      var wrap = ensureLineWrapped(lineView);
      lineView.gutterBackground = elt("div", null, "CodeMirror-gutter-background " + lineView.line.gutterClass,
                                      ("left: " + (cm.options.fixedGutter ? dims.fixedPos : -dims.gutterTotalWidth) + "px; width: " + (dims.gutterTotalWidth) + "px"));
      cm.display.input.setUneditable(lineView.gutterBackground);
      wrap.insertBefore(lineView.gutterBackground, lineView.text);
    }
    # 获取行的 gutterMarkers
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
        // 如果找到标记，则将其添加到装载行号的容器中
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
    // 创建一个测试函数，用于检查是否为行部件
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
    // 获取行内容的构建结果
    var built = getLineContent(cm, lineView);
    // 将行内容的预格式化文本赋给行视图的文本属性
    lineView.text = lineView.node = built.pre;
    // 如果有背景样式类，则赋给行视图的背景样式类属性
    if (built.bgClass) { lineView.bgClass = built.bgClass; }
    # 如果存在文本类别，则将行视图的文本类别设置为已构建的文本类别
    if (built.textClass) { lineView.textClass = built.textClass; }

    # 更新行的类别
    updateLineClasses(cm, lineView);
    # 更新行的行号
    updateLineGutter(cm, lineView, lineN, dims);
    # 插入行部件
    insertLineWidgets(cm, lineView, dims);
    # 返回行视图的节点
    return lineView.node
  }

  # 一个行视图可能包含多个逻辑行（通过合并的跨度）。所有这些行的部件都需要被绘制。
  function insertLineWidgets(cm, lineView, dims) {
    insertLineWidgetsFor(cm, lineView.line, lineView, dims, true);
    # 如果存在额外的行，则为每个额外的行插入行部件
    if (lineView.rest) { for (var i = 0; i < lineView.rest.length; i++)
      { insertLineWidgetsFor(cm, lineView.rest[i], lineView, dims, false); } }
  }

  # 为每个行插入行部件
  function insertLineWidgetsFor(cm, line, lineView, dims, allowAbove) {
    # 如果行没有部件，则返回
    if (!line.widgets) { return }
    # 确保行被包裹
    var wrap = ensureLineWrapped(lineView);
    # 遍历行的部件
    for (var i = 0, ws = line.widgets; i < ws.length; ++i) {
      var widget = ws[i], node = elt("div", [widget.node], "CodeMirror-linewidget" + (widget.className ? " " + widget.className : ""));
      # 如果部件不处理鼠标事件，则设置属性
      if (!widget.handleMouseEvents) { node.setAttribute("cm-ignore-events", "true"); }
      # 定位行部件
      positionLineWidget(widget, node, lineView, dims);
      cm.display.input.setUneditable(node);
      # 如果允许在上方插入部件，并且部件在上方，则在行的开头插入部件
      if (allowAbove && widget.above)
        { wrap.insertBefore(node, lineView.gutter || lineView.text); }
      else
        { wrap.appendChild(node); }
      signalLater(widget, "redraw");
    }
  }

  # 定位行部件
  function positionLineWidget(widget, node, lineView, dims) {
    # 如果部件不需要水平滚动，则设置样式
    if (widget.noHScroll) {
  (lineView.alignable || (lineView.alignable = [])).push(node);
      var width = dims.wrapperWidth;
      node.style.left = dims.fixedPos + "px";
      if (!widget.coverGutter) {
        width -= dims.gutterTotalWidth;
        node.style.paddingLeft = dims.gutterTotalWidth + "px";
      }
      node.style.width = width + "px";
    }
    # 如果部件需要覆盖行号，则设置样式
    if (widget.coverGutter) {
      node.style.zIndex = 5;
      node.style.position = "relative";
      if (!widget.noHScroll) { node.style.marginLeft = -dims.gutterTotalWidth + "px"; }
  // 返回小部件的高度
  function widgetHeight(widget) {
    // 如果小部件的高度不为空，则返回该高度
    if (widget.height != null) { return widget.height }
    // 获取小部件所在的 CodeMirror 对象
    var cm = widget.doc.cm;
    // 如果小部件没有所在的 CodeMirror 对象，则返回 0
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
      // 移除原有的子元素，并添加小部件的节点作为子元素
      removeChildrenAndAdd(cm.display.measure, elt("div", [widget.node], null, parentStyle));
    }
    // 返回小部件的高度，并将其赋值给 widget.height
    return widget.height = widget.node.parentNode.offsetHeight
  }

  // 当给定的鼠标事件发生在小部件中时返回 true
  function eventInWidget(display, e) {
    // 遍历鼠标事件的目标元素及其父元素
    for (var n = e_target(e); n != display.wrapper; n = n.parentNode) {
      // 如果目标元素为空，或者是被忽略的事件元素，或者是 display.sizer 的子元素但不是 display.mover，则返回 true
      if (!n || (n.nodeType == 1 && n.getAttribute("cm-ignore-events") == "true") ||
          (n.parentNode == display.sizer && n != display.mover))
        { return true }
    }
  }

  // 位置测量

  // 返回行间距的上偏移量
  function paddingTop(display) {return display.lineSpace.offsetTop}
  // 返回行间距的垂直内边距
  function paddingVert(display) {return display.mover.offsetHeight - display.lineSpace.offsetHeight}
  // 返回行间距的水平内边距
  function paddingH(display) {
    // 如果已经缓存了水平内边距，则直接返回缓存的值
    if (display.cachedPaddingH) { return display.cachedPaddingH }
    // 创建一个包含单个字符的 pre 元素，并获取其样式
    var e = removeChildrenAndAdd(display.measure, elt("pre", "x", "CodeMirror-line-like"));
    var style = window.getComputedStyle ? window.getComputedStyle(e) : e.currentStyle;
    // 解析样式中的左右内边距值
    var data = {left: parseInt(style.paddingLeft), right: parseInt(style.paddingRight)};
    // 如果左右内边距值都是数字，则缓存这些值并返回
    if (!isNaN(data.left) && !isNaN(data.right)) { display.cachedPaddingH = data; }
    return data
  }

  // 返回滚动条的间隙
  function scrollGap(cm) { return scrollerGap - cm.display.nativeBarWidth }
  // 返回显示区域的宽度
  function displayWidth(cm) {
    return cm.display.scroller.clientWidth - scrollGap(cm) - cm.display.barWidth
  }
  // 返回显示区域的高度
  function displayHeight(cm) {
  // 计算并返回滚动条可视区域的高度减去滚动条的高度和滚动条与编辑器内容之间的间隙
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

// 找到给定行号的行映射（将字符偏移映射到文本节点）和测量缓存。（当折叠范围存在时，行视图可能包含多行。）
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
  // 获取给定行的行号
  var lineN = lineNo(line);
  // 创建一个新的 LineView 对象，用于显示给定行的内容
  var view = cm.display.externalMeasured = new LineView(cm.doc, line, lineN);
  // 将行号保存到 LineView 对象中
  view.lineN = lineN;
  // 构建 LineView 对象的内容
  var built = view.built = buildLineContent(cm, view);
  // 将构建好的内容保存到 LineView 对象中
  view.text = built.pre;
  // 移除之前的子元素，并添加新的内容到 lineMeasure 元素中
  removeChildrenAndAdd(cm.display.lineMeasure, built.pre);
  // 返回 LineView 对象
  return view
}

// 获取给定字符的位置信息（在行内坐标系中的 {top, bottom, left, right} 盒子）
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

// 准备测量工作，分为两步：整行的设置工作和实际字符的测量
function prepareMeasureForLine(cm, line) {
  // 获取行号
  var lineN = lineNo(line);
  // 查找与给定行号对应的 LineView 对象
  var view = findViewForLine(cm, lineN);
  // 如果存在对应的 LineView 对象且没有内容，则置空
  if (view && !view.text) {
    view = null;
  } else if (view && view.changes) {
    // 更新 LineView 对象的内容
    updateLineForChanges(cm, view, lineN, getDimensions(cm));
    cm.curOp.forceUpdate = true;
  }
  // 如果不存在对应的 LineView 对象，则更新外部测量
  if (!view)
    { view = updateExternalMeasurement(cm, line); }

  // 根据 LineView 对象和行信息，返回准备好的测量对象
  var info = mapFromLineView(view, line, lineN);
  return {
    line: line, view: view, rect: null,
    map: info.map, cache: info.cache, before: info.before,
    hasHeights: false
  }
}

// 根据准备好的测量对象，测量实际字符的位置
function measureCharPrepared(cm, prepared, ch, bias, varHeight) {
  // 如果存在 before 属性，则将字符位置设为 -1
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
      # 如果没有高度信息，则确保行高信息
      if (!prepared.hasHeights) {
        ensureLineHeights(cm, prepared.view, prepared.rect);
        prepared.hasHeights = true;
      }
      # 测量字符的宽度和位置
      found = measureCharInner(cm, prepared, ch, bias);
      # 如果测量结果有效，则将其存入缓存
      if (!found.bogus) { prepared.cache[key] = found; }
    }
    # 返回测量结果
    return {left: found.left, right: found.right,
            top: varHeight ? found.rtop : found.top,
            bottom: varHeight ? found.rbottom : found.bottom}
  }

  # 定义一个空矩形对象
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
  function measureCharInner(cm, prepared, ch, bias) {
    // 在行映射中获取字符的节点和偏移量
    var place = nodeAndOffsetInLineMap(prepared.map, ch, bias);
    // 获取节点、起始位置、结束位置、折叠状态
    var node = place.node, start = place.start, end = place.end, collapse = place.collapse;

    var rect;
    // 如果节点类型为文本节点，则使用范围来检索坐标
    if (node.nodeType == 3) {
      // 最多重试4次，当返回无意义的矩形时
      for (var i$1 = 0; i$1 < 4; i$1++) {
        // 当起始位置大于0且当前字符为扩展字符时，向前移动起始位置
        while (start && isExtendingChar(prepared.line.text.charAt(place.coverStart + start))) { --start; }
        // 当覆盖起始位置加上结束位置小于覆盖结束位置且当前字符为扩展字符时，向后移动结束位置
        while (place.coverStart + end < place.coverEnd && isExtendingChar(prepared.line.text.charAt(place.coverStart + end))) { ++end; }
        // 如果是IE并且版本小于9且起始位置为0且结束位置为覆盖结束位置减去覆盖起始位置，则获取父节点的矩形
        if (ie && ie_version < 9 && start == 0 && end == place.coverEnd - place.coverStart) { rect = node.parentNode.getBoundingClientRect(); }
        else { rect = getUsefulRect(range(node, start, end).getClientRects(), bias); }
        // 如果矩形左边、右边存在，或者起始位置为0，则跳出循环
        if (rect.left || rect.right || start == 0) { break }
        // 更新起始位置和结束位置，折叠状态为右
        end = start;
        start = start - 1;
        collapse = "right";
      }
      // 如果是IE并且版本小于11，则可能更新缩放后的矩形
      if (ie && ie_version < 11) { rect = maybeUpdateRectForZooming(cm.display.measure, rect); }
    } else {
      // 如果是小部件，则简单地获取整个小部件的框
      if (start > 0) { collapse = bias = "right"; }
      var rects;
      // 如果代码编辑器选项中启用了行包装并且矩形数量大于1，则获取最后一个或第一个矩形
      if (cm.options.lineWrapping && (rects = node.getClientRects()).length > 1) { rect = rects[bias == "right" ? rects.length - 1 : 0]; }
      else { rect = node.getBoundingClientRect(); }
    }
  }
    // 如果满足条件：ie存在、ie版本小于9、start不存在、rect不存在或者left和right都不存在
    if (ie && ie_version < 9 && !start && (!rect || !rect.left && !rect.right)) {
      // 获取父节点的第一个客户端矩形
      var rSpan = node.parentNode.getClientRects()[0];
      // 如果rSpan存在，则使用其left、right、top和bottom创建rect对象
      if (rSpan)
        { rect = {left: rSpan.left, right: rSpan.left + charWidth(cm.display), top: rSpan.top, bottom: rSpan.bottom}; }
      // 否则将rect设置为nullRect
      else
        { rect = nullRect; }
    }

    // 计算rtop和rbot
    var rtop = rect.top - prepared.rect.top, rbot = rect.bottom - prepared.rect.top;
    // 计算mid
    var mid = (rtop + rbot) / 2;
    // 获取prepared.view.measure.heights
    var heights = prepared.view.measure.heights;
    // 初始化i为0，遍历heights数组
    var i = 0;
    for (; i < heights.length - 1; i++)
      { if (mid < heights[i]) { break } }
    // 计算top和bot
    var top = i ? heights[i - 1] : 0, bot = heights[i];
    // 创建result对象，根据collapse的值设置left和right，设置top和bottom
    var result = {left: (collapse == "right" ? rect.right : rect.left) - prepared.rect.left,
                  right: (collapse == "left" ? rect.left : rect.right) - prepared.rect.left,
                  top: top, bottom: bot};
    // 如果rect的left和right都不存在，则设置result的bogus属性为true
    if (!rect.left && !rect.right) { result.bogus = true; }
    // 如果cm.options.singleCursorHeightPerLine为false，则设置result的rtop和rbottom属性
    if (!cm.options.singleCursorHeightPerLine) { result.rtop = rtop; result.rbottom = rbot; }

    // 返回result对象
    return result
  }

  // 解决IE10及以下缩放时边界客户端矩形返回不正确的问题
  function maybeUpdateRectForZooming(measure, rect) {
    // 如果不满足条件：window.screen不存在、screen.logicalXDPI为null、logicalXDPI等于deviceXDPI、或者没有错误的缩放矩形
    if (!window.screen || screen.logicalXDPI == null ||
        screen.logicalXDPI == screen.deviceXDPI || !hasBadZoomedRects(measure))
      { return rect }
    // 计算缩放比例
    var scaleX = screen.logicalXDPI / screen.deviceXDPI;
    var scaleY = screen.logicalYDPI / screen.deviceYDPI;
    // 返回根据缩放比例计算后的rect对象
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
        # 如果小部件在行上方，则累加小部件高度
        height += widgetHeight(lineObj.widgets[i]); 
      } } }
    return height
  }

  # 将行本地坐标系的{top, bottom, left, right}框转换为另一个坐标系
  # 上下文可能是"line"、"div"（display.lineDiv）、"local"/null（编辑器）、"window"或"page"
  function intoCoordSystem(cm, lineObj, rect, context, includeWidgets) {
    # 如果不包括小部件
    if (!includeWidgets) {
      # 获取小部件顶部高度
      var height = widgetTopHeight(lineObj);
      rect.top += height; rect.bottom += height;
    }
    # 如果上下文是"line"，则直接返回rect
    if (context == "line") { return rect }
    # 如果上下文为空，则设置为"local"
    if (!context) { context = "local"; }
    # 获取行的高度
    var yOff = heightAtLine(lineObj);
    // 如果上下文是“local”，则增加垂直偏移量
    if (context == "local") { yOff += paddingTop(cm.display); }
    // 如果上下文不是“local”，则减去视图偏移量
    else { yOff -= cm.display.viewOffset; }
    // 如果上下文是“page”或“window”，则进行坐标转换
    if (context == "page" || context == "window") {
      // 获取行间距的位置信息
      var lOff = cm.display.lineSpace.getBoundingClientRect();
      yOff += lOff.top + (context == "window" ? 0 : pageScrollY());
      var xOff = lOff.left + (context == "window" ? 0 : pageScrollX());
      rect.left += xOff; rect.right += xOff;
    }
    // 增加垂直偏移量到矩形的上下边界
    rect.top += yOff; rect.bottom += yOff;
    // 返回转换后的矩形坐标
    return rect
  }

  // 将一个框从“div”坐标系转换到另一个坐标系
  // 上下文可以是“window”、“page”、“div”或“local”/null
  function fromCoordSystem(cm, coords, context) {
    // 如果上下文是“div”，则直接返回坐标
    if (context == "div") { return coords }
    var left = coords.left, top = coords.top;
    // 首先转换到“page”坐标系
    if (context == "page") {
      left -= pageScrollX();
      top -= pageScrollY();
    } else if (context == "local" || !context) {
      var localBox = cm.display.sizer.getBoundingClientRect();
      left += localBox.left;
      top += localBox.top;
    }
    // 获取行间距的位置信息
    var lineSpaceBox = cm.display.lineSpace.getBoundingClientRect();
    return {left: left - lineSpaceBox.left, top: top - lineSpaceBox.top}
  }

  // 获取字符的坐标信息
  function charCoords(cm, pos, context, lineObj, bias) {
    // 如果没有给定行对象，则获取指定行的行对象
    if (!lineObj) { lineObj = getLine(cm.doc, pos.line); }
  # 返回给定坐标系中的坐标
  return intoCoordSystem(cm, lineObj, measureChar(cm, lineObj, pos.ch, bias), context)
}

// 返回给定光标位置的框，可能包含一个'other'属性，其中包含次要光标在双向文本边界上的位置。
// 光标 Pos(line, char, "before") 与 `char - 1` 的写入顺序相同，并且在 `char - 1` 之后的可视行上
// 光标 Pos(line, char, "after") 与 `char` 的写入顺序相同，并且在 `char` 之前的可视行上
// 例子（大写字母是 RTL，小写字母是 LTR）：
//     Pos(0, 1, ...)
//     before   after
// ab     a|b     a|b
// aB     a|B     aB|
// Ab     |Ab     A|b
// AB     B|A     B|A
// 每个位置在行上最后一个字符之后被认为是粘在行上的最后一个字符上。
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
    # 如果传入的 other 不为空，则根据条件获取双向绑定的值
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
    # 如果 outside 为 true，则将其添加到位置对象中
    if (outside) { pos.outside = outside; }
    # 返回位置对象
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
    # 如果 x 坐标小于 0，则将其设置为 0
    if (x < 0) { x = 0; }
    # 获取行对象
    var lineObj = getLine(doc, lineN);
    # 无限循环，直到条件不满足
    for (;;) {
      # 在给定坐标处找到对应的字符位置
      var found = coordsCharInner(cm, lineObj, lineN, x, y);
      # 查找给定位置是否处于折叠的范围内
      var collapsed = collapsedSpanAround(lineObj, found.ch + (found.xRel > 0 || found.outside > 0 ? 1 : 0));
      # 如果没有折叠，则返回找到的字符位置
      if (!collapsed) { return found }
      # 查找折叠范围的结束位置
      var rangeEnd = collapsed.find(1);
      # 如果结束位置在同一行，则返回结束位置
      if (rangeEnd.line == lineN) { return rangeEnd }
      # 获取结束位置所在的行对象
      lineObj = getLine(doc, lineN = rangeEnd.line);
    }
  }

  # 获取折叠行的范围
  function wrappedLineExtent(cm, lineObj, preparedMeasure, y) {
    # 减去折叠行顶部的高度
    y -= widgetTopHeight(lineObj);
    # 获取行文本的结束位置
    var end = lineObj.text.length;
    # 查找第一个满足条件的字符位置
    var begin = findFirst(function (ch) { return measureCharPrepared(cm, preparedMeasure, ch - 1).bottom <= y; }, end, 0);
    # 查找最后一个满足条件的字符位置
    end = findFirst(function (ch) { return measureCharPrepared(cm, preparedMeasure, ch).top > y; }, begin, end);
    # 返回折叠行的范围
    return {begin: begin, end: end}
  }

  # 获取折叠行的字符范围
  function wrappedLineExtentChar(cm, lineObj, preparedMeasure, target) {
    # 如果没有准备好的测量数据，则进行准备
    if (!preparedMeasure) { preparedMeasure = prepareMeasureForLine(cm, lineObj); }
    # 将目标字符位置转换为行内坐标系
    var targetTop = intoCoordSystem(cm, lineObj, measureCharPrepared(cm, preparedMeasure, target), "line").top;
    # 返回折叠行的字符范围
    return wrappedLineExtent(cm, lineObj, preparedMeasure, targetTop)
  }

  # 判断给定的盒子边界是否在指定坐标之后
  function boxIsAfter(box, x, y, left) {
    # 如果盒子底部在指定坐标之下，则返回false
    return box.bottom <= y ? false : box.top > y ? true : (left ? box.left : box.right) > x
  }

  # 获取给定坐标处的字符位置
  function coordsCharInner(cm, lineObj, lineNo, x, y) {
    # 将y坐标转换为行内局部坐标系
    y -= heightAtLine(lineObj);
    # 准备行内测量数据
    var preparedMeasure = prepareMeasureForLine(cm, lineObj);
    # 获取行内部件的高度
    var widgetHeight = widgetTopHeight(lineObj);
    # 初始化起始位置、结束位置和文本方向
    var begin = 0, end = lineObj.text.length, ltr = true;

    # 获取文本的排列顺序
    var order = getOrder(lineObj, cm.doc.direction);
    # 如果行不是纯左到右文本，则首先确定坐标落入哪个双向文本段落中
    # 如果存在文字方向，则计算文字方向
    if (order) {
      # 根据文字方向计算部分坐标
      var part = (cm.options.lineWrapping ? coordsBidiPartWrapped : coordsBidiPart)
                   (cm, lineObj, lineNo, preparedMeasure, order, x, y);
      # 判断文字方向是否为从左到右
      ltr = part.level != 1;
      # 根据文字方向确定起始和结束位置
      # 注意：-1 的偏移是因为 findFirst 方法的边界处理方式
      begin = ltr ? part.from : part.to - 1;
      end = ltr ? part.to : part.from - 1;
    }

    # 二分查找，找到第一个边界框起始位置在坐标之后的字符
    # 如果遇到边界框包裹坐标的情况，则存储下来
    var chAround = null, boxAround = null;
    var ch = findFirst(function (ch) {
      # 计算字符的边界框
      var box = measureCharPrepared(cm, preparedMeasure, ch);
      box.top += widgetHeight; box.bottom += widgetHeight;
      # 判断坐标是否在边界框之后
      if (!boxIsAfter(box, x, y, false)) { return false }
      # 如果坐标在边界框内，则存储字符和边界框
      if (box.top <= y && box.left <= x) {
        chAround = ch;
        boxAround = box;
      }
      return true
    }, begin, end);

    var baseX, sticky, outside = false;
    # 如果存在包裹坐标的边界框，则使用该边界框
    if (boxAround) {
      # 区分坐标靠近边界框左侧还是右侧
      var atLeft = x - boxAround.left < boxAround.right - x, atStart = atLeft == ltr;
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
    # 如果索引大于0
    if (index > 0) {
      # 判断当前部分是否为一级部分
      var ltr = part.level != 1;
      # 获取当前部分的起始坐标
      var start = cursorCoords(cm, Pos(lineNo, ltr ? part.from : part.to, ltr ? "after" : "before"),
                               "line", lineObj, preparedMeasure);
      # 如果起始坐标在指定坐标之后且在指定坐标的上方
      if (boxIsAfter(start, x, y, true) && start.top > y)
        { part = order[index - 1]; }
    }
    # 返回当前部分
    return part
  }

  # 在包裹的行中查找双向文本部分
  function coordsBidiPartWrapped(cm, lineObj, _lineNo, preparedMeasure, order, x, y) {
    # 在包裹的行中，rtl文本在包裹边界上可能会做一些与我们的`order`数组中的顺序不符的事情，因此二分搜索不起作用，我们希望返回一个只跨越一行的部分，以便在coordsCharInner中的二分搜索是安全的。因此，我们首先找到包裹行的范围，然后进行一个扁平搜索，在这个搜索中，我们丢弃任何不在该行上的跨度。
    var ref = wrappedLineExtent(cm, lineObj, preparedMeasure, y);
    var begin = ref.begin;
    var end = ref.end;
    # 如果行文本末尾是空白字符，则结束位置减一
    if (/\s/.test(lineObj.text.charAt(end - 1))) { end--; }
    var part = null, closestDist = null;
    # 遍历部分顺序数组
    for (var i = 0; i < order.length; i++) {
      var p = order[i];
      # 如果部分的起始位置大于等于结束位置，或者部分的结束位置小于等于开始位置，则继续下一次循环
      if (p.from >= end || p.to <= begin) { continue }
      # 判断当前部分是否为一级部分
      var ltr = p.level != 1;
      # 获取部分结束位置的x坐标
      var endX = measureCharPrepared(cm, preparedMeasure, ltr ? Math.min(end, p.to) - 1 : Math.max(begin, p.from)).right;
      # 如果结束位置的x坐标小于指定的x坐标
      # 则计算距离，并与最近距离比较
      var dist = endX < x ? x - endX + 1e9 : endX - x;
      if (!part || closestDist > dist) {
        part = p;
        closestDist = dist;
      }
    }
    # 如果没有找到部分，则选择顺序数组中的最后一个部分
    if (!part) { part = order[order.length - 1]; }
    # 将部分裁剪到包裹的行
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
  // 如果 measureText 为空，则创建一个 pre 元素，并添加一些文本用于测量文本高度
  if (measureText == null) {
    measureText = elt("pre", null, "CodeMirror-line-like");
    // 测量一系列行的高度，用于一些计算高度的浏览器
    for (var i = 0; i < 49; ++i) {
      measureText.appendChild(document.createTextNode("x"));
      measureText.appendChild(elt("br"));
    }
    measureText.appendChild(document.createTextNode("x"));
  }
  // 移除 display.measure 下的所有子元素，并添加 measureText
  removeChildrenAndAdd(display.measure, measureText);
  // 计算平均行高
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
  // 创建一个 span 元素，并添加到 pre 元素中，用于测量字符宽度
  var anchor = elt("span", "xxxxxxxxxx");
  var pre = elt("pre", [anchor], "CodeMirror-line-like");
  removeChildrenAndAdd(display.measure, pre);
  // 获取字符宽度
  var rect = anchor.getBoundingClientRect(), width = (rect.right - rect.left) / 10;
  // 如果宽度大于 2，则缓存字符宽度
  if (width > 2) { display.cachedCharWidth = width; }
  return width || 10
}

// 执行一次性读取所需的 DOM 位置和大小，以便在绘制视图时不会交错读写 DOM
function getDimensions(cm) {
  var d = cm.display, left = {}, width = {};
  var gutterLeft = d.gutters.clientLeft;
  for (var n = d.gutters.firstChild, i = 0; n; n = n.nextSibling, ++i) {
    var id = cm.display.gutterSpecs[i].className;
    left[id] = n.offsetLeft + n.clientLeft + gutterLeft;
    width[id] = n.clientWidth;
  }
}
  # 返回一个对象，包含固定位置、垂直滚动条宽度、左侧位置、宽度和包裹宽度
  return {fixedPos: compensateForHScroll(d),
          gutterTotalWidth: d.gutters.offsetWidth,
          gutterLeft: left,
          gutterWidth: width,
          wrapperWidth: d.wrapper.clientWidth}
}

# 计算 display.scroller.scrollLeft + display.gutters.offsetWidth，使用 getBoundingClientRect 获取子像素精确结果
function compensateForHScroll(display) {
  return display.scroller.getBoundingClientRect().left - display.sizer.getBoundingClientRect().left
}

# 返回一个函数，估计行高，作为第一次近似，直到行变得可见（因此可以正确测量）
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

# 估计行高
function estimateLineHeights(cm) {
  var doc = cm.doc, est = estimateHeight(cm);
  doc.iter(function (line) {
    var estHeight = est(line);
    if (estHeight != line.height) { updateLineHeight(line, estHeight); }
  });
}

# 给定鼠标事件，找到相应的位置。如果 liberal 为 false，则检查是否点击了 gutter 或 scrollbar，如果是则返回 null。forRect 用于矩形选择，尝试估计字符位置，即使超出文本右侧的坐标。
function posFromMouse(cm, e, liberal, forRect) {
  var display = cm.display;
    // 如果不是宽容模式并且目标元素的属性为"cm-not-content"，则返回null
    if (!liberal && e_target(e).getAttribute("cm-not-content") == "true") { return null }

    // 获取display.lineSpace的边界矩形
    var x, y, space = display.lineSpace.getBoundingClientRect();
    // 在IE[67]上在鼠标快速拖动时不可预测地失败
    try { x = e.clientX - space.left; y = e.clientY - space.top; }
    catch (e$1) { return null }
    // 根据鼠标坐标计算出对应的字符位置
    var coords = coordsChar(cm, x, y), line;
    // 如果forRect为真且coords.xRel大于0并且line的长度等于coords.ch
    if (forRect && coords.xRel > 0 && (line = getLine(cm.doc, coords.line).text).length == coords.ch) {
      // 计算列差值
      var colDiff = countColumn(line, line.length, cm.options.tabSize) - line.length;
      // 根据x坐标计算出字符位置
      coords = Pos(coords.line, Math.max(0, Math.round((x - paddingH(cm.display).left) / charWidth(cm.display)) - colDiff));
    }
    // 返回字符位置
    return coords
  }

  // 查找与给定行对应的视图元素。当行不可见时返回null
  function findViewIndex(cm, n) {
    // 如果n大于等于display.viewTo，则返回null
    if (n >= cm.display.viewTo) { return null }
    // 将n减去display.viewFrom
    n -= cm.display.viewFrom;
    // 如果n小于0，则返回null
    if (n < 0) { return null }
    var view = cm.display.view;
    // 遍历视图元素数组
    for (var i = 0; i < view.length; i++) {
      // 将n减去当前视图元素的大小
      n -= view[i].size;
      // 如果n小于0，则返回当前索引i
      if (n < 0) { return i }
    }
  }

  // 更新display.view数据结构以适应对文档的更改。from和to是变更前的坐标。lendiff是变更的行数差异
  function regChange(cm, from, to, lendiff) {
    // 如果from为null，则设置为文档的第一行
    if (from == null) { from = cm.doc.first; }
    // 如果to为null，则设置为文档的最后一行
    if (to == null) { to = cm.doc.first + cm.doc.size; }
    // 如果lendiff为假，则设置为0
    if (!lendiff) { lendiff = 0; }

    var display = cm.display;
    // 如果lendiff不为0且to小于display.viewTo并且(display.updateLineNumbers为null或者display.updateLineNumbers大于from)
    { display.updateLineNumbers = from; }

    // 设置当前操作的视图已更改标志为true
    cm.curOp.viewChanged = true;
    if (from >= display.viewTo) { // 如果起始行大于或等于显示区域的结束行，则表示发生改变
      if (sawCollapsedSpans && visualLineNo(cm.doc, from) < display.viewTo)
        { resetView(cm); } // 如果存在折叠的跨度并且起始行在显示区域内，则重置视图
    } else if (to <= display.viewFrom) { // 如果结束行小于或等于显示区域的起始行，则表示发生改变
      if (sawCollapsedSpans && visualLineEndNo(cm.doc, to + lendiff) > display.viewFrom) {
        resetView(cm); // 如果存在折叠的跨度并且结束行在显示区域内，则重置视图
      } else {
        display.viewFrom += lendiff; // 否则，更新显示区域的起始行和结束行
        display.viewTo += lendiff;
      }
    } else if (from <= display.viewFrom && to >= display.viewTo) { // 如果起始行小于等于显示区域的起始行且结束行大于等于显示区域的结束行，则表示完全重叠
      resetView(cm); // 重置视图
    } else if (from <= display.viewFrom) { // 如果起始行小于等于显示区域的起始行，则表示顶部重叠
      var cut = viewCuttingPoint(cm, to, to + lendiff, 1); // 计算切割点
      if (cut) {
        display.view = display.view.slice(cut.index); // 更新显示区域的内容
        display.viewFrom = cut.lineN; // 更新显示区域的起始行
        display.viewTo += lendiff; // 更新显示区域的结束行
      } else {
        resetView(cm); // 否则，重置视图
      }
    } else if (to >= display.viewTo) { // 如果结束行大于等于显示区域的结束行，则表示底部重叠
      var cut$1 = viewCuttingPoint(cm, from, from, -1); // 计算切割点
      if (cut$1) {
        display.view = display.view.slice(0, cut$1.index); // 更新显示区域的内容
        display.viewTo = cut$1.lineN; // 更新显示区域的结束行
      } else {
        resetView(cm); // 否则，重置视图
      }
    } else { // 如果存在中间的间隙
      var cutTop = viewCuttingPoint(cm, from, from, -1); // 计算顶部切割点
      var cutBot = viewCuttingPoint(cm, to, to + lendiff, 1); // 计算底部切割点
      if (cutTop && cutBot) {
        display.view = display.view.slice(0, cutTop.index)
          .concat(buildViewArray(cm, cutTop.lineN, cutBot.lineN))
          .concat(display.view.slice(cutBot.index)); // 更新显示区域的内容
        display.viewTo += lendiff; // 更新显示区域的结束行
      } else {
        resetView(cm); // 否则，重置视图
      }
    }

    var ext = display.externalMeasured; // 获取外部测量值
    if (ext) {
      if (to < ext.lineN)
        { ext.lineN += lendiff; } // 如果结束行小于外部测量值的行数，则更新外部测量值的行数
      else if (from < ext.lineN + ext.size)
        { display.externalMeasured = null; } // 如果起始行小于外部测量值的行数加上大小，则将外部测量值置为null
    }
  }

  // 注册对单行的更改。类型必须是"text"、"gutter"、"class"、"widget"之一
  function regLineChange(cm, line, type) {
    cm.curOp.viewChanged = true; // 设置当前操作的视图更改为true
    // 保存对 CodeMirror 显示的引用，以及外部测量的引用
    var display = cm.display, ext = cm.display.externalMeasured;
    // 如果外部测量存在，并且行数在外部测量的范围内，则将外部测量置空
    if (ext && line >= ext.lineN && line < ext.lineN + ext.size)
      { display.externalMeasured = null; }

    // 如果行数小于显示区域的起始行或者大于等于结束行，则直接返回
    if (line < display.viewFrom || line >= display.viewTo) { return }
    // 获取指定行的视图
    var lineView = display.view[findViewIndex(cm, line)];
    // 如果视图的节点为空，则直接返回
    if (lineView.node == null) { return }
    // 获取视图的变化数组，如果不存在则创建一个空数组
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
    // 如果没有折叠的跨度或者新行数等于文档的第一行加上文档的大小，则返回新行数和索引
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

  // 强制视图覆盖给定范围，根据需要添加空的视图元素或者裁剪现有的元素
  function adjustView(cm, from, to) {
    var display = cm.display, view = display.view;
    // 如果视图数组为空，或者起始行大于等于结束行，或者结束行小于等于显示区域的起始行，则重新构建视图数组
    if (view.length == 0 || from >= display.viewTo || to <= display.viewFrom) {
      display.view = buildViewArray(cm, from, to);
      display.viewFrom = from;
    }
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
      // 如果视图的结束位置小于指定的结束位置
      if (display.viewTo < to)
        // 将构建的视图数组连接到当前视图
        { display.view = display.view.concat(buildViewArray(cm, display.viewTo, to)); }
      // 如果视图的结束位置大于指定的结束位置
      else if (display.viewTo > to)
        // 从当前视图中截取从开始到指定结束位置的部分
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

  // 更新选择
  function updateSelection(cm) {
    // 显示选择
    cm.display.input.showSelection(cm.display.input.prepareSelection());
  }

  // 准备选择
  function prepareSelection(cm, primary) {
    if ( primary === void 0 ) primary = true;

    var doc = cm.doc, result = {};
    var curFragment = result.cursors = document.createDocumentFragment();
    var selFragment = result.selection = document.createDocumentFragment();

    for (var i = 0; i < doc.sel.ranges.length; i++) {
      // 如果不是主要选择且索引等于主要索引，则继续下一次循环
      if (!primary && i == doc.sel.primIndex) { continue }
      var range = doc.sel.ranges[i];
      // 如果范围的起始行大于视图的结束位置或者结束行小于视图的起始位置，则继续下一次循环
      if (range.from().line >= cm.display.viewTo || range.to().line < cm.display.viewFrom) { continue }
      var collapsed = range.empty();
      // 如果范围是折叠的或者选中时显示光标，则绘制光标
      if (collapsed || cm.options.showCursorWhenSelecting)
        { drawSelectionCursor(cm, range.head, curFragment); }
      // 如果范围不是折叠的，则绘制选择范围
      if (!collapsed)
        { drawSelectionRange(cm, range, selFragment); }
    }
    return result
  }

  // 绘制给定范围的光标
  function drawSelectionCursor(cm, head, output) {
    # 获取光标位置的坐标信息
    var pos = cursorCoords(cm, head, "div", null, null, !cm.options.singleCursorHeightPerLine);

    # 创建光标元素并设置位置和高度
    var cursor = output.appendChild(elt("div", "\u00a0", "CodeMirror-cursor"));
    cursor.style.left = pos.left + "px";
    cursor.style.top = pos.top + "px";
    cursor.style.height = Math.max(0, pos.bottom - pos.top) * cm.options.cursorHeight + "px";

    # 如果存在另一个光标位置，则创建并设置其位置和高度
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

    # 添加高亮选择的样式和位置信息到文档片段中
    function add(left, top, width, bottom) {
      if (top < 0) { top = 0; }
      top = Math.round(top);
      bottom = Math.round(bottom);
      fragment.appendChild(elt("div", null, "CodeMirror-selected", ("position: absolute; left: " + left + "px;\n                             top: " + top + "px; width: " + (width == null ? rightSide - left : width) + "px;\n                             height: " + (bottom - top) + "px")));
    }

    # 获取选择范围的起始和结束位置
    var sFrom = range.from(), sTo = range.to();
    # 如果选择范围在同一行上，则绘制该行的高亮选择
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
        // 如果左侧结束位置的底部小于右侧开始位置的顶部减2
        if (leftEnd.top < rightStart.top - 2) {
          // 添加左侧结束位置到右侧开始位置之间的矩形
          add(leftEnd.right, leftEnd.top, null, leftEnd.bottom);
          // 添加左侧边界到右侧开始位置的矩形
          add(leftSide, rightStart.top, rightStart.left, rightStart.bottom);
        } else {
          // 添加左侧结束位置到右侧开始位置之间的矩形
          add(leftEnd.right, leftEnd.top, rightStart.left - leftEnd.right, leftEnd.bottom);
        }
      }
      // 如果左侧结束位置的底部小于右侧开始位置的顶部
      if (leftEnd.bottom < rightStart.top)
        { add(leftSide, leftEnd.bottom, null, rightStart.top); }
    }

    // 将fragment添加到output中
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
    // 如果光标闪烁速率大于0
    if (cm.options.cursorBlinkRate > 0)
      { 
        // 设置光标闪烁定时器
        display.blinker = setInterval(function () { return display.cursorDiv.style.visibility = (on = !on) ? "" : "hidden"; },
        cm.options.cursorBlinkRate); 
      }
    // 如果光标闪烁速率小于0
    else if (cm.options.cursorBlinkRate < 0)
      { display.cursorDiv.style.visibility = "hidden"; }
  }

  // 确保焦点在编辑器上
  function ensureFocus(cm) {
    // 如果编辑器失去焦点，则将焦点设置到编辑器上
    if (!cm.state.focused) { cm.display.input.focus(); onFocus(cm); }
  }

  // 延迟失去焦点事件
  function delayBlurEvent(cm) {
    // 设置延迟失去焦点事件状态为true
    cm.state.delayingBlurEvent = true;
    // 延迟100ms后执行失去焦点事件
    setTimeout(function () { if (cm.state.delayingBlurEvent) {
      cm.state.delayingBlurEvent = false;
      onBlur(cm);
    } }, 100);
  }

  // 获取焦点时的事件处理
  function onFocus(cm, e) {
    // 如果正在延迟失去焦点事件，则将延迟失去焦点事件状态设置为false
    if (cm.state.delayingBlurEvent) { cm.state.delayingBlurEvent = false; }

    // 如果编辑器为只读且没有光标，则返回
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
    // 遍历 display.view 数组
    for (var i = 0; i < display.view.length; i++) {
      // 获取当前行对象和是否启用换行
      var cur = display.view[i], wrapping = cm.options.lineWrapping;
      // 初始化高度和宽度变量
      var height = (void 0), width = 0;
      // 如果当前行被隐藏，则跳过
      if (cur.hidden) { continue }
      // 如果是 IE 并且版本小于 8
      if (ie && ie_version < 8) {
        // 获取当前行底部位置
        var bot = cur.node.offsetTop + cur.node.offsetHeight;
        // 计算高度差
        height = bot - prevBottom;
        prevBottom = bot;
      } else {
        // 获取当前行的盒子模型信息
        var box = cur.node.getBoundingClientRect();
        // 计算高度差
        height = box.bottom - box.top;
        // 如果不启用换行并且当前行有文本
        if (!wrapping && cur.text.firstChild)
          { width = cur.text.firstChild.getBoundingClientRect().right - box.left - 1; }
      }
      // 计算高度差
      var diff = cur.line.height - height;
      // 如果高度差超过阈值
      if (diff > .005 || diff < -.005) {
        // 更新行高
        updateLineHeight(cur.line, height);
        // 更新行部件高度
        updateWidgetHeight(cur.line);
        // 如果当前行有其他部件
        if (cur.rest) { for (var j = 0; j < cur.rest.length; j++)
          { updateWidgetHeight(cur.rest[j]); } }
      }
      // 如果宽度超过编辑器宽度
      if (width > cm.display.sizerWidth) {
        // 计算字符宽度
        var chWidth = Math.ceil(width / charWidth(cm.display));
        // 如果字符宽度超过最大行长度
        if (chWidth > cm.display.maxLineLength) {
          // 更新最大行长度和最大行对象
          cm.display.maxLineLength = chWidth;
          cm.display.maxLine = cur.line;
          cm.display.maxLineChanged = true;
        }
      }
    }
  }

  // 读取并存储与给定行相关的行部件的高度
  function updateWidgetHeight(line) {
    // 如果行有部件
    if (line.widgets) { for (var i = 0; i < line.widgets.length; ++i) {
      // 获取部件对象和父节点
      var w = line.widgets[i], parent = w.node.parentNode;
      // 如果有父节点，则更新部件高度
      if (parent) { w.height = parent.offsetHeight; }
    } }
  }

  // 计算在给定视口中可见的行（默认为当前滚动位置）
  // 视口可能包含 top、height 和 ensure（参见 op.scrollToPos）属性
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

  // 将编辑器中的内容滚动到视图中

  // 如果编辑器位于窗口顶部或底部，部分滚动出视图，这将确保光标可见
  function maybeScrollWindow(cm, rect) {
    // 如果触发了"scrollCursorIntoView"事件，则返回
    if (signalDOMEvent(cm, "scrollCursorIntoView")) { return }

    var display = cm.display, box = display.sizer.getBoundingClientRect(), doScroll = null;
    // 如果矩形的顶部加上显示区域的顶部小于0，则需要滚动
    if (rect.top + box.top < 0) { doScroll = true; }
    // 如果矩形的底部加上显示区域的顶部大于窗口高度，则不需要滚动
    else if (rect.bottom + box.top > (window.innerHeight || document.documentElement.clientHeight)) { doScroll = false; }
    # 如果 doScroll 不为空且不是幻影模式
    if (doScroll != null && !phantom) {
      # 创建一个滚动节点，设置其样式和位置
      var scrollNode = elt("div", "\u200b", null, ("position: absolute;\n                         top: " + (rect.top - display.viewOffset - paddingTop(cm.display)) + "px;\n                         height: " + (rect.bottom - rect.top + scrollGap(cm) + display.barHeight) + "px;\n                         left: " + (rect.left) + "px; width: " + (Math.max(2, rect.right - rect.left)) + "px;"));
      # 将滚动节点添加到显示区域的行间隔中
      cm.display.lineSpace.appendChild(scrollNode);
      # 将滚动节点滚动到视图中
      scrollNode.scrollIntoView(doScroll);
      # 从显示区域的行间隔中移除滚动节点
      cm.display.lineSpace.removeChild(scrollNode);
    }
  }

  # 将给定位置滚动到视图中（立即），验证其是否实际可见（因为行高度准确测量，某些位置在绘制过程中可能会“漂移”）
  function scrollPosIntoView(cm, pos, end, margin) {
    # 如果 margin 为空，则设置为 0
    if (margin == null) { margin = 0; }
    # 定义一个矩形变量
    var rect;
    # 如果不使用行包裹且位置等于结束位置
    if (!cm.options.lineWrapping && pos == end) {
      # 设置 pos 和 end 为光标位置周围的字符位置
      # 如果 pos.sticky == "before"，则在 pos.ch - 1 周围，否则在 pos.ch 周围
      # 如果 pos == Pos(_, 0, "before")，则 pos 和 end 不变
      pos = pos.ch ? Pos(pos.line, pos.sticky == "before" ? pos.ch - 1 : pos.ch, "after") : pos;
      end = pos.sticky == "before" ? Pos(pos.line, pos.ch + 1, "before") : pos;
    }
    // 循环执行5次，用于限制滚动次数
    for (var limit = 0; limit < 5; limit++) {
      // 标记是否发生了滚动
      var changed = false;
      // 获取光标位置的坐标
      var coords = cursorCoords(cm, pos);
      // 获取结束位置的坐标
      var endCoords = !end || end == pos ? coords : cursorCoords(cm, end);
      // 计算矩形区域的坐标
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
        // 如果滚动距离超过1个像素，标记为发生了滚动
        if (Math.abs(cm.doc.scrollTop - startTop) > 1) { changed = true; }
      }
      // 如果有需要滚动的水平位置
      if (scrollPos.scrollLeft != null) {
        // 更新水平滚动位置
        setScrollLeft(cm, scrollPos.scrollLeft);
        // 如果滚动距离超过1个像素，标记为发生了滚动
        if (Math.abs(cm.doc.scrollLeft - startLeft) > 1) { changed = true; }
      }
      // 如果没有发生滚动，跳出循环
      if (!changed) { break }
    }
    // 返回矩形区域的坐标
    return rect
  }

  // 立即将给定的坐标滚动到可视区域内
  function scrollIntoView(cm, rect) {
    // 计算滚动位置
    var scrollPos = calculateScrollPos(cm, rect);
    // 如果有需要滚动的垂直位置
    if (scrollPos.scrollTop != null) { updateScrollTop(cm, scrollPos.scrollTop); }
    // 如果有需要滚动的水平位置
    if (scrollPos.scrollLeft != null) { setScrollLeft(cm, scrollPos.scrollLeft); }
  }

  // 计算滚动到给定矩形区域的新滚动位置
  // 返回一个包含scrollTop和scrollLeft属性的对象
  // 当这些属性为undefined时，垂直/水平位置不需要调整
  function calculateScrollPos(cm, rect) {
    // 获取显示区域和文本高度
    var display = cm.display, snapMargin = textHeight(cm.display);
    // 如果矩形区域的顶部超出了可视区域，将其调整为0
    if (rect.top < 0) { rect.top = 0; }
    // 获取当前滚动位置
    var screentop = cm.curOp && cm.curOp.scrollTop != null ? cm.curOp.scrollTop : display.scroller.scrollTop;
    var screen = displayHeight(cm), result = {};
    // 如果矩形区域的高度超出了可视区域的高度，将其调整为可视区域的高度
    if (rect.bottom - rect.top > screen) { rect.bottom = rect.top + screen; }
    // 获取文档底部位置
    var docBottom = cm.doc.height + paddingVert(display);
    // 检查矩形是否在顶部或底部需要进行吸附
    var atTop = rect.top < snapMargin, atBottom = rect.bottom > docBottom - snapMargin;
    // 如果矩形在屏幕顶部之上
    if (rect.top < screentop) {
      // 如果在顶部，滚动到顶部；否则滚动到矩形的顶部
      result.scrollTop = atTop ? 0 : rect.top;
    } else if (rect.bottom > screentop + screen) {
      // 计算新的顶部位置，确保矩形在屏幕内
      var newTop = Math.min(rect.top, (atBottom ? docBottom : rect.bottom) - screen);
      // 如果新的顶部位置不等于当前顶部位置，进行滚动
      if (newTop != screentop) { result.scrollTop = newTop; }
    }

    // 获取屏幕左侧位置和宽度
    var screenleft = cm.curOp && cm.curOp.scrollLeft != null ? cm.curOp.scrollLeft : display.scroller.scrollLeft;
    var screenw = displayWidth(cm) - (cm.options.fixedGutter ? display.gutters.offsetWidth : 0);
    // 检查矩形是否过宽，如果是则调整右侧位置
    var tooWide = rect.right - rect.left > screenw;
    if (tooWide) { rect.right = rect.left + screenw; }
    // 根据矩形左侧位置进行滚动
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
    // 如果top为null，直接返回
    if (top == null) { return }
    // 解析滚动位置
    resolveScrollToPos(cm);
    // 计算新的滚动位置
    cm.curOp.scrollTop = (cm.curOp.scrollTop == null ? cm.doc.scrollTop : cm.curOp.scrollTop) + top;
  }

  // 确保在操作结束时当前光标可见
  function ensureCursorVisible(cm) {
    // 解析滚动位置
    resolveScrollToPos(cm);
    // 获取当前光标位置，并设置滚动位置
    var cur = cm.getCursor();
    cm.curOp.scrollToPos = {from: cur, to: cur, margin: cm.options.cursorScrollMargin};
  }

  // 滚动到指定坐标
  function scrollToCoords(cm, x, y) {
    // 如果x或y不为null，解析滚动位置
    if (x != null || y != null) { resolveScrollToPos(cm); }
    // 如果x不为null，设置水平滚动位置
    if (x != null) { cm.curOp.scrollLeft = x; }
    // 如果y不为null，设置垂直滚动位置
    if (y != null) { cm.curOp.scrollTop = y; }
  }

  // 滚动到指定范围
  function scrollToRange(cm, range) {
    // 解析滚动位置
    resolveScrollToPos(cm);
  // 设置当前操作的滚动位置为指定范围
  cm.curOp.scrollToPos = range;
}

// 当操作的 scrollToPos 属性被设置，并且在操作结束之前应用了另一个滚动操作时，这个函数会以一种简单的方式“模拟”将该位置滚动到视图中，以便中间滚动命令的效果不会被忽略。
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
    // 如果是滚动条，并且值等于文档的水平滚动位置，或者水平滚动位置的变化小于2，并且不是强制滚动，则返回
    if ((isScroller ? val == cm.doc.scrollLeft : Math.abs(cm.doc.scrollLeft - val) < 2) && !forceScroll) { return }
    // 设置文档的水平滚动位置为给定值
    cm.doc.scrollLeft = val;
    // 水平对齐
    alignHorizontally(cm);
    // 如果显示区域的水平滚动位置不等于给定值，则设置显示区域的水平滚动位置为给定值
    if (cm.display.scroller.scrollLeft != val) { cm.display.scroller.scrollLeft = val; }
    // 设置滚动条的水平滚动位置为给定值
    cm.display.scrollbars.setScrollLeft(val);
  }

  // SCROLLBARS

  // 准备更新滚动条所需的 DOM 读取。一次性完成以最小化更新/测量的往返
  function measureForScrollbars(cm) {
    // 获取显示区域的相关信息
    var d = cm.display, gutterW = d.gutters.offsetWidth;
    var docH = Math.round(cm.doc.height + paddingVert(cm.display));
    // 返回包含相关信息的对象
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
    // 将滚动条元素添加到指定位置
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

    # 返回需要添加的右边和底部偏移量
    return {right: needsV ? sWidth : 0, bottom: needsH ? sWidth : 0}
  };

  # 设置水平滚动条位置
  NativeScrollbars.prototype.setScrollLeft = function (pos) {
    if (this.horiz.scrollLeft != pos) { this.horiz.scrollLeft = pos; }
    # 如果禁用水平滚动条，则启用零宽度滚动条
    if (this.disableHoriz) { this.enableZeroWidthBar(this.horiz, this.disableHoriz, "horiz"); }
  };

  # 设置垂直滚动条位置
  NativeScrollbars.prototype.setScrollTop = function (pos) {
    if (this.vert.scrollTop != pos) { this.vert.scrollTop = pos; }
    # 如果禁用垂直滚动条，则启用零宽度滚动条
    if (this.disableVert) { this.enableZeroWidthBar(this.vert, this.disableVert, "vert"); }
  };

  # 修复零宽度滚动条的 hack
  NativeScrollbars.prototype.zeroWidthHack = function () {
    # 根据浏览器类型设置宽度
    var w = mac && !mac_geMountainLion ? "12px" : "18px";
  // 设置水平滚动条的高度和垂直滚动条的宽度为指定的宽度
  this.horiz.style.height = this.vert.style.width = w;
  // 设置水平滚动条和垂直滚动条的指针事件为"none"
  this.horiz.style.pointerEvents = this.vert.style.pointerEvents = "none";
  // 创建一个延迟对象来禁用水平滚动条
  this.disableHoriz = new Delayed;
  // 创建一个延迟对象来禁用垂直滚动条
  this.disableVert = new Delayed;
};

// 启用零宽度滚动条
NativeScrollbars.prototype.enableZeroWidthBar = function (bar, delay, type) {
  // 设置滚动条的指针事件为"auto"
  bar.style.pointerEvents = "auto";
  // 定义一个函数来检查滚动条是否仍然可见
  function maybeDisable() {
    // 通过获取滚动条的边界框和底部右侧像素下的元素来判断滚动条是否可见
    var box = bar.getBoundingClientRect();
    var elt = type == "vert" ? document.elementFromPoint(box.right - 1, (box.top + box.bottom) / 2)
        : document.elementFromPoint((box.right + box.left) / 2, box.bottom - 1);
    // 如果元素不是滚动条本身，则禁用指针事件，否则设置延迟来再次检查
    if (elt != bar) { bar.style.pointerEvents = "none"; }
    else { delay.set(1000, maybeDisable); }
  }
  // 设置延迟来检查滚动条是否可见
  delay.set(1000, maybeDisable);
};

// 清除滚动条
NativeScrollbars.prototype.clear = function () {
  // 获取水平滚动条的父元素，并移除水平滚动条和垂直滚动条
  var parent = this.horiz.parentNode;
  parent.removeChild(this.horiz);
  parent.removeChild(this.vert);
};

// 定义一个空的滚动条对象
var NullScrollbars = function () {};

// 更新滚动条
NullScrollbars.prototype.update = function () { return {bottom: 0, right: 0} };
// 设置滚动条的水平滚动位置
NullScrollbars.prototype.setScrollLeft = function () {};
// 设置滚动条的垂直滚动位置
NullScrollbars.prototype.setScrollTop = function () {};
// 清除滚动条
NullScrollbars.prototype.clear = function () {};

// 更新滚动条
function updateScrollbars(cm, measure) {
  // 如果没有测量值，则调用measureForScrollbars函数进行测量
  if (!measure) { measure = measureForScrollbars(cm); }
  // 获取滚动条的初始宽度和高度
  var startWidth = cm.display.barWidth, startHeight = cm.display.barHeight;
  // 调用updateScrollbarsInner函数来更新滚动条
  updateScrollbarsInner(cm, measure);
    // 循环执行以下操作，直到满足条件：i < 4 且 startWidth 不等于 cm.display.barWidth 或 startHeight 不等于 cm.display.barHeight
    for (var i = 0; i < 4 && startWidth != cm.display.barWidth || startHeight != cm.display.barHeight; i++) {
      // 如果 startWidth 不等于 cm.display.barWidth 并且 cm.options.lineWrapping 为真，则执行 updateHeightsInViewport 函数
      if (startWidth != cm.display.barWidth && cm.options.lineWrapping)
        { updateHeightsInViewport(cm); }
      // 调用 updateScrollbarsInner 函数，传入参数 cm 和 measureForScrollbars(cm)，更新滚动条
      updateScrollbarsInner(cm, measureForScrollbars(cm));
      // 更新 startWidth 和 startHeight 的值为 cm.display.barWidth 和 cm.display.barHeight
      startWidth = cm.display.barWidth; startHeight = cm.display.barHeight;
    }
  }

  // 重新同步虚拟滚动条与内容的实际大小
  function updateScrollbarsInner(cm, measure) {
    // 获取 cm.display 对象
    var d = cm.display;
    // 调用 d.scrollbars.update 函数，传入 measure 参数，更新滚动条的大小
    var sizes = d.scrollbars.update(measure);

    // 设置 sizer 元素的右内边距为滚动条的右侧宽度
    d.sizer.style.paddingRight = (d.barWidth = sizes.right) + "px";
    // 设置 sizer 元素的底部内边距为滚动条的底部高度
    d.sizer.style.paddingBottom = (d.barHeight = sizes.bottom) + "px";
    // 设置 heightForcer 元素的底部边框为滚动条的底部高度
    d.heightForcer.style.borderBottom = sizes.bottom + "px solid transparent";

    // 如果右侧滚动条和底部滚动条都存在
    if (sizes.right && sizes.bottom) {
      // 设置 scrollbarFiller 元素显示，并设置其高度和宽度
      d.scrollbarFiller.style.display = "block";
      d.scrollbarFiller.style.height = sizes.bottom + "px";
      d.scrollbarFiller.style.width = sizes.right + "px";
    } else { d.scrollbarFiller.style.display = ""; }
    // 如果底部滚动条存在，并且 coverGutterNextToScrollbar 和 fixedGutter 都为真
    if (sizes.bottom && cm.options.coverGutterNextToScrollbar && cm.options.fixedGutter) {
      // 设置 gutterFiller 元素显示，并设置其高度和宽度
      d.gutterFiller.style.display = "block";
      d.gutterFiller.style.height = sizes.bottom + "px";
      d.gutterFiller.style.width = measure.gutterWidth + "px";
    } else { d.gutterFiller.style.display = ""; }
  }

  // 定义 scrollbarModel 对象，包含 "native" 和 "null" 两个属性
  var scrollbarModel = {"native": NativeScrollbars, "null": NullScrollbars};

  // 初始化滚动条
  function initScrollbars(cm) {
    // 如果 cm.display.scrollbars 存在，则清除滚动条
    if (cm.display.scrollbars) {
      cm.display.scrollbars.clear();
      // 如果 cm.display.scrollbars.addClass 存在，则移除 cm.display.wrapper 的指定类名
      if (cm.display.scrollbars.addClass)
        { rmClass(cm.display.wrapper, cm.display.scrollbars.addClass); }
    }
  }
    # 设置滚动条样式，并将滚动条插入到编辑器显示区域中
    cm.display.scrollbars = new scrollbarModel[cm.options.scrollbarStyle](function (node) {
      cm.display.wrapper.insertBefore(node, cm.display.scrollbarFiller);
      # 防止在滚动条上的点击事件导致焦点丢失
      on(node, "mousedown", function () {
        if (cm.state.focused) { setTimeout(function () { return cm.display.input.focus(); }, 0); }
      });
      # 设置节点属性，表示不是编辑器内容
      node.setAttribute("cm-not-content", "true");
    }, function (pos, axis) {
      if (axis == "horizontal") { setScrollLeft(cm, pos); }
      else { updateScrollTop(cm, pos); }
    }, cm);
    # 如果滚动条有添加类的方法，则为编辑器显示区域添加滚动条类
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
      typing: false,           // 是否小心地保留现有文本（用于合成）
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

  // 操作完成时进行的 DOM 更新被批处理，以便需要最少的重排
  function endOperations(group) {
    var ops = group.ops;
    for (var i = 0; i < ops.length; i++) // 读取 DOM
      { endOperation_R1(ops[i]); }
    for (var i$1 = 0; i$1 < ops.length; i$1++) // 写入 DOM（可能）
      { endOperation_W1(ops[i$1]); }
    for (var i$2 = 0; i$2 < ops.length; i$2++) // 读取 DOM
      { endOperation_R2(ops[i$2]); }
    // 遍历操作数组，执行写入 DOM 操作
    for (var i$3 = 0; i$3 < ops.length; i$3++) // Write DOM (maybe)
      { endOperation_W2(ops[i$3]); }
    // 再次遍历操作数组，执行读取 DOM 操作
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
    // 如果需要更新，则创建更新对象
    op.update = op.mustUpdate &&
      new DisplayUpdate(cm, op.mustUpdate && {top: op.scrollTop, ensure: op.scrollToPos}, op.forceUpdate);
  }

  // 结束写入操作，处理相关逻辑
  function endOperation_W1(op) {
    // 如果需要更新显示，则执行更新显示操作
    op.updatedDisplay = op.mustUpdate && updateDisplayIfNeeded(op.cm, op.update);
  }

  // 再次结束读取操作，处理相关逻辑
  function endOperation_R2(op) {
    var cm = op.cm, display = cm.display;
    // 如果更新了显示，则更新视口高度
    if (op.updatedDisplay) { updateHeightsInViewport(cm); }

    // 计算滚动条的测量值
    op.barMeasure = measureForScrollbars(cm);

    // 如果最大行数发生变化且不换行，则测量最大行数的宽度
    if (display.maxLineChanged && !cm.options.lineWrapping) {
      op.adjustWidthTo = measureChar(cm, display.maxLine, display.maxLine.text.length).left + 3;
      cm.display.sizerWidth = op.adjustWidthTo;
      op.barMeasure.scrollWidth =
        Math.max(display.scroller.clientWidth, display.sizer.offsetLeft + op.adjustWidthTo + scrollGap(cm) + cm.display.barWidth);
      op.maxScrollLeft = Math.max(0, display.sizer.offsetLeft + op.adjustWidthTo - displayWidth(cm));
    }

    // 如果更新了显示或者选择发生变化，则准备选择
    if (op.updatedDisplay || op.selectionChanged)
      { op.preparedSelection = display.input.prepareSelection(); }
  }

  // 再次结束写入操作，处理相关逻辑
  function endOperation_W2(op) {
    var cm = op.cm;
    // 这里是函数的其余部分，未提供
  }
    // 如果需要调整宽度
    if (op.adjustWidthTo != null) {
      // 设置编辑器容器的最小宽度
      cm.display.sizer.style.minWidth = op.adjustWidthTo + "px";
      // 如果最大滚动距离小于文档的水平滚动位置
      if (op.maxScrollLeft < cm.doc.scrollLeft)
        // 设置水平滚动位置为最大滚动距离
        { setScrollLeft(cm, Math.min(cm.display.scroller.scrollLeft, op.maxScrollLeft), true); }
      // 标记最大行改变为false
      cm.display.maxLineChanged = false;
    }

    // 如果需要获取焦点并且焦点是活动元素
    var takeFocus = op.focus && op.focus == activeElt();
    // 如果有准备好的选择
    if (op.preparedSelection)
      // 显示准备好的选择
      { cm.display.input.showSelection(op.preparedSelection, takeFocus); }
    // 如果更新了显示或者起始高度不等于文档高度
    if (op.updatedDisplay || op.startHeight != cm.doc.height)
      // 更新滚动条
      { updateScrollbars(cm, op.barMeasure); }
    // 如果更新了显示
    if (op.updatedDisplay)
      // 设置文档高度
      { setDocumentHeight(cm, op.barMeasure); }

    // 如果选择改变了
    if (op.selectionChanged) { restartBlink(cm); }

    // 如果编辑器处于焦点状态并且需要更新输入
    if (cm.state.focused && op.updateInput)
      // 重置输入
      { cm.display.input.reset(op.typing); }
    // 如果需要获取焦点
    if (takeFocus) { ensureFocus(op.cm); }
  }

  // 结束操作的最后步骤
  function endOperation_finish(op) {
    var cm = op.cm, display = cm.display, doc = cm.doc;

    // 如果更新了显示
    if (op.updatedDisplay) { postUpdateDisplay(cm, op.update); }

    // 当显式滚动时，中止鼠标滚轮的delta测量
    if (display.wheelStartX != null && (op.scrollTop != null || op.scrollLeft != null || op.scrollToPos))
      { display.wheelStartX = display.wheelStartY = null; }

    // 将滚动位置传播到实际的DOM滚动器
    if (op.scrollTop != null) { setScrollTop(cm, op.scrollTop, op.forceScroll); }

    // 如果需要设置水平滚动位置
    if (op.scrollLeft != null) { setScrollLeft(cm, op.scrollLeft, true, true); }
    // 如果需要将特定位置滚动到视图中
    if (op.scrollToPos) {
      // 将位置滚动到视图中
      var rect = scrollPosIntoView(cm, clipPos(doc, op.scrollToPos.from),
                                   clipPos(doc, op.scrollToPos.to), op.scrollToPos.margin);
      // 可能滚动窗口
      maybeScrollWindow(cm, rect);
    }

    // 触发由编辑或撤销隐藏/显示的标记的事件
    var hidden = op.maybeHiddenMarkers, unhidden = op.maybeUnhiddenMarkers;
    // 如果存在隐藏的内容，遍历隐藏的内容
    if (hidden) { for (var i = 0; i < hidden.length; ++i)
      { if (!hidden[i].lines.length) { signal(hidden[i], "hide"); } } }
    // 如果存在未隐藏的内容，遍历未隐藏的内容
    if (unhidden) { for (var i$1 = 0; i$1 < unhidden.length; ++i$1)
      { if (unhidden[i$1].lines.length) { signal(unhidden[i$1], "unhide"); } } }

    // 如果显示区域的高度不为0，设置文档的滚动位置为编辑器的滚动位置
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
    // 如果文档的高亮边界小于显示区域的结束位置，设置高亮状态
    if (cm.doc.highlightFrontier < cm.display.viewTo)
      { cm.state.highlight.set(time, bind(highlightWorker, cm)); }
  }

  // 高亮工作器
  function highlightWorker(cm) {
    var doc = cm.doc;
    // 如果文档的高亮边界大于等于显示区域的结束位置，返回
    if (doc.highlightFrontier >= cm.display.viewTo) { return }
    // 计算结束时间
    var end = +new Date + cm.options.workTime;
    // 获取当前光标位置之前的上下文
    var context = getContextBefore(cm, doc.highlightFrontier);
    // 存储发生变化的行
    var changedLines = [];

    // 迭代处理文档中的每一行
    doc.iter(context.line, Math.min(doc.first + doc.size, cm.display.viewTo + 500), function (line) {
      // 如果当前行在可见区域内
      if (context.line >= cm.display.viewFrom) { // Visible
        // 存储当前行的样式
        var oldStyles = line.styles;
        // 如果当前行的文本长度超过最大高亮长度，则复制当前状态
        var resetState = line.text.length > cm.options.maxHighlightLength ? copyState(doc.mode, context.state) : null;
        // 对当前行进行高亮处理
        var highlighted = highlightLine(cm, line, context, true);
        // 如果存在复制的状态，则将当前状态重置为复制的状态
        if (resetState) { context.state = resetState; }
        // 更新当前行的样式
        line.styles = highlighted.styles;
        // 更新当前行的类名
        var oldCls = line.styleClasses, newCls = highlighted.classes;
        if (newCls) { line.styleClasses = newCls; }
        else if (oldCls) { line.styleClasses = null; }
        // 检查当前行是否发生了变化
        var ischange = !oldStyles || oldStyles.length != line.styles.length ||
          oldCls != newCls && (!oldCls || !newCls || oldCls.bgClass != newCls.bgClass || oldCls.textClass != newCls.textClass);
        for (var i = 0; !ischange && i < oldStyles.length; ++i) { ischange = oldStyles[i] != line.styles[i]; }
        // 如果当前行发生了变化，则将其添加到变化行列表中
        if (ischange) { changedLines.push(context.line); }
        // 保存当前行的状态
        line.stateAfter = context.save();
        // 移动到下一行
        context.nextLine();
      } else {
        // 如果当前行不在可见区域内且文本长度小于等于最大高亮长度，则处理当前行
        if (line.text.length <= cm.options.maxHighlightLength)
          { processLine(cm, line.text, context); }
        // 设置当前行的状态
        line.stateAfter = context.line % 5 == 0 ? context.save() : null;
        // 移动到下一行
        context.nextLine();
      }
      // 如果当前时间超过了结束时间，则启动工作线程并返回true
      if (+new Date > end) {
        startWorker(cm, cm.options.workDelay);
        return true
      }
    });
    // 更新文档的高亮边界
    doc.highlightFrontier = context.line;
    // 更新文档的模式边界
    doc.modeFrontier = Math.max(doc.modeFrontier, context.line);
    // 如果存在变化的行，则在操作中运行
    if (changedLines.length) { runInOp(cm, function () {
      for (var i = 0; i < changedLines.length; i++)
        { regLineChange(cm, changedLines[i], "text"); }
    }); }
  }

  // 显示绘制

  // 定义 DisplayUpdate 类
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
  // 设置行号和边线标记的高度
  cm.display.gutters.style.height = (measure.docHeight + cm.display.barHeight + scrollGap(cm)) + "px";
}

// 重新对齐行号和边线标记，以补偿水平滚动
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

// 用于确保行号边线的大小适合当前文档大小。当需要更新时返回 true
function maybeUpdateLineNumberWidth(cm) {
  if (!cm.options.lineNumbers) { return false }
  var doc = cm.doc, last = lineNumberFor(cm.options, doc.first + doc.size - 1), display = cm.display;
    # 如果最后一行的长度不等于显示行号的字符数
    if (last.length != display.lineNumChars) {
      # 创建一个测试元素，用于测量行号的宽度
      var test = display.measure.appendChild(elt("div", [elt("div", last)],
                                                 "CodeMirror-linenumber CodeMirror-gutter-elt"));
      # 获取内部宽度和内外宽度的差值，用于计算行号的宽度
      var innerW = test.firstChild.offsetWidth, padding = test.offsetWidth - innerW;
      # 重置行号的样式和宽度
      display.lineGutter.style.width = "";
      # 计算行号的内部宽度
      display.lineNumInnerWidth = Math.max(innerW, display.lineGutter.offsetWidth - padding) + 1;
      # 计算行号的总宽度
      display.lineNumWidth = display.lineNumInnerWidth + padding;
      # 更新显示行号的字符数
      display.lineNumChars = display.lineNumInnerWidth ? last.length : -1;
      # 设置行号的宽度
      display.lineGutter.style.width = display.lineNumWidth + "px";
      # 更新 gutter 空间
      updateGutterSpace(cm.display);
      # 返回 true
      return true
    }
    # 返回 false
    return false
  }

  # 获取 gutter 的样式和类名
  function getGutters(gutters, lineNumbers) {
    var result = [], sawLineNumbers = false;
    for (var i = 0; i < gutters.length; i++) {
      var name = gutters[i], style = null;
      # 如果 gutter 不是字符串，获取样式和类名
      if (typeof name != "string") { style = name.style; name = name.className; }
      # 如果类名是 "CodeMirror-linenumbers"
      if (name == "CodeMirror-linenumbers") {
        # 如果不需要显示行号，跳过
        if (!lineNumbers) { continue }
        # 否则标记已经看到行号
        else { sawLineNumbers = true; }
      }
      # 将类名和样式添加到结果数组中
      result.push({className: name, style: style});
    }
    # 如果需要显示行号且没有看到行号，添加行号的类名和样式到结果数组中
    if (lineNumbers && !sawLineNumbers) { result.push({className: "CodeMirror-linenumbers", style: null}); }
    # 返回结果数组
    return result
  }

  # 重建 gutter 元素，确保代码左侧的边距与它们的宽度匹配
  function renderGutters(display) {
    # 获取 gutter 和 gutter 的规格
    var gutters = display.gutters, specs = display.gutterSpecs;
    # 移除所有 gutter 的子元素
    removeChildren(gutters);
    # 重置 lineGutter 为 null
    display.lineGutter = null;
    # 遍历 specs 数组
    for (var i = 0; i < specs.length; ++i) {
      # 获取当前 specs 元素
      var ref = specs[i];
      # 获取当前 specs 元素的 className
      var className = ref.className;
      # 获取当前 specs 元素的 style
      var style = ref.style;
      # 在 gutters 中添加一个 div 元素，类名为 "CodeMirror-gutter " + className
      var gElt = gutters.appendChild(elt("div", null, "CodeMirror-gutter " + className));
      # 如果 style 存在，则将 gElt 的样式设置为 style
      if (style) { gElt.style.cssText = style; }
      # 如果 className 为 "CodeMirror-linenumbers"，则将 display.lineGutter 设置为 gElt，并设置其宽度
      if (className == "CodeMirror-linenumbers") {
        display.lineGutter = gElt;
        gElt.style.width = (display.lineNumWidth || 1) + "px";
      }
    }
    # 如果 specs 数组长度不为 0，则设置 gutters 的显示为默认，否则设置为隐藏
    gutters.style.display = specs.length ? "" : "none";
    # 更新 gutter 空间
    updateGutterSpace(display);
  }

  # 更新 gutter
  function updateGutters(cm) {
    # 渲染 gutter
    renderGutters(cm.display);
    # 注册改变
    regChange(cm);
    # 水平对齐
    alignHorizontally(cm);
  }

  # display 处理 DOM 集成，包括输入读取和内容绘制。它保存对 DOM 节点和与显示相关的状态的引用。
  function Display(place, doc, input, options) {
    var d = this;
    # 保存输入
    this.input = input;

    # 当两个滚动条都存在时，覆盖底部右侧的方块
    d.scrollbarFiller = elt("div", null, "CodeMirror-scrollbar-filler");
    d.scrollbarFiller.setAttribute("cm-not-content", "true");
    # 当 coverGutterNextToScrollbar 打开且水平滚动条存在时，覆盖 gutter 底部
    d.gutterFiller = elt("div", null, "CodeMirror-gutter-filler");
    d.gutterFiller.setAttribute("cm-not-content", "true");
    # 将包含实际代码的元素定位到覆盖视口
    d.lineDiv = eltP("div", null, "CodeMirror-code");
    # 用于表示选择和光标的元素添加到这些元素中
    d.selectionDiv = elt("div", null, null, "position: relative; z-index: 1");
    d.cursorDiv = elt("div", null, "CodeMirror-cursors");
    # 用于查找大小的 visibility: hidden 元素
    d.measure = elt("div", null, "CodeMirror-measure");
    # 当视口外的行被测量时，它们会在这里绘制
    d.lineMeasure = elt("div", null, "CodeMirror-measure");
    // 创建一个包含需要存在于垂直填充坐标系统内的所有内容的容器
    d.lineSpace = eltP("div", [d.measure, d.lineMeasure, d.selectionDiv, d.cursorDiv, d.lineDiv],
                      null, "position: relative; outline: none");
    // 创建一个包含行空间的容器，用于移动到父容器以覆盖可见视图
    var lines = eltP("div", [d.lineSpace], "CodeMirror-lines");
    // 创建一个相对定位的容器，用于移动到父容器
    d.mover = elt("div", [lines], null, "position: relative");
    // 设置为文档的高度，允许滚动
    d.sizer = elt("div", [d.mover], "CodeMirror-sizer");
    d.sizerWidth = null;
    // 元素的行为，具有溢出:自动和填充的行为在不同浏览器中不一致。这用于确保可滚动区域足够大
    d.heightForcer = elt("div", null, null, "position: absolute; height: " + scrollerGap + "px; width: 1px;");
    // 将包含 gutter 的容器
    d.gutters = elt("div", null, "CodeMirror-gutters");
    d.lineGutter = null;
    // 实际可滚动的元素
    d.scroller = elt("div", [d.sizer, d.heightForcer, d.gutters], "CodeMirror-scroll");
    d.scroller.setAttribute("tabIndex", "-1");
    // 编辑器所在的元素
    d.wrapper = elt("div", [d.scrollbarFiller, d.gutterFiller, d.scroller], "CodeMirror");

    // 解决 IE7 的 z-index bug（不完美，因此 IE7 实际上并不被支持）
    if (ie && ie_version < 8) { d.gutters.style.zIndex = -1; d.scroller.style.paddingRight = 0; }
    // 如果不是 webkit 并且不是 gecko 并且不是移动设备，则设置可拖动
    if (!webkit && !(gecko && mobile)) { d.scroller.draggable = true; }

    // 如果有指定的位置
    if (place) {
      // 如果 place 是一个 DOM 元素，则将编辑器添加到该元素中
      if (place.appendChild) { place.appendChild(d.wrapper); }
      // 否则，将编辑器添加到指定的位置
      else { place(d.wrapper); }
    }

    // 当前渲染的范围（可能比视图窗口更大）
    d.viewFrom = d.viewTo = doc.first;
    d.reportedViewFrom = d.reportedViewTo = doc.first;
    // 关于渲染行的信息
    d.view = [];
    d.renderedView = null;
    // 在渲染时保存有关单个渲染行的信息
    // 用于测量，当不在视图中时。
    d.externalMeasured = null;
    // 视图上方的空白空间（以像素为单位）
    d.viewOffset = 0;
    d.lastWrapHeight = d.lastWrapWidth = 0;
    d.updateLineNumbers = null;

    d.nativeBarWidth = d.barHeight = d.barWidth = 0;
    d.scrollbarsClipped = false;

    // 用于仅在必要时调整行号边栏的宽度（当行数跨越边界时，其宽度会改变）
    d.lineNumWidth = d.lineNumInnerWidth = d.lineNumChars = null;
    // 当添加非水平滚动的行部件时设置为true。作为优化，当此值为false时，跳过行部件的对齐。
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

// 由于鼠标滚轮事件报告的增量值在不同浏览器甚至浏览器版本之间是不标准的，
// 通常是难以预测的，因此这段代码首先测量前几个鼠标滚轮事件的滚动效果，
// 并从中检测出如何将增量转换为像素偏移量。
//
// 我们想知道滚轮事件将滚动的量的原因是，它给了我们一个机会在实际滚动发生之前更新显示，
// 从而减少闪烁。

var wheelSamples = 0, wheelPixelsPerUnit = null;
// 在我们知道的浏览器中填入一个浏览器检测到的起始值。这些值不必准确，
// 如果它们错误，结果只会是第一次滚轮滚动时轻微的闪烁（如果足够大）。
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

  // Webkit 浏览器在 OS X 上中止动量滚动时，目标
    // 如果发生垂直滚动并且是在 Mac 平台上使用 Webkit 浏览器
    // 则执行以下操作
    if (dy && mac && webkit) {
      // 遍历当前元素及其父元素，直到滚动元素
      outer: for (var cur = e.target, view = display.view; cur != scroll; cur = cur.parentNode) {
        // 遍历视图数组，找到当前元素对应的视图节点
        for (var i = 0; i < view.length; i++) {
          if (view[i].node == cur) {
            // 设置当前滚动目标为当前元素
            cm.display.currentWheelTarget = cur;
            break outer
          }
        }
      }
    }

    // 在某些浏览器上，水平滚动会导致重新绘制发生在 gutter 重新对齐之前
    // 为了避免 gutter 在滚动时出现不正常的抖动，我们在这里处理水平滚动
    if (dx && !gecko && !presto && wheelPixelsPerUnit != null) {
      if (dy && canScrollY)
        { updateScrollTop(cm, Math.max(0, scroll.scrollTop + dy * wheelPixelsPerUnit)); }
      setScrollLeft(cm, Math.max(0, scroll.scrollLeft + dx * wheelPixelsPerUnit));
      // 只有在垂直滚动实际上是可能的情况下才阻止默认滚动
      // 否则，当 deltaX 很小而 deltaY 很大时，会在 OSX 触控板上引起垂直滚动抖动
      if (!dy || (dy && canScrollY))
        { e_preventDefault(e); }
      display.wheelStartX = null; // 中止测量，如果正在进行中
      return
    }

    // 如果发生垂直滚动并且已知每单位滚动的像素值
    // 则执行以下操作
    if (dy && wheelPixelsPerUnit != null) {
      var pixels = dy * wheelPixelsPerUnit;
      var top = cm.doc.scrollTop, bot = top + display.wrapper.clientHeight;
      if (pixels < 0) { top = Math.max(0, top + pixels - 50); }
      else { bot = Math.min(cm.doc.height, bot + pixels + 50); }
      // 更新显示视图，使滚动区域可见
      updateDisplaySimple(cm, {top: top, bottom: bot});
    }
    // 如果滚轮样本小于20
    if (wheelSamples < 20) {
      // 如果显示对象的滚轮起始X坐标为空
      if (display.wheelStartX == null) {
        // 设置滚轮起始X和Y坐标为滚动条的左偏移和上偏移
        display.wheelStartX = scroll.scrollLeft; display.wheelStartY = scroll.scrollTop;
        // 设置滚轮的水平和垂直位移
        display.wheelDX = dx; display.wheelDY = dy;
        // 设置一个定时器，延迟200毫秒执行
        setTimeout(function () {
          // 如果滚轮起始X坐标为空，则返回
          if (display.wheelStartX == null) { return }
          // 计算滚动条的水平和垂直位移
          var movedX = scroll.scrollLeft - display.wheelStartX;
          var movedY = scroll.scrollTop - display.wheelStartY;
          // 计算滚轮的样本值
          var sample = (movedY && display.wheelDY && movedY / display.wheelDY) ||
            (movedX && display.wheelDX && movedX / display.wheelDX);
          // 重置滚轮起始X和Y坐标为空
          display.wheelStartX = display.wheelStartY = null;
          // 如果样本值不存在，则返回
          if (!sample) { return }
          // 更新滚轮每单位像素的值
          wheelPixelsPerUnit = (wheelPixelsPerUnit * wheelSamples + sample) / (wheelSamples + 1);
          // 增加滚轮样本数量
          ++wheelSamples;
        }, 200);
      } else {
        // 更新滚轮的水平和垂直位移
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
    // 遍历范围数组，比较其中的anchor和head是否相等
    for (var i = 0; i < this.ranges.length; i++) {
      var here = this.ranges[i], there = other.ranges[i];
      if (!equalCursorPos(here.anchor, there.anchor) || !equalCursorPos(here.head, there.head)) { return false }
    }
    return true
  };

  // 深拷贝选择对象
  Selection.prototype.deepCopy = function () {
    var out = [];
  // 遍历this.ranges数组，复制每个范围的锚点和头部，存入out数组中
  for (var i = 0; i < this.ranges.length; i++)
    { out[i] = new Range(copyPos(this.ranges[i].anchor), copyPos(this.ranges[i].head)); }
  // 返回一个新的Selection对象，传入out数组和this.primIndex
  return new Selection(out, this.primIndex)
};

// 检查是否有选中内容
Selection.prototype.somethingSelected = function () {
  // 遍历this.ranges数组，如果有范围不为空，则返回true，否则返回false
  for (var i = 0; i < this.ranges.length; i++)
    { if (!this.ranges[i].empty()) { return true } }
  return false
};

// 检查是否包含指定的位置范围
Selection.prototype.contains = function (pos, end) {
  // 如果end不存在，则将end设置为pos
  if (!end) { end = pos; }
  // 遍历this.ranges数组，比较pos和end与范围的起始和结束位置，返回匹配的范围索引，否则返回-1
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

// 为Range对象添加from方法，返回锚点和头部位置中较小的位置
Range.prototype.from = function () { return minPos(this.anchor, this.head) };
// 为Range对象添加to方法，返回锚点和头部位置中较大的位置
Range.prototype.to = function () { return maxPos(this.anchor, this.head) };
// 为Range对象添加empty方法，判断范围是否为空
Range.prototype.empty = function () { return this.head.line == this.anchor.line && this.head.ch == this.anchor.ch };

// 标准化选择范围，处理重叠的范围
function normalizeSelection(cm, ranges, primIndex) {
  // 获取cm.options.selectionsMayTouch属性
  var mayTouch = cm && cm.options.selectionsMayTouch;
  // 获取ranges数组中指定primIndex的范围
  var prim = ranges[primIndex];
  // 对ranges数组进行排序，根据范围的起始位置进行比较
  ranges.sort(function (a, b) { return cmp(a.from(), b.from()); });
  // 获取排序后primIndex在ranges中的索引
  primIndex = indexOf(ranges, prim);
  // 遍历ranges数组，处理重叠的范围
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

// 调整一个位置，使其指向相同文本的变化后位置，或者变化覆盖的结束位置
function adjustForChange(pos, change) {
  if (cmp(pos, change.from) < 0) { return pos }
  if (cmp(pos, change.to) <= 0) { return changeEnd(change) }

  var line = pos.line + change.text.length - (change.to.line - change.from.line) - 1, ch = pos.ch;
  if (pos.line == change.to.line) { ch += changeEnd(change).ch - change.to.ch; }
  return Pos(line, ch)
}

// 根据文档和变化计算变化后的选择
function computeSelAfterChange(doc, change) {
  var out = [];
  for (var i = 0; i < doc.sel.ranges.length; i++) {
    var range = doc.sel.ranges[i];
    out.push(new Range(adjustForChange(range.anchor, change),
                       adjustForChange(range.head, change)));
  }
  return normalizeSelection(doc.cm, out, doc.sel.primIndex)
}

// 根据旧的位置和新的位置，计算偏移后的位置
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
      # 计算起始位置 from 和结束位置 to
      var from = offsetPos(change.from, oldPrev, newPrev);
      var to = offsetPos(changeEnd(change), oldPrev, newPrev);
      # 更新 oldPrev 和 newPrev
      oldPrev = change.to;
      newPrev = to;
      # 根据 hint 的值进行不同的处理
      if (hint == "around") {
        # 获取 range 对象
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

  # 默认情况下，处理从行首开始到行尾结束的更新
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
      # 发出 "change" 信号
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
  // 在文档发生改变时，延迟触发 change 事件
  signalLater(doc, "change", doc, change);
}

// 对所有链接的文档调用 f 函数
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

// 方向改变时的处理
function directionChanged(cm) {
  runInOp(cm, function () {
    setDirectionClass(cm);
    regChange(cm);
  });
}

// 历史记录对象
function History(startGen) {
  // 改变事件和选择的数组。执行操作会将事件添加到 done 并清除 undone。撤销操作将事件从 done 移动到 undone，重做操作则相反。
  this.done = []; this.undone = [];
  this.undoDepth = Infinity;
  // 用于跟踪何时可以将更改合并为单个撤销事件
  this.lastModTime = this.lastSelTime = 0;
  this.lastOp = this.lastSelOp = null;
  this.lastOrigin = this.lastSelOrigin = null;
  // 由 isClean() 方法使用
  this.generation = this.maxGeneration = startGen || 1;
}

// 从 updateDoc 风格的更改对象创建历史更改事件
function historyChangeFromChange(doc, change) {
  // 创建一个包含历史变化信息的对象，包括变化的起始位置、结束位置和文本内容
  var histChange = {from: copyPos(change.from), to: changeEnd(change), text: getBetween(doc, change.from, change.to)};
  // 在文档中附加本地跨度
  attachLocalSpans(doc, histChange, change.from.line, change.to.line + 1);
  // 在关联的文档中附加本地跨度
  linkedDocs(doc, function (doc) { return attachLocalSpans(doc, histChange, change.from.line, change.to.line + 1); }, true);
  // 返回历史变化对象
  return histChange
}

// 从历史数组中弹出所有选择事件。在遇到变化事件之前停止。
function clearSelectionEvents(array) {
  while (array.length) {
    var last = lst(array);
    // 如果最后一个元素包含范围信息，则弹出
    if (last.ranges) { array.pop(); }
    // 否则跳出循环
    else { break }
  }
}

// 查找历史中的顶级变化事件。弹出在其路径上的选择事件。
function lastChangeEvent(hist, force) {
  if (force) {
    // 清除历史中的选择事件
    clearSelectionEvents(hist.done);
    // 返回历史中的最后一个变化事件
    return lst(hist.done)
  } else if (hist.done.length && !lst(hist.done).ranges) {
    // 如果历史中存在变化事件且最后一个元素不包含范围信息，则返回最后一个变化事件
    return lst(hist.done)
  } else if (hist.done.length > 1 && !hist.done[hist.done.length - 2].ranges) {
    // 如果历史中存在多个元素且倒数第二个元素不包含范围信息，则弹出最后一个元素并返回
    hist.done.pop();
    return lst(hist.done)
  }
}

// 将变化注册到历史中。将在单个操作内或接近的变化与允许合并的起源（以“+”开头）合并为单个事件。
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
         ((change.origin.charAt(0) == "+" && hist.lastModTime > time - (doc.cm ? doc.cm.options.historyEventDelay : 500)) ||  # 且当前操作是插入操作且上一个操作距离当前时间小于延迟时间
          change.origin.charAt(0) == "*")) &&  # 或者当前操作是特殊操作
        (cur = lastChangeEvent(hist, hist.lastOp == opId))) {  # 并且当前事件是上一个事件的最后一个变化
      # 将当前变化合并到上一个事件中
      last = lst(cur.changes);
      if (cmp(change.from, change.to) == 0 && cmp(change.from, last.to) == 0) {  # 如果当前变化是简单的插入操作
        # 优化简单插入的情况，不需要为每个字符输入添加新的变化集
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
      while (hist.done.length > hist.undoDepth) {  # 如果历史记录长度超过了撤销深度
        hist.done.shift();  # 移除最早的事件
        if (!hist.done[0].ranges) { hist.done.shift(); }  # 如果下一个事件没有范围，则再移除一个
      }
    }
    hist.done.push(selAfter);  # 将当前选择推入历史记录
    hist.generation = ++hist.maxGeneration;  # 更新历史记录的最大代数
    hist.lastModTime = hist.lastSelTime = time;  # 更新最后修改时间和最后选择时间
    hist.lastOp = hist.lastSelOp = opId;  # 更新最后操作和最后选择的操作
    hist.lastOrigin = hist.lastSelOrigin = change.origin;  # 更新最后操作的来源和最后选择的来源

    if (!last) { signal(doc, "historyAdded"); }  # 如果没有上一个事件，触发历史添加事件
  }

  function selectionEventCanBeMerged(doc, origin, prev, sel) {
    var ch = origin.charAt(0);
    // 如果字符是 * 或者 +，并且前一个选择范围的长度等于当前选择范围的长度，并且前一个选择范围是否有选中内容与当前选择范围是否有选中内容相同，并且当前时间与上一次选择的时间间隔小于等于500毫秒（如果有 CodeMirror 实例，则使用其选项中的 historyEventDelay，否则默认为500）
    return ch == "*" ||
      ch == "+" &&
      prev.ranges.length == sel.ranges.length &&
      prev.somethingSelected() == sel.somethingSelected() &&
      new Date - doc.history.lastSelTime <= (doc.cm ? doc.cm.options.historyEventDelay : 500)
  }

  // 每当选择发生变化时调用，将新的选择设置为历史记录中的待处理选择，并在选择范围数量、是否为空或时间上有显著差异时，将旧的待处理选择推入“已完成”数组中。
  function addSelectionToHistory(doc, sel, opId, options) {
    var hist = doc.history, origin = options && options.origin;

    // 当前操作与上一个选择操作相同时，或者当前操作的来源与上一个选择操作的来源相同时，并且上一个选择操作的修改时间与上一个选择操作的时间相同时，或者选择事件可以合并时，将最后一个已完成的选择替换为当前选择
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

  // 用于在历史记录中存储标记的跨度信息
  function attachLocalSpans(doc, change, from, to) {
    var existing = change["spans_" + doc.id], n = 0;
    doc.iter(Math.max(doc.first, from), Math.min(doc.first + doc.size, to), function (line) {
      if (line.markedSpans)
        { (existing || (existing = change["spans_" + doc.id] = {}))[n] = line.markedSpans; }
      ++n;
  // 当撤销/重做操作时，恢复包含标记范围的文本，那些已经明确清除的标记范围不应该被恢复。
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

  // 用于撤销/重做历史中的更改。将现有标记范围的计算结果与历史中存在的标记范围集合合并（以便在标记范围周围进行删除然后撤消操作时将标记范围恢复）。
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
        // 继续下一次循环
        continue
      }
      // 获取事件的改变和新改变数组
      var changes = event.changes, newChanges = [];
      // 将包含新改变数组的对象添加到副本数组中
      copy.push({changes: newChanges});
      // 遍历改变数组
      for (var j = 0; j < changes.length; ++j) {
        // 获取当前改变
        var change = changes[j], m = (void 0);
        // 将改变的起始位置、结束位置和文本添加到新改变数组中
        newChanges.push({from: change.from, to: change.to, text: change.text});
        // 如果有新组，则处理改变对象中的属性
        if (newGroup) { for (var prop in change) { if (m = prop.match(/^spans_(\d+)$/)) {
          // 如果新组中包含属性对应的数字，则将属性值添加到新改变数组中，并删除原改变对象中的属性
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

  // 给定的'scroll'参数表示在修改选择后是否应将新的光标位置滚动到视图中

  // 如果按住shift键或设置了extend标志，则扩展范围以包括给定位置（和可选的第二个位置）。
  // 否则，只返回给定位置之间的范围。
  // 用于光标移动等。
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
          // 交换头部位置和锚点位置
          anchor = head;
          head = other;
        } else if (posBefore != (cmp(head, other) < 0)) {
          // 如果头部位置和锚点位置的大小不同于头部位置和第二个位置的大小，则更新头部位置
          head = other;
        }
      }
      // 返回新的范围对象
      return new Range(anchor, head)
    } else {
      // 返回给定位置之间的范围
      return new Range(other || head, head)
    }
  }

  // 扩展主要选择范围，丢弃其余部分。
  function extendSelection(doc, head, other, options, extend) {
    // 如果extend为null，则根据文档的cm属性和shift或extend属性来确定extend的值
    if (extend == null) { extend = doc.cm && (doc.cm.display.shift || doc.extend); }
  // 设置新的选择，包括扩展选择和选项
  function setSelection(doc, newSel, options) {
    // 设置选择
    doc.setSelection(newSel, options);
  }

  // 扩展所有选择
  function extendSelections(doc, heads, options) {
    // 创建一个空数组
    var out = [];
    // 获取扩展状态
    var extend = doc.cm && (doc.cm.display.shift || doc.extend);
    // 遍历选择范围
    for (var i = 0; i < doc.sel.ranges.length; i++) {
      // 扩展选择范围
      out[i] = extendRange(doc.sel.ranges[i], heads[i], null, extend);
    }
    // 标准化新的选择
    var newSel = normalizeSelection(doc.cm, out, doc.sel.primIndex);
    // 设置新的选择
    setSelection(doc, newSel, options);
  }

  // 替换选择中的单个范围
  function replaceOneSelection(doc, i, range, options) {
    // 复制选择范围数组
    var ranges = doc.sel.ranges.slice(0);
    // 替换指定位置的选择范围
    ranges[i] = range;
    // 设置新的选择
    setSelection(doc, normalizeSelection(doc.cm, ranges, doc.sel.primIndex), options);
  }

  // 重置选择为单个范围
  function setSimpleSelection(doc, anchor, head, options) {
    // 设置新的选择
    setSelection(doc, simpleSelection(anchor, head), options);
  }

  // 在选择更新之前，允许 beforeSelectionChange 处理程序影响选择更新
  function filterSelectionChange(doc, sel, options) {
    // 创建一个对象
    var obj = {
      ranges: sel.ranges,
      update: function(ranges) {
        this.ranges = [];
        // 更新选择范围
        for (var i = 0; i < ranges.length; i++) {
          this.ranges[i] = new Range(clipPos(doc, ranges[i].anchor), clipPos(doc, ranges[i].head));
        }
      },
      origin: options && options.origin
    };
    // 触发 beforeSelectionChange 事件
    signal(doc, "beforeSelectionChange", doc, obj);
    // 如果存在 CodeMirror 实例，则触发其 beforeSelectionChange 事件
    if (doc.cm) {
      signal(doc.cm, "beforeSelectionChange", doc.cm, obj);
    }
    // 返回标准化后的选择
    if (obj.ranges != sel.ranges) {
      return normalizeSelection(doc.cm, obj.ranges, obj.ranges.length - 1);
    } else {
      return sel;
    }
  }

  // 设置选择并替换历史记录
  function setSelectionReplaceHistory(doc, sel, options) {
    // 获取历史记录
    var done = doc.history.done, last = lst(done);
    // 如果存在历史记录并且包含选择范围，则替换最后一个历史记录
    if (last && last.ranges) {
      done[done.length - 1] = sel;
      // 设置新的选择，不包含撤销操作
      setSelectionNoUndo(doc, sel, options);
    }
  }
  // 如果条件为真，则执行第一个代码块，否则执行第二个代码块
  } else {
    // 调用setSelection函数，设置新的选择
    setSelection(doc, sel, options);
  }

  // 设置新的选择
  function setSelection(doc, sel, options) {
    // 调用setSelectionNoUndo函数，设置新的选择，不记录撤销操作
    setSelectionNoUndo(doc, sel, options);
    // 将选择添加到历史记录中
    addSelectionToHistory(doc, doc.sel, doc.cm ? doc.cm.curOp.id : NaN, options);
  }

  // 设置新的选择，不记录撤销操作
  function setSelectionNoUndo(doc, sel, options) {
    // 如果存在"beforeSelectionChange"事件处理程序或者doc.cm存在并且存在"beforeSelectionChange"事件处理程序
    if (hasHandler(doc, "beforeSelectionChange") || doc.cm && hasHandler(doc.cm, "beforeSelectionChange"))
      // 调用filterSelectionChange函数，过滤选择改变
      { sel = filterSelectionChange(doc, sel, options); }

    // 设置偏移量为options.bias，如果不存在则根据选择的位置设置偏移量
    var bias = options && options.bias ||
      (cmp(sel.primary().head, doc.sel.primary().head) < 0 ? -1 : 1);
    // 调用skipAtomicInSelection函数，跳过选择中的原子标记范围
    setSelectionInner(doc, skipAtomicInSelection(doc, sel, bias, true));

    // 如果不存在options.scroll或者doc.cm存在，则确保光标可见
    if (!(options && options.scroll === false) && doc.cm)
      { ensureCursorVisible(doc.cm); }
  }

  // 设置内部选择
  function setSelectionInner(doc, sel) {
    // 如果选择与文档中的选择相同，则返回
    if (sel.equals(doc.sel)) { return }

    // 设置文档的选择为新的选择
    doc.sel = sel;

    // 如果doc.cm存在
    if (doc.cm) {
      // 更新输入
      doc.cm.curOp.updateInput = 1;
      // 选择改变
      doc.cm.curOp.selectionChanged = true;
      // 发送光标活动信号
      signalCursorActivity(doc.cm);
    }
    // 延迟发送"cursorActivity"信号
    signalLater(doc, "cursorActivity", doc);
  }

  // 验证选择是否没有部分选择任何原子标记范围
  function reCheckSelection(doc) {
    // 调用setSelectionInner函数，跳过选择中的原子标记范围
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
  // 如果out为真，则调用normalizeSelection函数，传入doc.cm、out、sel.primIndex参数，否则返回sel
  return out ? normalizeSelection(doc.cm, out, sel.primIndex) : sel
  }

  // 跳过原子范围内的位置
  function skipAtomicInner(doc, pos, oldPos, dir, mayClear) {
    // 获取指定行的内容
    var line = getLine(doc, pos.line);
    // 如果该行有标记范围，则遍历标记范围
    if (line.markedSpans) { for (var i = 0; i < line.markedSpans.length; ++i) {
      var sp = line.markedSpans[i], m = sp.marker;

      // 确定是否应防止将光标放置在原子标记的左侧/右侧
      // 历史上，这是使用inclusiveLeft/Right选项来确定的，但现在控制它的新方法是使用selectLeft/Right
      var preventCursorLeft = ("selectLeft" in m) ? !m.selectLeft : m.inclusiveLeft;
      var preventCursorRight = ("selectRight" in m) ? !m.selectRight : m.inclusiveRight;

      // 如果光标位置在标记范围内，则执行以下操作
      if ((sp.from == null || (preventCursorLeft ? sp.from <= pos.ch : sp.from < pos.ch)) &&
          (sp.to == null || (preventCursorRight ? sp.to >= pos.ch : sp.to > pos.ch))) {
        // 如果mayClear为真，则触发m的beforeCursorEnter信号，并检查是否已明确清除
        if (mayClear) {
          signal(m, "beforeCursorEnter");
          if (m.explicitlyCleared) {
            // 如果line.markedSpans不存在，则跳出循环，否则继续下一次循环
            if (!line.markedSpans) { break }
            else {--i; continue}
          }
        }
        // 如果不是原子标记，则继续下一次循环
        if (!m.atomic) { continue }

        // 如果oldPos存在，则查找与dir方向相邻的位置
        if (oldPos) {
          var near = m.find(dir < 0 ? 1 : -1), diff = (void 0);
          if (dir < 0 ? preventCursorRight : preventCursorLeft)
            { near = movePos(doc, near, -dir, near && near.line == pos.line ? line : null); }
          if (near && near.line == pos.line && (diff = cmp(near, oldPos)) && (dir < 0 ? diff < 0 : diff > 0))
            { return skipAtomicInner(doc, near, pos, dir, mayClear) }
        }

        // 查找与dir方向相邻的位置
        var far = m.find(dir < 0 ? -1 : 1);
        if (dir < 0 ? preventCursorLeft : preventCursorRight)
          { far = movePos(doc, far, dir, far.line == pos.line ? line : null); }
        // 如果far存在，则递归调用skipAtomicInner函数，传入far、pos、dir、mayClear参数，否则返回null
        return far ? skipAtomicInner(doc, far, pos, dir, mayClear) : null
      }
    } }
    // 返回pos
    return pos
  }

  // 确保给定位置不在原子范围内
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
    # 触发“beforeChange”事件，并传递相关参数
    signal(doc, "beforeChange", doc, obj);
    # 如果文档关联了 CodeMirror 编辑器，则也触发相应事件
    if (doc.cm) { signal(doc.cm, "beforeChange", doc.cm, obj); }

    # 如果取消了更改，则将编辑器的输入更新标记为 2
    if (obj.canceled) {
      if (doc.cm) { doc.cm.curOp.updateInput = 2; }
      return null
    }
  // 返回一个包含指定属性的对象
  return {from: obj.from, to: obj.to, text: obj.text, origin: obj.origin}
  }

  // 对文档应用更改，并将其添加到文档的历史记录中，并传播到所有链接的文档
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
    // 如果遍历完所有事件都没有符合条件的事件，则直接返回
    if (i == source.length) { return }
    // 重置历史记录的最后来源和最后选择来源
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
    // 定义循环函数，参数为索引 i
    var loop = function ( i ) {
      // 获取第 i 个变化对象
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

      // 计算变化后的选择位置
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
    // 调整选择位置的行号
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
    // 如果存在代码镜像并且当前操作不是代码操作，则返回
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
    # 设置选择内容，不记录操作
    setSelectionNoUndo(doc, selAfter, sel_dontScroll);

    # 如果文档不能编辑并且跳过原子操作，则重置文档的编辑状态
    if (doc.cantEdit && skipAtomic(doc, Pos(doc.firstLine(), 0)))
      { doc.cantEdit = false; }
  }

  // 处理文档变化与编辑器的交互
  function makeChangeSingleDocInEditor(cm, change, spans) {
    var doc = cm.doc, display = cm.display, from = change.from, to = change.to;

    var recomputeMaxLength = false, checkWidthStart = from.line;
    # 如果不换行，则检查宽度
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

    # 更新文档内容和显示
    updateDoc(doc, change, spans, estimateHeight(cm));
    # 如果编辑器选项中没有启用行包裹
    if (!cm.options.lineWrapping) {
      # 从指定行开始迭代文档，检查每行的长度
      doc.iter(checkWidthStart, from.line + change.text.length, function (line) {
        # 获取当前行的长度
        var len = lineLength(line);
        # 如果当前行长度超过了最大行长度
        if (len > display.maxLineLength) {
          # 更新最大行信息
          display.maxLine = line;
          display.maxLineLength = len;
          display.maxLineChanged = true;
          recomputeMaxLength = false;
        }
      });
      # 如果需要重新计算最大行长度
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
      # 如果有变化处理程序，则延迟发送变化信号
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

    # 如果没有指定 to，则将其设为 from
    if (!to) { to = from; }
    # 如果 to 在 from 之前，则交换它们的位置
    if (cmp(to, from) < 0) { (assign = [to, from], from = assign[0], to = assign[1]); }
    # 如果 code 是字符串，则将其拆分为行数组
    if (typeof code == "string") { code = doc.splitLines(code); }
    # 进行文本替换
    makeChange(doc, {from: from, to: to, text: code, origin: origin});
  }

  # 重新基于/重置历史记录以处理外部变化
  function rebaseHistSelSingle(pos, from, to, diff) {
    # 如果 to 在 pos 行之前，则将 pos 行向后移动 diff 行
    if (to < pos.line) {
      pos.line += diff;
    } else if (from < pos.line) {
      # 如果 from 在 pos 行之前，则将 pos 行设为 from 行，列设为 0
      pos.line = from;
      pos.ch = 0;
  // 结束当前函数的定义
  }
}

// 尝试重新定位历史事件数组，根据文档的变化。如果变化触及事件的相同行，事件及其后的所有内容将被丢弃。如果变化在事件之前，事件的位置将被更新。使用写时复制方案来处理位置，以避免在每次重新定位时重新分配它们，但也避免共享位置对象被不安全地更新。
function rebaseHistArray(array, from, to, diff) {
  // 遍历历史事件数组
  for (var i = 0; i < array.length; ++i) {
    // 获取当前事件
    var sub = array[i], ok = true;
    // 如果事件包含范围
    if (sub.ranges) {
      // 如果事件未被复制，则进行深拷贝
      if (!sub.copied) { sub = array[i] = sub.deepCopy(); sub.copied = true; }
      // 遍历事件的范围，更新位置
      for (var j = 0; j < sub.ranges.length; j++) {
        rebaseHistSelSingle(sub.ranges[j].anchor, from, to, diff);
        rebaseHistSelSingle(sub.ranges[j].head, from, to, diff);
      }
      continue
    }
    // 如果事件包含变化
    for (var j$1 = 0; j$1 < sub.changes.length; ++j$1) {
      var cur = sub.changes[j$1];
      // 如果变化在事件之后
      if (to < cur.from.line) {
        // 更新位置
        cur.from = Pos(cur.from.line + diff, cur.from.ch);
        cur.to = Pos(cur.to.line + diff, cur.to.ch);
      } else if (from <= cur.to.line) {
        // 标记为不可用
        ok = false;
        break
      }
    }
    // 如果不可用，则删除事件及其之前的所有事件
    if (!ok) {
      array.splice(0, i + 1);
      i = 0;
    }
  }
}

// 重新定位历史记录
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
    // 如果操作函数返回 true 并且文档有变化，则注册文档行变化
    if (op(line, no) && doc.cm) { regLineChange(doc.cm, no, changeType); }
    // 返回行对象
    return line
  }

  // 文档被表示为一个 BTree，由叶子节点和包含行的分支节点组成，每个分支节点最多有十个叶子节点或其他分支节点。顶部节点始终是一个分支节点，也是文档对象本身（意味着它有额外的方法和属性）。
  //
  // 所有节点都有父链接。该树用于从行号到行对象的转换，以及从对象到行号的转换。它还按高度进行索引，并用于在高度和行对象之间进行转换，并找到文档的总高度。
  //
  // 另请参阅 http://marijnhaverbeke.nl/blog/codemirror-line-tree.html

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

    // 在 'at' 处删除 n 行
    removeInner: function(at, n) {
      for (var i = at, e = at + n; i < e; ++i) {
        var line = this.lines[i];
        this.height -= line.height;
        cleanUpLine(line);
        signalLater(line, "delete");
      }
      this.lines.splice(at, n);
    },

    // 用于将一个小分支合并为单个叶子的辅助函数
    collapse: function(lines) {
      lines.push.apply(lines, this.lines);
    },

    // 在 'at' 处插入给定的行数组，将它们视为具有给定高度
    insertInner: function(at, lines, height) {
      this.height += height;
      this.lines = this.lines.slice(0, at).concat(lines).concat(this.lines.slice(at));
      for (var i = 0; i < lines.length; ++i) { lines[i].parent = this; }
    },

    // 用于遍历树的一部分
  // 定义一个名为 iterN 的方法，用于迭代处理指定数量的行
  iterN: function(at, n, op) {
    // 从 at 开始，迭代 n 次
    for (var e = at + n; at < e; ++at)
      // 如果 op 方法返回 true，则立即返回 true
      { if (op(this.lines[at])) { return true } }
  }
};

// 定义一个名为 BranchChunk 的构造函数，用于创建分支节点
function BranchChunk(children) {
  this.children = children;
  var size = 0, height = 0;
  // 遍历子节点，计算总大小和高度，并设置父节点
  for (var i = 0; i < children.length; ++i) {
    var ch = children[i];
    size += ch.chunkSize(); height += ch.height;
    ch.parent = this;
  }
  this.size = size;
  this.height = height;
  this.parent = null;
}

// 设置 BranchChunk 的原型对象
BranchChunk.prototype = {
  // 定义 chunkSize 方法，返回节点的大小
  chunkSize: function() { return this.size },

  // 定义 removeInner 方法，用于移除节点内的内容
  removeInner: function(at, n) {
    this.size -= n;
    // 遍历子节点
    for (var i = 0; i < this.children.length; ++i) {
      var child = this.children[i], sz = child.chunkSize();
      // 如果 at 小于子节点的大小
      if (at < sz) {
        var rm = Math.min(n, sz - at), oldHeight = child.height;
        // 递归调用 removeInner 方法
        child.removeInner(at, rm);
        this.height -= oldHeight - child.height;
        // 如果子节点的大小等于 rm，则从 children 中移除该子节点
        if (sz == rm) { this.children.splice(i--, 1); child.parent = null; }
        // 如果 n 减去 rm 等于 0，则跳出循环
        if ((n -= rm) == 0) { break }
        at = 0;
      } else { at -= sz; }
    }
    // 如果结果小于 25 行，则确保它是单个叶子节点
    if (this.size - n < 25 &&
        (this.children.length > 1 || !(this.children[0] instanceof LeafChunk))) {
      var lines = [];
      // 调用 collapse 方法，将子节点的内容合并到 lines 中
      this.collapse(lines);
      this.children = [new LeafChunk(lines)];
      this.children[0].parent = this;
    }
  },

  // 定义 collapse 方法，用于将子节点的内容合并到指定的数组中
  collapse: function(lines) {
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
            // 为了避免内存抖动，当子节点的行数很大时（例如大文件的第一个视图），不会进行切片。
            // 取小的切片，按顺序取，因为顺序内存访问最快。
            var remaining = child.lines.length % 25 + 25;  // 计算剩余行数
            for (var pos = remaining; pos < child.lines.length;) {  // 循环取小的切片
              var leaf = new LeafChunk(child.lines.slice(pos, pos += 25));  // 创建新的叶子节点
              child.height -= leaf.height;  // 更新子节点的高度
              this.children.splice(++i, 0, leaf);  // 在当前节点的子节点中插入新的叶子节点
              leaf.parent = this;  // 设置新叶子节点的父节点
            }
            child.lines = child.lines.slice(0, remaining);  // 更新子节点的行数
            this.maybeSpill();  // 可能需要分裂当前节点
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
          me.children = [copy, sibling];  // 更新当前节点的子节点
          me = copy;  // 更新当前节点为复制节点
       } else {
          me.size -= sibling.size;  // 更新当前节点的大小
          me.height -= sibling.height;  // 更新当前节点的高度
          var myIndex = indexOf(me.parent.children, me);  // 获取当前节点在父节点中的索引
          me.parent.children.splice(myIndex + 1, 0, sibling);  // 在父节点中插入新的分支节点
        }
        sibling.parent = me.parent;  // 设置新的分支节点的父节点
      } while (me.children.length > 10)  // 当子节点数量大于10时循环
      me.parent.maybeSpill();  // 可能需要分裂父节点
    },
    // 定义一个名为 iterN 的方法，接受三个参数：at（起始位置）、n（长度）、op（操作）
    iterN: function(at, n, op) {
      // 遍历子元素数组
      for (var i = 0; i < this.children.length; ++i) {
        // 获取当前子元素和其大小
        var child = this.children[i], sz = child.chunkSize();
        // 如果起始位置小于当前子元素大小
        if (at < sz) {
          // 计算实际使用的长度
          var used = Math.min(n, sz - at);
          // 调用子元素的 iterN 方法，如果返回 true 则结束循环
          if (child.iterN(at, used, op)) { return true }
          // 减去已经使用的长度
          if ((n -= used) == 0) { break }
          // 重置起始位置
          at = 0;
        } else { at -= sz; }
      }
    }
  };

  // LineWidget 是显示在行上方或下方的块元素。

  // 定义 LineWidget 构造函数，接受三个参数：doc（文档）、node（节点）、options（选项）
  var LineWidget = function(doc, node, options) {
    // 如果有选项，则遍历选项并赋值给 LineWidget 实例
    if (options) { for (var opt in options) { if (options.hasOwnProperty(opt))
      { this[opt] = options[opt]; } } }
    this.doc = doc;
    this.node = node;
  };

  // 给 LineWidget 原型添加 clear 方法
  LineWidget.prototype.clear = function () {
    // 获取文档、行的小部件、行号
    var cm = this.doc.cm, ws = this.line.widgets, line = this.line, no = lineNo(line);
    // 如果行号为空或者小部件为空，则返回
    if (no == null || !ws) { return }
    // 遍历小部件数组，删除当前小部件
    for (var i = 0; i < ws.length; ++i) { if (ws[i] == this) { ws.splice(i--, 1); } }
    // 如果小部件数组为空，则置空
    if (!ws.length) { line.widgets = null; }
    // 计算小部件高度，更新行高
    var height = widgetHeight(this);
    updateLineHeight(line, Math.max(0, line.height - height));
    // 如果存在 CodeMirror 实例，则执行以下操作
    if (cm) {
      runInOp(cm, function () {
        // 调整滚动条
        adjustScrollWhenAboveVisible(cm, line, -height);
        // 注册行变化事件
        regLineChange(cm, no, "widget");
      });
      // 延迟触发行小部件清除事件
      signalLater(cm, "lineWidgetCleared", cm, this, no);
    }
  };

  // 给 LineWidget 原型添加 changed 方法
  LineWidget.prototype.changed = function () {
      var this$1 = this;

    // 获取旧高度、CodeMirror 实例、行
    var oldH = this.height, cm = this.doc.cm, line = this.line;
    this.height = null;
    // 计算高度差
    var diff = widgetHeight(this) - oldH;
    // 如果差值为 0，则返回
    if (!diff) { return }
    // 如果行不是隐藏的，则更新行高
    if (!lineIsHidden(this.doc, line)) { updateLineHeight(line, line.height + diff); }
    // 如果存在 CodeMirror 实例，则执行以下操作
    if (cm) {
      runInOp(cm, function () {
        // 强制更新
        cm.curOp.forceUpdate = true;
        // 调整滚动条
        adjustScrollWhenAboveVisible(cm, line, diff);
        // 延迟触发行小部件变化事件
        signalLater(cm, "lineWidgetChanged", cm, this$1, lineNo(line));
      });
    }
  };
  // 给 LineWidget 添加事件混合器
  eventMixin(LineWidget);

  // 调整滚动条当小部件在可见范围之上
  function adjustScrollWhenAboveVisible(cm, line, diff) {
    // 如果当前行的高度小于编辑器的滚动条位置，则将滚动条位置调整到当前行
    if (heightAtLine(line) < ((cm.curOp && cm.curOp.scrollTop) || cm.doc.scrollTop))
      { addToScrollTop(cm, diff); }
  }

  // 向文档中添加一个行部件
  function addLineWidget(doc, handle, node, options) {
    // 创建一个行部件对象
    var widget = new LineWidget(doc, node, options);
    var cm = doc.cm;
    // 如果编辑器存在并且部件不需要水平滚动，则设置 alignWidgets 为 true
    if (cm && widget.noHScroll) { cm.display.alignWidgets = true; }
    // 在指定行添加部件
    changeLine(doc, handle, "widget", function (line) {
      var widgets = line.widgets || (line.widgets = []);
      // 如果未指定插入位置，则将部件添加到行的末尾
      if (widget.insertAt == null) { widgets.push(widget); }
      // 否则将部件插入到指定位置
      else { widgets.splice(Math.min(widgets.length - 1, Math.max(0, widget.insertAt)), 0, widget); }
      widget.line = line;
      // 如果编辑器存在并且行不是隐藏的，则更新行高度并调整滚动条位置
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

  // 使用 markText 和 setBookmark 方法创建。TextMarker 是一个可以用于清除或查找文档中标记位置的句柄。
  // 行对象包含数组（markedSpans），其中包含指向这些标记对象的 {from, to, marker} 对象，并指示该行上存在这样的标记。
  // 当标记跨越多行时，多行可能指向同一个标记。当标记延伸到行的起始/结束位置之外时，它们的 from/to 属性将为 null。
  // 标记具有指向它们当前触及的行的链接。

  // 折叠的标记具有唯一的 id，以便能够对它们进行排序，这对于在它们重叠时唯一确定外部标记是必要的（它们可以嵌套，但不能部分重叠）。
  var nextMarkerId = 0;

  // TextMarker 构造函数
  var TextMarker = function(doc, type) {
    this.lines = [];
  // 设置当前对象的类型
  this.type = type;
  // 设置当前对象的文档
  this.doc = doc;
  // 设置当前对象的ID，并递增下一个标记ID
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

  // 如果最小行号和编辑器存在并且标记折叠了，则注册变化事件
  if (min != null && cm && this.collapsed) { regChange(cm, min, max + 1); }
  // 清空标记所在的行
  this.lines.length = 0;
  // 设置标记已经明确清除
  this.explicitlyCleared = true;
  // 如果标记是原子的并且文档不可编辑，则设置文档可编辑并重新检查选择
  if (this.atomic && this.doc.cantEdit) {
    this.doc.cantEdit = false;
    if (cm) { reCheckSelection(cm.doc); }
  }
  // 如果编辑器存在，则延迟发送标记清除事件
  if (cm) { signalLater(cm, "markerCleared", cm, this, min, max); }
  // 如果存在操作，则结束操作
  if (withOp) { endOperation(cm); }
  // 如果存在父元素，则清除父元素
  if (this.parent) { this.parent.clear(); }
  };

  // 在文档中查找标记的位置。默认情况下返回一个 {from, to} 对象。可以传递 side 来获取特定的边 -- 0 (两侧), -1 (左侧), 或 1 (右侧)。当 lineObj 为 true 时，返回的 Pos 对象包含一个行对象，而不是行号（用于防止两次查找相同的行）。
  TextMarker.prototype.find = function (side, lineObj) {
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
  // 将事件混合到文本标记中
  eventMixin(TextMarker);

  // 创建一个标记，将其连接到正确的行，并且
  function markText(doc, from, to, options, type) {
    // 共享标记（跨链接文档）单独处理
    // （markTextShared将再次调用此函数，每个文档一次）。
    if (options && options.shared) { return markTextShared(doc, from, to, options, type) }
    // 确保我们在一个操作中
    if (doc.cm && !doc.cm.curOp) { return operation(doc.cm, markText)(doc, from, to, options, type) }

    // 创建一个文本标记对象，并计算起始和结束位置的差值
    var marker = new TextMarker(doc, type), diff = cmp(from, to);
    // 如果存在选项，则将选项复制到标记对象中
    if (options) { copyObj(options, marker, false); }
    // 如果差值大于0或者等于0并且clearWhenEmpty不为false，则返回标记对象
    if (diff > 0 || diff == 0 && marker.clearWhenEmpty !== false)
      { return marker }
    // 如果标记有替换内容
    if (marker.replacedWith) {
      // 显示为小部件意味着折叠（小部件替换文本）
      marker.collapsed = true;
      // 创建小部件节点
      marker.widgetNode = eltP("span", [marker.replacedWith], "CodeMirror-widget");
      // 如果不处理鼠标事件，则设置小部件节点属性
      if (!options.handleMouseEvents) { marker.widgetNode.setAttribute("cm-ignore-events", "true"); }
      // 如果设置了insertLeft选项，则设置小部件节点属性
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

    # 如果标记需要添加到历史记录中
    if (marker.addToHistory)
      { addChangeToHistory(doc, {from: from, to: to, origin: "markText"}, doc.sel, NaN); }

    # 当前行等于起始行，编辑器对象存在，标记已折叠且编辑器选项中不包含换行，并且当前行是显示的最大行
    var curLine = from.line, cm = doc.cm, updateMaxLine;
    doc.iter(curLine, to.line + 1, function (line) {
      if (cm && marker.collapsed && !cm.options.lineWrapping && visualLine(line) == cm.display.maxLine)
        { updateMaxLine = true; }
      # 如果标记已折叠且当前行不等于起始行，则更新行高
      if (marker.collapsed && curLine != from.line) { updateLineHeight(line, 0); }
      # 添加标记范围
      addMarkedSpan(line, new MarkedSpan(marker,
                                         curLine == from.line ? from.ch : null,
                                         curLine == to.line ? to.ch : null));
      ++curLine;
    });
    # lineIsHidden 依赖于标记范围的存在，因此需要第二次遍历
    # 如果标记已折叠
    if (marker.collapsed) { doc.iter(from.line, to.line + 1, function (line) {
      # 如果行被隐藏，则更新行高
      if (lineIsHidden(doc, line)) { updateLineHeight(line, 0); }
    }); }

    # 如果标记在输入时清除
    if (marker.clearOnEnter) { on(marker, "beforeCursorEnter", function () { return marker.clear(); }); }

    # 如果标记是只读的
    if (marker.readOnly) {
      # 查看只读范围
      seeReadOnlySpans();
      # 如果历史记录中存在已完成的操作或者存在已撤销的操作，则清除历史记录
      if (doc.history.done.length || doc.history.undone.length)
        { doc.clearHistory(); }
    }
    # 如果标记已折叠
    if (marker.collapsed) {
      # 分配标记 ID，并设置为原子操作
      marker.id = ++nextMarkerId;
      marker.atomic = true;
    }
    if (cm) {
      // 如果存在 CodeMirror 编辑器对象
      // 同步编辑器状态
      if (updateMaxLine) { cm.curOp.updateMaxLine = true; }  // 如果需要更新最大行数，则设置编辑器操作对象的更新最大行数属性为 true
      if (marker.collapsed)
        { regChange(cm, from.line, to.line + 1); }  // 如果标记已折叠，则注册编辑器内容改变事件
      else if (marker.className || marker.startStyle || marker.endStyle || marker.css ||
               marker.attributes || marker.title)
        { for (var i = from.line; i <= to.line; i++) { regLineChange(cm, i, "text"); } }  // 如果标记具有类名、起始样式、结束样式、CSS、属性或标题，则注册编辑器行内容改变事件
      if (marker.atomic) { reCheckSelection(cm.doc); }  // 如果标记是原子标记，则重新检查编辑器文档的选择状态
      signalLater(cm, "markerAdded", cm, marker);  // 发送标记添加事件
    }
    return marker  // 返回标记对象
  }

  // SHARED TEXTMARKERS

  // 共享文本标记
  // 一个共享标记跨越多个链接的文档。它被实现为控制多个普通标记的元标记对象。
  var SharedTextMarker = function(markers, primary) {
    this.markers = markers;  // 设置共享标记对象的标记数组
    this.primary = primary;  // 设置共享标记对象的主要标记
    for (var i = 0; i < markers.length; ++i)
      { markers[i].parent = this; }  // 遍历标记数组，设置每个标记的父级为当前共享标记对象
  };

  SharedTextMarker.prototype.clear = function () {
    if (this.explicitlyCleared) { return }  // 如果已经明确清除，则返回
    this.explicitlyCleared = true;  // 设置为已明确清除
    for (var i = 0; i < this.markers.length; ++i)
      { this.markers[i].clear(); }  // 遍历标记数组，清除每个标记
    signalLater(this, "clear");  // 发送清除事件
  };

  SharedTextMarker.prototype.find = function (side, lineObj) {
    return this.primary.find(side, lineObj)  // 返回主要标记的查找结果
  };
  eventMixin(SharedTextMarker);  // 将事件混合到共享文本标记对象中

  function markTextShared(doc, from, to, options, type) {
    options = copyObj(options);  // 复制选项对象
    options.shared = false;  // 设置选项对象的共享属性为 false
    var markers = [markText(doc, from, to, options, type)], primary = markers[0];  // 创建标记数组，设置主要标记
    var widget = options.widgetNode;  // 获取选项对象的小部件节点
    linkedDocs(doc, function (doc) {
      if (widget) { options.widgetNode = widget.cloneNode(true); }  // 如果存在小部件节点，则克隆小部件节点
      markers.push(markText(doc, clipPos(doc, from), clipPos(doc, to), options, type));  // 将标记添加到标记数组中
      for (var i = 0; i < doc.linked.length; ++i)
        { if (doc.linked[i].isParent) { return } }  // 遍历链接的文档数组，如果是父级文档，则返回
      primary = lst(markers);  // 设置主要标记为标记数组的最后一个标记
    });
    return new SharedTextMarker(markers, primary)  // 返回新的共享文本标记对象
  }

  function findSharedMarkers(doc) {
  # 返回文档中指定位置范围内的所有标记
  return doc.findMarks(Pos(doc.first, 0), doc.clipPos(Pos(doc.lastLine())), function (m) { return m.parent; })
}

# 复制共享标记到文档中
function copySharedMarkers(doc, markers) {
  for (var i = 0; i < markers.length; i++) {
    var marker = markers[i], pos = marker.find();
    var mFrom = doc.clipPos(pos.from), mTo = doc.clipPos(pos.to);
    if (cmp(mFrom, mTo)) {
      var subMark = markText(doc, mFrom, mTo, marker.primary, marker.primary.type);
      marker.markers.push(subMark);
      subMark.parent = marker;
    }
  }
}

# 分离共享标记
function detachSharedMarkers(markers) {
  var loop = function ( i ) {
    var marker = markers[i], linked = [marker.primary.doc];
    linkedDocs(marker.primary.doc, function (d) { return linked.push(d); });
    for (var j = 0; j < marker.markers.length; j++) {
      var subMarker = marker.markers[j];
      if (indexOf(linked, subMarker.doc) == -1) {
        subMarker.parent = null;
        marker.markers.splice(j--, 1);
      }
    }
  };

  for (var i = 0; i < markers.length; i++) loop( i );
}

# 初始化文档对象
var nextDocId = 0;
var Doc = function(text, mode, firstLine, lineSep, direction) {
  if (!(this instanceof Doc)) { return new Doc(text, mode, firstLine, lineSep, direction) }
  if (firstLine == null) { firstLine = 0; }

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

  if (typeof text == "string") { text = this.splitLines(text); }
  updateDoc(this, {from: start, to: start, text: text});
}
    // 设置选择区域，不滚动
    setSelection(this, simpleSelection(start), sel_dontScroll);
  };

  // 将 Doc 对象的原型设置为 BranchChunk 对象
  Doc.prototype = createObj(BranchChunk.prototype, {
    constructor: Doc,
    // 遍历文档。支持两种形式 -- 只有一个参数时，对文档中的每一行调用该函数。有三个参数时，遍历给定范围（第二个参数不包括在内）。
    iter: function(from, to, op) {
      if (op) { this.iterN(from - this.first, to - from, op); }
      else { this.iterN(this.first, this.first + this.size, from); }
    },

    // 非公共接口，用于添加和删除行
    insert: function(at, lines) {
      var height = 0;
      for (var i = 0; i < lines.length; ++i) { height += lines[i].height; }
      this.insertInner(at - this.first, lines, height);
    },
    remove: function(at, n) { this.removeInner(at - this.first, n); },

    // 从这里开始，这些方法是公共接口的一部分。大多数也可以从 CodeMirror (editor) 实例中使用。

    // 获取文档的值
    getValue: function(lineSep) {
      var lines = getLines(this, this.first, this.first + this.size);
      if (lineSep === false) { return lines }
      return lines.join(lineSep || this.lineSeparator())
    },
    // 设置文档的值
    setValue: docMethodOp(function(code) {
      var top = Pos(this.first, 0), last = this.first + this.size - 1;
      makeChange(this, {from: top, to: Pos(last, getLine(this, last).text.length),
                        text: this.splitLines(code), origin: "setValue", full: true}, true);
      if (this.cm) { scrollToCoords(this.cm, 0, 0); }
      setSelection(this, simpleSelection(top), sel_dontScroll);
    }),
    // 替换指定范围的文本
    replaceRange: function(code, from, to, origin) {
      from = clipPos(this, from);
      to = to ? clipPos(this, to) : from;
      replaceRange(this, code, from, to, origin);
    },
    # 获取指定范围内的文本行
    getRange: function(from, to, lineSep) {
      # 获取指定范围内的文本行
      var lines = getBetween(this, clipPos(this, from), clipPos(this, to));
      # 如果不需要行分隔符，直接返回文本行数组
      if (lineSep === false) { return lines }
      # 否则使用指定的行分隔符连接文本行数组并返回
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
      # 如果行号是数字，获取对应行的句柄
      if (typeof line == "number") { line = getLine(this, line); }
      # 返回行的可视起始位置
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
      # 获取主选区的范围
      var range = this.sel.primary(), pos;
      # 根据参数确定返回的光标位置
      if (start == null || start == "head") { pos = range.head; }
      else if (start == "anchor") { pos = range.anchor; }
      else if (start == "end" || start == "to" || start === false) { pos = range.to(); }
      else { pos = range.from(); }
      return pos
    },
    
    # 返回所有选区的范围
    listSelections: function() { return this.sel.ranges },
    
    # 检查是否有选中内容
    somethingSelected: function() {return this.sel.somethingSelected()},

    # 设置光标位置
    setCursor: docMethodOp(function(line, ch, options) {
      setSimpleSelection(this, clipPos(this, typeof line == "number" ? Pos(line, ch || 0) : line), null, options);
    }),
    
    # 设置选区范围
    setSelection: docMethodOp(function(anchor, head, options) {
      setSimpleSelection(this, clipPos(this, anchor), clipPos(this, head || anchor), options);
    }),
    
    # 扩展选区范围
    extendSelection: docMethodOp(function(head, other, options) {
      extendSelection(this, clipPos(this, head), other && clipPos(this, other), options);
    }),
    
    # 扩展多个选区范围
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
      // 如果未指定 primary，则默认为选择范围数组的第一个范围
      if (primary == null) { primary = Math.min(ranges.length - 1, this.sel.primIndex); }
      // 设置选择范围
      setSelection(this, normalizeSelection(this.cm, out, primary), options);
    }),
    addSelection: docMethodOp(function(anchor, head, options) {
      // 复制当前选择范围数组
      var ranges = this.sel.ranges.slice(0);
      // 向复制的选择范围数组中添加新的范围
      ranges.push(new Range(clipPos(this, anchor), clipPos(this, head || anchor)));
      // 设置选择范围
      setSelection(this, normalizeSelection(this.cm, ranges, ranges.length - 1), options);
    }),

    getSelection: function(lineSep) {
      // 获取选择范围数组
      var ranges = this.sel.ranges, lines;
      for (var i = 0; i < ranges.length; i++) {
        // 获取每个选择范围的文本内容
        var sel = getBetween(this, ranges[i].from(), ranges[i].to());
        // 将文本内容合并到 lines 数组中
        lines = lines ? lines.concat(sel) : sel;
      }
      // 如果 lineSep 为 false，则直接返回 lines 数组
      if (lineSep === false) { return lines }
      // 否则将 lines 数组中的文本内容用 lineSep 连接起来，并返回
      else { return lines.join(lineSep || this.lineSeparator()) }
    },
    getSelections: function(lineSep) {
      // 存储每个选择范围的文本内容
      var parts = [], ranges = this.sel.ranges;
      for (var i = 0; i < ranges.length; i++) {
        // 获取每个选择范围的文本内容
        var sel = getBetween(this, ranges[i].from(), ranges[i].to());
        // 如果 lineSep 不为 false，则将文本内容用 lineSep 连接起来
        if (lineSep !== false) { sel = sel.join(lineSep || this.lineSeparator()); }
        // 将处理后的文本内容存入 parts 数组
        parts[i] = sel;
      }
      // 返回处理后的文本内容数组
      return parts
    },
    replaceSelection: function(code, collapse, origin) {
      // 复制当前选择范围数组
      var dup = [];
      for (var i = 0; i < this.sel.ranges.length; i++)
        { dup[i] = code; }
      // 替换选择范围数组中的文本内容
      this.replaceSelections(dup, collapse, origin || "+input");
    },
    // 替换选定文本的方法，接受代码、折叠标志和原始信息作为参数
    replaceSelections: docMethodOp(function(code, collapse, origin) {
      var changes = [], sel = this.sel;
      for (var i = 0; i < sel.ranges.length; i++) {
        var range = sel.ranges[i];
        // 为每个选定范围创建变化对象
        changes[i] = {from: range.from(), to: range.to(), text: this.splitLines(code[i]), origin: origin};
      }
      // 计算新的选定范围
      var newSel = collapse && collapse != "end" && computeReplacedSel(this, changes, collapse);
      // 逆序应用变化对象
      for (var i$1 = changes.length - 1; i$1 >= 0; i$1--)
        { makeChange(this, changes[i$1]); }
      // 如果有新的选定范围，则设置替换历史
      if (newSel) { setSelectionReplaceHistory(this, newSel); }
      // 否则，确保光标可见
      else if (this.cm) { ensureCursorVisible(this.cm); }
    }),
    // 撤销操作
    undo: docMethodOp(function() {makeChangeFromHistory(this, "undo");}),
    // 重做操作
    redo: docMethodOp(function() {makeChangeFromHistory(this, "redo");}),
    // 撤销选定范围的操作
    undoSelection: docMethodOp(function() {makeChangeFromHistory(this, "undo", true);}),
    // 重做选定范围的操作
    redoSelection: docMethodOp(function() {makeChangeFromHistory(this, "redo", true);}),
    
    // 设置扩展标志
    setExtending: function(val) {this.extend = val;},
    // 获取扩展标志
    getExtending: function() {return this.extend},
    
    // 历史记录大小
    historySize: function() {
      var hist = this.history, done = 0, undone = 0;
      for (var i = 0; i < hist.done.length; i++) { if (!hist.done[i].ranges) { ++done; } }
      for (var i$1 = 0; i$1 < hist.undone.length; i$1++) { if (!hist.undone[i$1].ranges) { ++undone; } }
      return {undo: done, redo: undone}
    },
    // 清除历史记录
    clearHistory: function() {
      var this$1 = this;
      // 创建新的历史记录对象，并将其链接到文档
      this.history = new History(this.history.maxGeneration);
      linkedDocs(this, function (doc) { return doc.history = this$1.history; }, true);
    },
    
    // 标记为干净状态
    markClean: function() {
      this.cleanGeneration = this.changeGeneration(true);
    },
    // 改变历史记录的代数
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

    // 设置编辑器的历史记录，包括已完成和未完成的操作
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
      # 改变行的样式
      return changeLine(this, handle, where == "gutter" ? "gutter" : "class", function (line) {
        # 根据位置确定属性
        var prop = where == "text" ? "textClass"
                 : where == "background" ? "bgClass"
                 : where == "gutter" ? "gutterClass" : "wrapClass";
        # 如果属性为空，则设置为给定的类
        if (!line[prop]) { line[prop] = cls; }
        # 如果属性不为空，且已经包含给定的类，则返回 false
        else if (classTest(cls).test(line[prop])) { return false }
        # 如果属性不为空，且不包含给定的类，则添加给定的类
        else { line[prop] += " " + cls; }
        # 返回 true
        return true
      })
    }),
    # 移除行的类
    removeLineClass: docMethodOp(function(handle, where, cls) {
      # 改变行的样式
      return changeLine(this, handle, where == "gutter" ? "gutter" : "class", function (line) {
        # 根据位置确定属性
        var prop = where == "text" ? "textClass"
                 : where == "background" ? "bgClass"
                 : where == "gutter" ? "gutterClass" : "wrapClass";
        # 获取当前属性值
        var cur = line[prop];
        # 如果当前属性值为空，则返回 false
        if (!cur) { return false }
        # 如果给定的类为空，则将属性值设置为 null
        else if (cls == null) { line[prop] = null; }
        # 如果给定的类不为空
        else {
          # 查找属性值中是否包含给定的类
          var found = cur.match(classTest(cls));
          # 如果没有找到，则返回 false
          if (!found) { return false }
          # 找到给定的类，将其从属性值中移除
          var end = found.index + found[0].length;
          line[prop] = cur.slice(0, found.index) + (!found.index || end == cur.length ? "" : " ") + cur.slice(end) || null;
        }
        # 返回 true
        return true
      })
    }),

    # 添加行小部件
    addLineWidget: docMethodOp(function(handle, node, options) {
      # 调用添加行小部件的函数
      return addLineWidget(this, handle, node, options)
    }),
    # 移除行小部件
    removeLineWidget: function(widget) { widget.clear(); },

    # 标记文本
    markText: function(from, to, options) {
      # 调用标记文本的函数
      return markText(this, clipPos(this, from), clipPos(this, to), options, options && options.type || "range")
    },
    # 设置书签，将选项转换为真实选项
    setBookmark: function(pos, options) {
      var realOpts = {replacedWith: options && (options.nodeType == null ? options.widget : options),
                      insertLeft: options && options.insertLeft,
                      clearWhenEmpty: false, shared: options && options.shared,
                      handleMouseEvents: options && options.handleMouseEvents};
      # 限制位置在编辑器范围内
      pos = clipPos(this, pos);
      # 在指定位置标记文本
      return markText(this, pos, pos, realOpts, "bookmark")
    },
    # 查找指定位置的标记
    findMarksAt: function(pos) {
      # 限制位置在编辑器范围内
      pos = clipPos(this, pos);
      var markers = [], spans = getLine(this, pos.line).markedSpans;
      if (spans) { for (var i = 0; i < spans.length; ++i) {
        var span = spans[i];
        # 如果标记范围包含指定位置，则将标记添加到结果数组中
        if ((span.from == null || span.from <= pos.ch) &&
            (span.to == null || span.to >= pos.ch))
          { markers.push(span.marker.parent || span.marker); }
      } }
      return markers
    },
    # 查找指定范围内的标记
    findMarks: function(from, to, filter) {
      # 限制范围在编辑器范围内
      from = clipPos(this, from); to = clipPos(this, to);
      var found = [], lineNo = from.line;
      # 遍历指定范围内的每一行
      this.iter(from.line, to.line + 1, function (line) {
        var spans = line.markedSpans;
        if (spans) { for (var i = 0; i < spans.length; i++) {
          var span = spans[i];
          # 如果标记范围与指定范围有交集，则将标记添加到结果数组中
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
      # 遍历每一行，获取标记
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
      var lineNo = this.first, 
      // 获取行分隔符的长度
      sepSize = this.lineSeparator().length;
      // 遍历每一行
      this.iter(function (line) {
        // 计算当前行的长度加上分隔符长度
        var sz = line.text.length + sepSize;
        // 如果长度大于偏移量，则找到对应的位置
        if (sz > off) { ch = off; return true }
        // 减去当前行的长度和分隔符长度
        off -= sz;
        // 行号加一
        ++lineNo;
      });
      // 返回计算出的位置
      return clipPos(this, Pos(lineNo, ch))
    },
    
    // 根据位置计算索引
    indexFromPos: function (coords) {
      // 调用 clipPos 方法，确保位置在有效范围内
      coords = clipPos(this, coords);
      // 初始化索引为列号
      var index = coords.ch;
      // 如果行号小于第一行或者列号小于0，则返回0
      if (coords.line < this.first || coords.ch < 0) { return 0 }
      // 获取行分隔符的长度
      var sepSize = this.lineSeparator().length;
      // 遍历每一行
      this.iter(this.first, coords.line, function (line) { // iter aborts when callback returns a truthy value
        // 索引加上当前行的长度和分隔符长度
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
      // 复制滚动条位置
      doc.scrollTop = this.scrollTop; doc.scrollLeft = this.scrollLeft;
      // 复制选择内容
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

    // 创建一个与当前文档关联的文档
    linkedDoc: function(options) {
      // 如果没有传入选项，则初始化为空对象
      if (!options) { options = {}; }
      // 初始化起始行和结束行
      var from = this.first, to = this.first + this.size;
      // 如果传入了起始行，并且大于起始行，则更新起始行
      if (options.from != null && options.from > from) { from = options.from; }
      // 如果传入了结束行，并且小于结束行，则更新结束行
      if (options.to != null && options.to < to) { to = options.to; }
      // 创建一个新的文档对象
      var copy = new Doc(getLines(this, from, to), options.mode || this.modeOption, from, this.lineSep, this.direction);
      // 如果需要共享历史记录
      if (options.sharedHist) { 
        // 将当前文档的历史记录赋值给新文档
        copy.history = this.history
      ; }
      // 将新文档添加到当前文档的关联文档列表中
      (this.linked || (this.linked = [])).push({doc: copy, sharedHist: options.sharedHist});
      // 将当前文档添加到新文档的关联文档列表中
      copy.linked = [{doc: this, isParent: true, sharedHist: options.sharedHist}];
      // 复制共享标记
      copySharedMarkers(copy, findSharedMarkers(this));
      // 返回关联的新文档
      return copy
    },
    // 如果传入的参数是 CodeMirror 对象，则获取其 doc 属性
    unlinkDoc: function(other) {
      if (other instanceof CodeMirror) { other = other.doc; }
      // 如果当前文档对象有关联的文档对象
      if (this.linked) { for (var i = 0; i < this.linked.length; ++i) {
        var link = this.linked[i];
        // 如果找到了要解除关联的文档对象
        if (link.doc != other) { continue }
        // 从关联列表中移除当前文档对象
        this.linked.splice(i, 1);
        // 调用其他文档对象的 unlinkDoc 方法，解除关联
        other.unlinkDoc(this);
        // 移除共享标记
        detachSharedMarkers(findSharedMarkers(this));
        break
      } }
      // 如果当前文档对象和传入的文档对象有共享的历史记录，则分割历史记录
      if (other.history == this.history) {
        var splitIds = [other.id];
        // 遍历关联的文档对象，获取其 id
        linkedDocs(other, function (doc) { return splitIds.push(doc.id); }, true);
        // 创建新的历史记录对象，分割原有的历史记录
        other.history = new History(null);
        other.history.done = copyHistoryArray(this.history.done, splitIds);
        other.history.undone = copyHistoryArray(this.history.undone, splitIds);
      }
    },
    // 遍历关联的文档对象，并执行指定的函数
    iterLinkedDocs: function(f) {linkedDocs(this, f);},

    // 获取当前文档对象的编辑模式
    getMode: function() {return this.mode},
    // 获取当前文档对象的编辑器对象
    getEditor: function() {return this.cm},

    // 将字符串按行分割
    splitLines: function(str) {
      // 如果有指定行分隔符，则按指定的分隔符进行分割
      if (this.lineSep) { return str.split(this.lineSep) }
      // 否则自动识别行分隔符进行分割
      return splitLinesAuto(str)
    },
    // 获取当前文档对象的行分隔符
    lineSeparator: function() { return this.lineSep || "\n" },

    // 设置文本方向
    setDirection: docMethodOp(function (dir) {
      // 如果方向不是从右到左，则设置为从左到右
      if (dir != "rtl") { dir = "ltr"; }
      // 如果方向和当前文档对象的方向相同，则直接返回
      if (dir == this.direction) { return }
      // 设置文本方向
      this.direction = dir;
      // 遍历文档对象的每一行，重置行的顺序
      this.iter(function (line) { return line.order = null; });
      // 如果有编辑器对象，则通知编辑器文本方向已改变
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
    // 如果 pos 为假或者编辑器为只读状态，则直接返回，不执行后续操作
    if (!pos || cm.isReadOnly()) { return }
    // 可能是文件拖放操作，此时我们简单提取文本并插入
    if (files && files.length && window.FileReader && window.File) {
      // 获取文件数量和文本数组，以及已读取文件数量
      var n = files.length, text = Array(n), read = 0;
      // 定义一个函数，用于标记已读取并在所有文件都读取完毕后执行粘贴操作
      var markAsReadAndPasteIfAllFilesAreRead = function () {
        // 如果已读取文件数量等于文件总数，则执行粘贴操作
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
      // 定义一个函数，用于读取文件内容
      var readTextFromFile = function (file, i) {
        // 如果允许拖放的文件类型不包含当前文件类型，则标记已读取并执行粘贴操作
        if (cm.options.allowDropFileTypes &&
            indexOf(cm.options.allowDropFileTypes, file.type) == -1) {
          markAsReadAndPasteIfAllFilesAreRead();
          return
        }
        // 创建一个文件读取器
        var reader = new FileReader;
        // 处理读取错误
        reader.onerror = function () { return markAsReadAndPasteIfAllFilesAreRead(); };
        // 处理读取完成
        reader.onload = function () {
          var content = reader.result;
          // 如果内容包含不可见字符，则标记已读取并执行粘贴操作
          if (/[\x00-\x08\x0e-\x1f]{2}/.test(content)) {
            markAsReadAndPasteIfAllFilesAreRead();
            return
          }
          // 将文件内容存入文本数组，并标记已读取并执行粘贴操作
          text[i] = content;
          markAsReadAndPasteIfAllFilesAreRead();
        };
        // 以文本形式读取文件
        reader.readAsText(file);
      };
      // 遍历所有文件，依次读取文件内容
      for (var i = 0; i < files.length; i++) { readTextFromFile(files[i], i); }
    } else { // 如果是正常的拖放操作
      // 如果拖放发生在选定文本内部，则不执行替换操作
      if (cm.state.draggingText && cm.doc.sel.contains(pos) > -1) {
        // 调用拖放文本的方法
        cm.state.draggingText(e);
        // 确保编辑器重新获得焦点
        setTimeout(function () { return cm.display.input.focus(); }, 20);
        return
      }
      try {
        // 获取拖放的文本数据
        var text$1 = e.dataTransfer.getData("Text");
        if (text$1) {
          var selected;
          // 如果正在拖放文本并且不是复制操作，则保存当前选中的文本
          if (cm.state.draggingText && !cm.state.draggingText.copy)
            { selected = cm.listSelections(); }
          // 设置新的选中文本，不记录撤销操作
          setSelectionNoUndo(cm.doc, simpleSelection(pos, pos));
          // 如果有保存的选中文本，则删除它们
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

  // 当拖动开始时触发的事件处理函数
  function onDragStart(cm, e) {
    // 如果是 IE 并且不是拖放文本，或者距离上次拖放不足 100 毫秒，则阻止默认行为
    if (ie && (!cm.state.draggingText || +new Date - lastDrop < 100)) { e_stop(e); return }
    // 如果信号是 DOM 事件，或者事件在小部件内部，则返回
    if (signalDOMEvent(cm, e) || eventInWidget(cm.display, e)) { return }

    // 设置拖放的文本数据
    e.dataTransfer.setData("Text", cm.getSelection());
    // 设置拖放效果为复制或移动
    e.dataTransfer.effectAllowed = "copyMove";

    // 使用虚拟图像代替默认的浏览器图像
    // 最近的 Safari（~6.0.2）在这种情况下有时会崩溃，所以我们在那里不这样做
    if (e.dataTransfer.setDragImage && !safari) {
      // 创建一个虚拟图像
      var img = elt("img", null, null, "position: fixed; left: 0; top: 0;");
      // 设置图像的源
      img.src = "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==";
      // 如果是 Presto 引擎，则设置图像的宽高并添加到编辑器容器中
      if (presto) {
        img.width = img.height = 1;
        cm.display.wrapper.appendChild(img);
        // 强制重新布局，否则 Opera 会因为某种隐晦的原因而不使用我们的图像
        img._top = img.offsetTop;
      }
      // 设置拖放图像
      e.dataTransfer.setDragImage(img, 0, 0);
      // 如果是 Presto 引擎，则移除图像
      if (presto) { img.parentNode.removeChild(img); }
  }
}

function onDragOver(cm, e) {
  // 获取鼠标位置对应的编辑器位置
  var pos = posFromMouse(cm, e);
  if (!pos) { return }
  // 创建一个文档片段
  var frag = document.createDocumentFragment();
  // 在编辑器位置绘制选择光标
  drawSelectionCursor(cm, pos, frag);
  // 如果不存在拖拽光标，则创建一个
  if (!cm.display.dragCursor) {
    cm.display.dragCursor = elt("div", null, "CodeMirror-cursors CodeMirror-dragcursors");
    cm.display.lineSpace.insertBefore(cm.display.dragCursor, cm.display.cursorDiv);
  }
  // 移除拖拽光标的子元素，并添加新的文档片段
  removeChildrenAndAdd(cm.display.dragCursor, frag);
}

function clearDragCursor(cm) {
  // 如果存在拖拽光标，则移除
  if (cm.display.dragCursor) {
    cm.display.lineSpace.removeChild(cm.display.dragCursor);
    cm.display.dragCursor = null;
  }
}

// 这些必须小心处理，因为简单地为每个编辑器注册处理程序将导致编辑器永远不会被垃圾回收。

function forEachCodeMirror(f) {
  if (!document.getElementsByClassName) { return }
  // 获取所有类名为 "CodeMirror" 的元素
  var byClass = document.getElementsByClassName("CodeMirror"), editors = [];
  for (var i = 0; i < byClass.length; i++) {
    var cm = byClass[i].CodeMirror;
    if (cm) { editors.push(cm); }
  }
  if (editors.length) { editors[0].operation(function () {
    // 对每个编辑器执行指定的函数
    for (var i = 0; i < editors.length; i++) { f(editors[i]); }
  }); }
}

var globalsRegistered = false;
function ensureGlobalHandlers() {
  // 如果全局处理程序已注册，则返回
  if (globalsRegistered) { return }
  // 注册全局处理程序
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
  // 当窗口失去焦点时，显示编辑器为模糊状态
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
  // 请注意，默认情况下，保存和查找相关的命令未定义。用户代码或插件可以定义它们。未知命令将被简单地忽略。
  keyMap.pcDefault = {
    // 定义键盘快捷键映射对象，将按键组合映射到对应的操作
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
        "Alt-Left": "goLineStart", // 光标移动到当前行开头
        "Alt-Right": "goLineEnd", // 光标移动到当前行结尾
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
        "Ctrl-A": "goLineStart", // 光标移动到当前行开头
        "Ctrl-E": "goLineEnd", // 光标移动到当前行结尾
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
        "Cmd-Left": "goLineLeft", // 光标向左移动一行
        "Cmd-Right": "goLineRight", // 光标向右移动一行
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
    // 定义键盘快捷键映射对象的默认值，根据操作系统类型选择不同的默认键盘映射
    "fallthrough": ["basic", "emacsy"]
  };
  // 根据操作系统类型选择默认键盘映射
  keyMap["default"] = mac ? keyMap.macDefault : keyMap.pcDefault;

  // 键盘映射分发函数

  // 标准化键名，将键名转换为标准格式
  function normalizeKeyName(name) {
    var parts = name.split(/-(?!$)/);
    name = parts[parts.length - 1];
    var alt, ctrl, shift, cmd;
    // 遍历键名的修饰键，将修饰键转换为标准格式
    for (var i = 0; i < parts.length - 1; i++) {
      var mod = parts[i];
      if (/^(cmd|meta|m)$/i.test(mod)) { cmd = true; }
      else if (/^a(lt)?$/i.test(mod)) { alt = true; }
      else if (/^(c|ctrl|control)$/i.test(mod)) { ctrl = true; }
      else if (/^s(hift)?$/i.test(mod)) { shift = true; }
      else { throw new Error("Unrecognized modifier name: " + mod) }
    }
    if (alt) { name = "Alt-" + name; }
    if (ctrl) { name = "Ctrl-" + name; }
    if (cmd) { name = "Cmd-" + name; }
    if (shift) { name = "Shift-" + name; }
    return name
  }

  // 这是一个修补程序，用于保持键盘映射对象的原始对象工作方式（向后兼容性），同时支持标准化和多键绑定等功能。
  // 它编译一个新的标准化键盘映射，并更新旧对象以反映这一变化。
  function normalizeKeyMap(keymap) {
    var copy = {};
    // 遍历 keymap 对象的属性
    for (var keyname in keymap) { if (keymap.hasOwnProperty(keyname)) {
      // 获取 keymap 对象的属性值
      var value = keymap[keyname];
      // 如果属性名为 name、fallthrough、detach，或属性值为 "..."，则跳过当前循环
      if (/^(name|fallthrough|(de|at)tach)$/.test(keyname)) { continue }
      if (value == "...") { delete keymap[keyname]; continue }

      // 将属性名按空格分割成数组，然后规范化处理
      var keys = map(keyname.split(" "), normalizeKeyName);
      // 遍历处理后的属性名数组
      for (var i = 0; i < keys.length; i++) {
        // 初始化变量 val 和 name
        var val = (void 0), name = (void 0);
        // 如果当前循环到最后一个属性名
        if (i == keys.length - 1) {
          // 将属性名数组连接成字符串，赋值给 name，属性值赋值给 val
          name = keys.join(" ");
          val = value;
        } else {
          // 将属性名数组的部分连接成字符串，赋值给 name，属性值赋值为 "..."
          name = keys.slice(0, i + 1).join(" ");
          val = "...";
        }
        // 获取 copy 对象中的属性值
        var prev = copy[name];
        // 如果 copy 对象中不存在当前属性名，则将属性名和属性值添加到 copy 对象中
        if (!prev) { copy[name] = val; }
        // 如果 copy 对象中存在当前属性名，并且属性值不等于当前值，则抛出错误
        else if (prev != val) { throw new Error("Inconsistent bindings for " + name) }
      }
      // 删除 keymap 对象中的当前属性
      delete keymap[keyname];
    } }
    // 将 copy 对象中的属性和属性值复制到 keymap 对象中
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

    // 如果 map 对象中存在 fallthrough 属性
    if (map.fallthrough) {
      // 如果 fallthrough 属性不是数组，则递归调用 lookupKey 函数
      if (Object.prototype.toString.call(map.fallthrough) != "[object Array]")
        { return lookupKey(key, map.fallthrough, handle, context) }
      // 遍历 fallthrough 属性数组
      for (var i = 0; i < map.fallthrough.length; i++) {
        // 递归调用 lookupKey 函数
        var result = lookupKey(key, map.fallthrough[i], handle, context);
        // 如果返回结果存在，则返回该结果
        if (result) { return result }
      }
    }
  }

  // 判断按键是否为修饰键
  function isModifierKey(value) {
    // 获取按键名称
    var name = typeof value == "string" ? value : keyNames[value.keyCode];
    // 判断按键是否为 Ctrl、Alt、Shift、Mod
    return name == "Ctrl" || name == "Alt" || name == "Shift" || name == "Mod"
  }

  // 添加修饰键的名称
  function addModifierNames(name, event, noShift) {
    // 获取基础名称
    var base = name;
    // 如果按下了 Alt 键且基础名称不为 "Alt"，则在基础名称前添加 "Alt-"
    if (event.altKey && base != "Alt") { name = "Alt-" + name; }
    # 如果 flipCtrlCmd 为真，则判断 event.metaKey，否则判断 event.ctrlKey，如果都为真且 base 不为 "Ctrl"，则在 name 前加上 "Ctrl-"
    if ((flipCtrlCmd ? event.metaKey : event.ctrlKey) && base != "Ctrl") { name = "Ctrl-" + name; }
    # 如果 flipCtrlCmd 为真，则判断 event.ctrlKey，否则判断 event.metaKey，如果都为真且 base 不为 "Cmd"，则在 name 前加上 "Cmd-"
    if ((flipCtrlCmd ? event.ctrlKey : event.metaKey) && base != "Cmd") { name = "Cmd-" + name; }
    # 如果 noShift 为假且 event.shiftKey 为真且 base 不为 "Shift"，则在 name 前加上 "Shift-"
    if (!noShift && event.shiftKey && base != "Shift") { name = "Shift-" + name; }
    # 返回处理后的 name
    return name
  }

  // Look up the name of a key as indicated by an event object.
  function keyName(event, noShift) {
    # 如果是 Presto 引擎且 keyCode 为 34 且有字符输入，则返回假
    if (presto && event.keyCode == 34 && event["char"]) { return false }
    # 从 keyNames 对象中获取 keyCode 对应的键名
    var name = keyNames[event.keyCode];
    # 如果键名为 null 或者 event.altGraphKey 为真，则返回假
    if (name == null || event.altGraphKey) { return false }
    # 如果 keyCode 为 3 且 event.code 存在，则使用 event.code 作为键名
    if (event.keyCode == 3 && event.code) { name = event.code; }
    # 返回添加修饰键名后的键名
    return addModifierNames(name, event, noShift)
  }

  # 获取键映射
  function getKeyMap(val) {
    # 如果 val 是字符串，则从 keyMap 对象中获取对应的值，否则直接返回 val
    return typeof val == "string" ? keyMap[val] : val
  }

  // Helper for deleting text near the selection(s), used to implement
  // backspace, delete, and similar functionality.
  # 删除选择区域附近的文本的辅助函数，用于实现退格、删除等功能
  function deleteNearSelection(cm, compute) {
    var ranges = cm.doc.sel.ranges, kill = [];
    # 首先构建要删除的范围集合，合并重叠的范围
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
    # 然后，删除这些实际的范围
    runInOp(cm, function () {
      for (var i = kill.length - 1; i >= 0; i--)
        { replaceRange(cm.doc, "", kill[i].from, kill[i].to, "+delete"); }
      ensureCursorVisible(cm);
    });
  }

  # 在逻辑上移动字符
  function moveCharLogically(line, ch, dir) {
    # 跳过 line.text 中的扩展字符，获取目标位置
    var target = skipExtendingChars(line.text, ch + dir, dir);
    // 如果目标位置小于0或大于行文本长度，返回null；否则返回目标位置
    return target < 0 || target > line.text.length ? null : target
  }

  // 在文本行内逻辑移动光标
  function moveLogically(line, start, dir) {
    // 在文本行内逻辑移动光标
    var ch = moveCharLogically(line, start.ch, dir);
    // 如果移动结果为null，返回null；否则返回新的位置对象
    return ch == null ? null : new Pos(start.line, ch, dir < 0 ? "after" : "before")
  }

  // 获取行的可视末尾位置
  function endOfLine(visually, cm, lineObj, lineNo, dir) {
    // 如果是可视模式
    if (visually) {
      // 如果文档方向为rtl，则改变方向
      if (cm.doc.direction == "rtl") { dir = -dir; }
      // 获取行的文本顺序
      var order = getOrder(lineObj, cm.doc.direction);
      // 如果存在文本顺序
      if (order) {
        // 获取部分的末尾位置
        var part = dir < 0 ? lst(order) : order[0];
        // 根据移动方向和部分级别确定移动顺序
        var moveInStorageOrder = (dir < 0) == (part.level == 1);
        var sticky = moveInStorageOrder ? "after" : "before";
        var ch;
        // 如果部分级别大于0或文档方向为rtl
        if (part.level > 0 || cm.doc.direction == "rtl") {
          // 准备测量行
          var prep = prepareMeasureForLine(cm, lineObj);
          // 根据移动方向确定起始位置
          ch = dir < 0 ? lineObj.text.length - 1 : 0;
          // 获取目标位置的top值
          var targetTop = measureCharPrepared(cm, prep, ch).top;
          // 找到第一个top值与目标top值相同的字符位置
          ch = findFirst(function (ch) { return measureCharPrepared(cm, prep, ch).top == targetTop; }, (dir < 0) == (part.level == 1) ? part.from : part.to - 1, ch);
          // 如果sticky为"before"，根据逻辑移动字符
          if (sticky == "before") { ch = moveCharLogically(lineObj, ch, 1); }
        } else { 
          // 根据移动方向确定起始位置
          ch = dir < 0 ? part.to : part.from; 
        }
        // 返回新的位置对象
        return new Pos(lineNo, ch, sticky)
      }
    }
    // 返回新的位置对象
    return new Pos(lineNo, dir < 0 ? lineObj.text.length : 0, dir < 0 ? "before" : "after")
  }

  // 在文本行内可视移动光标
  function moveVisually(cm, line, start, dir) {
    // 获取文本行的文本顺序
    var bidi = getOrder(line, cm.doc.direction);
    // 如果不存在文本顺序，返回逻辑移动光标的结果
    if (!bidi) { return moveLogically(line, start, dir) }
    # 如果起始位置超出了文本长度，则将起始位置设置为文本长度，并且设置粘性为“before”
    if (start.ch >= line.text.length) {
      start.ch = line.text.length;
      start.sticky = "before";
    } 
    # 如果起始位置小于等于0，则将起始位置设置为0，并且设置粘性为“after”
    else if (start.ch <= 0) {
      start.ch = 0;
      start.sticky = "after";
    }
    # 获取起始位置所在的双向文本段落的位置和段落信息
    var partPos = getBidiPartAt(bidi, start.ch, start.sticky), part = bidi[partPos];
    # 如果编辑器的文本方向为“ltr”，且当前段落为“ltr”且方向为偶数，并且（dir > 0 ? part.to > start.ch : part.from < start.ch）条件成立
    if (cm.doc.direction == "ltr" && part.level % 2 == 0 && (dir > 0 ? part.to > start.ch : part.from < start.ch)) {
      # 情况1：在ltr编辑器中移动ltr部分。即使有换行，也不会发生有趣的事情。
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

    # 如果文档的文本方向为“rtl”或者当前段落的级别为1
    if (cm.doc.direction == "rtl" || part.level == 1) {
      # 根据条件判断是否按照存储顺序移动
      var moveInStorageOrder = (part.level == 1) == (dir < 0);
      # 根据移动方向和条件判断是否在同一可视行内移动
      var ch = mv(start, moveInStorageOrder ? 1 : -1);
      if (ch != null && (!moveInStorageOrder ? ch >= part.from && ch >= wrappedLineExtent.begin : ch <= part.to && ch <= wrappedLineExtent.end)) {
        # 情况2：在rtl部分内移动或在同一可视行上的rtl编辑器内移动
        var sticky = moveInStorageOrder ? "before" : "after";
        return new Pos(start.line, ch, sticky)
      }
    }

    # 情况3：无法在当前可视行内的双向文本段落内移动，因此离开当前双向文本段落
    // 定义函数，用于在可视行中搜索指定位置的文本
    var searchInVisualLine = function (partPos, dir, wrappedLineExtent) {
      // 定义内部函数，用于获取结果位置
      var getRes = function (ch, moveInStorageOrder) { return moveInStorageOrder
        ? new Pos(start.line, mv(ch, 1), "before")
        : new Pos(start.line, ch, "after"); };

      // 循环遍历双向文本数组
      for (; partPos >= 0 && partPos < bidi.length; partPos += dir) {
        // 获取当前部分
        var part = bidi[partPos];
        // 判断是否按存储顺序移动
        var moveInStorageOrder = (dir > 0) == (part.level != 1);
        // 根据移动顺序获取字符位置
        var ch = moveInStorageOrder ? wrappedLineExtent.begin : mv(wrappedLineExtent.end, -1);
        // 如果字符位置在当前部分范围内，则返回结果位置
        if (part.from <= ch && ch < part.to) { return getRes(ch, moveInStorageOrder) }
        // 根据移动顺序获取字符位置
        ch = moveInStorageOrder ? part.from : mv(part.to, -1);
        // 如果字符位置在当前行范围内，则返回结果位置
        if (wrappedLineExtent.begin <= ch && ch < wrappedLineExtent.end) { return getRes(ch, moveInStorageOrder) }
      }
    };

    // Case 3a: Look for other bidi parts on the same visual line
    // 在同一可视行上查找其他双向文本部分
    var res = searchInVisualLine(partPos + dir, dir, wrappedLineExtent);
    if (res) { return res }

    // Case 3b: Look for other bidi parts on the next visual line
    // 在下一行可视行上查找其他双向文本部分
    var nextCh = dir > 0 ? wrappedLineExtent.end : mv(wrappedLineExtent.begin, -1);
    if (nextCh != null && !(dir > 0 && nextCh == line.text.length)) {
      res = searchInVisualLine(dir > 0 ? 0 : bidi.length - 1, dir, getWrappedLineExtent(nextCh));
      if (res) { return res }
    }

    // Case 4: Nowhere to move
    // 没有地方可以移动
    return null
  }

  // Commands are parameter-less actions that can be performed on an
  // editor, mostly used for keybindings.
  // 定义命令对象，用于在编辑器上执行无参数操作，主要用于键绑定
  var commands = {
    // 选择全部文本的命令
    selectAll: selectAll,
    // 单选命令，用于在编辑器中设置单个选择
    singleSelection: function (cm) { return cm.setSelection(cm.getCursor("anchor"), cm.getCursor("head"), sel_dontScroll); },
    // 删除光标所在行的内容
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
    // 删除光标所在行
    deleteLine: function (cm) { return deleteNearSelection(cm, function (range) { return ({
      from: Pos(range.from().line, 0),
      to: clipPos(cm.doc, Pos(range.to().line + 1, 0))
    }); }); },
    // 删除光标左侧的内容
    delLineLeft: function (cm) { return deleteNearSelection(cm, function (range) { return ({
      from: Pos(range.from().line, 0), to: range.from()
    }); }); },
    // 删除光标所在行的左侧内容
    delWrappedLineLeft: function (cm) { return deleteNearSelection(cm, function (range) {
      // 获取光标所在行的顶部坐标
      var top = cm.charCoords(range.head, "div").top + 5;
      // 获取左侧位置的字符坐标
      var leftPos = cm.coordsChar({left: 0, top: top}, "div");
      return {from: leftPos, to: range.from()}
    }); },
    // 删除光标所在行的右侧内容
    delWrappedLineRight: function (cm) { return deleteNearSelection(cm, function (range) {
      // 获取光标所在行的顶部坐标
      var top = cm.charCoords(range.head, "div").top + 5;
      // 获取右侧位置的字符坐标
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
    # 定义函数，将光标移动到当前行的开头
    goLineStartSmart: function (cm) { return cm.extendSelectionsBy(function (range) { return lineStartSmart(cm, range.head); },
      {origin: "+move", bias: 1}
    ); },
    # 定义函数，将光标移动到当前行的末尾
    goLineEnd: function (cm) { return cm.extendSelectionsBy(function (range) { return lineEnd(cm, range.head.line); },
      {origin: "+move", bias: -1}
    ); },
    # 定义函数，将光标移动到当前行的右侧
    goLineRight: function (cm) { return cm.extendSelectionsBy(function (range) {
      # 获取光标位置的顶部坐标
      var top = cm.cursorCoords(range.head, "div").top + 5;
      # 将光标移动到当前行的右侧
      return cm.coordsChar({left: cm.display.lineDiv.offsetWidth + 100, top: top}, "div")
    }, sel_move); },
    # 定义函数，将光标移动到当前行的左侧
    goLineLeft: function (cm) { return cm.extendSelectionsBy(function (range) {
      # 获取光标位置的顶部坐标
      var top = cm.cursorCoords(range.head, "div").top + 5;
      # 将光标移动到当前行的左侧
      return cm.coordsChar({left: 0, top: top}, "div")
    }, sel_move); },
    # 定义函数，智能地将光标移动到当前行的左侧
    goLineLeftSmart: function (cm) { return cm.extendSelectionsBy(function (range) {
      # 获取光标位置的顶部坐标
      var top = cm.cursorCoords(range.head, "div").top + 5;
      # 将光标移动到当前行的左侧
      var pos = cm.coordsChar({left: 0, top: top}, "div");
      # 如果光标位置在当前行的空白字符之前，则智能地将光标移动到当前行的开头
      if (pos.ch < cm.getLine(pos.line).search(/\S/)) { return lineStartSmart(cm, range.head) }
      return pos
    }, sel_move); },
    # 定义函数，将光标向上移动一行
    goLineUp: function (cm) { return cm.moveV(-1, "line"); },
    # 定义函数，将光标向下移动一行
    goLineDown: function (cm) { return cm.moveV(1, "line"); },
    # 定义函数，将光标向上翻页
    goPageUp: function (cm) { return cm.moveV(-1, "page"); },
    # 定义函数，将光标向下翻页
    goPageDown: function (cm) { return cm.moveV(1, "page"); },
    # 定义函数，将光标向左移动一个字符
    goCharLeft: function (cm) { return cm.moveH(-1, "char"); },
    # 定义函数，将光标向右移动一个字符
    goCharRight: function (cm) { return cm.moveH(1, "char"); },
    # 定义函数，将光标向左移动一列
    goColumnLeft: function (cm) { return cm.moveH(-1, "column"); },
    # 定义函数，将光标向右移动一列
    goColumnRight: function (cm) { return cm.moveH(1, "column"); },
    # 定义函数，将光标向左移动一个单词
    goWordLeft: function (cm) { return cm.moveH(-1, "word"); },
    # 定义函数，将光标向右移动一个组
    goGroupRight: function (cm) { return cm.moveH(1, "group"); },
    # 定义函数，将光标向左移动一个组
    goGroupLeft: function (cm) { return cm.moveH(-1, "group"); },
    # 定义函数，将光标向右移动一个单词
    goWordRight: function (cm) { return cm.moveH(1, "word"); },
    # 定义函数，删除光标前的字符
    delCharBefore: function (cm) { return cm.deleteH(-1, "char"); },
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
      # 计算需要插入的空格数，以保持对齐
      var spaces = [], ranges = cm.listSelections(), tabSize = cm.options.tabSize;
      for (var i = 0; i < ranges.length; i++) {
        var pos = ranges[i].from();
        var col = countColumn(cm.getLine(pos.line), pos.ch, tabSize);
        spaces.push(spaceStr(tabSize - col % tabSize));
      }
      # 替换选定文本为计算出的空格
      cm.replaceSelections(spaces);
    },
    # 默认的制表符行为
    defaultTab: function (cm) {
      # 如果有选定文本，则增加缩进
      if (cm.somethingSelected()) { cm.indentSelection("add"); }
      # 否则插入制表符
      else { cm.execCommand("insertTab"); }
    },
    # 交换光标位置左右两个字符的内容
    # 然后将光标移动到交换后的字符后面
    #
    # 不考虑换行符
    # 不扫描多于一行来查找字符
    # 在空行上不执行任何操作
    # 不对非空选定文本执行任何操作
    // 定义一个名为transposeChars的函数，接受CodeMirror实例作为参数，返回一个操作函数
    transposeChars: function (cm) { return runInOp(cm, function () {
      // 获取当前选区的范围
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
            // 将光标位置的字符和前一个字符交换
            cur = new Pos(cur.line, cur.ch + 1);
            cm.replaceRange(line.charAt(cur.ch - 1) + line.charAt(cur.ch - 2),
                            Pos(cur.line, cur.ch - 2), cur, "+transpose");
          } else if (cur.line > cm.doc.first) {
            // 获取上一行的文本内容
            var prev = getLine(cm.doc, cur.line - 1).text;
            // 如果上一行内容不为空
            if (prev) {
              // 将当前行的第一个字符和上一行的最后一个字符交换
              cur = new Pos(cur.line, 1);
              cm.replaceRange(line.charAt(0) + cm.doc.lineSeparator() +
                              prev.charAt(prev.length - 1),
                              Pos(cur.line - 1, prev.length - 1), cur, "+transpose");
            }
          }
        }
        // 将新的光标位置添加到newSel数组中
        newSel.push(new Range(cur, cur));
      }
      // 设置新的选区范围
      cm.setSelections(newSel);
    }); },
    // 定义一个名为newlineAndIndent的函数，接受CodeMirror实例作为参数，返回一个操作函数
    newlineAndIndent: function (cm) { return runInOp(cm, function () {
      // 获取当前所有选区的范围
      var sels = cm.listSelections();
      // 遍历选区范围
      for (var i = sels.length - 1; i >= 0; i--)
        { cm.replaceRange(cm.doc.lineSeparator(), sels[i].anchor, sels[i].head, "+input"); }
      // 获取更新后的选区范围
      sels = cm.listSelections();
      // 遍历选区范围
      for (var i$1 = 0; i$1 < sels.length; i$1++)
        { cm.indentLine(sels[i$1].from().line, null, true); }
      // 确保光标可见
      ensureCursorVisible(cm);
    }); },
    // 定义一个名为openLine的函数，接受CodeMirror实例作为参数，将光标位置替换为换行符
    openLine: function (cm) { return cm.replaceSelection("\n", "start"); },
    // 定义一个名为toggleOverwrite的函数，接受CodeMirror实例作为参数，切换覆盖模式
    toggleOverwrite: function (cm) { return cm.toggleOverwrite(); }
  };

  // 定义一个名为lineStart的函数，接受CodeMirror实例和行号作为参数，返回行的起始位置
  function lineStart(cm, lineN) {
    // 获取指定行的文本内容
    var line = getLine(cm.doc, lineN);
    // 获取可视行
    var visual = visualLine(line);
    // 如果可视行和实际行不一致，则更新行号
    if (visual != line) { lineN = lineNo(visual); }
    // 返回行的起始位置
    return endOfLine(true, cm, visual, lineN, 1)
  }
  // 定义一个名为lineEnd的函数，接受CodeMirror实例和行号作为参数，返回行的结束位置
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
      // 根据文本内容和文档的方向获取排序顺序
      var order = getOrder(line, cm.doc.direction);
      // 如果没有排序顺序或者排序顺序的级别为0
      if (!order || order[0].level == 0) {
        // 获取第一个非空白字符的位置
        var firstNonWS = Math.max(start.ch, line.text.search(/\S/));
        // 判断当前位置是否在空白字符内
        var inWS = pos.line == start.line && pos.ch <= firstNonWS && pos.ch;
        // 返回起始位置
        return Pos(start.line, inWS ? 0 : firstNonWS, start.sticky)
      }
      // 返回起始位置
      return start
    }
    
    // 执行绑定到按键的处理程序
    function doHandleBinding(cm, bound, dropShift) {
      // 如果绑定是字符串，则获取对应的命令
      if (typeof bound == "string") {
        bound = commands[bound];
        // 如果没有对应的命令，则返回false
        if (!bound) { return false }
      }
      // 确保之前的输入已被读取，以便处理程序看到文档的一致视图
      cm.display.input.ensurePolled();
      var prevShift = cm.display.shift, done = false;
      try {
        // 如果编辑器是只读的，则设置状态为禁止编辑
        if (cm.isReadOnly()) { cm.state.suppressEdits = true; }
        // 如果需要丢弃shift键，则设置shift为false
        if (dropShift) { cm.display.shift = false; }
        // 执行绑定的处理程序
        done = bound(cm) != Pass;
      } finally {
        cm.display.shift = prevShift;
        cm.state.suppressEdits = false;
      }
      return done
    }
    
    // 查找编辑器中指定名称的按键绑定
    function lookupKeyForEditor(cm, name, handle) {
      for (var i = 0; i < cm.state.keyMaps.length; i++) {
        var result = lookupKey(name, cm.state.keyMaps[i], handle, cm);
        if (result) { return result }
      }
      // 如果存在额外的按键绑定，则查找额外的按键绑定
      return (cm.options.extraKeys && lookupKey(name, cm.options.extraKeys, handle, cm))
        // 否则查找默认的按键绑定
        || lookupKey(name, cm.options.keyMap, handle, cm)
    }
    
    // 注意，尽管名称是dispatchKey，但此函数也用于检查绑定的鼠标点击事件
    
    // 停止序列的延迟执行
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
            # 清空按键序列
            cm.state.keySeq = null;
            # 重置输入
            cm.display.input.reset();
          }
        }); }
      # 如果按键序列与当前按键组合后的结果可以触发命令
      if (dispatchKeyInner(cm, seq + " " + name, e, handle)) { return true }
    }
    # 触发按键命令
    return dispatchKeyInner(cm, name, e, handle)
  }

  # 处理按键命令
  function dispatchKeyInner(cm, name, e, handle) {
    # 查找编辑器中按键对应的命令
    var result = lookupKeyForEditor(cm, name, handle);

    # 如果结果是"multi"，设置按键序列
    if (result == "multi")
      { cm.state.keySeq = name; }
    # 如果结果是"handled"，触发"keyHandled"事件
    if (result == "handled")
      { signalLater(cm, "keyHandled", cm, name, e); }

    # 如果结果是"handled"或"multi"，阻止默认行为，重启光标闪烁
    if (result == "handled" || result == "multi") {
      e_preventDefault(e);
      restartBlink(cm);
    }

    # 返回结果是否为真
    return !!result
  }

  # 处理键盘按下事件
  function handleKeyBinding(cm, e) {
    # 获取按键名称
    var name = keyName(e, true);
    # 如果按键名称不存在，返回false
    if (!name) { return false }

    # 如果按下Shift键且按键序列为空
    if (e.shiftKey && !cm.state.keySeq) {
      # 首先尝试解析包括'Shift-'的完整名称，如果失败，查看是否有绑定到不包括'Shift-'的按键名称的光标移动命令
      return dispatchKey(cm, "Shift-" + name, e, function (b) { return doHandleBinding(cm, b, true); })
          || dispatchKey(cm, name, e, function (b) {
               if (typeof b == "string" ? /^go[A-Z]/.test(b) : b.motion)
                 { return doHandleBinding(cm, b) }
             })
    } else {
      # 否则，触发按键命令
      return dispatchKey(cm, name, e, function (b) { return doHandleBinding(cm, b); })
    }
  }

  # 处理按键事件
  function handleCharBinding(cm, e, ch) {
    # 触发按键命令
    return dispatchKey(cm, "'" + ch + "'", e, function (b) { return doHandleBinding(cm, b, true); })
  }

  # 上一个停止的按键
  var lastStoppedKey = null;
  # 键盘按下事件处理函数
  function onKeyDown(e) {
    var cm = this;
    # 如果事件目标不是输入框
    if (e.target && e.target != cm.display.input.getField()) { return }
    # 将当前操作的焦点设置为活动元素
    cm.curOp.focus = activeElt();
    # 如果触发了 CodeMirror 的 DOM 事件，则返回
    if (signalDOMEvent(cm, e)) { return }
    # 处理 IE 对于 escape 键的特殊情况
    if (ie && ie_version < 11 && e.keyCode == 27) { e.returnValue = false; }
    # 获取按键的键码
    var code = e.keyCode;
    # 设置 shift 键的状态
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

    # 当按下 Alt 键时，将鼠标变为十字线（仅在 Mac 上）
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
    # 如果事件目标不是 CodeMirror 的输入框，则返回
    if (e.target && e.target != cm.display.input.getField()) { return }
    # 如果事件在小部件中或者触发了 CodeMirror 的 DOM 事件，或者按下了 Ctrl 键但没有按下 Alt 键（或者在 Mac 上按下了 Meta 键），则返回
    if (eventInWidget(cm.display, e) || signalDOMEvent(cm, e) || e.ctrlKey && !e.altKey || mac && e.metaKey) { return }
    # 获取键码和字符码
    var keyCode = e.keyCode, charCode = e.charCode;
    # 处理 Opera 浏览器的特殊情况
    if (presto && keyCode == lastStoppedKey) {lastStoppedKey = null; e_preventDefault(e); return}
    # 处理键盘绑定
    if ((presto && (!e.which || e.which < 10)) && handleKeyBinding(cm, e)) { return }
    var ch = String.fromCharCode(charCode == null ? keyCode : charCode);
    // 某些浏览器会对退格键触发键盘按键事件
    if (ch == "\x08") { return }
    // 如果字符绑定处理函数返回 true，则返回
    if (handleCharBinding(cm, e, ch)) { return }
    // 触发输入框的按键事件处理函数
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

  // 比较当前点击事件和上一次点击事件的时间、位置和按钮信息
  PastClick.prototype.compare = function (time, pos, button) {
    return this.time + DOUBLECLICK_DELAY > time &&
      cmp(pos, this.pos) == 0 && button == this.button
  };

  // 上一次单击和双击的点击事件
  var lastClick, lastDoubleClick;
  function clickRepeat(pos, button) {
    var now = +new Date;
    // 如果上一次是双击事件且符合条件，则返回 "triple"
    if (lastDoubleClick && lastDoubleClick.compare(now, pos, button)) {
      lastClick = lastDoubleClick = null;
      return "triple"
    // 如果上一次是单击事件且符合条件，则返回 "double"
    } else if (lastClick && lastClick.compare(now, pos, button)) {
      lastDoubleClick = new PastClick(now, pos, button);
      lastClick = null;
      return "double"
    // 否则，记录当前点击事件为上一次单击事件，并返回 "single"
    } else {
      lastClick = new PastClick(now, pos, button);
      lastDoubleClick = null;
      return "single"
    }
  }

  // 鼠标按下事件可能是单击、双击、三击、开始选择拖拽、开始文本拖拽、新的光标（ctrl+单击）、矩形拖拽（alt+拖拽）或者 xwin 中键粘贴。或者可能是点击到不应干扰的东西，比如滚动条或小部件。
  function onMouseDown(e) {
    var cm = this, display = cm.display;
    // 如果触发了 DOM 事件或者是触摸屏事件，则返回
    if (signalDOMEvent(cm, e) || display.activeTouch && display.input.supportsTouch()) { return }
    // 确保输入框已经轮询
    display.input.ensurePolled();
    // 记录是否按下了 shift 键
    display.shift = e.shiftKey;

    // 如果点击在小部件上，则返回
    if (eventInWidget(display, e)) {
      if (!webkit) {
        // 短暂关闭可拖动性，以允许小部件进行正常拖拽操作
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
    # 如果 value.addNew 为 null，则根据操作系统类型和事件按键设置其值
    if (value.addNew == null) { value.addNew = mac ? event.metaKey : event.ctrlKey; }
    # 如果 value.moveOnDrag 为 null，则根据操作系统类型和事件按键设置其值
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
    # 如果允许拖放并且不是只读模式，并且是单击事件，并且鼠标位置在选择文本范围内
    # 则开始拖放操作，否则进行文本选择操作
    if (cm.options.dragDrop && dragAndDrop && !cm.isReadOnly() &&
        repeat == "single" && (contained = sel.contains(pos)) > -1 &&
        (cmp((contained = sel.ranges[contained]).from(), pos) < 0 || pos.xRel > 0) &&
        (cmp(contained.to(), pos) > 0 || pos.xRel < 0))
      { leftButtonStartDrag(cm, event, pos, behavior); }
    else
      { leftButtonSelect(cm, event, pos, behavior); }
  }

  # 开始文本拖放操作
  function leftButtonStartDrag(cm, event, pos, behavior) {
    var display = cm.display, moved = false;
    # 拖放结束时的处理函数
    var dragEnd = operation(cm, function (e) {
      # 如果是 Webkit 浏览器，则禁止滚动
      if (webkit) { display.scroller.draggable = false; }
      # 设置拖放状态为 false
      cm.state.draggingText = false;
      # 移除鼠标事件监听
      off(display.wrapper.ownerDocument, "mouseup", dragEnd);
      off(display.wrapper.ownerDocument, "mousemove", mouseMove);
      off(display.scroller, "dragstart", dragStart);
      off(display.scroller, "drop", dragEnd);
      # 如果没有移动过，则视为点击事件
      if (!moved) {
        e_preventDefault(e);
        # 如果不是添加新的选择，则扩展选择范围
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
    // 监听 display.scroller 的 dragstart 事件，触发 dragStart 函数
    on(display.scroller, "dragstart", dragStart);
    // 监听 display.scroller 的 drop 事件，触发 dragEnd 函数
    on(display.scroller, "drop", dragEnd);

    // 延迟触发 cm 的 blur 事件
    delayBlurEvent(cm);
    // 延迟 20 毫秒后，让 display.input 获取焦点
    setTimeout(function () { return display.input.focus(); }, 20);
  }

  // 根据单位返回对应的 Range 对象
  function rangeForUnit(cm, pos, unit) {
    if (unit == "char") { return new Range(pos, pos) }
    if (unit == "word") { return cm.findWordAt(pos) }
    if (unit == "line") { return new Range(Pos(pos.line, 0), clipPos(cm.doc, Pos(pos.line + 1, 0))) }
    var result = unit(cm, pos);
    return new Range(result.from, result.to)
  }

  // 处理左键选择文本的函数
  function leftButtonSelect(cm, event, start, behavior) {
    var display = cm.display, doc = cm.doc;
    // 阻止默认事件
    e_preventDefault(event);

    var ourRange, ourIndex, startSel = doc.sel, ranges = startSel.ranges;
    // 如果是添加新的选择区域且不是扩展选择，则根据 start 获取对应的 Range 对象
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

    // 如果是矩形选择，则根据 behavior.addNew 判断是否创建新的 Range 对象
    if (behavior.unit == "rectangle") {
      if (!behavior.addNew) { ourRange = new Range(start, start); }
      // 根据鼠标位置获取对应的 pos
      start = posFromMouse(cm, event, true, true);
      ourIndex = -1;
    } else {
      // 如果不是添加新的选择范围，则根据行为单位和起始位置计算范围
      var range = rangeForUnit(cm, start, behavior.unit);
      // 如果是扩展选择，则扩展当前范围
      if (behavior.extend)
        { ourRange = extendRange(ourRange, range.anchor, range.head, behavior.extend); }
      // 否则直接使用计算得到的范围
      else
        { ourRange = range; }
    }

    // 如果不是添加新的选择范围
    if (!behavior.addNew) {
      // 重置选择范围的索引为0
      ourIndex = 0;
      // 设置文档的选择范围为新的选择范围
      setSelection(doc, new Selection([ourRange], 0), sel_mouse);
      // 记录起始选择范围
      startSel = doc.sel;
    } else if (ourIndex == -1) {
      // 如果选择范围索引为-1，则将其设置为范围数组的长度
      ourIndex = ranges.length;
      // 设置文档的选择范围为包含新范围的范围数组
      setSelection(doc, normalizeSelection(cm, ranges.concat([ourRange]), ourIndex),
                   {scroll: false, origin: "*mouse"});
    } else if (ranges.length > 1 && ranges[ourIndex].empty() && behavior.unit == "char" && !behavior.extend) {
      // 如果范围数组长度大于1且当前选择范围为空且行为单位为字符且不是扩展选择
      // 则设置文档的选择范围为排除当前选择范围的范围数组
      setSelection(doc, normalizeSelection(cm, ranges.slice(0, ourIndex).concat(ranges.slice(ourIndex + 1)), 0),
                   {scroll: false, origin: "*mouse"});
      // 记录起始选择范围
      startSel = doc.sel;
    } else {
      // 替换文档中的一个选择范围
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
        // 计算左右边界
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
    // 计数器，用于跟踪鼠标事件的次数
    var counter = 0;
    
    // 扩展选区的函数
    function extend(e) {
      // 当前计数
      var curCount = ++counter;
      // 获取鼠标位置对应的编辑器位置
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
        // 如果存在滚动，则延迟执行滚动操作
        if (outside) { setTimeout(operation(cm, function () {
          if (counter != curCount) { return }
          display.scroller.scrollTop += outside;
          extend(e);
        }), 50); }
      }
    }
    
    // 完成选区操作
    function done(e) {
      // 取消选区状态
      cm.state.selectingText = false;
      // 重置计数器
      counter = Infinity;
      // 如果 e 为 null 或 undefined，则取消选区，否则设置焦点到编辑器
      if (e) {
        e_preventDefault(e);
        display.input.focus();
      }
      // 移除鼠标移动和鼠标释放事件监听
      off(display.wrapper.ownerDocument, "mousemove", move);
      off(display.wrapper.ownerDocument, "mouseup", up);
      // 重置历史记录的选区来源
      doc.history.lastSelOrigin = null;
    }
    
    // 鼠标移动事件处理
    var move = operation(cm, function (e) {
      // 如果鼠标按钮为0或没有按下鼠标，则完成选区操作，否则继续扩展选区
      if (e.buttons === 0 || !e_button(e)) { done(e); }
      else { extend(e); }
    });
    
    // 鼠标释放事件处理
    var up = operation(cm, done);
    // 设置编辑器状态为选区操作中
    cm.state.selectingText = up;
    // 添加鼠标移动事件监听
    on(display.wrapper.ownerDocument, "mousemove", move);
    # 在显示器上注册鼠标抬起事件的监听器，当事件发生时调用 up 函数
    on(display.wrapper.ownerDocument, "mouseup", up);
  }

  // 用于在鼠标选择时调整锚点到正确的双向跳转的一侧
  function bidiSimplify(cm, range) {
    // 获取锚点和头部的位置
    var anchor = range.anchor;
    var head = range.head;
    // 获取锚点所在行的内容
    var anchorLine = getLine(cm.doc, anchor.line);
    // 如果锚点和头部位置相同且粘性相同，则返回原始范围
    if (cmp(anchor, head) == 0 && anchor.sticky == head.sticky) { return range }
    // 获取锚点所在行的双向文本顺序
    var order = getOrder(anchorLine);
    // 如果没有双向文本顺序，则返回原始范围
    if (!order) { return range }
    // 获取锚点所在的双向文本部分的索引和部分信息
    var index = getBidiPartAt(order, anchor.ch, anchor.sticky), part = order[index];
    // 如果锚点不在部分的起始或结束位置，则返回原始范围
    if (part.from != anchor.ch && part.to != anchor.ch) { return range }
    // 计算头部相对于锚点的视觉位置（<0表示在左侧，>0表示在右侧）
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
    // 使用左侧的部分来确定新的锚点位置
    var usePart = order[boundary + (leftSide ? -1 : 0)];
    var from = leftSide == (usePart.level == 1);
    var ch = from ? usePart.from : usePart.to, sticky = from ? "after" : "before";
    // 如果锚点位置和粘性与新位置相同，则返回原始范围，否则返回新的范围
    return anchor.ch == ch && anchor.sticky == sticky ? range : new Range(new Pos(anchor.line, ch, sticky), head)
  }

  // 确定事件是否发生在行号区域，并触发相应事件的处理程序
  function gutterEvent(cm, e, type, prevent) {
    var mX, mY;
    if (e.touches) {
      mX = e.touches[0].clientX;
      mY = e.touches[0].clientY;
  } else {
    // 如果鼠标事件无法获取坐标，则返回 false
    try { mX = e.clientX; mY = e.clientY; }
    catch(e$1) { return false }
  }
  // 如果鼠标横坐标大于编辑器的 gutter 右边界，则返回 false
  if (mX >= Math.floor(cm.display.gutters.getBoundingClientRect().right)) { return false }
  // 如果需要阻止默认行为，则调用 e_preventDefault 函数
  if (prevent) { e_preventDefault(e); }

  // 获取编辑器的 display 对象
  var display = cm.display;
  // 获取编辑器行的位置信息
  var lineBox = display.lineDiv.getBoundingClientRect();

  // 如果鼠标纵坐标大于行的底部位置或者没有指定类型的事件处理函数，则返回 e_defaultPrevented(e)
  if (mY > lineBox.bottom || !hasHandler(cm, type)) { return e_defaultPrevented(e) }
  // 计算相对于编辑器内容区域的纵坐标
  mY -= lineBox.top - display.viewOffset;

  // 遍历编辑器的 gutterSpecs 数组
  for (var i = 0; i < cm.display.gutterSpecs.length; ++i) {
    // 获取指定索引的 gutter 元素
    var g = display.gutters.childNodes[i];
    // 如果 gutter 存在且右边界大于等于鼠标横坐标，则执行以下操作
    if (g && g.getBoundingClientRect().right >= mX) {
      // 获取鼠标所在行的行号
      var line = lineAtHeight(cm.doc, mY);
      // 获取指定索引的 gutter 规格
      var gutter = cm.display.gutterSpecs[i];
      // 触发指定类型的事件处理函数，并返回 e_defaultPrevented(e)
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

// 为了使上下文菜单生效，我们需要暂时显示文本区域（尽可能不显眼），让右键单击在其上生效
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
  // 如果没有指定类型的事件处理函数，则返回 false
  if (!hasHandler(cm, "gutterContextMenu")) { return false }
  // 返回 gutterEvent 函数的执行结果
  return gutterEvent(cm, e, "gutterContextMenu", false)
}

// 主题更改处理函数
function themeChanged(cm) {
  // 替换编辑器 wrapper 元素的类名，以更新主题样式
  cm.display.wrapper.className = cm.display.wrapper.className.replace(/\s*cm-s-\S+/g, "") +
    cm.options.theme.replace(/(^|\s)\s*/g, " cm-s-");
  // 清空缓存
  clearCaches(cm);
}

// 初始化对象
var Init = {toString: function(){return "CodeMirror.Init"}};

// 默认选项对象
var defaults = {};

// 选项处理函数对象
var optionHandlers = {};

// 定义选项处理函数
function defineOptions(CodeMirror) {
  // 获取 CodeMirror 的选项处理函数对象
  var optionHandlers = CodeMirror.optionHandlers;
    # 定义一个函数，用于设置 CodeMirror 的默认选项值，并且可以在初始化时执行处理函数
    function option(name, deflt, handle, notOnInit) {
      # 将默认值设置到 CodeMirror 的默认选项中
      CodeMirror.defaults[name] = deflt;
      # 如果存在处理函数，则将处理函数添加到选项处理函数列表中
      if (handle) { optionHandlers[name] =
        # 如果不是初始化时执行处理函数，则在旧值不是初始化值时执行处理函数
        notOnInit ? function (cm, val, old) {if (old != Init) { handle(cm, val, old); }} : handle; }
    }

    # 定义一个函数，用于设置 CodeMirror 的选项
    CodeMirror.defineOption = option;

    # 当没有旧值时传递给选项处理函数的值
    CodeMirror.Init = Init;

    # 这两个选项在初始化时从构造函数中调用，因为它们必须在编辑器可以开始之前初始化
    option("value", "", function (cm, val) { return cm.setValue(val); }, true);
    option("mode", null, function (cm, val) {
      # 设置编辑器的模式选项，并加载对应的模式
      cm.doc.modeOption = val;
      loadMode(cm);
    }, true);

    # 设置缩进单位选项，并在初始化时加载模式
    option("indentUnit", 2, loadMode, true);
    # 设置是否使用制表符进行缩进的选项
    option("indentWithTabs", false);
    # 设置智能缩进选项
    option("smartIndent", true);
    # 设置制表符大小选项，并在初始化时重置模式状态、清除缓存、注册变化
    option("tabSize", 4, function (cm) {
      resetModeState(cm);
      clearCaches(cm);
      regChange(cm);
    }, true);

    # 设置行分隔符选项，并在初始化时处理行分隔符
    option("lineSeparator", null, function (cm, val) {
      cm.doc.lineSep = val;
      if (!val) { return }
      # 处理新的行分隔符
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
    # 设置特殊字符选项，并在初始化时刷新编辑器
    option("specialChars", /[\u0000-\u001f\u007f-\u009f\u00ad\u061c\u200b-\u200c\u200e\u200f\u2028\u2029\ufeff\ufff9-\ufffc]/g, function (cm, val, old) {
      cm.state.specialChars = new RegExp(val.source + (val.test("\t") ? "" : "|\t"), "g");
      if (old != Init) { cm.refresh(); }
    });
    # 设置特殊字符占位符选项，并在初始化时刷新编辑器
    option("specialCharPlaceholder", defaultSpecialCharPlaceholder, function (cm) { return cm.refresh(); }, true);
    # 设置电气字符选项
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
    // 设置在整行更新之前先更新
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
      // 如果旧键盘映射存在且有 detach 方法，则调用 detach 方法
      if (prev && prev.detach) { prev.detach(cm, next); }
      // 如果新键盘映射存在且有 attach 方法，则调用 attach 方法
      if (next.attach) { next.attach(cm, prev || null); }
    });
    // 设置额外按键为 null
    option("extraKeys", null);
    // 设置鼠标配置为 null
    option("configureMouse", null);

    // 设置是否自动换行为 false，触发换行改变函数
    option("lineWrapping", false, wrappingChanged, true);
    // 设置 gutter 为一个空数组，触发更新 gutter 函数
    option("gutters", [], function (cm, val) {
      cm.display.gutterSpecs = getGutters(val, cm.options.lineNumbers);
      updateGutters(cm);
    }, true);
    // 设置固定 gutter 为 true，根据值调整 gutter 样式并刷新编辑器
    option("fixedGutter", true, function (cm, val) {
      cm.display.gutters.style.left = val ? compensateForHScroll(cm.display) + "px" : "0";
      cm.refresh();
    }, true);
    // 设置是否覆盖滚动条旁边的 gutter 为 false，触发更新滚动条函数
    option("coverGutterNextToScrollbar", false, function (cm) { return updateScrollbars(cm); }, true);
    // 设置滚动条样式为原生，初始化滚动条并更新滚动条
    option("scrollbarStyle", "native", function (cm) {
      initScrollbars(cm);
      updateScrollbars(cm);
      cm.display.scrollbars.setScrollTop(cm.doc.scrollTop);
      cm.display.scrollbars.setScrollLeft(cm.doc.scrollLeft);
    }, true);
    // 设置是否显示行号为 false，根据值更新 gutter 函数
    option("lineNumbers", false, function (cm, val) {
      cm.display.gutterSpecs = getGutters(cm.options.gutters, val);
      updateGutters(cm);
    }, true);
    // 设置第一行行号为 1，触发更新 gutter 函数
    option("firstLineNumber", 1, updateGutters, true);
    # 设置行号格式化函数，将整数转换为字符串，更新行号区域
    option("lineNumberFormatter", function (integer) { return integer; }, updateGutters, true);
    # 设置在选择文本时是否显示光标
    option("showCursorWhenSelecting", false, updateSelection, true);

    # 设置右键菜单重置选择
    option("resetSelectionOnContextMenu", true);
    # 设置按行复制/剪切
    option("lineWiseCopyCut", true);
    # 设置每次选择粘贴的行数
    option("pasteLinesPerSelection", true);
    # 设置选择是否可以触摸相邻
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
    # 设置是否允许拖放
    option("dragDrop", true, dragDropChanged);
    # 设置允许拖放的文件类型
    option("allowDropFileTypes", null);

    # 设置光标闪烁速率
    option("cursorBlinkRate", 530);
    # 设置光标滚动边距
    option("cursorScrollMargin", 0);
    # 设置光标高度
    option("cursorHeight", 1, updateSelection, true);
    # 设置单个光标高度
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
    // 判断拖拽操作是否开启或关闭
    var wasOn = old && old != Init;
    if (!value != !wasOn) {
      var funcs = cm.display.dragFunctions;
      var toggle = value ? on : off;
      // 根据拖拽操作的开启或关闭状态，切换相应的事件监听器
      toggle(cm.display.scroller, "dragstart", funcs.start);
      toggle(cm.display.scroller, "dragenter", funcs.enter);
      toggle(cm.display.scroller, "dragover", funcs.over);
      toggle(cm.display.scroller, "dragleave", funcs.leave);
      toggle(cm.display.scroller, "drop", funcs.drop);
    }
  }

  // 当换行设置改变时触发的函数
  function wrappingChanged(cm) {
    // 如果开启了换行设置
    if (cm.options.lineWrapping) {
      // 给显示区域添加 CodeMirror-wrap 类
      addClass(cm.display.wrapper, "CodeMirror-wrap");
      // 重置显示区域的最小宽度
      cm.display.sizer.style.minWidth = "";
      cm.display.sizerWidth = null;
    } else {
      // 移除显示区域的 CodeMirror-wrap 类
      rmClass(cm.display.wrapper, "CodeMirror-wrap");
      // 重新计算最大行数
      findMaxLine(cm);
    }
    // 估算行高
    estimateLineHeights(cm);
    // 注册改变
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

    // 将传入的选项复制到实例的 options 属性中
    this.options = options = options ? copyObj(options) : {};
    // 根据给定值和默认值确定有效选项
    copyObj(defaults, options, false);

    // 获取文档内容
    var doc = options.value;
    // 如果文档是字符串类型，则创建一个新的 Doc 对象
    if (typeof doc == "string") { doc = new Doc(doc, options.mode, null, options.lineSeparator, options.direction); }
    // 如果设置了 mode 选项，则将其赋值给 doc.modeOption
    else if (options.mode) { doc.modeOption = options.mode; }
    this.doc = doc;

    // 创建输入对象
    var input = new CodeMirror.inputStyles[options.inputStyle](this);
    // 创建显示对象
    var display = this.display = new Display(place, doc, input, options);
    // 将显示对象绑定到 wrapper 上
    display.wrapper.CodeMirror = this;
    // 主题改变时触发的函数
    themeChanged(this);
    // 如果开启了换行设置，则给显示区域添加 CodeMirror-wrap 类
    if (options.lineWrapping)
      { this.display.wrapper.className += " CodeMirror-wrap"; }
    // 初始化滚动条
    initScrollbars(this);
    // 定义组件的状态对象
    this.state = {
      keyMaps: [],  // 用于存储通过 addKeyMap 添加的键映射
      overlays: [], // 用于存储通过 addOverlay 添加的高亮覆盖层
      modeGen: 0,   // 当模式/覆盖层更改时递增，用于使高亮信息无效
      overwrite: false, // 是否覆盖模式
      delayingBlurEvent: false, // 是否延迟模糊事件
      focused: false, // 是否聚焦
      suppressEdits: false, // 用于在只读模式下禁用编辑
      pasteIncoming: -1, cutIncoming: -1, // 用于识别输入.poll中的粘贴/剪切编辑
      selectingText: false, // 是否正在选择文本
      draggingText: false, // 是否正在拖动文本
      highlight: new Delayed(), // 用于存储高亮工作器超时
      keySeq: null,  // 未完成的键序列
      specialChars: null // 特殊字符
    };

    // 如果设置了自动聚焦并且不是移动设备，则让输入框获得焦点
    if (options.autofocus && !mobile) { display.input.focus(); }

    // 覆盖 IE 在重新加载时有时对我们的隐藏文本区域执行的神奇文本内容恢复
    if (ie && ie_version < 11) { setTimeout(function () { return this$1.display.input.reset(true); }, 20); }

    // 注册事件处理程序
    registerEventHandlers(this);
    // 确保全局处理程序
    ensureGlobalHandlers();

    // 开始操作
    startOperation(this);
    this.curOp.forceUpdate = true;
    // 附加文档
    attachDoc(this, doc);

    // 如果设置了自动聚焦并且不是移动设备，或者已经聚焦，则延迟20毫秒后执行聚焦函数，否则执行失焦函数
    if ((options.autofocus && !mobile) || this.hasFocus())
      { setTimeout(bind(onFocus, this), 20); }
    else
      { onBlur(this); }

    // 遍历选项处理程序，并执行初始化
    for (var opt in optionHandlers) { if (optionHandlers.hasOwnProperty(opt))
      { optionHandlers[opt](this, options[opt], Init); } }
    // 可能更新行号宽度
    maybeUpdateLineNumberWidth(this);
    // 如果设置了完成初始化函数，则执行
    if (options.finishInit) { options.finishInit(this); }
    // 遍历初始化钩子，并执行
    for (var i = 0; i < initHooks.length; ++i) { initHooks[i](this); }
    // 结束操作
    endOperation(this);
    // 在 Webkit 中禁用 optimizelegibility，因为它会破坏文本在换行边界的测量
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
    // 旧版 IE 不会为双击触发第二次 mousedown 事件
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
    // 一些浏览器在打开菜单后才触发 contextmenu 事件，这时我们无法再进行处理。这些浏览器的右键菜单在 onMouseDown 中处理
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
    // 判断事件是否类似于鼠标事件
    function isMouseLikeTouchEvent(e) {
      // 如果触摸点数量不为1，则返回false
      if (e.touches.length != 1) { return false }
      // 获取触摸点的信息，判断是否类似于鼠标事件
      var touch = e.touches[0];
      return touch.radiusX <= 1 && touch.radiusY <= 1
    }
    // 判断触摸点是否远离其他点
    function farAway(touch, other) {
      // 如果其他点的左边界为null，则返回true
      if (other.left == null) { return true }
      // 计算触摸点和其他点的水平和垂直距离，判断是否远离
      var dx = other.left - touch.left, dy = other.top - touch.top;
      return dx * dx + dy * dy > 20 * 20
    }
    // 监听触摸开始事件
    on(d.scroller, "touchstart", function (e) {
      // 如果事件不是DOM事件信号、不类似于鼠标事件、不在行号区域点击，则执行以下操作
      if (!signalDOMEvent(cm, e) && !isMouseLikeTouchEvent(e) && !clickInGutter(cm, e)) {
        // 确保输入已轮询
        d.input.ensurePolled();
        // 清除触摸结束的定时器
        clearTimeout(touchFinished);
        var now = +new Date;
        // 设置活动触摸点的信息
        d.activeTouch = {start: now, moved: false,
                         prev: now - prevTouch.end <= 300 ? prevTouch : null};
        // 如果触摸点数量为1，则记录触摸点的位置
        if (e.touches.length == 1) {
          d.activeTouch.left = e.touches[0].pageX;
          d.activeTouch.top = e.touches[0].pageY;
        }
      }
    });
    // 监听触摸移动事件
    on(d.scroller, "touchmove", function () {
      // 如果存在活动触摸点，则将moved属性设置为true
      if (d.activeTouch) { d.activeTouch.moved = true; }
    });
    // 监听触摸结束事件
    on(d.scroller, "touchend", function (e) {
      var touch = d.activeTouch;
      // 如果存在活动触摸点且不在小部件内、触摸点有位置信息且未移动、触摸时间小于300ms，则执行以下操作
      if (touch && !eventInWidget(d, e) && touch.left != null &&
          !touch.moved && new Date - touch.start < 300) {
        // 根据触摸点的位置获取光标位置
        var pos = cm.coordsChar(d.activeTouch, "page"), range;
        // 如果不存在前一个触摸点或者与前一个触摸点距离较远，则执行单击操作
        if (!touch.prev || farAway(touch, touch.prev)) { range = new Range(pos, pos); }
        // 如果不存在前一个前一个触摸点或者与前一个前一个触摸点距离较远，则执行双击操作
        else if (!touch.prev.prev || farAway(touch, touch.prev.prev)) { range = cm.findWordAt(pos); }
        // 否则执行三击操作
        else { range = new Range(Pos(pos.line, 0), clipPos(cm.doc, Pos(pos.line + 1, 0))); }
        // 设置选区并聚焦
        cm.setSelection(range.anchor, range.head);
        cm.focus();
        // 阻止默认事件
        e_preventDefault(e);
      }
      // 结束触摸操作
      finishTouch();
    });
    // 监听触摸取消事件
    on(d.scroller, "touchcancel", finishTouch);

    // 同步虚拟滚动条和真实可滚动区域的滚动，确保视口在滚动时更新
    // 监听滚动事件，更新滚动条位置，发送滚动信号
    on(d.scroller, "scroll", function () {
      // 如果滚动条有高度
      if (d.scroller.clientHeight) {
        // 更新垂直滚动条位置
        updateScrollTop(cm, d.scroller.scrollTop);
        // 设置水平滚动条位置
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

    // 定义拖拽函数
    d.dragFunctions = {
      // 进入拖拽区域
      enter: function (e) {if (!signalDOMEvent(cm, e)) { e_stop(e); }},
      // 在拖拽区域内移动
      over: function (e) {if (!signalDOMEvent(cm, e)) { onDragOver(cm, e); e_stop(e); }},
      // 开始拖拽
      start: function (e) { return onDragStart(cm, e); },
      // 放置拖拽内容
      drop: operation(cm, onDrop),
      // 离开拖拽区域
      leave: function (e) {if (!signalDOMEvent(cm, e)) { clearDragCursor(cm); }}
    };

    // 获取输入框元素
    var inp = d.input.getField();
    // 监听键盘按键抬起事件
    on(inp, "keyup", function (e) { return onKeyUp.call(cm, e); });
    // 监听键盘按键按下事件
    on(inp, "keydown", operation(cm, onKeyDown));
    // 监听键盘按键按下并产生字符事件
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

  // 缩进给定行。how参数可以是"smart"、"add"/null、"subtract"或"prev"。当aggressive为false时（通常设置为true以强制单行缩进），空行不缩进，模式返回Pass的地方保持不变。
  function indentLine(cm, n, how, aggressive) {
    // 获取文档对象和状态
    var doc = cm.doc, state;
    // 如果how为null，则设置为"add"
    if (how == null) { how = "add"; }
    // 如果缩进方式是 "smart"
    if (how == "smart") {
      // 当模式没有缩进方法时，回退到 "prev"
      if (!doc.mode.indent) { how = "prev"; }
      else { state = getContextBefore(cm, n).state; }
    }

    // 获取选项中的制表符大小
    var tabSize = cm.options.tabSize;
    // 获取当前行的文本和当前空格数
    var line = getLine(doc, n), curSpace = countColumn(line.text, null, tabSize);
    // 如果行的状态已经存在，则将其置为 null
    if (line.stateAfter) { line.stateAfter = null; }
    // 获取当前空格字符串和缩进
    var curSpaceString = line.text.match(/^\s*/)[0], indentation;
    // 如果不是侵略模式且当前行没有非空白字符
    if (!aggressive && !/\S/.test(line.text)) {
      indentation = 0;
      how = "not";
    } else if (how == "smart") {
      // 根据模式的缩进方法获取缩进
      indentation = doc.mode.indent(state, line.text.slice(curSpaceString.length), line.text);
      // 如果缩进为 Pass 或大于 150，则根据侵略模式返回
      if (indentation == Pass || indentation > 150) {
        if (!aggressive) { return }
        how = "prev";
      }
    }
    // 如果缩进方式是 "prev"
    if (how == "prev") {
      // 如果当前行大于文档的第一行，则获取前一行的缩进
      if (n > doc.first) { indentation = countColumn(getLine(doc, n-1).text, null, tabSize); }
      else { indentation = 0; }
    } else if (how == "add") {
      // 如果缩进方式是 "add"，则在当前空格数的基础上增加一个缩进单位
      indentation = curSpace + cm.options.indentUnit;
    } else if (how == "subtract") {
      // 如果缩进方式是 "subtract"，则在当前空格数的基础上减少一个缩进单位
      indentation = curSpace - cm.options.indentUnit;
    } else if (typeof how == "number") {
      // 如果缩进方式是一个数字，则在当前空格数的基础上增加该数字
      indentation = curSpace + how;
    }
    // 缩进至少为 0
    indentation = Math.max(0, indentation);

    // 初始化缩进字符串和位置
    var indentString = "", pos = 0;
    // 如果选项中使用制表符缩进
    if (cm.options.indentWithTabs)
      { for (var i = Math.floor(indentation / tabSize); i; --i) {pos += tabSize; indentString += "\t";} }
    // 如果位置小于缩进，则添加空格
    if (pos < indentation) { indentString += spaceStr(indentation - pos); }

    // 如果缩进字符串不等于当前空格字符串
    if (indentString != curSpaceString) {
      // 替换当前行的缩进字符串
      replaceRange(doc, indentString, Pos(n, 0), Pos(n, curSpaceString.length), "+input");
      // 将行的状态置为 null
      line.stateAfter = null;
      // 返回 true
      return true
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
    // 保持文本输入框靠近光标位置，防止��为输入而滚动光标位置出视野
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
      # 如果是 IE 浏览器版本小于等于 11，则延迟 20 毫秒后执行操作
      if (ie_version <= 11) { setTimeout(operation(cm, function () { return this$1.updateFromDOM(); }), 20); }
    });

    # 给 div 元素添加开始组合事件监听器
    on(div, "compositionstart", function (e) {
      # 设置 composing 对象的 data 和 done 属性
      this$1.composing = {data: e.data, done: false};
    });

    # 给 div 元素添加组合更新事件监听器
    on(div, "compositionupdate", function (e) {
      # 如果 composing 对象不存在，则设置其 data 和 done 属性
      if (!this$1.composing) { this$1.composing = {data: e.data, done: false}; }
    });

    # 给 div 元素添加组合结束事件监听器
    on(div, "compositionend", function (e) {
      # 如果 composing 对象存在
      if (this$1.composing) {
        # 如果事件数据不等于 composing 对象的数据，则调用 readFromDOMSoon 方法
        if (e.data != this$1.composing.data) { this$1.readFromDOMSoon(); }
        # 设置 composing 对象的 done 属性为 true
        this$1.composing.done = true;
      }
    });

    # 给 div 元素添加触摸开始事件监听器
    on(div, "touchstart", function () { return input.forceCompositionEnd(); });

    # 给 div 元素添加输入事件监听器
    on(div, "input", function () {
      # 如果 composing 对象不存在，则调用 readFromDOMSoon 方法
      if (!this$1.composing) { this$1.readFromDOMSoon(); }
    });
    // 定义处理复制和剪切事件的函数
    function onCopyCut(e) {
      // 如果不属于输入框或者是 DOM 事件，则返回
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
      // 如果支持剪贴板操作
      if (e.clipboardData) {
        // 清空剪贴板数据
        e.clipboardData.clearData();
        // 获取最后复制的文本，设置到剪贴板中
        var content = lastCopied.text.join("\n");
        // iOS 暴露了剪贴板 API，但似乎会丢弃插入其中的内容
        e.clipboardData.setData("Text", content);
        // 如果剪贴板中的文本与设置的内容相同，则阻止默认行为
        if (e.clipboardData.getData("Text") == content) {
          e.preventDefault();
          return
        }
      }
      // 旧式的焦点到文本区域的 hack
      var kludge = hiddenTextarea(), te = kludge.firstChild;
      // 在编辑器的行空间前插入文本区域
      cm.display.lineSpace.insertBefore(kludge, cm.display.lineSpace.firstChild);
      // 设置文本区域的值为最后复制的文本
      te.value = lastCopied.text.join("\n");
      var hadFocus = document.activeElement;
      // 选择文本区域
      selectInput(te);
      // 延迟执行，移除文本区域，恢复焦点
      setTimeout(function () {
        cm.display.lineSpace.removeChild(kludge);
        hadFocus.focus();
        if (hadFocus == div) { input.showPrimarySelection(); }
      }, 50);
    }
    // 绑定复制事件
    on(div, "copy", onCopyCut);
    // 绑定剪切事件
    on(div, "cut", onCopyCut);
  };

  // 屏幕阅读器标签变化的函数
  ContentEditableInput.prototype.screenReaderLabelChanged = function (label) {
    // 如果有标签，则设置 aria-label 属性，否则移除该属性
    if(label) {
      this.div.setAttribute('aria-label', label);
    } else {
      this.div.removeAttribute('aria-label');
    }
  };

  // 准备选择的函数
  ContentEditableInput.prototype.prepareSelection = function () {
    // 调用 prepareSelection 函数，传入编辑器对象和 false 参数
    var result = prepareSelection(this.cm, false);
  // 检查当前焦点是否在结果元素上
  result.focus = document.activeElement == this.div;
  // 返回结果对象
  return result
};

// 显示选择内容
ContentEditableInput.prototype.showSelection = function (info, takeFocus) {
  // 如果没有选择信息或者编辑器没有内容，则返回
  if (!info || !this.cm.display.view.length) { return }
  // 如果有焦点或者需要获取焦点，则显示主要选择内容
  if (info.focus || takeFocus) { this.showPrimarySelection(); }
  // 显示多重选择内容
  this.showMultipleSelections(info);
};

// 获取选择内容
ContentEditableInput.prototype.getSelection = function () {
  return this.cm.display.wrapper.ownerDocument.getSelection()
};

// 显示主要选择内容
ContentEditableInput.prototype.showPrimarySelection = function () {
  var sel = this.getSelection(), cm = this.cm, prim = cm.doc.sel.primary();
  var from = prim.from(), to = prim.to();

  // 如果编辑器没有内容或者选择内容不在可视区域内，则移除所有选择内容并返回
  if (cm.display.viewTo == cm.display.viewFrom || from.line >= cm.display.viewTo || to.line < cm.display.viewFrom) {
    sel.removeAllRanges();
    return
  }

  // 获取当前锚点和焦点的位置
  var curAnchor = domToPos(cm, sel.anchorNode, sel.anchorOffset);
  var curFocus = domToPos(cm, sel.focusNode, sel.focusOffset);
  // 如果当前位置有效且和选择内容的起始和结束位置一致，则返回
  if (curAnchor && !curAnchor.bad && curFocus && !curFocus.bad &&
      cmp(minPos(curAnchor, curFocus), from) == 0 &&
      cmp(maxPos(curAnchor, curFocus), to) == 0)
    { return }

  var view = cm.display.view;
  // 获取起始和结束位置的 DOM 元素
  var start = (from.line >= cm.display.viewFrom && posToDOM(cm, from)) ||
      {node: view[0].measure.map[2], offset: 0};
  var end = to.line < cm.display.viewTo && posToDOM(cm, to);
  if (!end) {
    var measure = view[view.length - 1].measure;
    var map = measure.maps ? measure.maps[measure.maps.length - 1] : measure.map;
    end = {node: map[map.length - 1], offset: map[map.length - 2] - map[map.length - 3]};
  }

  // 如果起始或结束位置无效，则移除所有选择内容并返回
  if (!start || !end) {
    sel.removeAllRanges();
    return
  }

  var old = sel.rangeCount && sel.getRangeAt(0), rng;
  try { rng = range(start.node, start.offset, end.offset, end.node); }
  catch(e) {} // Our model of the DOM might be outdated, in which case the range we try to set can be impossible
}
    // 如果存在选区
    if (rng) {
      // 如果不是 gecko 浏览器，并且编辑器处于焦点状态
      if (!gecko && cm.state.focused) {
        // 折叠选区到指定的起始节点和偏移位置
        sel.collapse(start.node, start.offset);
        // 如果选区没有折叠
        if (!rng.collapsed) {
          // 清除所有选区
          sel.removeAllRanges();
          // 添加指定的选区
          sel.addRange(rng);
        }
      } else {
        // 清除所有选区
        sel.removeAllRanges();
        // 添加指定的选区
        sel.addRange(rng);
      }
      // 如果存在旧选区，并且当前选区的锚点节点为 null
      if (old && sel.anchorNode == null) { sel.addRange(old); }
      // 如果是 gecko 浏览器，启动优雅期
      else if (gecko) { this.startGracePeriod(); }
    }
    // 记住当前选区状态
    this.rememberSelection();
  };

  // 启动优雅期
  ContentEditableInput.prototype.startGracePeriod = function () {
      var this$1 = this;

    // 清除之前的优雅期定时器
    clearTimeout(this.gracePeriod);
    // 设置新的优雅期定时器
    this.gracePeriod = setTimeout(function () {
      this$1.gracePeriod = false;
      // 如果选区发生变化，则设置编辑器操作的选区变化标志
      if (this$1.selectionChanged())
        { this$1.cm.operation(function () { return this$1.cm.curOp.selectionChanged = true; }); }
    }, 20);
  };

  // 显示多重选区
  ContentEditableInput.prototype.showMultipleSelections = function (info) {
    // 移除光标和选区，并添加新的光标和选区
    removeChildrenAndAdd(this.cm.display.cursorDiv, info.cursors);
    removeChildrenAndAdd(this.cm.display.selectionDiv, info.selection);
  };

  // 记住当前选区状态
  ContentEditableInput.prototype.rememberSelection = function () {
    // 获取当前选区
    var sel = this.getSelection();
    // 记住当前选区的锚点节点、偏移位置、焦点节点和偏移位置
    this.lastAnchorNode = sel.anchorNode; this.lastAnchorOffset = sel.anchorOffset;
    this.lastFocusNode = sel.focusNode; this.lastFocusOffset = sel.focusOffset;
  };

  // 判断选区是否在编辑器内
  ContentEditableInput.prototype.selectionInEditor = function () {
    // 获取当前选区
    var sel = this.getSelection();
    // 如果没有选区范围，返回 false
    if (!sel.rangeCount) { return false }
    // 获取选区的公共祖先节点
    var node = sel.getRangeAt(0).commonAncestorContainer;
    // 判断编辑器是否包含该节点
    return contains(this.div, node)
  };

  // 设置编辑器焦点
  ContentEditableInput.prototype.focus = function () {
    // 如果编辑器不是只读模式
    if (this.cm.options.readOnly != "nocursor") {
      // 如果选区不在编辑器内，或者当前活动元素不是编辑器
      if (!this.selectionInEditor() || document.activeElement != this.div)
        { this.showSelection(this.prepareSelection(), true); }
      // 设置编辑器焦点
      this.div.focus();
  // 定义 ContentEditableInput 对象的 blur 方法，使其调用 div 的 blur 方法
  ContentEditableInput.prototype.blur = function () { this.div.blur(); };
  // 定义 ContentEditableInput 对象的 getField 方法，返回 div 对象
  ContentEditableInput.prototype.getField = function () { return this.div };
  // 定义 ContentEditableInput 对象的 supportsTouch 方法，始终返回 true
  ContentEditableInput.prototype.supportsTouch = function () { return true };
  // 定义 ContentEditableInput 对象的 receivedFocus 方法，处理输入框获取焦点时的逻辑
  ContentEditableInput.prototype.receivedFocus = function () {
    // 保存当前 input 对象
    var input = this;
    // 如果光标在编辑器中，则轮询选择
    if (this.selectionInEditor())
      { this.pollSelection(); }
    // 否则，设置 input.cm.curOp.selectionChanged 为 true
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
    // 在 Android Chrome 上，如果光标在不可编辑的块元素中，按退格键会隐藏虚拟键盘，这里模拟按退格键
    if (android && chrome && this.cm.display.gutterSpecs.length && isInGutter(sel.anchorNode)) {
      this.cm.triggerOnKeyDown({type: "keydown", keyCode: 8, preventDefault: Math.abs});
      this.blur();
      this.focus();
      return
    }
    // 如果正在输入中，则返回
    if (this.composing) { return }
    // 记住当前选择
    this.rememberSelection();
    // 将 DOM 元素转换为编辑器中的位置
    var anchor = domToPos(cm, sel.anchorNode, sel.anchorOffset);
    # 获取光标位置的起始位置
    var head = domToPos(cm, sel.focusNode, sel.focusOffset);
    # 如果锚点和头部位置都存在，则执行操作
    if (anchor && head) { runInOp(cm, function () {
      # 设置编辑器的选择区域为给定的范围
      setSelection(cm.doc, simpleSelection(anchor, head), sel_dontScroll);
      # 如果锚点或头部位置不正确，则标记选择区域已改变
      if (anchor.bad || head.bad) { cm.curOp.selectionChanged = true; }
    }); }
  };

  # 定义方法用于轮询内容
  ContentEditableInput.prototype.pollContent = function () {
    # 如果读取 DOM 的超时存在，则清除超时
    if (this.readDOMTimeout != null) {
      clearTimeout(this.readDOMTimeout);
      this.readDOMTimeout = null;
    }

    # 获取编辑器、显示区域和选择区域的相关信息
    var cm = this.cm, display = cm.display, sel = cm.doc.sel.primary();
    var from = sel.from(), to = sel.to();
    # 如果起始位置在行首且不是第一行，则将起始位置移动到上一行的末尾
    if (from.ch == 0 && from.line > cm.firstLine())
      { from = Pos(from.line - 1, getLine(cm.doc, from.line - 1).length); }
    # 如果结束位置在行末且不是最后一行，则将结束位置移动到下一行的开头
    if (to.ch == getLine(cm.doc, to.line).text.length && to.line < cm.lastLine())
      { to = Pos(to.line + 1, 0); }
    # 如果起始行小于显示区域的起始行或结束行大于显示区域的结束行，则返回 false
    if (from.line < display.viewFrom || to.line > display.viewTo - 1) { return false }

    # 获取起始位置的相关信息
    var fromIndex, fromLine, fromNode;
    if (from.line == display.viewFrom || (fromIndex = findViewIndex(cm, from.line)) == 0) {
      fromLine = lineNo(display.view[0].line);
      fromNode = display.view[0].node;
    } else {
      fromLine = lineNo(display.view[fromIndex].line);
      fromNode = display.view[fromIndex - 1].node.nextSibling;
    }
    # 获取结束位置的相关信息
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
    # 获取起始节点到结束节点之间的文本内容，并将其拆分为多行
    var newText = cm.doc.splitLines(domTextBetween(cm, fromNode, toNode, fromLine, toLine));
    # 获取起始位置到结束位置之间的文本内容
    var oldText = getBetween(cm.doc, Pos(fromLine, 0), Pos(toLine, getLine(cm.doc, toLine).text.length));
    // 当新旧文本长度都大于1时，进行循环比较
    while (newText.length > 1 && oldText.length > 1) {
      // 如果新旧文本的末尾相同，则删除末尾字符，行数减一
      if (lst(newText) == lst(oldText)) { newText.pop(); oldText.pop(); toLine--; }
      // 如果新旧文本的开头相同，则删除开头字符，行数加一
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

    // 计算变化的起始和结束位置
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
    # 清除读取 DOM 的定时器
    clearTimeout(this.readDOMTimeout);
    # 清空正在输入的内容
    this.composing = null;
    # 从 DOM 更新输入内容
    this.updateFromDOM();
    # 失去焦点
    this.div.blur();
    # 获取焦点
    this.div.focus();
  };
  # 延迟读取 DOM 内容
  ContentEditableInput.prototype.readFromDOMSoon = function () {
      var this$1 = this;

    if (this.readDOMTimeout != null) { return }
    # 设置定时器，延迟80毫秒后执行读取 DOM 内容的操作
    this.readDOMTimeout = setTimeout(function () {
      this$1.readDOMTimeout = null;
      # 如果正在输入中，则不执行读取 DOM 内容的操作
      if (this$1.composing) {
        if (this$1.composing.done) { this$1.composing = null; }
        else { return }
      }
      # 更新输入内容
      this$1.updateFromDOM();
    }, 80);
  };

  # 从 DOM 更新输入内容
  ContentEditableInput.prototype.updateFromDOM = function () {
      var this$1 = this;

    # 如果编辑器是只读的，或者无法获取内容，则执行注册更改的操作
    if (this.cm.isReadOnly() || !this.pollContent())
      { runInOp(this.cm, function () { return regChange(this$1.cm); }); }
  };

  # 设置不可编辑
  ContentEditableInput.prototype.setUneditable = function (node) {
    node.contentEditable = "false";
  };

  # 处理按键事件
  ContentEditableInput.prototype.onKeyPress = function (e) {
    # 如果按键码为0或者正在输入中，则不执行操作
    if (e.charCode == 0 || this.composing) { return }
    # 阻止默认行为
    e.preventDefault();
    # 如果编辑器不是只读的，则执行应用文本输入的操作
    if (!this.cm.isReadOnly())
      { operation(this.cm, applyTextInput)(this.cm, String.fromCharCode(e.charCode == null ? e.keyCode : e.charCode), 0); }
  };

  # 只读状态改变时的处理
  ContentEditableInput.prototype.readOnlyChanged = function (val) {
    this.div.contentEditable = String(val != "nocursor");
  };

  # 右键菜单事件处理
  ContentEditableInput.prototype.onContextMenu = function () {};
  # 重置位置
  ContentEditableInput.prototype.resetPosition = function () {};

  # 需要内容属性
  ContentEditableInput.prototype.needsContentAttribute = true;

  # 将位置转换为 DOM
  function posToDOM(cm, pos) {
    var view = findViewForLine(cm, pos.line);
    # 如果视图不存在或者被隐藏，则返回空
    if (!view || view.hidden) { return null }
    var line = getLine(cm.doc, pos.line);
    var info = mapFromLineView(view, line, pos.line);

    var order = getOrder(line, cm.doc.direction), side = "left";
    if (order) {
      var partPos = getBidiPartAt(order, pos.ch);
      side = partPos % 2 ? "right" : "left";
    }
    # 获取节点和偏移量
    var result = nodeAndOffsetInLineMap(info.map, pos.ch, side);
    // 如果折叠方向为右，则偏移量为结束位置，否则为起始位置
    result.offset = result.collapse == "right" ? result.end : result.start;
    // 返回结果对象
    return result
  }

  // 判断节点是否在 gutter 区域
  function isInGutter(node) {
    for (var scan = node; scan; scan = scan.parentNode)
      { if (/CodeMirror-gutter-wrapper/.test(scan.className)) { return true } }
    return false
  }

  // 如果存在错误，则标记位置为 bad
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
      // 遍历 DOM 节点
      if (node.nodeType == 1) {
        // 如果节点类型为元素节点
        var cmText = node.getAttribute("cm-text");
        // 获取节点的 cm-text 属性
        if (cmText) {
          // 如果存在 cm-text 属性
          addText(cmText);
          // 添加文本内容
          return
        }
        var markerID = node.getAttribute("cm-marker"), range;
        // 获取节点的 cm-marker 属性
        if (markerID) {
          // 如果存在 cm-marker 属性
          var found = cm.findMarks(Pos(fromLine, 0), Pos(toLine + 1, 0), recognizeMarker(+markerID));
          // 查找标记
          if (found.length && (range = found[0].find(0)))
            { addText(getBetween(cm.doc, range.from, range.to).join(lineSep)); }
          // 如果找到标记，则添加文本内容
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
        // 如果是块级元素，则设置关闭为 true
      } else if (node.nodeType == 3) {
        // 如果节点类型为文本节点
        addText(node.nodeValue.replace(/\u200b/g, "").replace(/\u00a0/g, " "));
        // 添加文本内容，替换特殊字符
      }
    }
    for (;;) {
      walk(from);
      // 遍历起始节点
      if (from == to) { break }
      from = from.nextSibling;
      // 获取下一个兄弟节点
      extraLinebreak = false;
    }
    return text
    // 返回文本内容
    }
    
    function domToPos(cm, node, offset) {
      var lineNode;
      // 定义变量 lineNode
      if (node == cm.display.lineDiv) {
        // 如果节点等于编辑器的行 div
        lineNode = cm.display.lineDiv.childNodes[offset];
        // 获取行 div 的子节点
        if (!lineNode) { return badPos(cm.clipPos(Pos(cm.display.viewTo - 1)), true) }
        // 如果行节点不存在，则返回错误位置
        node = null; offset = 0;
      } else {
        for (lineNode = node;; lineNode = lineNode.parentNode) {
          // 循环遍历节点的父节点
          if (!lineNode || lineNode == cm.display.lineDiv) { return null }
          // 如果行节点不存在或者等于编辑器的行 div，则返回 null
          if (lineNode.parentNode && lineNode.parentNode == cm.display.lineDiv) { break }
          // 如果行节点的父节点存在且等于编辑器的行 div，则跳出循环
        }
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
    // 获取 lineView 的文本包裹节点
    var wrapper = lineView.text.firstChild, bad = false;
    // 如果 node 为空或不包含在 wrapper 中，则返回错误位置
    if (!node || !contains(wrapper, node)) { return badPos(Pos(lineNo(lineView.line), 0), true) }
    // 如果 node 等于 wrapper，则进行一些处理
    if (node == wrapper) {
      bad = true;
      // 获取 wrapper 的子节点和偏移量
      node = wrapper.childNodes[offset];
      offset = 0;
      // 如果 node 为空，则返回最后一个位置
      if (!node) {
        var line = lineView.rest ? lst(lineView.rest) : lineView.line;
        return badPos(Pos(lineNo(line), line.text.length), bad)
      }
    }

    // 初始化 textNode 和 topNode
    var textNode = node.nodeType == 3 ? node : null, topNode = node;
    // 如果 node 不是文本节点且只有一个子节点且子节点是文本节点，则进行一些处理
    if (!textNode && node.childNodes.length == 1 && node.firstChild.nodeType == 3) {
      textNode = node.firstChild;
      if (offset) { offset = textNode.nodeValue.length; }
    }
    // 循环直到 topNode 的父节点等于 wrapper
    while (topNode.parentNode != wrapper) { topNode = topNode.parentNode; }
    // 获取 measure 和 maps
    var measure = lineView.measure, maps = measure.maps;

    // 定义 find 函数，用于查找节点在行中的位置
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
    // 调用 find 函数查找节点的位置
    var found = find(textNode, topNode, offset);
    // 如果找到位置，则返回该位置，否则返回错误位置
    if (found) { return badPos(found, bad) }

    // FIXME this is all really shaky. might handle the few cases it needs to handle, but likely to cause problems
    # 遍历 topNode 后面的兄弟节点，计算文本节点的偏移量
    for (var after = topNode.nextSibling, dist = textNode ? textNode.nodeValue.length - offset : 0; after; after = after.nextSibling) {
      # 在兄弟节点中查找目标位置
      found = find(after, after.firstChild, 0);
      # 如果找到目标位置，返回错误位置
      if (found)
        { return badPos(Pos(found.line, found.ch - dist), bad) }
      # 如果未找到目标位置，更新偏移量
      else
        { dist += after.textContent.length; }
    }
    # 遍历 topNode 前面的兄弟节点，计算文本节点的偏移量
    for (var before = topNode.previousSibling, dist$1 = offset; before; before = before.previousSibling) {
      # 在兄弟节点中查找目标位置
      found = find(before, before.firstChild, -1);
      # 如果找到目标位置，返回错误位置
      if (found)
        { return badPos(Pos(found.line, found.ch + dist$1), bad) }
      # 如果未找到目标位置，更新偏移量
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
    # 用于轮询的自重置超时
    this.polling = new Delayed();
    # 用于解决 IE 中失去焦点时选择被忘记的问题
    this.hasSelection = false;
    this.composing = null;
  };

  # 初始化 TextareaInput 实例
  TextareaInput.prototype.init = function (display) {
      var this$1 = this;

    var input = this, cm = this.cm;
    # 创建文本输入框
    this.createField(display);
    var te = this.textarea;

    # 将文本输入框插入到显示区域中
    display.wrapper.insertBefore(this.wrapper, display.wrapper.firstChild);

    # 在输入框输入时触发的事件
    on(te, "input", function () {
      # 处理 IE 中的选择被忘记的问题
      if (ie && ie_version >= 9 && this$1.hasSelection) { this$1.hasSelection = null; }
      # 轮询输入
      input.poll();
    });

    # 在粘贴事件发生时触发的事件
    on(te, "paste", function (e) {
      # 如果事件被处理或者粘贴事件被处理，返回
      if (signalDOMEvent(cm, e) || handlePaste(e, cm)) { return }
      # 记录粘贴事件发生的时间
      cm.state.pasteIncoming = +new Date;
      # 快速轮询输入
      input.fastPoll();
    });
    // 准备复制或剪切操作的处理函数
    function prepareCopyCut(e) {
      // 如果是 DOM 事件则返回
      if (signalDOMEvent(cm, e)) { return }
      // 如果有选中内容
      if (cm.somethingSelected()) {
        // 设置最后复制的内容为非行级别，获取选中的文本
        setLastCopied({lineWise: false, text: cm.getSelections()});
      } else if (!cm.options.lineWiseCopyCut) {
        // 如果不支持行级别复制剪切则返回
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
      // 如果是剪切操作，则记录剪切的时间
      if (e.type == "cut") { cm.state.cutIncoming = +new Date; }
    }
    // 绑定复制和剪切事件处理函数
    on(te, "cut", prepareCopyCut);
    on(te, "copy", prepareCopyCut);

    // 粘贴事件处理函数
    on(display.scroller, "paste", function (e) {
      // 如果在小部件内或是 DOM 事件则返回
      if (eventInWidget(display, e) || signalDOMEvent(cm, e)) { return }
      // 如果不支持 dispatchEvent
      if (!te.dispatchEvent) {
        // 记录粘贴的时间，聚焦输入框
        cm.state.pasteIncoming = +new Date;
        input.focus();
        return
      }

      // 将 `paste` 事件传递给文本输入框，以便由其事件监听器处理
      var event = new Event("paste");
      event.clipboardData = e.clipboardData;
      te.dispatchEvent(event);
    });

    // 阻止编辑器内的正常选择（我们自己处理）
    on(display.lineSpace, "selectstart", function (e) {
      // 如果不在小部件内，则阻止默认事件
      if (!eventInWidget(display, e)) { e_preventDefault(e); }
    });

    // 开始输入法编辑事件处理函数
    on(te, "compositionstart", function () {
      // 获取光标起始位置
      var start = cm.getCursor("from");
      // 如果正在输入法编辑，则清除输入法编辑范围
      if (input.composing) { input.composing.range.clear(); }
      // 设置正在输入法编辑的范围
      input.composing = {
        start: start,
        range: cm.markText(start, cm.getCursor("to"), {className: "CodeMirror-composing"})
      };
    });
    // 结束输入法编辑事件处理函数
    on(te, "compositionend", function () {
      // 如果正在输入法编辑
      if (input.composing) {
        // 轮询输入
        input.poll();
        // 清除输入法编辑范围
        input.composing.range.clear();
        input.composing = null;
      }
    });
  };

  // 创建输入框
  TextareaInput.prototype.createField = function (_display) {
    // 包装并隐藏输入文本框
    // 创建一个隐藏的文本域，用于在编辑器聚焦时接收输入
    this.wrapper = hiddenTextarea();
    // 获取隐藏文本域的第一个子元素，即实际的文本域
    this.textarea = this.wrapper.firstChild;
  };

  // 更改屏幕阅读器的标签，提高可访问性
  TextareaInput.prototype.screenReaderLabelChanged = function (label) {
    // 如果有标签，则设置文本域的 aria-label 属性
    if(label) {
      this.textarea.setAttribute('aria-label', label);
    } else {
      // 否则移除文本域的 aria-label 属性
      this.textarea.removeAttribute('aria-label');
    }
  };

  // 准备重新绘制选择和/或光标
  TextareaInput.prototype.prepareSelection = function () {
    // 获取编辑器、显示区域和文档对象
    var cm = this.cm, display = cm.display, doc = cm.doc;
    // 准备选择
    var result = prepareSelection(cm);

    // 如果选项中设置了移动输入光标，则将隐藏的文本域移动到光标附近，以防止滚动效果
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

  // 显示选择内容
  TextareaInput.prototype.showSelection = function (drawn) {
    var cm = this.cm, display = cm.display;
    // 移除光标和选择内容，然后添加新的光标和选择内容
    removeChildrenAndAdd(display.cursorDiv, drawn.cursors);
    removeChildrenAndAdd(display.selectionDiv, drawn.selection);
    // 如果有绘制的位置信息，则设置隐藏文本域的位置
    if (drawn.teTop != null) {
      this.wrapper.style.top = drawn.teTop + "px";
      this.wrapper.style.left = drawn.teLeft + "px";
    }
  };

  // 重置输入以匹配选择（或为空，当未输入且未选择任何内容时）
  TextareaInput.prototype.reset = function (typing) {
    // 如果上下文菜单挂起或正在输入中，则直接返回
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
      # 如果是 IE 并且版本大于等于 9，则记录选中的内容
      if (ie && ie_version >= 9) { this.hasSelection = content; }
    } 
    # 如果没有选中内容并且不是在输入状态
    else if (!typing) {
      # 重置之前的输入内容和文本框的值
      this.prevInput = this.textarea.value = "";
      # 如果是 IE 并且版本大于等于 9，则清空选中的内容
      if (ie && ie_version >= 9) { this.hasSelection = null; }
    }
  };

  # 获取文本框对象
  TextareaInput.prototype.getField = function () { return this.textarea };

  # 判断是否支持触摸操作
  TextareaInput.prototype.supportsTouch = function () { return false };

  # 设置文本框获取焦点
  TextareaInput.prototype.focus = function () {
    # 如果编辑器不是只读模式并且不是在移动设备上或者当前活动元素不是文本框，则尝试让文本框获取焦点
    if (this.cm.options.readOnly != "nocursor" && (!mobile || activeElt() != this.textarea)) {
      try { this.textarea.focus(); }
      catch (e) {} # 如果是 IE8 并且文本框的显示属性为 none 或者不在 DOM 中，则会抛出异常
    }
  };

  # 设置文本框失去焦点
  TextareaInput.prototype.blur = function () { this.textarea.blur(); };

  # 重置文本框位置
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
    # 标记为正在快速轮询
    input.pollingFast = true;
    # 定义轮询函数
    function p() {
      # 执行轮询操作，并记录是否有内容改变
      var changed = input.poll();
      # 如果没有内容改变并且之前没有错过轮询，则继续快速轮询
      if (!changed && !missed) {missed = true; input.polling.set(60, p);}
      # 否则，标记为不再快速轮询，并继续慢速轮询
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
    // 找到实际新的输入部分
    var same = 0, l = Math.min(prevInput.length, text.length);
    while (same < l && prevInput.charCodeAt(same) == text.charCodeAt(same)) { ++same; }
    // 在操作中运行代码块
    runInOp(cm, function () {
      // 应用输入文本到 CodeMirror 编辑器
      applyTextInput(cm, text.slice(same), prevInput.length - same,
                     null, this$1.composing ? "*compose" : null);

      // 如果文本长度超过1000或者包含换行符，则清空输入框
      if (text.length > 1000 || text.indexOf("\n") > -1) { input.value = this$1.prevInput = ""; }
      // 否则将当前文本保存到 prevInput 中
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
    // 如果是 IE 并且版本大于等于 9，则重置选择状态
    if (ie && ie_version >= 9) { this.hasSelection = null; }
    // 快速轮询
    this.fastPoll();
  };

  // 处理右键菜单事件
  TextareaInput.prototype.onContextMenu = function (e) {
    var input = this, cm = input.cm, display = cm.display, te = input.textarea;
    // 如果右键菜单事件挂起，则执行挂起操作
    if (input.contextMenuPending) { input.contextMenuPending(); }
    // 获取鼠标位置和滚动位置
    var pos = posFromMouse(cm, e), scrollPos = display.scroller.scrollTop;
    // 如果位置不存在或者是 Opera 浏览器，则返回
    if (!pos || presto) { return } // Opera is difficult.

    // 如果 'resetSelectionOnContextMenu' 选项为真且点击位置不在当前选择范围内，则重置选择状态
    var reset = cm.options.resetSelectionOnContextMenu;
    if (reset && cm.doc.sel.contains(pos) == -1)
      { operation(cm, setSelection)(cm.doc, simpleSelection(pos), sel_dontScroll); }

    // 保存旧的 CSS 样式，获取输入框的位置信息
    var oldCSS = te.style.cssText, oldWrapperCSS = input.wrapper.style.cssText;
    var wrapperBox = input.wrapper.offsetParent.getBoundingClientRect();
    // 设置输入框的样式为静态定位
    input.wrapper.style.cssText = "position: static";
    // 设置文本编辑器样式，包括位置、大小、背景等
    te.style.cssText = "position: absolute; width: 30px; height: 30px;\n      top: " + (e.clientY - wrapperBox.top - 5) + "px; left: " + (e.clientX - wrapperBox.left - 5) + "px;\n      z-index: 1000; background: " + (ie ? "rgba(255, 255, 255, .05)" : "transparent") + ";\n      outline: none; border-width: 0; outline: none; overflow: hidden; opacity: .05; filter: alpha(opacity=5);";
    // 保存旧的滚动位置，用于解决 Chrome 的问题
    var oldScrollY;
    if (webkit) { oldScrollY = window.scrollY; } // 解决 Chrome 问题 (#2712)
    // 让输入框获得焦点
    display.input.focus();
    // 在 Chrome 中恢复滚动位置
    if (webkit) { window.scrollTo(null, oldScrollY); }
    // 重置输入框
    display.input.reset();
    // 在 Firefox 中添加 "Select all" 到右键菜单
    if (!cm.somethingSelected()) { te.value = input.prevInput = " "; }
    // 设置上下文菜单挂起状态
    input.contextMenuPending = rehide;
    // 保存上下文菜单的选择状态
    display.selForContextMenu = cm.doc.sel;
    // 清除检测全选的定时器
    clearTimeout(display.detectingSelectAll);

    // 准备全选的 hack
    function prepareSelectAllHack() {
      // 如果支持 selectionStart 属性
      if (te.selectionStart != null) {
        // 检查是否有选中内容
        var selected = cm.somethingSelected();
        // 添加零宽空格以便后续检查是否被选中
        var extval = "\u200b" + (selected ? te.value : "");
        // 设置输入框的值，用于捕获右键菜单的撤销操作
        te.value = "\u21da";
        te.value = extval;
        // 设置输入框的选中范围
        te.selectionStart = 1; te.selectionEnd = extval.length;
        // 重新设置上下文菜单的选择状态
        display.selForContextMenu = cm.doc.sel;
      }
    }
    # 定义函数 rehide
    function rehide() {
      # 如果 input.contextMenuPending 不等于 rehide，则返回
      if (input.contextMenuPending != rehide) { return }
      # 将 input.contextMenuPending 设置为 false
      input.contextMenuPending = false;
      # 恢复 input.wrapper 和 te 的 CSS 样式
      input.wrapper.style.cssText = oldWrapperCSS;
      te.style.cssText = oldCSS;
      # 如果是 IE 并且版本小于 9，则设置滚动条位置
      if (ie && ie_version < 9) { display.scrollbars.setScrollTop(display.scroller.scrollTop = scrollPos); }

      # 尝试检测用户是否选择了全选
      if (te.selectionStart != null) {
        # 如果不是 IE 或者是 IE 并且版本小于 9，则准备全选的 hack
        if (!ie || (ie && ie_version < 9)) { prepareSelectAllHack(); }
        var i = 0, poll = function () {
          if (display.selForContextMenu == cm.doc.sel && te.selectionStart == 0 &&
              te.selectionEnd > 0 && input.prevInput == "\u200b") {
            operation(cm, selectAll)(cm);
          } else if (i++ < 10) {
            display.detectingSelectAll = setTimeout(poll, 500);
          } else {
            display.selForContextMenu = null;
            display.input.reset();
          }
        };
        display.detectingSelectAll = setTimeout(poll, 200);
      }
    }

    # 如果是 IE 并且版本大于等于 9，则准备全选的 hack
    if (ie && ie_version >= 9) { prepareSelectAllHack(); }
    # 如果 captureRightClick 为真
    if (captureRightClick) {
      # 阻止默认事件
      e_stop(e);
      # 定义 mouseup 函数
      var mouseup = function () {
        off(window, "mouseup", mouseup);
        setTimeout(rehide, 20);
      };
      on(window, "mouseup", mouseup);
    } else {
      # 延迟 50 毫秒后执行 rehide 函数
      setTimeout(rehide, 50);
    }
  };

  # 定义 TextareaInput 原型的 readOnlyChanged 方法
  TextareaInput.prototype.readOnlyChanged = function (val) {
    # 如果 val 为假，则重置输入框
    if (!val) { this.reset(); }
    # 如果 val 为 "nocursor"，则禁用输入框
    this.textarea.disabled = val == "nocursor";
  };

  # 定义 TextareaInput 原型的 setUneditable 方法
  TextareaInput.prototype.setUneditable = function () {};

  # 设置 TextareaInput 原型的 needsContentAttribute 属性为假
  TextareaInput.prototype.needsContentAttribute = false;

  # 定义 fromTextArea 函数，接受一个 textarea 和 options 参数
  function fromTextArea(textarea, options) {
    # 如果 options 存在，则复制一份
    options = options ? copyObj(options) : {};
    # 设置 options 的 value 为 textarea 的值
    options.value = textarea.value;
    # 如果 options 中没有设置 tabindex 且 textarea 有设置 tabIndex，则设置 options 的 tabindex 为 textarea 的 tabIndex
    if (!options.tabindex && textarea.tabIndex)
      { options.tabindex = textarea.tabIndex; }
    # 如果 options 中没有设置 placeholder 且 textarea 有设置 placeholder，则设置 options 的 placeholder 为 textarea 的 placeholder
    if (!options.placeholder && textarea.placeholder)
      { options.placeholder = textarea.placeholder; }
    # 设置 autofocus 为真，如果 textarea 被聚焦，或者它有
    // 如果 options.autofocus 为 null，则判断当前是否有焦点元素，如果没有则设置 options.autofocus 为 true
    if (options.autofocus == null) {
      var hasFocus = activeElt();
      options.autofocus = hasFocus == textarea ||
        textarea.getAttribute("autofocus") != null && hasFocus == document.body;
    }

    // 定义保存函数，将 CodeMirror 编辑器中的内容保存到 textarea 中
    function save() {textarea.value = cm.getValue();}

    // 如果 textarea 所在的表单存在，则在表单提交时执行保存函数
    if (textarea.form) {
      on(textarea.form, "submit", save);
      // 用于修复表单提交方法的 hack
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

    // 完成初始化后执行的操作，包括保存函数的绑定和相关方法的定义
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

    // 隐藏原始的 textarea 元素，创建 CodeMirror 编辑器并返回
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

  # 定义选项方法
  defineOptions(CodeMirror);

  # 添加编辑器方法
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
    // 调用 defineMode 方法
    defineMode.apply(this, arguments);
  };

  // 定义 MIME 类型
  CodeMirror.defineMIME = defineMIME;

  // 最小的默认模式
 // 定义一个名为 "null" 的最小默认模式
  CodeMirror.defineMode("null", function () { return ({token: function (stream) { return stream.skipToEnd(); }}); });
  // 定义 MIME 类型为 "text/plain" 的最小默认模式
  CodeMirror.defineMIME("text/plain", "null");

  // 扩展

  // 定义扩展方法
  CodeMirror.defineExtension = function (name, func) {
    // 为 CodeMirror 对象添加扩展方法
    CodeMirror.prototype[name] = func;
  };
  // 定义文档扩展方法
  CodeMirror.defineDocExtension = function (name, func) {
    // 为 Doc 对象添加扩展方法
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
// 定义一个匿名函数，传入一个模块参数
(function(mod) {
  // 如果是 CommonJS 环境，使用 require 引入 CodeMirror 模块
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境，使用 define 引入 CodeMirror 模块
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境，直接使用 CodeMirror 模块
  else
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义一个简单模式的函数
  CodeMirror.defineSimpleMode = function(name, states) {
    // 定义一个模式，返回一个简单模式的配置
    CodeMirror.defineMode(name, function(config) {
      return CodeMirror.simpleMode(config, states);
    });
  };

  // 定义一个简单模式的函数
  CodeMirror.simpleMode = function(config, states) {
    // 确保状态中包含起始状态
    ensureState(states, "start");
    var states_ = {}, meta = states.meta || {}, hasIndentation = false;
    // 遍历状态
    for (var state in states) if (state != meta && states.hasOwnProperty(state)) {
      var list = states_[state] = [], orig = states[state];
      // 遍历状态中的规则
      for (var i = 0; i < orig.length; i++) {
        var data = orig[i];
        // 将规则封装成 Rule 对象，并添加到列表中
        list.push(new Rule(data, states));
        // 如果规则中包含缩进或减少缩进的操作，设置标志位为 true
        if (data.indent || data.dedent) hasIndentation = true;
      }
    }
    # 定义一个名为 mode 的对象，包含了 startState、copyState、token、innerMode 和 indent 方法
    var mode = {
      # 定义 startState 方法，返回初始状态对象
      startState: function() {
        return {state: "start", pending: null,
                local: null, localState: null,
                indent: hasIndentation ? [] : null};
      },
      # 定义 copyState 方法，用于复制状态对象
      copyState: function(state) {
        var s = {state: state.state, pending: state.pending,
                 local: state.local, localState: null,
                 indent: state.indent && state.indent.slice(0)};
        if (state.localState)
          s.localState = CodeMirror.copyState(state.local.mode, state.localState);
        if (state.stack)
          s.stack = state.stack.slice(0);
        for (var pers = state.persistentStates; pers; pers = pers.next)
          s.persistentStates = {mode: pers.mode,
                                spec: pers.spec,
                                state: pers.state == state.localState ? s.localState : CodeMirror.copyState(pers.mode, pers.state),
                                next: s.persistentStates};
        return s;
      },
      # 定义 token 方法，用于生成 token
      token: tokenFunction(states_, config),
      # 定义 innerMode 方法，返回内部模式对象
      innerMode: function(state) { return state.local && {mode: state.local.mode, state: state.localState}; },
      # 定义 indent 方法，用于缩进
      indent: indentFunction(states_, meta)
    };
    # 如果存在 meta 对象，则将其属性添加到 mode 对象中
    if (meta) for (var prop in meta) if (meta.hasOwnProperty(prop))
      mode[prop] = meta[prop];
    # 返回 mode 对象
    return mode;
  };

  # 定义 ensureState 方法，用于确保状态存在
  function ensureState(states, name) {
    if (!states.hasOwnProperty(name))
      throw new Error("Undefined state " + name + " in simple mode");
  }

  # 定义 toRegex 方法，用于将输入值转换为正则表达式
  function toRegex(val, caret) {
    if (!val) return /(?:)/;
    var flags = "";
    if (val instanceof RegExp) {
      if (val.ignoreCase) flags = "i";
      val = val.source;
    } else {
      val = String(val);
    }
    return new RegExp((caret === false ? "" : "^") + "(?:" + val + ")", flags);
  }

  # 定义 asToken 方法，用于将输入值转换为 token
  function asToken(val) {
    if (!val) return null;
    if (val.apply) return val
    if (typeof val == "string") return val.replace(/\./g, " ");
    var result = [];
  // 遍历数组 val，将每个元素经过替换操作后加入到结果数组 result 中
  for (var i = 0; i < val.length; i++)
    result.push(val[i] && val[i].replace(/\./g, " "));
  // 返回结果数组
  return result;
}

// Rule 对象的构造函数，根据传入的 data 和 states 创建 Rule 对象
function Rule(data, states) {
  // 如果 data 中有 next 或 push 属性，则确保 states 中存在对应的状态
  if (data.next || data.push) ensureState(states, data.next || data.push);
  // 将 data 中的 regex 转换为正则表达式，并赋值给 this.regex
  this.regex = toRegex(data.regex);
  // 将 data 中的 token 转换为 token 对象，并赋值给 this.token
  this.token = asToken(data.token);
  // 将 data 赋值给 this.data
  this.data = data;
}

// tokenFunction 函数，接受 states 和 config 作为参数
function tokenFunction(states, config) {
  // 空函数体
};

// 比较函数 cmp，用于比较两个对象是否相等
function cmp(a, b) {
  // 如果 a 与 b 相等，则返回 true
  if (a === b) return true;
  // 如果 a 或 b 为空，或者不是对象，则返回 false
  if (!a || typeof a != "object" || !b || typeof b != "object") return false;
  // 初始化属性计数器 props
  var props = 0;
  // 遍历对象 a 的属性
  for (var prop in a) if (a.hasOwnProperty(prop)) {
    // 如果 b 中不包含属性 prop，或者 a[prop] 与 b[prop] 不相等，则返回 false
    if (!b.hasOwnProperty(prop) || !cmp(a[prop], b[prop])) return false;
    // 属性计数器加一
    props++;
  }
  // 遍历对象 b 的属性
  for (var prop in b) if (b.hasOwnProperty(prop)) props--;
  // 返回属性计数器是否为 0
  return props == 0;
}

// 进入局部模式的函数，接受 config、state、spec 和 token 作为参数
function enterLocalMode(config, state, spec, token) {
  // 声明局部变量 pers
  var pers;
  // 如果 spec 中包含 persistent 属性，则遍历 state.persistentStates，查找对应的状态
  if (spec.persistent) for (var p = state.persistentStates; p && !pers; p = p.next)
    // 如果 spec.spec 存在，则比较 spec.spec 和 p.spec，否则比较 spec.mode 和 p.mode
    if (spec.spec ? cmp(spec.spec, p.spec) : spec.mode == p.mode) pers = p;
  // 根据 pers 是否存在，确定 mode 和 lState 的值
  var mode = pers ? pers.mode : spec.mode || CodeMirror.getMode(config, spec.spec);
  var lState = pers ? pers.state : CodeMirror.startState(mode);
  // 如果 spec 中包含 persistent 属性，并且 pers 不存在，则将 mode、spec.spec 和 lState 添加到 state.persistentStates 中
  if (spec.persistent && !pers)
    state.persistentStates = {mode: mode, spec: spec.spec, state: lState, next: state.persistentStates};

  // 设置 state.localState 为 lState
  state.localState = lState;
  // 设置 state.local 对象的属性
  state.local = {mode: mode,
                 end: spec.end && toRegex(spec.end),
                 endScan: spec.end && spec.forceEnd !== false && toRegex(spec.end, false),
                 endToken: token && token.join ? token[token.length - 1] : token};
}

// 查找函数 indexOf，用于在数组 arr 中查找元素 val
function indexOf(val, arr) {
  // 遍历数组 arr，如果找到元素 val，则返回 true
  for (var i = 0; i < arr.length; i++) if (arr[i] === val) return true;
}

// 缩进函数 indentFunction，接受 states 和 meta 作为参数
    # 定义一个函数，用于处理缩进
    return function(state, textAfter, line) {
      # 如果存在本地状态并且本地状态有缩进模式，则使用本地状态的缩进模式
      if (state.local && state.local.mode.indent)
        return state.local.mode.indent(state.localState, textAfter, line);
      # 如果缩进为空或者存在本地状态或者当前状态在不缩进状态列表中，则返回 CodeMirror.Pass
      if (state.indent == null || state.local || meta.dontIndentStates && indexOf(state.state, meta.dontIndentStates) > -1)
        return CodeMirror.Pass;

      # 初始化位置和规则
      var pos = state.indent.length - 1, rules = states[state.state];
      # 循环扫描规则
      scan: for (;;) {
        for (var i = 0; i < rules.length; i++) {
          var rule = rules[i];
          # 如果规则包含取消缩进并且不是行首，则执行以下操作
          if (rule.data.dedent && rule.data.dedentIfLineStart !== false) {
            # 使用正则表达式匹配文本
            var m = rule.regex.exec(textAfter);
            # 如果匹配成功
            if (m && m[0]) {
              # 位置减一
              pos--;
              # 如果存在下一个状态或者推入状态，则更新规则
              if (rule.next || rule.push) rules = states[rule.next || rule.push];
              # 更新文本内容
              textAfter = textAfter.slice(m[0].length);
              # 继续扫描
              continue scan;
            }
          }
        }
        # 结束扫描
        break;
      }
      # 如果位置小于0，则返回0，否则返回对应位置的缩进值
      return pos < 0 ? 0 : state.indent[pos];
    };
  }
// 闭合大括号
});


/* ---- extension/sublime.js ---- */


// CodeMirror, 版权归 Marijn Haverbeke 和其他人所有
// 根据 MIT 许可证分发：https://codemirror.net/LICENSE

// Sublime Text 键绑定的粗略近似
// 依赖于 addon/search/searchcursor.js，可选依赖于 addon/dialog/dialogs.js

(function(mod) {
  if (typeof exports == "object" && typeof module == "object") // CommonJS
    mod(require("../lib/codemirror"), require("../addon/search/searchcursor"), require("../addon/edit/matchbrackets"));
  else if (typeof define == "function" && define.amd) // AMD
    define(["../lib/codemirror", "../addon/search/searchcursor", "../addon/edit/matchbrackets"], mod);
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  var cmds = CodeMirror.commands;
  var Pos = CodeMirror.Pos;

  // 这并不完全是 Sublime 的算法。我无法理解那个。
  function findPosSubword(doc, start, dir) {
    if (dir < 0 && start.ch == 0) return doc.clipPos(Pos(start.line - 1));
    var line = doc.getLine(start.line);
    if (dir > 0 && start.ch >= line.length) return doc.clipPos(Pos(start.line + 1, 0));
    var state = "start", type, startPos = start.ch;
    for (var pos = startPos, e = dir < 0 ? 0 : line.length, i = 0; pos != e; pos += dir, i++) {
      var next = line.charAt(dir < 0 ? pos - 1 : pos);
      var cat = next != "_" && CodeMirror.isWordChar(next) ? "w" : "o";
      if (cat == "w" && next.toUpperCase() == next) cat = "W";
      if (state == "start") {
        if (cat != "o") { state = "in"; type = cat; }
        else startPos = pos + dir
      } else if (state == "in") {
        if (type != cat) {
          if (type == "w" && cat == "W" && dir < 0) pos--;
          if (type == "W" && cat == "w" && dir > 0) { // 从大写转换为小写
            if (pos == startPos + 1) { type = "w"; continue; }
            else pos--;
          }
          break;
        }
      }
    }
  // 返回一个包含指定行和位置的位置对象
  return Pos(start.line, pos);
}

function moveSubword(cm, dir) {
  // 通过扩展选择范围的方式移动光标到下一个子词
  cm.extendSelectionsBy(function(range) {
    if (cm.display.shift || cm.doc.extend || range.empty())
      // 如果按下了 shift 键或者正在扩展选择范围，或者选择范围为空，则查找下一个子词的位置
      return findPosSubword(cm.doc, range.head, dir);
    else
      // 否则，根据方向返回当前选择范围的起始或结束位置
      return dir < 0 ? range.from() : range.to();
  });
}

// 向左移动光标到下一个子词
cmds.goSubwordLeft = function(cm) { moveSubword(cm, -1); };
// 向右移动光标到下一个子词
cmds.goSubwordRight = function(cm) { moveSubword(cm, 1); };

cmds.scrollLineUp = function(cm) {
  var info = cm.getScrollInfo();
  if (!cm.somethingSelected()) {
    var visibleBottomLine = cm.lineAtHeight(info.top + info.clientHeight, "local");
    if (cm.getCursor().line >= visibleBottomLine)
      cm.execCommand("goLineUp");
  }
  // 向上滚动一行
  cm.scrollTo(null, info.top - cm.defaultTextHeight());
};
cmds.scrollLineDown = function(cm) {
  var info = cm.getScrollInfo();
  if (!cm.somethingSelected()) {
    var visibleTopLine = cm.lineAtHeight(info.top, "local")+1;
    if (cm.getCursor().line <= visibleTopLine)
      cm.execCommand("goLineDown");
  }
  // 向下滚动一行
  cm.scrollTo(null, info.top + cm.defaultTextHeight());
};

cmds.splitSelectionByLine = function(cm) {
  var ranges = cm.listSelections(), lineRanges = [];
  for (var i = 0; i < ranges.length; i++) {
    var from = ranges[i].from(), to = ranges[i].to();
    for (var line = from.line; line <= to.line; ++line)
      if (!(to.line > from.line && line == to.line && to.ch == 0))
        lineRanges.push({anchor: line == from.line ? from : Pos(line, 0),
                         head: line == to.line ? to : Pos(line)});
  }
  // 根据行范围设置选择范围
  cm.setSelections(lineRanges, 0);
};

cmds.singleSelectionTop = function(cm) {
  var range = cm.listSelections()[0];
  // 将选择范围设置为单个光标，并将其滚动到可见区域的顶部
  cm.setSelection(range.anchor, range.head, {scroll: false});
};

cmds.selectLine = function(cm) {
  var ranges = cm.listSelections(), extended = [];
  // ...
};
    # 遍历 ranges 数组
    for (var i = 0; i < ranges.length; i++) {
      # 获取当前 range
      var range = ranges[i];
      # 将当前 range 转换为 anchor 和 head 对象，并添加到 extended 数组中
      extended.push({anchor: Pos(range.from().line, 0),
                     head: Pos(range.to().line + 1, 0)});
    }
    # 设置编辑器的选择范围为 extended 数组中的内容
    cm.setSelections(extended);
  };

  # 在当前光标位置上方或下方插入新行
  function insertLine(cm, above) {
    # 如果编辑器为只读状态，则返回 CodeMirror.Pass
    if (cm.isReadOnly()) return CodeMirror.Pass
    # 执行编辑操作
    cm.operation(function() {
      # 获取当前选择范围的长度和新的选择范围数组
      var len = cm.listSelections().length, newSelection = [], last = -1;
      # 遍历当前选择范围
      for (var i = 0; i < len; i++) {
        # 获取当前选择范围的头部位置
        var head = cm.listSelections()[i].head;
        # 如果头部位置在上一个选择范围的行以下，则继续下一次循环
        if (head.line <= last) continue;
        # 计算新行的位置
        var at = Pos(head.line + (above ? 0 : 1), 0);
        # 在新行位置插入换行符
        cm.replaceRange("\n", at, null, "+insertLine");
        # 自动缩进新行
        cm.indentLine(at.line, null, true);
        # 将新行的位置添加到新的选择范围数组中
        newSelection.push({head: at, anchor: at});
        # 更新上一个选择范围的行号
        last = head.line + 1;
      }
      # 设置编辑器的选择范围为新的选择范围数组
      cm.setSelections(newSelection);
    });
    # 执行自动缩进命令
    cm.execCommand("indentAuto");
  }

  # 定义在当前光标位置下方插入新行的命令
  cmds.insertLineAfter = function(cm) { return insertLine(cm, false); };

  # 定义在当前光标位置上方插入新行的命令
  cmds.insertLineBefore = function(cm) { return insertLine(cm, true); };

  # 获取指定位置的单词信息
  function wordAt(cm, pos) {
    # 初始化单词的起始位置和结束位置
    var start = pos.ch, end = start, line = cm.getLine(pos.line);
    # 向前查找单词的起始位置
    while (start && CodeMirror.isWordChar(line.charAt(start - 1))) --start;
    # 向后查找单词的结束位置
    while (end < line.length && CodeMirror.isWordChar(line.charAt(end))) ++end;
    # 返回单词的起始位置、结束位置和内容
    return {from: Pos(pos.line, start), to: Pos(pos.line, end), word: line.slice(start, end)};
  }

  # 定义选择下一个相同单词的命令
  cmds.selectNextOccurrence = function(cm) {
    # 获取当前选择范围的起始位置和结束位置
    var from = cm.getCursor("from"), to = cm.getCursor("to");
    # 判断是否需要选择整个单词
    var fullWord = cm.state.sublimeFindFullWord == cm.doc.sel;
    # 如果起始位置和结束位置相同
    if (CodeMirror.cmpPos(from, to) == 0) {
      # 获取当前位置的单词信息
      var word = wordAt(cm, from);
      # 如果没有找到单词，则返回
      if (!word.word) return;
      # 选择当前单词
      cm.setSelection(word.from, word.to);
      # 设置需要选择整个单词
      fullWord = true;
  } else {
    // 获取选中文本
    var text = cm.getRange(from, to);
    // 创建查询正则表达式
    var query = fullWord ? new RegExp("\\b" + text + "\\b") : text;
    // 获取搜索光标
    var cur = cm.getSearchCursor(query, to);
    // 查找下一个匹配项
    var found = cur.findNext();
    // 如果没有找到，则从文档开头重新查找
    if (!found) {
      cur = cm.getSearchCursor(query, Pos(cm.firstLine(), 0));
      found = cur.findNext();
    }
    // 如果没有找到或者选中范围已经包含了当前匹配项，则返回
    if (!found || isSelectedRange(cm.listSelections(), cur.from(), cur.to())) return
    // 添加选中范围
    cm.addSelection(cur.from(), cur.to());
  }
  // 如果是全词匹配，则保存当前选中状态
  if (fullWord)
    cm.state.sublimeFindFullWord = cm.doc.sel;
};

// 跳过当前匹配项并选择下一个匹配项
cmds.skipAndSelectNextOccurrence = function(cm) {
  // 保存当前光标位置
  var prevAnchor = cm.getCursor("anchor"), prevHead = cm.getCursor("head");
  // 选择下一个匹配项
  cmds.selectNextOccurrence(cm);
  // 如果光标位置发生变化，则更新选择范围
  if (CodeMirror.cmpPos(prevAnchor, prevHead) != 0) {
    cm.doc.setSelections(cm.doc.listSelections()
        .filter(function (sel) {
          return sel.anchor != prevAnchor || sel.head != prevHead;
        }));
  }
}

// 在指定方向上添加光标到选择范围
function addCursorToSelection(cm, dir) {
  var ranges = cm.listSelections(), newRanges = [];
  for (var i = 0; i < ranges.length; i++) {
    var range = ranges[i];
    var newAnchor = cm.findPosV(
        range.anchor, dir, "line", range.anchor.goalColumn);
    var newHead = cm.findPosV(
        range.head, dir, "line", range.head.goalColumn);
    newAnchor.goalColumn = range.anchor.goalColumn != null ?
        range.anchor.goalColumn : cm.cursorCoords(range.anchor, "div").left;
    newHead.goalColumn = range.head.goalColumn != null ?
        range.head.goalColumn : cm.cursorCoords(range.head, "div").left;
    var newRange = {anchor: newAnchor, head: newHead};
    newRanges.push(range);
    newRanges.push(newRange);
  }
  cm.setSelections(newRanges);
}
// 向上添加光标到选择范围
cmds.addCursorToPrevLine = function(cm) { addCursorToSelection(cm, -1); };
// 向下添加光标到选择范围
cmds.addCursorToNextLine = function(cm) { addCursorToSelection(cm, 1); };

// 检查选中范围是否包含指定范围
function isSelectedRange(ranges, from, to) {
  // 遍历 ranges 数组
  for (var i = 0; i < ranges.length; i++)
    // 检查当前 range 是否与给定 from 和 to 相等，如果相等则返回 true
    if (CodeMirror.cmpPos(ranges[i].from(), from) == 0 &&
        CodeMirror.cmpPos(ranges[i].to(), to) == 0) return true
  // 如果没有相等的 range，则返回 false
  return false
}

// 定义 mirror 变量，包含括号和方括号
var mirror = "(){}[]";
// 定义 selectBetweenBrackets 函数，参数为 cm
function selectBetweenBrackets(cm) {
  // 获取当前选中的 ranges
  var ranges = cm.listSelections(), newRanges = []
  // 遍历 ranges 数组
  for (var i = 0; i < ranges.length; i++) {
    // 获取当前 range 的头部位置
    var range = ranges[i], pos = range.head, opening = cm.scanForBracket(pos, -1);
    // 如果没有找到匹配的开括号，则返回 false
    if (!opening) return false;
    // 循环查找匹配的闭括号
    for (;;) {
      var closing = cm.scanForBracket(pos, 1);
      // 如果没有找到匹配的闭括号，则返回 false
      if (!closing) return false;
      // 如果找到匹配的闭括号，则判断是否与对应的开括号匹配
      if (closing.ch == mirror.charAt(mirror.indexOf(opening.ch) + 1)) {
        // 获取起始位置
        var startPos = Pos(opening.pos.line, opening.pos.ch + 1);
        // 如果起始位置和结束位置与 range 的 from 和 to 相等，则继续查找下一个开括号
        if (CodeMirror.cmpPos(startPos, range.from()) == 0 &&
            CodeMirror.cmpPos(closing.pos, range.to()) == 0) {
          opening = cm.scanForBracket(opening.pos, -1);
          // 如果没有找到匹配的开括号，则返回 false
          if (!opening) return false;
        } else {
          // 将匹配的括号范围添加到 newRanges 数组中
          newRanges.push({anchor: startPos, head: closing.pos});
          break;
        }
      }
      // 更新位置为下一个字符
      pos = Pos(closing.pos.line, closing.pos.ch + 1);
    }
  }
  // 设置新的选中范围
  cm.setSelections(newRanges);
  return true;
}

// 定义 selectScope 命令，参数为 cm
cmds.selectScope = function(cm) {
  // 调用 selectBetweenBrackets 函数，如果返回 false 则执行 selectAll 命令
  selectBetweenBrackets(cm) || cm.execCommand("selectAll");
};
// 定义 selectBetweenBrackets 命令，参数为 cm
cmds.selectBetweenBrackets = function(cm) {
  // 如果 selectBetweenBrackets 函数返回 false，则返回 CodeMirror.Pass
  if (!selectBetweenBrackets(cm)) return CodeMirror.Pass;
};

// 定义 puncType 函���，参数为 type
function puncType(type) {
  // 如果 type 为假值，则返回 null，否则返回 type 是否包含 'punctuation' 的正则匹配结果
  return !type ? null : /\bpunctuation\b/.test(type) ? type : undefined
}

// 定义 goToBracket 命令，参数为 cm
cmds.goToBracket = function(cm) {
  // 根据当前位置和类型查找下一个匹配的括号位置
  cm.extendSelectionsBy(function(range) {
    var next = cm.scanForBracket(range.head, 1, puncType(cm.getTokenTypeAt(range.head)));
    // 如果找到下一个匹配的括号位置，则返回该位置
    if (next && CodeMirror.cmpPos(next.pos, range.head) != 0) return next.pos;
    // 否则，查找前一个匹配的括号位置
    var prev = cm.scanForBracket(range.head, -1, puncType(cm.getTokenTypeAt(Pos(range.head.line, range.head.ch + 1))));
    // 如果找到前一个匹配的括号位置，则返回该位置，否则返回当前位置
    return prev && Pos(prev.pos.line, prev.pos.ch + 1) || range.head;
  // 定义 swapLineUp 命令，用于将选中的行向上移动
  cmds.swapLineUp = function(cm) {
    // 如果编辑器是只读的，则返回 CodeMirror.Pass
    if (cm.isReadOnly()) return CodeMirror.Pass
    // 获取选中的文本范围和行数
    var ranges = cm.listSelections(), linesToMove = [], at = cm.firstLine() - 1, newSels = [];
    // 遍历选中的文本范围
    for (var i = 0; i < ranges.length; i++) {
      var range = ranges[i], from = range.from().line - 1, to = range.to().line;
      // 更新新的选择范围
      newSels.push({anchor: Pos(range.anchor.line - 1, range.anchor.ch),
                    head: Pos(range.head.line - 1, range.head.ch)});
      // 如果选中的行不为空且结束位置为0，则将结束位置减1
      if (range.to().ch == 0 && !range.empty()) --to;
      // 判断行是否需要移动，并更新行数
      if (from > at) linesToMove.push(from, to);
      else if (linesToMove.length) linesToMove[linesToMove.length - 1] = to;
      at = to;
    }
    // 执行操作
    cm.operation(function() {
      // 遍历需要移动的行
      for (var i = 0; i < linesToMove.length; i += 2) {
        var from = linesToMove[i], to = linesToMove[i + 1];
        var line = cm.getLine(from);
        // 移动行
        cm.replaceRange("", Pos(from, 0), Pos(from + 1, 0), "+swapLine");
        // 判断是否需要在末尾添加新行
        if (to > cm.lastLine())
          cm.replaceRange("\n" + line, Pos(cm.lastLine()), null, "+swapLine");
        else
          cm.replaceRange(line + "\n", Pos(to, 0), null, "+swapLine");
      }
      // 更新选择范围并滚动视图
      cm.setSelections(newSels);
      cm.scrollIntoView();
    });
  };

  // 定义 swapLineDown 命令，用于将选中的行向下移动
  cmds.swapLineDown = function(cm) {
    // 如果编辑器是只读的，则返回 CodeMirror.Pass
    if (cm.isReadOnly()) return CodeMirror.Pass
    // 获取选中的文本范围和行数
    var ranges = cm.listSelections(), linesToMove = [], at = cm.lastLine() + 1;
    // 倒序遍历选中的文本范围
    for (var i = ranges.length - 1; i >= 0; i--) {
      var range = ranges[i], from = range.to().line + 1, to = range.from().line;
      // 如果选中的行不为空且结束位置为0，则将结束位置减1
      if (range.to().ch == 0 && !range.empty()) from--;
      // 判断行是否需要移动，并更新行数
      if (from < at) linesToMove.push(from, to);
      else if (linesToMove.length) linesToMove[linesToMove.length - 1] = to;
      at = to;
    }
    // 执行操作函数
    cm.operation(function() {
      // 遍历需要移动的行
      for (var i = linesToMove.length - 2; i >= 0; i -= 2) {
        // 获取起始行和目标行
        var from = linesToMove[i], to = linesToMove[i + 1];
        // 获取起始行的内容
        var line = cm.getLine(from);
        // 如果起始行是最后一行，则删除起始行
        if (from == cm.lastLine())
          cm.replaceRange("", Pos(from - 1), Pos(from), "+swapLine");
        // 否则删除起始行到下一行的内容
        else
          cm.replaceRange("", Pos(from, 0), Pos(from + 1, 0), "+swapLine");
        // 在目标行插入起始行的内容
        cm.replaceRange(line + "\n", Pos(to, 0), null, "+swapLine");
      }
      // 滚动视图以确保操作后的内容可见
      cm.scrollIntoView();
    });
  };

  // 切换缩进注释
  cmds.toggleCommentIndented = function(cm) {
    cm.toggleComment({ indent: true });
  }

  // 合并多行为一行
  cmds.joinLines = function(cm) {
    // 获取选中的文本范围
    var ranges = cm.listSelections(), joined = [];
    // 遍历选中的文本范围
    for (var i = 0; i < ranges.length; i++) {
      var range = ranges[i], from = range.from();
      var start = from.line, end = range.to().line;
      // 合并相邻的选中文本范围
      while (i < ranges.length - 1 && ranges[i + 1].from().line == end)
        end = ranges[++i].to().line;
      joined.push({start: start, end: end, anchor: !range.empty() && from});
    }
    // 执行操作函数
    cm.operation(function() {
      var offset = 0, ranges = [];
      // 遍历合并后的文本范围
      for (var i = 0; i < joined.length; i++) {
        var obj = joined[i];
        var anchor = obj.anchor && Pos(obj.anchor.line - offset, obj.anchor.ch), head;
        // 遍历合并后的每一行
        for (var line = obj.start; line <= obj.end; line++) {
          var actual = line - offset;
          // 如果不是最后一行，则删除行尾空白字符并增加偏移量
          if (line == obj.end) head = Pos(actual, cm.getLine(actual).length + 1);
          if (actual < cm.lastLine()) {
            cm.replaceRange(" ", Pos(actual), Pos(actual + 1, /^\s*/.exec(cm.getLine(actual + 1))[0].length));
            ++offset;
          }
        }
        ranges.push({anchor: anchor || head, head: head});
      }
      // 设置新的文本范围
      cm.setSelections(ranges, 0);
    });
  };

  // 复制当前行
  cmds.duplicateLine = function(cm) {
  // 定义操作函数，接受一个匿名函数作为参数
  cm.operation(function() {
    // 获取当前选择的文本范围数量
    var rangeCount = cm.listSelections().length;
    // 遍历每个选择的文本范围
    for (var i = 0; i < rangeCount; i++) {
      // 获取当前文本范围
      var range = cm.listSelections()[i];
      // 如果范围为空，则在范围末尾插入一个换行符
      if (range.empty())
        cm.replaceRange(cm.getLine(range.head.line) + "\n", Pos(range.head.line, 0));
      // 如果范围不为空，则用范围内的文本替换范围起始位置到结束位置的文本
      else
        cm.replaceRange(cm.getRange(range.from(), range.to()), range.from());
    }
    // 滚动视图以确保所有更改都可见
    cm.scrollIntoView();
  });
};

// 定义排序函数，接受编辑器对象和是否区分大小写作为参数
function sortLines(cm, caseSensitive) {
  // 如果编辑器为只读状态，则返回
  if (cm.isReadOnly()) return CodeMirror.Pass
  // 获取选择的文本范围
  var ranges = cm.listSelections(), toSort = [], selected;
  // 遍历每个选择的文本范围
  for (var i = 0; i < ranges.length; i++) {
    var range = ranges[i];
    // 如果范围为空，则继续下一次循环
    if (range.empty()) continue;
    // 获取范围的起始行和结束行
    var from = range.from().line, to = range.to().line;
    // 合并相邻的范围
    while (i < ranges.length - 1 && ranges[i + 1].from().line == to)
      to = ranges[++i].to().line;
    // 如果范围的结束位置为行首，则结束行减一
    if (!ranges[i].to().ch) to--;
    // 将需要排序的范围起始行和结束行添加到数组中
    toSort.push(from, to);
  }
  // 如果存在需要排序的范围，则设置 selected 为 true
  if (toSort.length) selected = true;
  // 否则将整个文档的范围添加到数组中
  else toSort.push(cm.firstLine(), cm.lastLine());

  // 执行操作函数
  cm.operation(function() {
    var ranges = [];
    // 遍历需要排序的范围
    for (var i = 0; i < toSort.length; i += 2) {
      var from = toSort[i], to = toSort[i + 1];
      var start = Pos(from, 0), end = Pos(to);
      // 获取范围内的文本
      var lines = cm.getRange(start, end, false);
      // 如果区分大小写，则直接排序
      if (caseSensitive)
        lines.sort();
      // 否则不区分大小写排序
      else
        lines.sort(function(a, b) {
          var au = a.toUpperCase(), bu = b.toUpperCase();
          if (au != bu) { a = au; b = bu; }
          return a < b ? -1 : a == b ? 0 : 1;
        });
      // 用排序后的文本替换原来的文本
      cm.replaceRange(lines, start, end);
      // 如果存在选择的范围，则将排序后的范围添加到 ranges 数组中
      if (selected) ranges.push({anchor: start, head: Pos(to + 1, 0)});
    }
    // 如果存在选择的范围，则设置编辑器的选择范围为 ranges 数组中的范围
    if (selected) cm.setSelections(ranges, 0);
  });
}

// 定义排序函数的忽略大小写版本
cmds.sortLines = function(cm) { sortLines(cm, true); };
cmds.sortLinesInsensitive = function(cm) { sortLines(cm, false); };

// 定义下一个书签函数
cmds.nextBookmark = function(cm) {
  // 获取编辑器状态中的 sublimeBookmarks
  var marks = cm.state.sublimeBookmarks;
  // 如果存在书签，则循环处理每个书签
  if (marks) while (marks.length) {
    // 从书签数组中取出第一个书签
    var current = marks.shift();
    // 查找当前书签的位置
    var found = current.find();
    // 如果找到了位置
    if (found) {
      // 将当前书签重新放回数组末尾
      marks.push(current);
      // 设置编辑器的选中范围为找到的位置
      return cm.setSelection(found.from, found.to);
    }
  }
};

// 向前查找书签
cmds.prevBookmark = function(cm) {
  // 获取书签数组
  var marks = cm.state.sublimeBookmarks;
  // 如果存在书签，则循环处理每个书签
  if (marks) while (marks.length) {
    // 将最后一个书签移到数组开头
    marks.unshift(marks.pop());
    // 查找最后一个书签的位置
    var found = marks[marks.length - 1].find();
    // 如果未找到位置
    if (!found)
      // 移除最后一个书签
      marks.pop();
    else
      // 设置编辑器的选中范围为找到的位置
      return cm.setSelection(found.from, found.to);
  }
};

// 切换书签状态
cmds.toggleBookmark = function(cm) {
  // 获取当前选中的范围
  var ranges = cm.listSelections();
  // 获取书签数组，如果不存在则初始化为空数组
  var marks = cm.state.sublimeBookmarks || (cm.state.sublimeBookmarks = []);
  // 遍历选中范围
  for (var i = 0; i < ranges.length; i++) {
    var from = ranges[i].from(), to = ranges[i].to();
    // 查找选中范围内的书签
    var found = ranges[i].empty() ? cm.findMarksAt(from) : cm.findMarks(from, to);
    // 遍历找到的书签
    for (var j = 0; j < found.length; j++) {
      // 如果找到了 Sublime 书签
      if (found[j].sublimeBookmark) {
        // 清除该书签
        found[j].clear();
        // 从书签数组中移除该书签
        for (var k = 0; k < marks.length; k++)
          if (marks[k] == found[j])
            marks.splice(k--, 1);
        break;
      }
    }
    // 如果未找到 Sublime 书签
    if (j == found.length)
      // 将选中范围标记为 Sublime 书签
      marks.push(cm.markText(from, to, {sublimeBookmark: true, clearWhenEmpty: false}));
  }
};

// 清除所有书签
cmds.clearBookmarks = function(cm) {
  // 获取书签数组
  var marks = cm.state.sublimeBookmarks;
  // 如果存在书签，则循环清除每个书签
  if (marks) for (var i = 0; i < marks.length; i++) marks[i].clear();
  // 清空书签数组
  marks.length = 0;
};

// 选择所有书签
cmds.selectBookmarks = function(cm) {
  // 获取书签数组和范围数组
  var marks = cm.state.sublimeBookmarks, ranges = [];
  // 如果存在书签，则循环处理每个书签
  if (marks) for (var i = 0; i < marks.length; i++) {
    // 查找书签的位置
    var found = marks[i].find();
    // 如果未找到位置，则移除该书签
    if (!found)
      marks.splice(i--, 0);
    else
      // 将书签的位置添加到范围数组中
      ranges.push({anchor: found.from, head: found.to});
  }
  // 如果范围数组不为空，则设置编辑器的选中范围为范围数组
  if (ranges.length)
    cm.setSelections(ranges, 0);
};

// 修改单词或选中范围
function modifyWordOrSelection(cm, mod) {
    # 执行编辑操作
    cm.operation(function() {
      # 获取当前选中文本的范围
      var ranges = cm.listSelections(), indices = [], replacements = [];
      # 遍历选中文本的范围
      for (var i = 0; i < ranges.length; i++) {
        var range = ranges[i];
        # 如果范围为空，则将索引添加到数组中，并将替换内容置为空字符串
        if (range.empty()) { indices.push(i); replacements.push(""); }
        # 否则，将替换内容设置为经过特定操作后的文本
        else replacements.push(mod(cm.getRange(range.from(), range.to())));
      }
      # 用替换内容替换选中文本
      cm.replaceSelections(replacements, "around", "case");
      # 遍历索引数组，逆序处理
      for (var i = indices.length - 1, at; i >= 0; i--) {
        var range = ranges[indices[i]];
        # 如果存在at且当前范围的头部位置大于at，则继续下一次循环
        if (at && CodeMirror.cmpPos(range.head, at) > 0) continue;
        # 获取当前范围头部位置的单词信息
        var word = wordAt(cm, range.head);
        at = word.from;
        # 用处理后的单词替换当前范围的内容
        cm.replaceRange(mod(word.word), word.from, word.to);
      }
    });
  }

  # 定义智能退格操作
  cmds.smartBackspace = function(cm) {
    # 如果有选中文本，则返回CodeMirror.Pass
    if (cm.somethingSelected()) return CodeMirror.Pass;

    # 执行编辑操作
    cm.operation(function() {
      # 获取当前光标位置
      var cursors = cm.listSelections();
      var indentUnit = cm.getOption("indentUnit");

      # 遍历光标位置，逆序处理
      for (var i = cursors.length - 1; i >= 0; i--) {
        var cursor = cursors[i].head;
        # 获取光标所在行的起始位置到光标位置的文本
        var toStartOfLine = cm.getRange({line: cursor.line, ch: 0}, cursor);
        # 计算光标所在列的位置
        var column = CodeMirror.countColumn(toStartOfLine, null, cm.getOption("tabSize"));

        # 默认向左删除一个字符
        var deletePos = cm.findPosH(cursor, -1, "char", false);

        # 如果光标所在行为空且光标所在列为缩进单位的整数倍
        if (toStartOfLine && !/\S/.test(toStartOfLine) && column % indentUnit == 0) {
          # 获取前一个缩进单位的位置
          var prevIndent = new Pos(cursor.line,
            CodeMirror.findColumn(toStartOfLine, column - indentUnit, indentUnit));

          # 如果找到有效的前一个缩进单位位置，则将删除位置设置为前一个缩进单位位置
          if (prevIndent.ch != cursor.ch) deletePos = prevIndent;
        }

        # 删除指定范围的内容
        cm.replaceRange("", deletePos, cursor, "+delete");
      }
    });
  };

  # 定义向右删除整行操作
  cmds.delLineRight = function(cm) {
  // 定义一个操作函数，接受一个匿名函数作为参数
  cm.operation(function() {
    // 获取当前编辑器中所有选中文本的范围
    var ranges = cm.listSelections();
    // 遍历选中文本的范围，逐个删除
    for (var i = ranges.length - 1; i >= 0; i--)
      cm.replaceRange("", ranges[i].anchor, Pos(ranges[i].to().line), "+delete");
    // 滚动编辑器视图以确保光标可见
    cm.scrollIntoView();
  });

  // 定义一个在光标处将文本转换为大写的函数
  cmds.upcaseAtCursor = function(cm) {
    modifyWordOrSelection(cm, function(str) { return str.toUpperCase(); });
  };
  // 定义一个在光标处将文本转换为小写的函数
  cmds.downcaseAtCursor = function(cm) {
    modifyWordOrSelection(cm, function(str) { return str.toLowerCase(); });
  };

  // 定义一个设置Sublime标记的函数
  cmds.setSublimeMark = function(cm) {
    // 如果已经存在Sublime标记，则清除
    if (cm.state.sublimeMark) cm.state.sublimeMark.clear();
    // 设置Sublime标记为当前光标位置
    cm.state.sublimeMark = cm.setBookmark(cm.getCursor());
  };
  // 定义一个选择到Sublime标记的函数
  cmds.selectToSublimeMark = function(cm) {
    // 查找Sublime标记的位置
    var found = cm.state.sublimeMark && cm.state.sublimeMark.find();
    // 如果找到了，则将光标移动到Sublime标记位置
    if (found) cm.setSelection(cm.getCursor(), found);
  };
  // 定义一个删除到Sublime标记的函数
  cmds.deleteToSublimeMark = function(cm) {
    // 查找Sublime标记的位置
    var found = cm.state.sublimeMark && cm.state.sublimeMark.find();
    // 如果找到了，则删除从当前光标位置到Sublime标记位置的文本
    if (found) {
      var from = cm.getCursor(), to = found;
      if (CodeMirror.cmpPos(from, to) > 0) { var tmp = to; to = from; from = tmp; }
      cm.state.sublimeKilled = cm.getRange(from, to);
      cm.replaceRange("", from, to);
    }
  };
  // 定义一个与Sublime标记交换位置的函数
  cmds.swapWithSublimeMark = function(cm) {
    // 查找Sublime标记的位置
    var found = cm.state.sublimeMark && cm.state.sublimeMark.find();
    // 如果找到了，则清除Sublime标记，设置新的Sublime标记为当前光标位置，并将光标移动到原Sublime标记位置
    if (found) {
      cm.state.sublimeMark.clear();
      cm.state.sublimeMark = cm.setBookmark(cm.getCursor());
      cm.setCursor(found);
    }
  };
  // 定义一个Sublime粘贴的函数
  cmds.sublimeYank = function(cm) {
    // 如果存在Sublime删除的文本，则将其粘贴到当前光标位置
    if (cm.state.sublimeKilled != null)
      cm.replaceSelection(cm.state.sublimeKilled, null, "paste");
  };

  // 定义一个在编辑器中将光标移动到中心的函数
  cmds.showInCenter = function(cm) {
    // 获取光标位置的坐标信息
    var pos = cm.cursorCoords(null, "local");
    // 将编辑器滚动到使光标位置处于视图中央
    cm.scrollTo(null, (pos.top + pos.bottom) / 2 - cm.getScrollInfo().clientHeight / 2);
  };

  // 定义一个获取目标范围的函数
  function getTarget(cm) {
    // 获取当前光标的起始位置和结束位置
    var from = cm.getCursor("from"), to = cm.getCursor("to");
    # 如果起始位置和结束位置相同
    if (CodeMirror.cmpPos(from, to) == 0):
      # 获取起始位置的单词
      var word = wordAt(cm, from);
      # 如果没有单词则返回
      if (!word.word) return;
      # 更新起始位置和结束位置为单词的起始和结束位置
      from = word.from;
      to = word.to;
    # 返回起始位置、结束位置、查询内容和单词
    return {from: from, to: to, query: cm.getRange(from, to), word: word};
  }

  # 查找并跳转到目标位置
  function findAndGoTo(cm, forward):
    # 获取目标位置
    var target = getTarget(cm);
    # 如果没有目标位置则返回
    if (!target) return;
    # 获取查询内容
    var query = target.query;
    # 获取搜索光标
    var cur = cm.getSearchCursor(query, forward ? target.to : target.from);

    # 如果向前查找则找到下一个匹配，否则找到上一个匹配
    if (forward ? cur.findNext() : cur.findPrevious()):
      # 设置选中内容为匹配的位置
      cm.setSelection(cur.from(), cur.to());
    else:
      # 重新获取搜索光标
      cur = cm.getSearchCursor(query, forward ? Pos(cm.firstLine(), 0)
                                              : cm.clipPos(Pos(cm.lastLine())));
      # 如果向前查找则找到下一个匹配，否则找到上一个匹配
      if (forward ? cur.findNext() : cur.findPrevious()):
        # 设置选中内容为匹配的位置
        cm.setSelection(cur.from(), cur.to());
      # 如果存在单词
      else if (target.word)
        # 设置选中内容为单词的起始和结束位置
        cm.setSelection(target.from, target.to);
    # 定义命令 findUnder
  cmds.findUnder = function(cm) { findAndGoTo(cm, true); };
  # 定义命令 findUnderPrevious
  cmds.findUnderPrevious = function(cm) { findAndGoTo(cm,false); };
  # 定义命令 findAllUnder
  cmds.findAllUnder = function(cm):
    # 获取目标位置
    var target = getTarget(cm);
    # 如果没有目标位置则返回
    if (!target) return;
    # 获取搜索光标
    var cur = cm.getSearchCursor(target.query);
    # 定义匹配数组
    var matches = [];
    # 定义主要索引
    var primaryIndex = -1;
    # 循环查找所有匹配
    while (cur.findNext()):
      # 将匹配的起始和结束位置添加到匹配数组
      matches.push({anchor: cur.from(), head: cur.to()});
      # 如果匹配的起始位置在目标位置之前，则主要索引加一
      if (cur.from().line <= target.from.line && cur.from().ch <= target.from.ch)
        primaryIndex++;
    # 设置选中内容为匹配数组中的位置，主要索引为主要索引
    cm.setSelections(matches, primaryIndex);


  # 定义键盘映射
  var keyMap = CodeMirror.keyMap;
  keyMap.macSublime = {
    "Cmd-Left": "goLineStartSmart",
    "Shift-Tab": "indentLess",
    "Shift-Ctrl-K": "deleteLine",
    "Alt-Q": "wrapLines",
    "Ctrl-Left": "goSubwordLeft",
    "Ctrl-Right": "goSubwordRight",
    "Ctrl-Alt-Up": "scrollLineUp",
    "Ctrl-Alt-Down": "scrollLineDown",
    "Cmd-L": "selectLine",
    "Shift-Cmd-L": "splitSelectionByLine",
    "Esc": "singleSelectionTop",
    "Cmd-Enter": "insertLineAfter",
    "Shift-Cmd-Enter": "insertLineBefore",
    # 定义键盘快捷键映射，选择下一个出现的内容
    "Cmd-D": "selectNextOccurrence",
    # 定义键盘快捷键映射，选择范围
    "Shift-Cmd-Space": "selectScope",
    # 定义键盘快捷键映射，选择括号之间的内容
    "Shift-Cmd-M": "selectBetweenBrackets",
    # 定义键盘快捷键映射，跳转到括号
    "Cmd-M": "goToBracket",
    # 定义键盘快捷键映射，向上交换当前行和上一行
    "Cmd-Ctrl-Up": "swapLineUp",
    # 定义键盘快捷键映射，向下交换当前行和下一行
    "Cmd-Ctrl-Down": "swapLineDown",
    # 定义键盘快捷键映射，切换缩进的注释
    "Cmd-/": "toggleCommentIndented",
    # 定义键盘快捷键映射，合并当前行和下一行
    "Cmd-J": "joinLines",
    # 定义键盘快捷键映射，复制当前行
    "Shift-Cmd-D": "duplicateLine",
    # 定义键盘快捷键映射，对选中的行进行排序
    "F5": "sortLines",
    # 定义键盘快捷键映射，对选中的行进行不区分大小写的排序
    "Cmd-F5": "sortLinesInsensitive",
    # 定义键盘快捷键映射，跳转到下一个书签
    "F2": "nextBookmark",
    # 定义键盘快捷键映射，跳转到上一个书签
    "Shift-F2": "prevBookmark",
    # 定义键盘快捷键映射，切换书签
    "Cmd-F2": "toggleBookmark",
    # 定义键盘快捷键映射，清除所有书签
    "Shift-Cmd-F2": "clearBookmarks",
    # 定义键盘快捷键映射，选择所有书签
    "Alt-F2": "selectBookmarks",
    # 定义键盘快捷键映射，智能退格
    "Backspace": "smartBackspace",
    # 定义键盘快捷键映射，跳过并选择下一个出现的内容
    "Cmd-K Cmd-D": "skipAndSelectNextOccurrence",
    # 定义键盘快捷键映射，删除右侧的行
    "Cmd-K Cmd-K": "delLineRight",
    # 定义键盘快捷键映射，将光标处的内容转换为大写
    "Cmd-K Cmd-U": "upcaseAtCursor",
    # 定义键盘快捷键映射，将光标处的内容转换为小写
    "Cmd-K Cmd-L": "downcaseAtCursor",
    # 定义键盘快捷键映射，设置 Sublime 标记
    "Cmd-K Cmd-Space": "setSublimeMark",
    # 定义键盘快捷键映射，选择到 Sublime 标记
    "Cmd-K Cmd-A": "selectToSublimeMark",
    # 定义键盘快捷键映射，删除到 Sublime 标记
    "Cmd-K Cmd-W": "deleteToSublimeMark",
    # 定义键盘快捷键映射，与 Sublime 标记交换内容
    "Cmd-K Cmd-X": "swapWithSublimeMark",
    # 定义键盘快捷键映射，Sublime 粘贴
    "Cmd-K Cmd-Y": "sublimeYank",
    # 定义键盘快捷键映射，在中心显示
    "Cmd-K Cmd-C": "showInCenter",
    # 定义键盘快捷键映射，清除所有书签
    "Cmd-K Cmd-G": "clearBookmarks",
    # 定义键盘快捷键映射，删除左侧的行
    "Cmd-K Cmd-Backspace": "delLineLeft",
    # 定义键盘快捷键映射，折叠所有内容
    "Cmd-K Cmd-1": "foldAll",
    # 定义键盘快捷键映射，展开所有内容
    "Cmd-K Cmd-0": "unfoldAll",
    # 定义键盘快捷键映射，展开所有内容
    "Cmd-K Cmd-J": "unfoldAll",
    # 定义键盘快捷键映射，将光标添加到上一行
    "Ctrl-Shift-Up": "addCursorToPrevLine",
    # 定义键盘快捷键映射，将光标添加到下一行
    "Ctrl-Shift-Down": "addCursorToNextLine",
    # 定义键盘快捷键映射，查找并选择下一个匹配内容
    "Cmd-F3": "findUnder",
    # 定义键盘快捷键映射，查找并选择上一个匹配内容
    "Shift-Cmd-F3": "findUnderPrevious",
    # 定义键盘快捷键映射，查找并选择所有匹配内容
    "Alt-F3": "findAllUnder",
    # 定义键盘快捷键映射，���叠内容
    "Shift-Cmd-[": "fold",
    # 定义键盘快捷键映射，展开内容
    "Shift-Cmd-]": "unfold",
    # 定义键盘快捷键映射，增量查找
    "Cmd-I": "findIncremental",
    # 定义键盘快捷键映射，反向增量查找
    "Shift-Cmd-I": "findIncrementalReverse",
    # 定义键盘快捷键映射，替换
    "Cmd-H": "replace",
    # 定义键盘快捷键映射，查找下一个匹配内容
    "F3": "findNext",
    # 定义键盘快捷键映射，查找上一个匹配内容
    "Shift-F3": "findPrev",
    # 定义键盘快捷键映射，mac 默认
    "fallthrough": "macDefault"
  };
  # 标准化键盘映射
  CodeMirror.normalizeKeyMap(keyMap.macSublime);

  # 定义 PC Sublime 键盘映射
  keyMap.pcSublime = {
    # 定义键盘快捷键映射，减少缩进
    "Shift-Tab": "indentLess",
    # 定义键盘快捷键映射，删除当前行
    "Shift-Ctrl-K": "deleteLine",
    # 定义键盘快捷键映射，换行
    "Alt-Q": "wrapLines",
    # 定义键盘快捷键映射，交换字符
    "Ctrl-T": "transposeChars",
    # 定义键盘快捷键映射，向左跳转一个子词
    "Alt-Left": "goSubwordLeft",
    # 定义键盘快捷键映射，向右跳转一个子词
    "Alt-Right": "goSubwordRight",
    # 定义键盘快捷键映射，向上滚动一行
    "Ctrl-Up": "scrollLineUp",
    # 定义键盘快捷键映射，向下滚动一行
    "Ctrl-Down": "scrollLineDown",
    # 定义键盘快捷键映射，选择当前行
    "Ctrl-L": "selectLine",
    # 定义键盘快捷键映射，根据行进行分割选择
    "Shift-Ctrl-L": "splitSelectionByLine",
    # 定义键盘快捷键映射，将"Esc"映射到"singleSelectionTop"
    "Esc": "singleSelectionTop",
    # 将"Ctrl-Enter"映射到"insertLineAfter"
    "Ctrl-Enter": "insertLineAfter",
    # 将"Shift-Ctrl-Enter"映射到"insertLineBefore"
    "Shift-Ctrl-Enter": "insertLineBefore",
    # 将"Ctrl-D"映射到"selectNextOccurrence"
    "Ctrl-D": "selectNextOccurrence",
    # 将"Shift-Ctrl-Space"映射到"selectScope"
    "Shift-Ctrl-Space": "selectScope",
    # 将"Shift-Ctrl-M"映射到"selectBetweenBrackets"
    "Shift-Ctrl-M": "selectBetweenBrackets",
    # 将"Ctrl-M"映射到"goToBracket"
    "Ctrl-M": "goToBracket",
    # 将"Shift-Ctrl-Up"映射到"swapLineUp"
    "Shift-Ctrl-Up": "swapLineUp",
    # 将"Shift-Ctrl-Down"映射到"swapLineDown"
    "Shift-Ctrl-Down": "swapLineDown",
    # 将"Ctrl-/"映射到"toggleCommentIndented"
    "Ctrl-/": "toggleCommentIndented",
    # 将"Ctrl-J"映射到"joinLines"
    "Ctrl-J": "joinLines",
    # 将"Shift-Ctrl-D"映射到"duplicateLine"
    "Shift-Ctrl-D": "duplicateLine",
    # 将"F9"映射到"sortLines"
    "F9": "sortLines",
    # 将"Ctrl-F9"映射到"sortLinesInsensitive"
    "Ctrl-F9": "sortLinesInsensitive",
    # 将"F2"映射到"nextBookmark"
    "F2": "nextBookmark",
    # 将"Shift-F2"映射到"prevBookmark"
    "Shift-F2": "prevBookmark",
    # 将"Ctrl-F2"映射到"toggleBookmark"
    "Ctrl-F2": "toggleBookmark",
    # 将"Shift-Ctrl-F2"映射到"clearBookmarks"
    "Shift-Ctrl-F2": "clearBookmarks",
    # 将"Alt-F2"映射到"selectBookmarks"
    "Alt-F2": "selectBookmarks",
    # 将"Backspace"映射到"smartBackspace"
    "Backspace": "smartBackspace",
    # 将"Ctrl-K Ctrl-D"映射到"skipAndSelectNextOccurrence"
    "Ctrl-K Ctrl-D": "skipAndSelectNextOccurrence",
    # 将"Ctrl-K Ctrl-K"映射到"delLineRight"
    "Ctrl-K Ctrl-K": "delLineRight",
    # 将"Ctrl-K Ctrl-U"映射到"upcaseAtCursor"
    "Ctrl-K Ctrl-U": "upcaseAtCursor",
    # 将"Ctrl-K Ctrl-L"映射到"downcaseAtCursor"
    "Ctrl-K Ctrl-L": "downcaseAtCursor",
    # 将"Ctrl-K Ctrl-Space"映射到"setSublimeMark"
    "Ctrl-K Ctrl-Space": "setSublimeMark",
    # 将"Ctrl-K Ctrl-A"映射到"selectToSublimeMark"
    "Ctrl-K Ctrl-A": "selectToSublimeMark",
    # 将"Ctrl-K Ctrl-W"映射到"deleteToSublimeMark"
    "Ctrl-K Ctrl-W": "deleteToSublimeMark",
    # 将"Ctrl-K Ctrl-X"映射到"swapWithSublimeMark"
    "Ctrl-K Ctrl-X": "swapWithSublimeMark",
    # 将"Ctrl-K Ctrl-Y"映射到"sublimeYank"
    "Ctrl-K Ctrl-Y": "sublimeYank",
    # 将"Ctrl-K Ctrl-C"映射到"showInCenter"
    "Ctrl-K Ctrl-C": "showInCenter",
    # 将"Ctrl-K Ctrl-G"映射到"clearBookmarks"
    "Ctrl-K Ctrl-G": "clearBookmarks",
    # 将"Ctrl-K Ctrl-Backspace"映射到"delLineLeft"
    "Ctrl-K Ctrl-Backspace": "delLineLeft",
    # 将"Ctrl-K Ctrl-1"映射到"foldAll"
    "Ctrl-K Ctrl-1": "foldAll",
    # 将"Ctrl-K Ctrl-0"映射到"unfoldAll"
    "Ctrl-K Ctrl-0": "unfoldAll",
    # 将"Ctrl-K Ctrl-J"映射到"unfoldAll"
    "Ctrl-K Ctrl-J": "unfoldAll",
    # 将"Ctrl-Alt-Up"映射到"addCursorToPrevLine"
    "Ctrl-Alt-Up": "addCursorToPrevLine",
    # 将"Ctrl-Alt-Down"映射到"addCursorToNextLine"
    "Ctrl-Alt-Down": "addCursorToNextLine",
    # 将"Ctrl-F3"映射到"findUnder"
    "Ctrl-F3": "findUnder",
    # 将"Shift-Ctrl-F3"映射到"findUnderPrevious"
    "Shift-Ctrl-F3": "findUnderPrevious",
    # 将"Alt-F3"映射到"findAllUnder"
    "Alt-F3": "findAllUnder",
    # 将"Shift-Ctrl-["映射到"fold"
    "Shift-Ctrl-[": "fold",
    # 将"Shift-Ctrl-]"映射到"unfold"
    "Shift-Ctrl-]": "unfold",
    # 将"Ctrl-I"映射到"findIncremental"
    "Ctrl-I": "findIncremental",
    # 将"Shift-Ctrl-I"映射到"findIncrementalReverse"
    "Shift-Ctrl-I": "findIncrementalReverse",
    # 将"Ctrl-H"映射到"replace"
    "Ctrl-H": "replace",
    # 将"F3"映射到"findNext"
    "F3": "findNext",
    # 将"Shift-F3"映射到"findPrev"
    "Shift-F3": "findPrev",
    # 将"fallthrough"映射到"pcDefault"
    "fallthrough": "pcDefault"
    # 标准化键盘映射
    CodeMirror.normalizeKeyMap(keyMap.pcSublime);
    # 判断是否为 Mac 系统，选择相应的键盘映射
    var mac = keyMap.default == keyMap.macDefault;
    keyMap.sublime = mac ? keyMap.macSublime : keyMap.pcSublime;
// 导入模块
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror"], mod);
  // 如果是普通浏览器环境
  else
    mod(CodeMirror);
})(function(CodeMirror) {
  // 定义 dialogDiv 函数
  function dialogDiv(cm, template, bottom) {
    // 获取编辑器的包裹元素
    var wrap = cm.getWrapperElement();
    var dialog;
    // 创建一个 div 元素作为对话框
    dialog = wrap.appendChild(document.createElement("div"));
    // 根据参数决定对话框的位置
    if (bottom)
      dialog.className = "CodeMirror-dialog CodeMirror-dialog-bottom";
    else
      dialog.className = "CodeMirror-dialog CodeMirror-dialog-top";

    // 根据模板类型设置对话框内容
    if (typeof template == "string") {
      dialog.innerHTML = template;
    } else { // 假设是一个独立的 DOM 元素
      dialog.appendChild(template);
    }
    // 给编辑器的包裹元素添加类名，表示对话框已打开
    CodeMirror.addClass(wrap, 'dialog-opened');
    return dialog;
  }

  // 定义 closeNotification 函数
  function closeNotification(cm, newVal) {
    // 如果当前存在通知关闭函数，则调用它
    if (cm.state.currentNotificationClose)
      cm.state.currentNotificationClose();
    // 将新的通知关闭函数赋值给当前通知关闭函数
    cm.state.currentNotificationClose = newVal;
  }

  // 定义 openDialog 方法
  CodeMirror.defineExtension("openDialog", function(template, callback, options) {
    // 如果没有传入 options，则设置为空对象
    if (!options) options = {};

    // 关闭当前通知
    closeNotification(this, null);

    // 创建对话框并返回
    var dialog = dialogDiv(this, template, options.bottom);
    var closed = false, me = this;
    // 定义关闭对话框的函数
    function close(newVal) {
      // 如果 newVal 是字符串，则将其赋值给输入框的值
      if (typeof newVal == 'string') {
        inp.value = newVal;
      } else {
        // 如果已经关闭，则直接返回
        if (closed) return;
        // 标记对话框已关闭
        closed = true;
        // 移除对话框及其父元素的类名，表示对话框已关闭
        CodeMirror.rmClass(dialog.parentNode, 'dialog-opened');
        dialog.parentNode.removeChild(dialog);
        // 让编辑器获得焦点
        me.focus();

        // 如果存在 onClose 回调函数，则调用它
        if (options.onClose) options.onClose(dialog);
      }
    }

    // 获取对话框中的输入框和按钮
    var inp = dialog.getElementsByTagName("input")[0], button;
    // 如果输入框存在
    if (inp) {
      // 让输入框获得焦点
      inp.focus();

      // 如果选项中有数值，则将输入框的值设置为该数值，并根据选项决定是否在打开时选择输入框中的值
      if (options.value) {
        inp.value = options.value;
        if (options.selectValueOnOpen !== false) {
          inp.select();
        }
      }

      // 如果选项中有 onInput 回调函数，则在输入框输入时触发该回调函数
      if (options.onInput)
        CodeMirror.on(inp, "input", function(e) { options.onInput(e, inp.value, close);});
      // 如果选项中有 onKeyUp 回调函数，则在输入框按键弹起时触发该回调函数
      if (options.onKeyUp)
        CodeMirror.on(inp, "keyup", function(e) {options.onKeyUp(e, inp.value, close);});

      // 在输入框按键按下时触发的事件处理函数
      CodeMirror.on(inp, "keydown", function(e) {
        // 如果选项中有 onKeyDown 回调函数，则在按键按下时触发该回调函数
        if (options && options.onKeyDown && options.onKeyDown(e, inp.value, close)) { return; }
        // 如果按下的是 ESC 键或者按下的是回车键且选项中没有设置不关闭，则让输入框失去焦点并关闭对话框
        if (e.keyCode == 27 || (options.closeOnEnter !== false && e.keyCode == 13)) {
          inp.blur();
          CodeMirror.e_stop(e);
          close();
        }
        // 如果按下的是回车键，则调用回调函数，并传入输入框的值和事件对象
        if (e.keyCode == 13) callback(inp.value, e);
      });

      // 如果选项中没有设置不关闭，则在对话框失去焦点时关闭对话框
      if (options.closeOnBlur !== false) CodeMirror.on(dialog, "focusout", function (evt) {
        if (evt.relatedTarget !== null) close();
      });
    } 
    // 如果输入框不存在，且对话框中有按钮
    else if (button = dialog.getElementsByTagName("button")[0]) {
      // 在按钮点击时关闭对话框并让编辑器获得焦点
      CodeMirror.on(button, "click", function() {
        close();
        me.focus();
      });

      // 如果选项中没有设置不关闭，则在按钮失去焦点时关闭对话框
      if (options.closeOnBlur !== false) CodeMirror.on(button, "blur", close);

      // 让按钮获得焦点
      button.focus();
    }
    // 返回关闭函数
    return close;
  });

  // 定义一个名为 openConfirm 的编辑器扩展方法
  CodeMirror.defineExtension("openConfirm", function(template, callbacks, options) {
    // 关闭当前通知
    closeNotification(this, null);
    // 创建对话框
    var dialog = dialogDiv(this, template, options && options.bottom);
    var buttons = dialog.getElementsByTagName("button");
    var closed = false, me = this, blurring = 1;
    // 定义关闭函数
    function close() {
      if (closed) return;
      closed = true;
      // 移除对话框的打开样式，并移除对话框
      CodeMirror.rmClass(dialog.parentNode, 'dialog-opened');
      dialog.parentNode.removeChild(dialog);
      // 让编辑器获得焦点
      me.focus();
    }
    // 让第一个按钮获得焦点
    buttons[0].focus();
    // 遍历按钮数组
    for (var i = 0; i < buttons.length; ++i) {
      // 获取当前按钮
      var b = buttons[i];
      // 创建闭包，将回调函数作为参数传入
      (function(callback) {
        // 给按钮添加点击事件监听器
        CodeMirror.on(b, "click", function(e) {
          // 阻止默认事件
          CodeMirror.e_preventDefault(e);
          // 关闭通知
          close();
          // 如果存在回调函数，则执行回调函数
          if (callback) callback(me);
        });
      })(callbacks[i]);
      // 给按钮添加失焦事件监听器
      CodeMirror.on(b, "blur", function() {
        // 减少失焦计数
        --blurring;
        // 设置定时器，在200ms后如果失焦计数小于等于0，则关闭通知
        setTimeout(function() { if (blurring <= 0) close(); }, 200);
      });
      // 给按钮添加获取焦点事件监听器
      CodeMirror.on(b, "focus", function() { ++blurring; });
    }
  });

  /*
   * openNotification
   * 打开一个通知，可以使用可选的定时器关闭（默认5000ms定时器），并且始终在点击时关闭。
   *
   * 如果在打开另一个通知时已经有一个通知打开，则会关闭当前打开的通知，并立即打开新的通知。
   */
  CodeMirror.defineExtension("openNotification", function(template, options) {
    // 关闭通知
    closeNotification(this, close);
    // 创建对话框
    var dialog = dialogDiv(this, template, options && options.bottom);
    var closed = false, doneTimer;
    // 设置持续时间，默认为5000ms
    var duration = options && typeof options.duration !== "undefined" ? options.duration : 5000;

    function close() {
      // 如果已关闭，则直接返回
      if (closed) return;
      // 设置已关闭标志
      closed = true;
      // 清除定时器
      clearTimeout(doneTimer);
      // 移除对话框的打开样式类，并移除对话框
      CodeMirror.rmClass(dialog.parentNode, 'dialog-opened');
      dialog.parentNode.removeChild(dialog);
    }

    // 给对话框添加点击事件监听器
    CodeMirror.on(dialog, 'click', function(e) {
      // 阻止默认事件
      CodeMirror.e_preventDefault(e);
      // 关闭通知
      close();
    });

    // 如果设置了持续时间，则设置定时器，在持续时间后关闭通知
    if (duration)
      doneTimer = setTimeout(close, duration);

    // 返回关闭函数
    return close;
  });
// 定义一个自执行函数，传入一个 mod 参数
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境
  else
    mod(CodeMirror);
})(function(CodeMirror) {
  // 默认配置
  var defaults = {
    pairs: "()[]{}''\"\"",
    closeBefore: ")]}'\":;>",
    triples: "",
    explode: "[]{}"
  };

  // 位置对象
  var Pos = CodeMirror.Pos;

  // 定义 CodeMirror 的 autoCloseBrackets 选项
  CodeMirror.defineOption("autoCloseBrackets", false, function(cm, val, old) {
    // 移除旧的按键映射
    if (old && old != CodeMirror.Init) {
      cm.removeKeyMap(keyMap);
      cm.state.closeBrackets = null;
    }
    // 如果新值存在
    if (val) {
      // 确保绑定
      ensureBound(getOption(val, "pairs"))
      cm.state.closeBrackets = val;
      cm.addKeyMap(keyMap);
    }
  });

  // 获取配置选项
  function getOption(conf, name) {
    if (name == "pairs" && typeof conf == "string") return conf;
    if (typeof conf == "object" && conf[name] != null) return conf[name];
    return defaults[name];
  }

  // 按键映射
  var keyMap = {Backspace: handleBackspace, Enter: handleEnter};

  // 确保绑定
  function ensureBound(chars) {
    for (var i = 0; i < chars.length; i++) {
      var ch = chars.charAt(i), key = "'" + ch + "'"
      if (!keyMap[key]) keyMap[key] = handler(ch)
    }
  }
  ensureBound(defaults.pairs + "`")

  // 处理函数
  function handler(ch) {
    return function(cm) { return handleChar(cm, ch); };
  }

  // 获取配置
  function getConfig(cm) {
    var deflt = cm.state.closeBrackets;
    if (!deflt || deflt.override) return deflt;
    var mode = cm.getModeAt(cm.getCursor());
    return mode.closeBrackets || deflt;
  }

  // 处理退格键
  function handleBackspace(cm) {
    var conf = getConfig(cm);
    if (!conf || cm.getOption("disableInput")) return CodeMirror.Pass;

    var pairs = getOption(conf, "pairs");
    var ranges = cm.listSelections();
    // 遍历ranges数组，对每个range进行处理
    for (var i = 0; i < ranges.length; i++) {
      // 如果range不为空，则返回CodeMirror.Pass
      if (!ranges[i].empty()) return CodeMirror.Pass;
      // 获取range头部字符周围的字符
      var around = charsAround(cm, ranges[i].head);
      // 如果周围字符不存在或者不在pairs数组中，返回CodeMirror.Pass
      if (!around || pairs.indexOf(around) % 2 != 0) return CodeMirror.Pass;
    }
    // 逆序遍历ranges数组
    for (var i = ranges.length - 1; i >= 0; i--) {
      // 获取当前range的头部位置
      var cur = ranges[i].head;
      // 在当前位置替换字符为空字符串
      cm.replaceRange("", Pos(cur.line, cur.ch - 1), Pos(cur.line, cur.ch + 1), "+delete");
    }
  }

  // 处理回车键事件
  function handleEnter(cm) {
    // 获取配置信息
    var conf = getConfig(cm);
    // 获取是否explode配置
    var explode = conf && getOption(conf, "explode");
    // 如果不存在explode配置或者输入被禁用，则返回CodeMirror.Pass
    if (!explode || cm.getOption("disableInput")) return CodeMirror.Pass;

    // 获取选择范围
    var ranges = cm.listSelections();
    // 遍历ranges数组
    for (var i = 0; i < ranges.length; i++) {
      // 如果range不为空，则返回CodeMirror.Pass
      if (!ranges[i].empty()) return CodeMirror.Pass;
      // 获取range头部字符周围的字符
      var around = charsAround(cm, ranges[i].head);
      // 如果周围字符不存在或者不在explode数组中，返回CodeMirror.Pass
      if (!around || explode.indexOf(around) % 2 != 0) return CodeMirror.Pass;
    }
    // 执行操作
    cm.operation(function() {
      // 获取行分隔符
      var linesep = cm.lineSeparator() || "\n";
      // 在选择范围插入两个行分隔符
      cm.replaceSelection(linesep + linesep, null);
      // 执行goCharLeft命令
      cm.execCommand("goCharLeft");
      // 重新获取选择范围
      ranges = cm.listSelections();
      // 遍历ranges数组
      for (var i = 0; i < ranges.length; i++) {
        // 获取行号
        var line = ranges[i].head.line;
        // 对当前行和下一行进行缩进
        cm.indentLine(line, null, true);
        cm.indentLine(line + 1, null, true);
      }
    });
  }

  // 收缩选择范围
  function contractSelection(sel) {
    // 判断选择范围是否倒置
    var inverted = CodeMirror.cmpPos(sel.anchor, sel.head) > 0;
    // 返回收缩后的选择范围
    return {anchor: new Pos(sel.anchor.line, sel.anchor.ch + (inverted ? -1 : 1)),
            head: new Pos(sel.head.line, sel.head.ch + (inverted ? 1 : -1))};
  }

  // 处理字符输入事件
  function handleChar(cm, ch) {
    // 获取配置信息
    var conf = getConfig(cm);
    // 如果不存在配置信息或者输入被禁用，则返回CodeMirror.Pass
    if (!conf || cm.getOption("disableInput")) return CodeMirror.Pass;

    // 获取pairs配置
    var pairs = getOption(conf, "pairs");
    // 获取字符在pairs中的位置
    var pos = pairs.indexOf(ch);
    // 如果字符不在pairs中，则返回CodeMirror.Pass
    if (pos == -1) return CodeMirror.Pass;

    // 获取closeBefore配置
    var closeBefore = getOption(conf,"closeBefore");

    // 获取triples配置
    var triples = getOption(conf, "triples");

    // 判断是否为相同字符
    var identical = pairs.charAt(pos + 1) == ch;
    // 获取选择范围
    var ranges = cm.listSelections();
    # 判断当前光标位置是否在偶数列
    var opening = pos % 2 == 0;

    # 声明变量 type
    var type;
    # 遍历 ranges 数组
    for (var i = 0; i < ranges.length; i++) {
      # 获取当前 range 的头部位置和当前类型
      var range = ranges[i], cur = range.head, curType;
      # 获取当前位置的下一个字符
      var next = cm.getRange(cur, Pos(cur.line, cur.ch + 1));
      # 判断是否为开放符号并且 range 不为空
      if (opening && !range.empty()) {
        curType = "surround";
      } else if ((identical || !opening) && next == ch) {
        # 判断是否为相同符号或者不是开放符号，并且下一个字符等于当前字符
        if (identical && stringStartsAfter(cm, cur))
          curType = "both";
        else if (triples.indexOf(ch) >= 0 && cm.getRange(cur, Pos(cur.line, cur.ch + 3)) == ch + ch + ch)
          curType = "skipThree";
        else
          curType = "skip";
      } else if (identical && cur.ch > 1 && triples.indexOf(ch) >= 0 &&
                 cm.getRange(Pos(cur.line, cur.ch - 2), cur) == ch + ch) {
        # 判断是否为相同符号，并且当前列大于1且是三重符号
        if (cur.ch > 2 && /\bstring/.test(cm.getTokenTypeAt(Pos(cur.line, cur.ch - 2)))) return CodeMirror.Pass;
        curType = "addFour";
      } else if (identical) {
        # 判断是否为相同符号，并且前一个字符不是单词字符并且下一个字符不是当前字符并且前一个字符不是单词字符
        var prev = cur.ch == 0 ? " " : cm.getRange(Pos(cur.line, cur.ch - 1), cur)
        if (!CodeMirror.isWordChar(next) && prev != ch && !CodeMirror.isWordChar(prev)) curType = "both";
        else return CodeMirror.Pass;
      } else if (opening && (next.length === 0 || /\s/.test(next) || closeBefore.indexOf(next) > -1)) {
        # 判断是否为开放符号并且下一个字符为空或者是空白字符或者在 closeBefore 数组中
        curType = "both";
      } else {
        return CodeMirror.Pass;
      }
      # 如果 type 为空，则赋值为当前类型，否则如果 type 不等于当前类型，则返回 CodeMirror.Pass
      if (!type) type = curType;
      else if (type != curType) return CodeMirror.Pass;
    }

    # 根据当前列的奇偶性获取左右符号
    var left = pos % 2 ? pairs.charAt(pos - 1) : ch;
    var right = pos % 2 ? ch : pairs.charAt(pos + 1);
    # 执行操作函数
    cm.operation(function() {
      # 如果类型为"skip"，执行向右移动一个字符的命令
      if (type == "skip") {
        cm.execCommand("goCharRight");
      } 
      # 如果类型为"skipThree"，执行向右移动三个字符的命令
      else if (type == "skipThree") {
        for (var i = 0; i < 3; i++)
          cm.execCommand("goCharRight");
      } 
      # 如果类型为"surround"，执行选中文本并在其周围添加左右字符的操作
      else if (type == "surround") {
        var sels = cm.getSelections();
        for (var i = 0; i < sels.length; i++)
          sels[i] = left + sels[i] + right;
        cm.replaceSelections(sels, "around");
        sels = cm.listSelections().slice();
        for (var i = 0; i < sels.length; i++)
          sels[i] = contractSelection(sels[i]);
        cm.setSelections(sels);
      } 
      # 如果类型为"both"，在当前位置插入左右字符并触发电动力
      else if (type == "both") {
        cm.replaceSelection(left + right, null);
        cm.triggerElectric(left + right);
        cm.execCommand("goCharLeft");
      } 
      # 如果类型为"addFour"，在当前位置之前插入四个左字符
      else if (type == "addFour") {
        cm.replaceSelection(left + left + left + left, "before");
        cm.execCommand("goCharRight");
      }
    });
  }

  # 获取指定位置周围的字符
  function charsAround(cm, pos) {
    var str = cm.getRange(Pos(pos.line, pos.ch - 1),
                          Pos(pos.line, pos.ch + 1));
    return str.length == 2 ? str : null;
  }

  # 检查指定位置之后是否是字符串的起始位置
  function stringStartsAfter(cm, pos) {
    var token = cm.getTokenAt(Pos(pos.line, pos.ch + 1))
    return /\bstring/.test(token.type) && token.start == pos.ch &&
      (pos.ch == 0 || !/\bstring/.test(cm.getTokenTypeAt(pos)))
  }
// 闭包，用于封装代码，避免变量污染全局作用域
(function(mod) {
  // 判断是否为 CommonJS 环境，如果是则加载相应模块
  if (typeof exports == "object" && typeof module == "object") 
    mod(require("../../lib/codemirror"), require("../fold/xml-fold"));
  // 判断是否为 AMD 环境，如果是则加载相应模块
  else if (typeof define == "function" && define.amd) 
    define(["../../lib/codemirror", "../fold/xml-fold"], mod);
  // 如果都不是，则直接加载 CodeMirror
  else 
    mod(CodeMirror);
})(function(CodeMirror) {
  // 定义 CodeMirror 的 autoCloseTags 选项
  CodeMirror.defineOption("autoCloseTags", false, function(cm, val, old) {
    // 如果旧值不是初始值并且不为空，则移除之前的键盘映射
    if (old != CodeMirror.Init && old)
      cm.removeKeyMap("autoCloseTags");
    // 如果值为空，则直接返回
    if (!val) return;
    // 定义键盘映射对象
    var map = {name: "autoCloseTags"};
    // 如果值不是对象或者 whenClosing 不为 false，则设置 '/' 键的处理函数为 autoCloseSlash
    if (typeof val != "object" || val.whenClosing !== false)
      map["'/'"] = function(cm) { return autoCloseSlash(cm); };
    // 如果值不是对象或者 whenOpening 不为 false，则设置 '>' 键的处理函数为 autoCloseGT
    if (typeof val != "object" || val.whenOpening !== false)
      map["'>'"] = function(cm) { return autoCloseGT(cm); };
  # 将键盘映射添加到代码编辑器中
  cm.addKeyMap(map);
  });

  # 不需要自动闭合的 HTML 标签
  var htmlDontClose = ["area", "base", "br", "col", "command", "embed", "hr", "img", "input", "keygen", "link", "meta", "param",
                       "source", "track", "wbr"];
  # 需要缩进的 HTML 标签
  var htmlIndent = ["applet", "blockquote", "body", "button", "div", "dl", "fieldset", "form", "frameset", "h1", "h2", "h3", "h4",
                    "h5", "h6", "head", "html", "iframe", "layer", "legend", "object", "ol", "p", "select", "table", "ul"];

  # 自动闭合 HTML 标签的函数
  function autoCloseGT(cm) {
    # 如果禁用输入，则返回
    if (cm.getOption("disableInput")) return CodeMirror.Pass;
    # 获取选中的文本范围和替换内容
    var ranges = cm.listSelections(), replacements = [];
    # 获取自动闭合标签的选项
    var opt = cm.getOption("autoCloseTags");
    # 遍历ranges数组
    for (var i = 0; i < ranges.length; i++) {
      # 如果当前range不为空，则返回CodeMirror.Pass
      if (!ranges[i].empty()) return CodeMirror.Pass;
      # 获取当前range的头部位置和token
      var pos = ranges[i].head, tok = cm.getTokenAt(pos);
      # 获取当前token所在的inner mode和state
      var inner = CodeMirror.innerMode(cm.getMode(), tok.state), state = inner.state;
      # 获取当前XML标签的信息
      var tagInfo = inner.mode.xmlCurrentTag && inner.mode.xmlCurrentTag(state)
      # 获取当前XML标签的名称
      var tagName = tagInfo && tagInfo.name
      # 如果没有标签名，则返回CodeMirror.Pass
      if (!tagName) return CodeMirror.Pass

      # 判断当前mode是否为html
      var html = inner.mode.configuration == "html";
      # 获取不需要闭合的标签列表
      var dontCloseTags = (typeof opt == "object" && opt.dontCloseTags) || (html && htmlDontClose);
      # 获取需要缩进的标签列表
      var indentTags = (typeof opt == "object" && opt.indentTags) || (html && htmlIndent);

      # 如果token的结束位置大于当前位置的字符数，则修正tagName
      if (tok.end > pos.ch) tagName = tagName.slice(0, tagName.length - tok.end + pos.ch);
      # 将tagName转换为小写
      var lowerTagName = tagName.toLowerCase();
      # 不处理结束标签或自闭合标签末尾的'>'
      if (!tagName ||
          tok.type == "string" && (tok.end != pos.ch || !/[\"\']/.test(tok.string.charAt(tok.string.length - 1)) || tok.string.length == 1) ||
          tok.type == "tag" && tagInfo.close ||
          tok.string.indexOf("/") == (pos.ch - tok.start - 1) || 
          dontCloseTags && indexOf(dontCloseTags, lowerTagName) > -1 ||
          closingTagExists(cm, inner.mode.xmlCurrentContext && inner.mode.xmlCurrentContext(state) || [], tagName, pos, true))
        return CodeMirror.Pass;

      # 获取空标签列表
      var emptyTags = typeof opt == "object" && opt.emptyTags;
      # 如果当前标签在空标签列表中，则替换为自闭合标签
      if (emptyTags && indexOf(emptyTags, tagName) > -1) {
        replacements[i] = { text: "/>", newPos: CodeMirror.Pos(pos.line, pos.ch + 2) };
        continue;
      }

      # 判断是否需要缩进
      var indent = indentTags && indexOf(indentTags, lowerTagName) > -1;
      # 生成替换文本和新位置
      replacements[i] = {indent: indent,
                         text: ">" + (indent ? "\n\n" : "") + "</" + tagName + ">",
                         newPos: indent ? CodeMirror.Pos(pos.line + 1, 0) : CodeMirror.Pos(pos.line, pos.ch + 1)};
    }
    # 检查是否存在 opt 对象并且 opt.dontIndentOnAutoClose 为真，如果是则设置 dontIndentOnAutoClose 为真
    var dontIndentOnAutoClose = (typeof opt == "object" && opt.dontIndentOnAutoClose);
    # 从后向前遍历 ranges 数组
    for (var i = ranges.length - 1; i >= 0; i--) {
      # 获取当前替换信息
      var info = replacements[i];
      # 在指定范围内替换文本
      cm.replaceRange(info.text, ranges[i].head, ranges[i].anchor, "+insert");
      # 复制当前选择
      var sel = cm.listSelections().slice(0);
      # 设置新的选择范围
      sel[i] = {head: info.newPos, anchor: info.newPos};
      cm.setSelections(sel);
      # 如果不是 dontIndentOnAutoClose 或者 info.indent 为真，则进行自动缩进
      if (!dontIndentOnAutoClose && info.indent) {
        cm.indentLine(info.newPos.line, null, true);
        cm.indentLine(info.newPos.line + 1, null, true);
      }
    }
  }

  # 自动关闭当前标签
  function autoCloseCurrent(cm, typingSlash) {
    # 获取当前选择范围
    var ranges = cm.listSelections(), replacements = [];
    # 设置头部文本
    var head = typingSlash ? "/" : "</";
    # 获取自动关闭标签的选项
    var opt = cm.getOption("autoCloseTags");
    # 检查是否存在 opt 对象并且 opt.dontIndentOnSlash 为真，如果是则设置 dontIndentOnAutoClose 为真
    var dontIndentOnAutoClose = (typeof opt == "object" && opt.dontIndentOnSlash);
    // 遍历ranges数组，对每个元素执行以下操作
    for (var i = 0; i < ranges.length; i++) {
        // 如果当前range不为空，则返回CodeMirror.Pass
        if (!ranges[i].empty()) return CodeMirror.Pass;
        // 获取当前range的头部位置和对应的token
        var pos = ranges[i].head, tok = cm.getTokenAt(pos);
        // 获取当前token所在的inner mode和对应的state
        var inner = CodeMirror.innerMode(cm.getMode(), tok.state), state = inner.state;
        // 如果正在输入斜杠并且当前token是字符串，或者token的第一个字符不是"<"，或者token的起始位置不是pos.ch - 1，则返回CodeMirror.Pass
        if (typingSlash && (tok.type == "string" || tok.string.charAt(0) != "<" ||
                            tok.start != pos.ch - 1))
            return CodeMirror.Pass;
        // 用于解决在htmlmixed模式下在JS/CSS片段中自动补全时不在XML模式下的问题
        var replacement, mixed = inner.mode.name != "xml" && cm.getMode().name == "htmlmixed"
        // 如果mixed为true并且inner mode的名称为javascript
        if (mixed && inner.mode.name == "javascript") {
            replacement = head + "script";
        } else if (mixed && inner.mode.name == "css") {
            replacement = head + "style";
        } else {
            // 获取当前inner mode的xmlCurrentContext并执行对应的操作
            var context = inner.mode.xmlCurrentContext && inner.mode.xmlCurrentContext(state)
            // 如果context不存在或者context长度大于0并且在当前位置存在closing tag，则返回CodeMirror.Pass
            if (!context || (context.length && closingTagExists(cm, context, context[context.length - 1], pos)))
                return CodeMirror.Pass;
            // 根据context的最后一个元素构建replacement
            replacement = head + context[context.length - 1]
        }
        // 如果当前位置的行不是">"，则在replacement后面添加">"
        if (cm.getLine(pos.line).charAt(tok.end) != ">") replacement += ">";
        // 将replacement添加到replacements数组中
        replacements[i] = replacement;
    }
    // 用replacements数组替换当前选择的文本
    cm.replaceSelections(replacements);
    // 更新ranges数组
    ranges = cm.listSelections();
    // 如果dontIndentOnAutoClose为false，则对每个range执行以下操作
    if (!dontIndentOnAutoClose) {
        for (var i = 0; i < ranges.length; i++)
            if (i == ranges.length - 1 || ranges[i].head.line < ranges[i + 1].head.line)
                cm.indentLine(ranges[i].head.line);
    }
  }

  // 自动关闭斜杠
  function autoCloseSlash(cm) {
    // 如果disableInput选项为true，则返回CodeMirror.Pass
    if (cm.getOption("disableInput")) return CodeMirror.Pass;
    // 调用autoCloseCurrent函数并传入true作为参数
    return autoCloseCurrent(cm, true);
  }

  // 将closeTag命令绑定到autoCloseCurrent函数
  CodeMirror.commands.closeTag = function(cm) { return autoCloseCurrent(cm); };

  // 查找collection中elt的索引
  function indexOf(collection, elt) {
    // 如果集合对象有 indexOf 方法，则使用该方法查找元素的索引
    if (collection.indexOf) return collection.indexOf(elt);
    // 遍历集合对象，查找元素的索引
    for (var i = 0, e = collection.length; i < e; ++i)
      if (collection[i] == elt) return i;
    // 如果未找到元素，则返回 -1
    return -1;
    }
    
    // 如果加载了 xml-fold 模块，使用其功能尝试验证给定标签是否未闭合
    function closingTagExists(cm, context, tagName, pos, newTag) {
      // 如果未加载 CodeMirror.scanForClosingTag 模块，则返回 false
      if (!CodeMirror.scanForClosingTag) return false;
      // 计算结束位置，最大为当前行号加上 500
      var end = Math.min(cm.lastLine() + 1, pos.line + 500);
      // 查找下一个闭合标签
      var nextClose = CodeMirror.scanForClosingTag(cm, pos, null, end);
      // 如果未找到下一个闭合标签，或者闭合标签不是指定的标签，则返回 false
      if (!nextClose || nextClose.tag != tagName) return false;
      // 如果新标签存在，则将 onCx 设置为 1，否则为 0
      var onCx = newTag ? 1 : 0
      // 遍历上下文数组，查找相同标签的实例数量
      for (var i = context.length - 1; i >= 0; i--) {
        if (context[i] == tagName) ++onCx
        else break
      }
      // 更新位置为下一个闭合标签的位置
      pos = nextClose.to;
      // 遍历 onCx 次，查找后续的闭合标签
      for (var i = 1; i < onCx; i++) {
        var next = CodeMirror.scanForClosingTag(cm, pos, null, end);
        // 如果未找到下一个闭合标签，或者闭合标签不是指定的标签，则返回 false
        if (!next || next.tag != tagName) return false;
        pos = next.to;
      }
      // 如果满足条件，则返回 true
      return true;
    }
// 定义一个匿名函数，并传入一个 mod 参数
(function(mod) {
  // 如果 exports 和 module 都是对象，则使用 CommonJS 规范
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror"));
  // 如果 define 是一个函数且支持 AMD 规范，则使用 AMD
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror"], mod);
  // 否则在普通的浏览器环境中使用
  else
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义正则表达式，用于匹配 Markdown 列表的不同格式
  var listRE = /^(\s*)(>[> ]*|[*+-] \[[x ]\]\s|[*+-]\s|(\d+)([.)]))(\s*)/,
      emptyListRE = /^(\s*)(>[> ]*|[*+-] \[[x ]\]|[*+-]|(\d+)[.)])(\s*)$/,
      unorderedListRE = /[*+-]\s/;

  // 定义 CodeMirror 命令，用于在 Markdown 列表中插入新行并缩进
  CodeMirror.commands.newlineAndIndentContinueMarkdownList = function(cm) {
    // 如果禁用输入，则返回 CodeMirror.Pass
    if (cm.getOption("disableInput")) return CodeMirror.Pass;
    // 获取当前选择的文本范围
    var ranges = cm.listSelections(), replacements = [];
    // 遍历ranges数组，处理每个范围
    for (var i = 0; i < ranges.length; i++) {
      // 获取当前范围的头部位置
      var pos = ranges[i].head;

      // 获取当前行的状态，如果不是Markdown模式，则执行默认的newlineAndIndent命令
      var eolState = cm.getStateAfter(pos.line);
      var inner = CodeMirror.innerMode(cm.getMode(), eolState);
      if (inner.mode.name !== "markdown") {
        cm.execCommand("newlineAndIndent");
        return;
      } else {
        eolState = inner.state;
      }

      // 判断当前行是否在列表中
      var inList = eolState.list !== false;
      // 判断当前行是否在引用中
      var inQuote = eolState.quote !== 0;

      // 获取当前行的内容和匹配列表项的正则表达式
      var line = cm.getLine(pos.line), match = listRE.exec(line);
      // 判断当前范围是否为空，或者不在列表或引用中，或者不匹配列表项的正则，或者光标前为空白
      var cursorBeforeBullet = /^\s*$/.test(line.slice(0, pos.ch));
      if (!ranges[i].empty() || (!inList && !inQuote) || !match || cursorBeforeBullet) {
        cm.execCommand("newlineAndIndent");
        return;
      }
      // 如果当前行是空列表项
      if (emptyListRE.test(line)) {
        // 判断是否在引用的末尾或者列表的末尾，如果是则删除当前行
        var endOfQuote = inQuote && />\s*$/.test(line)
        var endOfList = !/>\s*$/.test(line)
        if (endOfQuote || endOfList) cm.replaceRange("", {
          line: pos.line, ch: 0
        }, {
          line: pos.line, ch: pos.ch + 1
        });
        // 将当前范围的替换内容设置为换行符
        replacements[i] = "\n";
      } else {
        // 如果不是空列表项，则处理列表项的缩进和标记
        var indent = match[1], after = match[5];
        var numbered = !(unorderedListRE.test(match[2]) || match[2].indexOf(">") >= 0);
        var bullet = numbered ? (parseInt(match[3], 10) + 1) + match[4] : match[2].replace("x", " ");
        // 将当前范围的替换内容设置为新的列表项
        replacements[i] = "\n" + indent + bullet + after;

        // 如果是有序列表，则增加后续列表项的编号
        if (numbered) incrementRemainingMarkdownListNumbers(cm, pos);
      }
    }

    // 执行替换操作，将replacements数组中的内容替换到对应的范围中
    cm.replaceSelections(replacements);
  };

  // 当在列表中间添加新项时，自动更新Markdown列表的编号
  function incrementRemainingMarkdownListNumbers(cm, pos) {
    var startLine = pos.line, lookAhead = 0, skipCount = 0;
    var startItem = listRE.exec(cm.getLine(startLine)), startIndent = startItem[1];
    // 使用 do-while 循环来处理下一个列表项的缩进情况
    do {
      // 增加 lookAhead 变量的值，表示向下查找的行数
      lookAhead += 1;
      // 计算下一行的行号
      var nextLineNumber = startLine + lookAhead;
      // 获取下一行的内容和列表项信息
      var nextLine = cm.getLine(nextLineNumber), nextItem = listRE.exec(nextLine);

      // 如果存在下一个列表项
      if (nextItem) {
        // 获取下一个列表项的缩进
        var nextIndent = nextItem[1];
        // 计算新的列表项编号
        var newNumber = (parseInt(startItem[3], 10) + lookAhead - skipCount);
        // 获取下一个列表项的编号
        var nextNumber = (parseInt(nextItem[3], 10)), itemNumber = nextNumber;

        // 如果起始缩进和下一个列表项的缩进相同，并且下一个列表项的编号是有效的
        if (startIndent === nextIndent && !isNaN(nextNumber)) {
          // 如果新的编号和下一个编号相同，则将列表项编号设置为下一个编号加1
          if (newNumber === nextNumber) itemNumber = nextNumber + 1;
          // 如果新的编号大于下一个编号，则将列表项编号设置为新的编号加1
          if (newNumber > nextNumber) itemNumber = newNumber + 1;
          // 替换下一行的内容，更新列表项编号
          cm.replaceRange(
            nextLine.replace(listRE, nextIndent + itemNumber + nextItem[4] + nextItem[5]),
          {
            line: nextLineNumber, ch: 0
          }, {
            line: nextLineNumber, ch: nextLine.length
          });
        } else {
          // 如果起始缩进大于下一个列表项的缩进，则直接返回
          if (startIndent.length > nextIndent.length) return;
          // 如果起始缩进小于下一个列表项的缩进，并且 lookAhead 为1，则直接返回
          // 这表示下一行立即缩进，不清楚用户的意图（是新的缩进项还是同一级别）
          if ((startIndent.length < nextIndent.length) && (lookAhead === 1)) return;
          // 增加跳过计数
          skipCount += 1;
        }
      }
    } while (nextItem);
  }
// 匿名函数，传入 mod 函数
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror")); // 使用 mod 函数引入 codemirror 模块
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror"], mod); // 使用 define 函数引入 codemirror 模块
  // 如果是普通浏览器环境
  else
    mod(CodeMirror); // 使用 mod 函数传入 CodeMirror 对象
})(function(CodeMirror) {
  // 判断是否是 IE8 及以下版本
  var ie_lt8 = /MSIE \d/.test(navigator.userAgent) &&
    (document.documentMode == null || document.documentMode < 8);
  
  // 定义 Pos 变量为 CodeMirror.Pos
  var Pos = CodeMirror.Pos;
  
  // 定义匹配括号的对象
  var matching = {"(": ")>", ")": "(<", "[": "]>", "]": "[<", "{": "}>", "}": "{<", "<": ">>", ">": "<<"};
  
  // 定义 bracketRegex 函数，根据配置返回括号的正则表达式
  function bracketRegex(config) {
    return config && config.bracketRegex || /[(){}[\]]/
  }
  
  // 定义 findMatchingBracket 函数，查找匹配的括号
  function findMatchingBracket(cm, where, config) {
    // 获取当前行的内容和光标位置
    var line = cm.getLineHandle(where.line), pos = where.ch - 1;
    var afterCursor = config && config.afterCursor
    if (afterCursor == null)
      afterCursor = /(^| )cm-fat-cursor($| )/.test(cm.getWrapperElement().className)
    var re = bracketRegex(config)
  
    // 判断光标位置的字符是否是括号，并且获取匹配的括号
    var match = (!afterCursor && pos >= 0 && re.test(line.text.charAt(pos)) && matching[line.text.charAt(pos)]) ||
        re.test(line.text.charAt(pos + 1)) && matching[line.text.charAt(++pos)];
    if (!match) return null;
    var dir = match.charAt(1) == ">" ? 1 : -1;
    if (config && config.strict && (dir > 0) != (pos == where.ch)) return null;
    var style = cm.getTokenTypeAt(Pos(where.line, pos + 1));
  
    // 在指定方向上扫描匹配的括号
    var found = scanForBracket(cm, Pos(where.line, pos + (dir > 0 ? 1 : 0)), dir, style || null, config);
    if (found == null) return null;
  # 返回一个对象，包含匹配的括号位置信息和是否匹配的信息
  return {from: Pos(where.line, pos), to: found && found.pos,
          match: found && found.ch == match.charAt(0), forward: dir > 0};
}

// bracketRegex is used to specify which type of bracket to scan
// should be a regexp, e.g. /[[\]]/
//
// Note: If "where" is on an open bracket, then this bracket is ignored.
//
// Returns false when no bracket was found, null when it reached
// maxScanLines and gave up
function scanForBracket(cm, where, dir, style, config) {
  var maxScanLen = (config && config.maxScanLineLength) || 10000;
  var maxScanLines = (config && config.maxScanLines) || 1000;

  var stack = [];
  var re = bracketRegex(config)
  var lineEnd = dir > 0 ? Math.min(where.line + maxScanLines, cm.lastLine() + 1)
                        : Math.max(cm.firstLine() - 1, where.line - maxScanLines);
  for (var lineNo = where.line; lineNo != lineEnd; lineNo += dir) {
    var line = cm.getLine(lineNo);
    if (!line) continue;
    var pos = dir > 0 ? 0 : line.length - 1, end = dir > 0 ? line.length : -1;
    if (line.length > maxScanLen) continue;
    if (lineNo == where.line) pos = where.ch - (dir < 0 ? 1 : 0);
    for (; pos != end; pos += dir) {
      var ch = line.charAt(pos);
      if (re.test(ch) && (style === undefined || cm.getTokenTypeAt(Pos(lineNo, pos + 1)) == style)) {
        var match = matching[ch];
        if (match && (match.charAt(1) == ">") == (dir > 0)) stack.push(ch);
        else if (!stack.length) return {pos: Pos(lineNo, pos), ch: ch};
        else stack.pop();
      }
    }
  }
  return lineNo - dir == (dir > 0 ? cm.lastLine() : cm.firstLine()) ? false : null;
}

function matchBrackets(cm, autoclear, config) {
  // Disable brace matching in long lines, since it'll cause hugely slow updates
  var maxHighlightLen = cm.state.matchBrackets.maxHighlightLineLength || 1000;
  var marks = [], ranges = cm.listSelections();
    // 遍历 ranges 数组，对每个元素执行以下操作
    for (var i = 0; i < ranges.length; i++) {
      // 检查当前 range 是否为空，如果是，则查找匹配的括号
      var match = ranges[i].empty() && findMatchingBracket(cm, ranges[i].head, config);
      // 如果找到匹配的括号，并且匹配的行长度小于等于 maxHighlightLen
      if (match && cm.getLine(match.from.line).length <= maxHighlightLen) {
        // 根据匹配结果选择样式
        var style = match.match ? "CodeMirror-matchingbracket" : "CodeMirror-nonmatchingbracket";
        // 在匹配的位置创建标记
        marks.push(cm.markText(match.from, Pos(match.from.line, match.from.ch + 1), {className: style}));
        // 如果有匹配的结束位置，并且结束位置所在行长度小于等于 maxHighlightLen
        if (match.to && cm.getLine(match.to.line).length <= maxHighlightLen)
          // 在结束位置创建标记
          marks.push(cm.markText(match.to, Pos(match.to.line, match.to.ch + 1), {className: style}));
      }
    }

    // 如果存在标记
    if (marks.length) {
      // 修复 IE bug，解决 issue #1193，当触发此事件时，文本输入停止输入到文本区域
      if (ie_lt8 && cm.state.focused) cm.focus();

      // 定义清除标记的函数
      var clear = function() {
        cm.operation(function() {
          for (var i = 0; i < marks.length; i++) marks[i].clear();
        });
      };
      // 如果 autoclear 为真，则延迟 800 毫秒后执行清除函数
      if (autoclear) setTimeout(clear, 800);
      // 否则返回清除函数
      else return clear;
    }
  }

  // 执行匹配括号的操作
  function doMatchBrackets(cm) {
    cm.operation(function() {
      // 如果已经有高亮的匹配括号，则清除
      if (cm.state.matchBrackets.currentlyHighlighted) {
        cm.state.matchBrackets.currentlyHighlighted();
        cm.state.matchBrackets.currentlyHighlighted = null;
      }
      // 执行匹配括号操作
      cm.state.matchBrackets.currentlyHighlighted = matchBrackets(cm, false, cm.state.matchBrackets);
    });
  }

  // 定义 matchBrackets 选项
  CodeMirror.defineOption("matchBrackets", false, function(cm, val, old) {
    // 定义清除函数
    function clear(cm) {
      if (cm.state.matchBrackets && cm.state.matchBrackets.currentlyHighlighted) {
        cm.state.matchBrackets.currentlyHighlighted();
        cm.state.matchBrackets.currentlyHighlighted = null;
      }
    }

    // 如果旧值存在且不等于 CodeMirror.Init
    if (old && old != CodeMirror.Init) {
      // 移除事件监听
      cm.off("cursorActivity", doMatchBrackets);
      cm.off("focus", doMatchBrackets)
      cm.off("blur", clear)
      // 执行清除函数
      clear(cm);
    }
    # 如果val存在
    if (val) {
      # 设置代码镜像的匹配括号状态为val的对象形式，如果val不是对象，则设置为空对象
      cm.state.matchBrackets = typeof val == "object" ? val : {};
      # 当光标活动时执行匹配括号函数
      cm.on("cursorActivity", doMatchBrackets);
      # 当焦点在代码镜像上时执行匹配括号函数
      cm.on("focus", doMatchBrackets)
      # 当代码镜像失去焦点时清除匹配括号
      cm.on("blur", clear)
    }
  });

  # 定义代码镜像的匹配括号函数
  CodeMirror.defineExtension("matchBrackets", function() {matchBrackets(this, true);});
  # 定义查找匹配括号的函数
  CodeMirror.defineExtension("findMatchingBracket", function(pos, config, oldConfig){
    # 向后兼容的修补措施
    if (oldConfig || typeof config == "boolean") {
      # 如果没有旧配置，根据config的布尔值设置严格模式，或者设置为空
      if (!oldConfig) {
        config = config ? {strict: true} : null
      } else {
        # 如果有旧配置，将旧配置的严格模式设置为config的值
        oldConfig.strict = config
        config = oldConfig
      }
    }
    # 返回查找到的匹配括号的位置
    return findMatchingBracket(this, pos, config)
  });
  # 定义扫描括号的函数
  CodeMirror.defineExtension("scanForBracket", function(pos, dir, style, config){
    # 返回扫描到的括号位置
    return scanForBracket(this, pos, dir, style, config);
  });
// 匿名函数，传入 mod 参数
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror"), require("../fold/xml-fold"));
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror", "../fold/xml-fold"], mod);
  // 如果是普通的浏览器环境
  else
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义 CodeMirror 的 matchTags 选项
  CodeMirror.defineOption("matchTags", false, function(cm, val, old) {
    // 如果旧值存在且不是初始化值
    if (old && old != CodeMirror.Init) {
      // 移除光标活动事件的监听器
      cm.off("cursorActivity", doMatchTags);
      // 移除视口变化事件的监听器
      cm.off("viewportChange", maybeUpdateMatch);
      // 清除标签匹配
      clear(cm);
    }
    // 如果新值存在
    if (val) {
      // 设置匹配标签的状态
      cm.state.matchBothTags = typeof val == "object" && val.bothTags;
      // 添加光标活动事件的监听器
      cm.on("cursorActivity", doMatchTags);
      // 添加视口变化事件的监听器
      cm.on("viewportChange", maybeUpdateMatch);
      // 执行标签匹配
      doMatchTags(cm);
    }
  });

  // 清除标签匹配
  function clear(cm) {
    if (cm.state.tagHit) cm.state.tagHit.clear();
    if (cm.state.tagOther) cm.state.tagOther.clear();
    cm.state.tagHit = cm.state.tagOther = null;
  }

  // 执行标签匹配
  function doMatchTags(cm) {
    cm.state.failedTagMatch = false;
    # 执行操作函数
    cm.operation(function() {
      # 清除编辑器中的选择
      clear(cm);
      # 如果有选中内容，则直接返回
      if (cm.somethingSelected()) return;
      # 获取当前光标位置和可视区域范围
      var cur = cm.getCursor(), range = cm.getViewport();
      range.from = Math.min(range.from, cur.line); range.to = Math.max(cur.line + 1, range.to);
      # 查找匹配的标签
      var match = CodeMirror.findMatchingTag(cm, cur, range);
      # 如果没有找到匹配的标签，则直接返回
      if (!match) return;
      # 如果编辑器状态中包含匹配双标签的信息
      if (cm.state.matchBothTags) {
        # 获取匹配的标签
        var hit = match.at == "open" ? match.open : match.close;
        # 如果找到匹配的标签，则在编辑器中标记出来
        if (hit) cm.state.tagHit = cm.markText(hit.from, hit.to, {className: "CodeMirror-matchingtag"});
      }
      # 获取另一半匹配的标签
      var other = match.at == "close" ? match.open : match.close;
      # 如果找到另一半匹配的标签，则在编辑器中标记出来
      if (other)
        cm.state.tagOther = cm.markText(other.from, other.to, {className: "CodeMirror-matchingtag"});
      else
        # 如果没有找到另一半匹配的标签，则在编辑器状态中标记匹配失败
        cm.state.failedTagMatch = true;
    });
  }

  # 可能更新匹配的标签
  function maybeUpdateMatch(cm) {
    # 如果编辑器状态中标记了匹配失败，则重新匹配标签
    if (cm.state.failedTagMatch) doMatchTags(cm);
  }

  # 定义命令，跳转到匹配的标签
  CodeMirror.commands.toMatchingTag = function(cm) {
    # 查找匹配的标签
    var found = CodeMirror.findMatchingTag(cm, cm.getCursor());
    # 如果找到匹配的标签
    if (found) {
      # 获取另一半匹配的标签
      var other = found.at == "close" ? found.open : found.close;
      # 如果找到另一半匹配的标签，则将光标扩展到另一半标签的位置
      if (other) cm.extendSelection(other.to, other.from);
    }
  };
// 定义了一个匿名函数，该函数用于在 CodeMirror 中添加一个选项，用于显示或隐藏行尾空格
(function(mod) {
  // 如果是 CommonJS 环境，使用 require 导入 CodeMirror 模块
  if (typeof exports == "object" && typeof module == "object") 
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境，使用 define 导入 CodeMirror 模块
  else if (typeof define == "function" && define.amd) 
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境，直接使用 CodeMirror 对象
  else 
    mod(CodeMirror);
})(function(CodeMirror) {
  // 定义一个名为 showTrailingSpace 的选项，初始值为 false
  CodeMirror.defineOption("showTrailingSpace", false, function(cm, val, prev) {
    // 如果之前的值是初始化状态，则将其设为 false
    if (prev == CodeMirror.Init) prev = false;
    // 如果之前的值为 true，且当前值为 false，则移除 trailingspace 的覆盖
    if (prev && !val)
      cm.removeOverlay("trailingspace");
    // 如果之前的值为 false，且当前值为 true，则添加 trailingspace 的覆盖
    else if (!prev && val)
      cm.addOverlay({
        // 定义 token 函数，用于标记行尾空格
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



// 定义了一个匿名函数，该函数用于在 CodeMirror 中注册一个名为 brace 的折叠辅助函数
(function(mod) {
  // 如果是 CommonJS 环境，使用 require 导入 CodeMirror 模块
  if (typeof exports == "object" && typeof module == "object") 
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境，使用 define 导入 CodeMirror 模块
  else if (typeof define == "function" && define.amd) 
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境，直接使用 CodeMirror 对象
  else 
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";
  // 注册一个名为 brace 的折叠辅助函数
  CodeMirror.registerHelper("fold", "brace", function(cm, start) {
    var line = start.line, lineText = cm.getLine(line);
    var tokenType;
    // 查找匹配的开放括号
    function findOpening(openCh) {
    # 从起始位置开始遍历
    for (var at = start.ch, pass = 0;;) {
      # 在当前行文本中查找指定字符的位置
      var found = at <= 0 ? -1 : lineText.lastIndexOf(openCh, at - 1);
      # 如果未找到指定字符
      if (found == -1) {
        # 如果已经遍历两次，则退出循环
        if (pass == 1) break;
        pass = 1;
        at = lineText.length;
        continue;
      }
      # 如果已经遍历两次且找到的位置在起始位置之前，则退出循环
      if (pass == 1 && found < start.ch) break;
      # 获取指定位置的 token 类型
      tokenType = cm.getTokenTypeAt(CodeMirror.Pos(line, found + 1));
      # 如果 token 类型不是注释或字符串，则返回找到的位置
      if (!/^(comment|string)/.test(tokenType)) return found + 1;
      at = found - 1;
    }
  }

  # 设置起始和结束的 token
  var startToken = "{", endToken = "}", startCh = findOpening("{");
  # 如果未找到起始 token，则设置新的起始和结束 token，并重新查找起始位置
  if (startCh == null) {
    startToken = "[", endToken = "]";
    startCh = findOpening("[");
  }

  # 如果未找到起始位置，则返回
  if (startCh == null) return;
  var count = 1, lastLine = cm.lastLine(), end, endCh;
  # 遍历每一行文本
  outer: for (var i = line; i <= lastLine; ++i) {
    var text = cm.getLine(i), pos = i == line ? startCh : 0;
    for (;;) {
      var nextOpen = text.indexOf(startToken, pos), nextClose = text.indexOf(endToken, pos);
      if (nextOpen < 0) nextOpen = text.length;
      if (nextClose < 0) nextClose = text.length;
      pos = Math.min(nextOpen, nextClose);
      if (pos == text.length) break;
      # 获取指定位置的 token 类型
      if (cm.getTokenTypeAt(CodeMirror.Pos(i, pos + 1)) == tokenType) {
        if (pos == nextOpen) ++count;
        else if (!--count) { end = i; endCh = pos; break outer; }
      }
      ++pos;
    }
  }
  # 如果未找到结束位置或起始和结束在同一行，则返回
  if (end == null || line == end) return;
  # 返回起始和结束位置的对象
  return {from: CodeMirror.Pos(line, startCh),
          to: CodeMirror.Pos(end, endCh)};
// 注册一个名为 "import" 的折叠辅助函数
CodeMirror.registerHelper("fold", "import", function(cm, start) {
  // 检查指定行是否包含 import 关键字
  function hasImport(line) {
    if (line < cm.firstLine() || line > cm.lastLine()) return null;
    // 获取指定行的第一个 token
    var start = cm.getTokenAt(CodeMirror.Pos(line, 1));
    // 如果第一个 token 是空白字符，则获取下一个 token
    if (!/\S/.test(start.string)) start = cm.getTokenAt(CodeMirror.Pos(line, start.end + 1));
    // 如果第一个 token 不是关键字 import，则返回 null
    if (start.type != "keyword" || start.string != "import") return null;
    // 寻找下一个分号的位置，并返回其位置
    for (var i = line, e = Math.min(cm.lastLine(), line + 10); i <= e; ++i) {
      var text = cm.getLine(i), semi = text.indexOf(";");
      if (semi != -1) return {startCh: start.end, end: CodeMirror.Pos(i, semi)};
    }
  }

  // 获取起始行的行号和是否包含 import 的结果
  var startLine = start.line, has = hasImport(startLine), prev;
  // 如果没有 import 或者前一行也包含 import 或者前一行的结束位置和当前行相同，则返回 null
  if (!has || hasImport(startLine - 1) || ((prev = hasImport(startLine - 2)) && prev.end.line == startLine - 1))
    return null;
  // 寻找 import 块的结束位置
  for (var end = has.end;;) {
    var next = hasImport(end.line + 1);
    if (next == null) break;
    end = next.end;
  }
  // 返回 import 块的起始位置和结束位置
  return {from: cm.clipPos(CodeMirror.Pos(startLine, has.startCh + 1)), to: end};
});

// 注册一个名为 "include" 的折叠辅助函数
CodeMirror.registerHelper("fold", "include", function(cm, start) {
  // 检查指定行是否包含 #include
  function hasInclude(line) {
    if (line < cm.firstLine() || line > cm.lastLine()) return null;
    // 获取指定行的第一个 token
    var start = cm.getTokenAt(CodeMirror.Pos(line, 1));
    // 如果第一个 token 是空白字符，则获取下一个 token
    if (!/\S/.test(start.string)) start = cm.getTokenAt(CodeMirror.Pos(line, start.end + 1));
    // 如果第一个 token 是 meta 类型且以 #include 开头，则返回起始位置
    if (start.type == "meta" && start.string.slice(0, 8) == "#include") return start.start + 8;
  }

  // 获取起始行的行号和是否包含 #include 的结果
  var startLine = start.line, has = hasInclude(startLine);
  // 如果没有 #include 或者前一行也包含 #include，则返回 null
  if (has == null || hasInclude(startLine - 1) != null) return null;
  // 寻找 #include 块的结束位置
  for (var end = startLine;;) {
    var next = hasInclude(end + 1);
    if (next == null) break;
    ++end;
  }
  // 返回 #include 块的起始位置和结束位置
  return {from: CodeMirror.Pos(startLine, has + 1),
          to: cm.clipPos(CodeMirror.Pos(end))};
});
// 使用 MIT 许可证分发
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

  // 注册全局的折叠帮助函数，用于折叠注释
  CodeMirror.registerGlobalHelper("fold", "comment", function(mode) {
    // 判断当前模式是否有块注释的开始和结束标记
    return mode.blockCommentStart && mode.blockCommentEnd;
  }, function(cm, start) {
    // 获取当前光标位置的模式和块注释的开始和结束标记
    var mode = cm.getModeAt(start), startToken = mode.blockCommentStart, endToken = mode.blockCommentEnd;
    if (!startToken || !endToken) return;
    var line = start.line, lineText = cm.getLine(line);

    var startCh;
    // 查找块注释的开始位置
    for (var at = start.ch, pass = 0;;) {
      var found = at <= 0 ? -1 : lineText.lastIndexOf(startToken, at - 1);
      if (found == -1) {
        if (pass == 1) return;
        pass = 1;
        at = lineText.length;
        continue;
      }
      if (pass == 1 && found < start.ch) return;
      // 判断找到的开始位置是否是注释类型
      if (/comment/.test(cm.getTokenTypeAt(CodeMirror.Pos(line, found + 1))) &&
          (found == 0 || lineText.slice(found - endToken.length, found) == endToken ||
           !/comment/.test(cm.getTokenTypeAt(CodeMirror.Pos(line, found))))) {
        startCh = found + startToken.length;
        break;
      }
      at = found - 1;
    }

    var depth = 1, lastLine = cm.lastLine(), end, endCh;
    // 遍历每一行，查找块注释的结束位置
    outer: for (var i = line; i <= lastLine; ++i) {
      var text = cm.getLine(i), pos = i == line ? startCh : 0;
      for (;;) {
        var nextOpen = text.indexOf(startToken, pos), nextClose = text.indexOf(endToken, pos);
        if (nextOpen < 0) nextOpen = text.length;
        if (nextClose < 0) nextClose = text.length;
        pos = Math.min(nextOpen, nextClose);
        if (pos == text.length) break;
        if (pos == nextOpen) ++depth;
        else if (!--depth) { end = i; endCh = pos; break outer; }
        ++pos;
      }
    }
    // 折叠块注释
    cm.foldCode(CodeMirror.Pos(line, startCh), CodeMirror.Pos(end, endCh));
  });
});
    }
  }
  // 如果结束位置为空，或者行数等于结束行数并且结束字符等于开始字符，则返回
  if (end == null || line == end && endCh == startCh) return;
  // 返回一个对象，包含起始位置和结束位置
  return {from: CodeMirror.Pos(line, startCh),
          to: CodeMirror.Pos(end, endCh)};
// 定义一个匿名函数，传入一个 mod 参数
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境
  else
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 折叠代码的函数
  function doFold(cm, pos, options, force) {
    // 如果 options 存在并且是一个函数
    if (options && options.call) {
      var finder = options;
      options = null;
    } else {
      var finder = getOption(cm, options, "rangeFinder");
    }
    // 如果 pos 是一个数字，将其转换为 CodeMirror.Pos 对象
    if (typeof pos == "number") pos = CodeMirror.Pos(pos, 0);
    // 获取最小折叠大小
    var minSize = getOption(cm, options, "minFoldSize");

    // 获取折叠范围
    function getRange(allowFolded) {
      var range = finder(cm, pos);
      if (!range || range.to.line - range.from.line < minSize) return null;
      var marks = cm.findMarksAt(range.from);
      for (var i = 0; i < marks.length; ++i) {
        if (marks[i].__isFold && force !== "fold") {
          if (!allowFolded) return null;
          range.cleared = true;
          marks[i].clear();
        }
      }
      return range;
    }

    var range = getRange(true);
    // 如果设置了 scanUp 选项，向上扫描
    if (getOption(cm, options, "scanUp")) while (!range && pos.line > cm.firstLine()) {
      pos = CodeMirror.Pos(pos.line - 1, 0);
      range = getRange(false);
    }
    // 如果没有范围或者已经清除，或者强制展开，则返回
    if (!range || range.cleared || force === "unfold") return;

    // 创建折叠小部件
    var myWidget = makeWidget(cm, options, range);
    // 监听鼠标按下事件
    CodeMirror.on(myWidget, "mousedown", function(e) {
      myRange.clear();
      CodeMirror.e_preventDefault(e);
    });
    // 标记文本范围
    var myRange = cm.markText(range.from, range.to, {
      replacedWith: myWidget,
      clearOnEnter: getOption(cm, options, "clearOnEnter"),
      __isFold: true
    });
    // 监听清除事件
    myRange.on("clear", function(from, to) {
      CodeMirror.signal(cm, "unfold", cm, from, to);
  });

  // 触发 CodeMirror 的折叠事件
  CodeMirror.signal(cm, "fold", cm, range.from, range.to);
}

// 创建折叠小部件
function makeWidget(cm, options, range) {
  // 从选项中获取小部件
  var widget = getOption(cm, options, "widget");

  // 如果小部件是函数，则调用函数获取小部件
  if (typeof widget == "function") {
    widget = widget(range.from, range.to);
  }

  // 如果小部件是字符串，则创建文本节点和 span 元素
  if (typeof widget == "string") {
    var text = document.createTextNode(widget);
    widget = document.createElement("span");
    widget.appendChild(text);
    widget.className = "CodeMirror-foldmarker";
  } else if (widget) {
    widget = widget.cloneNode(true)
  }
  return widget;
}

// 兼容老版本的折叠函数
CodeMirror.newFoldFunction = function(rangeFinder, widget) {
  return function(cm, pos) { doFold(cm, pos, {rangeFinder: rangeFinder, widget: widget}); };
};

// 新版本的折叠函数
CodeMirror.defineExtension("foldCode", function(pos, options, force) {
  doFold(this, pos, options, force);
});

// 判断指定位置是否被折叠
CodeMirror.defineExtension("isFolded", function(pos) {
  var marks = this.findMarksAt(pos);
  for (var i = 0; i < marks.length; ++i)
    if (marks[i].__isFold) return true;
});

// 折叠命令
CodeMirror.commands.toggleFold = function(cm) {
  cm.foldCode(cm.getCursor());
};
CodeMirror.commands.fold = function(cm) {
  cm.foldCode(cm.getCursor(), null, "fold");
};
CodeMirror.commands.unfold = function(cm) {
  cm.foldCode(cm.getCursor(), null, "unfold");
};
CodeMirror.commands.foldAll = function(cm) {
  cm.operation(function() {
    for (var i = cm.firstLine(), e = cm.lastLine(); i <= e; i++)
      cm.foldCode(CodeMirror.Pos(i, 0), null, "fold");
  });
};
CodeMirror.commands.unfoldAll = function(cm) {
  cm.operation(function() {
    for (var i = cm.firstLine(), e = cm.lastLine(); i <= e; i++)
      cm.foldCode(CodeMirror.Pos(i, 0), null, "unfold");
  });
};

// 注册折叠辅助函数
CodeMirror.registerHelper("fold", "combine", function() {
  var funcs = Array.prototype.slice.call(arguments, 0);
    // 定义一个函数，接受参数 cm 和 start
    return function(cm, start) {
      // 遍历 funcs 数组
      for (var i = 0; i < funcs.length; ++i) {
        // 调用 funcs[i] 函数，传入参数 cm 和 start
        var found = funcs[i](cm, start);
        // 如果找到结果，则返回
        if (found) return found;
      }
    };
  });

  // 注册一个自动折叠的辅助函数
  CodeMirror.registerHelper("fold", "auto", function(cm, start) {
    // 获取指定位置的折叠辅助函数
    var helpers = cm.getHelpers(start, "fold");
    // 遍历折叠辅助函数数组
    for (var i = 0; i < helpers.length; i++) {
      // 调用折叠辅助函数，传入参数 cm 和 start
      var cur = helpers[i](cm, start);
      // 如果找到结果，则返回
      if (cur) return cur;
    }
  });

  // 定义默认选项对象
  var defaultOptions = {
    rangeFinder: CodeMirror.fold.auto,
    widget: "\u2194",
    minFoldSize: 0,
    scanUp: false,
    clearOnEnter: true
  };

  // 定义折叠选项
  CodeMirror.defineOption("foldOptions", null);

  // 获取选项值的函数
  function getOption(cm, options, name) {
    // 如果 options 中存在指定选项，则返回其值
    if (options && options[name] !== undefined)
      return options[name];
    // 否则，获取编辑器的折叠选项值
    var editorOptions = cm.options.foldOptions;
    if (editorOptions && editorOptions[name] !== undefined)
      return editorOptions[name];
    // 否则，返回默认选项值
    return defaultOptions[name];
  }

  // 定义获取折叠选项值的扩展函数
  CodeMirror.defineExtension("foldOption", function(options, name) {
    return getOption(this, options, name);
  });
// 定义一个匿名函数，传入 mod 参数
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror"), require("./foldcode"));
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror", "./foldcode"], mod);
  // 如果是普通的浏览器环境
  else
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义 foldGutter 选项
  CodeMirror.defineOption("foldGutter", false, function(cm, val, old) {
    // 如果旧值存在且不是初始化值
    if (old && old != CodeMirror.Init) {
      // 清除折叠边栏
      cm.clearGutter(cm.state.foldGutter.options.gutter);
      cm.state.foldGutter = null;
      // 移除事件监听
      cm.off("gutterClick", onGutterClick);
      cm.off("changes", onChange);
      cm.off("viewportChange", onViewportChange);
      cm.off("fold", onFold);
      cm.off("unfold", onFold);
      cm.off("swapDoc", onChange);
    }
    // 如果新值存在
    if (val) {
      // 创建 foldGutter 状态对象
      cm.state.foldGutter = new State(parseOptions(val));
      // 更新视口中的折叠状态
      updateInViewport(cm);
      // 添加事件监听
      cm.on("gutterClick", onGutterClick);
      cm.on("changes", onChange);
      cm.on("viewportChange", onViewportChange);
      cm.on("fold", onFold);
      cm.on("unfold", onFold);
      cm.on("swapDoc", onChange);
    }
  });

  // 定义 Pos 函数
  var Pos = CodeMirror.Pos;

  // 定义 State 构造函数
  function State(options) {
    this.options = options;
    this.from = this.to = 0;
  }

  // 解析选项
  function parseOptions(opts) {
    if (opts === true) opts = {};
    if (opts.gutter == null) opts.gutter = "CodeMirror-foldgutter";
    if (opts.indicatorOpen == null) opts.indicatorOpen = "CodeMirror-foldgutter-open";
    if (opts.indicatorFolded == null) opts.indicatorFolded = "CodeMirror-foldgutter-folded";
    return opts;
  }

  // 判断行是否被折叠
  function isFolded(cm, line) {
    var marks = cm.findMarks(Pos(line, 0), Pos(line + 1, 0));
    # 遍历 marks 数组
    for (var i = 0; i < marks.length; ++i) {
      # 如果 marks[i] 的 __isFold 属性为真
      if (marks[i].__isFold) {
        # 查找 marks[i] 中值为 -1 的位置
        var fromPos = marks[i].find(-1);
        # 如果 fromPos 存在且其行号等于 line
        if (fromPos && fromPos.line === line)
          # 返回 marks[i]
          return marks[i];
      }
    }
  }

  # 根据 spec 参数创建 marker 元素
  function marker(spec) {
    # 如果 spec 的类型为字符串
    if (typeof spec == "string") {
      # 创建一个 div 元素
      var elt = document.createElement("div");
      # 设置 div 元素的类名
      elt.className = spec + " CodeMirror-guttermarker-subtle";
      # 返回创建的 div 元素
      return elt;
    } else {
      # 返回 spec 的克隆节点
      return spec.cloneNode(true);
    }
  }

  # 更新折叠信息
  function updateFoldInfo(cm, from, to) {
    # 获取折叠选项和当前行号
    var opts = cm.state.foldGutter.options, cur = from - 1;
    # 获取最小折叠大小和范围查找器
    var minSize = cm.foldOption(opts, "minFoldSize");
    var func = cm.foldOption(opts, "rangeFinder");
    # 如果内置指示器元素的类名与新状态匹配，则可以重用它
    var clsFolded = typeof opts.indicatorFolded == "string" && classTest(opts.indicatorFolded);
    var clsOpen = typeof opts.indicatorOpen == "string" && classTest(opts.indicatorOpen);
    # 遍历 from 到 to 之间的每一行
    cm.eachLine(from, to, function(line) {
      ++cur;
      var mark = null;
      var old = line.gutterMarkers;
      # 如果 old 存在且其类名与 gutter 匹配
      if (old) old = old[opts.gutter];
      # 如果当前行被折叠
      if (isFolded(cm, cur)) {
        # 如果 clsFolded 存在且 old 存在且其类名与 clsFolded 匹配，则返回
        if (clsFolded && old && clsFolded.test(old.className)) return;
        # 创建折叠指示器元素
        mark = marker(opts.indicatorFolded);
      } else {
        # 获取当前行的位置和范围
        var pos = Pos(cur, 0);
        var range = func && func(cm, pos);
        # 如果范围存在且其行数大于等于最小折叠大小
        if (range && range.to.line - range.from.line >= minSize) {
          # 如果 clsOpen 存在且 old 存在且其类名与 clsOpen 匹配，则返回
          if (clsOpen && old && clsOpen.test(old.className)) return;
          # 创建展开指示器元素
          mark = marker(opts.indicatorOpen);
        }
      }
      # 如果 mark 和 old 都不存在，则返回
      if (!mark && !old) return;
      # 设置当前行的 gutterMarker
      cm.setGutterMarker(line, opts.gutter, mark);
    });
  }

  # 从 CodeMirror/src/util/dom.js 复制过来的函数
  function classTest(cls) { return new RegExp("(^|\\s)" + cls + "(?:$|\\s)\\s*") }

  # 更新视口中的内容
  function updateInViewport(cm) {
    # 获取编辑器的视口和折叠状态
    var vp = cm.getViewport(), state = cm.state.foldGutter;
    # 如果折叠状态不存在，则返回
    if (!state) return;
    # 执行更新折叠信息的操作
    cm.operation(function() {
      updateFoldInfo(cm, vp.from, vp.to);
    });
  // 将视口的起始和结束行号赋值给状态对象
  state.from = vp.from; state.to = vp.to;
}

function onGutterClick(cm, line, gutter) {
  // 获取折叠栏状态对象
  var state = cm.state.foldGutter;
  // 如果状态对象不存在，则返回
  if (!state) return;
  // 获取状态对象的选项
  var opts = state.options;
  // 如果点击的折叠栏不是指定的折叠栏，则返回
  if (gutter != opts.gutter) return;
  // 判断当前行是否已经折叠，如果是则清除折叠，否则折叠当前行
  var folded = isFolded(cm, line);
  if (folded) folded.clear();
  else cm.foldCode(Pos(line, 0), opts);
}

function onChange(cm) {
  // 获取折叠栏状态对象
  var state = cm.state.foldGutter;
  // 如果状态对象不存在，则返回
  if (!state) return;
  // 获取状态对象的选项
  var opts = state.options;
  // 将状态对象的起始和结束行号都设置为0
  state.from = state.to = 0;
  // 清除之前的定时器，设置新的定时器来更新视口中的折叠信息
  clearTimeout(state.changeUpdate);
  state.changeUpdate = setTimeout(function() { updateInViewport(cm); }, opts.foldOnChangeTimeSpan || 600);
}

function onViewportChange(cm) {
  // 获取折叠栏状态对象
  var state = cm.state.foldGutter;
  // 如果状态对象不存在，则返回
  if (!state) return;
  // 获取状态对象的选项
  var opts = state.options;
  // 清除之前的定时器，设置新的定时器来更新视口中的折叠信息
  clearTimeout(state.changeUpdate);
  state.changeUpdate = setTimeout(function() {
    // 获取当前视口的起始和结束行号
    var vp = cm.getViewport();
    // 如果状态对象的起始和结束行号相等，或者视口的起始行号与状态对象的结束行号之差大于20，或者状态对象的起始行号与视口的结束行号之差大于20，则更新视口中的折叠信息
    if (state.from == state.to || vp.from - state.to > 20 || state.from - vp.to > 20) {
      updateInViewport(cm);
    } else {
      // 否则，进行操作来更新折叠信息
      cm.operation(function() {
        if (vp.from < state.from) {
          updateFoldInfo(cm, vp.from, state.from);
          state.from = vp.from;
        }
        if (vp.to > state.to) {
          updateFoldInfo(cm, state.to, vp.to);
          state.to = vp.to;
        }
      });
    }
  }, opts.updateViewportTimeSpan || 400);
}

function onFold(cm, from) {
  // 获取折叠栏状态对象
  var state = cm.state.foldGutter;
  // 如果状态对象不存在，则返回
  if (!state) return;
  // 获取折叠的行号
  var line = from.line;
  // 如果折叠的行号在状态对象的起始和结束行号之间，则更新折叠信息
  if (line >= state.from && line < state.to)
    updateFoldInfo(cm, line, line + 1);
}
// 定义一个匿名函数，传入一个 mod 参数
(function(mod) {
  // 如果是 CommonJS 环境，使用 mod 函数
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境，使用 define 函数
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境，直接使用 CodeMirror 对象
  else
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义函数 lineIndent，用于获取指定行的缩进
  function lineIndent(cm, lineNo) {
    var text = cm.getLine(lineNo)
    var spaceTo = text.search(/\S/)
    // 如果没有找到非空白字符，或者该行是注释行，则返回 -1
    if (spaceTo == -1 || /\bcomment\b/.test(cm.getTokenTypeAt(CodeMirror.Pos(lineNo, spaceTo + 1))))
      return -1
    // 否则返回该行的缩进
    return CodeMirror.countColumn(text, null, cm.getOption("tabSize"))
  }

  // 注册折叠功能，根据缩进进行折叠
  CodeMirror.registerHelper("fold", "indent", function(cm, start) {
    var myIndent = lineIndent(cm, start.line)
    // 如果该行没有缩进，返回
    if (myIndent < 0) return
    var lastLineInFold = null

    // 遍历行，找到属于同一折叠块的最后一行
    for (var i = start.line + 1, end = cm.lastLine(); i <= end; ++i) {
      var indent = lineIndent(cm, i)
      if (indent == -1) {
        // 如果是注释行，继续遍历
      } else if (indent > myIndent) {
        // 如果缩进大于起始行的缩进，认为是同一折叠块的行
        lastLineInFold = i;
      } else {
        // 如果该行有非空白字符，并且缩进小于等于起始行，认为是另一个折叠块的开始
        break;
      }
    }
    // 返回折叠的起始和结束位置
    if (lastLineInFold) return {
      from: CodeMirror.Pos(start.line, cm.getLine(start.line).length),
      to: CodeMirror.Pos(lastLineInFold, cm.getLine(lastLineInFold).length)
    };
  });

});
(function(mod) {
  if (typeof exports == "object" && typeof module == "object") // CommonJS
    mod(require("../../lib/codemirror"));
  else if (typeof define == "function" && define.amd) // AMD
    define(["../../lib/codemirror"], mod);
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
"use strict";

CodeMirror.registerHelper("fold", "markdown", function(cm, start) {
  var maxDepth = 100;

  function isHeader(lineNo) {
    var tokentype = cm.getTokenTypeAt(CodeMirror.Pos(lineNo, 0));
    return tokentype && /\bheader\b/.test(tokentype);
  }

  function headerLevel(lineNo, line, nextLine) {
    var match = line && line.match(/^#+/);
    if (match && isHeader(lineNo)) return match[0].length;
    match = nextLine && nextLine.match(/^[=\-]+\s*$/);
    if (match && isHeader(lineNo + 1)) return nextLine[0] == "=" ? 1 : 2;
    return maxDepth;
  }

  var firstLine = cm.getLine(start.line), nextLine = cm.getLine(start.line + 1);
  var level = headerLevel(start.line, firstLine, nextLine);
  if (level === maxDepth) return undefined;

  var lastLineNo = cm.lastLine();
  var end = start.line, nextNextLine = cm.getLine(end + 2);
  while (end < lastLineNo) {
    if (headerLevel(end + 1, nextLine, nextNextLine) <= level) break;
    ++end;
    nextLine = nextNextLine;
    nextNextLine = cm.getLine(end + 2);
  }

  return {
    from: CodeMirror.Pos(start.line, firstLine.length),
    to: CodeMirror.Pos(end, cm.getLine(end).length)
  };
});

});


/* ---- extension/fold/xml-fold.js ---- */


// CodeMirror, copyright (c) by Marijn Haverbeke and others
// Distributed under an MIT license: https://codemirror.net/LICENSE

(function(mod) {
  if (typeof exports == "object" && typeof module == "object") // CommonJS
    mod(require("../../lib/codemirror"));
  else if (typeof define == "function" && define.amd) // AMD
    define(["../../lib/codemirror"], mod);
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  var Pos = CodeMirror.Pos;  # 定义变量 Pos 为 CodeMirror.Pos
  function cmp(a, b) { return a.line - b.line || a.ch - b.ch; }  # 定义函数 cmp，用于比较两个位置的行和列

  var nameStartChar = "A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD";
  var nameChar = nameStartChar + "\-\:\.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040";
  var xmlTagStart = new RegExp("<(/?)([" + nameStartChar + "][" + nameChar + "]*)", "g");  # 定义正则表达式 xmlTagStart，用于匹配 XML 标签的起始部分

  function Iter(cm, line, ch, range) {  # 定义迭代器 Iter
    this.line = line; this.ch = ch;
    this.cm = cm; this.text = cm.getLine(line);
    this.min = range ? Math.max(range.from, cm.firstLine()) : cm.firstLine();
    this.max = range ? Math.min(range.to - 1, cm.lastLine()) : cm.lastLine();
  }

  function tagAt(iter, ch) {  # 定义函数 tagAt，用于判断指定位置是否为标签
    var type = iter.cm.getTokenTypeAt(Pos(iter.line, ch));
    return type && /\btag\b/.test(type);
  }

  function nextLine(iter) {  # 定义函数 nextLine，用于获取下一行的内容
    if (iter.line >= iter.max) return;
    iter.ch = 0;
    iter.text = iter.cm.getLine(++iter.line);
    return true;
  }
  function prevLine(iter) {  # 定义函数 prevLine，用于获取上一行的内容
    if (iter.line <= iter.min) return;
    iter.text = iter.cm.getLine(--iter.line);
    iter.ch = iter.text.length;
    return true;
  }

  function toTagEnd(iter) {  # 定义函数 toTagEnd，用于定位标签的结束位置
    for (;;) {
      var gt = iter.text.indexOf(">", iter.ch);
      if (gt == -1) { if (nextLine(iter)) continue; else return; }
      if (!tagAt(iter, gt + 1)) { iter.ch = gt + 1; continue; }
      var lastSlash = iter.text.lastIndexOf("/", gt);
      var selfClose = lastSlash > -1 && !/\S/.test(iter.text.slice(lastSlash + 1, gt));
      iter.ch = gt + 1;
      return selfClose ? "selfClose" : "regular";
    }
  }
  function toTagStart(iter) {  # 定义函数 toTagStart，用于定位标签的起始位置
  // 无限循环，查找最近的 "<" 符号的位置
  for (;;) {
    // 如果迭代器中有字符，则查找最近的 "<" 符号的位置
    var lt = iter.ch ? iter.text.lastIndexOf("<", iter.ch - 1) : -1;
    // 如果找不到 "<" 符号，则继续查找上一行
    if (lt == -1) { if (prevLine(iter)) continue; else return; }
    // 如果最近的 "<" 符号不是标签的起始位置，则将迭代器的位置移到该符号处
    if (!tagAt(iter, lt + 1)) { iter.ch = lt; continue; }
    // 设置正则表达式的起始位置，将迭代器的位置移到最近的 "<" 符号处
    xmlTagStart.lastIndex = lt;
    iter.ch = lt;
    // 在文本中查找匹配的标签，并返回匹配结果
    var match = xmlTagStart.exec(iter.text);
    if (match && match.index == lt) return match;
  }
}

// 移动迭代器到下一个标签的位置
function toNextTag(iter) {
  // 无限循环，查找下一个标签的位置
  for (;;) {
    // 设置正则表达式的起始位置，查找下一个标签的位置
    xmlTagStart.lastIndex = iter.ch;
    var found = xmlTagStart.exec(iter.text);
    // 如果找不到下一个标签，则继续查找下一行
    if (!found) { if (nextLine(iter)) continue; else return; }
    // 如果找到的标签不是有效的标签，则将迭代器的位置移到下一个位置
    if (!tagAt(iter, found.index + 1)) { iter.ch = found.index + 1; continue; }
    // 将迭代器的位置移到下一个标签的末尾
    iter.ch = found.index + found[0].length;
    return found;
  }
}

// 移动迭代器到上一个标签的位置
function toPrevTag(iter) {
  // 无限循环，查找上一个标签的位置
  for (;;) {
    // 如果迭代器中有字符，则查找最近的 ">" 符号的位置
    var gt = iter.ch ? iter.text.lastIndexOf(">", iter.ch - 1) : -1;
    // 如果找不到 ">" 符号，则继续查找上一行
    if (gt == -1) { if (prevLine(iter)) continue; else return; }
    // 如果最近的 ">" 符号不是标签的结束位置，则将迭代器的位置移到该符号处
    if (!tagAt(iter, gt + 1)) { iter.ch = gt; continue; }
    // 查找最近的 "/" 符号的位置
    var lastSlash = iter.text.lastIndexOf("/", gt);
    // 判断标签是否是自闭合标签
    var selfClose = lastSlash > -1 && !/\S/.test(iter.text.slice(lastSlash + 1, gt));
    // 将迭代器的位置移到 ">" 符号的下一个位置
    iter.ch = gt + 1;
    // 如果是自闭合标签，则返回 "selfClose"，否则返回 "regular"
    return selfClose ? "selfClose" : "regular";
  }
}

// 查找匹配的闭合标签
function findMatchingClose(iter, tag) {
  var stack = [];
  // 无限循环，查找匹配的闭合标签
  for (;;) {
    // 移动迭代器到下一个标签的位置，并记录起始位置
    var next = toNextTag(iter), end, startLine = iter.line, startCh = iter.ch - (next ? next[0].length : 0);
    // 如果找不到下一个标签，或者找不到匹配的结束标签，则结束循环
    if (!next || !(end = toTagEnd(iter))) return;
    // 如果是自闭合标签，则继续下一次循环
    if (end == "selfClose") continue;
    // 如果是闭合标签，则将其从堆栈中移除
    if (next[1]) { // closing tag
      for (var i = stack.length - 1; i >= 0; --i) if (stack[i] == next[2]) {
        stack.length = i;
        break;
      }
      // 如果堆栈中没有对应的开始标签，则返回匹配结果
      if (i < 0 && (!tag || tag == next[2])) return {
        tag: next[2],
        from: Pos(startLine, startCh),
        to: Pos(iter.line, iter.ch)
      };
    } else { // opening tag
      // 将开始标签添加到堆栈中
      stack.push(next[2]);
    }
  }
}

// 查找匹配的开始标签
function findMatchingOpen(iter, tag) {
  var stack = [];
    // 无限循环，直到条件不满足
    for (;;) {
      // 获取前一个标签
      var prev = toPrevTag(iter);
      // 如果前一个标签不存在，则返回
      if (!prev) return;
      // 如果前一个标签是自闭合标签，则跳过当前循环
      if (prev == "selfClose") { toTagStart(iter); continue; }
      // 记录当前行号和字符位置
      var endLine = iter.line, endCh = iter.ch;
      // 获取当前标签的起始位置
      var start = toTagStart(iter);
      // 如果起始位置不存在，则返回
      if (!start) return;
      // 如果起始位置是闭合标签
      if (start[1]) { // closing tag
        // 将标签名加入栈中
        stack.push(start[2]);
      } else { // opening tag
        // 遍历栈，找到对应的闭合标签并移除
        for (var i = stack.length - 1; i >= 0; --i) if (stack[i] == start[2]) {
          stack.length = i;
          break;
        }
        // 如果找不到对应的闭合标签，则返回标签信息
        if (i < 0 && (!tag || tag == start[2])) return {
          tag: start[2],
          from: Pos(iter.line, iter.ch),
          to: Pos(endLine, endCh)
        };
      }
    }
  }

  // 注册 XML 代码折叠功能
  CodeMirror.registerHelper("fold", "xml", function(cm, start) {
    // 创建迭代器
    var iter = new Iter(cm, start.line, 0);
    // 无限循环，直到条件不满足
    for (;;) {
      // 获取下一个标签
      var openTag = toNextTag(iter)
      // 如果下一个标签不存在或者行号不等于起始行号，则返回
      if (!openTag || iter.line != start.line) return
      // 获取标签的结束位置
      var end = toTagEnd(iter)
      // 如果结束位置不存在，则返回
      if (!end) return
      // 如果标签不是自闭合标签
      if (!openTag[1] && end != "selfClose") {
        // 获取起始位置和匹配的闭合标签位置
        var startPos = Pos(iter.line, iter.ch);
        var endPos = findMatchingClose(iter, openTag[2]);
        // 如果闭合标签位置存在且在起始位置之后，则返回折叠范围
        return endPos && cmp(endPos.from, startPos) > 0 ? {from: startPos, to: endPos.from} : null
      }
    }
  });

  // 查找匹配的标签
  CodeMirror.findMatchingTag = function(cm, pos, range) {
    // 创建迭代器
    var iter = new Iter(cm, pos.line, pos.ch, range);
    // 如果当前行不包含 "<" 或 ">"，则返回
    if (iter.text.indexOf(">") == -1 && iter.text.indexOf("<") == -1) return;
    // 获取标签的结束位置和起始位置
    var end = toTagEnd(iter), to = end && Pos(iter.line, iter.ch);
    var start = end && toTagStart(iter);
    // 如果结束位置或起始位置不存在，或者当前位置在指定位置之后，则返回
    if (!end || !start || cmp(iter, pos) > 0) return;
    // 记录当前标签信息
    var here = {from: Pos(iter.line, iter.ch), to: to, tag: start[2]};
    // 如果是自闭合标签，则返回标签信息
    if (end == "selfClose") return {open: here, close: null, at: "open"};

    // 如果是闭合标签
    if (start[1]) { // closing tag
      // 返回匹配的开放标签位置和当前闭合标签位置
      return {open: findMatchingOpen(iter, start[2]), close: here, at: "close"};
    } else { // opening tag
      // 创建新的迭代器
      iter = new Iter(cm, to.line, to.ch, range);
      // 返回当前开放标签位置和匹配的闭合标签位置
      return {open: here, close: findMatchingClose(iter, start[2]), at: "open"};
  // 定义一个匿名函数，用于查找匹配的开放标签和关闭标签
  }
};

// 在CodeMirror中查找包围标签
CodeMirror.findEnclosingTag = function(cm, pos, range, tag) {
  // 创建一个迭代器对象
  var iter = new Iter(cm, pos.line, pos.ch, range);
  // 无限循环，直到找到匹配的开放标签
  for (;;) {
    // 查找匹配的开放标签
    var open = findMatchingOpen(iter, tag);
    // 如果没有找到开放标签，则跳出循环
    if (!open) break;
    // 创建一个新的迭代器对象
    var forward = new Iter(cm, pos.line, pos.ch, range);
    // 查找匹配的关闭标签
    var close = findMatchingClose(forward, open.tag);
    // 如果找到了关闭标签，则返回开放标签和关闭标签的对象
    if (close) return {open: open, close: close};
  }
};

// 由addon/edit/closetag.js使用
// 在CodeMirror中扫描查找关闭标签
CodeMirror.scanForClosingTag = function(cm, pos, name, end) {
  // 创建一个迭代器对象
  var iter = new Iter(cm, pos.line, pos.ch, end ? {from: 0, to: end} : null);
  // 查找匹配的关闭标签
  return findMatchingClose(iter, name);
};
// 定义一个匿名函数，传入一个 mod 参数
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror")); // 调用 mod 函数并传入参数
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror"], mod); // 调用 define 函数并传入参数
  // 如果是普通的浏览器环境
  else
    mod(CodeMirror); // 调用 mod 函数并传入参数
})(function(CodeMirror) {
  "use strict"; // 开启严格模式

  var WORD = /[\w$]+/, RANGE = 500; // 定义正则表达式和范围

  // 注册一个名为 "anyword" 的提示辅助函数
  CodeMirror.registerHelper("hint", "anyword", function(editor, options) {
    var word = options && options.word || WORD; // 获取单词正则表达式
    var range = options && options.range || RANGE; // 获取范围
    var cur = editor.getCursor(), curLine = editor.getLine(cur.line); // 获取当前光标位置和当前行内容
    var end = cur.ch, start = end; // 设置结束位置和开始位置为光标位置
    while (start && word.test(curLine.charAt(start - 1))) --start; // 循环找到当前单词的起始位置
    var curWord = start != end && curLine.slice(start, end); // 获取当前单词

    var list = options && options.list || [], seen = {}; // 初始化列表和已见过的单词对象
    var re = new RegExp(word.source, "g"); // 创建正则表达式对象
    for (var dir = -1; dir <= 1; dir += 2) { // 循环两次
      var line = cur.line, endLine = Math.min(Math.max(line + dir * range, editor.firstLine()), editor.lastLine()) + dir; // 计算起始行和结束行
      for (; line != endLine; line += dir) { // 循环遍历行
        var text = editor.getLine(line), m; // 获取当前行内容和匹配结果
        while (m = re.exec(text)) { // 循环匹配单词
          if (line == cur.line && m[0] === curWord) continue; // 如果是当前行且是当前单词则跳过
          if ((!curWord || m[0].lastIndexOf(curWord, 0) == 0) && !Object.prototype.hasOwnProperty.call(seen, m[0])) { // 如果是以当前单词开头且未见过
            seen[m[0]] = true; // 标记为已见过
            list.push(m[0]); // 添加到列表中
          }
        }
      }
    }
    return {list: list, from: CodeMirror.Pos(cur.line, start), to: CodeMirror.Pos(cur.line, end)}; // 返回提示列表和光标位置范围
  });
});
(function(mod) {
  // 如果是 CommonJS 环境，使用 require 导入依赖
  if (typeof exports == "object" && typeof module == "object") 
    mod(require("../../lib/codemirror"), require("./xml-hint"));
  // 如果是 AMD 环境，使用 define 定义模块
  else if (typeof define == "function" && define.amd) 
    define(["../../lib/codemirror", "./xml-hint"], mod);
  // 如果是普通的浏览器环境，直接使用全局变量 CodeMirror
  else 
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 支持的语言列表
  var langs = "ab aa af ak sq am ar an hy as av ae ay az bm ba eu be bn bh bi bs br bg my ca ch ce ny zh cv kw co cr hr cs da dv nl dz en eo et ee fo fj fi fr ff gl ka de el gn gu ht ha he hz hi ho hu ia id ie ga ig ik io is it iu ja jv kl kn kr ks kk km ki rw ky kv kg ko ku kj la lb lg li ln lo lt lu lv gv mk mg ms ml mt mi mr mh mn na nv nb nd ne ng nn no ii nr oc oj cu om or os pa pi fa pl ps pt qu rm rn ro ru sa sc sd se sm sg sr gd sn si sk sl so st es su sw ss sv ta te tg th ti bo tk tl tn to tr ts tt tw ty ug uk ur uz ve vi vo wa cy wo fy xh yi yo za zu".split(" ");
  // 支持的链接打开方式
  var targets = ["_blank", "_self", "_top", "_parent"];
  // 支持的字符集
  var charsets = ["ascii", "utf-8", "utf-16", "latin1", "latin1"];
  // 支持的请求方法
  var methods = ["get", "post", "put", "delete"];
  // 支持的编码方式
  var encs = ["application/x-www-form-urlencoded", "multipart/form-data", "text/plain"];
  // 支持的媒体类型
  var media = ["all", "screen", "print", "embossed", "braille", "handheld", "print", "projection", "screen", "tty", "tv", "speech",
               "3d-glasses", "resolution [>][<][=] [X]", "device-aspect-ratio: X/Y", "orientation:portrait",
               "orientation:landscape", "device-height: [X]", "device-width: [X]"];
  // 简单的标签，用于一大堆标签的重复使用
  var s = { attrs: {} };

  // HTML 标签的属性配置
  var data = {
    a: {
      attrs: {
        href: null, ping: null, type: null,
        media: media,
        target: targets,
        hreflang: langs
      }
    },
    abbr: s,
    acronym: s,
    address: s,
    applet: s,
    # 定义 HTML 元素 area 的属性
    area: {
      attrs: {
        alt: null, coords: null, href: null, target: null, ping: null,
        media: media, hreflang: langs, type: null,
        shape: ["default", "rect", "circle", "poly"]
      }
    },
    # 定义 HTML 元素 article 和 aside，它们没有特定的属性
    article: s,
    aside: s,
    # 定义 HTML 元素 audio 的属性
    audio: {
      attrs: {
        src: null, mediagroup: null,
        crossorigin: ["anonymous", "use-credentials"],
        preload: ["none", "metadata", "auto"],
        autoplay: ["", "autoplay"],
        loop: ["", "loop"],
        controls: ["", "controls"]
      }
    },
    # 定义 HTML 元素 b，它没有特定的属性
    b: s,
    # 定义 HTML 元素 base 的属性
    base: { attrs: { href: null, target: targets } },
    # 定义 HTML 元素 basefont，它没有特定的属性
    basefont: s,
    # 定义 HTML 元素 bdi 和 bdo，它们没有特定的属性
    bdi: s,
    bdo: s,
    # 定义 HTML 元素 big，它没有特定的属性
    big: s,
    # 定义 HTML 元素 blockquote 的属性
    blockquote: { attrs: { cite: null } },
    # 定义 HTML 元素 body，它没有特定的属性
    body: s,
    # 定义 HTML 元素 br，它没有特定的属性
    br: s,
    # 定义 HTML 元素 button 的属性
    button: {
      attrs: {
        form: null, formaction: null, name: null, value: null,
        autofocus: ["", "autofocus"],
        disabled: ["", "autofocus"],
        formenctype: encs,
        formmethod: methods,
        formnovalidate: ["", "novalidate"],
        formtarget: targets,
        type: ["submit", "reset", "button"]
      }
    },
    # 定义 HTML 元素 canvas 的属性
    canvas: { attrs: { width: null, height: null } },
    # 定义 HTML 元素 caption，它没有特定的属性
    caption: s,
    # 定义 HTML 元素 center，它没有特定的属性
    center: s,
    # 定义 HTML 元素 cite，它没有特定的属性
    cite: s,
    # 定义 HTML 元素 code，它没有特定的属性
    code: s,
    # 定义 HTML 元素 col 的属性
    col: { attrs: { span: null } },
    # 定义 HTML 元素 colgroup 的属性
    colgroup: { attrs: { span: null } },
    # 定义 HTML 元素 command 的属性
    command: {
      attrs: {
        type: ["command", "checkbox", "radio"],
        label: null, icon: null, radiogroup: null, command: null, title: null,
        disabled: ["", "disabled"],
        checked: ["", "checked"]
      }
    },
    # 定义 HTML 元素 data 的属性
    data: { attrs: { value: null } },
    # 定义 HTML 元素 datagrid 的属性
    datagrid: { attrs: { disabled: ["", "disabled"], multiple: ["", "multiple"] } },
    # 定义 HTML 元素 datalist 的属性
    datalist: { attrs: { data: null } },
    # 定义 HTML 元素 dd，它没有特定的属性
    dd: s,
    # 定义 HTML 元素 del 的属性
    del: { attrs: { cite: null, datetime: null } },
    # 定义 HTML 元素 details 的属性
    details: { attrs: { open: ["", "open"] } },
    # 定义 HTML 元素 dfn，它没有特定的属性
    dfn: s,
    # 定义 HTML 元素 dir，它没有特定的属性
    dir: s,
    # 定义 HTML 元素 div，它没有特定的属性
    div: s,
    # 定义 HTML 元素 dl，它没有特定的属性
    dl: s,
    # 定义 HTML 元素 dt，它没有特定的属性
    dt: s,
    # 定义 HTML 元素 em，它没有特定的属性
    em: s,
    # 定义 HTML 元素 embed 的属性
    embed: { attrs: { src: null, type: null, width: null, height: null } },
    # 定义 HTML 元素 eventsource 的属性
    eventsource: { attrs: { src: null } },
    # 定义字段集合，包含禁用属性和表单属性
    fieldset: { attrs: { disabled: ["", "disabled"], form: null, name: null } },
    # 定义图例元素
    figcaption: s,
    # 定义图像和图形元素
    figure: s,
    # 定义字体元素
    font: s,
    # 定义页脚元素
    footer: s,
    # 定义表单元素
    form: {
      attrs: {
        action: null, name: null,
        "accept-charset": charsets,
        autocomplete: ["on", "off"],
        enctype: encs,
        method: methods,
        novalidate: ["", "novalidate"],
        target: targets
      }
    },
    # 定义框架元素
    frame: s,
    # 定义框架集元素
    frameset: s,
    # 定义标题元素
    h1: s, h2: s, h3: s, h4: s, h5: s, h6: s,
    # 定义头部元素
    head: {
      attrs: {},
      children: ["title", "base", "link", "style", "meta", "script", "noscript", "command"]
    },
    # 定义页眉元素
    header: s,
    # 定义标题组元素
    hgroup: s,
    # 定义水平线元素
    hr: s,
    # 定义 HTML 元素
    html: {
      attrs: { manifest: null },
      children: ["head", "body"]
    },
    # 定义斜体文本元素
    i: s,
    # 定义内联框架元素
    iframe: {
      attrs: {
        src: null, srcdoc: null, name: null, width: null, height: null,
        sandbox: ["allow-top-navigation", "allow-same-origin", "allow-forms", "allow-scripts"],
        seamless: ["", "seamless"]
      }
    },
    # 定义图像元素
    img: {
      attrs: {
        alt: null, src: null, ismap: null, usemap: null, width: null, height: null,
        crossorigin: ["anonymous", "use-credentials"]
      }
    },
    # 定义输入对象，包含各种 HTML 元素及其属性
    input: {
      attrs: {
        alt: null, dirname: null, form: null, formaction: null,
        height: null, list: null, max: null, maxlength: null, min: null,
        name: null, pattern: null, placeholder: null, size: null, src: null,
        step: null, value: null, width: null,
        accept: ["audio/*", "video/*", "image/*"],  # 可接受的文件类型
        autocomplete: ["on", "off"],  # 自动完成属性
        autofocus: ["", "autofocus"],  # 自动聚焦属性
        checked: ["", "checked"],  # 选中属性
        disabled: ["", "disabled"],  # 禁用属性
        formenctype: encs,  # 表单编码类型
        formmethod: methods,  # 表单提交方法
        formnovalidate: ["", "novalidate"],  # 表单不验证属性
        formtarget: targets,  # 表单提交目标
        multiple: ["", "multiple"],  # 多选属性
        readonly: ["", "readonly"],  # 只读属性
        required: ["", "required"],  # 必填属性
        type: ["hidden", "text", "search", "tel", "url", "email", "password", "datetime", "date", "month",
               "week", "time", "datetime-local", "number", "range", "color", "checkbox", "radio",
               "file", "submit", "image", "reset", "button"]  # 输入类型
      }
    },
    ins: { attrs: { cite: null, datetime: null } },  # 插入标签
    kbd: s,  # 键盘输入标签
    keygen: {
      attrs: {
        challenge: null, form: null, name: null,
        autofocus: ["", "autofocus"],  # 自动聚焦属性
        disabled: ["", "disabled"],  # 禁用属性
        keytype: ["RSA"]  # 密钥类型
      }
    },
    label: { attrs: { "for": null, form: null } },  # 标签标记
    legend: s,  # 图例标签
    li: { attrs: { value: null } },  # 列表项标签
    link: {
      attrs: {
        href: null, type: null,
        hreflang: langs,  # 链接语言
        media: media,  # 媒体查询条件
        sizes: ["all", "16x16", "16x16 32x32", "16x16 32x32 64x64"]  # 图标尺寸
      }
    },
    map: { attrs: { name: null } },  # 图像映射标签
    mark: s,  # 标记标签
    menu: { attrs: { label: null, type: ["list", "context", "toolbar"] } },  # 菜单标签
    meta: {
      attrs: {
        content: null,  # 元数据内容
        charset: charsets,  # 字符集
        name: ["viewport", "application-name", "author", "description", "generator", "keywords"],  # 元数据名称
        "http-equiv": ["content-language", "content-type", "default-style", "refresh"]  # HTTP 头部设置
      }
    },
    # 定义 meter 元素，包含 value、min、low、high、max、optimum 属性
    meter: { attrs: { value: null, min: null, low: null, high: null, max: null, optimum: null } },
    # 定义 nav 元素，无属性
    nav: s,
    # 定义 noframes 元素，无属性
    noframes: s,
    # 定义 noscript 元素，无属性
    noscript: s,
    # 定义 object 元素，包含 data、type、name、usemap、form、width、height、typemustmatch 属性
    object: {
      attrs: {
        data: null, type: null, name: null, usemap: null, form: null, width: null, height: null,
        typemustmatch: ["", "typemustmatch"]
      }
    },
    # 定义 ol 元素，包含 reversed、start、type 属性
    ol: { attrs: { reversed: ["", "reversed"], start: null, type: ["1", "a", "A", "i", "I"] } },
    # 定义 optgroup 元素，包含 disabled、label 属性
    optgroup: { attrs: { disabled: ["", "disabled"], label: null } },
    # 定义 option 元素，包含 disabled、label、selected、value 属性
    option: { attrs: { disabled: ["", "disabled"], label: null, selected: ["", "selected"], value: null } },
    # 定义 output 元素，包含 for、form、name 属性
    output: { attrs: { "for": null, form: null, name: null } },
    # 定义 p 元素，无属性
    p: s,
    # 定义 param 元素，包含 name、value 属性
    param: { attrs: { name: null, value: null } },
    # 定义 pre 元素，无属性
    pre: s,
    # 定义 progress 元素，包含 value、max 属性
    progress: { attrs: { value: null, max: null } },
    # 定义 q 元素，包含 cite 属性
    q: { attrs: { cite: null } },
    # 定义 rp 元素，无属性
    rp: s,
    # 定义 rt 元素，无属性
    rt: s,
    # 定义 ruby 元素，无属性
    ruby: s,
    # 定义 s 元素，无属性
    s: s,
    # 定义 samp 元素，无属性
    samp: s,
    # 定义 script 元素，包含 type、src、async、defer、charset 属性
    script: {
      attrs: {
        type: ["text/javascript"],
        src: null,
        async: ["", "async"],
        defer: ["", "defer"],
        charset: charsets
      }
    },
    # 定义 section 元素，无属性
    section: s,
    # 定义 select 元素，包含 form、name、size、autofocus、disabled、multiple 属性
    select: {
      attrs: {
        form: null, name: null, size: null,
        autofocus: ["", "autofocus"],
        disabled: ["", "disabled"],
        multiple: ["", "multiple"]
      }
    },
    # 定义 small 元素，无属性
    small: s,
    # 定义 source 元素，包含 src、type、media 属性
    source: { attrs: { src: null, type: null, media: null } },
    # 定义 span 元素，无属性
    span: s,
    # 定义 strike 元素，无属性
    strike: s,
    # 定义 strong 元素，无属性
    strong: s,
    # 定义 style 元素，包含 type、media、scoped 属性
    style: {
      attrs: {
        type: ["text/css"],
        media: media,
        scoped: null
      }
    },
    # 定义 sub 元素，无属性
    sub: s,
    # 定义 summary 元素，无属性
    summary: s,
    # 定义 sup 元素，无属性
    sup: s,
    # 定义 table 元素，无属性
    table: s,
    # 定义 tbody 元素，无属性
    tbody: s,
    # 定义 td 元素，包含 colspan、rowspan、headers 属性
    td: { attrs: { colspan: null, rowspan: null, headers: null } },
    # 定义 textarea 元素，包含 dirname、form、maxlength、name、placeholder、rows、cols、autofocus、disabled、readonly、required、wrap 属性
    textarea: {
      attrs: {
        dirname: null, form: null, maxlength: null, name: null, placeholder: null,
        rows: null, cols: null,
        autofocus: ["", "autofocus"],
        disabled: ["", "disabled"],
        readonly: ["", "readonly"],
        required: ["", "required"],
        wrap: ["soft", "hard"]
      }
    },
  // 定义 HTML 元素的默认属性和值
  var htmlElements = {
    // 定义 tfoot 元素的默认属性
    tfoot: s,
    // 定义 th 元素的默认属性
    th: { attrs: { colspan: null, rowspan: null, headers: null, scope: ["row", "col", "rowgroup", "colgroup"] } },
    // 定义 thead 元素的默认属性
    thead: s,
    // 定义 time 元素的默认属性
    time: { attrs: { datetime: null } },
    // 定义 title 元素的默认属性
    title: s,
    // 定义 tr 元素的默认属性
    tr: s,
    // 定义 track 元素的默认属性
    track: {
      attrs: {
        src: null, label: null, "default": null,
        kind: ["subtitles", "captions", "descriptions", "chapters", "metadata"],
        srclang: langs
      }
    },
    // 定义 tt 元素的默认属性
    tt: s,
    // 定义 u 元素的默认属性
    u: s,
    // 定义 ul 元素的默认属性
    ul: s,
    // 定义 var 元素的默认属性
    "var": s,
    // 定义 video 元素的默认属性
    video: {
      attrs: {
        src: null, poster: null, width: null, height: null,
        crossorigin: ["anonymous", "use-credentials"],
        preload: ["auto", "metadata", "none"],
        autoplay: ["", "autoplay"],
        mediagroup: ["movie"],
        muted: ["", "muted"],
        controls: ["", "controls"]
      }
    },
    // 定义 wbr 元素的默认属性
    wbr: s
  };

  // 定义全局属性的默认值
  var globalAttrs = {
    // 定义 accesskey 属性的默认值
    accesskey: ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    // 定义 class 属性的默认值
    "class": null,
    // 定义 contenteditable 属性的默认值
    contenteditable: ["true", "false"],
    // 定义 contextmenu 属性的默认值
    contextmenu: null,
    // 定义 dir 属性的默认值
    dir: ["ltr", "rtl", "auto"],
    // 定义 draggable 属性的默认值
    draggable: ["true", "false", "auto"],
    // 定义 dropzone 属性的默认值
    dropzone: ["copy", "move", "link", "string:", "file:"],
    // 定义 hidden 属性的默认值
    hidden: ["hidden"],
    // 定义 id 属性的默认值
    id: null,
    // 定义 inert 属性的默认值
    inert: ["inert"],
    // 定义 itemid 属性的默认值
    itemid: null,
    // 定义 itemprop 属性的默认值
    itemprop: null,
    // 定义 itemref 属性的默认值
    itemref: null,
    // 定义 itemscope 属性的默认值
    itemscope: ["itemscope"],
    // 定义 itemtype 属性的默认值
    itemtype: null,
    // 定义 lang 属性的默认值
    lang: ["en", "es"],
    // 定义 spellcheck 属性的默认值
    spellcheck: ["true", "false"],
    // 定义 autocorrect 属性的默认值
    autocorrect: ["true", "false"],
    // 定义 autocapitalize 属性的默认值
    autocapitalize: ["true", "false"],
    // 定义 style 属性的默认值
    style: null,
    // 定义 tabindex 属性的默认值
    tabindex: ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
    // 定义 title 属性的默认值
    title: null,
    // 定义 translate 属性的默认值
    translate: ["yes", "no"],
    // 定义 onclick 属性的默认值
    onclick: null,
    // 定义 rel 属性的默认值
    rel: ["stylesheet", "alternate", "author", "bookmark", "help", "license", "next", "nofollow", "noreferrer", "prefetch", "prev", "search", "tag"]
  };
  // 定义 populate 函数
  function populate(obj) {
  // 遍历全局属性对象，将属性添加到目标对象的属性中
  for (var attr in globalAttrs) if (globalAttrs.hasOwnProperty(attr))
    obj.attrs[attr] = globalAttrs[attr];
}

// 使用给定的数据填充目标对象
populate(s);
// 遍历数据对象，如果属性是数据对象的自有属性且不等于s，则填充数据对象
for (var tag in data) if (data.hasOwnProperty(tag) && data[tag] != s)
  populate(data[tag]);

// 将数据对象赋值给CodeMirror.htmlSchema
CodeMirror.htmlSchema = data;
// 定义htmlHint函数，接受CodeMirror编辑器和选项参数
function htmlHint(cm, options) {
  // 创建本地变量，包含schemaInfo属性
  var local = {schemaInfo: data};
  // 如果有选项参数，则将选项参数添加到本地变量中
  if (options) for (var opt in options) local[opt] = options[opt];
  // 返回xml提示
  return CodeMirror.hint.xml(cm, local);
}
// 将htmlHint函数注册为CodeMirror的html提示助手
CodeMirror.registerHelper("hint", "html", htmlHint);
// 模块定义
(function(mod) {
  // CommonJS
  if (typeof exports == "object" && typeof module == "object") 
    mod(require("../../lib/codemirror"));
  // AMD
  else if (typeof define == "function" && define.amd) 
    define(["../../lib/codemirror"], mod);
  // Plain browser env
  else 
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义类名常量
  var HINT_ELEMENT_CLASS        = "CodeMirror-hint";
  var ACTIVE_HINT_ELEMENT_CLASS = "CodeMirror-hint-active";

  // 旧的接口，为了向后兼容而保留
  CodeMirror.showHint = function(cm, getHints, options) {
    // 如果没有获取到提示，则调用 cm.showHint(options)
    if (!getHints) return cm.showHint(options);
    // 如果选项中包含 async 属性，则设置 getHints.async 为 true
    if (options && options.async) getHints.async = true;
    // 创建新的选项对象
    var newOpts = {hint: getHints};
    // 将 options 中的属性复制到 newOpts 中
    if (options) for (var prop in options) newOpts[prop] = options[prop];
    // 调用 cm.showHint(newOpts)
    return cm.showHint(newOpts);
  };

  // 定义 showHint 方法的扩展
  CodeMirror.defineExtension("showHint", function(options) {
    // 解析选项
    options = parseOptions(this, this.getCursor("start"), options);
    // 获取光标所在位置的选择
    var selections = this.listSelections()
    // 如果选择的数量大于 1，则返回
    if (selections.length > 1) return;
    // 默认情况下，当有选择时不允许自动完成
    // 提示函数可以有 supportsSelection 属性来指示它是否可以处理选择
    if (this.somethingSelected()) {
      if (!options.hint.supportsSelection) return;
      // 不允许跨行选择
      for (var i = 0; i < selections.length; i++)
        if (selections[i].head.line != selections[i].anchor.line) return;
    }
    // 如果存在 completionActive，则关闭它
    if (this.state.completionActive) this.state.completionActive.close();
    // 创建新的 Completion 对象
    var completion = this.state.completionActive = new Completion(this, options);
    // 如果选项中没有 hint 属性，则返回
    if (!completion.options.hint) return;
    // 触发 startCompletion 事件
    CodeMirror.signal(this, "startCompletion", this);
    // 更新补全提示的状态为 true
    completion.update(true);
  });

  // 定义关闭补全提示的方法
  CodeMirror.defineExtension("closeHint", function() {
    // 如果补全提示处于激活状态，则关闭补全提示
    if (this.state.completionActive) this.state.completionActive.close()
  })

  // 定义补全提示的构造函数
  function Completion(cm, options) {
    this.cm = cm;
    this.options = options;
    this.widget = null;
    this.debounce = 0;
    this.tick = 0;
    this.startPos = this.cm.getCursor("start");
    this.startLen = this.cm.getLine(this.startPos.line).length - this.cm.getSelection().length;

    var self = this;
    // 监听光标活动事件
    cm.on("cursorActivity", this.activityFunc = function() { self.cursorActivity(); });
  }

  // 定义 requestAnimationFrame 方法
  var requestAnimationFrame = window.requestAnimationFrame || function(fn) {
    return setTimeout(fn, 1000/60);
  };
  // 定义 cancelAnimationFrame 方法
  var cancelAnimationFrame = window.cancelAnimationFrame || clearTimeout;

  // 补全提示的原型方法
  Completion.prototype = {
    // 关闭补全提示
    close: function() {
      if (!this.active()) return;
      this.cm.state.completionActive = null;
      this.tick = null;
      this.cm.off("cursorActivity", this.activityFunc);

      if (this.widget && this.data) CodeMirror.signal(this.data, "close");
      if (this.widget) this.widget.close();
      CodeMirror.signal(this.cm, "endCompletion", this.cm);
    },

    // 判断补全提示是否处于激活状态
    active: function() {
      return this.cm.state.completionActive == this;
    },

    // 选择补全提示中的某个选项
    pick: function(data, i) {
      var completion = data.list[i], self = this;
      this.cm.operation(function() {
        if (completion.hint)
          completion.hint(self.cm, data, completion);
        else
          self.cm.replaceRange(getText(completion), completion.from || data.from,
                               completion.to || data.to, "complete");
        CodeMirror.signal(data, "pick", completion);
        self.cm.scrollIntoView();
      })
      this.close();
    },
    // 定义一个名为 cursorActivity 的方法
    cursorActivity: function() {
      // 如果存在 debounce 属性，则取消动画帧请求，并将 debounce 属性重置为 0
      if (this.debounce) {
        cancelAnimationFrame(this.debounce);
        this.debounce = 0;
      }

      // 初始化 identStart 变量为 this.startPos，如果存在 this.data，则使用 this.data.from 覆盖 identStart
      var identStart = this.startPos;
      if(this.data) {
        identStart = this.data.from;
      }

      // 获取光标位置和当前行内容
      var pos = this.cm.getCursor(), line = this.cm.getLine(pos.line);
      // 检查光标位置是否满足关闭提示框的条件，如果满足则关闭提示框，否则更新提示框
      if (pos.line != this.startPos.line || line.length - pos.ch != this.startLen - this.startPos.ch ||
          pos.ch < identStart.ch || this.cm.somethingSelected() ||
          (!pos.ch || this.options.closeCharacters.test(line.charAt(pos.ch - 1)))) {
        this.close();
      } else {
        // 使用 requestAnimationFrame 方法延迟更新提示框内容
        var self = this;
        this.debounce = requestAnimationFrame(function() {self.update();});
        // 如果存在提示框，则禁用提示框
        if (this.widget) this.widget.disable();
      }
    },

    // 定义一个名为 update 的方法，用于更新提示框内容
    update: function(first) {
      // 如果 tick 为 null，则直接返回
      if (this.tick == null) return
      var self = this, myTick = ++this.tick
      // 调用 fetchHints 方法获取提示框内容
      fetchHints(this.options.hint, this.cm, this.options, function(data) {
        // 如果 tick 等于 myTick，则调用 finishUpdate 方法更新提示框内容
        if (self.tick == myTick) self.finishUpdate(data, first)
      })
    },

    // 定义一个名为 finishUpdate 的方法，用于完成提示框内容的更新
    finishUpdate: function(data, first) {
      // 如果存在 this.data，则触发 "update" 事件
      if (this.data) CodeMirror.signal(this.data, "update");

      // 检查是否已选择提示框中的内容，如果已选择或者设置了 completeSingle 选项，则关闭提示框
      var picked = (this.widget && this.widget.picked) || (first && this.options.completeSingle);
      if (this.widget) this.widget.close();

      // 更新 this.data 为新的提示框内容
      this.data = data;

      // 如果存在提示框内容且列表长度大于 0，则根据条件选择或者创建新的提示框
      if (data && data.list.length) {
        if (picked && data.list.length == 1) {
          this.pick(data, 0);
        } else {
          this.widget = new Widget(this, data);
          CodeMirror.signal(data, "shown");
        }
      }
    }
  };

  // 定义一个名为 parseOptions 的函数，用于解析提示框的选项
  function parseOptions(cm, pos, options) {
    // 获取编辑器的提示框选项
    var editor = cm.options.hintOptions;
    var out = {};
    // 将默认选项复制到 out 对象中
    for (var prop in defaultOptions) out[prop] = defaultOptions[prop];
    // 如果存在编辑器选项，则将其覆盖到 out 对象中
    if (editor) for (var prop in editor)
      if (editor[prop] !== undefined) out[prop] = editor[prop];
    // 如果存在传入的选项，则将其覆盖到 out 对象中
    if (options) for (var prop in options)
      if (options[prop] !== undefined) out[prop] = options[prop];
    // 如果提示对象有解析方法，则调用解析方法，传入上下文和位置参数
    if (out.hint.resolve) out.hint = out.hint.resolve(cm, pos)
    // 返回提示对象
    return out;
  }

  // 获取提示的文本内容
  function getText(completion) {
    // 如果提示是字符串，则直接返回
    if (typeof completion == "string") return completion;
    // 否则返回提示对象的文本属性
    else return completion.text;
  }

  // 构建键盘映射
  function buildKeyMap(completion, handle) {
    // 基础键盘映射
    var baseMap = {
      Up: function() {handle.moveFocus(-1);},
      Down: function() {handle.moveFocus(1);},
      PageUp: function() {handle.moveFocus(-handle.menuSize() + 1, true);},
      PageDown: function() {handle.moveFocus(handle.menuSize() - 1, true);},
      Home: function() {handle.setFocus(0);},
      End: function() {handle.setFocus(handle.length - 1);},
      Enter: handle.pick,
      Tab: handle.pick,
      Esc: handle.close
    };

    // 检测是否为 Mac 系统
    var mac = /Mac/.test(navigator.platform);

    // 如果是 Mac 系统，则添加额外的键盘映射
    if (mac) {
      baseMap["Ctrl-P"] = function() {handle.moveFocus(-1);};
      baseMap["Ctrl-N"] = function() {handle.moveFocus(1);};
    }

    // 获取自定义键盘映射
    var custom = completion.options.customKeys;
    var ourMap = custom ? {} : baseMap;
    // 添加绑定
    function addBinding(key, val) {
      var bound;
      if (typeof val != "string")
        bound = function(cm) { return val(cm, handle); };
      // 此机制已被弃用
      else if (baseMap.hasOwnProperty(val))
        bound = baseMap[val];
      else
        bound = val;
      ourMap[key] = bound;
    }
    // 如果存在自定义键盘映射，则遍历添加
    if (custom)
      for (var key in custom) if (custom.hasOwnProperty(key))
        addBinding(key, custom[key]);
    // 获取额外的键盘映射
    var extra = completion.options.extraKeys;
    if (extra)
      for (var key in extra) if (extra.hasOwnProperty(key))
        addBinding(key, extra[key]);
    // 返回最终的键盘映射
    return ourMap;
  }

  // 获取提示元素
  function getHintElement(hintsElement, el) {
    while (el && el != hintsElement) {
      if (el.nodeName.toUpperCase() === "LI" && el.parentNode == hintsElement) return el;
      el = el.parentNode;
    }
  }

  // 提示小部件构造函数
  function Widget(completion, data) {
    this.completion = completion;
    this.data = data;
    this.picked = false;
    var widget = this, cm = completion.cm;
    # 获取输入字段的所属文档对象
    var ownerDocument = cm.getInputField().ownerDocument;
    # 获取所属文档对象的默认视图或父窗口
    var parentWindow = ownerDocument.defaultView || ownerDocument.parentWindow;

    # 创建提示列表元素
    var hints = this.hints = ownerDocument.createElement("ul");
    # 获取代码编辑器的主题
    var theme = completion.cm.options.theme;
    # 设置提示列表元素的类名
    hints.className = "CodeMirror-hints " + theme;
    # 设置选中的提示索引
    this.selectedHint = data.selectedHint || 0;

    # 获取提示列表
    var completions = data.list;
    # 遍历提示列表
    for (var i = 0; i < completions.length; ++i) {
      # 创建提示列表项元素
      var elt = hints.appendChild(ownerDocument.createElement("li")), cur = completions[i];
      # 设置提示列表项元素的类名
      var className = HINT_ELEMENT_CLASS + (i != this.selectedHint ? "" : " " + ACTIVE_HINT_ELEMENT_CLASS);
      if (cur.className != null) className = cur.className + " " + className;
      elt.className = className;
      # 如果有自定义渲染函数，则调用渲染函数
      if (cur.render) cur.render(elt, data, cur);
      else elt.appendChild(ownerDocument.createTextNode(cur.displayText || getText(cur)));
      # 设置提示列表项元素的索引
      elt.hintId = i;
    }

    # 获取提示列表容器，如果没有指定则使用文档对象的 body
    var container = completion.options.container || ownerDocument.body;
    # 获取光标位置的坐标
    var pos = cm.cursorCoords(completion.options.alignWithWord ? data.from : null);
    var left = pos.left, top = pos.bottom, below = true;
    var offsetLeft = 0, offsetTop = 0;
    # 如果容器不是文档对象的 body
    if (container !== ownerDocument.body) {
      # 计算偏移量
      var isContainerPositioned = ['absolute', 'relative', 'fixed'].indexOf(parentWindow.getComputedStyle(container).position) !== -1;
      var offsetParent = isContainerPositioned ? container : container.offsetParent;
      var offsetParentPosition = offsetParent.getBoundingClientRect();
      var bodyPosition = ownerDocument.body.getBoundingClientRect();
      offsetLeft = (offsetParentPosition.left - bodyPosition.left - offsetParent.scrollLeft);
      offsetTop = (offsetParentPosition.top - bodyPosition.top - offsetParent.scrollTop);
    }
    # 设置提示列表的位置
    hints.style.left = (left - offsetLeft) + "px";
    hints.style.top = (top - offsetTop) + "px";
    // 获取窗口的宽度和高度，考虑了不同浏览器的兼容性
    var winW = parentWindow.innerWidth || Math.max(ownerDocument.body.offsetWidth, ownerDocument.documentElement.offsetWidth);
    var winH = parentWindow.innerHeight || Math.max(ownerDocument.body.offsetHeight, ownerDocument.documentElement.offsetHeight);
    // 将提示框添加到容器中
    container.appendChild(hints);
    // 获取提示框的位置信息和与窗口底部的重叠情况
    var box = hints.getBoundingClientRect(), overlapY = box.bottom - winH;
    // 判断提示框是否需要滚动
    var scrolls = hints.scrollHeight > hints.clientHeight + 1
    // 获取滚动信息
    var startScroll = cm.getScrollInfo();

    // 如果提示框与窗口底部有重叠
    if (overlapY > 0) {
      // 计算提示框的高度和位置
      var height = box.bottom - box.top, curTop = pos.top - (pos.bottom - box.top);
      // 如果提示框可以放在光标上方
      if (curTop - height > 0) {
        hints.style.top = (top = pos.top - height - offsetTop) + "px";
        below = false;
      } 
      // 如果提示框的高度超过了窗口高度
      else if (height > winH) {
        hints.style.height = (winH - 5) + "px";
        hints.style.top = (top = pos.bottom - box.top - offsetTop) + "px";
        // 获取光标位置，调整提示框位置
        var cursor = cm.getCursor();
        if (data.from.ch != cursor.ch) {
          pos = cm.cursorCoords(cursor);
          hints.style.left = (left = pos.left - offsetLeft) + "px";
          box = hints.getBoundingClientRect();
        }
      }
    }
    // 获取提示框与窗口右侧的重叠情况
    var overlapX = box.right - winW;
    // 如果提示框与窗口右侧有重叠
    if (overlapX > 0) {
      // 如果提示框宽度超过了窗口宽度
      if (box.right - box.left > winW) {
        hints.style.width = (winW - 5) + "px";
        overlapX -= (box.right - box.left) - winW;
      }
      hints.style.left = (left = pos.left - overlapX - offsetLeft) + "px";
    }
    // 如果提示框需要滚动，调整内部元素的右边距
    if (scrolls) for (var node = hints.firstChild; node; node = node.nextSibling)
      node.style.paddingRight = cm.display.nativeBarWidth + "px"
    # 将键盘映射添加到代码编辑器中，并构建键盘映射
    cm.addKeyMap(this.keyMap = buildKeyMap(completion, {
      # 移动焦点到指定位置，避免循环
      moveFocus: function(n, avoidWrap) { widget.changeActive(widget.selectedHint + n, avoidWrap); },
      # 设置焦点到指定位置
      setFocus: function(n) { widget.changeActive(n); },
      # 获取菜单大小
      menuSize: function() { return widget.screenAmount(); },
      # 获取完成选项的长度
      length: completions.length,
      # 关闭完成选项
      close: function() { completion.close(); },
      # 选择当前的完成选项
      pick: function() { widget.pick(); },
      # 数据
      data: data
    }));

    # 如果完成选项的选项为关闭状态
    if (completion.options.closeOnUnfocus) {
      # 定义在失去焦点时关闭的变量
      var closingOnBlur;
      # 当编辑器失去焦点时触发的事件
      cm.on("blur", this.onBlur = function() { closingOnBlur = setTimeout(function() { completion.close(); }, 100); });
      # 当编辑器获得焦点时触发的事件
      cm.on("focus", this.onFocus = function() { clearTimeout(closingOnBlur); });
    }

    # 当编辑器滚动时触发的事件
    cm.on("scroll", this.onScroll = function() {
      # 获取当前滚动信息和编辑器的位置信息
      var curScroll = cm.getScrollInfo(), editor = cm.getWrapperElement().getBoundingClientRect();
      var newTop = top + startScroll.top - curScroll.top;
      var point = newTop - (parentWindow.pageYOffset || (ownerDocument.documentElement || ownerDocument.body).scrollTop);
      # 如果提示框在编辑器上方或下方，则关闭完成选项
      if (!below) point += hints.offsetHeight;
      if (point <= editor.top || point >= editor.bottom) return completion.close();
      hints.style.top = newTop + "px";
      hints.style.left = (left + startScroll.left - curScroll.left) + "px";
    });

    # 双击提示框时触发的事件
    CodeMirror.on(hints, "dblclick", function(e) {
      # 获取双击的提示元素，并改变活动状态并选择
      var t = getHintElement(hints, e.target || e.srcElement);
      if (t && t.hintId != null) {widget.changeActive(t.hintId); widget.pick();}
    });

    # 单击提示框时触发的事件
    CodeMirror.on(hints, "click", function(e) {
      # 获取单击的提示元素，并改变活动状态并选择
      var t = getHintElement(hints, e.target || e.srcElement);
      if (t && t.hintId != null) {
        widget.changeActive(t.hintId);
        # 如果选项为单击完成，则选择
        if (completion.options.completeOnSingleClick) widget.pick();
      }
    });

    # 鼠标按下提示框时触发的事件
    CodeMirror.on(hints, "mousedown", function() {
      # 延迟20毫秒后聚焦到编辑器
      setTimeout(function(){cm.focus();}, 20);
    });
    # 滚动到活动状态
    this.scrollToActive()
    # 发出“select”信号，传递选定的提示和对应的节点
    CodeMirror.signal(data, "select", completions[this.selectedHint], hints.childNodes[this.selectedHint]);
    # 返回 true
    return true;
  }

  Widget.prototype = {
    # 关闭小部件
    close: function() {
      # 如果小部件不是当前小部件，则返回
      if (this.completion.widget != this) return;
      # 将小部件从父节点中移除
      this.hints.parentNode.removeChild(this.hints);
      # 移除键盘映射
      this.completion.cm.removeKeyMap(this.keyMap);

      var cm = this.completion.cm;
      # 如果选项为在失焦时关闭，则移除失焦和获焦事件监听
      if (this.completion.options.closeOnUnfocus) {
        cm.off("blur", this.onBlur);
        cm.off("focus", this.onFocus);
      }
      # 移除滚动事件监听
      cm.off("scroll", this.onScroll);
    },

    # 禁用小部件
    disable: function() {
      # 移除键盘映射
      this.completion.cm.removeKeyMap(this.keyMap);
      var widget = this;
      # 设置键盘映射为按下回车键时标记为已选
      this.keyMap = {Enter: function() { widget.picked = true; }};
      this.completion.cm.addKeyMap(this.keyMap);
    },

    # 选择提示
    pick: function() {
      this.completion.pick(this.data, this.selectedHint);
    },

    # 改变活动状态
    changeActive: function(i, avoidWrap) {
      # 如果 i 大于列表长度，则根据 avoidWrap 决定是否循环
      if (i >= this.data.list.length)
        i = avoidWrap ? this.data.list.length - 1 : 0;
      # 如果 i 小于 0，则根据 avoidWrap 决定是否循环
      else if (i < 0)
        i = avoidWrap ? 0  : this.data.list.length - 1;
      # 如果选定的提示和 i 相同，则返回
      if (this.selectedHint == i) return;
      # 移除之前选定提示的活动样式类
      var node = this.hints.childNodes[this.selectedHint];
      if (node) node.className = node.className.replace(" " + ACTIVE_HINT_ELEMENT_CLASS, "");
      # 添加新选定提示的活动样式类
      node = this.hints.childNodes[this.selectedHint = i];
      node.className += " " + ACTIVE_HINT_ELEMENT_CLASS;
      # 滚动到活动提示
      this.scrollToActive()
      # 发出“select”信号，传递选定的提示和对应的节点
      CodeMirror.signal(this.data, "select", this.data.list[this.selectedHint], node);
    },
    # 滚动到活动提示的位置
    scrollToActive: function() {
      # 获取滚动边距，如果没有设置则默认为0
      var margin = this.completion.options.scrollMargin || 0;
      # 获取当前选中提示前后指定边距的节点
      var node1 = this.hints.childNodes[Math.max(0, this.selectedHint - margin)];
      var node2 = this.hints.childNodes[Math.min(this.data.list.length - 1, this.selectedHint + margin)];
      # 获取提示列表的第一个节点
      var firstNode = this.hints.firstChild;
      # 如果node1的顶部位置小于提示列表的滚动位置，则将滚动位置设置为node1的顶部位置减去第一个节点的顶部位置
      if (node1.offsetTop < this.hints.scrollTop)
        this.hints.scrollTop = node1.offsetTop - firstNode.offsetTop;
      # 如果node2的底部位置大于提示列表的滚动位置加上提示列表的高度，则将滚动位置设置为node2的底部位置加上node2的高度减去提示列表的高度再加上第一个节点的顶部位置
      else if (node2.offsetTop + node2.offsetHeight > this.hints.scrollTop + this.hints.clientHeight)
        this.hints.scrollTop = node2.offsetTop + node2.offsetHeight - this.hints.clientHeight + firstNode.offsetTop;
    },

    # 计算屏幕上能显示的提示数量
    screenAmount: function() {
      return Math.floor(this.hints.clientHeight / this.hints.firstChild.offsetHeight) || 1;
    }
  };

  # 获取适用的辅助函数
  function applicableHelpers(cm, helpers) {
    # 如果没有选中内容，则返回所有辅助函数
    if (!cm.somethingSelected()) return helpers
    var result = []
    # 遍历辅助函数，将支持选择的函数加入结果数组
    for (var i = 0; i < helpers.length; i++)
      if (helpers[i].supportsSelection) result.push(helpers[i])
    return result
  }

  # 获取提示列表
  function fetchHints(hint, cm, options, callback) {
    # 如果提示是异步的，则调用异步提示函数
    if (hint.async) {
      hint(cm, callback, options)
    } else {
      # 否则调用同步提示函数，并根据返回结果调用回调函数
      var result = hint(cm, options)
      if (result && result.then) result.then(callback)
      else callback(result)
    }
  }

  # 解析自动提示
  function resolveAutoHints(cm, pos) {
    # 获取当前位置的辅助函数和单词
    var helpers = cm.getHelpers(pos, "hint"), words
    if (helpers.length) {
      # 创建一个解析函数，根据当前位置的辅助函数获取适用的辅助函数，并依次调用获取提示列表的函数
      var resolved = function(cm, callback, options) {
        var app = applicableHelpers(cm, helpers);
        function run(i) {
          if (i == app.length) return callback(null)
          fetchHints(app[i], cm, options, function(result) {
            if (result && result.list.length > 0) callback(result)
            else run(i + 1)
          })
        }
        run(0)
      }
      resolved.async = true
      resolved.supportsSelection = true
      return resolved
  // 如果存在提示词，则返回一个函数，该函数返回基于提示词的提示列表
  } else if (words = cm.getHelper(cm.getCursor(), "hintWords")) {
    return function(cm) { return CodeMirror.hint.fromList(cm, {words: words}) }
  // 如果存在任意单词的提示，则返回一个函数，该函数返回基于任意单词的提示列表
  } else if (CodeMirror.hint.anyword) {
    return function(cm, options) { return CodeMirror.hint.anyword(cm, options) }
  // 否则返回一个空函数
  } else {
    return function() {}
  }
}

// 注册自动提示的解析函数
CodeMirror.registerHelper("hint", "auto", {
  resolve: resolveAutoHints
});

// 注册基于列表的提示函数
CodeMirror.registerHelper("hint", "fromList", function(cm, options) {
  var cur = cm.getCursor(), token = cm.getTokenAt(cur)
  var term, from = CodeMirror.Pos(cur.line, token.start), to = cur
  if (token.start < cur.ch && /\w/.test(token.string.charAt(cur.ch - token.start - 1))) {
    term = token.string.substr(0, cur.ch - token.start)
  } else {
    term = ""
    from = cur
  }
  var found = [];
  for (var i = 0; i < options.words.length; i++) {
    var word = options.words[i];
    if (word.slice(0, term.length) == term)
      found.push(word);
  }

  // 如果找到匹配的单词，则返回匹配列表和位置信息
  if (found.length) return {list: found, from: from, to: to};
});

// 设置命令 autocomplete 为 showHint
CodeMirror.commands.autocomplete = CodeMirror.showHint;

// 默认选项
var defaultOptions = {
  hint: CodeMirror.hint.auto,
  completeSingle: true,
  alignWithWord: true,
  closeCharacters: /[\s()\[\]{};:>,]/,
  closeOnUnfocus: true,
  completeOnSingleClick: true,
  container: null,
  customKeys: null,
  extraKeys: null
};

// 定义 hintOptions 选项
CodeMirror.defineOption("hintOptions", null);
// 导出模块
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror"), require("../../mode/sql/sql"));
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror", "../../mode/sql/sql"], mod);
  // 如果是普通浏览器环境
  else
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义变量
  var tables;
  var defaultTable;
  var keywords;
  var identifierQuote;
  var CONS = {
    QUERY_DIV: ";",
    ALIAS_KEYWORD: "AS"
  };
  var Pos = CodeMirror.Pos, cmpPos = CodeMirror.cmpPos;

  // 判断是否为数组
  function isArray(val) { return Object.prototype.toString.call(val) == "[object Array]" }

  // 获取关键字
  function getKeywords(editor) {
    var mode = editor.doc.modeOption;
    if (mode === "sql") mode = "text/x-sql";
    return CodeMirror.resolveMode(mode).keywords;
  }

  // 获取标识符引用符号
  function getIdentifierQuote(editor) {
    var mode = editor.doc.modeOption;
    if (mode === "sql") mode = "text/x-sql";
    return CodeMirror.resolveMode(mode).identifierQuote || "`";
  }

  // 获取文本
  function getText(item) {
    return typeof item == "string" ? item : item.text;
  }

  // 包装表
  function wrapTable(name, value) {
    if (isArray(value)) value = {columns: value}
    if (!value.text) value.text = name
    return value
  }

  // 解析表
  function parseTables(input) {
    var result = {}
    if (isArray(input)) {
      for (var i = input.length - 1; i >= 0; i--) {
        var item = input[i]
        result[getText(item).toUpperCase()] = wrapTable(getText(item), item)
      }
    } else if (input) {
      for (var name in input)
        result[name.toUpperCase()] = wrapTable(name, input[name])
    }
    return result
  }

  // 获取表
  function getTable(name) {
    return tables[name.toUpperCase()]
  }

  // 浅拷贝对象
  function shallowClone(object) {
    var result = {};
  // 遍历对象的属性，将属性和对应的值复制到新的对象中
  for (var key in object) if (object.hasOwnProperty(key))
    result[key] = object[key];
  // 返回复制后的对象
  return result;
}

// 匹配字符串和单词，忽略大小写
function match(string, word) {
  // 获取字符串的长度
  var len = string.length;
  // 获取单词的前 len 个字符
  var sub = getText(word).substr(0, len);
  // 比较字符串和单词的前 len 个字符，忽略大小写
  return string.toUpperCase() === sub.toUpperCase();
}

// 将匹配的单词添加到结果数组中
function addMatches(result, search, wordlist, formatter) {
  // 如果 wordlist 是数组
  if (isArray(wordlist)) {
    // 遍历数组中的单词
    for (var i = 0; i < wordlist.length; i++)
      // 如果单词匹配，则将格式化后的单词添加到结果数组中
      if (match(search, wordlist[i])) result.push(formatter(wordlist[i]))
  } else {
    // 如果 wordlist 是对象
    for (var word in wordlist) if (wordlist.hasOwnProperty(word)) {
      var val = wordlist[word]
      // 如果值为空或者为 true，则将单词添加到结果数组中
      if (!val || val === true)
        val = word
      else
        // 如果值有 displayText 属性，则将格式化后的值添加到结果数组中
        val = val.displayText ? {text: val.text, displayText: val.displayText} : val.text
      // 如果值匹配，则将格式化后的值添加到结果数组中
      if (match(search, val)) result.push(formatter(val))
    }
  }
}

// 清理名称，去除标识符引号和前导点（.）
function cleanName(name) {
  // 如果名称以点（.）开头，则去除点（.）
  if (name.charAt(0) == ".") {
    name = name.substr(1);
  }
  // 替换重复的标识符引号为单个标识符引号，并移除单个标识符引号
  var nameParts = name.split(identifierQuote+identifierQuote);
  for (var i = 0; i < nameParts.length; i++)
    nameParts[i] = nameParts[i].replace(new RegExp(identifierQuote,"g"), "");
  return nameParts.join(identifierQuote);
}

// 插入标识符引号
function insertIdentifierQuotes(name) {
  var nameParts = getText(name).split(".");
  for (var i = 0; i < nameParts.length; i++)
    nameParts[i] = identifierQuote +
      // 重复标识符引号
      nameParts[i].replace(new RegExp(identifierQuote,"g"), identifierQuote+identifierQuote) +
      identifierQuote;
  var escaped = nameParts.join(".");
  // 如果名称是字符串，则返回转义后的名称
  if (typeof name == "string") return escaped;
  // 否则，返回转义后的名称对象
  name = shallowClone(name);
  name.text = escaped;
  return name;
}

// 名称完成
function nameCompletion(cur, token, result, editor) {
    // 尝试完成表格、列名，并返回完成的起始位置
    var useIdentifierQuotes = false;  // 是否使用标识符引号
    var nameParts = [];  // 存储表格或列名的数组
    var start = token.start;  // 记录起始位置
    var cont = true;  // 控制循环的条件
    while (cont) {
      cont = (token.string.charAt(0) == ".");  // 判断是否继续循环
      useIdentifierQuotes = useIdentifierQuotes || (token.string.charAt(0) == identifierQuote);  // 判断是否使用标识符引号

      start = token.start;  // 更新起始位置
      nameParts.unshift(cleanName(token.string));  // 将清理后的表格或列名添加到数组中

      token = editor.getTokenAt(Pos(cur.line, token.start));  // 获取下一个标记
      if (token.string == ".") {
        cont = true;  // 如果下一个标记是"."，继续循环
        token = editor.getTokenAt(Pos(cur.line, token.start));  // 获取下一个标记
      }
    }

    // 尝试完成表格名
    var string = nameParts.join(".");  // 将数组中的表格或列名连接成字符串
    addMatches(result, string, tables, function(w) {  // 将结果添加到匹配列表中
      return useIdentifierQuotes ? insertIdentifierQuotes(w) : w;  // 如果使用标识符引号，则插入引号
    });

    // 尝试从默认表格中完成列名
    addMatches(result, string, defaultTable, function(w) {  // 将结果添加到匹配列表中
      return useIdentifierQuotes ? insertIdentifierQuotes(w) : w;  // 如果使用标识符引号，则插入引号
    });

    // 尝试完成列名
    string = nameParts.pop();  // 弹出数组中的最后一个元素
    var table = nameParts.join(".");  // 将数组中的元素连接成字符串

    var alias = false;  // 别名标识
    var aliasTable = table;  // 别名表格
    // 检查表格是否可用，如果不可用，则通过别名找到表格
    if (!getTable(table)) {
      var oldTable = table;  // 记录旧表格名
      table = findTableByAlias(table, editor);  // 通过别名找到表格
      if (table !== oldTable) alias = true;  // 如果找到的表格名与旧表格名不同，则设置别名标识为true
    }

    var columns = getTable(table);  // 获取表格的列
    if (columns && columns.columns)
      columns = columns.columns;  // 如果存在列，则获取列

    if (columns) {
      addMatches(result, string, columns, function(w) {  // 将结果添加到匹配列表中
        var tableInsert = table;  // 插入表格名
        if (alias == true) tableInsert = aliasTable;  // 如果存在别名，则使用别名表格名
        if (typeof w == "string") {
          w = tableInsert + "." + w;  // 如果是字符串，则添加表格名
        } else {
          w = shallowClone(w);  // 浅拷贝对象
          w.text = tableInsert + "." + w.text;  // 添加表格名
        }
        return useIdentifierQuotes ? insertIdentifierQuotes(w) : w;  // 如果使用标识符引号，则插入引号
      });
    }

    return start;  // 返回起始位置
  }

  function eachWord(lineText, f) {
    // 将文本按空格分割成单词数组
    var words = lineText.split(/\s+/)
    // 遍历单词数组
    for (var i = 0; i < words.length; i++)
      // 如果单词不为空，则调用函数 f 处理去除逗号和分号后的单词
      if (words[i]) f(words[i].replace(/[,;]/g, ''))
  }

  // 根据别名在编辑器中查找表
  function findTableByAlias(alias, editor) {
    // 获取编辑器文档
    var doc = editor.doc;
    // 获取完整查询语句
    var fullQuery = doc.getValue();
    // 将别名转换为大写
    var aliasUpperCase = alias.toUpperCase();
    // 初始化前一个单词和表名
    var previousWord = "";
    var table = "";
    // 初始化分隔符数组
    var separator = [];
    // 初始化有效范围
    var validRange = {
      start: Pos(0, 0),
      end: Pos(editor.lastLine(), editor.getLineHandle(editor.lastLine()).length)
    };

    // 添加分隔符位置
    var indexOfSeparator = fullQuery.indexOf(CONS.QUERY_DIV);
    while(indexOfSeparator != -1) {
      separator.push(doc.posFromIndex(indexOfSeparator));
      indexOfSeparator = fullQuery.indexOf(CONS.QUERY_DIV, indexOfSeparator+1);
    }
    separator.unshift(Pos(0, 0));
    separator.push(Pos(editor.lastLine(), editor.getLineHandle(editor.lastLine()).text.length));

    // 查找有效范围
    var prevItem = null;
    var current = editor.getCursor()
    for (var i = 0; i < separator.length; i++) {
      if ((prevItem == null || cmpPos(current, prevItem) > 0) && cmpPos(current, separator[i]) <= 0) {
        validRange = {start: prevItem, end: separator[i]};
        break;
      }
      prevItem = separator[i];
    }

    // 如果存在有效范围
    if (validRange.start) {
      // 获取有效范围内的查询语句
      var query = doc.getRange(validRange.start, validRange.end, false);

      // 遍历查询语句的每一行
      for (var i = 0; i < query.length; i++) {
        var lineText = query[i];
        // 对每一行的单词进行处理
        eachWord(lineText, function(word) {
          var wordUpperCase = word.toUpperCase();
          // 如果单词与别名相同且前一个单词是表名，则将表名赋值给 table
          if (wordUpperCase === aliasUpperCase && getTable(previousWord))
            table = previousWord;
          // 如果单词不是别名关键字，则更新前一个单词
          if (wordUpperCase !== CONS.ALIAS_KEYWORD)
            previousWord = word;
        });
        // 如果已找到表名，则跳出循环
        if (table) break;
      }
    }
    // 返回表名
    return table;
  }

  // 注册 SQL 提示功能
  CodeMirror.registerHelper("hint", "sql", function(editor, options) {
    // 解析表格
    tables = parseTables(options && options.tables)
    // 获取默认表名
    var defaultTableName = options && options.defaultTable;
    # 检查是否存在禁用关键字选项，如果存在则赋值给 disableKeywords，否则为 undefined
    var disableKeywords = options && options.disableKeywords;
    # 如果存在默认表名，则获取对应的表格，否则为 undefined
    defaultTable = defaultTableName && getTable(defaultTableName);
    # 获取编辑器中的关键字
    keywords = getKeywords(editor);
    # 获取编辑器中的标识符引用
    identifierQuote = getIdentifierQuote(editor);

    # 如果存在默认表名且默认表不存在，则通过别名查找表格
    if (defaultTableName && !defaultTable)
      defaultTable = findTableByAlias(defaultTableName, editor);

    # 如果默认表存在且包含列，则将默认表设置为其列
    defaultTable = defaultTable || [];

    # 获取当前光标位置
    var cur = editor.getCursor();
    # 初始化结果数组
    var result = [];
    # 获取当前光标处的标记
    var token = editor.getTokenAt(cur), start, end, search;
    # 如果标记的结束位置大于当前光标位置，则修正结束位置和字符串
    if (token.end > cur.ch) {
      token.end = cur.ch;
      token.string = token.string.slice(0, cur.ch - token.start);
    }

    # 如果标记的字符串匹配指定模式，则设置搜索字符串、开始位置和结束位置
    if (token.string.match(/^[.`"'\w@][\w$#]*$/g)) {
      search = token.string;
      start = token.start;
      end = token.end;
    } else {
      start = end = cur.ch;
      search = "";
    }
    # 如果搜索字符串以"."或标识符引用字符开头，则调用 nameCompletion 函数
    if (search.charAt(0) == "." || search.charAt(0) == identifierQuote) {
      start = nameCompletion(cur, token, result, editor);
    } else {
      # 定义一个函数，用于设置对象或类名
      var objectOrClass = function(w, className) {
        if (typeof w === "object") {
          w.className = className;
        } else {
          w = { text: w, className: className };
        }
        return w;
      };
      # 将默认表格中匹配搜索字符串的项添加到结果数组中
      addMatches(result, search, defaultTable, function(w) {
        return objectOrClass(w, "CodeMirror-hint-table CodeMirror-hint-default-table");
      });
      # 将所有表格中匹配搜索字符串的项添加到结果数组中
      addMatches(
        result,
        search,
        tables, function(w) {
          return objectOrClass(w, "CodeMirror-hint-table");
        }
      );
      # 如果禁用关键字选项不存在，则将匹配搜索字符串的关键字添加到结果数组中
      if (!disableKeywords)
        addMatches(result, search, keywords, function(w) {
          return objectOrClass(w.toUpperCase(), "CodeMirror-hint-keyword");
        });
    }
    # 返回结果数组和光标位置范围
    return {list: result, from: Pos(cur.line, start), to: Pos(cur.line, end)};
  });
// 导入模块
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object") 
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd) 
    define(["../../lib/codemirror"], mod);
  // 如果是普通浏览器环境
  else 
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 获取光标位置
  var Pos = CodeMirror.Pos;

  // 判断提示是否匹配
  function matches(hint, typed, matchInMiddle) {
    if (matchInMiddle) return hint.indexOf(typed) >= 0;
    else return hint.lastIndexOf(typed, 0) == 0;
  }

  // 获取提示
  function getHints(cm, options) {
    // 获取标签信息
    var tags = options && options.schemaInfo;
    // 获取引号字符
    var quote = (options && options.quoteChar) || '"';
    // 是否在中间匹配
    var matchInMiddle = options && options.matchInMiddle;
    if (!tags) return;
    // 获取光标位置和 token
    var cur = cm.getCursor(), token = cm.getTokenAt(cur);
    if (token.end > cur.ch) {
      token.end = cur.ch;
      token.string = token.string.slice(0, cur.ch - token.start);
    }
    // 获取内部模式
    var inner = CodeMirror.innerMode(cm.getMode(), token.state);
    if (!inner.mode.xmlCurrentTag) return
    var result = [], replaceToken = false, prefix;
    var tag = /\btag\b/.test(token.type) && !/>$/.test(token.string);
    var tagName = tag && /^\w/.test(token.string), tagStart;

    if (tagName) {
      var before = cm.getLine(cur.line).slice(Math.max(0, token.start - 2), token.start);
      var tagType = /<\/$/.test(before) ? "close" : /<$/.test(before) ? "open" : null;
      if (tagType) tagStart = token.start - (tagType == "close" ? 2 : 1);
    } else if (tag && token.string == "<") {
      tagType = "open";
    } else if (tag && token.string == "</") {
      tagType = "close";
    }

    // 获取标签信息
    var tagInfo = inner.mode.xmlCurrentTag(inner.state)
    # 如果标签不存在并且标签信息也不存在，或者标签类型存在
    if (!tag && !tagInfo || tagType) {
      # 如果存在标签名，则将其赋值给前缀
      if (tagName)
        prefix = token.string;
      # 将替换标记设置为标签类型
      replaceToken = tagType;
      # 获取当前 XML 上下文，如果存在则获取内部模式
      var context = inner.mode.xmlCurrentContext ? inner.mode.xmlCurrentContext(inner.state) : []
      # 获取内部标签
      var inner = context.length && context[context.length - 1]
      # 获取当前标签
      var curTag = inner && tags[inner]
      # 获取子标签列表
      var childList = inner ? curTag && curTag.children : tags["!top"];
      # 如果子标签列表存在并且标签类型不是关闭标签
      if (childList && tagType != "close") {
        # 遍历子标签列表，如果前缀存在或者匹配中间位置，则将标签名加入结果列表
        for (var i = 0; i < childList.length; ++i) if (!prefix || matches(childList[i], prefix, matchInMiddle))
          result.push("<" + childList[i]);
      } else if (tagType != "close") {
        # 如果子标签列表不存在并且标签类型不是关闭标签
        # 遍历所有标签，将标签名加入结果列表
        for (var name in tags)
          if (tags.hasOwnProperty(name) && name != "!top" && name != "!attrs" && (!prefix || matches(name, prefix, matchInMiddle)))
            result.push("<" + name);
      }
      # 如果内部标签存在并且前缀不存在或者标签类型是关闭标签并且匹配中间位置
      if (inner && (!prefix || tagType == "close" && matches(inner, prefix, matchInMiddle)))
        result.push("</" + inner + ">");
    }
    # 返回结果对象
    return {
      list: result,
      from: replaceToken ? Pos(cur.line, tagStart == null ? token.start : tagStart) : cur,
      to: replaceToken ? Pos(cur.line, token.end) : cur
    };
  }

  # 注册 XML 提示的帮助函数
  CodeMirror.registerHelper("hint", "xml", getHints);
// 声明一个匿名函数，传入一个 mod 参数
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境
  else
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 注册 JSON 的 lint 助手
  CodeMirror.registerHelper("lint", "json", function(text) {
    var found = [];
    // 如果 window.jsonlint 未定义
    if (!window.jsonlint) {
      // 如果有控制台，输出错误信息
      if (window.console) {
        window.console.error("Error: window.jsonlint not defined, CodeMirror JSON linting cannot run.");
      }
      return found;
    }
    // 对于 jsonlint 的 web dist，jsonlint 被导出为一个带有单个属性 parser 的对象，其中 parseError 是一个子属性
    var jsonlint = window.jsonlint.parser || window.jsonlint
    // 定义 jsonlint 的 parseError 方法
    jsonlint.parseError = function(str, hash) {
      var loc = hash.loc;
      // 将错误信息添加到 found 数组中
      found.push({from: CodeMirror.Pos(loc.first_line - 1, loc.first_column),
                  to: CodeMirror.Pos(loc.last_line - 1, loc.last_column),
                  message: str});
    };
    try { jsonlint.parse(text); }
    catch(e) {}
    return found;
  });
});
})(function(CodeMirror) {
  "use strict";
  // 定义一个常量，用于表示代码镜像的 lint 标记
  var GUTTER_ID = "CodeMirror-lint-markers";

  // 显示提示框
  function showTooltip(cm, e, content) {
    // 创建一个 div 元素作为提示框
    var tt = document.createElement("div");
    // 设置提示框的类名
    tt.className = "CodeMirror-lint-tooltip cm-s-" + cm.options.theme;
    // 将内容克隆到提示框中
    tt.appendChild(content.cloneNode(true));
    // 根据选项决定将提示框添加到代码镜像的包裹元素中还是添加到 body 中
    if (cm.state.lint.options.selfContain)
      cm.getWrapperElement().appendChild(tt);
    else
      document.body.appendChild(tt);

    // 定义提示框的位置
    function position(e) {
      if (!tt.parentNode) return CodeMirror.off(document, "mousemove", position);
      tt.style.top = Math.max(0, e.clientY - tt.offsetHeight - 5) + "px";
      tt.style.left = (e.clientX + 5) + "px";
    }
    CodeMirror.on(document, "mousemove", position);
    position(e);
    if (tt.style.opacity != null) tt.style.opacity = 1;
    return tt;
  }
  // 移除元素
  function rm(elt) {
    if (elt.parentNode) elt.parentNode.removeChild(elt);
  }
  // 隐藏提示框
  function hideTooltip(tt) {
    if (!tt.parentNode) return;
    if (tt.style.opacity == null) rm(tt);
    tt.style.opacity = 0;
    setTimeout(function() { rm(tt); }, 600);
  }

  // 为指定节点显示提示框
  function showTooltipFor(cm, e, content, node) {
    var tooltip = showTooltip(cm, e, content);
    function hide() {
      CodeMirror.off(node, "mouseout", hide);
      if (tooltip) { hideTooltip(tooltip); tooltip = null; }
    }
    var poll = setInterval(function() {
      if (tooltip) for (var n = node;; n = n.parentNode) {
        if (n && n.nodeType == 11) n = n.host;
        if (n == document.body) return;
        if (!n) { hide(); break; }
      }
      if (!tooltip) return clearInterval(poll);
    }, 400);
    CodeMirror.on(node, "mouseout", hide);
  }

  // LintState 类的构造函数
  function LintState(cm, options, hasGutter) {
    this.marked = [];
    this.options = options;
    this.timeout = null;
    this.hasGutter = hasGutter;
    this.onMouseOver = function(e) { onMouseOver(cm, e); };
    this.waitingFor = 0
  }

  // 解析选项
  function parseOptions(_cm, options) {
    if (options instanceof Function) return {getAnnotations: options};
    // 如果选项为空或者为 true，则将其设置为空对象
    if (!options || options === true) options = {};
    // 返回选项对象
    return options;
  }

  // 清除代码编辑器中的标记
  function clearMarks(cm) {
    // 获取 lint 状态
    var state = cm.state.lint;
    // 如果存在行号标记，则清除行号标记
    if (state.hasGutter) cm.clearGutter(GUTTER_ID);
    // 清除所有标记
    for (var i = 0; i < state.marked.length; ++i)
      state.marked[i].clear();
    // 重置标记数组
    state.marked.length = 0;
  }

  // 创建 lint 标记
  function makeMarker(cm, labels, severity, multiple, tooltips) {
    // 创建标记元素
    var marker = document.createElement("div"), inner = marker;
    // 设置标记元素的类名
    marker.className = "CodeMirror-lint-marker-" + severity;
    // 如果是多个标记，则创建内部标记元素
    if (multiple) {
      inner = marker.appendChild(document.createElement("div"));
      inner.className = "CodeMirror-lint-marker-multiple";
    }

    // 如果 tooltips 不为 false，则添加鼠标悬停事件
    if (tooltips != false) CodeMirror.on(inner, "mouseover", function(e) {
      showTooltipFor(cm, e, labels, inner);
    });

    // 返回标记元素
    return marker;
  }

  // 获取最大的严重程度
  function getMaxSeverity(a, b) {
    // 如果 a 是 "error"，则返回 a，否则返回 b
    if (a == "error") return a;
    else return b;
  }

  // 按行分组注释
  function groupByLine(annotations) {
    // 创建空数组
    var lines = [];
    // 遍历注释数组
    for (var i = 0; i < annotations.length; ++i) {
      var ann = annotations[i], line = ann.from.line;
      // 将注释按行号分组
      (lines[line] || (lines[line] = [])).push(ann);
    }
    // 返回分组后的数组
    return lines;
  }

  // 创建注释提示
  function annotationTooltip(ann) {
    // 获取严重程度
    var severity = ann.severity;
    // 如果没有严重程度，则默认为 "error"
    if (!severity) severity = "error";
    // 创建提示元素
    var tip = document.createElement("div");
    // 设置提示元素的类名
    tip.className = "CodeMirror-lint-message-" + severity;
    // 如果存在 messageHTML，则设置提示元素的 HTML 内容，否则设置文本内容
    if (typeof ann.messageHTML != 'undefined') {
      tip.innerHTML = ann.messageHTML;
    } else {
      tip.appendChild(document.createTextNode(ann.message));
    }
    // 返回提示元素
    return tip;
  }

  // 异步进行 lint 检查
  function lintAsync(cm, getAnnotations, passOptions) {
    // 获取 lint 状态
    var state = cm.state.lint
    // 增加等待计数
    var id = ++state.waitingFor
    // 定义中止函数
    function abort() {
      id = -1
      cm.off("change", abort)
    }
    // 监听编辑器变化事件，以中止 lint 检查
    cm.on("change", abort)
  // 获取编辑器内容的注解，并执行回调函数
  getAnnotations(cm.getValue(), function(annotations, arg2) {
    // 取消编辑器内容改变事件的监听
    cm.off("change", abort)
    // 如果等待的状态不是当前id，则返回
    if (state.waitingFor != id) return
    // 如果arg2存在且annotations是CodeMirror实例，则将arg2赋值给annotations
    if (arg2 && annotations instanceof CodeMirror) annotations = arg2
    // 执行操作，更新linting
    cm.operation(function() {updateLinting(cm, annotations)})
  }, passOptions, cm);
}

// 开始linting
function startLinting(cm) {
  var state = cm.state.lint, options = state.options;
  /*
   * 通过`options`属性传递规则，防止JSHint（和其他linters）抱怨无法识别的规则，如`onUpdateLinting`、`delay`、`lintOnChange`等。
   */
  var passOptions = options.options || options;
  var getAnnotations = options.getAnnotations || cm.getHelper(CodeMirror.Pos(0, 0), "lint");
  if (!getAnnotations) return;
  if (options.async || getAnnotations.async) {
    // 如果是异步的，则执行lintAsync函数
    lintAsync(cm, getAnnotations, passOptions)
  } else {
    var annotations = getAnnotations(cm.getValue(), passOptions, cm);
    if (!annotations) return;
    if (annotations.then) annotations.then(function(issues) {
      // 如果annotations是promise对象，则执行回调函数
      cm.operation(function() {updateLinting(cm, issues)})
    });
    else cm.operation(function() {updateLinting(cm, annotations)})
  }
}

// 更新linting
function updateLinting(cm, annotationsNotSorted) {
  // 清除所有标记
  clearMarks(cm);
  var state = cm.state.lint, options = state.options;

  // 按行分组注解
  var annotations = groupByLine(annotationsNotSorted);
    // 遍历注释数组，处理每一行的注释
    for (var line = 0; line < annotations.length; ++line) {
      // 获取当前行的注释
      var anns = annotations[line];
      // 如果当前行没有注释，则跳过
      if (!anns) continue;

      // 初始化最大严重性和提示标签
      var maxSeverity = null;
      var tipLabel = state.hasGutter && document.createDocumentFragment();

      // 遍历当前行的注释
      for (var i = 0; i < anns.length; ++i) {
        // 获取当前注释
        var ann = anns[i];
        // 获取注释的严重性，如果没有则默认为"error"
        var severity = ann.severity;
        if (!severity) severity = "error";
        // 更新最大严重性
        maxSeverity = getMaxSeverity(maxSeverity, severity);

        // 如果有自定义的注释格式化函数，则对注释进行格式化
        if (options.formatAnnotation) ann = options.formatAnnotation(ann);
        // 如果存在提示标签，则添加注释的提示
        if (state.hasGutter) tipLabel.appendChild(annotationTooltip(ann));

        // 如果注释有结束位置，则在编辑器中标记出注释的位置范围
        if (ann.to) state.marked.push(cm.markText(ann.from, ann.to, {
          className: "CodeMirror-lint-mark-" + severity,
          __annotation: ann
        }));
      }

      // 如果存在提示标签，则设置编辑器的行号标记
      if (state.hasGutter)
        cm.setGutterMarker(line, GUTTER_ID, makeMarker(cm, tipLabel, maxSeverity, anns.length > 1,
                                                       state.options.tooltips));
    }
    // 如果存在更新注释的回调函数，则调用该函数
    if (options.onUpdateLinting) options.onUpdateLinting(annotationsNotSorted, annotations, cm);
  }

  // 编辑器内容改变时的事件处理函数
  function onChange(cm) {
    var state = cm.state.lint;
    if (!state) return;
    // 清除之前的定时器，设置新的定时器用于延迟触发代码检查
    clearTimeout(state.timeout);
    state.timeout = setTimeout(function(){startLinting(cm);}, state.options.delay || 500);
  }

  // 鼠标悬停时显示提示信息
  function popupTooltips(cm, annotations, e) {
    var target = e.target || e.srcElement;
    var tooltip = document.createDocumentFragment();
    // 遍历注释数组，为每个注释添加提示信息
    for (var i = 0; i < annotations.length; i++) {
      var ann = annotations[i];
      tooltip.appendChild(annotationTooltip(ann));
    }
    // 在鼠标位置显示提示信息
    showTooltipFor(cm, e, tooltip, target);
  }

  // 鼠标悬停在标记上时的事件处理函数
  function onMouseOver(cm, e) {
    var target = e.target || e.srcElement;
    // 如果鼠标悬停在标记上，则获取标记的位置信息
    if (!/\bCodeMirror-lint-mark-/.test(target.className)) return;
    var box = target.getBoundingClientRect(), x = (box.left + box.right) / 2, y = (box.top + box.bottom) / 2;
    var spans = cm.findMarksAt(cm.coordsChar({left: x, top: y}, "client"));

    var annotations = [];
    // 遍历 spans 数组，获取每个元素的 __annotation 属性，如果存在则将其添加到 annotations 数组中
    for (var i = 0; i < spans.length; ++i) {
      var ann = spans[i].__annotation;
      if (ann) annotations.push(ann);
    }
    // 如果 annotations 数组不为空，则调用 popupTooltips 方法显示提示框
    if (annotations.length) popupTooltips(cm, annotations, e);
  }

  // 定义 lint 选项，当值发生变化时执行相应操作
  CodeMirror.defineOption("lint", false, function(cm, val, old) {
    // 如果旧值存在且不等于 CodeMirror.Init，则清除标记、解绑事件和定时器，并删除 lint 状态
    if (old && old != CodeMirror.Init) {
      clearMarks(cm);
      if (cm.state.lint.options.lintOnChange !== false)
        cm.off("change", onChange);
      CodeMirror.off(cm.getWrapperElement(), "mouseover", cm.state.lint.onMouseOver);
      clearTimeout(cm.state.lint.timeout);
      delete cm.state.lint;
    }

    // 如果新值存在
    if (val) {
      // 获取 gutters 和是否存在 lint gutter 的状态
      var gutters = cm.getOption("gutters"), hasLintGutter = false;
      for (var i = 0; i < gutters.length; ++i) if (gutters[i] == GUTTER_ID) hasLintGutter = true;
      // 创建 lint 状态对象，并根据选项绑定事件
      var state = cm.state.lint = new LintState(cm, parseOptions(cm, val), hasLintGutter);
      if (state.options.lintOnChange !== false)
        cm.on("change", onChange);
      // 如果选项中 tooltips 不为 false 且不为 "gutter"，则绑定鼠标悬停事件
      if (state.options.tooltips != false && state.options.tooltips != "gutter")
        CodeMirror.on(cm.getWrapperElement(), "mouseover", state.onMouseOver);

      // 开始 linting
      startLinting(cm);
    }
  });

  // 定义 performLint 方法，执行 lint 操作
  CodeMirror.defineExtension("performLint", function() {
    // 如果存在 lint 状态，则开始 linting
    if (this.state.lint) startLinting(this);
  });
});
// 结束匿名函数

/* ---- extension/scroll/annotatescrollbar.js ---- */
// 扩展插件的注释信息

// CodeMirror, 版权归 Marijn Haverbeke 和其他人所有
// 使用 MIT 许可证分发：https://codemirror.net/LICENSE

(function(mod) {
  if (typeof exports == "object" && typeof module == "object") // CommonJS
    mod(require("../../lib/codemirror"));
  else if (typeof define == "function" && define.amd) // AMD
    define(["../../lib/codemirror"], mod);
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义 annotateScrollbar 方法
  CodeMirror.defineExtension("annotateScrollbar", function(options) {
    if (typeof options == "string") options = {className: options};
    return new Annotation(this, options);
  });

  // 定义 scrollButtonHeight 选项
  CodeMirror.defineOption("scrollButtonHeight", 0);

  // Annotation 构造函数
  function Annotation(cm, options) {
    this.cm = cm;
    this.options = options;
    this.buttonHeight = options.scrollButtonHeight || cm.getOption("scrollButtonHeight");
    this.annotations = [];
    this.doRedraw = this.doUpdate = null;
    this.div = cm.getWrapperElement().appendChild(document.createElement("div"));
    this.div.style.cssText = "position: absolute; right: 0; top: 0; z-index: 7; pointer-events: none";
    this.computeScale();

    // 定义 scheduleRedraw 方法
    function scheduleRedraw(delay) {
      clearTimeout(self.doRedraw);
      self.doRedraw = setTimeout(function() { self.redraw(); }, delay);
    }

    var self = this;
    // 监听编辑器的 refresh 事件
    cm.on("refresh", this.resizeHandler = function() {
      clearTimeout(self.doUpdate);
      self.doUpdate = setTimeout(function() {
        if (self.computeScale()) scheduleRedraw(20);
      }, 100);
    });
    // 监听 markerAdded 事件
    cm.on("markerAdded", this.resizeHandler);
    // 监听 markerCleared 事件
    cm.on("markerCleared", this.resizeHandler);
    // 如果 listenForChanges 选项不为 false，则监听 changes 事件
    if (options.listenForChanges !== false)
      cm.on("changes", this.changeHandler = function() {
        scheduleRedraw(250);
      });
  }

  // 计算缩放比例
  Annotation.prototype.computeScale = function() {
    var cm = this.cm;
    # 计算垂直缩放比例，使得注释栏高度适应编辑器高度
    var hScale = (cm.getWrapperElement().clientHeight - cm.display.barHeight - this.buttonHeight * 2) /
      cm.getScrollerElement().scrollHeight
    # 如果计算得到的垂直缩放比例与之前的不同，则更新并返回 true
    if (hScale != this.hScale) {
      this.hScale = hScale;
      return true;
    }
  };

  # 更新注释内容
  Annotation.prototype.update = function(annotations) {
    this.annotations = annotations;
    this.redraw();
  };

  # 重新绘制注释
  Annotation.prototype.redraw = function(compute) {
    # 如果需要重新计算缩放比例，则调用 computeScale 方法
    if (compute !== false) this.computeScale();
    # 获取编辑器对象和垂直缩放比例
    var cm = this.cm, hScale = this.hScale;

    # 创建文档片段和注释数组
    var frag = document.createDocumentFragment(), anns = this.annotations;

    # 获取是否启用了自动换行和单行高度
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
      # 获取注释起始行的纵坐标乘以水平比例尺
      var top = nextTop || getY(ann.from, true) * hScale;
      # 获取注释结束行的纵坐标乘以水平比例尺
      var bottom = getY(ann.to, false) * hScale;
      # 当循环未结束时
      while (i < anns.length - 1) {
        # 如果下一个注释的结束行大于最后一行，则跳出循环
        if (anns[i + 1].to.line > lastLine) break;
        # 获取下一个注释起始行的纵坐标乘以水平比例尺
        nextTop = getY(anns[i + 1].from, true) * hScale;
        # 如果下一个注释的起始行的纵坐标大于当前注释的结束行的纵坐标加上0.9，则跳出循环
        if (nextTop > bottom + .9) break;
        # 获取下一个注释对象
        ann = anns[++i];
        # 获取下一个注释的结束行的纵坐标乘以水平比例尺
        bottom = getY(ann.to, false) * hScale;
      }
      # 如果注释的起始行和结束行的纵坐标相等，则继续下一次循环
      if (bottom == top) continue;
      # 获取注释的高度，取最大值为底部纵坐标减去顶部纵坐标或3
      var height = Math.max(bottom - top, 3);

      # 创建一个div元素并添加到frag中
      var elt = frag.appendChild(document.createElement("div"));
      # 设置div元素的样式
      elt.style.cssText = "position: absolute; right: 0px; width: " + Math.max(cm.display.barWidth - 1, 2) + "px; top: "
        + (top + this.buttonHeight) + "px; height: " + height + "px";
      # 设置div元素的类名
      elt.className = this.options.className;
      # 如果注释有id，则设置div元素的annotation-id属性为注释的id
      if (ann.id) {
        elt.setAttribute("annotation-id", ann.id);
      }
    }
    # 清空this.div的文本内容
    this.div.textContent = "";
    # 将frag添加到this.div中
    this.div.appendChild(frag);
  };

  # 清除注释
  Annotation.prototype.clear = function() {
    # 移除事件监听器
    this.cm.off("refresh", this.resizeHandler);
    this.cm.off("markerAdded", this.resizeHandler);
    this.cm.off("markerCleared", this.resizeHandler);
    if (this.changeHandler) this.cm.off("changes", this.changeHandler);
    # 移除this.div元素
    this.div.parentNode.removeChild(this.div);
  };
// 定义一个匿名函数，传入一个 mod 参数
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror")); // 调用 mod 函数并传入 codemirror 模块
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror"], mod); // 使用 define 定义模块，并传入 codemirror 模块和 mod 函数
  // 如果是普通的浏览器环境
  else
    mod(CodeMirror); // 调用 mod 函数并传入 CodeMirror 对象
})(function(CodeMirror) {
  "use strict"; // 开启严格模式

  // 定义 CodeMirror 的 scrollPastEnd 选项
  CodeMirror.defineOption("scrollPastEnd", false, function(cm, val, old) {
    // 如果旧值存在且不是初始化值
    if (old && old != CodeMirror.Init) {
      cm.off("change", onChange); // 移除 change 事件的监听器
      cm.off("refresh", updateBottomMargin); // 移除 refresh 事件的监听器
      cm.display.lineSpace.parentNode.style.paddingBottom = ""; // 设置行空间的底部内边距为空
      cm.state.scrollPastEndPadding = null; // 设置 scrollPastEndPadding 为 null
    }
    // 如果新值为 true
    if (val) {
      cm.on("change", onChange); // 添加 change 事件的监听器
      cm.on("refresh", updateBottomMargin); // 添加 refresh 事件的监听器
      updateBottomMargin(cm); // 调用 updateBottomMargin 函数
    }
  });

  // 当编辑器内容改变时的处理函数
  function onChange(cm, change) {
    // 如果改变的行是最后一行
    if (CodeMirror.changeEnd(change).line == cm.lastLine())
      updateBottomMargin(cm); // 调用 updateBottomMargin 函数
  }

  // 更新底部内边距的函数
  function updateBottomMargin(cm) {
    var padding = ""; // 初始化 padding 变量为空字符串
    // 如果编辑器有多于一行的内容
    if (cm.lineCount() > 1) {
      var totalH = cm.display.scroller.clientHeight - 30; // 获取编辑器可视区域的高度减去 30
      var lastLineH = cm.getLineHandle(cm.lastLine()).height; // 获取最后一行的高度
      padding = (totalH - lastLineH) + "px"; // 计算底部内边距
    }
    // 如果当前的底部内边距与新计算的底部内边距不一致
    if (cm.state.scrollPastEndPadding != padding) {
      cm.state.scrollPastEndPadding = padding; // 更新 scrollPastEndPadding
      cm.display.lineSpace.parentNode.style.paddingBottom = padding; // 设置行空间的底部内边距
      cm.off("refresh", updateBottomMargin); // 移除 refresh 事件的监听器
      cm.setSize(); // 重新设置编辑器的大小
      cm.on("refresh", updateBottomMargin); // 添加 refresh 事件的监听器
    }
  }
});
    // 如果当前环境支持 CommonJS 规范的模块化加载，则使用 require 引入 codemirror 模块
    mod(require("../../lib/codemirror"));
  // 如果当前环境支持 AMD 规范的模块化加载，则使用 define 定义 codemirror 模块
  else if (typeof define == "function" && define.amd) // AMD
    define(["../../lib/codemirror"], mod);
  // 如果当前环境不支持任何模块化加载规范，则直接将 codemirror 模块挂载到全局对象上
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义 Bar 类
  function Bar(cls, orientation, scroll) {
    // 初始化 Bar 对象的属性
    this.orientation = orientation;
    this.scroll = scroll;
    this.screen = this.total = this.size = 1;
    this.pos = 0;

    // 创建 div 元素作为 Bar 对象的节点
    this.node = document.createElement("div");
    this.node.className = cls + "-" + orientation;
    this.inner = this.node.appendChild(document.createElement("div"));

    var self = this;
    // 监听鼠标按下事件
    CodeMirror.on(this.inner, "mousedown", function(e) {
      if (e.which != 1) return;
      CodeMirror.e_preventDefault(e);
      var axis = self.orientation == "horizontal" ? "pageX" : "pageY";
      var start = e[axis], startpos = self.pos;
      // 定义鼠标移动和鼠标松开事件处理函数
      function done() {
        CodeMirror.off(document, "mousemove", move);
        CodeMirror.off(document, "mouseup", done);
      }
      function move(e) {
        if (e.which != 1) return done();
        self.moveTo(startpos + (e[axis] - start) * (self.total / self.size));
      }
      // 监听鼠标移动和鼠标松开事件
      CodeMirror.on(document, "mousemove", move);
      CodeMirror.on(document, "mouseup", done);
    });

    // 监听节点的点击事件
    CodeMirror.on(this.node, "click", function(e) {
      CodeMirror.e_preventDefault(e);
      var innerBox = self.inner.getBoundingClientRect(), where;
      if (self.orientation == "horizontal")
        where = e.clientX < innerBox.left ? -1 : e.clientX > innerBox.right ? 1 : 0;
      else
        where = e.clientY < innerBox.top ? -1 : e.clientY > innerBox.bottom ? 1 : 0;
      self.moveTo(self.pos + where * self.screen);
    });

    // 定义滚轮事件处理函数
    function onWheel(e) {
      var moved = CodeMirror.wheelEventPixels(e)[self.orientation == "horizontal" ? "x" : "y"];
      var oldPos = self.pos;
      self.moveTo(self.pos + moved);
      if (self.pos != oldPos) CodeMirror.e_preventDefault(e);
    }
    // 监听鼠标滚轮事件
    CodeMirror.on(this.node, "mousewheel", onWheel);
    CodeMirror.on(this.node, "DOMMouseScroll", onWheel);
  }

  // 设置 Bar 对象的位置
  Bar.prototype.setPos = function(pos, force) {
    if (pos < 0) pos = 0;
    // 如果滚动位置大于总长度减去屏幕长度，将滚动位置设置为总长度减去屏幕长度
    if (pos > this.total - this.screen) pos = this.total - this.screen;
    // 如果不是强制移动且滚动位置等于当前位置，返回 false
    if (!force && pos == this.pos) return false;
    // 设置当前位置为传入的位置
    this.pos = pos;
    // 根据水平或垂直方向设置滚动条的位置
    this.inner.style[this.orientation == "horizontal" ? "left" : "top"] =
      (pos * (this.size / this.total)) + "px";
    // 返回 true
    return true
  };

  // 移动滚动条到指定位置
  Bar.prototype.moveTo = function(pos) {
    // 如果设置位置成功，调用滚动方法
    if (this.setPos(pos)) this.scroll(pos, this.orientation);
  }

  // 最小按钮大小为 10
  var minButtonSize = 10;

  // 更新滚动条
  Bar.prototype.update = function(scrollSize, clientSize, barSize) {
    // 判断尺寸是否改变
    var sizeChanged = this.screen != clientSize || this.total != scrollSize || this.size != barSize
    // 如果尺寸改变
    if (sizeChanged) {
      // 更新屏幕长度、总长度和滚动条长度
      this.screen = clientSize;
      this.total = scrollSize;
      this.size = barSize;
    }
    // 计算按钮大小
    var buttonSize = this.screen * (this.size / this.total);
    // 如果按钮大小小于最小按钮大小
    if (buttonSize < minButtonSize) {
      // 减小滚动条长度，使按钮大小达到最小按钮大小
      this.size -= minButtonSize - buttonSize;
      buttonSize = minButtonSize;
    }
    // 根据水平或垂直方向设置滚动条内部的宽度或高度
    this.inner.style[this.orientation == "horizontal" ? "width" : "height"] =
      buttonSize + "px";
    // 设置滚动位置
    this.setPos(this.pos, sizeChanged);
  };

  // 创建简单滚动条
  function SimpleScrollbars(cls, place, scroll) {
    // 初始化水平和垂直滚动条
    this.addClass = cls;
    this.horiz = new Bar(cls, "horizontal", scroll);
    place(this.horiz.node);
    this.vert = new Bar(cls, "vertical", scroll);
    place(this.vert.node);
    this.width = null;
  }

  // 更新简单滚动条
  SimpleScrollbars.prototype.update = function(measure) {
    // 如果宽度为空
    if (this.width == null) {
      // 获取水平滚动条节点的样式
      var style = window.getComputedStyle ? window.getComputedStyle(this.horiz.node) : this.horiz.node.currentStyle;
      // 如果获取到样式，将宽度设置为样式的高度
      if (style) this.width = parseInt(style.height);
    }
    // 获取宽度
    var width = this.width || 0;

    // 判断是否需要水平滚动条和垂直滚动条
    var needsH = measure.scrollWidth > measure.clientWidth + 1;
    var needsV = measure.scrollHeight > measure.clientHeight + 1;
    // 根据需要显示或隐藏垂直滚动条和水平滚动条
    this.vert.node.style.display = needsV ? "block" : "none";
    this.horiz.node.style.display = needsH ? "block" : "none";
    // 如果需要垂直滚动条
    if (needsV) {
      // 更新垂直滚动条的位置和尺寸
      this.vert.update(measure.scrollHeight, measure.clientHeight,
                       measure.viewHeight - (needsH ? width : 0));
      // 根据水平滚动条的存在与否，设置垂直滚动条的底部位置
      this.vert.node.style.bottom = needsH ? width + "px" : "0";
    }
    // 如果需要水平滚动条
    if (needsH) {
      // 更新水平滚动条的位置和尺寸
      this.horiz.update(measure.scrollWidth, measure.clientWidth,
                        measure.viewWidth - (needsV ? width : 0) - measure.barLeft);
      // 根据垂直滚动条的存在与否，设置水平滚动条的右侧位置
      this.horiz.node.style.right = needsV ? width + "px" : "0";
      // 设置水平滚动条的左侧位置
      this.horiz.node.style.left = measure.barLeft + "px";
    }

    // 返回滚动条的右侧和底部位置
    return {right: needsV ? width : 0, bottom: needsH ? width : 0};
  };

  // 设置垂直滚动条的位置
  SimpleScrollbars.prototype.setScrollTop = function(pos) {
    this.vert.setPos(pos);
  };

  // 设置水平滚动条的位置
  SimpleScrollbars.prototype.setScrollLeft = function(pos) {
    this.horiz.setPos(pos);
  };

  // 清除滚动条
  SimpleScrollbars.prototype.clear = function() {
    // 获取水平滚动条的父节点，并移除水平和垂直滚动条
    var parent = this.horiz.node.parentNode;
    parent.removeChild(this.horiz.node);
    parent.removeChild(this.vert.node);
  };

  // 创建简单滚动条模型
  CodeMirror.scrollbarModel.simple = function(place, scroll) {
    return new SimpleScrollbars("CodeMirror-simplescroll", place, scroll);
  };
  // 创建覆盖滚动条模型
  CodeMirror.scrollbarModel.overlay = function(place, scroll) {
    return new SimpleScrollbars("CodeMirror-overlayscroll", place, scroll);
  };
// 定义了 jumpToLine 命令，如果存在 dialog.js 则使用它
(function(mod) {
  if (typeof exports == "object" && typeof module == "object") // CommonJS
    mod(require("../../lib/codemirror"), require("../dialog/dialog"));
  else if (typeof define == "function" && define.amd) // AMD
    define(["../../lib/codemirror", "../dialog/dialog"], mod);
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 弹出对话框函数，如果存在 openDialog 则使用它，否则使用 prompt
  function dialog(cm, text, shortText, deflt, f) {
    if (cm.openDialog) cm.openDialog(text, f, {value: deflt, selectValueOnOpen: true});
    else f(prompt(shortText, deflt));
  }

  // 获取跳转对话框内容
  function getJumpDialog(cm) {
    return cm.phrase("Jump to line:") + ' <input type="text" style="width: 10em" class="CodeMirror-search-field"/> <span style="color: #888" class="CodeMirror-search-hint">' + cm.phrase("(Use line:column or scroll% syntax)") + '</span>';
  }

  // 解释行号
  function interpretLine(cm, string) {
    var num = Number(string)
    if (/^[-+]/.test(string)) return cm.getCursor().line + num
    else return num - 1
  }

  // 定义 jumpToLine 命令
  CodeMirror.commands.jumpToLine = function(cm) {
    var cur = cm.getCursor();
    // 弹出跳转对话框
    dialog(cm, getJumpDialog(cm), cm.phrase("Jump to line:"), (cur.line + 1) + ":" + cur.ch, function(posStr) {
      if (!posStr) return;

      var match;
      // 解析输入的位置字符串
      if (match = /^\s*([\+\-]?\d+)\s*\:\s*(\d+)\s*$/.exec(posStr)) {
        cm.setCursor(interpretLine(cm, match[1]), Number(match[2]))
      } else if (match = /^\s*([\+\-]?\d+(\.\d+)?)\%\s*/.exec(posStr)) {
        var line = Math.round(cm.lineCount() * Number(match[1]) / 100);
        if (/^[-+]/.test(match[1])) line = cur.line + line + 1;
        cm.setCursor(line - 1, cur.ch);
      } else if (match = /^\s*\:?\s*([\+\-]?\d+)\s*/.exec(posStr)) {
        cm.setCursor(interpretLine(cm, match[1]), cur.ch);
      }
    // 在 CodeMirror 默认键映射中，将 Alt-G 键映射为跳转到指定行的功能
    CodeMirror.keyMap["default"]["Alt-G"] = "jumpToLine";
// 定义了一个匿名函数，接受一个 mod 参数
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror"), require("./matchesonscrollbar"));
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror", "./matchesonscrollbar"], mod);
  // 如果是普通的浏览器环境
  else
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 默认配置项
  var defaults = {
    style: "matchhighlight",
    minChars: 2,
    delay: 100,
    wordsOnly: false,
    annotateScrollbar: false,
    showToken: false,
    trim: true
  }

  // State 对象的构造函数
  function State(options) {
    this.options = {}
    // 遍历默认配置项，将传入的配置项覆盖默认配置项
    for (var name in defaults)
      this.options[name] = (options && options.hasOwnProperty(name) ? options : defaults)[name]
    this.overlay = this.timeout = null;
    # 初始化匹配滚动和激活状态
    this.matchesonscroll = null;
    this.active = false;
  }

  # 定义代码镜像的高亮选择匹配选项
  CodeMirror.defineOption("highlightSelectionMatches", false, function(cm, val, old) {
    # 如果旧值存在且不是初始化状态，则移除覆盖层，清除超时，取消事件监听
    if (old && old != CodeMirror.Init) {
      removeOverlay(cm);
      clearTimeout(cm.state.matchHighlighter.timeout);
      cm.state.matchHighlighter = null;
      cm.off("cursorActivity", cursorActivity);
      cm.off("focus", onFocus)
    }
    # 如果新值存在
    if (val) {
      # 创建状态对象
      var state = cm.state.matchHighlighter = new State(val);
      # 如果编辑器有焦点，则激活状态并高亮匹配
      if (cm.hasFocus()) {
        state.active = true
        highlightMatches(cm)
      } else {
        # 否则在焦点事件时激活状态
        cm.on("focus", onFocus)
      }
      # 在光标活动事件时触发匹配高亮
      cm.on("cursorActivity", cursorActivity);
    }
  });

  # 光标活动事件处理函数
  function cursorActivity(cm) {
    var state = cm.state.matchHighlighter;
    # 如果状态激活或编辑器有焦点，则调度高亮匹配
    if (state.active || cm.hasFocus()) scheduleHighlight(cm, state)
  }

  # 焦点事件处理函数
  function onFocus(cm) {
    var state = cm.state.matchHighlighter
    # 如果状态未激活，则激活状态并调度高亮匹配
    if (!state.active) {
      state.active = true
      scheduleHighlight(cm, state)
    }
  }

  # 调度高亮匹配
  function scheduleHighlight(cm, state) {
    clearTimeout(state.timeout);
    state.timeout = setTimeout(function() {highlightMatches(cm);}, state.options.delay);
  }

  # 添加覆盖层
  function addOverlay(cm, query, hasBoundary, style) {
    var state = cm.state.matchHighlighter;
    # 添加覆盖层并在滚动条上显示匹配
    cm.addOverlay(state.overlay = makeOverlay(query, hasBoundary, style));
    if (state.options.annotateScrollbar && cm.showMatchesOnScrollbar) {
      var searchFor = hasBoundary ? new RegExp((/\w/.test(query.charAt(0)) ? "\\b" : "") +
                                               query.replace(/[\\\[.+*?(){|^$]/g, "\\$&") +
                                               (/\w/.test(query.charAt(query.length - 1)) ? "\\b" : "")) : query;
      state.matchesonscroll = cm.showMatchesOnScrollbar(searchFor, false,
        {className: "CodeMirror-selection-highlight-scrollbar"});
    }
  }

  # 移除覆盖层
  function removeOverlay(cm) {
    var state = cm.state.matchHighlighter;
    # 如果存在覆盖层
    if (state.overlay):
      # 移除覆盖层
      cm.removeOverlay(state.overlay);
      # 重置覆盖层为 null
      state.overlay = null;
      # 如果滚动时匹配存在
      if (state.matchesonscroll):
        # 清空滚动时匹配
        state.matchesonscroll.clear();
        # 重置滚动时匹配为 null
        state.matchesonscroll = null;
  
  # 高亮匹配
  def highlightMatches(cm):
    # 执行操作
    cm.operation(function():
      # 获取匹配高亮状态
      var state = cm.state.matchHighlighter;
      # 移除覆盖层
      removeOverlay(cm);
      # 如果没有选中内容且选项为显示标记
      if (!cm.somethingSelected() && state.options.showToken):
        # 获取当前光标位置和所在行内容
        var cur = cm.getCursor(), line = cm.getLine(cur.line), start = cur.ch, end = start;
        # 查找标记的起始和结束位置
        while (start && re.test(line.charAt(start - 1))) --start;
        while (end < line.length && re.test(line.charAt(end))) ++end;
        # 如果存在标记
        if (start < end)
          # 添加覆盖层
          addOverlay(cm, line.slice(start, end), re, state.options.style);
        return;
      # 获取选中内容的起始和结束位置
      var from = cm.getCursor("from"), to = cm.getCursor("to");
      # 如果起始和结束在不同行
      if (from.line != to.line) return;
      # 如果选项为仅限单词且不是单词
      if (state.options.wordsOnly && !isWord(cm, from, to)) return;
      # 获取选中内容
      var selection = cm.getRange(from, to)
      # 如果选项为修剪空白字符
      if (state.options.trim) selection = selection.replace(/^\s+|\s+$/g, "")
      # 如果选中内容长度大于等于最小字符数
      if (selection.length >= state.options.minChars)
        # 添加覆盖层
        addOverlay(cm, selection, false, state.options.style);
    );
  
  # 判断是否为单词
  def isWord(cm, from, to):
    # 获取选中内容
    var str = cm.getRange(from, to);
    # 如果匹配单词
    if (str.match(/^\w+$/) !== null):
        # 如果起始位置大于 0
        if (from.ch > 0):
            # 获取前一个字符
            var pos = {line: from.line, ch: from.ch - 1};
            var chr = cm.getRange(pos, from);
            # 如果前一个字符不是非单词字符
            if (chr.match(/\W/) === null) return false;
        # 如果结束位置小于所在行长度
        if (to.ch < cm.getLine(from.line).length):
            # 获取后一个字���
            var pos = {line: to.line, ch: to.ch + 1};
            var chr = cm.getRange(to, pos);
            # 如果后一个字符不是非单词字符
            if (chr.match(/\W/) === null) return false;
        return true;
    else return false;
  
  # 获取边界周围的内容
  def boundariesAround(stream, re):
    # 返回一个布尔值，判断当前位置是否在字符串的开头或者前一个字符不符合正则表达式要求，并且当前位置在字符串末尾或者后一个字符不符合正则表达式要求
    return (!stream.start || !re.test(stream.string.charAt(stream.start - 1))) &&
      (stream.pos == stream.string.length || !re.test(stream.string.charAt(stream.pos)));
  }

  # 创建一个覆盖层，用于匹配指定的查询字符串，并根据条件和样式返回结果
  function makeOverlay(query, hasBoundary, style) {
    return {token: function(stream) {
      # 如果匹配到查询字符串，并且满足边界条件（如果有），则返回指定的样式
      if (stream.match(query) &&
          (!hasBoundary || boundariesAround(stream, hasBoundary)))
        return style;
      # 否则，继续向前匹配，直到匹配到查询字符串的第一个字符，或者匹配到字符串末尾
      stream.next();
      stream.skipTo(query.charAt(0)) || stream.skipToEnd();
    }};
  }
// 匿名函数，接受一个 mod 参数
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror"), require("./searchcursor"), require("../scroll/annotatescrollbar"));
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror", "./searchcursor", "../scroll/annotatescrollbar"], mod);
  // 如果是普通的浏览器环境
  else
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 在 CodeMirror 上定义 showMatchesOnScrollbar 方法
  CodeMirror.defineExtension("showMatchesOnScrollbar", function(query, caseFold, options) {
    // 如果 options 是字符串，则转换为对象
    if (typeof options == "string") options = {className: options};
    // 如果 options 不存在，则初始化为空对象
    if (!options) options = {};
    // 创建 SearchAnnotation 实例
    return new SearchAnnotation(this, query, caseFold, options);
  });

  // SearchAnnotation 构造函数
  function SearchAnnotation(cm, query, caseFold, options) {
    this.cm = cm;
    this.options = options;
    // 初始化 annotateOptions 对象
    var annotateOptions = {listenForChanges: false};
    // 将 options 的属性复制到 annotateOptions
    for (var prop in options) annotateOptions[prop] = options[prop];
    // 如果 annotateOptions 中没有 className 属性，则设置为 "CodeMirror-search-match"
    if (!annotateOptions.className) annotateOptions.className = "CodeMirror-search-match";
    // 在滚动条上创建注释
    this.annotation = cm.annotateScrollbar(annotateOptions);
    this.query = query;
    this.caseFold = caseFold;
    this.gap = {from: cm.firstLine(), to: cm.lastLine() + 1};
    this.matches = [];
    this.update = null;

    // 查找匹配项
    this.findMatches();
    // 更新注释
    this.annotation.update(this.matches);

    var self = this;
    // 监听编辑器内容变化事件
    cm.on("change", this.changeHandler = function(_cm, change) { self.onChange(change); });
  }

  // 最大匹配项数
  var MAX_MATCHES = 1000;

  // 查找匹配项方法
  SearchAnnotation.prototype.findMatches = function() {
    // 如果没有 gap，则返回
    if (!this.gap) return;
    // 遍历匹配项数组
    for (var i = 0; i < this.matches.length; i++) {
      var match = this.matches[i];
      // 如果匹配项的起始行大于等于 gap 的结束行，则跳出循环
      if (match.from.line >= this.gap.to) break;
      // 如果匹配项的结束行大于等于 gap 的起始行，则从数组中删除该匹配项
      if (match.to.line >= this.gap.from) this.matches.splice(i--, 1);
    }
  }
    # 根据查询内容和位置创建搜索游标对象
    var cursor = this.cm.getSearchCursor(this.query, CodeMirror.Pos(this.gap.from, 0), {caseFold: this.caseFold, multiline: this.options.multiline});
    # 设置最大匹配数
    var maxMatches = this.options && this.options.maxMatches || MAX_MATCHES;
    # 循环查找匹配项
    while (cursor.findNext()) {
      # 获取匹配项的起始和结束位置
      var match = {from: cursor.from(), to: cursor.to()};
      # 如果匹配项的起始行大于等于 gap 的结束行，则跳出循环
      if (match.from.line >= this.gap.to) break;
      # 将匹配项插入到 matches 数组中
      this.matches.splice(i++, 0, match);
      # 如果匹配项数量超过最大匹配数，则跳出循环
      if (this.matches.length > maxMatches) break;
    }
    # 重置 gap 为 null
    this.gap = null;
  };

  # 根据改变的起始行和大小改变，计算偏移后的行数
  function offsetLine(line, changeStart, sizeChange) {
    if (line <= changeStart) return line;
    return Math.max(changeStart, line + sizeChange);
  }

  # 当搜索注释对象发生改变时的处理函数
  SearchAnnotation.prototype.onChange = function(change) {
    # 获取改变的起始行和结束行
    var startLine = change.from.line;
    var endLine = CodeMirror.changeEnd(change).line;
    # 计算大小改变
    var sizeChange = endLine - change.to.line;
    # 如果存在 gap
    if (this.gap) {
      # 更新 gap 的起始行和结束行
      this.gap.from = Math.min(offsetLine(this.gap.from, startLine, sizeChange), change.from.line);
      this.gap.to = Math.max(offsetLine(this.gap.to, startLine, sizeChange), change.from.line);
    } else {
      # 创建新的 gap 对象
      this.gap = {from: change.from.line, to: endLine + 1};
    }

    # 如果存在大小改变，更新匹配项的位置
    if (sizeChange) for (var i = 0; i < this.matches.length; i++) {
      var match = this.matches[i];
      var newFrom = offsetLine(match.from.line, startLine, sizeChange);
      if (newFrom != match.from.line) match.from = CodeMirror.Pos(newFrom, match.from.ch);
      var newTo = offsetLine(match.to.line, startLine, sizeChange);
      if (newTo != match.to.line) match.to = CodeMirror.Pos(newTo, match.to.ch);
    }
    # 清除更新定时器
    clearTimeout(this.update);
    # 设置 self 为 this
    var self = this;
    # 设置更新定时器
    this.update = setTimeout(function() { self.updateAfterChange(); }, 250);
  };

  # 在改变后更新匹配项
  SearchAnnotation.prototype.updateAfterChange = function() {
    this.findMatches();
    this.annotation.update(this.matches);
  };

  # 清除搜索注释对象
  SearchAnnotation.prototype.clear = function() {
    this.cm.off("change", this.changeHandler);
    this.annotation.clear();
  };
// 代码块开始

// 代码块结束
  // 如果查询字符串全为小写，则进行不区分大小写的搜索
  return cm.getSearchCursor(query, pos, {caseFold: queryCaseInsensitive(query), multiline: true});
}

function persistentDialog(cm, text, deflt, onEnter, onKeyDown) {
  // 打开持久对话框，设置默认值、回车事件、按键事件等
  cm.openDialog(text, onEnter, {
    value: deflt,
    selectValueOnOpen: true,
    closeOnEnter: false,
    onClose: function() { clearSearch(cm); },
    onKeyDown: onKeyDown
  });
}

function dialog(cm, text, shortText, deflt, f) {
  // 如果支持对话框，则打开对话框，否则使用 prompt 函数
  if (cm.openDialog) cm.openDialog(text, f, {value: deflt, selectValueOnOpen: true});
  else f(prompt(shortText, deflt));
}

function confirmDialog(cm, text, shortText, fs) {
  // 如果支持确认对话框，则打开确认对话框，否则使用 confirm 函数
  if (cm.openConfirm) cm.openConfirm(text, fs);
  else if (confirm(shortText)) fs[0]();
}

function parseString(string) {
  // 解析字符串中的转义字符
  return string.replace(/\\([nrt\\])/g, function(match, ch) {
    if (ch == "n") return "\n"
    if (ch == "r") return "\r"
    if (ch == "t") return "\t"
    if (ch == "\\") return "\\"
    return match
  })
}

function parseQuery(query) {
  // 解析查询字符串，如果是正则表达式则转换为正则对象，否则解析转义字符
  var isRE = query.match(/^\/(.*)\/([a-z]*)$/);
  if (isRE) {
    try { query = new RegExp(isRE[1], isRE[2].indexOf("i") == -1 ? "" : "i"); }
    catch(e) {} // 如果不是正则表达式，则进行字符串搜索
  } else {
    query = parseString(query)
  }
  // 如果查询为空，则设置默认查询
  if (typeof query == "string" ? query == "" : query.test(""))
    query = /x^/;
  return query;
}

function startSearch(cm, state, query) {
  // 设置查询文本和查询对象
  state.queryText = query;
  state.query = parseQuery(query);
  // 移除旧的覆盖层，添加新的覆盖层
  cm.removeOverlay(state.overlay, queryCaseInsensitive(state.query));
  state.overlay = searchOverlay(state.query, queryCaseInsensitive(state.query));
  cm.addOverlay(state.overlay);
  // 如果支持在滚动条上显示匹配项，则进行相应操作
  if (cm.showMatchesOnScrollbar) {
    if (state.annotate) { state.annotate.clear(); state.annotate = null; }
    state.annotate = cm.showMatchesOnScrollbar(state.query, queryCaseInsensitive(state.query));
    }
  }

  // 执行搜索操作
  function doSearch(cm, rev, persistent, immediate) {
    // 获取搜索状态
    var state = getSearchState(cm);
    // 如果存在查询条件，则执行查找下一个匹配项
    if (state.query) return findNext(cm, rev);
    // 获取当前选中的文本，如果没有则使用上次的查询条件
    var q = cm.getSelection() || state.lastQuery;
    // 如果查询条件是正则表达式并且内容为"x^"，则置空
    if (q instanceof RegExp && q.source == "x^") q = null
    // 如果需要持久化对话框，并且编辑器支持对话框
    if (persistent && cm.openDialog) {
      var hiding = null
      // 定义搜索下一个匹配项的函数
      var searchNext = function(query, event) {
        CodeMirror.e_stop(event);
        if (!query) return;
        if (query != state.queryText) {
          startSearch(cm, state, query);
          state.posFrom = state.posTo = cm.getCursor();
        }
        if (hiding) hiding.style.opacity = 1
        // 查找下一个匹配项
        findNext(cm, event.shiftKey, function(_, to) {
          var dialog
          // 如果匹配项在前三行，并且编辑器支持查询对话框
          if (to.line < 3 && document.querySelector &&
              (dialog = cm.display.wrapper.querySelector(".CodeMirror-dialog")) &&
              dialog.getBoundingClientRect().bottom - 4 > cm.cursorCoords(to, "window").top)
            (hiding = dialog).style.opacity = .4
        })
      };
      // 持久化对话框
      persistentDialog(cm, getQueryDialog(cm), q, searchNext, function(event, query) {
        var keyName = CodeMirror.keyName(event)
        var extra = cm.getOption('extraKeys'), cmd = (extra && extra[keyName]) || CodeMirror.keyMap[cm.getOption("keyMap")][keyName]
        if (cmd == "findNext" || cmd == "findPrev" ||
          cmd == "findPersistentNext" || cmd == "findPersistentPrev") {
          CodeMirror.e_stop(event);
          startSearch(cm, getSearchState(cm), query);
          cm.execCommand(cmd);
        } else if (cmd == "find" || cmd == "findPersistent") {
          CodeMirror.e_stop(event);
          searchNext(query, event);
        }
      });
      // 如果需要立即搜索并且存在查询条件，则开始搜索并查找下一个匹配项
      if (immediate && q) {
        startSearch(cm, state, q);
        findNext(cm, rev);
      }
  } else {
    # 如果条件不成立，弹出对话框，获取查询内容
    dialog(cm, getQueryDialog(cm), "Search for:", q, function(query) {
      # 如果查询内容存在且状态中的查询内容为空，则执行操作
      if (query && !state.query) cm.operation(function() {
        # 开始搜索
        startSearch(cm, state, query);
        # 设置搜索起始和结束位置为当前光标位置
        state.posFrom = state.posTo = cm.getCursor();
        # 查找下一个匹配项
        findNext(cm, rev);
      });
    });
  }
}

# 查找下一个匹配项
function findNext(cm, rev, callback) {cm.operation(function() {
  # 获取搜索状态
  var state = getSearchState(cm);
  # 获取搜索光标
  var cursor = getSearchCursor(cm, state.query, rev ? state.posFrom : state.posTo);
  # 如果没有找到匹配项
  if (!cursor.find(rev)) {
    # 设置搜索光标为文档末尾或开头
    cursor = getSearchCursor(cm, state.query, rev ? CodeMirror.Pos(cm.lastLine()) : CodeMirror.Pos(cm.firstLine(), 0));
    # 如果还是没有找到匹配项，则返回
    if (!cursor.find(rev)) return;
  }
  # 设置编辑器选中匹配项
  cm.setSelection(cursor.from(), cursor.to());
  # 滚动编辑器视图到匹配项位置
  cm.scrollIntoView({from: cursor.from(), to: cursor.to()}, 20);
  # 设置状态中的搜索起始和结束位置
  state.posFrom = cursor.from(); state.posTo = cursor.to();
  # 如果有回调函数，则执行回调
  if (callback) callback(cursor.from(), cursor.to())
});}

# 清除搜索状态
function clearSearch(cm) {cm.operation(function() {
  # 获取搜索状态
  var state = getSearchState(cm);
  # 将上次的查询内容保存到状态中
  state.lastQuery = state.query;
  # 如果查询内容为空，则返回
  if (!state.query) return;
  # 清空查询内容和查询文本
  state.query = state.queryText = null;
  # 移除搜索效果
  cm.removeOverlay(state.overlay);
  # 如果有标注，则清除标注
  if (state.annotate) { state.annotate.clear(); state.annotate = null; }
});}

# 获取查询对话框内容
function getQueryDialog(cm)  {
  return '<span class="CodeMirror-search-label">' + cm.phrase("Search:") + '</span> <input type="text" style="width: 10em" class="CodeMirror-search-field"/> <span style="color: #888" class="CodeMirror-search-hint">' + cm.phrase("(Use /re/ syntax for regexp search)") + '</span>';
}
# 获取替换查询对话框内容
function getReplaceQueryDialog(cm) {
  return ' <input type="text" style="width: 10em" class="CodeMirror-search-field"/> <span style="color: #888" class="CodeMirror-search-hint">' + cm.phrase("(Use /re/ syntax for regexp search)") + '</span>';
}
# 获取替换内容对话框内容
function getReplacementQueryDialog(cm) {
  # 返回一个包含搜索标签和输入框的字符串，用于搜索功能
  function getSearchLabel(cm) {
    return '<span class="CodeMirror-search-label">' + cm.phrase("With:") + '</span> <input type="text" style="width: 10em" class="CodeMirror-search-field"/>';
  }
  # 返回一个包含确认替换按钮的字符串，用于确认替换操作
  function getDoReplaceConfirm(cm) {
    return '<span class="CodeMirror-search-label">' + cm.phrase("Replace?") + '</span> <button>' + cm.phrase("Yes") + '</button> <button>' + cm.phrase("No") + '</button> <button>' + cm.phrase("All") + '</button> <button>' + cm.phrase("Stop") + '</button> ';
  }

  # 替换所有匹配的文本
  function replaceAll(cm, query, text) {
    # 执行替换操作
    cm.operation(function() {
      # 遍历所有匹配的位置
      for (var cursor = getSearchCursor(cm, query); cursor.findNext();) {
        # 如果查询条件不是字符串，则进行特殊处理
        if (typeof query != "string") {
          # 获取匹配的文本
          var match = cm.getRange(cursor.from(), cursor.to()).match(query);
          # 替换文本
          cursor.replace(text.replace(/\$(\d)/g, function(_, i) {return match[i];}));
        } else 
          # 直接替换文本
          cursor.replace(text);
      }
    });
  }

  # 执行替换操作
  function replace(cm, all) {
    # 如果编辑器是只读模式，则直接返回
    if (cm.getOption("readOnly")) return;
    # 获取选中的文本或上次查询的文本
    var query = cm.getSelection() || getSearchState(cm).lastQuery;
    # 构建对话框文本
    var dialogText = '<span class="CodeMirror-search-label">' + (all ? cm.phrase("Replace all:") : cm.phrase("Replace:")) + '</span>';
    // 弹出对话框，显示替换查询的文本，并等待用户输入
    dialog(cm, dialogText + getReplaceQueryDialog(cm), dialogText, query, function(query) {
      // 如果用户没有输入查询内容，则直接返回
      if (!query) return;
      // 解析用户输入的查询内容
      query = parseQuery(query);
      // 弹出对话框，显示替换文本的输入框，并等待用户输入
      dialog(cm, getReplacementQueryDialog(cm), cm.phrase("Replace with:"), "", function(text) {
        // 解析用户输入的替换文本
        text = parseString(text)
        // 如果需要替换所有匹配项
        if (all) {
          // 调用替换所有匹配项的函数
          replaceAll(cm, query, text)
        } else {
          // 清除搜索高亮
          clearSearch(cm);
          // 获取搜索光标对象
          var cursor = getSearchCursor(cm, query, cm.getCursor("from"));
          // 定义光标移动的函数
          var advance = function() {
            // 获取当前匹配项的起始位置
            var start = cursor.from(), match;
            // 如果没有找到下一个匹配项
            if (!(match = cursor.findNext())) {
              // 重置搜索光标对象
              cursor = getSearchCursor(cm, query);
              // 如果还是没有找到下一个匹配项，或者匹配项的位置没有改变，则返回
              if (!(match = cursor.findNext()) ||
                  (start && cursor.from().line == start.line && cursor.from().ch == start.ch)) return;
            }
            // 选中匹配项
            cm.setSelection(cursor.from(), cursor.to());
            // 滚动到匹配项的位置
            cm.scrollIntoView({from: cursor.from(), to: cursor.to()});
            // 弹出确认对话框，询问用户是否替换当前匹配项
            confirmDialog(cm, getDoReplaceConfirm(cm), cm.phrase("Replace?"),
                          [function() {doReplace(match);}, advance,
                           function() {replaceAll(cm, query, text)}]);
          };
          // 定义替换匹配项的函数
          var doReplace = function(match) {
            // 替换匹配项
            cursor.replace(typeof query == "string" ? text :
                           text.replace(/\$(\d)/g, function(_, i) {return match[i];}));
            // 移动光标到下一个匹配项
            advance();
          };
          // 调用光标移动函数，开始替换操作
          advance();
        }
      });
  // 定义一个匿名函数，该函数用于清除搜索结果
  });
  
  // 定义一个命令，用于触发搜索功能
  CodeMirror.commands.find = function(cm) {clearSearch(cm); doSearch(cm);};
  
  // 定义一个命令，用于触发持续搜索功能
  CodeMirror.commands.findPersistent = function(cm) {clearSearch(cm); doSearch(cm, false, true);};
  
  // 定义一个命令，用于触发持续搜索下一个功能
  CodeMirror.commands.findPersistentNext = function(cm) {doSearch(cm, false, true, true);};
  
  // 定义一个命令，用于触发持续搜索上一个功能
  CodeMirror.commands.findPersistentPrev = function(cm) {doSearch(cm, true, true, true);};
  
  // 定义一个命令，用于触发搜索下一个功能
  CodeMirror.commands.findNext = doSearch;
  
  // 定义一个命令，用于触发搜索上一个功能
  CodeMirror.commands.findPrev = function(cm) {doSearch(cm, true);};
  
  // 定义一个命令，用于触发清除搜索结果功能
  CodeMirror.commands.clearSearch = clearSearch;
  
  // 定义一个命令，用于触发替换功能
  CodeMirror.commands.replace = replace;
  
  // 定义一个命令，用于触发替换所有功能
  CodeMirror.commands.replaceAll = function(cm) {replace(cm, true);};
// 导出模块
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror"))
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror"], mod)
  // 如果是普通浏览器环境
  else
    mod(CodeMirror)
})(function(CodeMirror) {
  "use strict"
  // 定义 Pos 变量
  var Pos = CodeMirror.Pos

  // 获取正则表达式的标志
  function regexpFlags(regexp) {
    var flags = regexp.flags
    return flags != null ? flags : (regexp.ignoreCase ? "i" : "")
      + (regexp.global ? "g" : "")
      + (regexp.multiline ? "m" : "")
  }

  // 确保正则表达式具有指定的标志
  function ensureFlags(regexp, flags) {
    var current = regexpFlags(regexp), target = current
    for (var i = 0; i < flags.length; i++) if (target.indexOf(flags.charAt(i)) == -1)
      target += flags.charAt(i)
    return current == target ? regexp : new RegExp(regexp.source, target)
  }

  // 检查正则表达式是否可能包含多行匹配
  function maybeMultiline(regexp) {
    return /\\s|\\n|\n|\\W|\\D|\[\^/.test(regexp.source)
  }

  // 向前搜索匹配正则表达式的内容
  function searchRegexpForward(doc, regexp, start) {
    regexp = ensureFlags(regexp, "g")
    for (var line = start.line, ch = start.ch, last = doc.lastLine(); line <= last; line++, ch = 0) {
      regexp.lastIndex = ch
      var string = doc.getLine(line), match = regexp.exec(string)
      if (match)
        return {from: Pos(line, match.index),
                to: Pos(line, match.index + match[0].length),
                match: match}
    }
  }

  // 向前搜索匹配多行的正则表达式的内容
  function searchRegexpForwardMultiline(doc, regexp, start) {
    if (!maybeMultiline(regexp)) return searchRegexpForward(doc, regexp, start)

    regexp = ensureFlags(regexp, "gm")
    var string, chunk = 1
    // 从起始行开始循环直到文档的最后一行
    for (var line = start.line, last = doc.lastLine(); line <= last;) {
      // 这里以指数大小的块来扩展搜索缓冲区，以便在匹配之间快速查找附近的匹配，而不需要连接整个文档（以防我们正在搜索具有大量匹配的内容），但同时，重试的次数是有限的。
      for (var i = 0; i < chunk; i++) {
        if (line > last) break
        var curLine = doc.getLine(line++)
        string = string == null ? curLine : string + "\n" + curLine
      }
      chunk = chunk * 2
      regexp.lastIndex = start.ch
      var match = regexp.exec(string)
      if (match) {
        var before = string.slice(0, match.index).split("\n"), inside = match[0].split("\n")
        var startLine = start.line + before.length - 1, startCh = before[before.length - 1].length
        return {from: Pos(startLine, startCh),
                to: Pos(startLine + inside.length - 1,
                        inside.length == 1 ? startCh + inside[0].length : inside[inside.length - 1].length),
                match: match}
      }
    }
  }

  // 在给定字符串中查找最后一个匹配项
  function lastMatchIn(string, regexp, endMargin) {
    var match, from = 0
    while (from <= string.length) {
      regexp.lastIndex = from
      var newMatch = regexp.exec(string)
      if (!newMatch) break
      var end = newMatch.index + newMatch[0].length
      if (end > string.length - endMargin) break
      if (!match || end > match.index + match[0].length)
        match = newMatch
      from = newMatch.index + 1
    }
    return match
  }

  // 向后搜索匹配给定正则表达式的内容
  function searchRegexpBackward(doc, regexp, start) {
    regexp = ensureFlags(regexp, "g")
  // 从给定位置开始向上搜索匹配指定正则表达式的多行文本
  function searchRegexpBackwardMultiline(doc, regexp, start) {
    // 如果正则表达式不包含多行匹配，则调用searchRegexpBackward函数进行搜索
    if (!maybeMultiline(regexp)) return searchRegexpBackward(doc, regexp, start)
    // 确保正则表达式包含全局匹配和多行匹配标志
    regexp = ensureFlags(regexp, "gm")
    // 初始化变量
    var string, chunkSize = 1, endMargin = doc.getLine(start.line).length - start.ch
    // 从给定位置开始向上搜索匹配多行文本
    for (var line = start.line, first = doc.firstLine(); line >= first;) {
      // 逐步增加chunkSize，合并多行文本
      for (var i = 0; i < chunkSize && line >= first; i++) {
        var curLine = doc.getLine(line--)
        string = string == null ? curLine : curLine + "\n" + string
      }
      chunkSize *= 2
      // 在合并的多行文本中搜索匹配的内容
      var match = lastMatchIn(string, regexp, endMargin)
      // 如果找到匹配的内容，则计算匹配内容的起始位置和结束位置
      if (match) {
        var before = string.slice(0, match.index).split("\n"), inside = match[0].split("\n")
        var startLine = line + before.length, startCh = before[before.length - 1].length
        return {from: Pos(startLine, startCh),
                to: Pos(startLine + inside.length - 1,
                        inside.length == 1 ? startCh + inside[0].length : inside[inside.length - 1].length),
                match: match}
      }
    }
  }

  // 初始化变量
  var doFold, noFold
  // 如果String对象包含normalize方法，则定义doFold和noFold函数
  if (String.prototype.normalize) {
    doFold = function(str) { return str.normalize("NFD").toLowerCase() }
    noFold = function(str) { return str.normalize("NFD") }
  } else {
    // 否则定义doFold和noFold函数
    doFold = function(str) { return str.toLowerCase() }
    noFold = function(str) { return str }
  }

  // 将折叠后的行中的位置映射回原始行中的位置
  // （补偿折叠过程中代码点数量的增加）
  function adjustPos(orig, folded, pos, foldFunc) {
    // 如果原始字符串长度等于折叠后字符串长度，则返回当前位置
    if (orig.length == folded.length) return pos
    // 计算最大搜索范围
    for (var min = 0, max = pos + Math.max(0, orig.length - folded.length);;) {
      // 如果最小值等于最大值，则返回最小值
      if (min == max) return min
      // 计算中间位置
      var mid = (min + max) >> 1
      // 计算折叠后的子字符串长度
      var len = foldFunc(orig.slice(0, mid)).length
      // 如果长度等于目标位置，则返回中间位置
      if (len == pos) return mid
      // 如果长度大于目标位置，则将最大值设为中间位置
      else if (len > pos) max = mid
      // 如果长度小于目标位置，则将最小值设为中间位置加一
      else min = mid + 1
    }
  }

  // 向前搜索字符串
  function searchStringForward(doc, query, start, caseFold) {
    // 如果查询字符串为空，则返回空
    if (!query.length) return null
    // 根据是否忽略大小写选择折叠函数
    var fold = caseFold ? doFold : noFold
    // 将查询字符串折叠并按换行符分割成数组
    var lines = fold(query).split(/\r|\n\r?/)

    // 遍历搜索
    search: for (var line = start.line, ch = start.ch, last = doc.lastLine() + 1 - lines.length; line <= last; line++, ch = 0) {
      var orig = doc.getLine(line).slice(ch), string = fold(orig)
      // 如果查询字符串只有一行
      if (lines.length == 1) {
        var found = string.indexOf(lines[0])
        // 如果找不到匹配的字符串，则继续搜索
        if (found == -1) continue search
        // 计算匹配字符串的起始位置和结束位置
        var start = adjustPos(orig, string, found, fold) + ch
        return {from: Pos(line, adjustPos(orig, string, found, fold) + ch),
                to: Pos(line, adjustPos(orig, string, found + lines[0].length, fold) + ch)}
      } else {
        var cutFrom = string.length - lines[0].length
        // 如果当前行末尾不是查询字符串的起始部分，则继续搜索
        if (string.slice(cutFrom) != lines[0]) continue search
        // 遍历查询字符串的每一行
        for (var i = 1; i < lines.length - 1; i++)
          // 如果折叠后的当前行与查询字符串不匹配，则继续搜索
          if (fold(doc.getLine(line + i)) != lines[i]) continue search
        var end = doc.getLine(line + lines.length - 1), endString = fold(end), lastLine = lines[lines.length - 1]
        // 如果当前行末尾不是查询字符串的起始部分，则继续���索
        if (endString.slice(0, lastLine.length) != lastLine) continue search
        // 计算匹配字符串的起始位置和结束位置
        return {from: Pos(line, adjustPos(orig, string, cutFrom, fold) + ch),
                to: Pos(line + lines.length - 1, adjustPos(end, endString, lastLine.length, fold))}
      }
    }
  }

  // 向后搜索字符串
  function searchStringBackward(doc, query, start, caseFold) {
    // 如果查询字符串为空，则返回空
    if (!query.length) return null
    // 根据是否忽略大小写选择折叠函数
    var fold = caseFold ? doFold : noFold
    # 将查询结果按行分割成数组
    var lines = fold(query).split(/\r|\n\r?/)

    # 从指定位置开始向上搜索匹配的文本
    search: for (var line = start.line, ch = start.ch, first = doc.firstLine() - 1 + lines.length; line >= first; line--, ch = -1) {
      # 获取当前行的原始文本
      var orig = doc.getLine(line)
      # 如果指定了列位置，则截取原始文本到指定列位置
      if (ch > -1) orig = orig.slice(0, ch)
      # 折叠原始文本
      var string = fold(orig)
      # 如果只有一行查询文本
      if (lines.length == 1) {
        # 在折叠后的原始文本中查找查询文本的最后出现位置
        var found = string.lastIndexOf(lines[0])
        # 如果未找到，则继续搜索下一行
        if (found == -1) continue search
        # 返回匹配结果的起始位置和结束位置
        return {from: Pos(line, adjustPos(orig, string, found, fold)),
                to: Pos(line, adjustPos(orig, string, found + lines[0].length, fold))}
      } else {
        # 获取查询文本的最后一行
        var lastLine = lines[lines.length - 1]
        # 如果折叠后的原始文本与查询文本的最后一行不匹配，则继续搜索下一行
        if (string.slice(0, lastLine.length) != lastLine) continue search
        # 遍历查询文本的每一行，与折叠后的原始文本进行比较
        for (var i = 1, start = line - lines.length + 1; i < lines.length - 1; i++)
          if (fold(doc.getLine(start + i)) != lines[i]) continue search
        # 获取当前行的上一行文本，并折叠
        var top = doc.getLine(line + 1 - lines.length), topString = fold(top)
        # 如果上一行文本与查询文本的第一行不匹配，则继续搜索下一行
        if (topString.slice(topString.length - lines[0].length) != lines[0]) continue search
        # 返回匹配结果的起始位置和结束位置
        return {from: Pos(line + 1 - lines.length, adjustPos(top, topString, top.length - lines[0].length, fold)),
                to: Pos(line, adjustPos(orig, string, lastLine.length, fold))}
      }
    }
  }

  # 定义搜索游标对象的构造函数
  function SearchCursor(doc, query, pos, options) {
    # 初始化搜索游标对象的属性
    this.atOccurrence = false
    this.doc = doc
    # 对位置进行裁剪，确保在文档范围内
    pos = pos ? doc.clipPos(pos) : Pos(0, 0)
    this.pos = {from: pos, to: pos}

    var caseFold
    # 检查是否指定了大小写折叠选项
    if (typeof options == "object") {
      caseFold = options.caseFold
    } else { # 为了向后兼容，当caseFold是第四个参数时
      caseFold = options
      options = null
    }

    # 如果查询是字符串类型
    if (typeof query == "string") {
      # 如果未指定大小写折叠选项，则默认为false
      if (caseFold == null) caseFold = false
      # 定义匹配函数，根据指定的方向和位置搜索匹配的文本
      this.matches = function(reverse, pos) {
        return (reverse ? searchStringBackward : searchStringForward)(doc, query, pos, caseFold)
      }
  } else {
    # 确保查询标志包含 "gm"
    query = ensureFlags(query, "gm")
    # 如果选项不存在或者选项中的多行标志不是 false，则定义匹配函数为多行匹配
    if (!options || options.multiline !== false)
      this.matches = function(reverse, pos) {
        return (reverse ? searchRegexpBackwardMultiline : searchRegexpForwardMultiline)(doc, query, pos)
      }
    # 否则定义匹配函数为非多行匹配
    else
      this.matches = function(reverse, pos) {
        return (reverse ? searchRegexpBackward : searchRegexpForward)(doc, query, pos)
      }
  }
}

# 定义 SearchCursor 对象的原型
SearchCursor.prototype = {
  # 查找下一个匹配项
  findNext: function() {return this.find(false)},
  # 查找上一个匹配项
  findPrevious: function() {return this.find(true)},

  # 查找匹配项的函数
  find: function(reverse) {
    # 获取匹配结果
    var result = this.matches(reverse, this.doc.clipPos(reverse ? this.pos.from : this.pos.to))

    # 处理空匹配的自动增长行为，以保持与 vim 代码的向后兼容
    while (result && CodeMirror.cmpPos(result.from, result.to) == 0) {
      if (reverse) {
        if (result.from.ch) result.from = Pos(result.from.line, result.from.ch - 1)
        else if (result.from.line == this.doc.firstLine()) result = null
        else result = this.matches(reverse, this.doc.clipPos(Pos(result.from.line - 1)))
      } else {
        if (result.to.ch < this.doc.getLine(result.to.line).length) result.to = Pos(result.to.line, result.to.ch + 1)
        else if (result.to.line == this.doc.lastLine()) result = null
        else result = this.matches(reverse, Pos(result.to.line + 1, 0))
      }
    }

    # 如果存在匹配结果，则更新位置并返回匹配结果
    if (result) {
      this.pos = result
      this.atOccurrence = true
      return this.pos.match || true
    } else {
      # 如果不存在匹配结果，则返回 false
      var end = Pos(reverse ? this.doc.firstLine() : this.doc.lastLine() + 1, 0)
      this.pos = {from: end, to: end}
      return this.atOccurrence = false
    }
  },

  # 获取匹配项的起始位置
  from: function() {if (this.atOccurrence) return this.pos.from},
  # 获取匹配项的结束位置
  to: function() {if (this.atOccurrence) return this.pos.to},
    // 定义一个名为 replace 的方法，用于替换文本
    replace: function(newText, origin) {
      // 如果没有指定替换位置，则直接返回
      if (!this.atOccurrence) return
      // 将新文本按行分割
      var lines = CodeMirror.splitLines(newText)
      // 用新文本替换指定位置的文本
      this.doc.replaceRange(lines, this.pos.from, this.pos.to, origin)
      // 更新替换后的位置
      this.pos.to = Pos(this.pos.from.line + lines.length - 1,
                        lines[lines.length - 1].length + (lines.length == 1 ? this.pos.from.ch : 0))
    }
  }

  // 定义一个名为 getSearchCursor 的方法，用于获取搜索光标
  CodeMirror.defineExtension("getSearchCursor", function(query, pos, caseFold) {
    return new SearchCursor(this.doc, query, pos, caseFold)
  })
  // 定义一个名为 getSearchCursor 的文档扩展方法，用于获取搜索光标
  CodeMirror.defineDocExtension("getSearchCursor", function(query, pos, caseFold) {
    return new SearchCursor(this, query, pos, caseFold)
  })

  // 定义一个名为 selectMatches 的方法，用于选择匹配项
  CodeMirror.defineExtension("selectMatches", function(query, caseFold) {
    var ranges = []
    // 获取搜索光标
    var cur = this.getSearchCursor(query, this.getCursor("from"), caseFold)
    // 循环查找匹配项
    while (cur.findNext()) {
      if (CodeMirror.cmpPos(cur.to(), this.getCursor("to")) > 0) break
      ranges.push({anchor: cur.from(), head: cur.to()})
    }
    // 如果存在匹配项，则设置选择范围
    if (ranges.length)
      this.setSelections(ranges, 0)
  })
// 定义了一个自执行函数，传入一个 mod 参数
(function(mod) {
  // 如果是 CommonJS 环境，使用 mod(require("../../lib/codemirror")) 引入模块
  if (typeof exports == "object" && typeof module == "object") 
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境，使用 define(["../../lib/codemirror"], mod) 引入模块
  else if (typeof define == "function" && define.amd) 
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境，直接使用 CodeMirror 对象
  else 
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";
  // 定义一些常量
  var WRAP_CLASS = "CodeMirror-activeline";
  var BACK_CLASS = "CodeMirror-activeline-background";
  var GUTT_CLASS = "CodeMirror-activeline-gutter";

  // 定义 CodeMirror 的 styleActiveLine 选项
  CodeMirror.defineOption("styleActiveLine", false, function(cm, val, old) {
    var prev = old == CodeMirror.Init ? false : old;
    // 如果新旧值相同，直接返回
    if (val == prev) return
    // 如果旧值存在，移除事件监听和清除活动行样式
    if (prev) {
      cm.off("beforeSelectionChange", selectionChange);
      clearActiveLines(cm);
      delete cm.state.activeLines;
    }
    // 如果新值存在，添加事件监听和更新活动行样式
    if (val) {
      cm.state.activeLines = [];
      updateActiveLines(cm, cm.listSelections());
      cm.on("beforeSelectionChange", selectionChange);
    }
  });

  // 清除活动行样式
  function clearActiveLines(cm) {
    for (var i = 0; i < cm.state.activeLines.length; i++) {
      cm.removeLineClass(cm.state.activeLines[i], "wrap", WRAP_CLASS);
      cm.removeLineClass(cm.state.activeLines[i], "background", BACK_CLASS);
      cm.removeLineClass(cm.state.activeLines[i], "gutter", GUTT_CLASS);
    }
  }

  // 判断两个数组是否相同
  function sameArray(a, b) {
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; i++)
      if (a[i] != b[i]) return false;
    return true;
  }

  // 更新活动行样式
  function updateActiveLines(cm, ranges) {
    var active = [];
    # 遍历ranges数组
    for (var i = 0; i < ranges.length; i++) {
      # 获取当前range
      var range = ranges[i];
      # 获取styleActiveLine选项
      var option = cm.getOption("styleActiveLine");
      # 判断styleActiveLine选项是否为对象且非空，如果是则判断range的起始行和结束行是否相同，如果不是则判断range是否为空
      if (typeof option == "object" && option.nonEmpty ? range.anchor.line != range.head.line : !range.empty())
        # 如果条件成立则跳过本次循环
        continue
      # 获取range头部行的可视起始行
      var line = cm.getLineHandleVisualStart(range.head.line);
      # 如果active数组中最后一个元素不等于line，则将line添加到active数组中
      if (active[active.length - 1] != line) active.push(line);
    }
    # 判断cm.state.activeLines和active数组是否相同，如果相同则返回
    if (sameArray(cm.state.activeLines, active)) return;
    # 执行操作
    cm.operation(function() {
      # 清除activeLines
      clearActiveLines(cm);
      # 遍历active数组
      for (var i = 0; i < active.length; i++) {
        # 为active数组中的每个元素添加wrap类
        cm.addLineClass(active[i], "wrap", WRAP_CLASS);
        # 为active数组中的每个元素添加background类
        cm.addLineClass(active[i], "background", BACK_CLASS);
        # 为active数组中的每个元素添加gutter类
        cm.addLineClass(active[i], "gutter", GUTT_CLASS);
      }
      # 将active数组赋值给cm.state.activeLines
      cm.state.activeLines = active;
    });
  }

  # 当选择发生变化时调用
  function selectionChange(cm, sel) {
    # 更新activeLines
    updateActiveLines(cm, sel.ranges);
  }
// 导入模块
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror"], mod);
  // 如果是普通浏览器环境
  else
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义 CodeMirror 的选项 'styleSelectedText'
  CodeMirror.defineOption("styleSelectedText", false, function(cm, val, old) {
    var prev = old && old != CodeMirror.Init;
    // 如果新值为真且旧值不为真
    if (val && !prev) {
      // 初始化标记选择的文本数组和样式
      cm.state.markedSelection = [];
      cm.state.markedSelectionStyle = typeof val == "string" ? val : "CodeMirror-selectedtext";
      // 重置
      reset(cm);
      // 监听光标活动事件
      cm.on("cursorActivity", onCursorActivity);
      // 监听文本改变事件
      cm.on("change", onChange);
    } 
    // 如果新值为假且旧值为真
    else if (!val && prev) {
      // 取消监听光标活动事件和文本改变事件
      cm.off("cursorActivity", onCursorActivity);
      cm.off("change", onChange);
      // 清除标记
      clear(cm);
      cm.state.markedSelection = cm.state.markedSelectionStyle = null;
    }
  });

  // 光标活动事件处理函数
  function onCursorActivity(cm) {
    if (cm.state.markedSelection)
      cm.operation(function() { update(cm); });
  }

  // 文本改变事件处理函数
  function onChange(cm) {
    if (cm.state.markedSelection && cm.state.markedSelection.length)
      cm.operation(function() { clear(cm); });
  }

  // 定义常量和变量
  var CHUNK_SIZE = 8;
  var Pos = CodeMirror.Pos;
  var cmp = CodeMirror.cmpPos;

  // 覆盖范围函数
  function coverRange(cm, from, to, addAt) {
    if (cmp(from, to) == 0) return;
    var array = cm.state.markedSelection;
    var cls = cm.state.markedSelectionStyle;
    # 从起始行开始循环处理
    for (var line = from.line;;) {
      # 如果当前行等于起始行，则将起始位置设为起始点，否则将起始位置设为当前行的第一个字符
      var start = line == from.line ? from : Pos(line, 0);
      # 计算结束行的位置，判断是否已经到达目标行
      var endLine = line + CHUNK_SIZE, atEnd = endLine >= to.line;
      # 如果已经到达目标行，则将结束位置设为目标点，否则将结束位置设为结束行的第一个字符
      var end = atEnd ? to : Pos(endLine, 0);
      # 在编辑器中标记起始位置到结束位置的文本，并设置样式
      var mark = cm.markText(start, end, {className: cls});
      # 如果没有指定插入位置，则将标记添加到数组末尾，否则在指定位置插入标记
      if (addAt == null) array.push(mark);
      else array.splice(addAt++, 0, mark);
      # 如果已经到达目标行，则跳出循环
      if (atEnd) break;
      # 更新当前行为结束行
      line = endLine;
    }
  }

  # 清除编辑器中的标记
  function clear(cm) {
    # 获取编辑器中的标记数组
    var array = cm.state.markedSelection;
    # 循环清除所有标记
    for (var i = 0; i < array.length; ++i) array[i].clear();
    # 清空标记数组
    array.length = 0;
  }

  # 重置编辑器中的标记
  function reset(cm) {
    # 清除编辑器中的标记
    clear(cm);
    # 获取编辑器中的选择范围数组
    var ranges = cm.listSelections();
    # 遍历选择范围数组，重新标记文本
    for (var i = 0; i < ranges.length; i++)
      coverRange(cm, ranges[i].from(), ranges[i].to());
  }

  # 更新编辑器中的标记
  function update(cm) {
    # 如果没有选中文本，则清除编辑器中的标记
    if (!cm.somethingSelected()) return clear(cm);
    # 如果选择范围数量大于1，则重置编辑器中的标记
    if (cm.listSelections().length > 1) return reset(cm);

    # 获取选中文本的起始位置和结束位置
    var from = cm.getCursor("start"), to = cm.getCursor("end");

    # 获取编辑器中的标记数组
    var array = cm.state.markedSelection;
    # 如果标记数组为空，则标记选中文本
    if (!array.length) return coverRange(cm, from, to);

    # 获取第一个标记的起始位置和最后一个标记的结束位置
    var coverStart = array[0].find(), coverEnd = array[array.length - 1].find();
    # 如果起始位置或结束位置为空，或者选中文本行数小于等于CHUNK_SIZE，或者选中文本超出已有标记范围，则重置编辑器中的标记
    if (!coverStart || !coverEnd || to.line - from.line <= CHUNK_SIZE ||
        cmp(from, coverEnd.to) >= 0 || cmp(to, coverStart.from) <= 0)
      return reset(cm);

    # 循环移除超出选中文本范围的标记
    while (cmp(from, coverStart.from) > 0) {
      array.shift().clear();
      coverStart = array[0].find();
    }
    # 如果起始位置小于第一个标记的起始位置，则根据条件重新标记文本
    if (cmp(from, coverStart.from) < 0) {
      if (coverStart.to.line - from.line < CHUNK_SIZE) {
        array.shift().clear();
        coverRange(cm, from, coverStart.to, 0);
      } else {
        coverRange(cm, from, coverStart.from, 0);
      }
    }

    # 循环移除超出选中文本范围的标记
    while (cmp(to, coverEnd.to) < 0) {
      array.pop().clear();
      coverEnd = array[array.length - 1].find();
    }
    # 如果 to 大于 coverEnd.to
    if (cmp(to, coverEnd.to) > 0) {
      # 如果 to 的行数减去 coverEnd.from 的行数小于 CHUNK_SIZE
      if (to.line - coverEnd.from.line < CHUNK_SIZE) {
        # 弹出数组最后一个元素并清空
        array.pop().clear();
        # 覆盖范围从 coverEnd.from 到 to
        coverRange(cm, coverEnd.from, to);
      } else {
        # 覆盖范围从 coverEnd.to 到 to
        coverRange(cm, coverEnd.to, to);
      }
    }
  }
// 定义一个匿名函数，传入 mod 参数
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror")); // 使用 mod 函数引入 codemirror 模块
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror"], mod); // 使用 define 函数引入 codemirror 模块
  // 如果是普通的浏览器环境
  else
    mod(CodeMirror); // 使用 mod 函数引入 CodeMirror 对象
})(function(CodeMirror) {
  "use strict"; // 开启严格模式

  // 定义 CodeMirror 的 selectionPointer 选项
  CodeMirror.defineOption("selectionPointer", false, function(cm, val) {
    var data = cm.state.selectionPointer; // 获取当前的 selectionPointer 数据
    // 如果已经存在 selectionPointer 数据
    if (data) {
      // 移除鼠标移动、鼠标移出、窗口滚动、光标活动、滚动事件的监听
      CodeMirror.off(cm.getWrapperElement(), "mousemove", data.mousemove);
      CodeMirror.off(cm.getWrapperElement(), "mouseout", data.mouseout);
      CodeMirror.off(window, "scroll", data.windowScroll);
      cm.off("cursorActivity", reset);
      cm.off("scroll", reset);
      cm.state.selectionPointer = null; // 清空 selectionPointer 数据
      cm.display.lineDiv.style.cursor = ""; // 恢复默认的鼠标样式
    }
    // 如果传入了新的 selectionPointer 数据
    if (val) {
      // 初始化 selectionPointer 数据
      data = cm.state.selectionPointer = {
        value: typeof val == "string" ? val : "default", // 设置 value 属性为传入的值或默认值
        mousemove: function(event) { mousemove(cm, event); }, // 设置 mousemove 函数
        mouseout: function(event) { mouseout(cm, event); }, // 设置 mouseout 函数
        windowScroll: function() { reset(cm); }, // 设置 windowScroll 函数
        rects: null, // 初始化 rects 属性为 null
        mouseX: null, // 初始化 mouseX 属性为 null
        mouseY: null, // 初始化 mouseY 属性为 null
        willUpdate: false // 初始化 willUpdate 属性为 false
      };
      // 添加鼠标移动、鼠标移出、窗口滚动的监听
      CodeMirror.on(cm.getWrapperElement(), "mousemove", data.mousemove);
      CodeMirror.on(cm.getWrapperElement(), "mouseout", data.mouseout);
      CodeMirror.on(window, "scroll", data.windowScroll);
      // 添加光标活动、滚动事件的监听
      cm.on("cursorActivity", reset);
      cm.on("scroll", reset);
    }
  });

  // 定义鼠标移动事件处理函数
  function mousemove(cm, event) {
    var data = cm.state.selectionPointer; // 获取当前的 selectionPointer 数据
    // 如果鼠标左键被按下
    if (event.buttons == null ? event.which : event.buttons) {
      data.mouseX = data.mouseY = null; // 将 mouseX 和 mouseY 属性设置为 null
    } else {
      data.mouseX = event.clientX; // 更新 mouseX 属性为鼠标的横坐标
      data.mouseY = event.clientY; // 更新 mouseY 属性为鼠标的纵坐标
    }
  // 调度更新操作，传入代码编辑器对象
  function scheduleUpdate(cm) {
    // 如果当前没有更新操作，则进行更新
    if (!cm.state.selectionPointer.willUpdate) {
      // 设置将要更新的标志为 true
      cm.state.selectionPointer.willUpdate = true;
      // 延迟 50 毫秒后执行更新操作
      setTimeout(function() {
        update(cm);
        // 更新完成后将将要更新的标志设置为 false
        cm.state.selectionPointer.willUpdate = false;
      }, 50);
    }
  }

  // 更新操作，传入代码编辑器对象
  function update(cm) {
    // 获取选择指针的数据
    var data = cm.state.selectionPointer;
    // 如果数据为空，则直接返回
    if (!data) return;
    // 如果矩形为空且鼠标 X 坐标不为空
    if (data.rects == null && data.mouseX != null) {
      // 初始化矩形数组
      data.rects = [];
      // 如果有选中内容，则遍历选中内容的矩形并添加到矩形数组中
      if (cm.somethingSelected()) {
        for (var sel = cm.display.selectionDiv.firstChild; sel; sel = sel.nextSibling)
          data.rects.push(sel.getBoundingClientRect());
      }
    }
    // 初始化内部标志为 false
    var inside = false;
    // 如果鼠标 X 坐标不为空，则遍历矩形数组
    if (data.mouseX != null) for (var i = 0; i < data.rects.length; i++) {
      var rect = data.rects[i];
      // 如果鼠标在矩形内部，则将内部标志设置为 true
      if (rect.left <= data.mouseX && rect.right >= data.mouseX &&
          rect.top <= data.mouseY && rect.bottom >= data.mouseY)
        inside = true;
    }
    // 根据内部标志设置光标样式
    var cursor = inside ? data.value : "";
    if (cm.display.lineDiv.style.cursor != cursor)
      cm.display.lineDiv.style.cursor = cursor;
  }
});

/* ---- mode/coffeescript.js ---- */

// CodeMirror, 版权归 Marijn Haverbeke 和其他人所有
// 根据 MIT 许可证分发：https://codemirror.net/LICENSE

/**
 * 项目的 GitHub 页面链接：
 * https://github.com/pickhardt/coffeescript-codemirror-mode
 */
(function(mod) {
  if (typeof exports == "object" && typeof module == "object") // CommonJS
    mod(require("../../lib/codemirror"));
  else if (typeof define == "function" && define.amd) // AMD
    define(["../../lib/codemirror"], mod);
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
"use strict";

CodeMirror.defineMode("coffeescript", function(conf, parserConf) {
  var ERRORCLASS = "error";

  function wordRegexp(words) {
    # 创建一个新的正则表达式对象，用于匹配单词边界
    return new RegExp("^((" + words.join(")|(") + "))\\b");
  }

  # 定义操作符的正则表达式模式
  var operators = /^(?:->|=>|\+[+=]?|-[\-=]?|\*[\*=]?|\/[\/=]?|[=!]=|<[><]?=?|>>?=?|%=?|&=?|\|=?|\^=?|\~|!|\?|(or|and|\|\||&&|\?)=)/;
  # 定义分隔符的正则表达式模式
  var delimiters = /^(?:[()\[\]{},:`=;]|\.\.?\.?)/;
  # 定义标识符的正则表达式模式
  var identifiers = /^[_A-Za-z$][_A-Za-z$0-9]*/;
  # 定义带有@符号的属性的正则表达式模式
  var atProp = /^@[_A-Za-z$][_A-Za-z$0-9]*/;

  # 定义包含特定单词的操作符的正则表达式模式
  var wordOperators = wordRegexp(["and", "or", "not",
                                  "is", "isnt", "in",
                                  "instanceof", "typeof"]);
  # 定义缩进关键字的数组
  var indentKeywords = ["for", "while", "loop", "if", "unless", "else",
                        "switch", "try", "catch", "finally", "class"];
  # 定义常见关键字的数组
  var commonKeywords = ["break", "by", "continue", "debugger", "delete",
                        "do", "in", "of", "new", "return", "then",
                        "this", "@", "throw", "when", "until", "extends"];

  # 定义关键字的正则表达式模式
  var keywords = wordRegexp(indentKeywords.concat(commonKeywords));

  # 重新定义缩进关键字的正则表达式模式
  indentKeywords = wordRegexp(indentKeywords);

  # 定义字符串前缀的正则表达式模式
  var stringPrefixes = /^('{3}|\"{3}|['\"])/;
  # 定义正则表达式前缀的正则表达式模式
  var regexPrefixes = /^(\/{3}|\/)/;
  # 定义常见常量的数组
  var commonConstants = ["Infinity", "NaN", "undefined", "null", "true", "false", "on", "off", "yes", "no"];
  # 定义常量的正则表达式模式
  var constants = wordRegexp(commonConstants);

  # Tokenizers
  # 定义 tokenBase 函数，处理流和状态
  function tokenBase(stream, state) {
    # 处理作用域变化
    if (stream.sol()) {
      if (state.scope.align === null) state.scope.align = false;
      var scopeOffset = state.scope.offset;
      if (stream.eatSpace()) {
        var lineOffset = stream.indentation();
        if (lineOffset > scopeOffset && state.scope.type == "coffee") {
          return "indent";
        } else if (lineOffset < scopeOffset) {
          return "dedent";
        }
        return null;
      } else {
        if (scopeOffset > 0) {
          dedent(stream, state);
        }
      }
    }
    if (stream.eatSpace()) {
      return null;
    }

    var ch = stream.peek();

    # 处理 docco 标题注释（单行）
    // 如果当前行以"####"开头，则将整行标记为注释
    if (stream.match("####")) {
      stream.skipToEnd();
      return "comment";
    }

    // 处理多行注释
    if (stream.match("###")) {
      state.tokenize = longComment;
      return state.tokenize(stream, state);
    }

    // 单行注释
    if (ch === "#") {
      stream.skipToEnd();
      return "comment";
    }

    // 处理数字字面量
    if (stream.match(/^-?[0-9\.]/, false)) {
      var floatLiteral = false;
      // 浮点数
      if (stream.match(/^-?\d*\.\d+(e[\+\-]?\d+)?/i)) {
        floatLiteral = true;
      }
      if (stream.match(/^-?\d+\.\d*/)) {
        floatLiteral = true;
      }
      if (stream.match(/^-?\.\d+/)) {
        floatLiteral = true;
      }

      if (floatLiteral) {
        // 防止出现额外的小数点，比如1..
        if (stream.peek() == "."){
          stream.backUp(1);
        }
        return "number";
      }
      // 整数
      var intLiteral = false;
      // 十六进制
      if (stream.match(/^-?0x[0-9a-f]+/i)) {
        intLiteral = true;
      }
      // 十进制
      if (stream.match(/^-?[1-9]\d*(e[\+\-]?\d+)?/)) {
        intLiteral = true;
      }
      // 单独的零，没有其他数字
      if (stream.match(/^-?0(?![\dx])/i)) {
        intLiteral = true;
      }
      if (intLiteral) {
        return "number";
      }
    }

    // 处理字符串
    if (stream.match(stringPrefixes)) {
      state.tokenize = tokenFactory(stream.current(), false, "string");
      return state.tokenize(stream, state);
    }
    // 处理正则表达式字面量
    if (stream.match(regexPrefixes)) {
      if (stream.current() != "/" || stream.match(/^.*\//, false)) { // 防止除法运算符被高亮显示
        state.tokenize = tokenFactory(stream.current(), true, "string-2");
        return state.tokenize(stream, state);
      } else {
        stream.backUp(1);
      }
    }

    // 处理运算符和分隔符
    # 如果流匹配操作符或者单词操作符，则返回"operator"
    if (stream.match(operators) || stream.match(wordOperators)) {
      return "operator";
    }
    # 如果流匹配分隔符，则返回"punctuation"
    if (stream.match(delimiters)) {
      return "punctuation";
    }

    # 如果流匹配常量，则返回"atom"
    if (stream.match(constants)) {
      return "atom";
    }

    # 如果流匹配@属性或者状态属性并且流匹配标识符，则返回"property"
    if (stream.match(atProp) || state.prop && stream.match(identifiers)) {
      return "property";
    }

    # 如果流匹配关键字，则返回"keyword"
    if (stream.match(keywords)) {
      return "keyword";
    }

    # 如果流匹配标识符，则返回"variable"
    if (stream.match(identifiers)) {
      return "variable";
    }

    # 处理未检测到的项
    stream.next();
    return ERRORCLASS;
  }

  # 创建标记工厂函数
  function tokenFactory(delimiter, singleline, outclass) {
    return function(stream, state) {
      while (!stream.eol()) {
        stream.eatWhile(/[^'"\/\\]/);
        if (stream.eat("\\")) {
          stream.next();
          if (singleline && stream.eol()) {
            return outclass;
          }
        } else if (stream.match(delimiter)) {
          state.tokenize = tokenBase;
          return outclass;
        } else {
          stream.eat(/['"\/]/);
        }
      }
      if (singleline) {
        if (parserConf.singleLineStringErrors) {
          outclass = ERRORCLASS;
        } else {
          state.tokenize = tokenBase;
        }
      }
      return outclass;
    };
  }

  # 处理长注释
  function longComment(stream, state) {
    while (!stream.eol()) {
      stream.eatWhile(/[^#]/);
      if (stream.match("###")) {
        state.tokenize = tokenBase;
        break;
      }
      stream.eatWhile("#");
    }
    return "comment";
  }

  # 缩进函数
  function indent(stream, state, type) {
    type = type || "coffee";
    var offset = 0, align = false, alignOffset = null;
    for (var scope = state.scope; scope; scope = scope.prev) {
      if (scope.type === "coffee" || scope.type == "}") {
        offset = scope.offset + conf.indentUnit;
        break;
      }
    }
    if (type !== "coffee") {
      align = null;
      alignOffset = stream.column() + stream.current().length;
  // 如果状态的作用域需要对齐，则将其对齐属性设置为false
  } else if (state.scope.align) {
    state.scope.align = false;
  }
  // 设置状态的作用域对象，包括偏移量、类型、前一个作用域、对齐属性和对齐偏移量
  state.scope = {
    offset: offset,
    type: type,
    prev: state.scope,
    align: align,
    alignOffset: alignOffset
  };
}

// 减少缩进
function dedent(stream, state) {
  // 如果没有前一个作用域，则返回
  if (!state.scope.prev) return;
  // 如果作用域类型为 "coffee"，则执行以下操作
  if (state.scope.type === "coffee") {
    var _indent = stream.indentation();
    var matched = false;
    // 遍历作用域链，查找匹配的偏移量
    for (var scope = state.scope; scope; scope = scope.prev) {
      if (_indent === scope.offset) {
        matched = true;
        break;
      }
    }
    // 如果没有匹配的偏移量，则返回true
    if (!matched) {
      return true;
    }
    // 将状态的作用域设置为前一个作用域，直到偏移量匹配
    while (state.scope.prev && state.scope.offset !== _indent) {
      state.scope = state.scope.prev;
    }
    return false;
  } else {
    // 将状态的作用域设置为前一个作用域
    state.scope = state.scope.prev;
    return false;
  }
}

// 标记词法分析器
function tokenLexer(stream, state) {
  var style = state.tokenize(stream, state);
  var current = stream.current();

  // 处理作用域变化
  if (current === "return") {
    state.dedent = true;
  }
  if (((current === "->" || current === "=>") && stream.eol())
      || style === "indent") {
    indent(stream, state);
  }
  var delimiter_index = "[({".indexOf(current);
  // 如果当前字符是左括号、左方括号或左大括号，则执行对齐操作
  if (delimiter_index !== -1) {
    indent(stream, state, "])}".slice(delimiter_index, delimiter_index+1));
  }
  // 如果当前字符是缩进关键字，则执行对齐操作
  if (indentKeywords.exec(current)){
    indent(stream, state);
  }
  // 如果当前字符是 "then"，则执行减少缩进操作
  if (current == "then"){
    dedent(stream, state);
  }

  // 如果样式为 "dedent"，则执行减少缩进操作
  if (style === "dedent") {
    if (dedent(stream, state)) {
      return ERRORCLASS;
    }
  }
  delimiter_index = "])}".indexOf(current);
  // 如果当前字符是右括号、右方括号或右大括号，则执行作用域变化操作
  if (delimiter_index !== -1) {
    while (state.scope.type == "coffee" && state.scope.prev)
      state.scope = state.scope.prev;
    if (state.scope.type == current)
      state.scope = state.scope.prev;
  }
    # 如果当前状态为dedent并且已经到达行尾
    if (state.dedent && stream.eol()) {
      # 如果当前作用域类型为"coffee"并且存在上一个作用域，则将当前作用域设置为上一个作用域
        state.scope = state.scope.prev;
      # 将dedent状态设置为false
        state.dedent = false;
    }

    # 返回样式
    return style;
  }

  # 定义外部接口
  var external = {
    # 初始化状态
    startState: function(basecolumn) {
      return {
        tokenize: tokenBase,
        scope: {offset:basecolumn || 0, type:"coffee", prev: null, align: false},
        prop: false,
        dedent: 0
      };
    },

    # 对输入的流进行标记
    token: function(stream, state) {
      # 如果scope的align为null并且scope存在，则将align设置为false
      var fillAlign = state.scope.align === null && state.scope;
      if (fillAlign && stream.sol()) fillAlign.align = false;

      # 调用tokenLexer函数对流进行标记
      var style = tokenLexer(stream, state);
      # 如果样式存在且不为"comment"
      if (style && style != "comment") {
        # 如果fillAlign存在，则将align设置为true
        if (fillAlign) fillAlign.align = true;
        # 如果样式为"punctuation"并且当前流的字符为"."，则将prop设置为true
        state.prop = style == "punctuation" && stream.current() == "."
      }

      # 返回样式
      return style;
    },

    # 根据状态和文本返回缩进值
    indent: function(state, text) {
      # 如果tokenize不等于tokenBase，则返回0
      if (state.tokenize != tokenBase) return 0;
      var scope = state.scope;
      var closer = text && "])}".indexOf(text.charAt(0)) > -1;
      # 如果closer为true，则在作用域类型为"coffee"且存在上一个作用域的情况下，将作用域设置为上一个作用域
      if (closer) while (scope.type == "coffee" && scope.prev) scope = scope.prev;
      var closes = closer && scope.type === text.charAt(0);
      # 如果作用域的align为true，则返回alignOffset减去1（如果closes为true）的值
      if (scope.align)
        return scope.alignOffset - (closes ? 1 : 0);
      else
        # 否则返回作用域的offset（如果closes为true，则返回上一个作用域的offset）
        return (closes ? scope.prev : scope).offset;
    },

    # 定义行注释符号
    lineComment: "#",
    # 定义折叠方式为缩进
    fold: "indent"
  };
  # 返回外部接口
  return external;
// 定义 MIME 类型为 application/vnd.coffeescript 的 CodeMirror 模式为 coffeescript
CodeMirror.defineMIME("application/vnd.coffeescript", "coffeescript");

// 定义 MIME 类型为 text/x-coffeescript 的 CodeMirror 模式为 coffeescript
CodeMirror.defineMIME("text/x-coffeescript", "coffeescript");

// 定义 MIME 类型为 text/coffeescript 的 CodeMirror 模式为 coffeescript
CodeMirror.defineMIME("text/coffeescript", "coffeescript");

// 定义 CSS 的 CodeMirror 模式
CodeMirror.defineMode("css", function(config, parserConfig) {
  // 获取配置中的 inline 属性
  var inline = parserConfig.inline
  // 如果没有指定 propertyKeywords，则使用 text/css 模式
  if (!parserConfig.propertyKeywords) parserConfig = CodeMirror.resolveMode("text/css");

  // 其他配置项
  var indentUnit = config.indentUnit,
      tokenHooks = parserConfig.tokenHooks,
      documentTypes = parserConfig.documentTypes || {},
      mediaTypes = parserConfig.mediaTypes || {},
      mediaFeatures = parserConfig.mediaFeatures || {},
      mediaValueKeywords = parserConfig.mediaValueKeywords || {},
      propertyKeywords = parserConfig.propertyKeywords || {},
      nonStandardPropertyKeywords = parserConfig.nonStandardPropertyKeywords || {},
      fontProperties = parserConfig.fontProperties || {},
      counterDescriptors = parserConfig.counterDescriptors || {},
      colorKeywords = parserConfig.colorKeywords || {},
      valueKeywords = parserConfig.valueKeywords || {},
      allowNested = parserConfig.allowNested,
      lineComment = parserConfig.lineComment,
      supportsAtComponent = parserConfig.supportsAtComponent === true;

  var type, override;
  function ret(style, tp) { type = tp; return style; }

  // Tokenizers

  function tokenBase(stream, state) {
    var ch = stream.next();
    // 如果存在与当前字符对应的tokenHooks函数，则调用该函数处理当前字符
    if (tokenHooks[ch]) {
      var result = tokenHooks[ch](stream, state);
      // 如果处理结果不为false，则返回结果
      if (result !== false) return result;
    }
    // 如果当前字符为"@"，则读取连续的字母、数字、反斜杠和连字符，并返回"def"类型和当前读取的内容
    if (ch == "@") {
      stream.eatWhile(/[\w\\\-]/);
      return ret("def", stream.current());
    } else if (ch == "=" || (ch == "~" || ch == "|") && stream.eat("=")) {
      // 如果当前字符为"="，或者为"~"或"|"并且下一个字符为"="，则返回null类型和"compare"内容
      return ret(null, "compare");
    } else if (ch == "\"" || ch == "'") {
      // 如果当前字符为双引号或单引号，则设置状态的tokenize函数为tokenString，并调用该函数处理当前字符
      state.tokenize = tokenString(ch);
      return state.tokenize(stream, state);
    } else if (ch == "#") {
      // 如果当前字符为"#"，则读取连续的字母、数字、反斜杠和连字符，并返回"atom"类型和"hash"内容
      stream.eatWhile(/[\w\\\-]/);
      return ret("atom", "hash");
    } else if (ch == "!") {
      // 如果当前字符为"!"，则匹配零或多个空白字符和字母数字字符，并返回"keyword"类型和"important"内容
      stream.match(/^\s*\w*/);
      return ret("keyword", "important");
    } else if (/\d/.test(ch) || ch == "." && stream.eat(/\d/)) {
      // 如果当前字符为数字，或者为"."并且下一个字符为数字，则读取连续的字母、数字、百分号，并返回"number"类型和"unit"内容
      stream.eatWhile(/[\w.%]/);
      return ret("number", "unit");
    } else if (ch === "-") {
      if (/[\d.]/.test(stream.peek())) {
        // 如果下一个字符为数字或"."，则读取连续的字母、数字、百分号，并返回"number"类型和"unit"内容
        stream.eatWhile(/[\w.%]/);
        return ret("number", "unit");
      } else if (stream.match(/^-[\w\\\-]*/)) {
        // 如果下一个字符为字母、数字或连字符，则读取连续的字母、数字、反斜杠和连字符
        stream.eatWhile(/[\w\\\-]/);
        // 如果下一个字符为":"，则返回"variable-2"类型和"variable-definition"内容，否则返回"variable-2"类型和"variable"内容
        if (stream.match(/^\s*:/, false))
          return ret("variable-2", "variable-definition");
        return ret("variable-2", "variable");
      } else if (stream.match(/^\w+-/)) {
        // 如果下一个字符为字母、数字或连字符，则返回"meta"类型和"meta"内容
        return ret("meta", "meta");
      }
    } else if (/[,+>*\/]/.test(ch)) {
      // 如果当前字符为",", "+", ">", "*"或"/"，则返回null类型和"select-op"内容
      return ret(null, "select-op");
    } else if (ch == "." && stream.match(/^-?[_a-z][_a-z0-9-]*/i)) {
      // 如果当前字符为"."，并且下一个字符为合法的CSS标识符，则返回"qualifier"类型和"qualifier"内容
      return ret("qualifier", "qualifier");
    } else if (/[:;{}\[\]\(\)]/.test(ch)) {
      // 如果当前字符为":", ";", "{", "}", "["或"]"，则返回null类型和当前字符内容
      return ret(null, ch);
    } else if (stream.match(/[\w-.]+(?=\()/)) {
      // 如果下一个字符为"("，则判断当前内容是否为"url"、"url-prefix"、"domain"或"regexp"，如果是则设置状态的tokenize函数为tokenParenthesized
      if (/^(url(-prefix)?|domain|regexp)$/.test(stream.current().toLowerCase())) {
        state.tokenize = tokenParenthesized;
      }
      return ret("variable callee", "variable");
    } else if (/[\w\\\-]/.test(ch)) {
      // 如果当前字符为字母、数字或连字符，则读取连续的字母、数字、反斜杠和连字符，并返回"property"类型和"word"内容
      stream.eatWhile(/[\w\\\-]/);
      return ret("property", "word");
    } else {
      // 其他情况返回null类型和null内容
      return ret(null, null);
    }
  }

  // 定义tokenString函数，用于处理字符串类型的token
  function tokenString(quote) {
    // 定义一个返回函数的函数，接受流和状态作为参数
    return function(stream, state) {
      // 初始化变量，用于判断是否转义
      var escaped = false, ch;
      // 循环遍历流中的字符
      while ((ch = stream.next()) != null) {
        // 如果字符是引号并且没有转义
        if (ch == quote && !escaped) {
          // 如果引号是右括号，则回退一个字符
          if (quote == ")") stream.backUp(1);
          // 退出循环
          break;
        }
        // 更新转义状态
        escaped = !escaped && ch == "\\";
      }
      // 如果字符是引号或者没有转义并且引号不是右括号，则将状态的tokenize设置为null
      if (ch == quote || !escaped && quote != ")") state.tokenize = null;
      // 返回字符串类型的token
      return ret("string", "string");
    };
  }

  // 定义处理括号的函数
  function tokenParenthesized(stream, state) {
    // 读取下一个字符，必须是'('
    stream.next(); // Must be '('
    // 如果后面没有匹配到空白、双引号或右括号，则将状态的tokenize设置为tokenString(")")
    if (!stream.match(/\s*[\"\')]/, false))
      state.tokenize = tokenString(")");
    else
      state.tokenize = null;
    // 返回null类型的token和'('
    return ret(null, "(");
  }

  // 上下文管理

  // 定义上下文对象
  function Context(type, indent, prev) {
    this.type = type;
    this.indent = indent;
    this.prev = prev;
  }

  // 推入上下文
  function pushContext(state, stream, type, indent) {
    // 创建新的上下文对象，设置为当前上下文的子上下文
    state.context = new Context(type, stream.indentation() + (indent === false ? 0 : indentUnit), state.context);
    // 返回类型
    return type;
  }

  // 弹出上下文
  function popContext(state) {
    // 如果有父上下文，则设置当前上下文为父上下文
    if (state.context.prev)
      state.context = state.context.prev;
    // 返回类型
    return state.context.type;
  }

  // 传递类型
  function pass(type, stream, state) {
    // 调用当前上下文类型对应的处理函数
    return states[state.context.type](type, stream, state);
  }
  // 弹出上下文并传递类型
  function popAndPass(type, stream, state, n) {
    // 弹出n个上下文
    for (var i = n || 1; i > 0; i--)
      state.context = state.context.prev;
    // 传递类型
    return pass(type, stream, state);
  }

  // 解析器

  // 将单词作为值处理
  function wordAsValue(stream) {
    // 获取当前单词并转换为小写
    var word = stream.current().toLowerCase();
    // 如果是值关键字，则覆盖为原子类型
    if (valueKeywords.hasOwnProperty(word))
      override = "atom";
    // 如果是颜色关键字，则覆盖为关键字类型
    else if (colorKeywords.hasOwnProperty(word))
      override = "keyword";
    // 否则覆盖为变量类型
    else
      override = "variable";
  }

  // 定义状态对象
  var states = {};

  // 顶层状态处理函数
  states.top = function(type, stream, state) {
    // 如果是'{'，则推入block类型的上下文
    if (type == "{") {
      return pushContext(state, stream, "block");
    } else if (type == "}" && state.context.prev) {
      // 如果是'}'并且有父上下文，则弹出上下文
      return popContext(state);
    } else if (supportsAtComponent && /@component/i.test(type)) {
      # 如果支持 @component 并且类型中包含 @component，则将上下文推入状态栈中
      return pushContext(state, stream, "atComponentBlock");
    } else if (/^@(-moz-)?document$/i.test(type)) {
      # 如果类型以 @document 开头，则将上下文推入状态栈中
      return pushContext(state, stream, "documentTypes");
    } else if (/^@(media|supports|(-moz-)?document|import)$/i.test(type)) {
      # 如果类型以 @media、@supports、@document 或 @import 开头，则将上下文推入状态栈中
      return pushContext(state, stream, "atBlock");
    } else if (/^@(font-face|counter-style)/i.test(type)) {
      # 如果类型以 @font-face 或 @counter-style 开头，则设置状态参数并返回 "restricted_atBlock_before"
      state.stateArg = type;
      return "restricted_atBlock_before";
    } else if (/^@(-(moz|ms|o|webkit)-)?keyframes$/i.test(type)) {
      # 如果类型以 @keyframes 开头，则返回 "keyframes"
      return "keyframes";
    } else if (type && type.charAt(0) == "@") {
      # 如果类型以 @ 开头，则将上下文推入状态栈中
      return pushContext(state, stream, "at");
    } else if (type == "hash") {
      # 如果类型为 "hash"，则覆盖为 "builtin"
      override = "builtin";
    } else if (type == "word") {
      # 如果类型为 "word"，则覆盖为 "tag"
      override = "tag";
    } else if (type == "variable-definition") {
      # 如果类型为 "variable-definition"，则返回 "maybeprop"
      return "maybeprop";
    } else if (type == "interpolation") {
      # 如果类型为 "interpolation"，则将上下文推入状态栈中
      return pushContext(state, stream, "interpolation");
    } else if (type == ":") {
      # 如果类型为 ":"，则返回 "pseudo"
      return "pseudo";
    } else if (allowNested && type == "(") {
      # 如果允许嵌套并且类型为 "("，则将上下文推入状态栈中
      return pushContext(state, stream, "parens");
    }
    # 返回当前状态的类型
    return state.context.type;
  };

  states.block = function(type, stream, state) {
    if (type == "word") {
      # 如果类型为 "word"
      var word = stream.current().toLowerCase();
      if (propertyKeywords.hasOwnProperty(word)) {
        # 如果属性关键字中包含当前单词，则覆盖为 "property"，并返回 "maybeprop"
        override = "property";
        return "maybeprop";
      } else if (nonStandardPropertyKeywords.hasOwnProperty(word)) {
        # 如果非标准属性关键字中包含当前单词，则覆盖为 "string-2"，并返回 "maybeprop"
        override = "string-2";
        return "maybeprop";
      } else if (allowNested) {
        # 如果允许嵌套
        override = stream.match(/^\s*:(?:\s|$)/, false) ? "property" : "tag";
        return "block";
      } else {
        # 否则，覆盖为 "error"，并返回 "maybeprop"
        override += " error";
        return "maybeprop";
      }
    } else if (type == "meta") {
      # 如果类型为 "meta"，则返回 "block"
      return "block";
    } else if (!allowNested && (type == "hash" || type == "qualifier")) {
      # 如果不允许嵌套并且类型为 "hash" 或 "qualifier"，则覆盖为 "error"，并返回 "block"
      override = "error";
      return "block";
    } else {
      # 否则，调用 states.top() 方法
      return states.top(type, stream, state);
  // 定义 states 对象的 maybeprop 方法，处理可能是属性的情况
  states.maybeprop = function(type, stream, state) {
    // 如果类型是冒号，则将上下文切换为 prop
    if (type == ":") return pushContext(state, stream, "prop");
    // 否则继续处理
    return pass(type, stream, state);
  };

  // 定义 states 对象的 prop 方法，处理属性的情况
  states.prop = function(type, stream, state) {
    // 如果类型是分号，则弹出上下文
    if (type == ";") return popContext(state);
    // 如果类型是左大括号并且允许嵌套，则将上下文切换为 propBlock
    if (type == "{" && allowNested) return pushContext(state, stream, "propBlock");
    // 如果类型是右大括号或左大括号，则弹出上下文并继续处理
    if (type == "}" || type == "{") return popAndPass(type, stream, state);
    // 如果类型是左括号，则将上下文切换为 parens
    if (type == "(") return pushContext(state, stream, "parens");

    // 如果类型是 hash 并且不符合颜色值的正则表达式，则添加 error 类
    if (type == "hash" && !/^#([0-9a-fA-f]{3,4}|[0-9a-fA-f]{6}|[0-9a-fA-f]{8})$/.test(stream.current())) {
      override += " error";
    } else if (type == "word") {
      wordAsValue(stream);
    } else if (type == "interpolation") {
      return pushContext(state, stream, "interpolation");
    }
    return "prop";
  };

  // 定义 states 对象的 propBlock 方法，处理属性块的情况
  states.propBlock = function(type, _stream, state) {
    // 如果类型是右大括号，则弹出上下文
    if (type == "}") return popContext(state);
    // 如果类型是单词，则将 override 设置为 property，并将上下文切换为 maybeprop
    if (type == "word") { override = "property"; return "maybeprop"; }
    // 否则返回当前上下文的类型
    return state.context.type;
  };

  // 定义 states 对象的 parens 方法，处理括号的情况
  states.parens = function(type, stream, state) {
    // 如果类型是左大括号或右大括号，则弹出上下文并继续处理
    if (type == "{" || type == "}") return popAndPass(type, stream, state);
    // 如果类型是右括号，则弹出上下文
    if (type == ")") return popContext(state);
    // 如果类型是左括号，则将上下文切换为 parens
    if (type == "(") return pushContext(state, stream, "parens");
    // 如果类型是插值，则将上下文切换为 interpolation
    if (type == "interpolation") return pushContext(state, stream, "interpolation");
    // 如果类型是单词，则将其作为值处理
    if (type == "word") wordAsValue(stream);
    // 返回当前上下文的类型
    return "parens";
  };

  // 定义 states 对象的 pseudo 方法，处理伪类的情况
  states.pseudo = function(type, stream, state) {
    // 如果类型是 meta，则返回伪类
    if (type == "meta") return "pseudo";

    // 如果类型是单词，则将 override 设置为 variable-3，并返回当前上下文的类型
    if (type == "word") {
      override = "variable-3";
      return state.context.type;
    }
    // 否则继续处理
    return pass(type, stream, state);
  };

  // 定义 states 对象的 documentTypes 方法，处理文档类型的情况
  states.documentTypes = function(type, stream, state) {
    // 如果类型是单词并且是已知的文档类型，则将 override 设置为 tag，并返回当前上下文的类型
    if (type == "word" && documentTypes.hasOwnProperty(stream.current())) {
      override = "tag";
      return state.context.type;
    } else {
      // 否则继续处理
      return states.atBlock(type, stream, state);
    }
  };

  // 定义 states 对象的 atBlock 方法，处理 at 规则的情况
  states.atBlock = function(type, stream, state) {
    // ...
  };
    # 如果类型为"("，则返回推送上下文到状态机，进入"atBlock_parens"状态
    if (type == "(") return pushContext(state, stream, "atBlock_parens");
    # 如果类型为"}"或";"，则返回弹出并传递类型、流和状态
    if (type == "}" || type == ";") return popAndPass(type, stream, state);
    # 如果类型为"{"，则返回弹出上下文并推送上下文到状态机，进入"block"或"top"状态（根据allowNested参数决定）
    if (type == "{") return popContext(state) && pushContext(state, stream, allowNested ? "block" : "top");

    # 如果类型为"interpolation"，则返回推送上下文到状态机，进入"interpolation"状态
    if (type == "interpolation") return pushContext(state, stream, "interpolation");

    # 如果类型为"word"
    if (type == "word") {
      # 获取当前单词并转换为小写
      var word = stream.current().toLowerCase();
      # 根据单词内容设置override的值
      if (word == "only" || word == "not" || word == "and" || word == "or")
        override = "keyword";
      else if (mediaTypes.hasOwnProperty(word))
        override = "attribute";
      else if (mediaFeatures.hasOwnProperty(word))
        override = "property";
      else if (mediaValueKeywords.hasOwnProperty(word))
        override = "keyword";
      else if (propertyKeywords.hasOwnProperty(word))
        override = "property";
      else if (nonStandardPropertyKeywords.hasOwnProperty(word))
        override = "string-2";
      else if (valueKeywords.hasOwnProperty(word))
        override = "atom";
      else if (colorKeywords.hasOwnProperty(word))
        override = "keyword";
      else
        override = "error";
    }
    # 返回状态机的上下文类型
    return state.context.type;
  };

  # 定义状态机的"atComponentBlock"状态处理函数
  states.atComponentBlock = function(type, stream, state) {
    # 如果类型为"}"，则返回弹出并传递类型、流和状态
    if (type == "}")
      return popAndPass(type, stream, state);
    # 如果类型为"{"，则返回弹出上下文并推送上下文到状态机，进入"block"或"top"状态（根据allowNested参数决定），并禁止嵌套
    if (type == "{")
      return popContext(state) && pushContext(state, stream, allowNested ? "block" : "top", false);
    # 如果类型为"word"，则设置override为"error"
    if (type == "word")
      override = "error";
    # 返回状态机的上下文类型
    return state.context.type;
  };

  # 定义状态机的"atBlock_parens"状态处理函数
  states.atBlock_parens = function(type, stream, state) {
    # 如果类型为")"，则返回弹出上下文
    if (type == ")") return popContext(state);
    # 如果类型为"{"或"}"，则返回弹出并传递类型、流和状态，传递参数2
    if (type == "{" || type == "}") return popAndPass(type, stream, state, 2);
    # 其他情况下，调用atBlock状态处理函数
    return states.atBlock(type, stream, state);
  };

  # 定义状态机的"restricted_atBlock_before"状态处理函数
  states.restricted_atBlock_before = function(type, stream, state) {
    # 如果类型为"{"，则返回推送上下文到状态机，进入"restricted_atBlock"状态
    if (type == "{")
      return pushContext(state, stream, "restricted_atBlock");
    # 如果类型为 "word" 并且状态参数为 "@counter-style"，则设置覆盖为 "variable"，并返回 "restricted_atBlock_before"
    if (type == "word" && state.stateArg == "@counter-style") {
      override = "variable";
      return "restricted_atBlock_before";
    }
    # 返回 pass 函数的结果
    return pass(type, stream, state);
  };

  # 定义状态为 restricted_atBlock 的函数
  states.restricted_atBlock = function(type, stream, state) {
    # 如果类型为 "}"，则重置状态参数并返回弹出上下文的结果
    if (type == "}") {
      state.stateArg = null;
      return popContext(state);
    }
    # 如果类型为 "word"
    if (type == "word") {
      # 如果状态参数为 "@font-face" 并且字体属性中不包含当前单词的小写形式，或者状态参数为 "@counter-style" 并且计数器描述中不包含当前单词的小写形式
      if ((state.stateArg == "@font-face" && !fontProperties.hasOwnProperty(stream.current().toLowerCase())) ||
          (state.stateArg == "@counter-style" && !counterDescriptors.hasOwnProperty(stream.current().toLowerCase())))
        # 设置覆盖为 "error"，否则设置为 "property"
        override = "error";
      else
        override = "property";
      # 返回 "maybeprop"
      return "maybeprop";
    }
    # 返回 "restricted_atBlock"
    return "restricted_atBlock";
  };

  # 定义状态为 keyframes 的函数
  states.keyframes = function(type, stream, state) {
    # 如果类型为 "word"，则设置覆盖为 "variable"，并返回 "keyframes"
    if (type == "word") { override = "variable"; return "keyframes"; }
    # 如果类型为 "{"，则推入上下文并返回 "top"
    if (type == "{") return pushContext(state, stream, "top");
    # 返回 pass 函数的结果
    return pass(type, stream, state);
  };

  # 定义状态为 at 的函数
  states.at = function(type, stream, state) {
    # 如果类型为 ";"，则返回弹出上下文的结果
    if (type == ";") return popContext(state);
    # 如果类型为 "{" 或 "}"，则返回弹出并传递的结果
    if (type == "{" || type == "}") return popAndPass(type, stream, state);
    # 如果类型为 "word"，则设置覆盖为 "tag"
    if (type == "word") override = "tag";
    # 如果类型为 "hash"，则设置覆盖为 "builtin"
    else if (type == "hash") override = "builtin";
    # 返回 "at"
    return "at";
  };

  # 定义状态为 interpolation 的函数
  states.interpolation = function(type, stream, state) {
    # 如果类型为 "}"，则返回弹出上下文的结果
    if (type == "}") return popContext(state);
    # 如果类型为 "{" 或 ";"，则返回弹出并传递的结果
    if (type == "{" || type == ";") return popAndPass(type, stream, state);
    # 如果类型为 "word"，则设置覆盖为 "variable"
    if (type == "word") override = "variable";
    # 如果类型不是 "variable"、"(" 或 ")"，则设置覆盖为 "error"
    else if (type != "variable" && type != "(" && type != ")") override = "error";
    # 返回 "interpolation"
    return "interpolation";
  };

  # 返回一个对象，其中包含 startState 函数
  return {
    startState: function(base) {
      # 返回一个对象，包含 tokenize、state、stateArg 和 context 属性
      return {tokenize: null,
              state: inline ? "block" : "top",
              stateArg: null,
              context: new Context(inline ? "block" : "top", base || 0, null)};
    },
    // 定义 token 方法，用于处理代码流和状态
    token: function(stream, state) {
      // 如果不在 tokenize 状态下并且遇到空白字符，则返回 null
      if (!state.tokenize && stream.eatSpace()) return null;
      // 调用 tokenBase 方法处理代码流和状态，获取样式
      var style = (state.tokenize || tokenBase)(stream, state);
      // 如果样式存在并且是对象类型
      if (style && typeof style == "object") {
        // 获取样式类型和样式值
        type = style[1];
        style = style[0];
      }
      // 覆盖样式
      override = style;
      // 如果类型不是注释
      if (type != "comment")
        // 根据当前状态和类型处理代码流和状态
        state.state = states[state.state](type, stream, state);
      // 返回覆盖样式
      return override;
    },

    // 定义 indent 方法，用于处理状态和文本后的内容
    indent: function(state, textAfter) {
      // 获取上下文和文本后的第一个字符
      var cx = state.context, ch = textAfter && textAfter.charAt(0);
      // 获取缩进值
      var indent = cx.indent;
      // 如果上下文类型是属性并且文本后的字符是 "}" 或 ")"
      if (cx.type == "prop" && (ch == "}" || ch == ")")) cx = cx.prev;
      // 如果存在上一个上下文
      if (cx.prev) {
        // 如果文本后的字符是 "}" 并且上下文类型是 "block"、"top"、"interpolation" 或 "restricted_atBlock"
        if (ch == "}" && (cx.type == "block" || cx.type == "top" ||
                          cx.type == "interpolation" || cx.type == "restricted_atBlock")) {
          // 从父上下文恢复缩进
          cx = cx.prev;
          indent = cx.indent;
        } else if (ch == ")" && (cx.type == "parens" || cx.type == "atBlock_parens") ||
            ch == "{" && (cx.type == "at" || cx.type == "atBlock")) {
          // 相对于当前上下文减少缩进
          indent = Math.max(0, cx.indent - indentUnit);
        }
      }
      // 返回缩进值
      return indent;
    },

    // 定义 electricChars 属性，表示触发自动缩进的字符
    electricChars: "}",
    // 定义 blockCommentStart 属性，表示块注释的起始标记
    blockCommentStart: "/*",
    // 定义 blockCommentEnd 属性，表示块注释的结束标记
    blockCommentEnd: "*/",
    // 定义 blockCommentContinue 属性，表示块注释的续行标记
    blockCommentContinue: " * ",
    // 定义 lineComment 属性，表示行注释的标记
    lineComment: lineComment,
    // 定义 fold 属性，表示折叠的标记
    fold: "brace"
  };
  // 定义一个函数，用于将数组转换为键值对，键为数组元素的小写形式，值为 true
  function keySet(array) {
    var keys = {};
    for (var i = 0; i < array.length; ++i) {
      keys[array[i].toLowerCase()] = true;
    }
    return keys;
  }

  // 定义文档类型数组和对应的键值对
  var documentTypes_ = [
    "domain", "regexp", "url", "url-prefix"
  ], documentTypes = keySet(documentTypes_);

  // 定义媒体类型数组和对应的键值对
  var mediaTypes_ = [
    "all", "aural", "braille", "handheld", "print", "projection", "screen",
    "tty", "tv", "embossed"
  ], mediaTypes = keySet(mediaTypes_);

  // 定义媒体特性数组和对应的键值对
  var mediaFeatures_ = [
    "width", "min-width", "max-width", "height", "min-height", "max-height",
    "device-width", "min-device-width", "max-device-width", "device-height",
    "min-device-height", "max-device-height", "aspect-ratio",
    "min-aspect-ratio", "max-aspect-ratio", "device-aspect-ratio",
    "min-device-aspect-ratio", "max-device-aspect-ratio", "color", "min-color",
    "max-color", "color-index", "min-color-index", "max-color-index",
    "monochrome", "min-monochrome", "max-monochrome", "resolution",
    "min-resolution", "max-resolution", "scan", "grid", "orientation",
    "device-pixel-ratio", "min-device-pixel-ratio", "max-device-pixel-ratio",
    "pointer", "any-pointer", "hover", "any-hover"
  ], mediaFeatures = keySet(mediaFeatures_);

  // 定义媒体值关键词数组和对应的键值对
  var mediaValueKeywords_ = [
    "landscape", "portrait", "none", "coarse", "fine", "on-demand", "hover",
    "interlace", "progressive"
  ], mediaValueKeywords = keySet(mediaValueKeywords_);

  // 定义属性关键词数组和对应的键值对
  var propertyKeywords_ = [
    "align-content", "align-items", "align-self", "alignment-adjust",
    "alignment-baseline", "anchor-point", "animation", "animation-delay",
    "animation-direction", "animation-duration", "animation-fill-mode",
    "animation-iteration-count", "animation-name", "animation-play-state",
    "animation-timing-function", "appearance", "azimuth", "backdrop-filter",
    "backface-visibility", "background", "background-attachment",
    "background-blend-mode", "background-clip", "background-color",
    // ... 其他属性关键词
  ];
    # CSS 属性列表，包含各种样式属性
    "background-image", "background-origin", "background-position",
    "background-position-x", "background-position-y", "background-repeat",
    "background-size", "baseline-shift", "binding", "bleed", "block-size",
    "bookmark-label", "bookmark-level", "bookmark-state", "bookmark-target",
    "border", "border-bottom", "border-bottom-color", "border-bottom-left-radius",
    "border-bottom-right-radius", "border-bottom-style", "border-bottom-width",
    "border-collapse", "border-color", "border-image", "border-image-outset",
    "border-image-repeat", "border-image-slice", "border-image-source",
    "border-image-width", "border-left", "border-left-color", "border-left-style",
    "border-left-width", "border-radius", "border-right", "border-right-color",
    "border-right-style", "border-right-width", "border-spacing", "border-style",
    "border-top", "border-top-color", "border-top-left-radius",
    "border-top-right-radius", "border-top-style", "border-top-width",
    "border-width", "bottom", "box-decoration-break", "box-shadow", "box-sizing",
    "break-after", "break-before", "break-inside", "caption-side", "caret-color",
    "clear", "clip", "color", "color-profile", "column-count", "column-fill",
    "column-gap", "column-rule", "column-rule-color", "column-rule-style",
    "column-rule-width", "column-span", "column-width", "columns", "contain",
    "content", "counter-increment", "counter-reset", "crop", "cue", "cue-after",
    "cue-before", "cursor", "direction", "display", "dominant-baseline",
    "drop-initial-after-adjust", "drop-initial-after-align",
    "drop-initial-before-adjust", "drop-initial-before-align", "drop-initial-size",
    "drop-initial-value", "elevation", "empty-cells", "fit", "fit-position",
    "flex", "flex-basis", "flex-direction", "flex-flow", "flex-grow",
    "flex-shrink", "flex-wrap", "float", "float-offset", "flow-from", "flow-into",
    "font", "font-family", "font-feature-settings", "font-kerning",
    # 定义一系列 CSS 属性名称
    "font-language-override", "font-optical-sizing", "font-size",
    "font-size-adjust", "font-stretch", "font-style", "font-synthesis",
    "font-variant", "font-variant-alternates", "font-variant-caps",
    "font-variant-east-asian", "font-variant-ligatures", "font-variant-numeric",
    "font-variant-position", "font-variation-settings", "font-weight", "gap",
    "grid", "grid-area", "grid-auto-columns", "grid-auto-flow", "grid-auto-rows",
    "grid-column", "grid-column-end", "grid-column-gap", "grid-column-start",
    "grid-gap", "grid-row", "grid-row-end", "grid-row-gap", "grid-row-start",
    "grid-template", "grid-template-areas", "grid-template-columns",
    "grid-template-rows", "hanging-punctuation", "height", "hyphens", "icon",
    "image-orientation", "image-rendering", "image-resolution", "inline-box-align",
    "inset", "inset-block", "inset-block-end", "inset-block-start", "inset-inline",
    "inset-inline-end", "inset-inline-start", "isolation", "justify-content",
    "justify-items", "justify-self", "left", "letter-spacing", "line-break",
    "line-height", "line-height-step", "line-stacking", "line-stacking-ruby",
    "line-stacking-shift", "line-stacking-strategy", "list-style",
    "list-style-image", "list-style-position", "list-style-type", "margin",
    "margin-bottom", "margin-left", "margin-right", "margin-top", "marks",
    "marquee-direction", "marquee-loop", "marquee-play-count", "marquee-speed",
    "marquee-style", "max-block-size", "max-height", "max-inline-size",
    "max-width", "min-block-size", "min-height", "min-inline-size", "min-width",
    "mix-blend-mode", "move-to", "nav-down", "nav-index", "nav-left", "nav-right",
    "nav-up", "object-fit", "object-position", "offset", "offset-anchor",
    "offset-distance", "offset-path", "offset-position", "offset-rotate",
    "opacity", "order", "orphans", "outline", "outline-color", "outline-offset",
    "outline-style", "outline-width", "overflow", "overflow-style",
    # CSS 属性列表，包括各种样式属性
    "overflow-wrap", "overflow-x", "overflow-y", "padding", "padding-bottom",
    "padding-left", "padding-right", "padding-top", "page", "page-break-after",
    "page-break-before", "page-break-inside", "page-policy", "pause",
    "pause-after", "pause-before", "perspective", "perspective-origin", "pitch",
    "pitch-range", "place-content", "place-items", "place-self", "play-during",
    "position", "presentation-level", "punctuation-trim", "quotes",
    "region-break-after", "region-break-before", "region-break-inside",
    "region-fragment", "rendering-intent", "resize", "rest", "rest-after",
    "rest-before", "richness", "right", "rotate", "rotation", "rotation-point",
    "row-gap", "ruby-align", "ruby-overhang", "ruby-position", "ruby-span",
    "scale", "scroll-behavior", "scroll-margin", "scroll-margin-block",
    "scroll-margin-block-end", "scroll-margin-block-start", "scroll-margin-bottom",
    "scroll-margin-inline", "scroll-margin-inline-end",
    "scroll-margin-inline-start", "scroll-margin-left", "scroll-margin-right",
    "scroll-margin-top", "scroll-padding", "scroll-padding-block",
    "scroll-padding-block-end", "scroll-padding-block-start",
    "scroll-padding-bottom", "scroll-padding-inline", "scroll-padding-inline-end",
    "scroll-padding-inline-start", "scroll-padding-left", "scroll-padding-right",
    "scroll-padding-top", "scroll-snap-align", "scroll-snap-type",
    "shape-image-threshold", "shape-inside", "shape-margin", "shape-outside",
    "size", "speak", "speak-as", "speak-header", "speak-numeral",
    "speak-punctuation", "speech-rate", "stress", "string-set", "tab-size",
    "table-layout", "target", "target-name", "target-new", "target-position",
    "text-align", "text-align-last", "text-combine-upright", "text-decoration",
    "text-decoration-color", "text-decoration-line", "text-decoration-skip",
    "text-decoration-skip-ink", "text-decoration-style", "text-emphasis",
    # 定义了一系列 CSS 属性关键字，包括文本样式、布局、动画等
    "text-emphasis-color", "text-emphasis-position", "text-emphasis-style",
    "text-height", "text-indent", "text-justify", "text-orientation",
    "text-outline", "text-overflow", "text-rendering", "text-shadow",
    "text-size-adjust", "text-space-collapse", "text-transform",
    "text-underline-position", "text-wrap", "top", "transform", "transform-origin",
    "transform-style", "transition", "transition-delay", "transition-duration",
    "transition-property", "transition-timing-function", "translate",
    "unicode-bidi", "user-select", "vertical-align", "visibility", "voice-balance",
    "voice-duration", "voice-family", "voice-pitch", "voice-range", "voice-rate",
    "voice-stress", "voice-volume", "volume", "white-space", "widows", "width",
    "will-change", "word-break", "word-spacing", "word-wrap", "writing-mode", "z-index",
    # SVG 特定的属性关键字
    "clip-path", "clip-rule", "mask", "enable-background", "filter", "flood-color",
    "flood-opacity", "lighting-color", "stop-color", "stop-opacity", "pointer-events",
    "color-interpolation", "color-interpolation-filters",
    "color-rendering", "fill", "fill-opacity", "fill-rule", "image-rendering",
    "marker", "marker-end", "marker-mid", "marker-start", "shape-rendering", "stroke",
    "stroke-dasharray", "stroke-dashoffset", "stroke-linecap", "stroke-linejoin",
    "stroke-miterlimit", "stroke-opacity", "stroke-width", "text-rendering",
    "baseline-shift", "dominant-baseline", "glyph-orientation-horizontal",
    "glyph-orientation-vertical", "text-anchor", "writing-mode"
    # 定义了一系列非标准的 CSS 属性关键字
    ], propertyKeywords = keySet(propertyKeywords_);
    
    var nonStandardPropertyKeywords_ = [
    "border-block", "border-block-color", "border-block-end",
    "border-block-end-color", "border-block-end-style", "border-block-end-width",
    "border-block-start", "border-block-start-color", "border-block-start-style",
    "border-block-start-width", "border-block-style", "border-block-width",
    # 定义包含 CSS 属性关键字的数组
    var cssPropertyKeywords_ = [
        "border-inline", "border-inline-color", "border-inline-end",
        "border-inline-end-color", "border-inline-end-style",
        "border-inline-end-width", "border-inline-start", "border-inline-start-color",
        "border-inline-start-style", "border-inline-start-width",
        "border-inline-style", "border-inline-width", "margin-block",
        "margin-block-end", "margin-block-start", "margin-inline", "margin-inline-end",
        "margin-inline-start", "padding-block", "padding-block-end",
        "padding-block-start", "padding-inline", "padding-inline-end",
        "padding-inline-start", "scroll-snap-stop", "scrollbar-3d-light-color",
        "scrollbar-arrow-color", "scrollbar-base-color", "scrollbar-dark-shadow-color",
        "scrollbar-face-color", "scrollbar-highlight-color", "scrollbar-shadow-color",
        "scrollbar-track-color", "searchfield-cancel-button", "searchfield-decoration",
        "searchfield-results-button", "searchfield-results-decoration", "shape-inside", "zoom"
      ], nonStandardPropertyKeywords = keySet(nonStandardPropertyKeywords_);
    
    # 将 CSS 属性关键字数组转换为关键字集合
    var cssPropertyKeywords = keySet(cssPropertyKeywords_);
    
    # 定义包含字体属性关键字的数组
    var fontProperties_ = [
        "font-display", "font-family", "src", "unicode-range", "font-variant",
         "font-feature-settings", "font-stretch", "font-weight", "font-style"
      ], fontProperties = keySet(fontProperties_);
    
    # 定义包含计数器描述符关键字的数组
    var counterDescriptors_ = [
        "additive-symbols", "fallback", "negative", "pad", "prefix", "range",
        "speak-as", "suffix", "symbols", "system"
      ], counterDescriptors = keySet(counterDescriptors_);
    
    # 定义包含颜色关键字的数组
    var colorKeywords_ = [
        "aliceblue", "antiquewhite", "aqua", "aquamarine", "azure", "beige",
        "bisque", "black", "blanchedalmond", "blue", "blueviolet", "brown",
        "burlywood", "cadetblue", "chartreuse", "chocolate", "coral", "cornflowerblue",
        "cornsilk", "crimson", "cyan", "darkblue", "darkcyan", "darkgoldenrod",
        "darkgray", "darkgreen", "darkkhaki", "darkmagenta", "darkolivegreen",
        "darkorange", "darkorchid", "darkred", "darksalmon", "darkseagreen",
        # ...（此处省略了部分颜色关键字）
      ];
    # 颜色关键词列表
    [
        "darkslateblue", "darkslategray", "darkturquoise", "darkviolet",
        "deeppink", "deepskyblue", "dimgray", "dodgerblue", "firebrick",
        "floralwhite", "forestgreen", "fuchsia", "gainsboro", "ghostwhite",
        "gold", "goldenrod", "gray", "grey", "green", "greenyellow", "honeydew",
        "hotpink", "indianred", "indigo", "ivory", "khaki", "lavender",
        "lavenderblush", "lawngreen", "lemonchiffon", "lightblue", "lightcoral",
        "lightcyan", "lightgoldenrodyellow", "lightgray", "lightgreen", "lightpink",
        "lightsalmon", "lightseagreen", "lightskyblue", "lightslategray",
        "lightsteelblue", "lightyellow", "lime", "limegreen", "linen", "magenta",
        "maroon", "mediumaquamarine", "mediumblue", "mediumorchid", "mediumpurple",
        "mediumseagreen", "mediumslateblue", "mediumspringgreen", "mediumturquoise",
        "mediumvioletred", "midnightblue", "mintcream", "mistyrose", "moccasin",
        "navajowhite", "navy", "oldlace", "olive", "olivedrab", "orange", "orangered",
        "orchid", "palegoldenrod", "palegreen", "paleturquoise", "palevioletred",
        "papayawhip", "peachpuff", "peru", "pink", "plum", "powderblue",
        "purple", "rebeccapurple", "red", "rosybrown", "royalblue", "saddlebrown",
        "salmon", "sandybrown", "seagreen", "seashell", "sienna", "silver", "skyblue",
        "slateblue", "slategray", "snow", "springgreen", "steelblue", "tan",
        "teal", "thistle", "tomato", "turquoise", "violet", "wheat", "white",
        "whitesmoke", "yellow", "yellowgreen"
      ], 
      # 颜色关键词集合
      colorKeywords = keySet(colorKeywords_);
    
      # 值关键词列表
      var valueKeywords_ = [
        "above", "absolute", "activeborder", "additive", "activecaption", "afar",
        "after-white-space", "ahead", "alias", "all", "all-scroll", "alphabetic", "alternate",
        "always", "amharic", "amharic-abegede", "antialiased", "appworkspace",
        "arabic-indic", "armenian", "asterisks", "attr", "auto", "auto-flow", "avoid", "avoid-column", "avoid-page",
    # 一系列 CSS 属性值，用于设置元素的样式和布局
    # 以下是一些常见的 CSS 属性值，包括字体、颜色、布局等
    "avoid-region", "background", "backwards", "baseline", "below", "bidi-override", "binary",
    "bengali", "blink", "block", "block-axis", "bold", "bolder", "border", "border-box",
    "both", "bottom", "break", "break-all", "break-word", "bullets", "button", "button-bevel",
    "buttonface", "buttonhighlight", "buttonshadow", "buttontext", "calc", "cambodian",
    "capitalize", "caps-lock-indicator", "caption", "captiontext", "caret",
    "cell", "center", "checkbox", "circle", "cjk-decimal", "cjk-earthly-branch",
    "cjk-heavenly-stem", "cjk-ideographic", "clear", "clip", "close-quote",
    "col-resize", "collapse", "color", "color-burn", "color-dodge", "column", "column-reverse",
    "compact", "condensed", "contain", "content", "contents",
    "content-box", "context-menu", "continuous", "copy", "counter", "counters", "cover", "crop",
    "cross", "crosshair", "currentcolor", "cursive", "cyclic", "darken", "dashed", "decimal",
    "decimal-leading-zero", "default", "default-button", "dense", "destination-atop",
    "destination-in", "destination-out", "destination-over", "devanagari", "difference",
    "disc", "discard", "disclosure-closed", "disclosure-open", "document",
    "dot-dash", "dot-dot-dash",
    "dotted", "double", "down", "e-resize", "ease", "ease-in", "ease-in-out", "ease-out",
    "element", "ellipse", "ellipsis", "embed", "end", "ethiopic", "ethiopic-abegede",
    "ethiopic-abegede-am-et", "ethiopic-abegede-gez", "ethiopic-abegede-ti-er",
    "ethiopic-abegede-ti-et", "ethiopic-halehame-aa-er",
    "ethiopic-halehame-aa-et", "ethiopic-halehame-am-et",
    "ethiopic-halehame-gez", "ethiopic-halehame-om-et",
    "ethiopic-halehame-sid-et", "ethiopic-halehame-so-et",
    "ethiopic-halehame-ti-er", "ethiopic-halehame-ti-et", "ethiopic-halehame-tig",
    "ethiopic-numeric", "ew-resize", "exclusion", "expanded", "extends", "extra-condensed",
    # 这是一个长字符串，包含了大量的样式属性值
    # 该字符串可能用于定义 CSS 样式或其他类型的配置
    # 由于没有上下文，无法确定具体用途，只能简单描述其内容
    # 以下是一系列的 CSS 样式属性值，包括字体、颜色、布局等
    # 这些值可以用于定义 HTML 元素的外观和行为
    "mix", "mongolian", "monospace", "move", "multiple", "multiply", "myanmar", "n-resize",
    "narrower", "ne-resize", "nesw-resize", "no-close-quote", "no-drop",
    "no-open-quote", "no-repeat", "none", "normal", "not-allowed", "nowrap",
    "ns-resize", "numbers", "numeric", "nw-resize", "nwse-resize", "oblique", "octal", "opacity", "open-quote",
    "optimizeLegibility", "optimizeSpeed", "oriya", "oromo", "outset",
    "outside", "outside-shape", "overlay", "overline", "padding", "padding-box",
    "painted", "page", "paused", "persian", "perspective", "plus-darker", "plus-lighter",
    "pointer", "polygon", "portrait", "pre", "pre-line", "pre-wrap", "preserve-3d",
    "progress", "push-button", "radial-gradient", "radio", "read-only",
    "read-write", "read-write-plaintext-only", "rectangle", "region",
    "relative", "repeat", "repeating-linear-gradient",
    "repeating-radial-gradient", "repeat-x", "repeat-y", "reset", "reverse",
    "rgb", "rgba", "ridge", "right", "rotate", "rotate3d", "rotateX", "rotateY",
    "rotateZ", "round", "row", "row-resize", "row-reverse", "rtl", "run-in", "running",
    "s-resize", "sans-serif", "saturation", "scale", "scale3d", "scaleX", "scaleY", "scaleZ", "screen",
    "scroll", "scrollbar", "scroll-position", "se-resize", "searchfield",
    "searchfield-cancel-button", "searchfield-decoration",
    "searchfield-results-button", "searchfield-results-decoration", "self-start", "self-end",
    "semi-condensed", "semi-expanded", "separate", "serif", "show", "sidama",
    "simp-chinese-formal", "simp-chinese-informal", "single",
    "skew", "skewX", "skewY", "skip-white-space", "slide", "slider-horizontal",
    "slider-vertical", "sliderthumb-horizontal", "sliderthumb-vertical", "slow",
    "small", "small-caps", "small-caption", "smaller", "soft-light", "solid", "somali",
    "source-atop", "source-in", "source-out", "source-over", "space", "space-around", "space-between", "space-evenly", "spell-out", "square",
    # 定义一系列 CSS 的关键字和值
    "square-button", "start", "static", "status-bar", "stretch", "stroke", "sub",
    "subpixel-antialiased", "super", "sw-resize", "symbolic", "symbols", "system-ui", "table",
    "table-caption", "table-cell", "table-column", "table-column-group",
    "table-footer-group", "table-header-group", "table-row", "table-row-group",
    "tamil",
    "telugu", "text", "text-bottom", "text-top", "textarea", "textfield", "thai",
    "thick", "thin", "threeddarkshadow", "threedface", "threedhighlight",
    "threedlightshadow", "threedshadow", "tibetan", "tigre", "tigrinya-er",
    "tigrinya-er-abegede", "tigrinya-et", "tigrinya-et-abegede", "to", "top",
    "trad-chinese-formal", "trad-chinese-informal", "transform",
    "translate", "translate3d", "translateX", "translateY", "translateZ",
    "transparent", "ultra-condensed", "ultra-expanded", "underline", "unset", "up",
    "upper-alpha", "upper-armenian", "upper-greek", "upper-hexadecimal",
    "upper-latin", "upper-norwegian", "upper-roman", "uppercase", "urdu", "url",
    "var", "vertical", "vertical-text", "visible", "visibleFill", "visiblePainted",
    "visibleStroke", "visual", "w-resize", "wait", "wave", "wider",
    "window", "windowframe", "windowtext", "words", "wrap", "wrap-reverse", "x-large", "x-small", "xor",
    "xx-large", "xx-small"
  ], valueKeywords = keySet(valueKeywords_);

  # 将所有关键字合并成一个列表
  var allWords = documentTypes_.concat(mediaTypes_).concat(mediaFeatures_).concat(mediaValueKeywords_)
    .concat(propertyKeywords_).concat(nonStandardPropertyKeywords_).concat(colorKeywords_)
    .concat(valueKeywords_);
  # 在 CodeMirror 中注册 CSS 的关键字
  CodeMirror.registerHelper("hintWords", "css", allWords);

  # 定义处理 CSS 注释的函数
  function tokenCComment(stream, state) {
    var maybeEnd = false, ch;
    while ((ch = stream.next()) != null) {
      if (maybeEnd && ch == "/") {
        state.tokenize = null;
        break;
      }
      maybeEnd = (ch == "*");
    }
    return ["comment", "comment"];
  }

  # 定义 CSS 的 MIME 类型
  CodeMirror.defineMIME("text/css", {
    documentTypes: documentTypes,
    mediaTypes: mediaTypes,  # 媒体类型
    mediaFeatures: mediaFeatures,  # 媒体特性
    mediaValueKeywords: mediaValueKeywords,  # 媒体值关键词
    propertyKeywords: propertyKeywords,  # 属性关键词
    nonStandardPropertyKeywords: nonStandardPropertyKeywords,  # 非标准属性关键词
    fontProperties: fontProperties,  # 字体属性
    counterDescriptors: counterDescriptors,  # 计数器描述符
    colorKeywords: colorKeywords,  # 颜色关键词
    valueKeywords: valueKeywords,  # 值关键词
    tokenHooks: {  # 标记钩子
      "/": function(stream, state) {  # 当遇到 "/" 时的处理函数
        if (!stream.eat("*")) return false;  # 如果下一个字符不是 "*"，则返回 false
        state.tokenize = tokenCComment;  # 设置状态的标记为 tokenCComment
        return tokenCComment(stream, state);  # 调用 tokenCComment 处理函数
      }
    },
    name: "css"  # 名称为 "css"
  });

  CodeMirror.defineMIME("text/x-scss", {  # 定义 MIME 类型为 "text/x-scss"
    mediaTypes: mediaTypes,  # 媒体类型
    mediaFeatures: mediaFeatures,  # 媒体特性
    mediaValueKeywords: mediaValueKeywords,  # 媒体值关键词
    propertyKeywords: propertyKeywords,  # 属性关键词
    nonStandardPropertyKeywords: nonStandardPropertyKeywords,  # 非标准属性关键词
    colorKeywords: colorKeywords,  # 颜色关键词
    valueKeywords: valueKeywords,  # 值关键词
    fontProperties: fontProperties,  # 字体属性
    allowNested: true,  # 允许嵌套
    lineComment: "//",  # 行注释为 "//"
    tokenHooks: {  # 标记钩子
      "/": function(stream, state) {  # 当遇到 "/" 时的处理函数
        if (stream.eat("/")) {  # 如果下一个字符是 "/"
          stream.skipToEnd();  # 跳过直到行尾
          return ["comment", "comment"];  # 返回注释类型
        } else if (stream.eat("*")) {  # 如果下一个字符是 "*"
          state.tokenize = tokenCComment;  # 设置状态的标记为 tokenCComment
          return tokenCComment(stream, state);  # 调用 tokenCComment 处理函数
        } else {  # 其他情况
          return ["operator", "operator"];  # 返回操作符类型
        }
      },
      ":": function(stream) {  # 当遇到 ":" 时的处理函数
        if (stream.match(/\s*\{/, false))  # 如果匹配到空格和 "{"
          return [null, null]  # 返回空
        return false;  # 返回 false
      },
      "$": function(stream) {  # 当遇到 "$" 时的处理函数
        stream.match(/^[\w-]+/);  # 匹配字���、数字、下划线、连字符
        if (stream.match(/^\s*:/, false))  # 如果匹配到空格和 ":"
          return ["variable-2", "variable-definition"];  # 返回变量定义类型
        return ["variable-2", "variable"];  # 返回变量类型
      },
      "#": function(stream) {  # 当遇到 "#" 时的处理函数
        if (!stream.eat("{")) return false;  # 如果下一个字符不是 "{"
        return [null, "interpolation"];  # 返回空和插值类型
      }
    },
    name: "css",  # 名称为 "css"
    helperType: "scss"  # 帮助类型为 "scss"
  });

  CodeMirror.defineMIME("text/x-less", {  # 定义 MIME 类型为 "text/x-less"
    mediaTypes: mediaTypes,  # 媒体类型
    mediaFeatures: mediaFeatures,  # 媒体特性
    mediaValueKeywords: mediaValueKeywords,  // 媒体值关键词
    propertyKeywords: propertyKeywords,  // 属性关键词
    nonStandardPropertyKeywords: nonStandardPropertyKeywords,  // 非标准属性关键词
    colorKeywords: colorKeywords,  // 颜色关键词
    valueKeywords: valueKeywords,  // 值关键词
    fontProperties: fontProperties,  // 字体属性
    allowNested: true,  // 允许嵌套
    lineComment: "//",  // 单行注释符号
    tokenHooks: {  // 标记钩子
      "/": function(stream, state) {  // 当遇到 "/" 时执行的函数
        if (stream.eat("/")) {  // 如果下一个字符是 "/"，则为单行注释
          stream.skipToEnd();  // 跳过注释内容
          return ["comment", "comment"];  // 返回注释类型
        } else if (stream.eat("*")) {  // 如果下一个字符是 "*"，则为多行注释
          state.tokenize = tokenCComment;  // 设置状态的标记为多行注释
          return tokenCComment(stream, state);  // 执行多行注释的函数
        } else {
          return ["operator", "operator"];  // 返回操作符类型
        }
      },
      "@": function(stream) {  // 当遇到 "@" 时执行的函数
        if (stream.eat("{")) return [null, "interpolation"];  // 如果下一个字符是 "{"，则为插值
        if (stream.match(/^(charset|document|font-face|import|(-(moz|ms|o|webkit)-)?keyframes|media|namespace|page|supports)\b/i, false)) return false;  // 匹配关键字
        stream.eatWhile(/[\w\\\-]/);  // 匹配字母、数字、下划线和连字符
        if (stream.match(/^\s*:/, false))  // 如果匹配到冒号
          return ["variable-2", "variable-definition"];  // 返回变量定义类型
        return ["variable-2", "variable"];  // 返回变量类型
      },
      "&": function() {  // 当遇到 "&" 时执行的函数
        return ["atom", "atom"];  // 返回原子类型
      }
    },
    name: "css",  // 设置名称为 CSS
    helperType: "less"  // 设置帮助类型为 less
  });

  CodeMirror.defineMIME("text/x-gss", {  // 定义 MIME 类型为 text/x-gss
    documentTypes: documentTypes,  // 文档类型
    mediaTypes: mediaTypes,  // 媒体类型
    mediaFeatures: mediaFeatures,  // 媒体特性
    propertyKeywords: propertyKeywords,  // 属性关键词
    nonStandardPropertyKeywords: nonStandardPropertyKeywords,  // 非标准属性关键词
    fontProperties: fontProperties,  // 字体属性
    counterDescriptors: counterDescriptors,  // 计数器描述
    colorKeywords: colorKeywords,  // 颜��关键词
    valueKeywords: valueKeywords,  // 值关键词
    supportsAtComponent: true,  // 支持 at 组件
    tokenHooks: {  // 标记钩子
      "/": function(stream, state) {  // 当遇到 "/" 时执行的函数
        if (!stream.eat("*")) return false;  // 如果下一个字符不是 "*"，则返回 false
        state.tokenize = tokenCComment;  // 设置状态的标记为多行注释
        return tokenCComment(stream, state);  // 执行多行注释的函数
      }
    },
    name: "css",  // 设置名称为 CSS
    helperType: "gss"  // 设置帮助类型为 gss
  });
// 定义一个匿名函数，传入 CodeMirror 对象
(function(mod) {
  // 如果是 CommonJS 环境，使用 require 引入 CodeMirror
  if (typeof exports == "object" && typeof module == "object") 
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境，使用 define 引入 CodeMirror
  else if (typeof define == "function" && define.amd) 
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境，直接使用 CodeMirror 对象
  else 
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义 go 语言的 CodeMirror 模式
  CodeMirror.defineMode("go", function(config) {
    // 获取缩进单位
    var indentUnit = config.indentUnit;

    // 定义关键字集合
    var keywords = {
      "break":true, "case":true, "chan":true, "const":true, "continue":true,
      "default":true, "defer":true, "else":true, "fallthrough":true, "for":true,
      "func":true, "go":true, "goto":true, "if":true, "import":true,
      "interface":true, "map":true, "package":true, "range":true, "return":true,
      "select":true, "struct":true, "switch":true, "type":true, "var":true,
      "bool":true, "byte":true, "complex64":true, "complex128":true,
      "float32":true, "float64":true, "int8":true, "int16":true, "int32":true,
      "int64":true, "string":true, "uint8":true, "uint16":true, "uint32":true,
      "uint64":true, "int":true, "uint":true, "uintptr":true, "error": true,
      "rune":true
    };

    // 定义原子集合
    var atoms = {
      "true":true, "false":true, "iota":true, "nil":true, "append":true,
      "cap":true, "close":true, "complex":true, "copy":true, "delete":true, "imag":true,
      "len":true, "make":true, "new":true, "panic":true, "print":true,
      "println":true, "real":true, "recover":true
    };

    // 定义操作符字符集合
    var isOperatorChar = /[+\-*&^%:=<>!|\/]/;

    var curPunc;

    // 定义基本的 token 处理函数
    function tokenBase(stream, state) {
      var ch = stream.next();
      // 如果是引号，进入字符串 token 处理函数
      if (ch == '"' || ch == "'" || ch == "`") {
        state.tokenize = tokenString(ch);
        return state.tokenize(stream, state);
      }
    # 如果字符是数字或者小数点
    if (/[\d\.]/.test(ch)) {
      # 如果字符是小数点
      if (ch == ".") {
        # 匹配小数点后面的数字，包括科学计数法
        stream.match(/^[0-9]+([eE][\-+]?[0-9]+)?/);
      } else if (ch == "0") {
        # 如果字符是0，匹配16进制或者8进制数字
        stream.match(/^[xX][0-9a-fA-F]+/) || stream.match(/^0[0-7]+/);
      } else {
        # 匹配普通的数字，包括小数和科学计数法
        stream.match(/^[0-9]*\.?[0-9]*([eE][\-+]?[0-9]+)?/);
      }
      # 返回数字类型
      return "number";
    }
    # 如果字符是特殊符号
    if (/[\[\]{}\(\),;\:\.]/.test(ch)) {
      # 记录当前的特殊符号
      curPunc = ch;
      # 返回空
      return null;
    }
    # 如果字符是斜杠
    if (ch == "/") {
      # 如果斜杠后面是星号，进入注释状态
      if (stream.eat("*")) {
        state.tokenize = tokenComment;
        return tokenComment(stream, state);
      }
      # 如果斜杠后面是斜杠，跳过整行
      if (stream.eat("/")) {
        stream.skipToEnd();
        return "comment";
      }
    }
    # 如果字符是操作符
    if (isOperatorChar.test(ch)) {
      # 匹配整个操作符
      stream.eatWhile(isOperatorChar);
      # 返回操作符类型
      return "operator";
    }
    # 匹配变量名
    stream.eatWhile(/[\w\$_\xa1-\uffff]/);
    # 获取当前匹配的字符串
    var cur = stream.current();
    # 如果是关键字，返回关键字类型
    if (keywords.propertyIsEnumerable(cur)) {
      if (cur == "case" || cur == "default") curPunc = "case";
      return "keyword";
    }
    # 如果是原子类型，返回原子类型
    if (atoms.propertyIsEnumerable(cur)) return "atom";
    # 否则返回变量类型
    return "variable";
  }

  # 处理字符串的函数
  function tokenString(quote) {
    return function(stream, state) {
      var escaped = false, next, end = false;
      while ((next = stream.next()) != null) {
        if (next == quote && !escaped) {end = true; break;}
        escaped = !escaped && quote != "`" && next == "\\";
      }
      if (end || !(escaped || quote == "`"))
        state.tokenize = tokenBase;
      return "string";
    };
  }

  # 处理注释的函数
  function tokenComment(stream, state) {
    var maybeEnd = false, ch;
    while (ch = stream.next()) {
      if (ch == "/" && maybeEnd) {
        state.tokenize = tokenBase;
        break;
      }
      maybeEnd = (ch == "*");
    }
    return "comment";
  }

  # 上下文对象的构造函数
  function Context(indented, column, type, align, prev) {
    this.indented = indented;
    this.column = column;
    this.type = type;
    this.align = align;
    this.prev = prev;
  }
  # 压入新的上下文
  function pushContext(state, col, type) {
  // 返回一个新的上下文对象，用于表示代码的上下文环境
  return state.context = new Context(state.indented, col, type, null, state.context);
  }
  // 弹出当前上下文对象，返回上一个上下文对象
  function popContext(state) {
    if (!state.context.prev) return;
    var t = state.context.type;
    if (t == ")" || t == "]" || t == "}")
      state.indented = state.context.indented;
    return state.context = state.context.prev;
  }

  // 接口

  return {
    // 初始化编辑器状态
    startState: function(basecolumn) {
      return {
        tokenize: null,
        context: new Context((basecolumn || 0) - indentUnit, 0, "top", false),
        indented: 0,
        startOfLine: true
      };
    },

    // 对输入的流进行标记化处理
    token: function(stream, state) {
      var ctx = state.context;
      if (stream.sol()) {
        if (ctx.align == null) ctx.align = false;
        state.indented = stream.indentation();
        state.startOfLine = true;
        if (ctx.type == "case") ctx.type = "}";
      }
      if (stream.eatSpace()) return null;
      curPunc = null;
      // 获取标记的样式
      var style = (state.tokenize || tokenBase)(stream, state);
      if (style == "comment") return style;
      if (ctx.align == null) ctx.align = true;

      // 根据不同的标点符号进行上下文的推入和弹出
      if (curPunc == "{") pushContext(state, stream.column(), "}");
      else if (curPunc == "[") pushContext(state, stream.column(), "]");
      else if (curPunc == "(") pushContext(state, stream.column(), ")");
      else if (curPunc == "case") ctx.type = "case";
      else if (curPunc == "}" && ctx.type == "}") popContext(state);
      else if (curPunc == ctx.type) popContext(state);
      state.startOfLine = false;
      return style;
    },
    # 定义一个函数，用于处理缩进
    indent: function(state, textAfter) {
      # 如果当前状态不是在基本标记或者不是空的，则返回 CodeMirror.Pass
      if (state.tokenize != tokenBase && state.tokenize != null) return CodeMirror.Pass;
      # 获取当前上下文和文本后的第一个字符
      var ctx = state.context, firstChar = textAfter && textAfter.charAt(0);
      # 如果上下文类型是 "case" 并且文本后的字符是 "case" 或 "default"，则修改上下文类型为 "}"
      if (ctx.type == "case" && /^(?:case|default)\b/.test(textAfter)) {
        state.context.type = "}";
        return ctx.indented;
      }
      # 判断是否是闭合字符
      var closing = firstChar == ctx.type;
      # 如果上下文有对齐属性，则返回上下文列数加上（如果是闭合字符则加0，否则加1）
      if (ctx.align) return ctx.column + (closing ? 0 : 1);
      # 否则返回上下文缩进加上（如果是闭合字符则加0，否则加缩进单位）
      else return ctx.indented + (closing ? 0 : indentUnit);
    },

    # 定义电气字符
    electricChars: "{}):",
    # 定义闭合括号
    closeBrackets: "()[]{}''\"\"``",
    # 定义折叠方式
    fold: "brace",
    # 定义块注释的起始标记
    blockCommentStart: "/*",
    # 定义块注释的结束标记
    blockCommentEnd: "*/",
    # 定义行注释的标记
    lineComment: "//"
  };
// 定义 MIME 类型为 "text/x-go" 的 CodeMirror 模式
CodeMirror.defineMIME("text/x-go", "go");



// 定义模式 "htmlembedded"，用于处理嵌入式 HTML
CodeMirror.defineMode("htmlembedded", function(config, parserConfig) {
  // 获取关闭注释的字符串，如果没有指定则使用默认值 "--%>"
  var closeComment = parserConfig.closeComment || "--%>"
  // 使用 CodeMirror.multiplexingMode 创建混合模式，处理嵌入式 HTML
  return CodeMirror.multiplexingMode(CodeMirror.getMode(config, "htmlmixed"), {
    // 设置开启注释的字符串，如果没有指定则使用默认值 "<%--"
    open: parserConfig.openComment || "<%--",
    // 设置关闭注释的字符串
    close: closeComment,
    // 设置注释的样式
    delimStyle: "comment",
    // 设置模式，处理注释内容
    mode: {token: function(stream) {
      // 跳过直到遇到关闭注释的字符串，或者跳到行尾
      stream.skipTo(closeComment) || stream.skipToEnd()
      // 返回注释的标记
      return "comment"
    }}
  }, {
    // 设置开启嵌入式代码的字符串
    open: parserConfig.open || parserConfig.scriptStartRegex || "<%",
    // 设置关闭嵌入式代码的字符串
    close: parserConfig.close || parserConfig.scriptEndRegex || "%>",
    // 设置嵌入式代码的模式
    mode: CodeMirror.getMode(config, parserConfig.scriptingModeSpec)
  });
}, "htmlmixed");



// 定义 MIME 类型为 "application/x-ejs" 的 CodeMirror 模式，使用 JavaScript 作为嵌入式脚本语言
CodeMirror.defineMIME("application/x-ejs", {name: "htmlembedded", scriptingModeSpec:"javascript"});
// 定义 MIME 类型为 "application/x-aspx" 的 CodeMirror 模式，使用 C# 作为嵌入式脚本语言
CodeMirror.defineMIME("application/x-aspx", {name: "htmlembedded", scriptingModeSpec:"text/x-csharp"});
// 定义 MIME 类型为 "application/x-jsp" 的 CodeMirror 模式，使用 Java 作为嵌入式脚本语言
CodeMirror.defineMIME("application/x-jsp", {name: "htmlembedded", scriptingModeSpec:"text/x-java"});
// 定义 MIME 类型为 "application/x-erb" 的 CodeMirror 模式，使用 Ruby 作为嵌入式脚本语言
CodeMirror.defineMIME("application/x-erb", {name: "htmlembedded", scriptingModeSpec:"ruby"});
(function(mod) {
  if (typeof exports == "object" && typeof module == "object") // CommonJS
    mod(require("../../lib/codemirror"), require("../xml/xml"), require("../javascript/javascript"), require("../css/css"));
  else if (typeof define == "function" && define.amd) // AMD
    define(["../../lib/codemirror", "../xml/xml", "../javascript/javascript", "../css/css"], mod);
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  var defaultTags = {
    script: [
      ["lang", /(javascript|babel)/i, "javascript"], // 定义 script 标签的 lang 属性匹配规则和样式
      ["type", /^(?:text|application)\/(?:x-)?(?:java|ecma)script$|^module$|^$/i, "javascript"], // 定义 script 标签的 type 属性匹配规则和样式
      ["type", /./, "text/plain"], // 定义 script 标签的 type 属性匹配规则和样式
      [null, null, "javascript"] // 默认样式
    ],
    style:  [
      ["lang", /^css$/i, "css"], // 定义 style 标签的 lang 属性匹配规则和样式
      ["type", /^(text\/)?(x-)?(stylesheet|css)$/i, "css"], // 定义 style 标签的 type 属性匹配规则和样式
      ["type", /./, "text/plain"], // 定义 style 标签的 type 属性匹配规则和样式
      [null, null, "css"] // 默认样式
    ]
  };

  function maybeBackup(stream, pat, style) {
    var cur = stream.current(), close = cur.search(pat);
    if (close > -1) {
      stream.backUp(cur.length - close);
    } else if (cur.match(/<\/?$/)) {
      stream.backUp(cur.length);
      if (!stream.match(pat, false)) stream.match(cur);
    }
    return style;
  }

  var attrRegexpCache = {};
  function getAttrRegexp(attr) {
    var regexp = attrRegexpCache[attr];
    if (regexp) return regexp;
    return attrRegexpCache[attr] = new RegExp("\\s+" + attr + "\\s*=\\s*('|\")?([^'\"]+)('|\")?\\s*"); // 缓存属性匹配规则
  }

  function getAttrValue(text, attr) {
    var match = text.match(getAttrRegexp(attr))
    return match ? /^\s*(.*?)\s*$/.exec(match[2])[1] : "" // 获取属性值
  }

  function getTagRegexp(tagName, anchored) {
    return new RegExp((anchored ? "^" : "") + "<\/\s*" + tagName + "\s*>", "i"); // 获取标签匹配规则
  }

  function addTags(from, to) {
    for (var tag in from) {
      var dest = to[tag] || (to[tag] = []);
      var source = from[tag];
      for (var i = source.length - 1; i >= 0; i--)
        dest.unshift(source[i]) // 添加标签匹配规则到目标对象
  // 寻找匹配的模式
  function findMatchingMode(tagInfo, tagText) {
    // 遍历标签信息数组
    for (var i = 0; i < tagInfo.length; i++) {
      // 获取当前标签信息
      var spec = tagInfo[i];
      // 如果标签信息为空或者匹配成功，则返回对应的模式
      if (!spec[0] || spec[1].test(getAttrValue(tagText, spec[0]))) return spec[2];
    }
  }

  // 定义 HTML 混合模式
  CodeMirror.defineMode("htmlmixed", function (config, parserConfig) {
    // 获取 HTML 模式
    var htmlMode = CodeMirror.getMode(config, {
      name: "xml",
      htmlMode: true,
      multilineTagIndentFactor: parserConfig.multilineTagIndentFactor,
      multilineTagIndentPastTag: parserConfig.multilineTagIndentPastTag
    });

    // 初始化标签对象
    var tags = {};
    // 获取配置中的标签和脚本类型
    var configTags = parserConfig && parserConfig.tags, configScript = parserConfig && parserConfig.scriptTypes;
    // 添加默认标签
    addTags(defaultTags, tags);
    // 如果有配置标签，则添加配置标签
    if (configTags) addTags(configTags, tags);
    // 如果有配置脚本类型，则遍历添加到标签对象中
    if (configScript) for (var i = configScript.length - 1; i >= 0; i--)
      tags.script.unshift(["type", configScript[i].matches, configScript[i].mode])
  }
    function html(stream, state) {
      // 获取当前标记的样式
      var style = htmlMode.token(stream, state.htmlState), tag = /\btag\b/.test(style), tagName
      // 如果当前是标签，并且当前字符不是 <、>、空格或斜杠，并且存在标签名
      if (tag && !/[<>\s\/]/.test(stream.current()) &&
          (tagName = state.htmlState.tagName && state.htmlState.tagName.toLowerCase()) &&
          tags.hasOwnProperty(tagName)) {
        // 设置当前标签状态
        state.inTag = tagName + " "
      } else if (state.inTag && tag && />$/.test(stream.current())) {
        // 获取当前标签的模式
        var inTag = /^([\S]+) (.*)/.exec(state.inTag)
        state.inTag = null
        var modeSpec = stream.current() == ">" && findMatchingMode(tags[inTag[1]], inTag[2])
        var mode = CodeMirror.getMode(config, modeSpec)
        var endTagA = getTagRegexp(inTag[1], true), endTag = getTagRegexp(inTag[1], false);
        // 设置当前标签的 token 函数
        state.token = function (stream, state) {
          if (stream.match(endTagA, false)) {
            state.token = html;
            state.localState = state.localMode = null;
            return null;
          }
          return maybeBackup(stream, endTag, state.localMode.token(stream, state.localState));
        };
        state.localMode = mode;
        state.localState = CodeMirror.startState(mode, htmlMode.indent(state.htmlState, "", ""));
      } else if (state.inTag) {
        state.inTag += stream.current()
        if (stream.eol()) state.inTag += " "
      }
      return style;
    };
    # 返回一个对象，包含了开始状态的函数和其他方法
    return {
      # 开始状态的函数
      startState: function () {
        # 使用 htmlMode 的开始状态作为基础状态
        var state = CodeMirror.startState(htmlMode);
        # 返回一个对象，包含 token、inTag、localMode、localState 和 htmlState
        return {token: html, inTag: null, localMode: null, localState: null, htmlState: state};
      },

      # 复制状态的函数
      copyState: function (state) {
        var local;
        # 如果存在 localState，则复制 localMode 和 localState
        if (state.localState) {
          local = CodeMirror.copyState(state.localMode, state.localState);
        }
        # 返回一个对象，包含 token、inTag、localMode、localState 和 htmlState
        return {token: state.token, inTag: state.inTag,
                localMode: state.localMode, localState: local,
                htmlState: CodeMirror.copyState(htmlMode, state.htmlState)};
      },

      # 返回 token 的函数
      token: function (stream, state) {
        # 调用 state 中的 token 方法
        return state.token(stream, state);
      },

      # 返回缩进的函数
      indent: function (state, textAfter, line) {
        # 如果不存在 localMode 或者 textAfter 以 </ 开头，则调用 htmlMode 的缩进方法
        if (!state.localMode || /^\s*<\//.test(textAfter))
          return htmlMode.indent(state.htmlState, textAfter, line);
        # 如果存在 localMode 并且 localMode 有 indent 方法，则调用 localMode 的缩进方法
        else if (state.localMode.indent)
          return state.localMode.indent(state.localState, textAfter, line);
        # 否则返回 CodeMirror.Pass
        else
          return CodeMirror.Pass;
      },

      # 返回内部模式的函数
      innerMode: function (state) {
        # 返回一个对象，包含 localState 或 htmlState 和 localMode 或 htmlMode
        return {state: state.localState || state.htmlState, mode: state.localMode || htmlMode};
      }
    };
  }, "xml", "javascript", "css");

  # 定义 MIME 类型为 text/html 的语法为 htmlmixed
  CodeMirror.defineMIME("text/html", "htmlmixed");
// 定义一个匿名函数，传入一个 mod 参数
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror")); // 使用 mod 函数引入 codemirror 模块
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror"], mod); // 使用 define 函数引入 codemirror 模块
  // 如果是普通的浏览器环境
  else
    mod(CodeMirror); // 使用 mod 函数引入 CodeMirror 对象
})(function(CodeMirror) {
  "use strict"; // 开启严格模式

  // 定义 JavaScript 语法模式
  CodeMirror.defineMode("javascript", function(config, parserConfig) {
    var indentUnit = config.indentUnit; // 缩进单位
    var statementIndent = parserConfig.statementIndent; // 语句缩进
    var jsonldMode = parserConfig.jsonld; // JSON-LD 模式
    var jsonMode = parserConfig.json || jsonldMode; // JSON 模式
    var isTS = parserConfig.typescript; // 是否是 TypeScript
    var wordRE = parserConfig.wordCharacters || /[\w$\xa1-\uffff]/; // 单词正则表达式

    // Tokenizer

    // 关键字定义
    var keywords = function(){
      function kw(type) {return {type: type, style: "keyword"};}
      var A = kw("keyword a"), B = kw("keyword b"), C = kw("keyword c"), D = kw("keyword d");
      var operator = kw("operator"), atom = {type: "atom", style: "atom"};

      return {
        "if": kw("if"), "while": A, "with": A, "else": B, "do": B, "try": B, "finally": B,
        "return": D, "break": D, "continue": D, "new": kw("new"), "delete": C, "void": C, "throw": C,
        "debugger": kw("debugger"), "var": kw("var"), "const": kw("var"), "let": kw("var"),
        "function": kw("function"), "catch": kw("catch"),
        "for": kw("for"), "switch": kw("switch"), "case": kw("case"), "default": kw("default"),
        "in": operator, "typeof": operator, "instanceof": operator,
        "true": atom, "false": atom, "null": atom, "undefined": atom, "NaN": atom, "Infinity": atom,
        "this": kw("this"), "class": kw("class"), "super": kw("atom"),
        "yield": C, "export": kw("export"), "import": kw("import"), "extends": C,
        "await": C
  // 匿名函数，用于创建一个作用域
  };
  // 判断是否为操作符字符的正则表达式
  var isOperatorChar = /[+\-*&%=<>!?|~^@]/;
  // 判断是否为 JSON-LD 关键字的正则表达式
  var isJsonldKeyword = /^@(context|id|value|language|type|container|list|set|reverse|index|base|vocab|graph)"/;

  // 读取正则表达式
  function readRegexp(stream) {
    var escaped = false, next, inSet = false;
    while ((next = stream.next()) != null) {
      if (!escaped) {
        if (next == "/" && !inSet) return;
        if (next == "[") inSet = true;
        else if (inSet && next == "]") inSet = false;
      }
      escaped = !escaped && next == "\\";
    }
  }

  // 用作临时变量，用于在不创建大量对象的情况下传递多个值
  var type, content;
  // 返回 token
  function ret(tp, style, cont) {
    type = tp; content = cont;
    return style;
  }
  // token 的基本处理函数
  function tokenBase(stream, state) {
    var ch = stream.next();
    if (ch == '"' || ch == "'") {
      state.tokenize = tokenString(ch);
      return state.tokenize(stream, state);
    } else if (ch == "." && stream.match(/^\d[\d_]*(?:[eE][+\-]?[\d_]+)?/)) {
      return ret("number", "number");
    } else if (ch == "." && stream.match("..")) {
      return ret("spread", "meta");
    } else if (/[\[\]{}\(\),;\:\.]/.test(ch)) {
      return ret(ch);
    } else if (ch == "=" && stream.eat(">")) {
      return ret("=>", "operator");
    } else if (ch == "0" && stream.match(/^(?:x[\dA-Fa-f_]+|o[0-7_]+|b[01_]+)n?/)) {
      return ret("number", "number");
    } else if (/\d/.test(ch)) {
      stream.match(/^[\d_]*(?:n|(?:\.[\d_]*)?(?:[eE][+\-]?[\d_]+)?)?/);
      return ret("number", "number");
    } else if (ch == "/") {
      // 如果当前字符是斜杠
      if (stream.eat("*")) {
        // 如果下一个字符是星号，表示注释开始
        state.tokenize = tokenComment;
        return tokenComment(stream, state);
      } else if (stream.eat("/")) {
        // 如果下一个字符是斜杠，表示单行注释
        stream.skipToEnd();
        return ret("comment", "comment");
      } else if (expressionAllowed(stream, state, 1)) {
        // 如果允许表达式，读取正则表达式
        readRegexp(stream);
        stream.match(/^\b(([gimyus])(?![gimyus]*\2))+\b/);
        return ret("regexp", "string-2");
      } else {
        // 否则，可能是赋值操作符
        stream.eat("=");
        return ret("operator", "operator", stream.current());
      }
    } else if (ch == "`") {
      // 如果当前字符是反引号，表示模板字符串
      state.tokenize = tokenQuasi;
      return tokenQuasi(stream, state);
    } else if (ch == "#" && stream.peek() == "!") {
      // 如果当前字符是井号，下一个字符是感叹号，表示元信息
      stream.skipToEnd();
      return ret("meta", "meta");
    } else if (ch == "#" && stream.eatWhile(wordRE)) {
      // 如果当前字符是井号，后面是单词字符，表示变量属性
      return ret("variable", "property")
    } else if (ch == "<" && stream.match("!--") ||
               (ch == "-" && stream.match("->") && !/\S/.test(stream.string.slice(0, stream.start)))) {
      // 如果当前字符是小于号，并且后面是注释开始标记，或者当前字符是减号，并且后面是注释结束标记并且后面没有非空白字符
      stream.skipToEnd()
      return ret("comment", "comment")
    } else if (isOperatorChar.test(ch)) {
      // 如果当前字符是操作符字符
      if (ch != ">" || !state.lexical || state.lexical.type != ">") {
        if (stream.eat("=")) {
          if (ch == "!" || ch == "=") stream.eat("=")
        } else if (/[<>*+\-]/.test(ch)) {
          stream.eat(ch)
          if (ch == ">") stream.eat(ch)
        }
      }
      if (ch == "?" && stream.eat(".")) return ret(".")
      return ret("operator", "operator", stream.current());
    } else if (wordRE.test(ch)) {
      // 如果当前字符是单词字符
      stream.eatWhile(wordRE);
      var word = stream.current()
      if (state.lastType != ".") {
        if (keywords.propertyIsEnumerable(word)) {
          var kw = keywords[word]
          return ret(kw.type, kw.style, word)
        }
        if (word == "async" && stream.match(/^(\s|\/\*.*?\*\/)*[\[\(\w]/, false))
          return ret("async", "keyword", word)
      }
      return ret("variable", "variable", word)
  }
}

function tokenString(quote) {
  // 返回一个函数，用于处理字符串类型的 token
  return function(stream, state) {
    var escaped = false, next;
    // 如果处于 JSON-LD 模式，并且下一个字符是 "@"，并且匹配 JSON-LD 关键字，则返回 "jsonld-keyword" 类型的 token
    if (jsonldMode && stream.peek() == "@" && stream.match(isJsonldKeyword)){
      state.tokenize = tokenBase;
      return ret("jsonld-keyword", "meta");
    }
    // 遍历字符串，处理转义字符
    while ((next = stream.next()) != null) {
      if (next == quote && !escaped) break;
      escaped = !escaped && next == "\\";
    }
    // 如果没有转义字符，则将状态切换回 tokenBase
    if (!escaped) state.tokenize = tokenBase;
    // 返回字符串类型的 token
    return ret("string", "string");
  };
}

function tokenComment(stream, state) {
  var maybeEnd = false, ch;
  // 处理注释类型的 token
  while (ch = stream.next()) {
    if (ch == "/" && maybeEnd) {
      state.tokenize = tokenBase;
      break;
    }
    maybeEnd = (ch == "*");
  }
  // 返回注释类型的 token
  return ret("comment", "comment");
}

function tokenQuasi(stream, state) {
  var escaped = false, next;
  // 处理模板字符串类型的 token
  while ((next = stream.next()) != null) {
    if (!escaped && (next == "`" || next == "$" && stream.eat("{"))) {
      state.tokenize = tokenBase;
      break;
    }
    escaped = !escaped && next == "\\";
  }
  // 返回模板字符串类型的 token
  return ret("quasi", "string-2", stream.current());
}

var brackets = "([{}])";
// 这是一个粗糙的前瞻技巧，用于尝试注意到我们在实际命中箭头标记之前正在解析一个箭头函数的参数模式。
// 如果箭头标记在与参数相同的行上，并且之间没有奇怪的噪音（注释），则它有效。
// 如果找不到箭头标记，则不将参数声明为箭头体的本地变量。
function findFatArrow(stream, state) {
  if (state.fatArrowAt) state.fatArrowAt = null;
  var arrow = stream.string.indexOf("=>", stream.start);
  if (arrow < 0) return;
    // 如果是 TypeScript，则尝试跳过参数后的返回类型声明
    if (isTS) { 
      // 在箭头符号之前，尝试匹配跳过 TypeScript 返回类型声明
      var m = /:\s*(?:\w+(?:<[^>]*>|\[\])?|\{[^}]*\})\s*$/.exec(stream.string.slice(stream.start, arrow))
      if (m) arrow = m.index
    }

    // 初始化深度和标记是否有内容
    var depth = 0, sawSomething = false;
    // 从箭头符号位置开始向前遍历
    for (var pos = arrow - 1; pos >= 0; --pos) {
      var ch = stream.string.charAt(pos);
      var bracket = brackets.indexOf(ch);
      // 如果是括号类型的字符
      if (bracket >= 0 && bracket < 3) {
        // 如果深度为 0，则跳出循环
        if (!depth) { ++pos; break; }
        // 如果深度减少到 0，则标记有内容，并跳出循环
        if (--depth == 0) { if (ch == "(") sawSomething = true; break; }
      } else if (bracket >= 3 && bracket < 6) {
        // 如果是另一种括号类型的字符，则深度加一
        ++depth;
      } else if (wordRE.test(ch)) {
        // 如果是单词字符，则标记有内容
        sawSomething = true;
      } else if (/["'\/`]/.test(ch)) {
        // 如果是引号或斜杠，则向前查找匹配的引号或斜杠
        for (;; --pos) {
          if (pos == 0) return
          var next = stream.string.charAt(pos - 1)
          if (next == ch && stream.string.charAt(pos - 2) != "\\") { pos--; break }
        }
      } else if (sawSomething && !depth) {
        // 如果已经标记有内容且深度为 0，则跳出循环
        ++pos;
        break;
      }
    }
    // 如果标记有内容且深度为 0，则记录箭头符号位置
    if (sawSomething && !depth) state.fatArrowAt = pos;
  }

  // Parser

  // 定义原子类型集合
  var atomicTypes = {"atom": true, "number": true, "variable": true, "string": true, "regexp": true, "this": true, "jsonld-keyword": true};

  // JSLexical 类的构造函数
  function JSLexical(indented, column, type, align, prev, info) {
    this.indented = indented;
    this.column = column;
    this.type = type;
    this.prev = prev;
    this.info = info;
    if (align != null) this.align = align;
  }

  // 判断变量是否在作用域内
  function inScope(state, varname) {
    for (var v = state.localVars; v; v = v.next)
      if (v.name == varname) return true;
    for (var cx = state.context; cx; cx = cx.prev) {
      for (var v = cx.vars; v; v = v.next)
        if (v.name == varname) return true;
    }
  }

  // 解析 JavaScript
  function parseJS(state, style, type, content, stream) {
    var cc = state.cc;
    // 将上下文传递给组合器
    // （比在每次调用时都创建闭包更节省资源）
    # 设置 cx 对象的属性值
    cx.state = state; cx.stream = stream; cx.marked = null, cx.cc = cc; cx.style = style;

    # 如果 state.lexical 对象中不包含 align 属性，则设置 align 为 true
    if (!state.lexical.hasOwnProperty("align"))
      state.lexical.align = true;

    # 进入循环，直到条件为 false
    while(true) {
      # 从 cc 数组中取出一个元素，如果为空则根据 jsonMode 的值选择 expression 或者 statement
      var combinator = cc.length ? cc.pop() : jsonMode ? expression : statement;
      # 如果 combinator 函数返回 true，则执行内部逻辑
      if (combinator(type, content)) {
        # 循环执行 cc 数组中的函数，直到遇到非 lex 函数
        while(cc.length && cc[cc.length - 1].lex)
          cc.pop()();
        # 如果 cx.marked 存在，则返回 cx.marked
        if (cx.marked) return cx.marked;
        # 如果 type 为 "variable" 并且在当前作用域内，则返回 "variable-2"，否则返回 style
        if (type == "variable" && inScope(state, content)) return "variable-2";
        return style;
      }
    }
  }

  # Combinator utils

  # 定义 cx 对象
  var cx = {state: null, column: null, marked: null, cc: null};
  # 将参数依次添加到 cx.cc 数组中
  function pass() {
    for (var i = arguments.length - 1; i >= 0; i--) cx.cc.push(arguments[i]);
  }
  # 调用 pass 函数，并返回 true
  function cont() {
    pass.apply(null, arguments);
    return true;
  }
  # 判断 name 是否在 list 中，如果在则返回 true，否则返回 false
  function inList(name, list) {
    for (var v = list; v; v = v.next) if (v.name == name) return true
    return false;
  }
  # 注册变量，设置 cx.marked 为 "def"，并根据作用域情况进行处理
  function register(varname) {
    var state = cx.state;
    cx.marked = "def";
    if (state.context) {
      if (state.lexical.info == "var" && state.context && state.context.block) {
        # 如果是 var 声明且在块级作用域内，则注册为局部变量
        var newContext = registerVarScoped(varname, state.context)
        if (newContext != null) {
          state.context = newContext
          return
        }
      } else if (!inList(varname, state.localVars)) {
        # 如果不在局部变量列表中，则添加到局部变量列表
        state.localVars = new Var(varname, state.localVars)
        return
      }
    }
    # 如果以上条件都不满足，则注册为全局变量
    if (parserConfig.globalVars && !inList(varname, state.globalVars))
      state.globalVars = new Var(varname, state.globalVars)
  }
  # 在特定作用域内注册变量
  function registerVarScoped(varname, context) {
    if (!context) {
      return null
    } else if (context.block) {
      var inner = registerVarScoped(varname, context.prev)
      if (!inner) return null
      if (inner == context.prev) return context
      return new Context(inner, context.vars, true)
  // 如果变量名在上下文的变量列表中，则返回上下文
  } else if (inList(varname, context.vars)) {
    return context
  // 否则，创建一个新的上下文，并添加变量名到变量列表中
  } else {
    return new Context(context.prev, new Var(varname, context.vars), false)
  }
}

// 判断是否为修饰符
function isModifier(name) {
  return name == "public" || name == "private" || name == "protected" || name == "abstract" || name == "readonly"
}

// 组合器

// 上下文构造函数
function Context(prev, vars, block) { this.prev = prev; this.vars = vars; this.block = block }
// 变量构造函数
function Var(name, next) { this.name = name; this.next = next }

// 默认变量列表
var defaultVars = new Var("this", new Var("arguments", null))
// 推入上下文
function pushcontext() {
  cx.state.context = new Context(cx.state.context, cx.state.localVars, false)
  cx.state.localVars = defaultVars
}
// 推入块上下文
function pushblockcontext() {
  cx.state.context = new Context(cx.state.context, cx.state.localVars, true)
  cx.state.localVars = null
}
// 弹出上下文
function popcontext() {
  cx.state.localVars = cx.state.context.vars
  cx.state.context = cx.state.context.prev
}
popcontext.lex = true
// 推入词法环境
function pushlex(type, info) {
  var result = function() {
    var state = cx.state, indent = state.indented;
    if (state.lexical.type == "stat") indent = state.lexical.indented;
    else for (var outer = state.lexical; outer && outer.type == ")" && outer.align; outer = outer.prev)
      indent = outer.indented;
    state.lexical = new JSLexical(indent, cx.stream.column(), type, null, state.lexical, info);
  };
  result.lex = true;
  return result;
}
// 弹出词法环境
function poplex() {
  var state = cx.state;
  if (state.lexical.prev) {
    if (state.lexical.type == ")")
      state.indented = state.lexical.indented;
    state.lexical = state.lexical.prev;
  }
}
poplex.lex = true;

// 期望特定类型的词法单元
function expect(wanted) {
  function exp(type) {
    if (type == wanted) return cont();
    else if (wanted == ";" || type == "}" || type == ")" || type == "]") return pass();
    else return cont(exp);
  };
  // 返回表达式
  return exp;
}

function statement(type, value) {
  // 如果类型为 "var"，则返回变量定义的上下文
  if (type == "var") return cont(pushlex("vardef", value), vardef, expect(";"), poplex);
  // 如果类型为 "keyword a"，则返回表达式的上下文
  if (type == "keyword a") return cont(pushlex("form"), parenExpr, statement, poplex);
  // 如果类型为 "keyword b"，则返回语句的上下文
  if (type == "keyword b") return cont(pushlex("form"), statement, poplex);
  // 如果类型为 "keyword d"，则返回可能的表达式的上下文
  if (type == "keyword d") return cx.stream.match(/^\s*$/, false) ? cont() : cont(pushlex("stat"), maybeexpression, expect(";"), poplex);
  // 如果类型为 "debugger"，则返回期望分号的上下文
  if (type == "debugger") return cont(expect(";"));
  // 如果类型为 "{"，则返回代码块的上下文
  if (type == "{") return cont(pushlex("}"), pushblockcontext, block, poplex, popcontext);
  // 如果类型为 ";"，则返回空的上下文
  if (type == ";") return cont();
  // 如果类型为 "if"，则返回 if 语句的上下文
  if (type == "if") {
    if (cx.state.lexical.info == "else" && cx.state.cc[cx.state.cc.length - 1] == poplex)
      cx.state.cc.pop()();
    return cont(pushlex("form"), parenExpr, statement, poplex, maybeelse);
  }
  // 如果类型为 "function"，则返回函数定义的上下文
  if (type == "function") return cont(functiondef);
  // 如果类型为 "for"，则返回 for 循环的上下文
  if (type == "for") return cont(pushlex("form"), forspec, statement, poplex);
  // 如果类型为 "class" 或者（如果是 TypeScript 并且值为 "interface"），则标记为关键字并返回类名的上下文
  if (type == "class" || (isTS && value == "interface")) {
    cx.marked = "keyword"
    return cont(pushlex("form", type == "class" ? type : value), className, poplex)
  }
}
    # 如果类型为"variable"
    if (type == "variable") {
      # 如果是 TypeScript 并且数值为"declare"
      if (isTS && value == "declare") {
        # 标记为关键字
        cx.marked = "keyword"
        # 继续解析语句
        return cont(statement)
      } else if (isTS && (value == "module" || value == "enum" || value == "type") && cx.stream.match(/^\s*\w/, false)) {
        # 标记为关键字
        cx.marked = "keyword"
        # 如果是"enum"，继续解析 enumdef
        if (value == "enum") return cont(enumdef);
        # 如果是"type"，继续解析 typename, 期望"operator"，typeexpr，期望";"
        else if (value == "type") return cont(typename, expect("operator"), typeexpr, expect(";"));
        # 否则，推入"form"，解析 pattern，期望"{"，推入"}"，解析 block，弹出"}"，弹出"form"
        else return cont(pushlex("form"), pattern, expect("{"), pushlex("}"), block, poplex, poplex)
      } else if (isTS && value == "namespace") {
        # 标记为关键字
        cx.marked = "keyword"
        # 推入"form"，解析 expression，解析 statement，弹出"form"
        return cont(pushlex("form"), expression, statement, poplex)
      } else if (isTS && value == "abstract") {
        # 标记为关键字
        cx.marked = "keyword"
        # 继续解析语句
        return cont(statement)
      } else {
        # 推入"stat"，解析 maybelabel
        return cont(pushlex("stat"), maybelabel);
      }
    }
    # 如果类型为"switch"，推入"form"，解析 parenExpr，期望"{"，推入"switch"，推入 block，弹出"}"，弹出"switch"，弹出上下文
    if (type == "switch") return cont(pushlex("form"), parenExpr, expect("{"), pushlex("}"), block, poplex, poplex, popcontext);
    # 如果类型为"case"，继续解析 expression，期望":"
    if (type == "case") return cont(expression, expect(":"));
    # 如果类型为"default"，期望":"
    if (type == "default") return cont(expect(":"));
    # 如果类型为"catch"，推入"form"，推入上下文，可能解析 maybeCatchBinding，解析 statement，弹出"form"，弹出上下文
    if (type == "catch") return cont(pushlex("form"), pushcontext, maybeCatchBinding, statement, poplex, popcontext);
    # 如果类型为"export"，推入"stat"，解析 afterExport，弹出"stat"
    if (type == "export") return cont(pushlex("stat"), afterExport, poplex);
    # 如果类型为"import"，推入"stat"，解析 afterImport，弹出"stat"
    if (type == "import") return cont(pushlex("stat"), afterImport, poplex);
    # 如果类型为"async"，继续解析语句
    if (type == "async") return cont(statement)
    # 如果数值为"@"，解析 expression，解析 statement
    if (value == "@") return cont(expression, statement)
    # 推入"stat"，解析 expression，期望";"，弹出"stat"
    return pass(pushlex("stat"), expression, expect(";"), poplex);
  }
  # 如果可能的捕获绑定
  function maybeCatchBinding(type) {
    # 如果类型为"("，解析 funarg，期望")"
    if (type == "(") return cont(funarg, expect(")"))
  }
  # 表达式
  function expression(type, value) {
    return expressionInner(type, value, false);
  }
  # 不包含逗号的表达式
  function expressionNoComma(type, value) {
    return expressionInner(type, value, true);
  }
  # 括号表达式
  function parenExpr(type) {
    # 如果类型不是"("，跳过
    if (type != "(") return pass()
  # 返回一个包含 pushlex(")") 的函数调用的结果
  return cont(pushlex(")"), maybeexpression, expect(")"), poplex)
  # 结束函数定义

  # 定义函数 expressionInner，接受 type, value, noComma 三个参数
  function expressionInner(type, value, noComma) {
    # 如果箭头函数在当前位置
    if (cx.state.fatArrowAt == cx.stream.start) {
      # 如果 noComma 为真，则调用 arrowBodyNoComma，否则调用 arrowBody
      var body = noComma ? arrowBodyNoComma : arrowBody;
      # 如果 type 为 "("，则返回一个包含 pushcontext, pushlex(")"), commasep(funarg, ")"), poplex, expect("=>"), body, popcontext 的函数调用的结果
      if (type == "(") return cont(pushcontext, pushlex(")"), commasep(funarg, ")"), poplex, expect("=>"), body, popcontext);
      # 如果 type 为 "variable"，则返回一个包含 pushcontext, pattern, expect("=>"), body, popcontext 的函数调用的结果
      else if (type == "variable") return pass(pushcontext, pattern, expect("=>"), body, popcontext);
    }

    # 如果 noComma 为真，则调用 maybeoperatorNoComma，否则调用 maybeoperatorComma
    var maybeop = noComma ? maybeoperatorNoComma : maybeoperatorComma;
    # 如果 atomicTypes 包含 type，则返回一个包含 maybeop 的函数调用的结果
    if (atomicTypes.hasOwnProperty(type)) return cont(maybeop);
    # 如果 type 为 "function"，则返回一个包含 functiondef, maybeop 的函数调用的结果
    if (type == "function") return cont(functiondef, maybeop);
    # 如果 type 为 "class" 或 (isTS 为真 且 value 为 "interface")，则将 cx.marked 设置为 "keyword"，并返回一个包含 pushlex("form"), classExpression, poplex 的函数调用的结果
    if (type == "class" || (isTS && value == "interface")) { cx.marked = "keyword"; return cont(pushlex("form"), classExpression, poplex); }
    # 如果 type 为 "keyword c" 或 type 为 "async"，则返回一个包含 noComma 为真则调用 expressionNoComma，否则调用 expression 的函数调用的结果
    if (type == "keyword c" || type == "async") return cont(noComma ? expressionNoComma : expression);
    # 如果 type 为 "("，则返回一个包含 pushlex(")"), maybeexpression, expect(")"), poplex, maybeop 的函数调用的结果
    if (type == "(") return cont(pushlex(")"), maybeexpression, expect(")"), poplex, maybeop);
    # 如果 type 为 "operator" 或 type 为 "spread"，则返回一个包含 noComma 为真则调用 expressionNoComma，否则调用 expression 的函数调用的结果
    if (type == "operator" || type == "spread") return cont(noComma ? expressionNoComma : expression);
    # 如果 type 为 "["，则返回一个包含 pushlex("]"), arrayLiteral, poplex, maybeop 的函数调用的结果
    if (type == "[") return cont(pushlex("]"), arrayLiteral, poplex, maybeop);
    # 如果 type 为 "{"，则返回一个包含 objprop, "}", null, maybeop 的函数调用的结果
    if (type == "{") return contCommasep(objprop, "}", null, maybeop);
    # 如果 type 为 "quasi"，则返回一个包含 quasi, maybeop 的函数调用的结果
    if (type == "quasi") return pass(quasi, maybeop);
    # 如果 type 为 "new"，则返回一个包含 noComma 的函数调用的结果
    if (type == "new") return cont(maybeTarget(noComma));
    # 如果 type 为 "import"，则返回一个包含 expression 的函数调用的结果
    if (type == "import") return cont(expression);
    # 返回一个空的函数调用的结果
    return cont();
  }

  # 定义函数 maybeexpression，接受 type 一个参数
  function maybeexpression(type) {
    # 如果 type 匹配 /[;\}\)\],]/，则返回一个空的函数调用的结果
    if (type.match(/[;\}\)\],]/)) return pass();
    # 返回一个包含 expression 的函数调用的结果
    return pass(expression);
  }

  # 定义函数 maybeoperatorComma，接受 type, value 两个参数
  function maybeoperatorComma(type, value) {
    # 如果 type 为 ","，则返回一个包含 maybeexpression 的函数调用的结果
    if (type == ",") return cont(maybeexpression);
    # 返回一个包含 maybeoperatorNoComma 的函数调用的结果
    return maybeoperatorNoComma(type, value, false);
  }

  # 定义函数 maybeoperatorNoComma，接受 type, value, noComma 三个参数
  function maybeoperatorNoComma(type, value, noComma) {
    # 根据 noComma 的值选择调用 maybeoperatorComma 或 maybeoperatorNoComma
    var me = noComma == false ? maybeoperatorComma : maybeoperatorNoComma;
    # 根据 noComma 的值选择调用 expression 或 expressionNoComma
    var expr = noComma == false ? expression : expressionNoComma;
    # 如果类型为 "=>"，则返回 pushcontext、arrowBodyNoComma 或 arrowBody 的结果
    if (type == "=>") return cont(pushcontext, noComma ? arrowBodyNoComma : arrowBody, popcontext);
    # 如果类型为 "operator"
    if (type == "operator") {
      # 如果值为 "++"、"--" 或者是 TypeScript 并且值为 "!"，则返回 me
      if (/\+\+|--/.test(value) || isTS && value == "!") return cont(me);
      # 如果是 TypeScript 并且值为 "<"，并且匹配到 "> ("，则返回 pushlex(">")、">" 类型表达式的逗号分隔、poplex、me
      if (isTS && value == "<" && cx.stream.match(/^([^<>]|<[^<>]*>)*>\s*\(/, false))
        return cont(pushlex(">"), commasep(typeexpr, ">"), poplex, me);
      # 如果值为 "?"，则返回 expression、expect(":")、expr
      if (value == "?") return cont(expression, expect(":"), expr);
      # 否则返回 expr
      return cont(expr);
    }
    # 如果类型为 "quasi"，则返回 pass(quasi, me)
    if (type == "quasi") { return pass(quasi, me); }
    # 如果类型为 ";"，则返回
    if (type == ";") return;
    # 如果类型为 "("，则返回 contCommasep(expressionNoComma, ")", "call", me)
    if (type == "(") return contCommasep(expressionNoComma, ")", "call", me);
    # 如果类型为 "."，则返回 cont(property, me)
    if (type == ".") return cont(property, me);
    # 如果类型为 "["，则返回 cont(pushlex("]"), maybeexpression, expect("]"), poplex, me)
    if (type == "[") return cont(pushlex("]"), maybeexpression, expect("]"), poplex, me);
    # 如果是 TypeScript 并且值为 "as"，则将 cx.marked 设置为 "keyword"，返回 cont(typeexpr, me)
    if (isTS && value == "as") { cx.marked = "keyword"; return cont(typeexpr, me) }
    # 如果类型为 "regexp"
    if (type == "regexp") {
      # 设置 cx.state.lastType 和 cx.marked 为 "operator"
      cx.state.lastType = cx.marked = "operator"
      # 回退到正则表达式的起始位置
      cx.stream.backUp(cx.stream.pos - cx.stream.start - 1)
      # 返回 cont(expr)
      return cont(expr)
    }
  }
  # 定义函数 quasi
  function quasi(type, value) {
    # 如果类型不是 "quasi"，则返回 pass()
    if (type != "quasi") return pass();
    # 如果值的后两个字符不是 "${"，则返回 cont(quasi)
    if (value.slice(value.length - 2) != "${") return cont(quasi);
    # 返回 cont(expression, continueQuasi)
    return cont(expression, continueQuasi);
  }
  # 定义函数 continueQuasi
  function continueQuasi(type) {
    # 如果类型为 "}"
    if (type == "}") {
      # 设置 cx.marked 为 "string-2"，将 cx.state.tokenize 设置为 tokenQuasi，返回 cont(quasi)
      cx.marked = "string-2";
      cx.state.tokenize = tokenQuasi;
      return cont(quasi);
    }
  }
  # 定义函数 arrowBody
  function arrowBody(type) {
    # 在流中查找箭头符号，返回 pass(type == "{" ? statement : expression)
    findFatArrow(cx.stream, cx.state);
    return pass(type == "{" ? statement : expression);
  }
  # 定义函数 arrowBodyNoComma
  function arrowBodyNoComma(type) {
    # 在流中查找箭头符号，返回 pass(type == "{" ? statement : expressionNoComma)
    findFatArrow(cx.stream, cx.state);
    return pass(type == "{" ? statement : expressionNoComma);
  }
  # 定义函数 maybeTarget
  function maybeTarget(noComma) {
    return function(type) {
      # 如果类型为 "."，则返回 cont(noComma ? targetNoComma : target)
      if (type == ".") return cont(noComma ? targetNoComma : target);
      # 如果类型为 "variable" 并且是 TypeScript，则返回 cont(maybeTypeArgs, noComma ? maybeoperatorNoComma : maybeoperatorComma)
      else if (type == "variable" && isTS) return cont(maybeTypeArgs, noComma ? maybeoperatorNoComma : maybeoperatorComma)
      # 否则返回 pass(noComma ? expressionNoComma : expression)
      else return pass(noComma ? expressionNoComma : expression);
    };
  }
  # 定义函数 target
  function target(_, value) {
    # 如果值等于"target"，则将cx.marked设置为"keyword"，然后调用maybeoperatorComma函数
    if (value == "target") { cx.marked = "keyword"; return cont(maybeoperatorComma); }
  }
  # 定义targetNoComma函数，如果值等于"target"，则将cx.marked设置为"keyword"，然后调用maybeoperatorNoComma函数
  function targetNoComma(_, value) {
    if (value == "target") { cx.marked = "keyword"; return cont(maybeoperatorNoComma); }
  }
  # 定义maybelabel函数，如果类型为":"，则调用poplex函数，然后调用statement函数；否则调用maybeoperatorComma函数，期望";"，最后调用poplex函数
  function maybelabel(type) {
    if (type == ":") return cont(poplex, statement);
    return pass(maybeoperatorComma, expect(";"), poplex);
  }
  # 定义property函数，如果类型为"variable"，则将cx.marked设置为"property"，然后调用cont函数
  function property(type) {
    if (type == "variable") {cx.marked = "property"; return cont();}
  }
  # 定义objprop函数，根据类型和值的不同进行不同的处理
  function objprop(type, value) {
    if (type == "async") {
      cx.marked = "property";
      return cont(objprop);
    } else if (type == "variable" || cx.style == "keyword") {
      cx.marked = "property";
      if (value == "get" || value == "set") return cont(getterSetter);
      var m // Work around fat-arrow-detection complication for detecting typescript typed arrow params
      if (isTS && cx.state.fatArrowAt == cx.stream.start && (m = cx.stream.match(/^\s*:\s*/, false)))
        cx.state.fatArrowAt = cx.stream.pos + m[0].length
      return cont(afterprop);
    } else if (type == "number" || type == "string") {
      cx.marked = jsonldMode ? "property" : (cx.style + " property");
      return cont(afterprop);
    } else if (type == "jsonld-keyword") {
      return cont(afterprop);
    } else if (isTS && isModifier(value)) {
      cx.marked = "keyword"
      return cont(objprop)
    } else if (type == "[") {
      return cont(expression, maybetype, expect("]"), afterprop);
    } else if (type == "spread") {
      return cont(expressionNoComma, afterprop);
    } else if (value == "*") {
      cx.marked = "keyword";
      return cont(objprop);
    } else if (type == ":") {
      return pass(afterprop)
    }
  }
  # 定义getterSetter函数，根据类型的不同进行不同的处理
  function getterSetter(type) {
    if (type != "variable") return pass(afterprop);
    cx.marked = "property";
    return cont(functiondef);
  }
  # 定义afterprop函数，如果类型为":"，则调用expressionNoComma函数
  function afterprop(type) {
    if (type == ":") return cont(expressionNoComma);
    # 如果类型为"("，则返回到functiondef函数
    if (type == "(") return pass(functiondef);
  }
  # 定义一个函数，用于处理逗号分隔的内容
  function commasep(what, end, sep) {
    # 定义一个内部函数，用于处理逗号分隔的内容
    function proceed(type, value) {
      # 如果存在分隔符并且当前类型在分隔符列表中，或者当前类型为逗号
      if (sep ? sep.indexOf(type) > -1 : type == ",") {
        # 获取当前状态的词法信息
        var lex = cx.state.lexical;
        if (lex.info == "call") lex.pos = (lex.pos || 0) + 1;
        # 继续处理下一个内容
        return cont(function(type, value) {
          if (type == end || value == end) return pass()
          return pass(what)
        }, proceed);
      }
      # 如果当前类型为结束符或值为结束符，则继续处理
      if (type == end || value == end) return cont();
      # 如果存在分隔符并且分隔符列表中包含";"，则继续处理what
      if (sep && sep.indexOf(";") > -1) return pass(what)
      # 否则，继续期望结束符
      return cont(expect(end));
    }
    # 返回一个函数，用于处理逗号分隔的内容
    return function(type, value) {
      if (type == end || value == end) return cont();
      return pass(what, proceed);
    };
  }
  # 定义一个函数，用于处理逗号分隔的内容
  function contCommasep(what, end, info) {
    # 将参数列表中从第三个参数开始的所有参数推入cx.cc数组
    for (var i = 3; i < arguments.length; i++)
      cx.cc.push(arguments[i]);
    # 返回一个函数，用于处理逗号分隔的内容
    return cont(pushlex(end, info), commasep(what, end), poplex);
  }
  # 定义一个函数，用于处理代码块
  function block(type) {
    # 如果类型为"}"，则返回到上一层处理
    if (type == "}") return cont();
    # 否则，继续处理语句和代码块
    return pass(statement, block);
  }
  # 定义一个函数，用于处理可能的类型
  function maybetype(type, value) {
    # 如果是TS语法
    if (isTS) {
      # 如果类型为":"，则继续处理类型表达式
      if (type == ":") return cont(typeexpr);
      # 如果值为"?"，则继续处理可能的类型
      if (value == "?") return cont(maybetype);
    }
  }
  # 定义一个函数，用于处理可能的类型或"in"关键字
  function maybetypeOrIn(type, value) {
    # 如果是TS语法并且类型为":"或值为"in"，则继续处理类型表达式
    if (isTS && (type == ":" || value == "in")) return cont(typeexpr)
  }
  # 定义一个函数，用于处理可能的返回类型
  function mayberettype(type) {
    # 如果是TS语法并且类型为":"，则继续处理类型表达式
    if (isTS && type == ":") {
      # 如果当前流匹配"\s*\w+\s+is\b"，则继续处理表达式、isKW和类型表达式
      if (cx.stream.match(/^\s*\w+\s+is\b/, false)) return cont(expression, isKW, typeexpr)
      # 否则，继续处理类型表达式
      else return cont(typeexpr)
    }
  }
  # 定义一个函数，用于处理"is"关键字
  function isKW(_, value) {
    # 如果值为"is"，则标记为关键字并继续处理
    if (value == "is") {
      cx.marked = "keyword"
      return cont()
    }
  }
  # 定义一个函数，用于处理类型表达式
  function typeexpr(type, value) {
    # 如果值为"keyof"、"typeof"或"infer"，则标记为关键字并继续处理
    if (value == "keyof" || value == "typeof" || value == "infer") {
      cx.marked = "keyword"
      return cont(value == "typeof" ? expressionNoComma : typeexpr)
    }
    # 如果类型为"variable"或值为"void"，则标记为类型并继续处理afterType
    if (type == "variable" || value == "void") {
      cx.marked = "type"
      return cont(afterType)
    }
    # 如果值为"|"或"&"，则继续处理类型表达式
    if (value == "|" || value == "&") return cont(typeexpr)
    # 如果类型为字符串、数字或原子，则返回到类型之后的继续处理
    if (type == "string" || type == "number" || type == "atom") return cont(afterType);
    # 如果类型为左方括号，则返回到左方括号之后的处理，包括推入右方括号的词法环境，解析类型表达式，弹出词法环境，最后返回到类型之后的处理
    if (type == "[") return cont(pushlex("]"), commasep(typeexpr, "]", ","), poplex, afterType)
    # 如果类型为左花括号，则返回到左花括号之后的处理，包括推入右花括号的词法环境，解析类型属性，弹出词法环境，最后返回到类型之后的处理
    if (type == "{") return cont(pushlex("}"), commasep(typeprop, "}", ",;"), poplex, afterType)
    # 如果类型为左括号，则返回到左括号之后的处理，包括解析类型参数，可能的返回类型，最后返回到类型之后的处理
    if (type == "(") return cont(commasep(typearg, ")"), maybeReturnType, afterType)
    # 如果类型为小于号，则返回到小于号之后的处理，包括解析类型表达式，最后返回到类型表达式的处理
    if (type == "<") return cont(commasep(typeexpr, ">"), typeexpr)
  }
  # 可能的返回类型处理函数
  function maybeReturnType(type) {
    # 如果类型为箭头符号，则返回到类型表达式的处理
    if (type == "=>") return cont(typeexpr)
  }
  # 类型属性处理函数
  function typeprop(type, value) {
    # 如果类型为变量或上下文样式为关键字，则标记为属性，返回到类型属性的处理
    if (type == "variable" || cx.style == "keyword") {
      cx.marked = "property"
      return cont(typeprop)
    } else if (value == "?" || type == "number" || type == "string") {
      return cont(typeprop)
    } else if (type == ":") {
      return cont(typeexpr)
    } else if (type == "[") {
      return cont(expect("variable"), maybetypeOrIn, expect("]"), typeprop)
    } else if (type == "(") {
      return pass(functiondecl, typeprop)
    }
  }
  # 类型参数处理函数
  function typearg(type, value) {
    # 如果类型为变量且上下文流匹配空格、问号或冒号，则返回到类型参数的处理
    if (type == "variable" && cx.stream.match(/^\s*[?:]/, false) || value == "?") return cont(typearg)
    # 如果类型为冒号，则返回到类型表达式的处理
    if (type == ":") return cont(typeexpr)
    # 如果类型为展开符，则返回到类型参数的处理
    if (type == "spread") return cont(typearg)
    # 否则，跳过类型表达式的处理
    return pass(typeexpr)
  }
  # 类型之后的处理函数
  function afterType(type, value) {
    # 如果值为小于号，则返回到小于号之后的处理，包括推入右尖括号的词法环境，解析类型表达式，弹出词法环境，最后返回到类型之后的处理
    if (value == "<") return cont(pushlex(">"), commasep(typeexpr, ">"), poplex, afterType)
    # 如果值为竖线、点或与符号，则返回到类型表达式的处理
    if (value == "|" || type == "." || value == "&") return cont(typeexpr)
    # 如果类型为左方括号，则返回到类型表达式的处理，包括解析类型表达式，期望右方括号，最后返回到类型之后的处理
    if (type == "[") return cont(typeexpr, expect("]"), afterType)
    # 如果值为extends或implements，则标记为关键字，返回到类型表达式的处理
    if (value == "extends" || value == "implements") { cx.marked = "keyword"; return cont(typeexpr) }
    # 如果值为问号，则返回到类型表达式的处理，期望冒号，解析类型表达式
    if (value == "?") return cont(typeexpr, expect(":"), typeexpr)
  }
  # 可能的类型参数处理函数
  function maybeTypeArgs(_, value) {
    # 如果值为小于号，则返回到小于号之后的处理，包括推入右尖括号的词法环境，解析类型表达式，弹出词法环境，最后返回到类型之后的处理
    if (value == "<") return cont(pushlex(">"), commasep(typeexpr, ">"), poplex, afterType)
  }
  # 类型参数处理函数
  function typeparam() {
    return pass(typeexpr, maybeTypeDefault)
  }
  # 可能的类型默认值处理函数
  function maybeTypeDefault(_, value) {
    # 如果值等于"="，则返回到typeexpr函数
    if (value == "=") return cont(typeexpr)
  }
  # 定义变量的函数
  function vardef(_, value) {
    # 如果值等于"enum"，则将标记设置为"keyword"，并返回到enumdef函数
    if (value == "enum") {cx.marked = "keyword"; return cont(enumdef)}
    # 否则，传递给pattern、maybetype、maybeAssign、vardefCont函数
    return pass(pattern, maybetype, maybeAssign, vardefCont);
  }
  # 匹配模式的函数
  function pattern(type, value) {
    # 如果是 TypeScript 并且是修饰符，则将标记设置为"keyword"，并返回到pattern函数
    if (isTS && isModifier(value)) { cx.marked = "keyword"; return cont(pattern) }
    # 如果类型是"variable"，则注册该值，并返回
    if (type == "variable") { register(value); return cont(); }
    # 如果类型是"spread"，则返回到pattern函数
    if (type == "spread") return cont(pattern);
    # 如果类型是"["，则返回到contCommasep(eltpattern, "]")函数
    if (type == "[") return contCommasep(eltpattern, "]");
    # 如果类型是"{"，则返回到contCommasep(proppattern, "}")函数
    if (type == "{") return contCommasep(proppattern, "}");
  }
  # 属性模式的函数
  function proppattern(type, value) {
    # 如果类型是"variable"并且不匹配":\s*"，则注册该值，并返回到maybeAssign函数
    if (type == "variable" && !cx.stream.match(/^\s*:/, false)) {
      register(value);
      return cont(maybeAssign);
    }
    # 如果类型是"variable"，则将标记设置为"property"
    if (type == "variable") cx.marked = "property";
    # 如果类型是"spread"，则返回到pattern函数
    if (type == "spread") return cont(pattern);
    # 如果类型是"}"，则传递
    if (type == "}") return pass();
    # 如果类型是"["，则返回到cont(expression, expect(']'), expect(':'), proppattern)函数
    if (type == "[") return cont(expression, expect(']'), expect(':'), proppattern);
    # 否则，返回到cont(expect(":"), pattern, maybeAssign)函数
    return cont(expect(":"), pattern, maybeAssign);
  }
  # 元素模式的函数
  function eltpattern() {
    return pass(pattern, maybeAssign)
  }
  # 可能赋值的函数
  function maybeAssign(_type, value) {
    # 如果值等于"="，则返回到expressionNoComma函数
    if (value == "=") return cont(expressionNoComma);
  }
  # 变量定义的继续函数
  function vardefCont(type) {
    # 如果类型是","，则返回到vardef函数
    if (type == ",") return cont(vardef);
  }
  # 可能是else的函数
  function maybeelse(type, value) {
    # 如果类型是"keyword b"并且值是"else"，则返回到pushlex("form", "else")函数
    if (type == "keyword b" && value == "else") return cont(pushlex("form", "else"), statement, poplex);
  }
  # for 循环规范的函数
  function forspec(type, value) {
    # 如果值是"await"，则返回到forspec函数
    if (value == "await") return cont(forspec);
    # 如果类型是"("，则返回到pushlex(")")，forspec1，poplex函数
    if (type == "(") return cont(pushlex(")"), forspec1, poplex);
  }
  # for 循环规范1的函数
  function forspec1(type) {
    # 如果类型是"var"，则返回到vardef，forspec2函数
    if (type == "var") return cont(vardef, forspec2);
    # 如果类型是"variable"，则返回到forspec2函数
    if (type == "variable") return cont(forspec2);
    # 否则，传递给forspec2函数
    return pass(forspec2)
  }
  # for 循环规范2的函数
  function forspec2(type, value) {
    # 如果类型是")"，则返回
    if (type == ")") return cont()
    # 如果类型是";"，则返回到forspec2函数
    if (type == ";") return cont(forspec2)
    # 如果值是"in"或"of"，则将标记设置为"keyword"，并返回到expression，forspec2函数
    if (value == "in" || value == "of") { cx.marked = "keyword"; return cont(expression, forspec2) }
    # 返回表达式和 forspec2
    return pass(expression, forspec2)
  }
  # 定义函数
  function functiondef(type, value) {
    # 如果值为 "*"，标记为关键字，继续定义函数
    if (value == "*") {cx.marked = "keyword"; return cont(functiondef);}
    # 如果类型为 "variable"，注册该值，继续定义函数
    if (type == "variable") {register(value); return cont(functiondef);}
    # 如果类型为 "("，推入上下文，推入 ")" 到词法环境，使用逗号分隔的参数列表，弹出词法环境，可能有返回类型，语句，弹出上下文
    if (type == "(") return cont(pushcontext, pushlex(")"), commasep(funarg, ")"), poplex, mayberettype, statement, popcontext);
    # 如果是 TypeScript 并且值为 "<"，推入 "<" 到词法环境，使用逗号分隔的类型参数列表，弹出词法环境，继续定义函数
    if (isTS && value == "<") return cont(pushlex(">"), commasep(typeparam, ">"), poplex, functiondef)
  }
  # 函数声明
  function functiondecl(type, value) {
    # 如果值为 "*"，标记为关键字，继续函数声明
    if (value == "*") {cx.marked = "keyword"; return cont(functiondecl);}
    # 如果类型为 "variable"，注册该值，继续函数声明
    if (type == "variable") {register(value); return cont(functiondecl);}
    # 如果类型为 "("，推入上下文，推入 ")" 到词法环境，使用逗号分隔的参数列表，弹出词法环境，可能有返回类型，弹出上下文
    if (type == "(") return cont(pushcontext, pushlex(")"), commasep(funarg, ")"), poplex, mayberettype, popcontext);
    # 如果是 TypeScript 并且值为 "<"，推入 "<" 到词法环境，使用逗号分隔的类型参数列表，弹出词法环境，继续函数声明
    if (isTS && value == "<") return cont(pushlex(">"), commasep(typeparam, ">"), poplex, functiondecl)
  }
  # 类型名称
  function typename(type, value) {
    # 如果类型为 "keyword" 或 "variable"，标记为类型，继续类型名称
    if (type == "keyword" || type == "variable") {
      cx.marked = "type"
      return cont(typename)
    } 
    # 如果值为 "<"，推入 "<" 到词法环境，使用逗号分隔的类型参数列表，弹出词法环境
    else if (value == "<") {
      return cont(pushlex(">"), commasep(typeparam, ">"), poplex)
    }
  }
  # 函数参数
  function funarg(type, value) {
    # 如果值为 "@"，继续表达式和函数参数
    if (value == "@") cont(expression, funarg)
    # 如果类型为 "spread"，继续函数参数
    if (type == "spread") return cont(funarg);
    # 如果是 TypeScript 并且是修饰符，标记为关键字，继续函数参数
    if (isTS && isModifier(value)) { cx.marked = "keyword"; return cont(funarg); }
    # 如果是 TypeScript 并且类型为 "this"，可能有类型，可能有赋值
    if (isTS && type == "this") return cont(maybetype, maybeAssign)
    # 返回模式，可能有类型，可能有赋值
    return pass(pattern, maybetype, maybeAssign);
  }
  # 类表达式
  function classExpression(type, value) {
    # 类表达式可能有可选名称
    if (type == "variable") return className(type, value);
    return classNameAfter(type, value);
  }
  # 类名称
  function className(type, value) {
    # 如果类型为 "variable"，注册该值，继续类名称
    if (type == "variable") {register(value); return cont(classNameAfter);}
  }
  # 类名称之后
  function classNameAfter(type, value) {
    # 如果值为 "<"，推入 "<" 到词法环境，使用逗号分隔的类型参数列表，弹出词法环境，继续类名称之后
    if (value == "<") return cont(pushlex(">"), commasep(typeparam, ">"), poplex, classNameAfter)
    # 如果值为"extends"、"implements"或(isTS为真且类型为",")，则执行以下操作
    if (value == "extends" || value == "implements" || (isTS && type == ",")) {
      # 如果值为"implements"，则将cx.marked标记为"keyword"
      if (value == "implements") cx.marked = "keyword";
      # 返回继续解析typeexpr或expression，然后执行classNameAfter函数
      return cont(isTS ? typeexpr : expression, classNameAfter);
    }
    # 如果类型为"{"，则返回继续解析classBody，并推入"}"到词法环境栈中
    if (type == "{") return cont(pushlex("}"), classBody, poplex);
  }
  # 定义classBody函数，解析类的主体部分
  function classBody(type, value) {
    # 如果类型为"async"或者(type为"variable"且(value为"static"、"get"、"set"或(isTS为真且是修饰符(value)))且cx.stream匹配到空白字符和字母、数字、下划线、或其他Unicode字符，则执行以下操作
    if (type == "async" ||
        (type == "variable" &&
         (value == "static" || value == "get" || value == "set" || (isTS && isModifier(value))) &&
         cx.stream.match(/^\s+[\w$\xa1-\uffff]/, false))) {
      # 将cx.marked标记为"keyword"，然后返回继续解析classBody
      cx.marked = "keyword";
      return cont(classBody);
    }
    # 如果类型为"variable"或者cx.style为"keyword"，则将cx.marked标记为"property"，然后返回继续解析classfield和classBody
    if (type == "variable" || cx.style == "keyword") {
      cx.marked = "property";
      return cont(classfield, classBody);
    }
    # 如果类型为"number"或"string"，则返回继续解析classfield和classBody
    if (type == "number" || type == "string") return cont(classfield, classBody);
    # 如果类型为"["，则返回继续解析expression、maybetype、"]"、classfield和classBody
    if (type == "[")
      return cont(expression, maybetype, expect("]"), classfield, classBody)
    # 如果值为"*"，则将cx.marked标记为"keyword"，然后返回继续解析classBody
    if (value == "*") {
      cx.marked = "keyword";
      return cont(classBody);
    }
    # 如果是TS并且类型为"("，则跳过functiondecl，返回继续解析classBody
    if (isTS && type == "(") return pass(functiondecl, classBody)
    # 如果类型为";"或","，则返回继续解析classBody
    if (type == ";" || type == ",") return cont(classBody);
    # 如果类型为"}"，则返回继续解析
    if (type == "}") return cont();
    # 如果值为"@"，则返回继续解析expression和classBody
    if (value == "@") return cont(expression, classBody)
  }
  # 定义classfield函数，解析类的字段
  function classfield(type, value) {
    # 如果值为"?"，则返回继续解析classfield
    if (value == "?") return cont(classfield)
    # 如果类型为":"，则返回继续解析typeexpr和maybeAssign
    if (type == ":") return cont(typeexpr, maybeAssign)
    # 如果值为"="，则返回继续解析expressionNoComma
    if (value == "=") return cont(expressionNoComma)
    # 获取上下文信息，判断是否为接口
    var context = cx.state.lexical.prev, isInterface = context && context.info == "interface"
    # 跳过isInterface为真则跳过functiondecl，否则跳过functiondef
    return pass(isInterface ? functiondecl : functiondef)
  }
  # 定义afterExport函数，处理导出后的操作
  function afterExport(type, value) {
    # 如果值为"*"，则将cx.marked标记为"keyword"，然后返回继续解析maybeFrom和";"
    if (value == "*") { cx.marked = "keyword"; return cont(maybeFrom, expect(";")); }
    # 如果值为"default"，则将cx.marked标记为"keyword"，然后返回继续解析expression和";"
    if (value == "default") { cx.marked = "keyword"; return cont(expression, expect(";")); }
    # 如果类型为"{"，则返回继续解析exportField，"}"，maybeFrom和";"
    if (type == "{") return cont(commasep(exportField, "}"), maybeFrom, expect(";"));
    # 跳过statement
    return pass(statement);
  }
  # 定义exportField函数，处理导出字段
  function exportField(type, value) {
    # 如果值等于 "as"，则将 cx.marked 设置为 "keyword"，然后返回继续解析变量
    if (value == "as") { cx.marked = "keyword"; return cont(expect("variable")); }
    # 如果类型为 "variable"，则返回表达式，不包括逗号，也包括导出字段
    if (type == "variable") return pass(expressionNoComma, exportField);
  }
  # 处理 import 后的语句
  function afterImport(type) {
    # 如果类型为 "string"，则返回继续解析
    if (type == "string") return cont();
    # 如果类型为 "("，则返回表达式
    if (type == "(") return pass(expression);
    # 否则返回 importSpec，可能有更多的导入，也可能有 from
    return pass(importSpec, maybeMoreImports, maybeFrom);
  }
  # 处理导入规范
  function importSpec(type, value) {
    # 如果类型为 "{"，则返回逗号分隔的 importSpec，直到遇到 "}"
    if (type == "{") return contCommasep(importSpec, "}");
    # 如果类型为 "variable"，则注册该变量
    if (type == "variable") register(value);
    # 如果值为 "*"，则将 cx.marked 设置为 "keyword"
    if (value == "*") cx.marked = "keyword";
    # 返回可能有 as 的继续解析
    return cont(maybeAs);
  }
  # 处理可能有更多导入的情况
  function maybeMoreImports(type) {
    # 如果类型为 ","，则返回继续解析 importSpec，可能有更多导入
    if (type == ",") return cont(importSpec, maybeMoreImports)
  }
  # 处理可能有 as 的情况
  function maybeAs(_type, value) {
    # 如果值为 "as"，则将 cx.marked 设置为 "keyword"，然后返回继续解析 importSpec
    if (value == "as") { cx.marked = "keyword"; return cont(importSpec); }
  }
  # 处理可能有 from 的情况
  function maybeFrom(_type, value) {
    # 如果值为 "from"，则将 cx.marked 设置为 "keyword"，然后返回继续解析表达式
    if (value == "from") { cx.marked = "keyword"; return cont(expression); }
  }
  # 处理数组字面量
  function arrayLiteral(type) {
    # 如果类型为 "]"，则返回继续解析
    if (type == "]") return cont();
    # 否则返回逗号分隔的表达式，不包括逗号，直到遇到 "]"
    return pass(commasep(expressionNoComma, "]"));
  }
  # 处理枚举定义
  function enumdef() {
    # 返回推入 "form"，解析模式，期望 "{", 推入 "}"，逗号分隔的枚举成员，弹出 "}"，弹出 "form"
    return pass(pushlex("form"), pattern, expect("{"), pushlex("}"), commasep(enummember, "}"), poplex, poplex)
  }
  # 处理枚举成员
  function enummember() {
    # 返回解析模式，可能有赋值
    return pass(pattern, maybeAssign);
  }

  # 判断语句是否继续
  function isContinuedStatement(state, textAfter) {
    return state.lastType == "operator" || state.lastType == "," ||
      isOperatorChar.test(textAfter.charAt(0)) ||
      /[,.]/.test(textAfter.charAt(0));
  }

  # 判断是否允许表达式
  function expressionAllowed(stream, state, backUp) {
    return state.tokenize == tokenBase &&
      /^(?:operator|sof|keyword [bcd]|case|new|export|default|spread|[\[{}\(,;:]|=>)$/.test(state.lastType) ||
      (state.lastType == "quasi" && /\{\s*$/.test(stream.string.slice(0, stream.pos - (backUp || 0))))
  }

  # 接口
  return {
    # 定义一个函数，用于初始化解析器状态
    startState: function(basecolumn) {
      # 初始化状态对象
      var state = {
        # 设置初始的标记函数
        tokenize: tokenBase,
        # 设置初始的类型为 "sof"
        lastType: "sof",
        # 初始化代码缩进级别
        cc: [],
        # 创建新的词法分析器对象
        lexical: new JSLexical((basecolumn || 0) - indentUnit, 0, "block", false),
        # 设置局部变量
        localVars: parserConfig.localVars,
        # 设置上下文
        context: parserConfig.localVars && new Context(null, null, false),
        # 设置缩进级别
        indented: basecolumn || 0
      };
      # 如果存在全局变量并且是对象，则设置全局变量
      if (parserConfig.globalVars && typeof parserConfig.globalVars == "object")
        state.globalVars = parserConfig.globalVars;
      # 返回状态对象
      return state;
    },

    # 定义一个函数，用于处理代码流的标记
    token: function(stream, state) {
      # 如果是行的开始
      if (stream.sol()) {
        # 如果状态对象的词法属性中没有 "align"，则设置为 false
        if (!state.lexical.hasOwnProperty("align"))
          state.lexical.align = false;
        # 设置缩进级别
        state.indented = stream.indentation();
        # 查找箭头函数
        findFatArrow(stream, state);
      }
      # 如果不是注释的标记函数并且存在空白字符，则返回空
      if (state.tokenize != tokenComment && stream.eatSpace()) return null;
      # 调用标记函数处理代码流，并返回样式
      var style = state.tokenize(stream, state);
      # 如果类型是注释，则返回样式
      if (type == "comment") return style;
      # 设置状态对象的最后类型
      state.lastType = type == "operator" && (content == "++" || content == "--") ? "incdec" : type;
      # 解析 JavaScript 代码
      return parseJS(state, style, type, content, stream);
    },
    # 定义一个函数，用于处理缩进
    indent: function(state, textAfter) {
      # 如果当前正在处理注释，则返回
      if (state.tokenize == tokenComment) return CodeMirror.Pass;
      # 如果不是在处理基本的token，则返回0
      if (state.tokenize != tokenBase) return 0;
      # 获取文本的第一个字符和当前的词法状态
      var firstChar = textAfter && textAfter.charAt(0), lexical = state.lexical, top
      # 修正，防止'maybelse'阻止词法范围的弹出
      if (!/^\s*else\b/.test(textAfter)) for (var i = state.cc.length - 1; i >= 0; --i) {
        var c = state.cc[i];
        if (c == poplex) lexical = lexical.prev;
        else if (c != maybeelse) break;
      }
      # 循环处理词法状态，直到找到合适的缩进位置
      while ((lexical.type == "stat" || lexical.type == "form") &&
             (firstChar == "}" || ((top = state.cc[state.cc.length - 1]) &&
                                   (top == maybeoperatorComma || top == maybeoperatorNoComma) &&
                                   !/^[,\.=+\-*:?[\(]/.test(textAfter))))
        lexical = lexical.prev;
      # 处理特殊情况，返回相应的缩进值
      if (statementIndent && lexical.type == ")" && lexical.prev.type == "stat")
        lexical = lexical.prev;
      var type = lexical.type, closing = firstChar == type;

      if (type == "vardef") return lexical.indented + (state.lastType == "operator" || state.lastType == "," ? lexical.info.length + 1 : 0);
      else if (type == "form" && firstChar == "{") return lexical.indented;
      else if (type == "form") return lexical.indented + indentUnit;
      else if (type == "stat")
        return lexical.indented + (isContinuedStatement(state, textAfter) ? statementIndent || indentUnit : 0);
      else if (lexical.info == "switch" && !closing && parserConfig.doubleIndentSwitch != false)
        return lexical.indented + (/^(?:case|default)\b/.test(textAfter) ? indentUnit : 2 * indentUnit);
      else if (lexical.align) return lexical.column + (closing ? 0 : 1);
      else return lexical.indented + (closing ? 0 : indentUnit);
    },

    # 定义一个正则表达式，用于匹配需要自动缩进的输入
    electricInput: /^\s*(?:case .*?:|default:|\{|\})$/,
    # 定义块注释的起始标记
    blockCommentStart: jsonMode ? null : "/*",
    # 定义块注释的结束标记
    blockCommentEnd: jsonMode ? null : "*/",
    // 如果处于 JSON 模式，则使用块注释的格式，否则使用普通的块注释格式
    blockCommentContinue: jsonMode ? null : " * ",
    // 如果处于 JSON 模式，则使用空注释，否则使用双斜杠注释
    lineComment: jsonMode ? null : "//",
    // 折叠代码块的方式为大括号
    fold: "brace",
    // 设置关闭括号的字符
    closeBrackets: "()[]{}''\"\"``",

    // 设置辅助类型为 JSON 或 JavaScript
    helperType: jsonMode ? "json" : "javascript",
    // 设置 JSON-LD 模式
    jsonldMode: jsonldMode,
    // 设置 JSON 模式
    jsonMode: jsonMode,

    // 表达式是否允许
    expressionAllowed: expressionAllowed,

    // 跳过表达式
    skipExpression: function(state) {
      // 获取当前状态的栈顶元素
      var top = state.cc[state.cc.length - 1]
      // 如果栈顶元素为表达式或无逗号表达式，则弹出栈顶元素
      if (top == expression || top == expressionNoComma) state.cc.pop()
    }
  };
});

// 注册 JavaScript 的单词字符
CodeMirror.registerHelper("wordChars", "javascript", /[\w$]/);

// 定义 MIME 类型为 JavaScript
CodeMirror.defineMIME("text/javascript", "javascript");
CodeMirror.defineMIME("text/ecmascript", "javascript");
CodeMirror.defineMIME("application/javascript", "javascript");
CodeMirror.defineMIME("application/x-javascript", "javascript");
CodeMirror.defineMIME("application/ecmascript", "javascript");
CodeMirror.defineMIME("application/json", {name: "javascript", json: true});
CodeMirror.defineMIME("application/x-json", {name: "javascript", json: true});
CodeMirror.defineMIME("application/ld+json", {name: "javascript", jsonld: true});
CodeMirror.defineMIME("text/typescript", { name: "javascript", typescript: true });
CodeMirror.defineMIME("application/typescript", { name: "javascript", typescript: true });

// 结束 JavaScript 代码块
});

/* ---- mode/markdown.js ---- */

// CodeMirror, 版权归 Marijn Haverbeke 和其他人所有，根据 MIT 许可证分发：https://codemirror.net/LICENSE
// 匿名函数，传入 mod 参数
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror"), require("../xml/xml"), require("../meta"));
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror", "../xml/xml", "../meta"], mod);
  // 如果是普通浏览器环境
  else
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义 Markdown 模式
  CodeMirror.defineMode("markdown", function(cmCfg, modeCfg) {

    // 获取 HTML 模式
    var htmlMode = CodeMirror.getMode(cmCfg, "text/html");
    // 检查 HTML 模式是否缺失
    var htmlModeMissing = htmlMode.name == "null"

    // 获取模式
    function getMode(name) {
      // 如果存在 CodeMirror.findModeByName 方法
      if (CodeMirror.findModeByName) {
        // 查找指定名称的模式
        var found = CodeMirror.findModeByName(name);
        // 如果找到了，使用其 MIME 类型
        if (found) name = found.mime || found.mimes[0];
      }
      // 获取指定名称的模式
      var mode = CodeMirror.getMode(cmCfg, name);
    // 如果模式名称为 "null"，则返回 null，否则返回模式名称
    return mode.name == "null" ? null : mode;
  }

  // 是否应该将影响高亮显示的字符单独高亮显示？
  // 不包括将被输出的字符（例如列表的 `1.` 和 `-`）
  if (modeCfg.highlightFormatting === undefined)
    modeCfg.highlightFormatting = false;

  // 最大嵌套引用块的数量。设置为 0 表示无限嵌套。
  // 多余的 `>` 将发出 `error` 标记。
  if (modeCfg.maxBlockquoteDepth === undefined)
    modeCfg.maxBlockquoteDepth = 0;

  // 打开任务列表？（"- [ ] " 和 "- [x] "）
  if (modeCfg.taskLists === undefined) modeCfg.taskLists = false;

  // 打开删除线语法
  if (modeCfg.strikethrough === undefined)
    modeCfg.strikethrough = false;

  // 打开表情符号
  if (modeCfg.emoji === undefined)
    modeCfg.emoji = false;

  // 打开围栏代码块高亮
  if (modeCfg.fencedCodeBlockHighlighting === undefined)
    modeCfg.fencedCodeBlockHighlighting = true;

  // 围栏代码块的默认模式
  if (modeCfg.fencedCodeBlockDefaultMode === undefined)
    modeCfg.fencedCodeBlockDefaultMode = 'text/plain';

  // 打开 XML 支持
  if (modeCfg.xml === undefined)
    modeCfg.xml = true;

  // 允许用户提供的标记类型覆盖标记类型。
  if (modeCfg.tokenTypeOverrides === undefined)
    modeCfg.tokenTypeOverrides = {};

  // 定义标记类型的默认值
  var tokenTypes = {
    header: "header",
    code: "comment",
    quote: "quote",
    list1: "variable-2",
    list2: "variable-3",
    list3: "keyword",
    hr: "hr",
    image: "image",
    imageAltText: "image-alt-text",
    imageMarker: "image-marker",
    formatting: "formatting",
    linkInline: "link",
    linkEmail: "link",
    linkText: "link",
    linkHref: "string",
    em: "em",
    strong: "strong",
    strikethrough: "strikethrough",
    emoji: "builtin"
  };

  // 遍历标记类型，如果用户提供了覆盖，则使用用户提供的标记类型
  for (var tokenType in tokenTypes) {
    if (tokenTypes.hasOwnProperty(tokenType) && modeCfg.tokenTypeOverrides[tokenType]) {
      tokenTypes[tokenType] = modeCfg.tokenTypeOverrides[tokenType];
    state.f = state.inline = f;
  // 返回函数 f 处理流和状态
  return f(stream, state);
}

function switchBlock(stream, state, f) {
  // 设置状态的函数和块
  state.f = state.block = f;
  // 返回函数 f 处理流和状态
  return f(stream, state);
}

function lineIsEmpty(line) {
  // 判断行是否为空
  return !line || !/\S/.test(line.string)
}

// Blocks

function blankLine(state) {
  // 重置链接标题状态
  state.linkTitle = false;
  state.linkHref = false;
  state.linkText = false;
  // 重置 EM 状态
  state.em = false;
  // 重置 STRONG 状态
  state.strong = false;
  // 重置删除线状态
  state.strikethrough = false;
  // 重置引用状态
  state.quote = 0;
  // 重置缩进代码状态
  state.indentedCode = false;
  if (state.f == htmlBlock) {
    var exit = htmlModeMissing
    if (!exit) {
      var inner = CodeMirror.innerMode(htmlMode, state.htmlState)
      exit = inner.mode.name == "xml" && inner.state.tagStart === null &&
        (!inner.state.context && inner.state.tokenize.isInText)
    }
    if (exit) {
      state.f = inlineNormal;
      state.block = blockNormal;
      state.htmlState = null;
    }
  }
  // 重置尾随空格状态
  state.trailingSpace = 0;
  state.trailingSpaceNewLine = false;
  // 标记此行为空行
  state.prevLine = state.thisLine
  state.thisLine = {stream: null}
  return null;
}

function blockNormal(stream, state) {
  var firstTokenOnLine = stream.column() === state.indentation;
  var prevLineLineIsEmpty = lineIsEmpty(state.prevLine.stream);
  var prevLineIsIndentedCode = state.indentedCode;
  var prevLineIsHr = state.prevLine.hr;
  var prevLineIsList = state.list !== false;
  var maxNonCodeIndentation = (state.listStack[state.listStack.length - 1] || 0) + 3;

  state.indentedCode = false;

  var lineIndentation = state.indentation;
  // 每行计算一次（在第一个标记上）
    # 如果缩进差为 null，则将其设置为当前缩进
    if (state.indentationDiff === null) {
      state.indentationDiff = state.indentation;
      # 如果前一行是列表，则将列表状态置为 null
      if (prevLineIsList) {
        state.list = null;
        # 当该列表项的标记缩进小于最深列表项内容的缩进时，从堆栈中弹出最深列表项的缩进，并更新块缩进状态
        while (lineIndentation < state.listStack[state.listStack.length - 1]) {
          state.listStack.pop();
          if (state.listStack.length) {
            state.indentation = state.listStack[state.listStack.length - 1];
          # 小于第一个列表的缩进 -> 该行不再是列表
          } else {
            state.list = false;
          }
        }
        # 如果列表不为 false，则计算缩进差值
        if (state.list !== false) {
          state.indentationDiff = lineIndentation - state.listStack[state.listStack.length - 1]
        }
      }
    }

    # 检查是否允许内联延续（目前仅用于 setext 检测）
    var allowsInlineContinuation = (
        !prevLineLineIsEmpty && !prevLineIsHr && !state.prevLine.header &&
        (!prevLineIsList || !prevLineIsIndentedCode) &&
        !state.prevLine.fencedCodeEnd
    );

    # 检查是否为水平线
    var isHr = (state.list === false || prevLineIsHr || prevLineLineIsEmpty) &&
      state.indentation <= maxNonCodeIndentation && stream.match(hrRE);

    var match = null;
    # 如果缩进差大于等于 4，并且（前一行是缩进代码或者前一行是围栏代码结束或者前一行是标题或者前一行是空行）
    if (state.indentationDiff >= 4 && (prevLineIsIndentedCode || state.prevLine.fencedCodeEnd ||
         state.prevLine.header || prevLineLineIsEmpty)) {
      stream.skipToEnd();
      state.indentedCode = true;
      return tokenTypes.code;
    # 如果是空格，则返回 null
    } else if (stream.eatSpace()) {
      return null;
    } else if (firstTokenOnLine && state.indentation <= maxNonCodeIndentation && (match = stream.match(atxHeaderRE)) && match[1].length <= 6) {
      # 如果当前行的第一个标记且缩进小于等于最大非代码缩进，并且匹配到 ATX 标题的正则表达式，并且标题长度小于等于6
      state.quote = 0;
      # 重置引用状态
      state.header = match[1].length;
      # 设置标题级别
      state.thisLine.header = true;
      # 标记当前行为标题行
      if (modeCfg.highlightFormatting) state.formatting = "header";
      # 如果启用了高亮格式，则设置当前格式为标题
      state.f = state.inline;
      # 设置状态为内联
      return getType(state);
      # 返回当前状态类型
    } else if (state.indentation <= maxNonCodeIndentation && stream.eat('>')) {
      # 如果缩进小于等于最大非代码缩进，并且当前字符为 '>'
      state.quote = firstTokenOnLine ? 1 : state.quote + 1;
      # 如果是行的第一个标记，则引用状态为1，否则引用状态加1
      if (modeCfg.highlightFormatting) state.formatting = "quote";
      # 如果启用了高亮格式，则设置当前格式为引用
      stream.eatSpace();
      # 跳过空格
      return getType(state);
      # 返回当前状态类型
    } else if (!isHr && !state.setext && firstTokenOnLine && state.indentation <= maxNonCodeIndentation && (match = stream.match(listRE))) {
      # 如果不是水平线且不是 setext 标题且是行的第一个标记且缩进小于等于最大非代码缩进，并且匹配到列表的正则表达式
      var listType = match[1] ? "ol" : "ul";
      # 根据匹配结果确定列表类型

      state.indentation = lineIndentation + stream.current().length;
      # 设置缩进为行缩进加上当前标记的长度
      state.list = true;
      # 设置状态为列表
      state.quote = 0;
      # 重置引用状态

      # 将此列表项的内容缩进添加到堆栈
      state.listStack.push(state.indentation);
      # 将当前缩进值添加到列表缩进堆栈
      # 重置不应传播到列表项的内联样式
      state.em = false;
      state.strong = false;
      state.code = false;
      state.strikethrough = false;

      if (modeCfg.taskLists && stream.match(taskListRE, false)) {
        state.taskList = true;
      }
      # 如果启用了任务列表并且匹配到任务列表的正则表达式，则设置状态为任务列表
      state.f = state.inline;
      # 设置状态为内联
      if (modeCfg.highlightFormatting) state.formatting = ["list", "list-" + listType];
      # 如果启用了高亮格式，则设置当前格式为列表和列表类型
      return getType(state);
      # 返回当前状态类型
    // 如果当前行的第一个标记并且缩进小于等于最大非代码缩进，并且匹配了 fencedCodeRE
    } else if (firstTokenOnLine && state.indentation <= maxNonCodeIndentation && (match = stream.match(fencedCodeRE, true))) {
      // 重置引用状态
      state.quote = 0;
      // 设置 fencedEndRE 为匹配结果的第一个分组加上零个或多个空格然后结束
      state.fencedEndRE = new RegExp(match[1] + "+ *$");
      // 尝试切换模式
      state.localMode = modeCfg.fencedCodeBlockHighlighting && getMode(match[2] || modeCfg.fencedCodeBlockDefaultMode );
      // 如果 localMode 存在，则设置 localState 为 localMode 的起始状态
      if (state.localMode) state.localState = CodeMirror.startState(state.localMode);
      // 设置 f 和 block 为 local
      state.f = state.block = local;
      // 如果 highlightFormatting 存在，则设置 formatting 为 "code-block"
      if (modeCfg.highlightFormatting) state.formatting = "code-block";
      // 设置 code 为 -1
      state.code = -1
      // 返回状态类型
      return getType(state);
    // SETEXT 在 HR 之后具有最低的块范围优先级，因此在其他情况（code, blockquote, list...）之后检查它
    } else if (
      // 如果 setext 存在，表示---/===后的行
      state.setext || (
        // ---/===前的行
        (!allowsInlineContinuation || !prevLineIsList) && !state.quote && state.list === false &&
        !state.code && !isHr && !linkDefRE.test(stream.string) &&
        (match = stream.lookAhead(1)) && (match = match.match(setextHeaderRE))
      )
    ) {
      // 如果 setext 不存在
      if ( !state.setext ) {
        // 设置 header 为 match[0] 的第一个字符是'='则为1，否则为2
        state.header = match[0].charAt(0) == '=' ? 1 : 2;
        // 设置 setext 为 header
        state.setext = state.header;
      } else {
        // 设置 header 为 setext
        state.header = state.setext;
        // 对类型没有影响，所以现在可以重置它
        state.setext = 0;
        // 跳过到行尾
        stream.skipToEnd();
        // 如果 highlightFormatting 存在，则设置 formatting 为 "header"
        if (modeCfg.highlightFormatting) state.formatting = "header";
      }
      // 设置当前行为 header
      state.thisLine.header = true;
      // 设置 f 为 inline
      state.f = state.inline;
      // 返回状态类型
      return getType(state);
    } else if (isHr) {
      // 跳过到行尾
      stream.skipToEnd();
      // 设置 hr 为 true
      state.hr = true;
      // 设置当前行为 hr
      state.thisLine.hr = true;
      // 返回 tokenTypes.hr
      return tokenTypes.hr;
    } else if (stream.peek() === '[') {
      // 切换到内联模式处理方括号
      return switchInline(stream, state, footnoteLink);
    }

    // 切换到内联模式处理其他情况
    return switchInline(stream, state, state.inline);
  }

  // 处理 HTML 块
  function htmlBlock(stream, state) {
    // 调用 htmlMode 的 token 方法处理 HTML 块
    var style = htmlMode.token(stream, state.htmlState);
    // 如果不缺少 htmlMode
    if (!htmlModeMissing) {
      // 获取内部模式
      var inner = CodeMirror.innerMode(htmlMode, state.htmlState)
      // 如果内部模式是 xml 并且没有标签开始，并且在文本中，或者在 md_inside 中并且当前流中包含 ">" 符号
      if ((inner.mode.name == "xml" && inner.state.tagStart === null &&
           (!inner.state.context && inner.state.tokenize.isInText)) ||
          (state.md_inside && stream.current().indexOf(">") > -1)) {
        // 设置状态为 inlineNormal
        state.f = inlineNormal;
        // 设置状态为 blockNormal
        state.block = blockNormal;
        // 重置 htmlState
        state.htmlState = null;
      }
    }
    // 返回样式
    return style;
  }

  // 本地模式
  function local(stream, state) {
    // 获取当前列表缩进
    var currListInd = state.listStack[state.listStack.length - 1] || 0;
    // 判断是否已经退出列表
    var hasExitedList = state.indentation < currListInd;
    // 计算最大 fencedEndInd
    var maxFencedEndInd = currListInd + 3;
    // 如果有 fencedEndRE 并且缩进小于等于 maxFencedEndInd 并且（已经退出列表或者匹配 fencedEndRE）
    if (state.fencedEndRE && state.indentation <= maxFencedEndInd && (hasExitedList || stream.match(state.fencedEndRE))) {
      // 如果 highlightFormatting 为真，则设置 formatting 为 "code-block"
      if (modeCfg.highlightFormatting) state.formatting = "code-block";
      var returnType;
      // 如果没有退出列表，则获取类型
      if (!hasExitedList) returnType = getType(state)
      // 重置 localMode 和 localState
      state.localMode = state.localState = null;
      // 设置状态为 blockNormal
      state.block = blockNormal;
      // 设置状态为 inlineNormal
      state.f = inlineNormal;
      // 重置 fencedEndRE
      state.fencedEndRE = null;
      // 重置 code
      state.code = 0
      // 设置当前行的 fencedCodeEnd 为 true
      state.thisLine.fencedCodeEnd = true;
      // 如果已经退出列表，则返回 switchBlock
      if (hasExitedList) return switchBlock(stream, state, state.block);
      // 返回类型
      return returnType;
    } else if (state.localMode) {
      // 如果存在 localMode，则调用其 token 方法
      return state.localMode.token(stream, state.localState);
    } else {
      // 跳过流中剩余部分
      stream.skipToEnd();
      // 返回代码类型
      return tokenTypes.code;
    }
  }

  // 内联
  function getType(state) {
    // 创建样式数组
    var styles = [];
    # 如果正在进行格式化处理
    if (state.formatting) {
      # 将格式化类型添加到样式数组中
      styles.push(tokenTypes.formatting);

      # 如果格式化类型是字符串，则转换为数组
      if (typeof state.formatting === "string") state.formatting = [state.formatting];

      # 遍历格式化类型数组
      for (var i = 0; i < state.formatting.length; i++) {
        # 将格式化类型和具体类型添加到样式数组中
        styles.push(tokenTypes.formatting + "-" + state.formatting[i]);

        # 如果是标题类型的格式化，添加具体标题级别的样式
        if (state.formatting[i] === "header") {
          styles.push(tokenTypes.formatting + "-" + state.formatting[i] + "-" + state.header);
        }

        # 对于引用类型的格式化，根据引用深度添加样式，或者添加错误样式
        if (state.formatting[i] === "quote") {
          if (!modeCfg.maxBlockquoteDepth || modeCfg.maxBlockquoteDepth >= state.quote) {
            styles.push(tokenTypes.formatting + "-" + state.formatting[i] + "-" + state.quote);
          } else {
            styles.push("error");
          }
        }
      }
    }

    # 如果存在未完成的任务
    if (state.taskOpen) {
      # 添加元信息样式
      styles.push("meta");
      return styles.length ? styles.join(' ') : null;
    }
    # 如果任务已完成
    if (state.taskClosed) {
      # 添加属性样式
      styles.push("property");
      return styles.length ? styles.join(' ') : null;
    }

    # 如果存在链接地址
    if (state.linkHref) {
      # 添加链接样式和 URL 样式
      styles.push(tokenTypes.linkHref, "url");
    } else { # 只对非 URL 文本应用内联样式
      if (state.strong) { styles.push(tokenTypes.strong); }
      if (state.em) { styles.push(tokenTypes.em); }
      if (state.strikethrough) { styles.push(tokenTypes.strikethrough); }
      if (state.emoji) { styles.push(tokenTypes.emoji); }
      if (state.linkText) { styles.push(tokenTypes.linkText); }
      if (state.code) { styles.push(tokenTypes.code); }
      if (state.image) { styles.push(tokenTypes.image); }
      if (state.imageAltText) { styles.push(tokenTypes.imageAltText, "link"); }
      if (state.imageMarker) { styles.push(tokenTypes.imageMarker); }
    }

    # 如果存在标题级别
    if (state.header) { styles.push(tokenTypes.header, tokenTypes.header + "-" + state.header); }
    // 如果当前状态为引用状态
    if (state.quote) {
      // 将引用样式添加到样式数组中
      styles.push(tokenTypes.quote);

      // 添加 `quote-#` 样式，其中 `#` 的最大值为 modeCfg.maxBlockquoteDepth
      if (!modeCfg.maxBlockquoteDepth || modeCfg.maxBlockquoteDepth >= state.quote) {
        styles.push(tokenTypes.quote + "-" + state.quote);
      } else {
        styles.push(tokenTypes.quote + "-" + modeCfg.maxBlockquoteDepth);
      }
    }

    // 如果当前状态为列表状态
    if (state.list !== false) {
      // 计算列表模数
      var listMod = (state.listStack.length - 1) % 3;
      // 根据模数添加不同的列表样式
      if (!listMod) {
        styles.push(tokenTypes.list1);
      } else if (listMod === 1) {
        styles.push(tokenTypes.list2);
      } else {
        styles.push(tokenTypes.list3);
      }
    }

    // 如果当前状态为有尾随空格和换行的状态
    if (state.trailingSpaceNewLine) {
      styles.push("trailing-space-new-line");
    } else if (state.trailingSpace) {
      // 如果有尾随空格，根据空格数量添加不同的样式
      styles.push("trailing-space-" + (state.trailingSpace % 2 ? "a" : "b"));
    }

    // 如果样式数组不为空，返回样式数组的字符串形式，否则返回 null
    return styles.length ? styles.join(' ') : null;
  }

  // 处理文本
  function handleText(stream, state) {
    // 如果匹配到文本正则表达式
    if (stream.match(textRE, true)) {
      // 返回当前状态的类型
      return getType(state);
    }
    // 否则返回 undefined
    return undefined;
  }

  // 处理普通内联文本
  function inlineNormal(stream, state) {
    // 调用 state 的 text 方法处理文本，获取样式
    var style = state.text(stream, state);
    // 如果样式不为 undefined，直接返回样式
    if (typeof style !== 'undefined')
      return style;

    // 如果当前状态为列表状态，清空列表状态并返回当前状态的类型
    if (state.list) { // List marker (*, +, -, 1., etc)
      state.list = null;
      return getType(state);
    }

    // 如果当前状态为任务列表状态
    if (state.taskList) {
      // 匹配任务列表正则表达式，判断任务是否打开或关闭
      var taskOpen = stream.match(taskListRE, true)[1] === " ";
      if (taskOpen) state.taskOpen = true;
      else state.taskClosed = true;
      // 如果需要高亮格式，则设置当前状态的格式为 "task"
      if (modeCfg.highlightFormatting) state.formatting = "task";
      state.taskList = false;
      return getType(state);
    }

    // 重置任务打开和关闭状态
    state.taskOpen = false;
    state.taskClosed = false;

    // 如果当前状态为标题状态，并且匹配到标题的正则表达式
    if (state.header && stream.match(/^#+$/, true)) {
      // 如果需要高亮格式，则设置当前状态的格式为 "header"
      if (modeCfg.highlightFormatting) state.formatting = "header";
      return getType(state);
    }

    // 获取下一个字符
    var ch = stream.next();

    // 匹配下一行上的链接标题
    // 如果状态中存在链接标题
    if (state.linkTitle) {
      // 将链接标题状态设置为false
      state.linkTitle = false;
      // 记录当前字符
      var matchCh = ch;
      // 如果当前字符是'('，则将匹配字符设置为')'
      if (ch === '(') {
        matchCh = ')';
      }
      // 将匹配字符转义，用于后续正则表达式匹配
      matchCh = (matchCh+'').replace(/([.?*+^\[\]\\(){}|-])/g, "\\$1");
      // 构建正则表达式，用于匹配链接地址
      var regex = '^\\s*(?:[^' + matchCh + '\\\\]+|\\\\\\\\|\\\\.)' + matchCh;
      // 如果匹配成功，则返回链接地址类型的token
      if (stream.match(new RegExp(regex), true)) {
        return tokenTypes.linkHref;
      }
    }

    // 如果代码块发生改变，可能需要在GFM模式下更新
    if (ch === '`') {
      // 保存之前的格式化状态
      var previousFormatting = state.formatting;
      // 如果启用了高亮格式化，将格式化状态设置为"code"
      if (modeCfg.highlightFormatting) state.formatting = "code";
      // 吃掉连续的反引号
      stream.eatWhile('`');
      // 获取连续反引号的数量
      var count = stream.current().length
      // 如果当前不在代码块中且不在引用块中，且反引号数量为1，则将状态设置为代码块，并返回对应的token类型
      if (state.code == 0 && (!state.quote || count == 1)) {
        state.code = count
        return getType(state)
      } else if (count == state.code) { // 必须完全匹配
        var t = getType(state)
        state.code = 0
        return t
      } else {
        // 恢复之前的格式化状态，并返回对应的token类型
        state.formatting = previousFormatting
        return getType(state)
      }
    } else if (state.code) {
      // 如果在代码块中，则返回对应的token类型
      return getType(state);
    }

    // 如果当前字符为反斜杠
    if (ch === '\\') {
      // 向前移动一个字符
      stream.next();
      // 如果启用了高亮格式化
      if (modeCfg.highlightFormatting) {
        // 获取当前token类型
        var type = getType(state);
        // 设置格式化转义的token类型
        var formattingEscape = tokenTypes.formatting + "-escape";
        // 如果存在token类型，则返回对应的token类型和格式化转义的token类型，否则只返回格式化转义的token类型
        return type ? type + " " + formattingEscape : formattingEscape;
      }
    }

    // 如果当前字符为'!'且紧跟着一个方括号和链接地址
    if (ch === '!' && stream.match(/\[[^\]]*\] ?(?:\(|\[)/, false)) {
      // 设置图片标记为true
      state.imageMarker = true;
      // 设置图片状态为true
      state.image = true;
      // 如果启用了高亮格式化，将格式化状态设置为"image"
      if (modeCfg.highlightFormatting) state.formatting = "image";
      // 返回对应的token类型
      return getType(state);
    }

    // 如果当前字符为'['且之前存在图片标记且紧跟着图片的描述文本和链接地址
    if (ch === '[' && state.imageMarker && stream.match(/[^\]]*\](\(.*?\)| ?\[.*?\])/, false)) {
      // 将图片标记设置为false
      state.imageMarker = false;
      // 设置图片描述文本状态为true
      state.imageAltText = true
      // 如果启用了高亮格式化，将格式化状态设置为"image"
      if (modeCfg.highlightFormatting) state.formatting = "image";
      // 返回对应的token类型
      return getType(state);
    }
    # 如果当前字符是 ']' 并且状态中存在图片的替代文本
    if (ch === ']' && state.imageAltText) {
      # 如果启用了高亮格式设置，将状态的格式设置为 "image"
      if (modeCfg.highlightFormatting) state.formatting = "image";
      # 获取当前类型
      var type = getType(state);
      # 重置状态中的图片相关标记
      state.imageAltText = false;
      state.image = false;
      state.inline = state.f = linkHref;
      # 返回当前类型
      return type;
    }

    # 如果当前字符是 '[' 并且状态中不存在图片
    if (ch === '[' && !state.image) {
      # 如果状态中存在链接文本并且匹配到了链接文本的结束标记
      if (state.linkText && stream.match(/^.*?\]/)) return getType(state)
      # 设置状态中存在链接文本的标记
      state.linkText = true;
      # 如果启用了高亮格式设置，将状态的格式设置为 "link"
      if (modeCfg.highlightFormatting) state.formatting = "link";
      # 返回当前类型
      return getType(state);
    }

    # 如果当前字符是 ']' 并且状态中存在链接文本
    if (ch === ']' && state.linkText) {
      # 如果启用了高亮格式设置，将状态的格式设置为 "link"
      if (modeCfg.highlightFormatting) state.formatting = "link";
      # 获取当前类型
      var type = getType(state);
      # 重置状态中的链接文本相关标记
      state.linkText = false;
      state.inline = state.f = stream.match(/\(.*?\)| ?\[.*?\]/, false) ? linkHref : inlineNormal
      # 返回当前类型
      return type;
    }

    # 如果当前字符是 '<' 并且匹配到了以 http、https、ftp 或 ftps 开头的链接
    if (ch === '<' && stream.match(/^(https?|ftps?):\/\/(?:[^\\>]|\\.)+>/, false)) {
      state.f = state.inline = linkInline;
      # 如果启用了高亮格式设置，将状态的格式设置为 "link"
      if (modeCfg.highlightFormatting) state.formatting = "link";
      # 获取当前类型
      var type = getType(state);
      # 如果存在类型，则在类型后面添加一个空格，否则设置类型为空字符串
      if (type){
        type += " ";
      } else {
        type = "";
      }
      # 返回类型和链接内联的 token 类型
      return type + tokenTypes.linkInline;
    }

    # 如果当前字符是 '<' 并且匹配到了以非空格或 '>' 开头的邮箱链接
    if (ch === '<' && stream.match(/^[^> \\]+@(?:[^\\>]|\\.)+>/, false)) {
      state.f = state.inline = linkInline;
      # 如果启用了高亮格式设置，将状态的格式设置为 "link"
      if (modeCfg.highlightFormatting) state.formatting = "link";
      # 获取当前类型
      var type = getType(state);
      # 如果存在类型，则在类型后面添加一个空格，否则设置类型为空字符串
      if (type){
        type += " ";
      } else {
        type = "";
      }
      # 返回类型和邮箱链接的 token 类型
      return type + tokenTypes.linkEmail;
    }
    # 如果配置为 XML 模式，并且当前字符为 '<'，并且下一个内容匹配 XML 标签的正则表达式
    if (modeCfg.xml && ch === '<' && stream.match(/^(!--|\?|!\[CDATA\[|[a-z][a-z0-9-]*(?:\s+[a-z_:.\-]+(?:\s*=\s*[^>]+)?)*\s*(?:>|$))/i, false)) {
      # 获取当前标签的结束位置
      var end = stream.string.indexOf(">", stream.pos);
      # 如果找到结束位置
      if (end != -1) {
        # 获取标签的属性部分
        var atts = stream.string.substring(stream.start, end);
        # 如果属性中包含 markdown=1，则设置状态为 md_inside
        if (/markdown\s*=\s*('|"){0,1}1('|"){0,1}/.test(atts)) state.md_inside = true;
      }
      # 回退一个字符
      stream.backUp(1);
      # 设置 HTML 状态为起始状态
      state.htmlState = CodeMirror.startState(htmlMode);
      # 返回切换到 HTML 模式的结果
      return switchBlock(stream, state, htmlBlock);
    }

    # 如果配置为 XML 模式，并且当前字符为 '<'，并且下一个内容匹配闭合标签的正则表达式
    if (modeCfg.xml && ch === '<' && stream.match(/^\/\w*?>/)) {
      # 设置状态为非 md_inside
      state.md_inside = false;
      # 返回标签类型为 "tag"
      return "tag";
    } else if (ch === "*" || ch === "_") {
      // 如果当前字符是 * 或者 _，则执行以下操作
      var len = 1, before = stream.pos == 1 ? " " : stream.string.charAt(stream.pos - 2)
      // 初始化 len 为 1，before 为当前字符前一个字符，如果当前字符在字符串的第一个位置，则 before 为一个空格
      while (len < 3 && stream.eat(ch)) len++
      // 当 len 小于 3 并且当前字符与后续字符相同，则 len 自增
      var after = stream.peek() || " "
      // 获取当前字符的后一个字符，如果不存在则为一个空格
      // See http://spec.commonmark.org/0.27/#emphasis-and-strong-emphasis
      // 参考链接，关于强调和加粗的规范
      var leftFlanking = !/\s/.test(after) && (!punctuation.test(after) || /\s/.test(before) || punctuation.test(before))
      var rightFlanking = !/\s/.test(before) && (!punctuation.test(before) || /\s/.test(after) || punctuation.test(after))
      // 判断左右是否有空格或标点符号
      var setEm = null, setStrong = null
      // 初始化 setEm 和 setStrong 为 null
      if (len % 2) { // Em
        // 如果 len 为奇数，则执行以下操作
        if (!state.em && leftFlanking && (ch === "*" || !rightFlanking || punctuation.test(before)))
          setEm = true
        else if (state.em == ch && rightFlanking && (ch === "*" || !leftFlanking || punctuation.test(after)))
          setEm = false
      }
      if (len > 1) { // Strong
        // 如果 len 大于 1，则执行以下操作
        if (!state.strong && leftFlanking && (ch === "*" || !rightFlanking || punctuation.test(before)))
          setStrong = true
        else if (state.strong == ch && rightFlanking && (ch === "*" || !leftFlanking || punctuation.test(after)))
          setStrong = false
      }
      if (setStrong != null || setEm != null) {
        // 如果 setStrong 或 setEm 不为 null，则执行以下操作
        if (modeCfg.highlightFormatting) state.formatting = setEm == null ? "strong" : setStrong == null ? "em" : "strong em"
        if (setEm === true) state.em = ch
        if (setStrong === true) state.strong = ch
        var t = getType(state)
        if (setEm === false) state.em = false
        if (setStrong === false) state.strong = false
        return t
      }
    } else if (ch === ' ') {
      // 如果当前字符是空格，则执行以下操作
      if (stream.eat('*') || stream.eat('_')) { // Probably surrounded by spaces
        // 如果后续字符是 * 或者 _，则执行以下操作
        if (stream.peek() === ' ') { // Surrounded by spaces, ignore
          // 如果后续字符是空格，则忽略当前操作
          return getType(state);
        } else { // Not surrounded by spaces, back up pointer
          // 如果后续字符不是空格，则后退一个字符
          stream.backUp(1);
        }
      }
    }
    // 如果 modeCfg.strikethrough 为真，则执行以下代码块
    if (modeCfg.strikethrough) {
      // 如果当前字符为'~'，并且后面的字符都是'~'，则执行以下代码块
      if (ch === '~' && stream.eatWhile(ch)) {
        // 如果当前状态为strikethrough，则移除strikethrough
        if (state.strikethrough) {
          // 如果 modeCfg.highlightFormatting 为真，则将当前格式设置为"strikethrough"
          if (modeCfg.highlightFormatting) state.formatting = "strikethrough";
          // 获取当前类型并返回
          var t = getType(state);
          state.strikethrough = false;
          return t;
        } else if (stream.match(/^[^\s]/, false)) {
          // 如果当前字符不是空格，则添加strikethrough
          state.strikethrough = true;
          // 如果 modeCfg.highlightFormatting 为真，则将当前格式设置为"strikethrough"
          if (modeCfg.highlightFormatting) state.formatting = "strikethrough";
          // 获取当前类型并返回
          return getType(state);
        }
      } else if (ch === ' ') {
        if (stream.match(/^~~/, true)) {
          // 如果前后都是'~'，则返回当前类型
          if (stream.peek() === ' ') {
            return getType(state);
          } else {
            // 如果前后不都是'~'，则回退指针
            stream.backUp(2);
          }
        }
      }
    }

    // 如果 modeCfg.emoji 为真，并且当前字符为':'，并且后面的字符符合表情的正则表达式，则执行以下代码块
    if (modeCfg.emoji && ch === ":" && stream.match(/^(?:[a-z_\d+][a-z_\d+-]*|\-[a-z_\d+][a-z_\d+-]*):/)) {
      state.emoji = true;
      // 如果 modeCfg.highlightFormatting 为真，则将当前格式设置为"emoji"
      if (modeCfg.highlightFormatting) state.formatting = "emoji";
      // 获取当前类型并返回
      var retType = getType(state);
      state.emoji = false;
      return retType;
    }

    // 如果当前字符为' '，则执行以下代码块
    if (ch === ' ') {
      if (stream.match(/^ +$/, false)) {
        // 如果后面都是空格，则增加尾随空格计数
        state.trailingSpace++;
      } else if (state.trailingSpace) {
        // 如果之前有尾随空格，则设置尾随空格新行为真
        state.trailingSpaceNewLine = true;
      }
    }

    // 获取当前类型并返回
    return getType(state);
  }

  // 内联链接处理函数
  function linkInline(stream, state) {
    // 获取下一个字符
    var ch = stream.next();

    // 如果当前字符为">"，则执行以下代码块
    if (ch === ">") {
      state.f = state.inline = inlineNormal;
      // 如果 modeCfg.highlightFormatting 为真，则将当前格式设置为"link"
      if (modeCfg.highlightFormatting) state.formatting = "link";
      // 获取当前类型并返回
      var type = getType(state);
      if (type){
        type += " ";
      } else {
        type = "";
      }
      return type + tokenTypes.linkInline;
    }

    // 匹配非">"的字符
    stream.match(/^[^>]+/, true);

    return tokenTypes.linkInline;
  }

  // 链接地址处理函数
  function linkHref(stream, state) {
    // 检查是否是空格，如果是则返回NULL（避免标记空格）
    if(stream.eatSpace()){
      return null;
  }
  // 获取下一个字符
  var ch = stream.next();
  // 如果下一个字符是 '(' 或者 '['
  if (ch === '(' || ch === '[') {
    // 设置状态的 f 和 inline 属性为获取链接地址的函数
    state.f = state.inline = getLinkHrefInside(ch === "(" ? ")" : "]");
    // 如果配置允许高亮格式，则设置状态的 formatting 属性为 "link-string"
    if (modeCfg.highlightFormatting) state.formatting = "link-string";
    // 设置状态的 linkHref 属性为 true
    state.linkHref = true;
    // 返回状态的类型
    return getType(state);
  }
  // 返回错误类型
  return 'error';
}

// 定义链接的正则表达式
var linkRE = {
  ")": /^(?:[^\\\(\)]|\\.|\((?:[^\\\(\)]|\\.)*\))*?(?=\))/,
  "]": /^(?:[^\\\[\]]|\\.|\[(?:[^\\\[\]]|\\.)*\])*?(?=\])/
}

// 获取链接地址的函数
function getLinkHrefInside(endChar) {
  return function(stream, state) {
    // 获取下一个字符
    var ch = stream.next();

    // 如果下一个字符是结束字符
    if (ch === endChar) {
      // 设置状态的 f 和 inline 属性为 inlineNormal 函数
      state.f = state.inline = inlineNormal;
      // 如果配置允许高亮格式，则设置状态的 formatting 属性为 "link-string"
      if (modeCfg.highlightFormatting) state.formatting = "link-string";
      // 获取状态的类型并返回
      var returnState = getType(state);
      // 设置状态的 linkHref 属性为 false
      state.linkHref = false;
      return returnState;
    }

    // 匹配链接地址的正则表达式
    stream.match(linkRE[endChar])
    // 设置状态的 linkHref 属性为 true
    state.linkHref = true;
    // 返回状态的类型
    return getType(state);
  };
}

// 处理脚注链接
function footnoteLink(stream, state) {
  // 如果匹配到脚注链接的格式
  if (stream.match(/^([^\]\\]|\\.)*\]:/, false)) {
    // 设置状态的 f 属性为 footnoteLinkInside 函数
    state.f = footnoteLinkInside;
    // 消耗掉 [
    stream.next();
    // 如果配置允许高亮格式，则设置状态的 formatting 属性为 "link"
    if (modeCfg.highlightFormatting) state.formatting = "link";
    // 设置状态的 linkText 属性为 true
    state.linkText = true;
    // 返回状态的类型
    return getType(state);
  }
  // 调用 switchInline 函数处理内联元素
  return switchInline(stream, state, inlineNormal);
}

// 处理脚注链接内部
function footnoteLinkInside(stream, state) {
  // 如果匹配到脚注链接的结束格式
  if (stream.match(/^\]:/, true)) {
    // 设置状态的 f 和 inline 属性为 footnoteUrl 函数
    state.f = state.inline = footnoteUrl;
    // 如果配置允许高亮格式，则设置状态的 formatting 属性为 "link"
    if (modeCfg.highlightFormatting) state.formatting = "link";
    // 获取状态的类型并返回
    var returnType = getType(state);
    // 设置状态的 linkText 属性为 false
    state.linkText = false;
    return returnType;
  }

  // 匹配脚注链接的文本部分
  stream.match(/^([^\]\\]|\\.)+/, true);

  return tokenTypes.linkText;
}

// 处理脚注链接的 URL 部分
function footnoteUrl(stream, state) {
  // 检查是否是空格，如果是则返回 null（避免标记空格）
  if(stream.eatSpace()){
    return null;
  }
  // 匹配 URL
  stream.match(/^[^\s]+/, true);
  // 检查链接标题
    // 如果流的下一个字符是未定义的，表示行末尾，设置标志以检查下一行
    if (stream.peek() === undefined) {
      state.linkTitle = true;
    } else { // 如果行上还有内容，检查是否为链接标题
      stream.match(/^(?:\s+(?:"(?:[^"\\]|\\\\|\\.)+"|'(?:[^'\\]|\\\\|\\.)+'|\((?:[^)\\]|\\\\|\\.)+\)))?/, true);
    }
    state.f = state.inline = inlineNormal; // 设置状态为内联普通状态
    return tokenTypes.linkHref + " url"; // 返回链接的类型和 URL
  }

  var mode = {
    startState: function() { // 定义初始状态
      return {
        f: blockNormal, // 设置初始状态为普通块状态

        prevLine: {stream: null}, // 上一行的流
        thisLine: {stream: null}, // 当前行的流

        block: blockNormal, // 块状态为普通块状态
        htmlState: null, // HTML 状态为空
        indentation: 0, // 缩进为 0

        inline: inlineNormal, // 内联状态为普通内联状态
        text: handleText, // 文本处理函数

        formatting: false, // 格式化标志为假
        linkText: false, // 链接文本标志为假
        linkHref: false, // 链接地址标志为假
        linkTitle: false, // 链接标题标志为假
        code: 0, // 代码标志为 0
        em: false, // 强调标志为假
        strong: false, // 加粗标志为假
        header: 0, // 标题级别为 0
        setext: 0, // Setext 标志为 0
        hr: false, // 水平线标志为假
        taskList: false, // 任务列表标志为假
        list: false, // 列表标志为假
        listStack: [], // 列表堆栈为空
        quote: 0, // 引用标志为 0
        trailingSpace: 0, // 尾随空格为 0
        trailingSpaceNewLine: false, // 尾随空格换行标志为假
        strikethrough: false, // 删除线标志为假
        emoji: false, // 表情标志为假
        fencedEndRE: null // 围栏结束正则为空
      };
    },
    // 复制给定状态对象的属性，创建并返回一个新的状态对象
    copyState: function(s) {
      return {
        // 复制属性 f
        f: s.f,
        // 复制属性 prevLine
        prevLine: s.prevLine,
        // 复制属性 thisLine
        thisLine: s.thisLine,
        // 复制属性 block
        block: s.block,
        // 复制属性 htmlState，并使用 htmlMode 复制其状态
        htmlState: s.htmlState && CodeMirror.copyState(htmlMode, s.htmlState),
        // 复制属性 indentation
        indentation: s.indentation,
        // 复制属性 localMode
        localMode: s.localMode,
        // 如果 localMode 存在，则使用其状态复制 localState
        localState: s.localMode ? CodeMirror.copyState(s.localMode, s.localState) : null,
        // 复制属性 inline
        inline: s.inline,
        // 复制属性 text
        text: s.text,
        // 设置 formatting 为 false
        formatting: false,
        // 复制属性 linkText
        linkText: s.linkText,
        // 复制属性 linkTitle
        linkTitle: s.linkTitle,
        // 复制属性 linkHref
        linkHref: s.linkHref,
        // 复制属性 code
        code: s.code,
        // 复制属性 em
        em: s.em,
        // 复制属性 strong
        strong: s.strong,
        // 复制属性 strikethrough
        strikethrough: s.strikethrough,
        // 复制属性 emoji
        emoji: s.emoji,
        // 复制属性 header
        header: s.header,
        // 复制属性 setext
        setext: s.setext,
        // 复制属性 hr
        hr: s.hr,
        // 复制属性 taskList
        taskList: s.taskList,
        // 复制属性 list
        list: s.list,
        // 复制属性 listStack，并复制其数组内容
        listStack: s.listStack.slice(0),
        // 复制属性 quote
        quote: s.quote,
        // 复制属性 indentedCode
        indentedCode: s.indentedCode,
        // 复制属性 trailingSpace
        trailingSpace: s.trailingSpace,
        // 复制属性 trailingSpaceNewLine
        trailingSpaceNewLine: s.trailingSpaceNewLine,
        // 复制属性 md_inside
        md_inside: s.md_inside,
        // 复制属性 fencedEndRE
        fencedEndRE: s.fencedEndRE
      };
    },
    // 定义 token 方法，用于处理代码中的标记
    token: function(stream, state) {

      // 重置 state.formatting
      state.formatting = false;

      // 如果当前流不等于 state.thisLine.stream
      if (stream != state.thisLine.stream) {
        // 重置 state.header 和 state.hr
        state.header = 0;
        state.hr = false;

        // 如果流匹配空白行
        if (stream.match(/^\s*$/, true)) {
          // 调用 blankLine 方法处理空白行
          blankLine(state);
          return null;
        }

        // 重置 state.prevLine 和 state.thisLine
        state.prevLine = state.thisLine
        state.thisLine = {stream: stream}

        // 重置 state.taskList
        state.taskList = false;

        // 重置 state.trailingSpace 和 state.trailingSpaceNewLine
        state.trailingSpace = 0;
        state.trailingSpaceNewLine = false;

        // 如果没有 localState
        if (!state.localState) {
          state.f = state.block;
          // 如果 state.f 不等于 htmlBlock
          if (state.f != htmlBlock) {
            // 匹配行首空白，计算缩进
            var indentation = stream.match(/^\s*/, true)[0].replace(/\t/g, expandedTab).length;
            state.indentation = indentation;
            state.indentationDiff = null;
            // 如果缩进大于 0，返回 null
            if (indentation > 0) return null;
          }
        }
      }
      // 调用 state.f 方法处理流
      return state.f(stream, state);
    },

    // 定义 innerMode 方法，用于处理内部模式
    innerMode: function(state) {
      // 如果 state.block 等于 htmlBlock，返回 state.htmlState 和 htmlMode
      if (state.block == htmlBlock) return {state: state.htmlState, mode: htmlMode};
      // 如果有 localState，返回 state.localState 和 state.localMode
      if (state.localState) return {state: state.localState, mode: state.localMode};
      // 否则返回 state 和 mode
      return {state: state, mode: mode};
    },

    // 定义 indent 方法，用于处理缩进
    indent: function(state, textAfter, line) {
      // 如果 state.block 等于 htmlBlock 并且 htmlMode.indent 存在，调用 htmlMode.indent 方法处理缩进
      if (state.block == htmlBlock && htmlMode.indent) return htmlMode.indent(state.htmlState, textAfter, line)
      // 如果有 localState 并且 state.localMode.indent 存在，调用 state.localMode.indent 方法处理缩进
      if (state.localState && state.localMode.indent) return state.localMode.indent(state.localState, textAfter, line)
      // 否则返回 CodeMirror.Pass
      return CodeMirror.Pass
    },

    // 定义 blankLine 方法
    blankLine: blankLine,

    // 定义 getType 方法
    getType: getType,

    // 定义 blockCommentStart、blockCommentEnd、closeBrackets 和 fold 属性
    blockCommentStart: "<!--",
    blockCommentEnd: "-->",
    closeBrackets: "()[]{}''\"\"``",
    fold: "markdown"
  };
  // 返回 mode
  return mode;
// 定义 MIME 类型为 "text/markdown" 的 CodeMirror 模式为 "markdown"
CodeMirror.defineMIME("text/markdown", "markdown");

// 定义 MIME 类型为 "text/x-markdown" 的 CodeMirror 模式为 "markdown"
CodeMirror.defineMIME("text/x-markdown", "markdown");

// 定义 Python 语言的 CodeMirror 模式
// 以下为 CodeMirror 源码，版权归 Marijn Haverbeke 及其他作者所有，基于 MIT 许可证发布
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境
  else
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义单词的正则表达式
  function wordRegexp(words) {
  // 创建一个新的正则表达式对象，用于匹配单词边界的模式
  return new RegExp("^((" + words.join(")|(") + "))\\b");
}

// 定义包含逻辑运算符的单词的正则表达式
var wordOperators = wordRegexp(["and", "or", "not", "is"]);
// 定义常见关键字列表
var commonKeywords = ["as", "assert", "break", "class", "continue",
                    "def", "del", "elif", "else", "except", "finally",
                    "for", "from", "global", "if", "import",
                    "lambda", "pass", "raise", "return",
                    "try", "while", "with", "yield", "in"];
// 定义常见内置函数列表
var commonBuiltins = ["abs", "all", "any", "bin", "bool", "bytearray", "callable", "chr",
                    "classmethod", "compile", "complex", "delattr", "dict", "dir", "divmod",
                    "enumerate", "eval", "filter", "float", "format", "frozenset",
                    "getattr", "globals", "hasattr", "hash", "help", "hex", "id",
                    "input", "int", "isinstance", "issubclass", "iter", "len",
                    "list", "locals", "map", "max", "memoryview", "min", "next",
                    "object", "oct", "open", "ord", "pow", "property", "range",
                    "repr", "reversed", "round", "set", "setattr", "slice",
                    "sorted", "staticmethod", "str", "sum", "super", "tuple",
                    "type", "vars", "zip", "__import__", "NotImplemented",
                    "Ellipsis", "__debug__"];
// 将 Python 语言的关键字和内置函数注册为代码提示的单词
CodeMirror.registerHelper("hintWords", "python", commonKeywords.concat(commonBuiltins));

// 返回作用域数组中的顶层作用域
function top(state) {
  return state.scopes[state.scopes.length - 1];
}

// 定义 Python 语言的代码模式
CodeMirror.defineMode("python", function(conf, parserConf) {
  var ERRORCLASS = "error";

  // 定义分隔符的正则表达式模式
  var delimiters = parserConf.delimiters || parserConf.singleDelimiters || /^[\(\)\[\]\{\}@,:`=;\.\\]/;
  // (与旧的、繁琐的配置系统向后兼容)
    # 定义操作符数组，包括单操作符、双操作符、双分隔符、三分隔符和自定义操作符
    var operators = [parserConf.singleOperators, parserConf.doubleOperators, parserConf.doubleDelimiters, parserConf.tripleDelimiters,
                     parserConf.operators || /^([-+*/%\/&|^]=?|[<>=]+|\/\/=?|\*\*=?|!=|[~!@]|\.\.\.)/]
    # 遍历操作符数组，如果某个元素为空，则删除该元素
    for (var i = 0; i < operators.length; i++) if (!operators[i]) operators.splice(i--, 1)

    # 定义悬挂缩进，如果未定义则使用默认缩进单位
    var hangingIndent = parserConf.hangingIndent || conf.indentUnit;

    # 定义关键字和内置函数数组，如果有额外关键字和内置函数则添加到数组中
    var myKeywords = commonKeywords, myBuiltins = commonBuiltins;
    if (parserConf.extra_keywords != undefined)
      myKeywords = myKeywords.concat(parserConf.extra_keywords);

    if (parserConf.extra_builtins != undefined)
      myBuiltins = myBuiltins.concat(parserConf.extra_builtins);

    # 判断是否为 Python 3 版本，如果是则定义对应的标识符、关键字和内置函数
    var py3 = !(parserConf.version && Number(parserConf.version) < 3)
    if (py3) {
      # 定义标识符正则表达式和特定于 Python 3 的关键字和内置函数
      var identifiers = parserConf.identifiers|| /^[_A-Za-z\u00A1-\uFFFF][_A-Za-z0-9\u00A1-\uFFFF]*/;
      myKeywords = myKeywords.concat(["nonlocal", "False", "True", "None", "async", "await"]);
      myBuiltins = myBuiltins.concat(["ascii", "bytes", "exec", "print"]);
      var stringPrefixes = new RegExp("^(([rbuf]|(br)|(fr))?('{3}|\"{3}|['\"]))", "i");
    } else {
      # 定义标识符正则表达式和特定于 Python 2 的关键字和内置函数
      var identifiers = parserConf.identifiers|| /^[_A-Za-z][_A-Za-z0-9]*/;
      myKeywords = myKeywords.concat(["exec", "print"]);
      myBuiltins = myBuiltins.concat(["apply", "basestring", "buffer", "cmp", "coerce", "execfile",
                                      "file", "intern", "long", "raw_input", "reduce", "reload",
                                      "unichr", "unicode", "xrange", "False", "True", "None"]);
      var stringPrefixes = new RegExp("^(([rubf]|(ur)|(br))?('{3}|\"{3}|['\"]))", "i");
    }
    # 定义关键字和内置函数的正则表达式
    var keywords = wordRegexp(myKeywords);
    var builtins = wordRegexp(myBuiltins);

    # tokenizers
    // 定义函数 tokenBase，接受输入流和状态对象作为参数
    function tokenBase(stream, state) {
      // 检查是否在行首且上一个标记不是反斜杠
      var sol = stream.sol() && state.lastToken != "\\"
      // 如果在行首，记录缩进值
      if (sol) state.indent = stream.indentation()
      // 处理作用域变化
      if (sol && top(state).type == "py") {
        // 获取当前作用域的偏移量
        var scopeOffset = top(state).offset;
        // 如果遇到空白字符，检查是否需要创建新的 Python 作用域
        if (stream.eatSpace()) {
          var lineOffset = stream.indentation();
          if (lineOffset > scopeOffset)
            pushPyScope(state);
          else if (lineOffset < scopeOffset && dedent(stream, state) && stream.peek() != "#")
            state.errorToken = true;
          return null;
        } else {
          // 调用 tokenBaseInner 处理非空白字符
          var style = tokenBaseInner(stream, state);
          // 如果需要缩进，添加错误样式
          if (scopeOffset > 0 && dedent(stream, state))
            style += " " + ERRORCLASS;
          return style;
        }
      }
      // 调用 tokenBaseInner 处理非 Python 作用域的情况
      return tokenBaseInner(stream, state);
    }

    // 定义函数 tokenStringFactory，接受定界符和外部标记函数作为参数
    function tokenStringFactory(delimiter, tokenOuter) {
      // 去除定界符前的修饰符
      while ("rubf".indexOf(delimiter.charAt(0).toLowerCase()) >= 0)
        delimiter = delimiter.substr(1);

      // 检查是否为单行字符串
      var singleline = delimiter.length == 1;
      var OUTCLASS = "string";

      // 定义函数 tokenString，处理字符串标记
      function tokenString(stream, state) {
        while (!stream.eol()) {
          // 跳过非定界符、单引号、双引号和反斜杠的字符
          stream.eatWhile(/[^'"\\]/);
          // 处理反斜杠转义
          if (stream.eat("\\")) {
            stream.next();
            if (singleline && stream.eol())
              return OUTCLASS;
          } else if (stream.match(delimiter)) {
            // 遇到定界符时，切换回外部标记函数
            state.tokenize = tokenOuter;
            return OUTCLASS;
          } else {
            stream.eat(/['"]/);
          }
        }
        // 处理单行字符串的结束
        if (singleline) {
          if (parserConf.singleLineStringErrors)
            return ERRORCLASS;
          else
            state.tokenize = tokenOuter;
        }
        return OUTCLASS;
      }
      // 标记该函数为字符串标记函数
      tokenString.isString = true;
      return tokenString;
    }
    # 将 Python 作用域推入状态栈
    function pushPyScope(state) {
      # 当栈顶元素类型不是 "py" 时，弹出栈顶元素，直到栈顶元素类型为 "py"
      while (top(state).type != "py") state.scopes.pop()
      # 将新的 Python 作用域推入状态栈
      state.scopes.push({offset: top(state).offset + conf.indentUnit,
                         type: "py",
                         align: null})
    }
    
    # 将括号作用域推入状态栈
    function pushBracketScope(stream, state, type) {
      # 匹配括号前的空白字符或注释，确定对齐位置
      var align = stream.match(/^([\s\[\{\(]|#.*)*$/, false) ? null : stream.column() + 1
      # 将新的括号作用域推入状态栈
      state.scopes.push({offset: state.indent + hangingIndent,
                         type: type,
                         align: align})
    }
    
    # 减少缩进级别
    function dedent(stream, state) {
      # 获取当前行的缩进级别
      var indented = stream.indentation();
      # 当栈中作用域数量大于 1 且栈顶作用域的偏移量大于当前行的缩进级别时，弹出栈顶作用域
      while (state.scopes.length > 1 && top(state).offset > indented) {
        # 如果栈顶作用域类型不是 "py"，返回 true
        if (top(state).type != "py") return true;
        state.scopes.pop();
      }
      # 返回栈顶作用域的偏移量是否不等于当前行的缩进级别
      return top(state).offset != indented;
    }
    # 定义一个函数，用于对输入的流进行词法分析
    function tokenLexer(stream, state) {
      # 如果流的位置在行首，设置状态为行首
      if (stream.sol()) state.beginningOfLine = true;

      # 调用状态中的tokenize方法对流进行词法分析，获取样式
      var style = state.tokenize(stream, state);
      # 获取当前流中的内容
      var current = stream.current();

      # 处理装饰器
      if (state.beginningOfLine && current == "@")
        # 如果在行首且当前字符为@，则返回meta样式，否则根据py3返回operator或ERRORCLASS样式
        return stream.match(identifiers, false) ? "meta" : py3 ? "operator" : ERRORCLASS;

      # 如果当前字符不为空白字符，设置状态为非行首
      if (/\S/.test(current)) state.beginningOfLine = false;

      # 如果样式为"variable"或"builtin"，且上一个标记为"meta"，则样式设置为"meta"
      if ((style == "variable" || style == "builtin")
          && state.lastToken == "meta")
        style = "meta";

      # 处理作用域变化
      if (current == "pass" || current == "return")
        state.dedent += 1;

      # 如果当前字符为"lambda"，设置状态中的lambda为true
      if (current == "lambda") state.lambda = true;
      # 如果当前字符为":"且不是lambda函数且状态栈顶为"py"，则推入py作用域
      if (current == ":" && !state.lambda && top(state).type == "py")
        pushPyScope(state);

      # 如果当前字符长度为1且样式不为"string"或"comment"
      if (current.length == 1 && !/string|comment/.test(style)) {
        # 获取当前字符在"[({"中的索引
        var delimiter_index = "[({".indexOf(current);
        # 如果索引不为-1，根据索引推入相应的括号作用域
        if (delimiter_index != -1)
          pushBracketScope(stream, state, "])}".slice(delimiter_index, delimiter_index+1));

        delimiter_index = "])}".indexOf(current);
        # 如果索引不为-1
        if (delimiter_index != -1) {
          # 如果状态栈顶类型与当前字符相同，设置缩进为作用域出栈的偏移量减去悬挂缩进
          if (top(state).type == current) state.indent = state.scopes.pop().offset - hangingIndent
          # 否则返回ERRORCLASS样式
          else return ERRORCLASS;
        }
      }
      # 如果需要减少缩进且流在行尾且状态栈顶类型为"py"
      if (state.dedent > 0 && stream.eol() && top(state).type == "py") {
        # 如果状态栈长度大于1，出栈一个作用域
        if (state.scopes.length > 1) state.scopes.pop();
        # 减少需要减少的缩进
        state.dedent -= 1;
      }

      # 返回样式
      return style;
    }
    # 定义名为 external 的变量，包含 startState、token、indent、electricInput、closeBrackets、lineComment 和 fold 属性
    var external = {
      # 定义 startState 方法，接受 basecolumn 参数，返回包含 tokenBase、scopes、indent、lastToken、lambda 和 dedent 属性的对象
      startState: function(basecolumn) {
        return {
          tokenize: tokenBase,  # 设置 tokenize 属性为 tokenBase 函数
          scopes: [{offset: basecolumn || 0, type: "py", align: null}],  # 设置 scopes 属性为包含 offset、type 和 align 属性的对象
          indent: basecolumn || 0,  # 设置 indent 属性为 basecolumn 或 0
          lastToken: null,  # 设置 lastToken 属性为 null
          lambda: false,  # 设置 lambda 属性为 false
          dedent: 0  # 设置 dedent 属性为 0
        };
      },

      # 定义 token 方法，接受 stream 和 state 参数，处理 token 的样式和错误
      token: function(stream, state) {
        var addErr = state.errorToken;  # 设置 addErr 变量为 state.errorToken
        if (addErr) state.errorToken = false;  # 如果 addErr 为真，将 state.errorToken 设置为 false
        var style = tokenLexer(stream, state);  # 设置 style 变量为 tokenLexer 函数处理 stream 和 state 后的结果

        if (style && style != "comment")  # 如果 style 存在且不是 "comment"
          state.lastToken = (style == "keyword" || style == "punctuation") ? stream.current() : style;  # 设置 state.lastToken 为 stream.current() 或 style
        if (style == "punctuation") style = null;  # 如果 style 是 "punctuation"，将 style 设置为 null

        if (stream.eol() && state.lambda)  # 如果 stream 到达行尾且 state.lambda 为真
          state.lambda = false;  # 将 state.lambda 设置为 false
        return addErr ? style + " " + ERRORCLASS : style;  # 如果 addErr 为真，返回 style + " " + ERRORCLASS，否则返回 style
      },

      # 定义 indent 方法，接受 state 和 textAfter 参数，处理缩进
      indent: function(state, textAfter) {
        if (state.tokenize != tokenBase)  # 如果 state.tokenize 不等于 tokenBase
          return state.tokenize.isString ? CodeMirror.Pass : 0;  # 如果 state.tokenize.isString 为真，返回 CodeMirror.Pass，否则返回 0

        var scope = top(state), closing = scope.type == textAfter.charAt(0)  # 设置 scope 变量为 state 的顶部，closing 变量为 scope.type 是否等于 textAfter 的第一个字符
        if (scope.align != null)  # 如果 scope.align 不为 null
          return scope.align - (closing ? 1 : 0)  # 返回 scope.align 减去（如果 closing 为真则 1，否则 0）
        else
          return scope.offset - (closing ? hangingIndent : 0)  # 返回 scope.offset 减去（如果 closing 为真则 hangingIndent，否则 0）
      },

      electricInput: /^\s*[\}\]\)]$/,  # 设置 electricInput 属性为匹配空白字符后跟 }、] 或 ) 的正则表达式
      closeBrackets: {triples: "'\""},  # 设置 closeBrackets 属性为包含 triples 属性的对象
      lineComment: "#",  # 设置 lineComment 属性为 "#"
      fold: "indent"  # 设置 fold 属性为 "indent"
    };
    return external;  # 返回 external 变量
  });

  # 定义 MIME 类型为 "text/x-python" 的 CodeMirror
  CodeMirror.defineMIME("text/x-python", "python");

  # 定义 words 函数，接受字符串参数 str，返回以空格分割的字符串数组
  var words = function(str) { return str.split(" "); };

  # 定义 MIME 类型为 "text/x-cython" 的 CodeMirror，包含 name 和 extra_keywords 属性
  CodeMirror.defineMIME("text/x-cython", {
    name: "python",  # 设置 name 属性为 "python"
    extra_keywords: words("by cdef cimport cpdef ctypedef enum except "+
                          "extern gil include nogil property public "+
                          "readonly struct union DEF IF ELIF ELSE")  # 设置 extra_keywords 属性为 words 函数处理后的结果
  });
});


/* ---- mode/rust.js ---- */


// CodeMirror, 版权归 Marijn Haverbeke 和其他人所有
// 根据 MIT 许可证分发：https://codemirror.net/LICENSE

(function(mod) {
  if (typeof exports == "object" && typeof module == "object") // CommonJS
    mod(require("../../lib/codemirror"), require("../../addon/mode/simple"));
  else if (typeof define == "function" && define.amd) // AMD
    define(["../../lib/codemirror", "../../addon/mode/simple"], mod);
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
"use strict";

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

    {regex: /(?:(?:[0-9][0-9_]*)(?:(?:[Ee][+-]?[0-9_]+)|\.[0-9_]+(?:[Ee][+-]?[0-9_]+)?)(?:f32|f64)?)|(?:0(?:b[01_]+|(?:o[0-7_]+)|(?:x[0-9a-fA-F_]+))|(?:[0-9][0-9_]*))(?:u8|u16|u32|u64|i8|i16|i32|i64|isize|usize)?/,
     token: "number"},
    {regex: /(let(?:\s+mut)?|fn|enum|mod|struct|type|union)(\s+)([a-zA-Z_][a-zA-Z0-9_]*)/, token: ["keyword", null, "def"]},
    {regex: /(?:abstract|alignof|as|async|await|box|break|continue|const|crate|do|dyn|else|enum|extern|fn|for|final|if|impl|in|loop|macro|match|mod|move|offsetof|override|priv|proc|pub|pure|ref|return|self|sizeof|static|struct|super|trait|type|typeof|union|unsafe|unsized|use|virtual|where|while|yield)\b/, token: "keyword"},
    {regex: /\b(?:Self|isize|usize|char|bool|u8|u16|u32|u64|f16|f32|f64|i8|i16|i32|i64|str|Option)\b/, token: "atom"},
    {regex: /\b(?:true|false|Some|None|Ok|Err)\b/, token: "builtin"},
    # 定义正则表达式匹配规则，匹配函数定义关键字
    {regex: /\b(fn)(\s+)([a-zA-Z_][a-zA-Z0-9_]*)/,
     token: ["keyword", null ,"def"]},
    # 匹配元数据标记
    {regex: /#!?\[.*\]/, token: "meta"},
    # 匹配单行注释
    {regex: /\/\/.*/, token: "comment"},
    # 匹配多行注释起始标记
    {regex: /\/\*/, token: "comment", next: "comment"},
    # 匹配运算符
    {regex: /[-+\/*=<>!]+/, token: "operator"},
    # 匹配变量名
    {regex: /[a-zA-Z_]\w*!/,token: "variable-3"},
    # 匹配变量名
    {regex: /[a-zA-Z_]\w*/, token: "variable"},
    # 匹配大括号、中括号、小括号，设置缩进
    {regex: /[\{\[\(]/, indent: true},
    # 匹配大括号、中括号、小括号，取消缩进
    {regex: /[\}\]\)]/, dedent: true}
  ],
  # 定义字符串匹配规则
  string: [
    # 匹配双引号，设置下一个状态为起始状态
    {regex: /"/, token: "string", next: "start"},
    # 匹配除反斜杠和双引号外的所有字符，设置为字符串
    {regex: /(?:[^\\"]|\\(?:.|$))*/, token: "string"}
  ],
  # 定义原始字符串匹配规则
  string_raw: [
    # 匹配双引号，设置下一个状态为起始状态
    {regex: /"/, token: "string", next: "start"},
    # 匹配除双引号外的所有字符，设置为字符串
    {regex: /[^"]*/, token: "string"}
  ],
  # 定义带有哈希标记的原始字符串匹配规则
  string_raw_hash: [
    # 匹配带有多个井号的双引号，设置下一个状态为起始状态
    {regex: /"#+/, token: "string", next: "start"},
    # 匹配除双引号外的所有字符，设置为字符串
    {regex: /(?:[^"]|"(?!#))*/, token: "string"}
  ],
  # 定义多行注释匹配规则
  comment: [
    # 匹配任意字符直到多行注释结束标记，设置为注释
    {regex: /.*?\*\//, token: "comment", next: "start"},
    # 匹配任意字符，设置为注释
    {regex: /.*/, token: "comment"}
  ],
  # 定义元数据配置
  meta: {
    # 不缩进的状态
    dontIndentStates: ["comment"],
    # 电动输入的正则表达式
    electricInput: /^\s*\}$/,
    # 块注释起始标记
    blockCommentStart: "/*",
    # 块注释结束标记
    blockCommentEnd: "*/",
    # 单行注释标记
    lineComment: "//",
    # 折叠标记
    fold: "brace"
  }
// 定义 MIME 类型为 text/x-rustsrc 的语言为 rust
CodeMirror.defineMIME("text/x-rustsrc", "rust");
// 定义 MIME 类型为 text/rust 的语言为 rust
CodeMirror.defineMIME("text/rust", "rust");

// mode/xml.js 文件的内容
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境
  else
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // HTML 配置对象
  var htmlConfig = {
    // 自动闭合的标签
    autoSelfClosers: {'area': true, 'base': true, 'br': true, 'col': true, 'command': true,
                      'embed': true, 'frame': true, 'hr': true, 'img': true, 'input': true,
                      'keygen': true, 'link': true, 'meta': true, 'param': true, 'source': true,
                      'track': true, 'wbr': true, 'menuitem': true},
    // 隐式闭合的标签
    implicitlyClosed: {'dd': true, 'li': true, 'optgroup': true, 'option': true, 'p': true,
                       'rp': true, 'rt': true, 'tbody': true, 'td': true, 'tfoot': true,
                       'th': true, 'tr': true},
    // 上下文抓取器
    contextGrabbers: {
      'dd': {'dd': true, 'dt': true},
      'dt': {'dd': true, 'dt': true},
      'li': {'li': true},
      'option': {'option': true, 'optgroup': true},
      'optgroup': {'optgroup': true},
      'p': {'address': true, 'article': true, 'aside': true, 'blockquote': true, 'dir': true,
            'div': true, 'dl': true, 'fieldset': true, 'footer': true, 'form': true,
            'h1': true, 'h2': true, 'h3': true, 'h4': true, 'h5': true, 'h6': true,
            'header': true, 'hgroup': true, 'hr': true, 'menu': true, 'nav': true, 'ol': true,
            'p': true, 'pre': true, 'section': true, 'table': true, 'ul': true},
      'rp': {'rp': true, 'rt': true},
      'rt': {'rp': true, 'rt': true},
      'tbody': {'tbody': true, 'tfoot': true},
      'td': {'td': true, 'th': true},
      'tfoot': {'tbody': true},
    # 定义一个对象，包含了一系列的键值对，每个键值对表示一个标签名和其允许包含的子标签
    'th': {'td': true, 'th': true},
    'thead': {'tbody': true, 'tfoot': true},
    'tr': {'tr': true}
  },
  # 定义一个对象，包含了一系列的键值对，每个键值对表示一个标签名和其不需要缩进的子标签
  doNotIndent: {"pre": true},
  # 允许属性值不使用引号
  allowUnquoted: true,
  # 允许标签不闭合
  allowMissing: true,
  # 不区分标签名的大小写
  caseFold: true
// 定义变量 xmlConfig，包含 XML 模式的配置信息
var xmlConfig = {
  autoSelfClosers: {},  // 自动关闭的标签
  implicitlyClosed: {},  // 隐式关闭的标签
  contextGrabbers: {},  // 上下文抓取器
  doNotIndent: {},  // 不缩进的标签
  allowUnquoted: false,  // 是否允许不带引号的属性值
  allowMissing: false,  // 是否允许缺失的标签
  allowMissingTagName: false,  // 是否允许缺失的标签名
  caseFold: false  // 是否忽略大小写
}

// 定义 XML 模式，传入编辑器配置和自定义配置
CodeMirror.defineMode("xml", function(editorConf, config_) {
  var indentUnit = editorConf.indentUnit  // 缩进单位
  var config = {}  // 配置对象
  var defaults = config_.htmlMode ? htmlConfig : xmlConfig  // 默认配置
  for (var prop in defaults) config[prop] = defaults[prop]  // 将默认配置复制到配置对象
  for (var prop in config_) config[prop] = config_[prop]  // 将自定义配置复制到配置对象

  // 为标记器返回变量
  var type, setStyle;

  // 处于文本状态的标记器
  function inText(stream, state) {
    function chain(parser) {
      state.tokenize = parser;  // 设置标记器
      return parser(stream, state);  // 调用标记器
    }

    var ch = stream.next();  // 读取下一个字符
    if (ch == "<") {  // 如果是左尖括号
      if (stream.eat("!")) {  // 如果是注释或 CDATA
        if (stream.eat("[")) {
          if (stream.match("CDATA[")) return chain(inBlock("atom", "]]>"));  // 如果是 CDATA
          else return null;
        } else if (stream.match("--")) {
          return chain(inBlock("comment", "-->"));  // 如果是注释
        } else if (stream.match("DOCTYPE", true, true)) {
          stream.eatWhile(/[\w\._\-]/);  // 读取 DOCTYPE 名称
          return chain(doctype(1));  // 处理 DOCTYPE
        } else {
          return null;
        }
      } else if (stream.eat("?")) {
        stream.eatWhile(/[\w\._\-]/);  // 读取处理指令名称
        state.tokenize = inBlock("meta", "?>");  // 设置为处理指令状态
        return "meta";
      } else {
        type = stream.eat("/") ? "closeTag" : "openTag";  // 判断是闭合标签还是开放标签
        state.tokenize = inTag;  // 设置为标签状态
        return "tag bracket";
      }
    } else if (ch == "&") {  // 如果是实体引用
      var ok;
      if (stream.eat("#")) {  // 如果是数字实体引用
        if (stream.eat("x")) {
          ok = stream.eatWhile(/[a-fA-F\d]/) && stream.eat(";");  // 读取十六进制数字
        } else {
          ok = stream.eatWhile(/[\d]/) && stream.eat(";");  // 读取十进制数字
        }
      } else {
        ok = stream.eatWhile(/[\w\.\-:]/) && stream.eat(";");  // 读取实体引用名称
      }
      return ok ? "atom" : "error";  // 返回实体引用类型
    } else {
      stream.eatWhile(/[^&<]/);  // 读取非实体引用和非左尖括号的字符
      return null;
    }
  }
  inText.isInText = true;  // 标记处于文本状态的标记器

  // 标签状态的标记器
  function inTag(stream, state) {
    // 从流中获取下一个字符
    var ch = stream.next();
    // 如果字符是">"或者是"/"并且下一个字符是">"，则设置tokenize状态为inText，设置type为"endTag"或者"selfcloseTag"，返回"tag bracket"
    if (ch == ">" || (ch == "/" && stream.eat(">"))) {
      state.tokenize = inText;
      type = ch == ">" ? "endTag" : "selfcloseTag";
      return "tag bracket";
    } 
    // 如果字符是"="，则设置type为"equals"，返回null
    else if (ch == "=") {
      type = "equals";
      return null;
    } 
    // 如果字符是"<"，则设置tokenize状态为inText，state状态为baseState，tagName和tagStart为null，调用state.tokenize方法，返回结果
    else if (ch == "<") {
      state.tokenize = inText;
      state.state = baseState;
      state.tagName = state.tagStart = null;
      var next = state.tokenize(stream, state);
      return next ? next + " tag error" : "tag error";
    } 
    // 如果字符是单引号或双引号，则设置tokenize状态为inAttribute，stringStartCol为当前列数，调用state.tokenize方法，返回结果
    else if (/[\'\"]/.test(ch)) {
      state.tokenize = inAttribute(ch);
      state.stringStartCol = stream.column();
      return state.tokenize(stream, state);
    } 
    // 如果字符不是空格、nbsp、=、<、>、单引号、双引号、/，则匹配非空白字符，返回"word"
    else {
      stream.match(/^[^\s\u00a0=<>\"\']*[^\s\u00a0=<>\"\'\/]/);
      return "word";
    }
  }

  // 返回一个函数，用于处理属性值
  function inAttribute(quote) {
    var closure = function(stream, state) {
      while (!stream.eol()) {
        if (stream.next() == quote) {
          state.tokenize = inTag;
          break;
        }
      }
      return "string";
    };
    closure.isInAttribute = true;
    return closure;
  }

  // 返回一个函数，用于处理块级元素
  function inBlock(style, terminator) {
    return function(stream, state) {
      while (!stream.eol()) {
        if (stream.match(terminator)) {
          state.tokenize = inText;
          break;
        }
        stream.next();
      }
      return style;
    }
  }

  // 返回一个函数，用于处理文档类型声明
  function doctype(depth) {
    return function(stream, state) {
      var ch;
      while ((ch = stream.next()) != null) {
        if (ch == "<") {
          state.tokenize = doctype(depth + 1);
          return state.tokenize(stream, state);
        } else if (ch == ">") {
          if (depth == 1) {
            state.tokenize = inText;
            break;
          } else {
            state.tokenize = doctype(depth - 1);
            return state.tokenize(stream, state);
          }
        }
      }
      return "meta";
    };
  }

  // 定义Context构造函数
  function Context(state, tagName, startOfLine) {
    this.prev = state.context;
    this.tagName = tagName;
    # 设置缩进等级为当前状态的缩进等级
    this.indent = state.indented;
    # 设置行首标记为给定的行首标记
    this.startOfLine = startOfLine;
    # 如果配置中包含不缩进的标签名，或者当前上下文存在且不需要缩进，则设置不缩进标记为真
    if (config.doNotIndent.hasOwnProperty(tagName) || (state.context && state.context.noIndent))
      this.noIndent = true;
  }
  # 弹出当前上下文
  function popContext(state) {
    if (state.context) state.context = state.context.prev;
  }
  # 如果可能，弹出当前上下文
  function maybePopContext(state, nextTagName) {
    var parentTagName;
    while (true) {
      if (!state.context) {
        return;
      }
      parentTagName = state.context.tagName;
      # 如果当前上下文不包含给定的标签名，或者配置中不包含当前上下文的标签名和给定的标签名的组合，则返回
      if (!config.contextGrabbers.hasOwnProperty(parentTagName) ||
          !config.contextGrabbers[parentTagName].hasOwnProperty(nextTagName)) {
        return;
      }
      # 弹出当前上下文
      popContext(state);
    }
  }

  # 基础状态函数，根据类型返回不同的状态函数
  function baseState(type, stream, state) {
    if (type == "openTag") {
      # 设置标签起始位置为当前流的列数
      state.tagStart = stream.column();
      return tagNameState;
    } else if (type == "closeTag") {
      return closeTagNameState;
    } else {
      return baseState;
    }
  }
  # 标签名状态函数，根据类型返回不同的状态函数
  function tagNameState(type, stream, state) {
    if (type == "word") {
      # 设置当前标签名为流的当前内容，设置样式为"tag"，并返回属性状态函数
      state.tagName = stream.current();
      setStyle = "tag";
      return attrState;
    } else if (config.allowMissingTagName && type == "endTag") {
      # 如果允许缺少标签名，并且类型为"endTag"，则设置样式为"tag bracket"，并返回属性状态函数
      setStyle = "tag bracket";
      return attrState(type, stream, state);
    } else {
      # 设置样式为"error"，并继续保持标签名状态
      setStyle = "error";
      return tagNameState;
    }
  }
  # 关闭标签名状态函数，根据类型返回不同的状态函数
  function closeTagNameState(type, stream, state) {
    if (type == "word") {
      # 获取当前标签名
      var tagName = stream.current();
      # 如果当前上下文存在且当前上下文的标签名不等于当前标签名，并且配置中包含当前上下文的隐式关闭标签名，则弹出当前上下文
      if (state.context && state.context.tagName != tagName &&
          config.implicitlyClosed.hasOwnProperty(state.context.tagName))
        popContext(state);
      # 如果当前上下文存在且当前上下文的标签名等于当前标签名，或者匹配关闭标签被禁用，则设置样式为"tag"，并返回关闭状态函数
      if ((state.context && state.context.tagName == tagName) || config.matchClosing === false) {
        setStyle = "tag";
        return closeState;
      } else {
        # 设置样式为"tag error"，并返回关闭状态错误函数
        setStyle = "tag error";
        return closeStateErr;
      }
    } else if (config.allowMissingTagName && type == "endTag") {
      # 如果允许缺少标签名，并且类型为"endTag"，则设置样式为"tag bracket"，并返回关闭状态函数
      setStyle = "tag bracket";
      return closeState(type, stream, state);
  } else {
    // 如果不是开始标签，则设置样式为错误，并返回关闭状态错误
    setStyle = "error";
    return closeStateErr;
  }
}

function closeState(type, _stream, state) {
  // 如果类型不是结束标签，则设置样式为错误，并返回关闭状态
  if (type != "endTag") {
    setStyle = "error";
    return closeState;
  }
  // 弹出当前上下文
  popContext(state);
  return baseState;
}
function closeStateErr(type, stream, state) {
  // 设置样式为错误，并返回关闭状态
  setStyle = "error";
  return closeState(type, stream, state);
}

function attrState(type, _stream, state) {
  // 如果类型是单词，则设置样式为属性，并返回属性等号状态
  if (type == "word") {
    setStyle = "attribute";
    return attrEqState;
  } else if (type == "endTag" || type == "selfcloseTag") {
    var tagName = state.tagName, tagStart = state.tagStart;
    state.tagName = state.tagStart = null;
    // 如果是自闭合标签或者自动自闭合标签，则可能弹出当前上下文
    if (type == "selfcloseTag" ||
        config.autoSelfClosers.hasOwnProperty(tagName)) {
      maybePopContext(state, tagName);
    } else {
      maybePopContext(state, tagName);
      state.context = new Context(state, tagName, tagStart == state.indented);
    }
    return baseState;
  }
  // 设置样式为错误，并返回属性状态
  setStyle = "error";
  return attrState;
}
function attrEqState(type, stream, state) {
  // 如果类型是等号，则返回属性值状态，否则如果不允许缺失，则设置样式为错误，并返回属性状态
  if (type == "equals") return attrValueState;
  if (!config.allowMissing) setStyle = "error";
  return attrState(type, stream, state);
}
function attrValueState(type, stream, state) {
  // 如果类型是字符串，则返回属性继续状态，否则如果类型是单词且允许不带引号，则设置样式为字符串，并返回属性状态，否则设置样式为错误，并返回属性状态
  if (type == "string") return attrContinuedState;
  if (type == "word" && config.allowUnquoted) {setStyle = "string"; return attrState;}
  setStyle = "error";
  return attrState(type, stream, state);
}
function attrContinuedState(type, stream, state) {
  // 如果类型是字符串，则返回属性继续状态，否则返回属性状态
  if (type == "string") return attrContinuedState;
  return attrState(type, stream, state);
}

return {
  startState: function(baseIndent) {
    // 返回初始状态对象
    var state = {tokenize: inText,
                 state: baseState,
                 indented: baseIndent || 0,
                 tagName: null, tagStart: null,
                 context: null}
    if (baseIndent != null) state.baseIndent = baseIndent
    return state
  },
    # 定义 token 函数，用于对代码进行词法分析
    token: function(stream, state) {
      # 如果当前行没有标签名，并且是行首，则记录缩进值
      if (!state.tagName && stream.sol())
        state.indented = stream.indentation();

      # 如果当前位置是空白字符，则返回空
      if (stream.eatSpace()) return null;
      type = null;
      # 调用 state 对象的 tokenize 方法，对当前位置的代码进行词法分析
      var style = state.tokenize(stream, state);
      # 如果有样式或类型，并且不是注释类型的样式
      if ((style || type) && style != "comment") {
        setStyle = null;
        # 根据样式或类型设置状态
        state.state = state.state(type || style, stream, state);
        # 如果设置了样式，则更新样式
        if (setStyle)
          style = setStyle == "error" ? style + " error" : setStyle;
      }
      # 返回样式
      return style;
    },
    # 定义一个函数，用于确定代码的缩进
    indent: function(state, textAfter, fullLine) {
      var context = state.context;
      # 如果正在处理属性中的多行字符串（例如css），则进行缩进
      if (state.tokenize.isInAttribute) {
        if (state.tagStart == state.indented)
          return state.stringStartCol + 1;
        else
          return state.indented + indentUnit;
      }
      # 如果上下文存在且不需要缩进，则返回 CodeMirror.Pass
      if (context && context.noIndent) return CodeMirror.Pass;
      # 如果不是在标签或文本中进行标记，则返回完整行的缩进或0
      if (state.tokenize != inTag && state.tokenize != inText)
        return fullLine ? fullLine.match(/^(\s*)/)[0].length : 0;
      # 缩进属性名的起始位置
      if (state.tagName) {
        if (config.multilineTagIndentPastTag !== false)
          return state.tagStart + state.tagName.length + 2;
        else
          return state.tagStart + indentUnit * (config.multilineTagIndentFactor || 1);
      }
      # 如果配置了 alignCDATA 并且 textAfter 包含 "<![CDATA["，则返回0
      if (config.alignCDATA && /<!\[CDATA\[/.test(textAfter)) return 0;
      # 检查是否有闭合标签
      var tagAfter = textAfter && /^<(\/)?([\w_:\.-]*)/.exec(textAfter);
      if (tagAfter && tagAfter[1]) { // Closing tag spotted
        while (context) {
          if (context.tagName == tagAfter[2]) {
            context = context.prev;
            break;
          } else if (config.implicitlyClosed.hasOwnProperty(context.tagName)) {
            context = context.prev;
          } else {
            break;
          }
        }
      } else if (tagAfter) { // Opening tag spotted
        while (context) {
          var grabbers = config.contextGrabbers[context.tagName];
          if (grabbers && grabbers.hasOwnProperty(tagAfter[2]))
            context = context.prev;
          else
            break;
        }
      }
      # 寻找上一个起始行的上下文，并返回相应的缩进
      while (context && context.prev && !context.startOfLine)
        context = context.prev;
      if (context) return context.indent + indentUnit;
      else return state.baseIndent || 0;
    },

    # 定义一个正则表达式，用于确定是否输入了闭合标签
    electricInput: /<\/[\s\w:]+>$/,
    # 定义块注释的起始标记
    blockCommentStart: "<!--",
    # 定义块注释的结束标记
    blockCommentEnd: "-->",

    # 根据配置决定使用 "html" 还是 "xml"
    configuration: config.htmlMode ? "html" : "xml",
    # 根据配置文件中的htmlMode属性确定helperType的取值，如果为true则为"html"，否则为"xml"
    helperType: config.htmlMode ? "html" : "xml",

    # 跳过属性的解析，如果当前状态为attrValueState，则将状态改为attrState
    skipAttribute: function(state) {
      if (state.state == attrValueState)
        state.state = attrState
    },

    # 返回当前的 XML 标签信息，包括标签名和类型（开标签或闭标签）
    xmlCurrentTag: function(state) {
      return state.tagName ? {name: state.tagName, close: state.type == "closeTag"} : null
    },

    # 返回当前的 XML 上下文信息，即当前标签的祖先标签列表
    xmlCurrentContext: function(state) {
      var context = []
      for (var cx = state.context; cx; cx = cx.prev)
        if (cx.tagName) context.push(cx.tagName)
      return context.reverse()
    }
  };
# 定义 MIME 类型为 "text/xml" 的 CodeMirror 模式为 "xml"
CodeMirror.defineMIME("text/xml", "xml");
# 定义 MIME 类型为 "application/xml" 的 CodeMirror 模式为 "xml"
CodeMirror.defineMIME("application/xml", "xml");
# 如果 CodeMirror.mimeModes 对象中不包含 "text/html" 属性，则定义 MIME 类型为 "text/html" 的 CodeMirror 模式为 {name: "xml", htmlMode: true}
if (!CodeMirror.mimeModes.hasOwnProperty("text/html"))
  CodeMirror.defineMIME("text/html", {name: "xml", htmlMode: true});
```