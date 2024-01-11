# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\hint\show-hint.js`

```
// 使用立即执行函数表达式（IIFE）来定义模块
(function(mod) {
  // 如果是 CommonJS 环境，则使用 require 导入 CodeMirror 模块
  if (typeof exports == "object" && typeof module == "object") 
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境，则使用 define 导入 CodeMirror 模块
  else if (typeof define == "function" && define.amd) 
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境，则直接使用 CodeMirror 模块
  else 
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义常量 HINT_ELEMENT_CLASS 和 ACTIVE_HINT_ELEMENT_CLASS
  var HINT_ELEMENT_CLASS        = "CodeMirror-hint";
  var ACTIVE_HINT_ELEMENT_CLASS = "CodeMirror-hint-active";

  // 旧的接口，为了向后兼容而保留
  CodeMirror.showHint = function(cm, getHints, options) {
    // 如果没有 getHints 函数，则调用 cm.showHint(options)
    if (!getHints) return cm.showHint(options);
    // 如果 options 存在且为异步，则设置 getHints.async 为 true
    if (options && options.async) getHints.async = true;
    // 创建新的选项对象 newOpts，设置 hint 属性为 getHints
    var newOpts = {hint: getHints};
    // 如果 options 存在，则将其属性复制到 newOpts 中
    if (options) for (var prop in options) newOpts[prop] = options[prop];
    // 调用 cm.showHint(newOpts) 并返回结果
    return cm.showHint(newOpts);
  };

  // 定义 showHint 方法的扩展
  CodeMirror.defineExtension("showHint", function(options) {
    // 解析选项，获取光标位置
    options = parseOptions(this, this.getCursor("start"), options);
    // 获取选择的文本
    var selections = this.listSelections()
    // 如果选择的文本大于 1 个，则返回
    if (selections.length > 1) return;
    // 默认情况下，不允许在选择文本时进行自动补全
    // 但是如果 hint 函数有 supportsSelection 属性，则允许处理选择的文本
    if (this.somethingSelected()) {
      if (!options.hint.supportsSelection) return;
      // 不允许跨行选择
      for (var i = 0; i < selections.length; i++)
        if (selections[i].head.line != selections[i].anchor.line) return;
    }
    // 如果存在 completionActive，则关闭它
    if (this.state.completionActive) this.state.completionActive.close();
    // 创建新的自动补全对象 completion
    var completion = this.state.completionActive = new Completion(this, options);
    // 如果没有 hint 属性，则返回
    if (!completion.options.hint) return;
    // 触发 startCompletion 事件，并更新自动补全
    CodeMirror.signal(this, "startCompletion", this);
    completion.update(true);
  });

  // 定义 closeHint 方法的扩展
  CodeMirror.defineExtension("closeHint", function() {
  // 如果自动补全处于激活状态，则关闭自动补全
  if (this.state.completionActive) this.state.completionActive.close()
  })

  // 定义自动补全的构造函数
  function Completion(cm, options) {
    this.cm = cm; // 保存 CodeMirror 对象
    this.options = options; // 保存选项
    this.widget = null; // 初始化小部件为空
    this.debounce = 0; // 初始化 debounce 为 0
    this.tick = 0; // 初始化 tick 为 0
    this.startPos = this.cm.getCursor("start"); // 获取光标起始位置
    this.startLen = this.cm.getLine(this.startPos.line).length - this.cm.getSelection().length; // 获取起始行的长度减去选择的长度

    var self = this; // 保存 this 到 self
    cm.on("cursorActivity", this.activityFunc = function() { self.cursorActivity(); }); // 监听光标活动事件
  }

  // 定义 requestAnimationFrame 函数
  var requestAnimationFrame = window.requestAnimationFrame || function(fn) {
    return setTimeout(fn, 1000/60);
  };
  // 定义 cancelAnimationFrame 函数
  var cancelAnimationFrame = window.cancelAnimationFrame || clearTimeout;

  // 定义自动补全的原型方法
  Completion.prototype = {
    // 关闭自动补全
    close: function() {
      if (!this.active()) return; // 如果自动补全不处于激活状态，则返回
      this.cm.state.completionActive = null; // 将自动补全状态设置为 null
      this.tick = null; // 将 tick 设置为 null
      this.cm.off("cursorActivity", this.activityFunc); // 取消光标活动事件监听

      if (this.widget && this.data) CodeMirror.signal(this.data, "close"); // 如果小部件和数据存在，则发送关闭信号
      if (this.widget) this.widget.close(); // 如果小部件存在，则关闭小部件
      CodeMirror.signal(this.cm, "endCompletion", this.cm); // 发送结束自动补全信号
    },

    // 判断自动补全是否处于激活状态
    active: function() {
      return this.cm.state.completionActive == this;
    },

    // 选择自动补全的选项
    pick: function(data, i) {
      var completion = data.list[i], self = this; // 获取选项并保存到 completion，保存 this 到 self
      this.cm.operation(function() {
        if (completion.hint)
          completion.hint(self.cm, data, completion); // 如果存在提示，则调用提示函数
        else
          self.cm.replaceRange(getText(completion), completion.from || data.from,
                               completion.to || data.to, "complete"); // 否则替换文本
        CodeMirror.signal(data, "pick", completion); // 发送选择信号
        self.cm.scrollIntoView(); // 滚动到视图
      })
      this.close(); // 关闭自动补全
    },
    // 定义一个名为 cursorActivity 的方法
    cursorActivity: function() {
      // 如果存在 debounce 属性，则取消动画帧并将 debounce 属性置为 0
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
      // 检查光标位置是否发生变化，如果发生变化则关闭提示框
      if (pos.line != this.startPos.line || line.length - pos.ch != this.startLen - this.startPos.ch ||
          pos.ch < identStart.ch || this.cm.somethingSelected() ||
          (!pos.ch || this.options.closeCharacters.test(line.charAt(pos.ch - 1)))) {
        this.close();
      } else {
        // 如果光标位置未发生变化，则更新提示框内容
        var self = this;
        this.debounce = requestAnimationFrame(function() {self.update();});
        // 如果存在 widget，则禁用 widget
        if (this.widget) this.widget.disable();
      }
    },

    // 定义一个名为 update 的方法，接受一个名为 first 的参数
    update: function(first) {
      // 如果 tick 为 null，则直接返回
      if (this.tick == null) return
      var self = this, myTick = ++this.tick
      // 调用 fetchHints 方法获取提示内容
      fetchHints(this.options.hint, this.cm, this.options, function(data) {
        // 如果 tick 等于 myTick，则调用 finishUpdate 方法
        if (self.tick == myTick) self.finishUpdate(data, first)
      })
    },

    // 定义一个名为 finishUpdate 的方法，接受名为 data 和 first 的参数
    finishUpdate: function(data, first) {
      // 如果存在 this.data，则触发 "update" 事件
      if (this.data) CodeMirror.signal(this.data, "update");

      // 检查是否已选择提示项或者 options.completeSingle 为 true，如果是则关闭 widget
      var picked = (this.widget && this.widget.picked) || (first && this.options.completeSingle);
      if (this.widget) this.widget.close();

      // 将 data 赋值给 this.data
      this.data = data;

      // 如果 data 存在且包含提示项，则根据条件选择提示项或者创建新的 widget
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

  // 定义一个名为 parseOptions 的函数，接受名为 cm、pos 和 options 的参数
  function parseOptions(cm, pos, options) {
    // 获取 cm.options.hintOptions 并赋值给 editor
    var editor = cm.options.hintOptions;
    // 初始化 out 对象
    var out = {};
    // 遍历 defaultOptions 对象的属性，并赋值给 out 对象
    for (var prop in defaultOptions) out[prop] = defaultOptions[prop];
    // 如果 editor 存在，则将 editor 的属性赋值给 out 对象
    if (editor) for (var prop in editor)
      if (editor[prop] !== undefined) out[prop] = editor[prop];
    // 如果 options 存在，则将 options 的属性赋值给 out 对象
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

    // 检测是否为 Mac 平台
    var mac = /Mac/.test(navigator.platform);

    // 如果是 Mac 平台，添加额外的键盘映射
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
    // 如果有自定义键盘映射，则遍历添加
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
    // 获取输入字段的所属文档对象
    var ownerDocument = cm.getInputField().ownerDocument;
    // 获取所属文档对象的默认视图或父窗口
    var parentWindow = ownerDocument.defaultView || ownerDocument.parentWindow;

    // 创建一个无序列表元素作为提示框
    var hints = this.hints = ownerDocument.createElement("ul");
    // 获取代码编辑器的主题
    var theme = completion.cm.options.theme;
    // 设置提示框的类名，包括CodeMirror-hints和主题类名
    hints.className = "CodeMirror-hints " + theme;
    // 设置当前选中的提示索引
    this.selectedHint = data.selectedHint || 0;

    // 获取提示列表
    var completions = data.list;
    // 遍历提示列表
    for (var i = 0; i < completions.length; ++i) {
      // 创建列表项元素
      var elt = hints.appendChild(ownerDocument.createElement("li")), cur = completions[i];
      // 设置列表项的类名
      var className = HINT_ELEMENT_CLASS + (i != this.selectedHint ? "" : " " + ACTIVE_HINT_ELEMENT_CLASS);
      // 如果提示对象有自定义类名，则添加到列表项的类名中
      if (cur.className != null) className = cur.className + " " + className;
      elt.className = className;
      // 如果提示对象有自定义渲染函数，则调用渲染函数
      if (cur.render) cur.render(elt, data, cur);
      // 否则将提示对象的显示文本添加到列表项中
      else elt.appendChild(ownerDocument.createTextNode(cur.displayText || getText(cur)));
      // 设置列表项的提示ID
      elt.hintId = i;
    }

    // 获取提示框的容器，如果没有指定则使用文档的body
    var container = completion.options.container || ownerDocument.body;
    // 获取光标位置的坐标
    var pos = cm.cursorCoords(completion.options.alignWithWord ? data.from : null);
    var left = pos.left, top = pos.bottom, below = true;
    var offsetLeft = 0, offsetTop = 0;
    // 如果容器不是文档的body，则需要计算偏移量
    if (container !== ownerDocument.body) {
      // 计算偏移量
      var isContainerPositioned = ['absolute', 'relative', 'fixed'].indexOf(parentWindow.getComputedStyle(container).position) !== -1;
      var offsetParent = isContainerPositioned ? container : container.offsetParent;
      var offsetParentPosition = offsetParent.getBoundingClientRect();
      var bodyPosition = ownerDocument.body.getBoundingClientRect();
      offsetLeft = (offsetParentPosition.left - bodyPosition.left - offsetParent.scrollLeft);
      offsetTop = (offsetParentPosition.top - bodyPosition.top - offsetParent.scrollTop);
    }
    // 设置提示框的位置
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
    // 获取编辑器的滚动信息
    var startScroll = cm.getScrollInfo();
    
    // 如果提示框与窗口底部有重叠
    if (overlapY > 0) {
      var height = box.bottom - box.top, curTop = pos.top - (pos.bottom - box.top);
      // 如果提示框可以放在光标上方
      if (curTop - height > 0) { // Fits above cursor
        hints.style.top = (top = pos.top - height - offsetTop) + "px";
        below = false;
      } else if (height > winH) {
        // 如果提示框高度超过窗口高度
        hints.style.height = (winH - 5) + "px";
        hints.style.top = (top = pos.bottom - box.top - offsetTop) + "px";
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
      if (box.right - box.left > winW) {
        // 如果提示框宽度超过窗口宽度
        hints.style.width = (winW - 5) + "px";
        overlapX -= (box.right - box.left) - winW;
      }
      hints.style.left = (left = pos.left - overlapX - offsetLeft) + "px";
    }
    // 如果提示框需要滚动，则设置内边距
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
      # 选择完成选项
      pick: function() { widget.pick(); },
      # 数据
      data: data
    }));

    # 如果完成选项设置为失去焦点时关闭
    if (completion.options.closeOnUnfocus) {
      # 定义失去焦点时的关闭操作
      var closingOnBlur;
      cm.on("blur", this.onBlur = function() { closingOnBlur = setTimeout(function() { completion.close(); }, 100); });
      # 定义获得焦点时的操作
      cm.on("focus", this.onFocus = function() { clearTimeout(closingOnBlur); });
    }

    # 滚动时的操作
    cm.on("scroll", this.onScroll = function() {
      # 获取当前滚动信息和编辑器信息
      var curScroll = cm.getScrollInfo(), editor = cm.getWrapperElement().getBoundingClientRect();
      # 计算新的顶部位置
      var newTop = top + startScroll.top - curScroll.top;
      var point = newTop - (parentWindow.pageYOffset || (ownerDocument.documentElement || ownerDocument.body).scrollTop);
      # 如果超出编辑器范围，则关闭完成选项
      if (!below) point += hints.offsetHeight;
      if (point <= editor.top || point >= editor.bottom) return completion.close();
      hints.style.top = newTop + "px";
      hints.style.left = (left + startScroll.left - curScroll.left) + "px";
    });

    # 双击事件
    CodeMirror.on(hints, "dblclick", function(e) {
      # 获取双击的提示元素，并改变活动状态并选择
      var t = getHintElement(hints, e.target || e.srcElement);
      if (t && t.hintId != null) {widget.changeActive(t.hintId); widget.pick();}
    });

    # 单击事件
    CodeMirror.on(hints, "click", function(e) {
      # 获取单击的提示元素，并改变活动状态并选择
      var t = getHintElement(hints, e.target || e.srcElement);
      if (t && t.hintId != null) {
        widget.changeActive(t.hintId);
        # 如果设置为单击即完成，则选择
        if (completion.options.completeOnSingleClick) widget.pick();
      }
    });

    # 鼠标按下事件
    CodeMirror.on(hints, "mousedown", function() {
      # 延迟20毫秒后聚焦到编辑器
      setTimeout(function(){cm.focus();}, 20);
    });
    # 滚动到活动状态
    this.scrollToActive()
    # 发出选择信号，将选定的提示和选定的提示节点传递给信号处理程序
    CodeMirror.signal(data, "select", completions[this.selectedHint], hints.childNodes[this.selectedHint]);
    # 返回 true
    return true;
  }

  Widget.prototype = {
    # 关闭小部件
    close: function() {
      # 如果当前小部件不是正在关闭的小部件，则返回
      if (this.completion.widget != this) return;
      # 将小部件从其父节点中移除
      this.hints.parentNode.removeChild(this.hints);
      # 从编辑器中移除键盘映射
      this.completion.cm.removeKeyMap(this.keyMap);

      var cm = this.completion.cm;
      # 如果选项为在失去焦点时关闭，则移除焦点和失去焦点事件处理程序
      if (this.completion.options.closeOnUnfocus) {
        cm.off("blur", this.onBlur);
        cm.off("focus", this.onFocus);
      }
      # 移除滚动事件处理程序
      cm.off("scroll", this.onScroll);
    },

    # 禁用小部件
    disable: function() {
      # 从编辑器中移除键盘映射
      this.completion.cm.removeKeyMap(this.keyMap);
      var widget = this;
      # 设置键盘映射，当按下 Enter 键时，设置 picked 为 true
      this.keyMap = {Enter: function() { widget.picked = true; }};
      this.completion.cm.addKeyMap(this.keyMap);
    },

    # 选择当前提示
    pick: function() {
      this.completion.pick(this.data, this.selectedHint);
    },

    # 更改活动提示
    changeActive: function(i, avoidWrap) {
      # 如果 i 大于提示列表的长度，则将 i 设置为提示列表长度减一，或者根据 avoidWrap 的值决定是否循环
      if (i >= this.data.list.length)
        i = avoidWrap ? this.data.list.length - 1 : 0;
      # 如果 i 小于 0，则将 i 设置为 0，或者根据 avoidWrap 的值决定是否循环
      else if (i < 0)
        i = avoidWrap ? 0  : this.data.list.length - 1;
      # 如果当前选定的提示与 i 相同，则返回
      if (this.selectedHint == i) return;
      # 移除当前选定提示节点的活动样式类
      var node = this.hints.childNodes[this.selectedHint];
      if (node) node.className = node.className.replace(" " + ACTIVE_HINT_ELEMENT_CLASS, "");
      # 将选定的提示设置为 i，并为新选定的提示节点添加活动样式类
      node = this.hints.childNodes[this.selectedHint = i];
      node.className += " " + ACTIVE_HINT_ELEMENT_CLASS;
      # 滚动到活动提示
      this.scrollToActive()
      # 发出选择信号，将选定的提示和选定的提示节点传递给信号处理程序
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
      # 如果node2的底部位置大于提示列表的滚动位置加上可见高度，则将滚动位置设置为node2的底部位置加上高度减去提示列表的可见高度再加上第一个节点的顶部位置
      else if (node2.offsetTop + node2.offsetHeight > this.hints.scrollTop + this.hints.clientHeight)
        this.hints.scrollTop = node2.offsetTop + node2.offsetHeight - this.hints.clientHeight + firstNode.offsetTop;
    },

    # 计算屏幕可容纳的提示数量
    screenAmount: function() {
      # 返回提示列表可容纳的整数个提示数量，如果无法整除则返回1
      return Math.floor(this.hints.clientHeight / this.hints.firstChild.offsetHeight) || 1;
    }
  };

  # 获取适用的辅助函数
  function applicableHelpers(cm, helpers) {
    # 如果没有选中内容，则返回所有辅助函数
    if (!cm.somethingSelected()) return helpers
    # 否则返回支持选择的辅助函数
    var result = []
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
      # 否则调用同步提示函数，并处理返回结果
      var result = hint(cm, options)
      if (result && result.then) result.then(callback)
      else callback(result)
    }
  }

  # 解析自动提示
  function resolveAutoHints(cm, pos) {
    # 获取指定位置的辅助函数和单词
    var helpers = cm.getHelpers(pos, "hint"), words
    if (helpers.length) {
      # 创建解析函数，用于处理多个辅助函数的提示
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
# 闭合了一个代码块或者函数的结束
```