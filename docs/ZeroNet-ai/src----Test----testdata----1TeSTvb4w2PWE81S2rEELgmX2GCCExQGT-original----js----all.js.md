# `ZeroNet\src\Test\testdata\1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT-original\js\all.js`

```py
/* ---- data/1BLogC9LN4oPDcruNz3qo1ysa133E9AGg8/js/lib/00-jquery.min.js ---- */
/*! jQuery v2.1.3 | (c) 2005, 2014 jQuery Foundation, Inc. | jquery.org/license */

/* ---- data/1BLogC9LN4oPDcruNz3qo1ysa133E9AGg8/js/lib/highlight.pack.js ---- */

/* ---- data/1BLogC9LN4oPDcruNz3qo1ysa133E9AGg8/js/lib/identicon.js ---- */
/**
 * Identicon.js v1.0
 * http://github.com/stewartlord/identicon.js
 *
 * Requires PNGLib
 * http://www.xarg.org/download/pnglib.js
 *
 * Copyright 2013, Stewart Lord
 * Released under the BSD license
 * http://www.opensource.org/licenses/bsd-license.php
 */
(function() {
    // 定义 Identicon 类
    Identicon = function(hash, size, margin){
        this.hash   = hash;
        this.size   = size   || 64;
        this.margin = margin || .08;
    }
    // 将 Identicon 类绑定到 window 对象上
    window.Identicon = Identicon;
})();

/* ---- data/1BLogC9LN4oPDcruNz3qo1ysa133E9AGg8/js/lib/jquery.cssanim.coffee ---- */
(function() {
  // 为 jQuery 对象添加 cssSlideDown 方法
  jQuery.fn.cssSlideDown = function() {
    var elem;
    elem = this;
    // 设置元素的样式
    elem.css({
      "opacity": 0,
      "margin-bottom": 0,
      "margin-top": 0,
      "padding-bottom": 0,
      "padding-top": 0,
      "display": "none",
      "transform": "scale(0.8)"
    });
    // 设置定时器
    setTimeout((function() {
      var height;
      // 显示元素
      elem.css("display", "");
      // 获取元素的高度
      height = elem.outerHeight();
      // 设置元素的样式
      elem.css({
        "height": 0,
        "display": ""
      }).cssLater("transition", "all 0.3s ease-out", 20);
      elem.cssLater({
        "height": height,
        "opacity": 1,
        "margin-bottom": "",
        "margin-top": "",
        "padding-bottom": "",
        "padding-top": "",
        "transform": "scale(1)"
      }, null, 40);
      // 清除过渡效果
      return elem.cssLater("transition", "", 1000, "noclear");
    }), 10);
    return this;
  };
  // 为 jQuery 对象添加 fancySlideDown 方法
  jQuery.fn.fancySlideDown = function() {
    var elem;
    elem = this;
    // 设置元素的样式
    return elem.css({
      "opacity": 0,
      "transform": "scale(0.9)"
    }).slideDown().animate({
      "opacity": 1,
      "scale": 1
  # 定义 jQuery 插件 fancySlideDown，用于元素下滑动画
  jQuery.fn.fancySlideDown = function() {
    # 声明变量 elem，指向当前 jQuery 对象
    var elem;
    elem = this;
    # 延迟 600 毫秒后执行下滑动画，持续 600 毫秒
    return elem.delay(600).slideDown(600).animate({
      # 设置动画效果为淡出和缩放至 0.9 倍
      "opacity": 1,
      "scale": 1.1
    }, {
      # 设置动画持续时间为 600 毫秒，不排队，使用 easeOutBack 缓动函数
      "duration": 600,
      "queue": false,
      "easing": "easeOutBack"
    });
  };

  # 定义 jQuery 插件 fancySlideUp，用于元素上滑动画
  jQuery.fn.fancySlideUp = function() {
    # 声明变量 elem，指向当前 jQuery 对象
    var elem;
    elem = this;
    # 延迟 600 毫秒后执行上滑动画，持续 600 毫秒
    return elem.delay(600).slideUp(600).animate({
      # 设置动画效果为淡出和缩放至 0.9 倍
      "opacity": 0,
      "scale": 0.9
    }, {
      # 设置动画持续时间为 600 毫秒，不排队，使用 easeOutQuad 缓动函数
      "duration": 600,
      "queue": false,
      "easing": "easeOutQuad"
    });
  };
  # 定义一个匿名函数，将其作为方法调用
  }).call(this);

  # 定义一个匿名函数，为 jQuery 对象添加一些延迟操作的方法
  # 定义一个计时器对象
  timers = {};

  # 为 jQuery 对象添加重新添加类名的方法
  jQuery.fn.readdClass = function(class_name) {
    # 获取当前元素
    var elem;
    elem = this;
    # 移除指定的类名
    elem.removeClass(class_name);
    # 延迟一段时间后重新添加指定的类名
    setTimeout((function() {
      return elem.addClass(class_name);
    }), 1);
    return this;
  };

  # 为 jQuery 对象添加延迟移除元素的方法
  jQuery.fn.removeLater = function(time) {
    # 如果未指定时间，默认为 500 毫秒
    var elem;
    if (time == null) {
      time = 500;
    }
    elem = this;
    # 延迟一段时间后移除元素
    setTimeout((function() {
      return elem.remove();
    }), time);
    return this;
  };

  # 为 jQuery 对象添加延迟隐藏元素的方法
  jQuery.fn.hideLater = function(time) {
    # 如果未指定时间，默认为 500 毫秒
    if (time == null) {
      time = 500;
    }
    # 调用 cssLater 方法，将 display 属性设置为 none，并延迟一段时间
    this.cssLater("display", "none", time);
    return this;
  };

  # 为 jQuery 对象添加延迟添加类名的方法
  jQuery.fn.addClassLater = function(class_name, time, mode) {
    # 如果未指定时间，默认为 5 毫秒
    if (time == null) {
      time = 5;
    }
    # 如果未指定模式，默认为 "clear"
    if (mode == null) {
      mode = "clear";
    }
    # 获取当前元素
    elem = this;
    # 如果已存在指定类名的计时器，并且模式为 "clear"，则清除计时器
    if (timers[class_name] && mode === "clear") {
      clearInterval(timers[class_name]);
    }
    # 设置一个延迟添加类名的计时器
    timers[class_name] = setTimeout((function() {
      return elem.addClass(class_name);
    }), time);
    return this;
  };

  # 为 jQuery 对象添加延迟移除类名的方法
  jQuery.fn.removeClassLater = function(class_name, time, mode) {
    # 如果未指定时间，默认为 500 毫秒
    if (time == null) {
      time = 500;
    }
    # 如果未指定模式，默认为 "clear"
    if (mode == null) {
      mode = "clear";
    }
    # 获取当前元素
    elem = this;
    # 如果已存在指定类名的计时器，并且模式为 "clear"，则清除计时器
    if (timers[class_name] && mode === "clear") {
      clearInterval(timers[class_name]);
    }
    # 设置一个延迟移除类名的计时器
    timers[class_name] = setTimeout((function() {
      return elem.removeClass(class_name);
    }), time);
    return this;
  };

  # 为 jQuery 对象添加延迟设置样式的方法
  jQuery.fn.cssLater = function(name, val, time, mode) {
    # 如果未指定时间，默认为 500 毫秒
    if (time == null) {
      time = 500;
    }
    # 如果未指定模式，默认为 "clear"
    if (mode == null) {
      mode = "clear";
    }
    # 获取当前元素
    elem = this;
    # 如果已存在指定样式的计时器，并且模式为 "clear"，则清除计时器
    if (timers[name] && mode === "clear") {
      clearInterval(timers[name]);
    }
    # 如果时间为 "now"，立即设置样式
    if (time === "now") {
      elem.css(name, val);
  } else {
    // 如果条件不成立，将定时器赋值给timers[name]
    timers[name] = setTimeout((function() {
      // 返回一个函数，设置元素的样式
      return elem.css(name, val);
    }), time);
  }
  // 返回当前jQuery对象
  return this;
};

// 在延迟后切换元素的类
jQuery.fn.toggleClassLater = function(name, val, time, mode) {
  var elem;
  // 如果time未定义，则默认为10
  if (time == null) {
    time = 10;
  }
  // 如果mode未定义，则默认为"clear"
  if (mode == null) {
    mode = "clear";
  }
  // 将当前jQuery对象赋值给elem
  elem = this;
  // 如果timers[name]存在且mode为"clear"
  if (timers[name] && mode === "clear") {
    // 清除定时器
    clearInterval(timers[name]);
  }
  // 设置一个定时器，在延迟后切换元素的类
  timers[name] = setTimeout((function() {
    return elem.toggleClass(name, val);
  }), time);
  // 返回当前jQuery对象
  return this;
};
// 调用匿名函数，将 this 绑定到全局对象
}).call(this);

/* ---- data/1BLogC9LN4oPDcruNz3qo1ysa133E9AGg8/js/lib/marked.min.js ---- */

/**
 * marked - a markdown parser
 * Copyright (c) 2011-2014, Christopher Jeffrey. (MIT Licensed)
 * https://github.com/chjj/marked
 */
/* ---- data/1BLogC9LN4oPDcruNz3qo1ysa133E9AGg8/js/lib/pnglib.js ---- */

/**
* A handy class to calculate color values.
*
* @version 1.0
* @author Robert Eisele <robert@xarg.org>
* @copyright Copyright (c) 2010, Robert Eisele
* @link http://www.xarg.org/2010/03/generate-client-side-png-files-using-javascript/
* @license http://www.opensource.org/licenses/bsd-license.php BSD License
*
*/

(function() {

    // helper functions for that ctx
    // 写入函数，将参数中的字符串写入缓冲区
    function write(buffer, offs) {
        for (var i = 2; i < arguments.length; i++) {
            for (var j = 0; j < arguments[i].length; j++) {
                buffer[offs++] = arguments[i].charAt(j);
            }
        }
    }

    // 将 16 位整数转换为 2 字节的字符串
    function byte2(w) {
        return String.fromCharCode((w >> 8) & 255, w & 255);
    }

    // 将 32 位整数转换为 4 字节的字符串
    function byte4(w) {
        return String.fromCharCode((w >> 24) & 255, (w >> 16) & 255, (w >> 8) & 255, w & 255);
    }

    // 将 16 位整数转换为 2 字节的字符串（低位在前）
    function byte2lsb(w) {
        return String.fromCharCode(w & 255, (w >> 8) & 255);
    }

    // 省略部分代码

})();

/* ---- data/1BLogC9LN4oPDcruNz3qo1ysa133E9AGg8/js/utils/Class.coffee ---- */

(function() {
  var Class,
    __slice = [].slice;

  Class = (function() {
    function Class() {}

    Class.prototype.trace = true;

    // 打印日志函数
    Class.prototype.log = function() {
      var args;
      args = 1 <= arguments.length ? __slice.call(arguments, 0) : [];
      if (!this.trace) {
        return;
      }
      if (typeof console === 'undefined') {
        return;
      }
      args.unshift("[" + this.constructor.name + "]");
      console.log.apply(console, args);
      return this;
    };
    # 定义 Class 原型的 logStart 方法
    Class.prototype.logStart = function() {
      # 获取参数中的 name 和 args
      var args, name;
      name = arguments[0], args = 2 <= arguments.length ? __slice.call(arguments, 1) : [];
      # 如果没有开启 trace，则直接返回
      if (!this.trace) {
        return;
      }
      # 如果 logtimers 不存在，则创建一个空对象
      this.logtimers || (this.logtimers = {});
      # 将当前时间存入 logtimers 对象中的 name 属性
      this.logtimers[name] = +(new Date);
      # 如果参数长度大于 0，则调用 log 方法，传入 name 和 args，并在最后加上 "(started)"
      if (args.length > 0) {
        this.log.apply(this, ["" + name].concat(__slice.call(args), ["(started)"]));
      }
      # 返回 this
      return this;
    };

    # 定义 Class 原型的 logEnd 方法
    Class.prototype.logEnd = function() {
      # 获取参数中的 name 和 args
      var args, ms, name;
      name = arguments[0], args = 2 <= arguments.length ? __slice.call(arguments, 1) : [];
      # 计算执行时间
      ms = +(new Date) - this.logtimers[name];
      # 调用 log 方法，传入 name、args 和执行时间，并在最后加上 "(Done in " + ms + "ms)"
      this.log.apply(this, ["" + name].concat(__slice.call(args), ["(Done in " + ms + "ms)"]));
      # 返回 this
      return this;
    };

    # 返回 Class
    return Class;

  })();
  
  # 将 Class 赋值给 window 对象的属性 Class
  window.Class = Class;
// 定义一个立即执行函数，将当前上下文绑定到 this
(function() {
  // 定义 InlineEditor 类
  var InlineEditor,
    // 将函数绑定到指定的上下文
    __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };

  // 初始化 InlineEditor 类
  InlineEditor = (function() {
    // 构造函数，接收四个参数
    function InlineEditor(_at_elem, _at_getContent, _at_saveContent, _at_getObject) {
      // 将参数赋值给对应的属性
      this.elem = _at_elem;
      this.getContent = _at_getContent;
      this.saveContent = _at_saveContent;
      this.getObject = _at_getObject;
      // 将方法绑定到当前实例
      this.cancelEdit = __bind(this.cancelEdit, this);
      this.deleteObject = __bind(this.deleteObject, this);
      this.saveEdit = __bind(this.saveEdit, this);
      this.stopEdit = __bind(this.stopEdit, this);
      this.startEdit = __bind(this.startEdit, this);
      // 创建编辑按钮元素
      this.edit_button = $("<a href='#Edit' class='editable-edit icon-edit'></a>");
      // 绑定点击事件处理函数
      this.edit_button.on("click", this.startEdit);
      // 在编辑元素之前添加编辑按钮
      this.elem.addClass("editable").before(this.edit_button);
      // 初始化编辑器为 null
      this.editor = null;
      // 鼠标进入编辑元素时的事件处理函数
      this.elem.on("mouseenter", (function(_this) {
        return function(e) {
          var scrolltop, top;
          // 设置编辑按钮的透明度
          _this.edit_button.css("opacity", "0.4");
          // 获取窗口滚动条的位置和编辑按钮的位置
          scrolltop = $(window).scrollTop();
          top = _this.edit_button.offset().top - parseInt(_this.edit_button.css("margin-top"));
          // 根据滚动条位置调整编辑按钮的位置
          if (scrolltop > top) {
            return _this.edit_button.css("margin-top", scrolltop - top + e.clientY - 20);
          } else {
            return _this.edit_button.css("margin-top", "");
          }
        };
      })(this));
      // 鼠标离开编辑元素时的事件处理函数
      this.elem.on("mouseleave", (function(_this) {
        return function() {
          return _this.edit_button.css("opacity", "");
        };
      })(this));
      // 如果编辑元素处于鼠标悬停状态，则触发鼠标进入事件
      if (this.elem.is(":hover")) {
        this.elem.trigger("mouseenter");
      }
    }
    // 定义内联编辑器的开始编辑方法
    InlineEditor.prototype.startEdit = function() {
      // 保存编辑前的内容
      this.content_before = this.elem.html();
      // 创建一个<textarea>元素作为编辑器
      this.editor = $("<textarea class='editor'></textarea>");
      // 设置编辑器的样式
      this.editor.css("outline", "10000px solid rgba(255,255,255,0)").cssLater("transition", "outline 0.3s", 5).cssLater("outline", "10000px solid rgba(255,255,255,0.9)", 10);
      // 将编辑器的值设置为元素的原始内容
      this.editor.val(this.getContent(this.elem, "raw"));
      // 在元素后面插入编辑器
      this.elem.after(this.editor);
      // 将元素的内容设置为数字 1 到 50，用 "fill the width" 连接
      this.elem.html((function() {
        _results = [];
        for (_i = 1; _i <= 50; _i++){ _results.push(_i); }
        return _results;
      }).apply(this).join("fill the width"));
      // 复制元素的样式到编辑器
      this.copyStyle(this.elem, this.editor);
      // 将元素的内容设置为编辑前的内容
      this.elem.html(this.content_before);
      // 自动扩展编辑器的高度
      this.autoExpand(this.editor);
      // 隐藏元素
      this.elem.css("display", "none");
      // 如果滚动条在顶部，将焦点设置在编辑器上
      if ($(window).scrollTop() === 0) {
        this.editor[0].selectionEnd = 0;
        this.editor.focus();
      }
      // 隐藏编辑按钮
      $(".editable-edit").css("display", "none");
      // 显示编辑工具栏
      $(".editbar").css("display", "inline-block").addClassLater("visible", 10);
      // 设置发布工具栏的透明度为 0
      $(".publishbar").css("opacity", 0);
      // 设置编辑工具栏的对象和按钮
      $(".editbar .object").text(this.getObject(this.elem).data("object") + "." + this.elem.data("editable"));
      $(".editbar .button").removeClass("loading");
      $(".editbar .save").off("click").on("click", this.saveEdit);
      $(".editbar .delete").off("click").on("click", this.deleteObject);
      $(".editbar .cancel").off("click").on("click", this.cancelEdit);
      // 如果元素可删除，显示删除按钮，否则隐藏
      if (this.getObject(this.elem).data("deletable")) {
        $(".editbar .delete").css("display", "").html("Delete " + this.getObject(this.elem).data("object").split(":")[0]);
      } else {
        $(".editbar .delete").css("display", "none");
      }
      // 在页面关闭前提示未保存的更改
      window.onbeforeunload = function() {
        return 'Your unsaved blog changes will be lost!';
      };
      // 返回 false
      return false;
    };
    # 停止编辑操作，移除编辑器元素，重置编辑器对象，显示原始元素，隐藏编辑按钮，隐藏编辑工具栏，显示发布工具栏，取消页面离开时的警告
    InlineEditor.prototype.stopEdit = function() {
      this.editor.remove();
      this.editor = null;
      this.elem.css("display", "");
      $(".editable-edit").css("display", "");
      $(".editbar").cssLater("display", "none", 1000).removeClass("visible");
      $(".publishbar").css("opacity", 1);
      return window.onbeforeunload = null;
    };

    # 保存编辑操作，获取编辑器内容，添加保存中的加载动画，保存内容并停止编辑，如果内容为字符串则更新元素内容，对代码块进行高亮处理
    InlineEditor.prototype.saveEdit = function() {
      var content;
      content = this.editor.val();
      $(".editbar .save").addClass("loading");
      this.saveContent(this.elem, content, (function(_this) {
        return function(content_html) {
          if (content_html) {
            $(".editbar .save").removeClass("loading");
            _this.stopEdit();
            if (typeof content_html === "string") {
              _this.elem.html(content_html);
            }
            return $('pre code').each(function(i, block) {
              return hljs.highlightBlock(block);
            });
          } else {
            return $(".editbar .save").removeClass("loading");
          }
        };
      })(this));
      return false;
    };

    # 删除对象操作，获取对象类型，弹出确认对话框，添加删除中的加载动画，保存内容并停止编辑
    InlineEditor.prototype.deleteObject = function() {
      var object_type;
      object_type = this.getObject(this.elem).data("object").split(":")[0];
      Page.cmd("wrapperConfirm", ["Are you sure you sure to delete this " + object_type + "?", "Delete"], (function(_this) {
        return function(confirmed) {
          $(".editbar .delete").addClass("loading");
          return Page.saveContent(_this.getObject(_this.elem), null, function() {
            return _this.stopEdit();
          });
        };
      })(this));
      return false;
    };

    # 取消编辑操作，停止编辑并恢复原始内容，对代码块进行高亮处理
    InlineEditor.prototype.cancelEdit = function() {
      this.stopEdit();
      this.elem.html(this.content_before);
      $('pre code').each(function(i, block) {
        return hljs.highlightBlock(block);
      });
      return false;
    };
    # 将元素 elem_from 的样式复制到元素 elem_to
    InlineEditor.prototype.copyStyle = function(elem_from, elem_to) {
      # 给 elem_to 添加与 elem_from 相同的类名
      elem_to.addClass(elem_from[0].className);
      # 获取 elem_from 的计算样式
      from_style = getComputedStyle(elem_from[0]);
      # 将获取的样式应用到 elem_to
      elem_to.css({
        fontFamily: from_style.fontFamily,
        fontSize: from_style.fontSize,
        fontWeight: from_style.fontWeight,
        marginTop: from_style.marginTop,
        marginRight: from_style.marginRight,
        marginBottom: from_style.marginBottom,
        marginLeft: from_style.marginLeft,
        paddingTop: from_style.paddingTop,
        paddingRight: from_style.paddingRight,
        paddingBottom: from_style.paddingBottom,
        paddingLeft: from_style.paddingLeft,
        lineHeight: from_style.lineHeight,
        textAlign: from_style.textAlign,
        color: from_style.color,
        letterSpacing: from_style.letterSpacing
      });
      # 如果 elem_from 的宽度小于 1000，则设置 elem_to 的最小宽度为 elem_from 的宽度
      if (elem_from.innerWidth() < 1000) {
        return elem_to.css("minWidth", elem_from.innerWidth());
      }
    };
    
    # 自动扩展元素的高度以适应内容
    InlineEditor.prototype.autoExpand = function(elem) {
      var editor;
      editor = elem[0];
      # 将元素的高度设置为 1
      elem.height(1);
      # 监听输入事件
      elem.on("input", function() {
        # 如果编辑器的滚动高度大于元素的高度，则将元素的高度设置为滚动高度加上上下边框的宽度
        if (editor.scrollHeight > elem.height()) {
          return elem.height(1).height(editor.scrollHeight + parseFloat(elem.css("borderTopWidth")) + parseFloat(elem.css("borderBottomWidth")));
        }
      });
      # 触发输入事件
      elem.trigger("input");
      # 监听键盘按下事件
      return elem.on('keydown', function(e) {
        var s, val;
        # 如果按下的是 Tab 键
        if (e.which === 9) {
          e.preventDefault();
          s = this.selectionStart;
          val = elem.val();
          # 在当前光标位置插入一个制表符
          elem.val(val.substring(0, this.selectionStart) + "\t" + val.substring(this.selectionEnd));
          return this.selectionEnd = s + 1;
        }
      });
    };
    
    # 将 InlineEditor 导出为全局变量
    return InlineEditor;
    
    })();
    # 将 InlineEditor 设置为 window 对象的属性
    window.InlineEditor = InlineEditor;
# 定义一个匿名函数，将其作用域设置为全局对象
(function() {
  # 定义一个空对象用于存储限制
  var limits = {};
  # 定义一个空对象用于存储调用间隔
  var call_after_interval = {};
  # 创建全局对象 RateLimit，接受两个参数：间隔时间和函数
  window.RateLimit = function(interval, fn) {
    # 如果限制对象中不存在该函数
    if (!limits[fn]) {
      # 将调用间隔对象中对应函数的值设置为 false
      call_after_interval[fn] = false;
      # 调用该函数
      fn();
      # 将该函数的限制存储在限制对象中，并设置一个定时器
      return limits[fn] = setTimeout((function() {
        # 如果调用间隔对象中对应函数的值为 true，则再次调用该函数
        if (call_after_interval[fn]) {
          fn();
        }
        # 删除限制对象中对应的函数
        delete limits[fn];
        # 删除调用间隔对象中对应的函数
        return delete call_after_interval[fn];
      }), interval);
    } else {
      # 如果限制对象中存在该函数，则将调用间隔对象中对应函数的值设置为 true
      return call_after_interval[fn] = true;
    }
  };
}).call(this);

# 定义一个匿名函数，将其作用域设置为全局对象
(function() {
  # 定义一个渲染器类，继承自 marked.Renderer 类
  var Renderer = (function(_super) {
    # 继承 marked.Renderer 类
    __extends(Renderer, _super);
    # 定义构造函数
    function Renderer() {
      return Renderer.__super__.constructor.apply(this, arguments);
    }
    # 定义渲染器的 image 方法
    Renderer.prototype.image = function(href, title, text) {
      return "<code>![" + text + "](" + href + ")</code>";
    };
    return Renderer;
  })(marked.Renderer);
  # 定义文本类
  var Text = (function() {
    # 定义构造函数
    function Text() {
      # 绑定 this 到 toUrl 方法
      this.toUrl = __bind(this.toUrl, this);
    }
    // 将文本转换为颜色值
    Text.prototype.toColor = function(text) {
      var color, hash, i, value, _i, _j, _ref;
      hash = 0;
      // 计算文本的哈希值
      for (i = _i = 0, _ref = text.length - 1; 0 <= _ref ? _i <= _ref : _i >= _ref; i = 0 <= _ref ? ++_i : --_i) {
        hash = text.charCodeAt(i) + ((hash << 5) - hash);
      }
      color = '#';
      // 返回基于哈希值的颜色
      return "hsl(" + (hash % 360) + ",30%,50%)";
      // 将哈希值转换为颜色值
      for (i = _j = 0; _j <= 2; i = ++_j) {
        value = (hash >> (i * 8)) & 0xFF;
        color += ('00' + value.toString(16)).substr(-2);
      }
      return color;
    };
    
    // 将文本转换为带有标记的文本
    Text.prototype.toMarked = function(text, options) {
      if (options == null) {
        options = {};
      }
      options["gfm"] = true;
      options["breaks"] = true;
      if (options.sanitize) {
        options["renderer"] = renderer;
      }
      // 使用 marked 库将文本转换为带有标记的文本
      text = marked(text, options);
      // 修复 HTML 链接
      return this.fixHtmlLinks(text);
    };
    
    // 修复带有 HTML 链接的文本
    Text.prototype.fixHtmlLinks = function(text) {
      if (window.is_proxy) {
        // 替换代理链接
        return text.replace(/href="http:\/\/(127.0.0.1|localhost):43110/g, 'href="http://zero');
      } else {
        // 移除代理链接
        return text.replace(/href="http:\/\/(127.0.0.1|localhost):43110/g, 'href="');
      }
    };
    
    // 修复链接
    Text.prototype.fixLink = function(link) {
      if (window.is_proxy) {
        // 替换代理链接
        return link.replace(/http:\/\/(127.0.0.1|localhost):43110/, 'http://zero');
      } else {
        // 移除代理链接
        return link.replace(/http:\/\/(127.0.0.1|localhost):43110/, '');
      }
    };
    
    // 将文本转换为 URL
    Text.prototype.toUrl = function(text) {
      // 替换非字母数字字符为 "+"
      return text.replace(/[^A-Za-z0-9]/g, "+").replace(/[+]+/g, "+").replace(/[+]+$/, "");
    };
    
    // 导出 Text 类
    return Text;
    
    })();
    
    // 设置代理状态
    window.is_proxy = window.location.pathname === "/";
    
    // 创建渲染器对象
    window.renderer = new Renderer();
    
    // 创建 Text 对象
    window.Text = new Text();
# 定义一个名为 Time 的类
Time = (function() {
  function Time() {}

  # 计算时间间隔
  Time.prototype.since = function(time) {
    # 获取当前时间的时间戳
    now = +(new Date) / 1000;
    # 计算时间间隔
    secs = now - time;
    # 根据时间间隔返回相应的时间描述
    if (secs < 60) {
      back = "Just now";
    } else if (secs < 60 * 60) {
      back = (Math.round(secs / 60)) + " minutes ago";
    } else if (secs < 60 * 60 * 24) {
      back = (Math.round(secs / 60 / 60)) + " hours ago";
    } else if (secs < 60 * 60 * 24 * 3) {
      back = (Math.round(secs / 60 / 60 / 24)) + " days ago";
    } else {
      back = "on " + this.date(time);
    }
    # 替换掉单数形式的时间单位
    back = back.replace(/1 ([a-z]+)s/, "1 $1");
    return back;
  };

  # 格式化时间戳
  Time.prototype.date = function(timestamp, format) {
    if (format == null) {
      format = "short";
    }
    # 将时间戳转换为日期格式
    parts = (new Date(timestamp * 1000)).toString().split(" ");
    # 根据格式返回相应的日期显示
    if (format === "short") {
      display = parts.slice(1, 4);
    } else {
      display = parts.slice(1, 5);
    }
    return display.join(" ").replace(/( [0-9]{4})/, ",$1");
  };

  # 将日期转换为时间戳
  Time.prototype.timestamp = function(date) {
    if (date == null) {
      date = "";
    }
    if (date === "now" || date === "") {
      return parseInt(+(new Date) / 1000);
    } else {
      return parseInt(Date.parse(date) / 1000);
    }
  };

  # 估算阅读时间
  Time.prototype.readtime = function(text) {
    chars = text.length;
    if (chars > 1500) {
      return parseInt(chars / 1500) + " min read";
    } else {
      return "less than 1 min read";
    }
  };

  return Time;

})();
# 将 Time 类绑定到全局对象 window 上
window.Time = new Time;

}).call(this);
    # 定义一个函数，用于实现继承
    __extends = function(child, parent) { for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
    # 定义一个函数，用于检查对象是否包含指定属性
    __hasProp = {}.hasOwnProperty;
    
    # 定义 ZeroFrame 类，继承自 _super
    ZeroFrame = (function(_super) {
        # 调用继承函数，将 ZeroFrame 类继承自 _super
        __extends(ZeroFrame, _super);
    
        # ZeroFrame 类的构造函数，接受一个 url 参数
        function ZeroFrame(url) {
          # 绑定 onCloseWebsocket 方法的上下文为 this
          this.onCloseWebsocket = __bind(this.onCloseWebsocket, this);
          # 绑定 onOpenWebsocket 方法的上下文为 this
          this.onOpenWebsocket = __bind(this.onOpenWebsocket, this);
          # 绑定 route 方法的上下文为 this
          this.route = __bind(this.route, this);
          # 绑定 onMessage 方法的上下文为 this
          this.onMessage = __bind(this.onMessage, this);
          # 初始化 url 属性为传入的 url 参数
          this.url = url;
          # 初始化 waiting_cb 属性为空对象
          this.waiting_cb = {};
          # 调用 connect 方法
          this.connect();
          # 初始化 next_message_id 属性为 1
          this.next_message_id = 1;
          # 调用 init 方法
          this.init();
        }
    
        # ZeroFrame 类的 init 方法
        ZeroFrame.prototype.init = function() {
          # 返回 this
          return this;
        };
    
        # ZeroFrame 类的 connect 方法
        ZeroFrame.prototype.connect = function() {
          # 初始化 target 属性为 window.parent
          this.target = window.parent;
          # 添加消息事件监听器，当消息到达时调用 onMessage 方法
          window.addEventListener("message", this.onMessage, false);
          # 调用 cmd 方法，发送 innerReady 命令
          return this.cmd("innerReady");
        };
    
        # ZeroFrame 类的 onMessage 方法，处理接收到的消息
        ZeroFrame.prototype.onMessage = function(e) {
          # 获取消息内容
          var message = e.data;
          # 获取消息命令
          var cmd = message.cmd;
          # 根据命令类型进行处理
          if (cmd === "response") {
            # 如果 waiting_cb 中存在对应的回调函数，则调用该函数并传入结果
            if (this.waiting_cb[message.to] != null) {
              return this.waiting_cb[message.to](message.result);
            } else {
              # 否则记录日志，提示未找到对应的回调函数
              return this.log("Websocket callback not found:", message);
            }
          } else if (cmd === "wrapperReady") {
            # 如果命令为 wrapperReady，则发送 innerReady 命令
            return this.cmd("innerReady");
          } else if (cmd === "ping") {
            # 如果命令为 ping，则发送 pong 响应
            return this.response(message.id, "pong");
          } else if (cmd === "wrapperOpenedWebsocket") {
            # 如果命令为 wrapperOpenedWebsocket，则调用 onOpenWebsocket 方法
            return this.onOpenWebsocket();
          } else if (cmd === "wrapperClosedWebsocket") {
            # 如果命令为 wrapperClosedWebsocket，则调用 onCloseWebsocket 方法
            return this.onCloseWebsocket();
          } else {
            # 其他命令则调用 onRequest 方法进行处理
            return this.onRequest(cmd, message);
          }
        };
    
        # ZeroFrame 类的 route 方法，处理未知命令
        ZeroFrame.prototype.route = function(cmd, message) {
          # 记录日志，提示未知命令
          return this.log("Unknown command", message);
        };
    # 定义 ZeroFrame 对象的 response 方法，用于发送响应消息
    ZeroFrame.prototype.response = function(to, result) {
      return this.send({
        "cmd": "response",
        "to": to,
        "result": result
      });
    };
    
    # 定义 ZeroFrame 对象的 cmd 方法，用于发送命令消息
    ZeroFrame.prototype.cmd = function(cmd, params, cb) {
      # 如果参数为空，则设置为空对象
      if (params == null) {
        params = {};
      }
      # 如果回调函数为空，则设置为 null
      if (cb == null) {
        cb = null;
      }
      return this.send({
        "cmd": cmd,
        "params": params
      }, cb);
    };
    
    # 定义 ZeroFrame 对象的 send 方法，用于发送消息
    ZeroFrame.prototype.send = function(message, cb) {
      # 如果回调函数为空，则设置为 null
      if (cb == null) {
        cb = null;
      }
      # 为消息设置唯一的 id
      message.id = this.next_message_id;
      this.next_message_id += 1;
      # 发送消息到目标窗口
      this.target.postMessage(message, "*");
      # 如果存在回调函数，则将其存储到等待回调的字典中
      if (cb) {
        return this.waiting_cb[message.id] = cb;
      }
    };
    
    # 定义 ZeroFrame 对象的 onOpenWebsocket 方法，用于处理 WebSocket 打开事件
    ZeroFrame.prototype.onOpenWebsocket = function() {
      return this.log("Websocket open");
    };
    
    # 定义 ZeroFrame 对象的 onCloseWebsocket 方法，用于处理 WebSocket 关闭事件
    ZeroFrame.prototype.onCloseWebsocket = function() {
      return this.log("Websocket close");
    };
    
    # 导出 ZeroFrame 对象
    return ZeroFrame;
    
    })(Class);
    
    # 将 ZeroFrame 对象绑定到全局对象 window 上
    window.ZeroFrame = ZeroFrame;
// 匿名函数，用于包裹整个代码块
(function() {
  // 定义 Comments 类
  var Comments,
    // 继承函数
    __extends = function(child, parent) { for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
    // 判断对象是否有指定属性的函数
    __hasProp = {}.hasOwnProperty;

  // Comments 类继承自 _super
  Comments = (function(_super) {
    // 继承 _super 的属性和方法
    __extends(Comments, _super);

    // Comments 类的构造函数
    function Comments() {
      return Comments.__super__.constructor.apply(this, arguments);
    }

    // Comments 类的 pagePost 方法
    Comments.prototype.pagePost = function(post_id, cb) {
      // 如果回调函数未定义，则设置为 false
      if (cb == null) {
        cb = false;
      }
      // 设置 post_id 属性
      this.post_id = post_id;
      // 初始化 rules 属性为空对象
      this.rules = {};
      // 绑定点击事件，调用 submitComment 方法
      $(".button-submit-comment").on("click", (function(_this) {
        return function() {
          _this.submitComment();
          return false;
        };
      })(this));
      // 调用 loadComments 方法，传入参数 "noanim" 和 cb
      this.loadComments("noanim", cb);
      // 调用 autoExpand 方法，传入参数 $(".comment-textarea")
      this.autoExpand($(".comment-textarea"));
      // 绑定点击事件，调用 certSelect 方法
      return $(".certselect").on("click", (function(_this) {
        return function() {
          // 如果 Page.server_info.rev 小于 160，则显示错误信息
          if (Page.server_info.rev < 160) {
            Page.cmd("wrapperNotification", ["error", "Comments requires at least ZeroNet 0.3.0 Please upgade!"]);
          } else {
            // 否则调用 certSelect 方法，传入参数 [["zeroid.bit"]]
            Page.cmd("certSelect", [["zeroid.bit"]]);
          }
          return false;
        };
      })(this));
    };
    // 定义一个方法用于加载评论，接受类型和回调函数作为参数
    Comments.prototype.loadComments = function(type, cb) {
      // 如果类型为空，则默认为 "show"
      var query;
      if (type == null) {
        type = "show";
      }
      // 如果回调函数为空，则默认为 false
      if (cb == null) {
        cb = false;
      }
      // 构建查询语句，查询评论相关信息
      query = "SELECT comment.*, json_content.json_id AS content_json_id, keyvalue.value AS cert_user_id, json.directory, (SELECT COUNT(*) FROM comment_vote WHERE comment_vote.comment_uri = comment.comment_id || '@' || json.directory)+1 AS votes FROM comment LEFT JOIN json USING (json_id) LEFT JOIN json AS json_content ON (json_content.directory = json.directory AND json_content.file_name='content.json') LEFT JOIN keyvalue ON (keyvalue.json_id = json_content.json_id AND key = 'cert_user_id') WHERE post_id = " + this.post_id + " ORDER BY date_added DESC";
      // 调用 Page.cmd 方法执行数据库查询
      return Page.cmd("dbQuery", query, (function(_this) {
        return function(comments) {
          var comment, comment_address, elem, user_address, _i, _len;
          // 更新评论数量显示
          $(".comments-num").text(comments.length);
          // 遍历评论列表
          for (_i = 0, _len = comments.length; _i < _len; _i++) {
            comment = comments[_i];
            user_address = comment.directory.replace("users/", "");
            comment_address = comment.comment_id + "_" + user_address;
            elem = $("#comment_" + comment_address);
            // 如果评论元素不存在，则创建并添加相应事件
            if (elem.length === 0) {
              elem = $(".comment.template").clone().removeClass("template").attr("id", "comment_" + comment_address).data("post_id", _this.post_id);
              if (type !== "noanim") {
                elem.cssSlideDown();
              }
              $(".reply", elem).on("click", function(e) {
                return _this.buttonReply($(e.target).parents(".comment"));
              });
            }
            // 应用评论数据到元素上
            _this.applyCommentData(elem, comment);
            // 将评论元素添加到页面中
            elem.appendTo(".comments");
          }
          // 延迟一秒后执行添加内联编辑器的方法
          return setTimeout((function() {
            return Page.addInlineEditors();
          }), 1000);
        };
      })(this));
    };
    // 将评论数据应用到指定元素上
    Comments.prototype.applyCommentData = function(elem, comment) {
      var cert_domain, user_address, user_name, _ref;
      // 通过分割字符串获取用户名称和证书域名
      _ref = comment.cert_user_id.split("@"), user_name = _ref[0], cert_domain = _ref[1];
      // 从评论目录中获取用户地址
      user_address = comment.directory.replace("users/", "");
      // 在指定元素中填充评论内容
      $(".comment-body", elem).html(Text.toMarked(comment.body, {
        "sanitize": true
      }));
      // 设置用户名称的显示和颜色
      $(".user_name", elem).text(user_name).css({
        "color": Text.toColor(comment.cert_user_id)
      }).attr("title", user_name + "@" + cert_domain + ": " + user_address);
      // 设置评论添加时间的显示和标题
      $(".added", elem).text(Time.since(comment.date_added)).attr("title", Time.date(comment.date_added, "long"));
      // 如果用户地址与当前站点的认证地址相同，则设置元素的数据属性和评论内容的可编辑属性
      if (user_address === Page.site_info.auth_address) {
        $(elem).attr("data-object", "Comment:" + comment.comment_id).attr("data-deletable", "yes");
        return $(".comment-body", elem).attr("data-editable", "body").data("content", comment.body);
      }
    };

    // 回复按钮的点击事件处理函数
    Comments.prototype.buttonReply = function(elem) {
      var body_add, elem_quote, post_id, user_name;
      this.log("Reply to", elem);
      // 获取评论中的用户名称和评论的ID
      user_name = $(".user_name", elem).text();
      post_id = elem.attr("id");
      // 构建回复内容
      body_add = "> [" + user_name + "](\#" + post_id + "): ";
      elem_quote = $(".comment-body", elem).clone();
      $("blockquote", elem_quote).remove();
      body_add += elem_quote.text().trim("\n").replace(/\n/g, "\n> ");
      body_add += "\n\n";
      // 将回复内容添加到评论输入框中
      $(".comment-new .comment-textarea").val($(".comment-new .comment-textarea").val() + body_add);
      $(".comment-new .comment-textarea").trigger("input").focus();
      return false;
    };
    // 定义提交评论的方法
    Comments.prototype.submitComment = function() {
      var body, inner_path;
      // 如果用户未登录，则提示选择账户并返回false
      if (!Page.site_info.cert_user_id) {
        Page.cmd("wrapperNotification", ["info", "Please, select your account."]);
        return false;
      }
      // 获取评论内容
      body = $(".comment-new .comment-textarea").val();
      // 如果评论内容为空，则聚焦到评论输入框并返回false
      if (!body) {
        $(".comment-new .comment-textarea").focus();
        return false;
      }
      // 添加加载状态样式
      $(".comment-new .button-submit").addClass("loading");
      // 构建用户数据文件路径
      inner_path = "data/users/" + Page.site_info.auth_address + "/data.json";
      // 请求用户数据文件
      return Page.cmd("fileGet", {
        "inner_path": inner_path,
        "required": false
      }, (function(_this) {
        return function(data) {
          var json_raw;
          // 如果存在用户数据文件，则解析为JSON对象
          if (data) {
            data = JSON.parse(data);
          } else {
            // 如果不存在用户数据文件，则创建空的数据对象
            data = {
              "next_comment_id": 1,
              "comment": [],
              "comment_vote": {}
            };
          }
          // 添加新评论到数据对象中
          data.comment.push({
            "comment_id": data.next_comment_id,
            "body": body,
            "post_id": _this.post_id,
            "date_added": Time.timestamp()
          });
          // 更新下一个评论ID
          data.next_comment_id += 1;
          // 将数据对象转换为字符串，并进行编码
          json_raw = unescape(encodeURIComponent(JSON.stringify(data, void 0, '\t')));
          // 写入并发布用户数据文件
          return Page.writePublish(inner_path, btoa(json_raw), function(res) {
            // 移除加载状态样式
            $(".comment-new .button-submit").removeClass("loading");
            // 重新加载评论
            _this.loadComments();
            // 检查用户权限
            _this.checkCert("updaterules");
            // 记录写入发布结果
            _this.log("Writepublish result", res);
            // 如果写入发布成功，则清空评论输入框
            if (res !== false) {
              return $(".comment-new .comment-textarea").val("");
            }
          });
        };
      })(this));
    };
    # 检查证书类型
    Comments.prototype.checkCert = function(type) {
      # 获取上一个证书用户的 ID
      var last_cert_user_id;
      last_cert_user_id = $(".comment-new .user_name").text();
      # 如果页面信息中存在证书用户 ID
      if (Page.site_info.cert_user_id) {
        # 移除评论框的无证书样式，并将用户名设置为证书用户 ID
        $(".comment-new").removeClass("comment-nocert");
        $(".comment-new .user_name").text(Page.site_info.cert_user_id);
      } else {
        # 添加评论框的无证书样式，并将用户名设置为"Please sign in"
        $(".comment-new").addClass("comment-nocert");
        $(".comment-new .user_name").text("Please sign in");
      }
      # 如果用户名发生变化或者类型为"updaterules"
      if ($(".comment-new .user_name").text() !== last_cert_user_id || type === "updaterules") {
        # 如果页面信息中存在证书用户 ID
        if (Page.site_info.cert_user_id) {
          # 调用Page.cmd方法，获取用户规则信息
          return Page.cmd("fileRules", "data/users/" + Page.site_info.auth_address + "/content.json", (function(_this) {
            return function(rules) {
              _this.rules = rules;
              # 如果规则中存在最大大小，则设置当前大小
              if (rules.max_size) {
                return _this.setCurrentSize(rules.current_size);
              } else {
                return _this.setCurrentSize(0);
              }
            };
          })(this));
        } else {
          # 否则设置当前大小为0
          return this.setCurrentSize(0);
        }
      }
    };

    # 设置当前大小
    Comments.prototype.setCurrentSize = function(current_size) {
      var current_size_kb;
      # 如果当前大小存在
      if (current_size) {
        # 将当前大小转换为KB，并更新页面显示
        current_size_kb = current_size / 1000;
        $(".user-size").text("used: " + (current_size_kb.toFixed(1)) + "k/" + (Math.round(this.rules.max_size / 1000)) + "k");
        return $(".user-size-used").css("width", Math.round(70 * current_size / this.rules.max_size));
      } else {
        # 否则清空页面显示
        return $(".user-size").text("");
      }
    };
    # 定义 Comments 对象的 autoExpand 方法，用于自动扩展元素的高度
    Comments.prototype.autoExpand = function(elem) {
      # 将 elem 转换为 editor 变量
      var editor;
      editor = elem[0];
      # 如果元素的高度大于 0，则将其高度设置为 1
      if (elem.height() > 0) {
        elem.height(1);
      }
      # 当输入框内容发生变化时执行以下操作
      elem.on("input", (function(_this) {
        return function() {
          # 获取当前高度、最小高度、新高度、旧高度
          var current_size, min_height, new_height, old_height;
          # 如果编辑器的滚动高度大于元素的高度
          if (editor.scrollHeight > elem.height()) {
            # 保存旧的高度
            old_height = elem.height();
            # 将元素的高度设置为 1
            elem.height(1);
            # 计算新的高度
            new_height = editor.scrollHeight;
            new_height += parseFloat(elem.css("borderTopWidth"));
            new_height += parseFloat(elem.css("borderBottomWidth"));
            new_height -= parseFloat(elem.css("paddingTop"));
            new_height -= parseFloat(elem.css("paddingBottom"));
            # 计算最小高度
            min_height = parseFloat(elem.css("lineHeight")) * 2;
            # 如果新高度小于最小高度，则将新高度设置为最小高度加 4
            if (new_height < min_height) {
              new_height = min_height + 4;
            }
            # 将元素的高度设置为新高度减 4
            elem.height(new_height - 4);
          }
          # 如果存在最大尺寸限制
          if (_this.rules.max_size) {
            # 如果输入框的值长度大于 0
            if (elem.val().length > 0) {
              # 计算当前尺寸
              current_size = _this.rules.current_size + elem.val().length + 90;
            } else {
              # 否则当前尺寸不变
              current_size = _this.rules.current_size;
            }
            # 设置当前尺寸
            return _this.setCurrentSize(current_size);
          }
        };
      })(this));
      # 如果元素的高度大于 0，则触发 input 事件
      if (elem.height() > 0) {
        return elem.trigger("input");
      } else {
        # 否则将元素的高度设置为 48px
        return elem.height("48px");
      }
    };

    # 返回 Comments 对象
    return Comments;

  # 将 Comments 对象赋值给 window.Comments
  })(Class);

  window.Comments = new Comments();
# 匿名函数，用于包裹 ZeroBlog 类的定义
(function() {
  # 创建 ZeroBlog 类
  var ZeroBlog,
    # 绑定函数上下文
    __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; },
    # 继承父类方法
    __extends = function(child, parent) { for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
    # 检查对象是否有指定属性
    __hasProp = {}.hasOwnProperty;

  # ZeroBlog 类继承自 _super 类
  ZeroBlog = (function(_super) {
    # 继承 _super 类的方法
    __extends(ZeroBlog, _super);

    # ZeroBlog 类的构造函数
    function ZeroBlog() {
      # 绑定 this 到 setSiteinfo 方法
      this.setSiteinfo = __bind(this.setSiteinfo, this);
      # 绑定 this 到 actionSetSiteInfo 方法
      this.actionSetSiteInfo = __bind(this.actionSetSiteInfo, this);
      # 绑定 this 到 saveContent 方法
      this.saveContent = __bind(this.saveContent, this);
      # 绑定 this 到 getContent 方法
      this.getContent = __bind(this.getContent, this);
      # 绑定 this 到 getObject 方法
      this.getObject = __bind(this.getObject, this);
      # 绑定 this 到 onOpenWebsocket 方法
      this.onOpenWebsocket = __bind(this.onOpenWebsocket, this);
      # 绑定 this 到 publish 方法
      this.publish = __bind(this.publish, this);
      # 绑定 this 到 pageLoaded 方法
      this.pageLoaded = __bind(this.pageLoaded, this);
      # 返回 ZeroBlog 类的构造函数
      return ZeroBlog.__super__.constructor.apply(this, arguments);
    }
    // 初始化 ZeroBlog 对象的方法
    ZeroBlog.prototype.init = function() {
      // 初始化数据、站点信息和服务器信息
      this.data = null;
      this.site_info = null;
      this.server_info = null;
      // 创建页面加载和站点信息加载的延迟对象
      this.event_page_load = $.Deferred();
      this.event_site_info = $.Deferred();
      // 当页面加载和站点信息加载完成后执行回调函数
      $.when(this.event_page_load, this.event_site_info).done((function(_this) {
        return function() {
          // 如果是自己的站点或者是演示数据，则添加内联编辑器
          if (_this.site_info.settings.own || _this.data.demo) {
            _this.addInlineEditors();
            // 检查发布栏
            _this.checkPublishbar();
            // 点击发布栏时执行发布函数
            $(".publishbar").on("click", _this.publish);
            // 显示新建按钮
            $(".posts .button.new").css("display", "inline-block");
            // 点击帮助图标时显示/隐藏 Markdown 帮助信息
            return $(".editbar .icon-help").on("click", function() {
              $(".editbar .markdown-help").css("display", "block");
              $(".editbar .markdown-help").toggleClassLater("visible", 10);
              $(".editbar .icon-help").toggleClass("active");
              return false;
            });
          }
        };
      })(this));
      // 当站点信息加载完成后执行回调函数
      $.when(this.event_site_info).done((function(_this) {
        return function() {
          var imagedata;
          // 记录事件信息
          _this.log("event site info");
          // 根据站点地址生成 Identicon 图像数据
          imagedata = new Identicon(_this.site_info.address, 70).toString();
          // 将生成的图像数据添加到页面样式中
          return $("body").append("<style>.avatar { background-image: url(data:image/png;base64," + imagedata + ") }</style>");
        };
      })(this));
      // 记录初始化完成信息
      return this.log("inited!");
    };
    // 加载数据的方法，根据查询条件加载数据
    ZeroBlog.prototype.loadData = function(query) {
      // 如果查询条件为空，则默认为"new"
      if (query == null) {
        query = "new";
      }
      // 如果查询条件为"old"，则执行指定的 SQL 查询语句
      if (query === "old") {
        query = "SELECT key, value FROM json LEFT JOIN keyvalue USING (json_id) WHERE path = 'data.json'";
      } else {
        // 否则执行另一条 SQL 查询语句
        query = "SELECT key, value FROM json LEFT JOIN keyvalue USING (json_id) WHERE directory = '' AND file_name = 'data.json'";
      }
      // 调用cmd方法执行数据库查询，并处理返回结果
      return this.cmd("dbQuery", [query], (function(_this) {
        return function(res) {
          var row, _i, _len;
          // 初始化数据对象
          _this.data = {};
          // 如果查询结果不为空，则遍历结果，将数据存入数据对象中
          if (res) {
            for (_i = 0, _len = res.length; _i < _len; _i++) {
              row = res[_i];
              _this.data[row.key] = row.value;
            }
            // 更新页面元素的内容
            $(".left h1 a:not(.editable-edit)").html(_this.data.title).data("content", _this.data.title);
            $(".left h2").html(Text.toMarked(_this.data.description)).data("content", _this.data.description);
            return $(".left .links").html(Text.toMarked(_this.data.links)).data("content", _this.data.links);
          }
        };
      })(this));
    };

    // 根据URL路由到对应的页面
    ZeroBlog.prototype.routeUrl = function(url) {
      var match;
      // 打印日志
      this.log("Routing url:", url);
      // 如果URL匹配到帖子的链接，则添加相应的类，并调用pagePost方法
      if (match = url.match(/Post:([0-9]+)/)) {
        $("body").addClass("page-post");
        this.post_id = parseInt(match[1]);
        return this.pagePost();
      } else {
        // 否则添加默认类，并调用pageMain方法
        $("body").addClass("page-main");
        return this.pageMain();
      }
    };

    // 加载帖子页面的方法
    ZeroBlog.prototype.pagePost = function() {
      var s;
      s = +(new Date);
      // 执行数据库查询，根据帖子ID加载帖子数据，并处理返回结果
      return this.cmd("dbQuery", ["SELECT * FROM post WHERE post_id = " + this.post_id + " LIMIT 1"], (function(_this) {
        return function(res) {
          // 如果查询结果不为空，则应用帖子数据到页面，并调用评论页面的方法
          if (res.length) {
            _this.applyPostdata($(".post-full"), res[0], true);
            Comments.pagePost(_this.post_id);
          } else {
            // 否则在帖子区域显示"Not found"
            $(".post-full").html("<h1>Not found</h1>");
          }
          // 页面加载完成
          return _this.pageLoaded();
        };
      })(this));
    };
    // 定义 ZeroBlog 对象的 pageMain 方法
    ZeroBlog.prototype.pageMain = function() {
      // 调用 cmd 方法执行数据库查询，返回结果后执行回调函数
      return this.cmd("dbQuery", ["SELECT post.*, COUNT(comment_id) AS comments FROM post LEFT JOIN comment USING (post_id) GROUP BY post_id ORDER BY date_published"], (function(_this) {
        return function(res) {
          // 获取当前时间戳
          var elem, post, s, _i, _len;
          s = +(new Date);
          // 遍历查询结果
          for (_i = 0, _len = res.length; _i < _len; _i++) {
            post = res[_i];
            // 根据 post_id 获取对应的元素
            elem = $("#post_" + post.post_id);
            // 如果元素不存在，则克隆模板元素并添加到页面中
            if (elem.length === 0) {
              elem = $(".post.template").clone().removeClass("template").attr("id", "post_" + post.post_id);
              elem.prependTo(".posts");
            }
            // 调用 applyPostdata 方法将数据应用到元素中
            _this.applyPostdata(elem, post);
          }
          // 页面加载完成后执行 pageLoaded 方法
          _this.pageLoaded();
          // 打印日志，记录加载时间
          _this.log("Posts loaded in", (+(new Date)) - s, "ms");
          // 给新添加的文章绑定点击事件
          return $(".posts .new").on("click", function() {
            // 从 data/data.json 文件中获取数据，返回结果后执行回调函数
            _this.cmd("fileGet", ["data/data.json"], function(res) {
              var data;
              // 解析获取的 JSON 数据
              data = JSON.parse(res);
              // 在数据中添加新的博客文章
              data.post.unshift({
                post_id: data.next_post_id,
                title: "New blog post",
                date_published: (+(new Date)) / 1000,
                body: "Blog post body"
              });
              data.next_post_id += 1;
              // 克隆模板元素并添加到页面中
              elem = $(".post.template").clone().removeClass("template");
              // 调用 applyPostdata 方法将数据应用到元素中
              _this.applyPostdata(elem, data.post[0]);
              // 隐藏元素并以动画效果显示
              elem.hide();
              elem.prependTo(".posts").slideDown();
              // 添加内联编辑器
              _this.addInlineEditors(elem);
              // 写入数据
              return _this.writeData(data);
            });
            return false;
          });
        };
      })(this));
    };

    // 定义 ZeroBlog 对象的 pageLoaded 方法
    ZeroBlog.prototype.pageLoaded = function() {
      // 给 body 添加 loaded 类
      $("body").addClass("loaded");
      // 遍历所有的 pre code 元素，对其进行代码高亮
      $('pre code').each(function(i, block) {
        return hljs.highlightBlock(block);
      });
      // 触发页面加载完成事件
      this.event_page_load.resolve();
      // 调用 cmd 方法通知页面加载完成
      return this.cmd("innerLoaded", true);
    };
    # 为 ZeroBlog 对象添加内联编辑器
    ZeroBlog.prototype.addInlineEditors = function(parent) {
      # 记录日志：开始添加内联编辑器
      this.logStart("Adding inline editors");
      # 获取所有可见的带有 data-editable 属性的元素
      elems = $("[data-editable]:visible", parent);
      # 遍历所有元素
      for (_i = 0, _len = elems.length; _i < _len; _i++) {
        elem = elems[_i];
        elem = $(elem);
        # 如果元素没有绑定编辑器并且不是编辑器本身
        if (!elem.data("editor") && !elem.hasClass("editor")) {
          # 创建内联编辑器对象，并绑定到元素上
          editor = new InlineEditor(elem, this.getContent, this.saveContent, this.getObject);
          elem.data("editor", editor);
        }
      }
      # 记录日志：结束添加内联编辑器
      return this.logEnd("Adding inline editors");
    };

    # 检查发布栏状态
    ZeroBlog.prototype.checkPublishbar = function() {
      # 如果站点未修改或者站点修改时间晚于内容修改时间
      if (!this.site_modified || this.site_modified > this.site_info.content.modified) {
        # 添加可见类
        return $(".publishbar").addClass("visible");
      } else {
        # 移除可见类
        return $(".publishbar").removeClass("visible");
      }
    };

    # 发布站点
    ZeroBlog.prototype.publish = function() {
      # 弹出输入私钥的对话框
      this.cmd("wrapperPrompt", ["Enter your private key:", "password"], (function(_this) {
        return function(privatekey) {
          # 添加加载状态类
          $(".publishbar .button").addClass("loading");
          # 调用站点发布命令
          return _this.cmd("sitePublish", [privatekey], function(res) {
            # 移除加载状态类
            $(".publishbar .button").removeClass("loading");
            # 记录发布结果
            return _this.log("Publish result:", res);
          });
        };
      })(this));
      # 阻止默认行为
      return false;
    };
    // 定义 ZeroBlog 对象的 applyPostdata 方法，用于将帖子数据应用到指定元素上
    ZeroBlog.prototype.applyPostdata = function(elem, post, full) {
      var body, date_published, title_hash;
      // 如果未传入 full 参数，则默认为 false
      if (full == null) {
        full = false;
      }
      // 根据帖子标题生成标题哈希
      title_hash = post.title.replace(/[#?& ]/g, "+").replace(/[+]+/g, "+");
      // 将元素的 data 属性设置为帖子对象的标识
      elem.data("object", "Post:" + post.post_id);
      // 设置元素内部可编辑的标题内容和链接
      $(".title .editable", elem).html(post.title).attr("href", "?Post:" + post.post_id + ":" + title_hash).data("content", post.title);
      // 根据帖子发布日期计算距今时间，并根据帖子内容判断是否显示阅读时间
      date_published = Time.since(post.date_published);
      if (post.body.match(/^---/m)) {
        date_published += " &middot; " + (Time.readtime(post.body));
        $(".more", elem).css("display", "inline-block").attr("href", "?Post:" + post.post_id + ":" + title_hash);
      }
      // 设置元素内部的发布日期和评论数
      $(".details .published", elem).html(date_published).data("content", post.date_published);
      if (post.comments > 0) {
        $(".details .comments-num", elem).css("display", "inline").attr("href", "?Post:" + post.post_id + ":" + title_hash + "#Comments");
        $(".details .comments-num .num", elem).text(post.comments + " comments");
      } else {
        $(".details .comments-num", elem).css("display", "none");
      }
      // 根据 full 参数决定是否显示完整帖子内容
      if (full) {
        body = post.body;
      } else {
        body = post.body.replace(/^([\s\S]*?)\n---\n[\s\S]*$/, "$1");
      }
      // 将帖子内容转换为 Markdown 格式并设置到元素内部
      return $(".body", elem).html(Text.toMarked(body)).data("content", post.body);
    };

    // 定义 ZeroBlog 对象的 onOpenWebsocket 方法，用于处理 WebSocket 打开事件
    ZeroBlog.prototype.onOpenWebsocket = function(e) {
      // 加载数据、路由 URL 和获取站点信息
      this.loadData();
      this.routeUrl(window.location.search.substring(1));
      this.cmd("siteInfo", {}, this.setSiteinfo);
      // 获取服务器信息，并根据版本号决定是否加载旧数据
      return this.cmd("serverInfo", {}, (function(_this) {
        return function(ret) {
          _this.server_info = ret;
          if (_this.server_info.rev < 160) {
            return _this.loadData("old");
          }
        };
      })(this));
    };

    // 定义 ZeroBlog 对象的 getObject 方法，用于获取指定元素的父元素中的对象
    ZeroBlog.prototype.getObject = function(elem) {
      return elem.parents("[data-object]:first");
    };
    # 定义 ZeroBlog 对象的 getContent 方法，用于获取元素的内容
    ZeroBlog.prototype.getContent = function(elem, raw) {
      # 声明变量 content, id, type, _ref
      var content, id, type, _ref;
      # 如果 raw 未定义，则设置为 false
      if (raw == null) {
        raw = false;
      }
      # 从元素获取对象数据，分割成 type 和 id
      _ref = this.getObject(elem).data("object").split(":"), type = _ref[0], id = _ref[1];
      # 将 id 转换为整数
      id = parseInt(id);
      # 获取元素的内容
      content = elem.data("content");
      # 如果元素的可编辑模式为 timestamp，则将内容转换为完整的时间格式
      if (elem.data("editable-mode") === "timestamp") {
        content = Time.date(content, "full");
      }
      # 如果元素的可编辑模式为 simple 或者 raw 为 true，则直接返回内容
      if (elem.data("editable-mode") === "simple" || raw) {
        return content;
      } else {
        # 否则将内容转换为 Markdown 格式并返回
        return Text.toMarked(content);
      }
    };

    # 定义 ZeroBlog 对象的 saveContent 方法，用于保存元素的内容
    ZeroBlog.prototype.saveContent = function(elem, content, cb) {
      # 声明变量 id, type, _ref
      var id, type, _ref;
      # 如果 cb 未定义，则设置为 false
      if (cb == null) {
        cb = false;
      }
      # 如果元素可删除且内容为 null，则删除该元素并执行回调函数
      if (elem.data("deletable") && content === null) {
        return this.deleteObject(elem, cb);
      }
      # 更新元素的内容
      elem.data("content", content);
      # 从元素获取对象数据，分割成 type 和 id
      _ref = this.getObject(elem).data("object").split(":"), type = _ref[0], id = _ref[1];
      # 将 id 转换为整数
      id = parseInt(id);
      # 根据对象类型调用相应的保存方法
      if (type === "Post" || type === "Site") {
        return this.saveSite(elem, type, id, content, cb);
      } else if (type === "Comment") {
        return this.saveComment(elem, type, id, content, cb);
      }
    };
    // 定义 ZeroBlog 对象的 saveSite 方法，用于保存站点信息
    ZeroBlog.prototype.saveSite = function(elem, type, id, content, cb) {
      // 调用 cmd 方法，获取 data.json 文件内容
      return this.cmd("fileGet", ["data/data.json"], (function(_this) {
        return function(res) {
          // 解析获取的 JSON 数据
          var data, post;
          data = JSON.parse(res);
          // 如果类型是 "Post"
          if (type === "Post") {
            // 查找指定 id 的 post
            post = ((function() {
              var _i, _len, _ref, _results;
              _ref = data.post;
              _results = [];
              for (_i = 0, _len = _ref.length; _i < _len; _i++) {
                post = _ref[_i];
                if (post.post_id === id) {
                  _results.push(post);
                }
              }
              return _results;
            })())[0];
            // 如果元素的可编辑模式是 "timestamp"，则将内容转换为时间戳
            if (elem.data("editable-mode") === "timestamp") {
              content = Time.timestamp(content);
            }
            // 更新 post 对象中的可编辑内容
            post[elem.data("editable")] = content;
          } else if (type === "Site") {
            // 如果类型是 "Site"，直接更新 data 对象中的可编辑内容
            data[elem.data("editable")] = content;
          }
          // 调用 writeData 方法，将更新后的 data 对象写入文件
          return _this.writeData(data, function(res) {
            // 如果有回调函数
            if (cb) {
              // 根据返回结果执行回调
              if (res === true) {
                if (elem.data("editable-mode") === "simple") {
                  return cb(content);
                } else if (elem.data("editable-mode") === "timestamp") {
                  return cb(Time.since(content));
                } else {
                  return cb(Text.toMarked(content));
                }
              } else {
                return cb(false);
              }
            }
          });
        };
      })(this));
    };
    // 保存评论的方法，接受元素、类型、ID、内容和回调函数作为参数
    ZeroBlog.prototype.saveComment = function(elem, type, id, content, cb) {
      // 打印保存评论的日志信息和评论ID
      this.log("Saving comment...", id);
      // 获取元素对象并设置其高度为自动
      this.getObject(elem).css("height", "auto");
      // 内部路径为用户数据的 JSON 文件路径
      inner_path = "data/users/" + Page.site_info.auth_address + "/data.json";
      // 调用 Page.cmd 方法，获取指定内部路径的文件内容
      return Page.cmd("fileGet", {
        "inner_path": inner_path,
        "required": false
      }, (function(_this) {
        return function(data) {
          // 解析获取的文件内容为 JSON 格式
          data = JSON.parse(data);
          // 从数据中找到指定评论ID对应的评论
          comment = ((function() {
            var _i, _len, _ref, _results;
            _ref = data.comment;
            _results = [];
            for (_i = 0, _len = _ref.length; _i < _len; _i++) {
              comment = _ref[_i];
              if (comment.comment_id === id) {
                _results.push(comment);
              }
            }
            return _results;
          })())[0];
          // 更新评论中指定元素的内容为传入的内容
          comment[elem.data("editable")] = content;
          // 打印更新后的数据
          _this.log(data);
          // 将更新后的 JSON 数据转换为 base64 编码的字符串，并写入指定的内部路径
          json_raw = unescape(encodeURIComponent(JSON.stringify(data, void 0, '\t')));
          return _this.writePublish(inner_path, btoa(json_raw), function(res) {
            // 如果写入成功，则检查证书并执行回调函数
            if (res === true) {
              Comments.checkCert("updaterules");
              if (cb) {
                return cb(Text.toMarked(content, {
                  "sanitize": true
                }));
              }
            } else {
              // 如果写入失败，则显示错误通知，并执行回调函数
              _this.cmd("wrapperNotification", ["error", "File write error: " + res]);
              if (cb) {
                return cb(false);
              }
            }
          });
        };
      })(this));
    };
    // 在 ZeroBlog 对象的原型上定义一个方法，用于写入数据
    ZeroBlog.prototype.writeData = function(data, cb) {
      var json_raw;
      // 如果回调函数未定义，则设为 null
      if (cb == null) {
        cb = null;
      }
      // 如果数据不存在，则记录日志并返回
      if (!data) {
        return this.log("Data missing");
      }
      // 更新数据的修改时间戳
      this.data["modified"] = data.modified = Time.timestamp();
      // 将数据转换为 JSON 字符串，然后进行 URI 编码
      json_raw = unescape(encodeURIComponent(JSON.stringify(data, void 0, '\t')));
      // 调用 cmd 方法，写入经过 base64 编码的 JSON 数据到指定路径的文件
      this.cmd("fileWrite", ["data/data.json", btoa(json_raw)], (function(_this) {
        return function(res) {
          // 如果写入成功，则执行回调函数
          if (res === "ok") {
            if (cb) {
              cb(true);
            }
          } else {
            // 如果写入失败，则显示错误通知，并执行回调函数
            _this.cmd("wrapperNotification", ["error", "File write error: " + res]);
            if (cb) {
              cb(false);
            }
          }
          // 检查发布栏状态
          return _this.checkPublishbar();
        };
      })(this));
      // 调用 cmd 方法，获取指定路径文件的内容
      return this.cmd("fileGet", ["content.json"], (function(_this) {
        return function(content) {
          // 替换内容中的标题字段为新的标题
          content = content.replace(/"title": ".*?"/, "\"title\": \"" + data.title + "\"");
          // 调用 cmd 方法，将经过 base64 编码的内容写入指定路径的文件
          return _this.cmd("fileWrite", ["content.json", btoa(content)], function(res) {
            // 如果写入失败，则显示错误通知
            if (res !== "ok") {
              return _this.cmd("wrapperNotification", ["error", "Content.json write error: " + res]);
            }
          });
        };
      })(this));
    };

    // 在 ZeroBlog 对象的原型上定义一个方法，用于写入发布内容
    ZeroBlog.prototype.writePublish = function(inner_path, data, cb) {
      // 调用 cmd 方法，将数据写入指定路径的文件
      return this.cmd("fileWrite", [inner_path, data], (function(_this) {
        return function(res) {
          // 如果写入失败，则显示错误通知，并执行回调函数
          if (res !== "ok") {
            _this.cmd("wrapperNotification", ["error", "File write error: " + res]);
            cb(false);
            return false;
          }
          // 调用 cmd 方法，发布指定路径的文件
          return _this.cmd("sitePublish", {
            "inner_path": inner_path
          }, function(res) {
            // 如果发布成功，则执行回调函数
            if (res === "ok") {
              return cb(true);
            } else {
              // 如果发布失败，则执行回调函数
              return cb(res);
            }
          });
        };
      })(this));
    };
    # 当收到请求时的处理函数，根据命令执行相应的操作
    ZeroBlog.prototype.onRequest = function(cmd, message) {
      # 如果命令是设置站点信息，则执行设置站点信息的操作
      if (cmd === "setSiteInfo") {
        return this.actionSetSiteInfo(message);
      } else {
        # 否则记录未知命令的日志
        return this.log("Unknown command", message);
      }
    };

    # 设置站点信息的操作函数
    ZeroBlog.prototype.actionSetSiteInfo = function(message) {
      # 调用设置站点信息的方法
      this.setSiteinfo(message.params);
      # 检查发布栏
      return this.checkPublishbar();
    };

    # 设置站点信息的方法
    ZeroBlog.prototype.setSiteinfo = function(site_info) {
      # 设置站点信息
      var _ref, _ref1;
      this.site_info = site_info;
      # 解析站点信息的事件
      this.event_site_info.resolve(site_info);
      # 如果页面是文章页面，则检查评论证书
      if ($("body").hasClass("page-post")) {
        Comments.checkCert();
      }
      # 如果事件是文件完成，并且文件名匹配特定模式
      if (((_ref = site_info.event) != null ? _ref[0] : void 0) === "file_done" && site_info.event[1].match(/.*users.*data.json$/)) {
        # 如果页面是文章页面，则加载评论
        if ($("body").hasClass("page-post")) {
          Comments.loadComments();
        }
        # 如果页面是主页面，则执行主页面操作
        if ($("body").hasClass("page-main")) {
          return RateLimit(500, (function(_this) {
            return function() {
              return _this.pageMain();
            };
          })(this));
        }
      } else if (((_ref1 = site_info.event) != null ? _ref1[0] : void 0) === "file_done" && site_info.event[1] === "data/data.json") {
        # 加载数据
        this.loadData();
        # 如果页面是主页面，则执行主页面操作
        if ($("body").hasClass("page-main")) {
          this.pageMain();
        }
        # 如果页面是文章页面，则执行文章页面操作
        if ($("body").hasClass("page-post")) {
          return this.pagePost();
        }
      } else {
        # 其他情况
      }
    };

    # 返回 ZeroBlog 类
    return ZeroBlog;

  })(ZeroFrame);

  # 创建 ZeroBlog 实例并赋值给 window.Page
  window.Page = new ZeroBlog();
# 调用匿名函数，并将当前上下文作为参数传入
}).call(this);
```