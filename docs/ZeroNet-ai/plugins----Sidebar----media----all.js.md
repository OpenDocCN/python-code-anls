# `ZeroNet\plugins\Sidebar\media\all.js`

```py
# 定义一个 Class 类
(function() {
  var Class,
    slice = [].slice;

  # 创建 Class 类
  Class = (function() {
    function Class() {}

    # 设置 trace 属性为 true
    Class.prototype.trace = true;

    # 定义 log 方法，用于打印日志
    Class.prototype.log = function() {
      var args;
      args = 1 <= arguments.length ? slice.call(arguments, 0) : [];
      # 如果 trace 为 false，则直接返回
      if (!this.trace) {
        return;
      }
      # 如果 console 未定义，则直接返回
      if (typeof console === 'undefined') {
        return;
      }
      # 在日志信息前添加类名，然后调用 console.log 方法打印日志
      args.unshift("[" + this.constructor.name + "]");
      console.log.apply(console, args);
      return this;
    };

    # 定义 logStart 方法，用于打印开始日志
    Class.prototype.logStart = function() {
      var args, name;
      name = arguments[0], args = 2 <= arguments.length ? slice.call(arguments, 1) : [];
      # 如果 trace 为 false，则直接返回
      if (!this.trace) {
        return;
      }
      # 如果 logtimers 未定义，则创建一个空对象
      this.logtimers || (this.logtimers = {});
      # 将当前时间存储到 logtimers 对象中
      this.logtimers[name] = +(new Date);
      # 如果参数个数大于 0，则调用 log 方法打印开始日志
      if (args.length > 0) {
        this.log.apply(this, ["" + name].concat(slice.call(args), ["(started)"]));
      }
      return this;
    };

    # 定义 logEnd 方法，用于打印结束日志
    Class.prototype.logEnd = function() {
      var args, ms, name;
      name = arguments[0], args = 2 <= arguments.length ? slice.call(arguments, 1) : [];
      # 计算时间差
      ms = +(new Date) - this.logtimers[name];
      # 调用 log 方法打印结束日志
      this.log.apply(this, ["" + name].concat(slice.call(args), ["(Done in " + ms + "ms)"]));
      return this;
    };

    return Class;

  })();

  # 将 Class 类绑定到 window 对象上
  window.Class = Class;

}).call(this);

# 定义一个 Console 类
(function() {
  var Console,
    bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; },
    extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
    hasProp = {}.hasOwnProperty;

  # 创建 Console 类，继承自 superClass
  Console = (function(superClass) {
    extend(Console, superClass);

    # ...
    # 此处省略了部分代码
    # ...

  });

}).call(this);
    // 检查文本是否滚动到底部，返回布尔值
    Console.prototype.checkTextIsBottom = function() {
      return this.text.is_bottom = Math.round(this.text_elem.scrollTop + this.text_elem.clientHeight) >= this.text_elem.scrollHeight - 15;
    };

    // 将文本转换为指定饱和度和亮度的颜色值，返回 HSL 颜色字符串
    Console.prototype.toColor = function(text, saturation, lightness) {
      var hash, i, j, ref;
      if (saturation == null) {
        saturation = 60;
      }
      if (lightness == null) {
        lightness = 70;
      }
      hash = 0;
      for (i = j = 0, ref = text.length - 1; 0 <= ref ? j <= ref : j >= ref; i = 0 <= ref ? ++j : --j) {
        hash += text.charCodeAt(i) * i;
        hash = hash % 1777;
      }
      return "hsl(" + (hash % 360) + ("," + saturation + "%," + lightness + "%)");
    };

    // 格式化一行文本，返回格式化后的文本
    Console.prototype.formatLine = function(line) {
      var added, level, match, module, ref, text;
      match = line.match(/(\[.*?\])[ ]+(.*?)[ ]+(.*?)[ ]+(.*)/);
      if (!match) {
        return line.replace(/\</g, "&lt;").replace(/\>/g, "&gt;");
      }
      ref = line.match(/(\[.*?\])[ ]+(.*?)[ ]+(.*?)[ ]+(.*)/), line = ref[0], added = ref[1], level = ref[2], module = ref[3], text = ref[4];
      added = "<span style='color: #dfd0fa'>" + added + "</span>";
      level = "<span style='color: " + (this.toColor(level, 100)) + ";'>" + level + "</span>";
      module = "<span style='color: " + (this.toColor(module, 60)) + "; font-weight: bold;'>" + module + "</span>";
      text = text.replace(/(Site:[A-Za-z0-9\.]+)/g, "<span style='color: #AAAAFF'>$1</span>");
      text = text.replace(/\</g, "&lt;").replace(/\>/g, "&gt;");
      return added + " " + level + " " + module + " " + text;
    };
    // 向控制台添加多行文本，可以选择是否使用动画效果
    Console.prototype.addLines = function(lines, animate) {
      var html_lines, j, len, line;
      // 如果未指定是否使用动画效果，默认为使用
      if (animate == null) {
        animate = true;
      }
      // 初始化 HTML 行数组
      html_lines = [];
      // 记录日志：开始格式化
      this.logStart("formatting");
      // 遍历每一行文本，格式化后添加到 HTML 行数组中
      for (j = 0, len = lines.length; j < len; j++) {
        line = lines[j];
        html_lines.push(this.formatLine(line));
      }
      // 记录日志：结束格式化
      this.logEnd("formatting");
      // 记录日志：开始添加
      this.text.append(html_lines.join("<br>") + "<br>");
      // 记录日志：结束添加
      this.logEnd("adding");
      // 如果文本框滚动条在底部且使用动画效果，则滚动到最底部
      if (this.text.is_bottom && animate) {
        return this.text.stop().animate({
          scrollTop: this.text_elem.scrollHeight - this.text_elem.clientHeight + 1
        }, 600, 'easeInOutCubic');
      }
    };

    // 加载控制台文本内容
    Console.prototype.loadConsoleText = function() {
      // 通过 WebSocket 发送命令，读取控制台日志
      this.sidebar.wrapper.ws.cmd("consoleLogRead", {
        filter: this.filter,
        read_size: this.read_size
      }, (function(_this) {
        return function(res) {
          var pos_diff, size_read, size_total;
          // 清空文本内容
          _this.text.html("");
          // 计算已读取日志的大小
          pos_diff = res["pos_end"] - res["pos_start"];
          size_read = Math.round(pos_diff / 1024);
          size_total = Math.round(res['pos_end'] / 1024);
          // 添加显示已读取日志的信息
          _this.text.append("<br><br>");
          _this.text.append("Displaying " + res.lines.length + " of " + res.num_found + " lines found in the last " + size_read + "kB of the log file. (" + size_total + "kB total)<br>");
          // 添加已读取日志的内容，不使用动画效果
          _this.addLines(res.lines, false);
          // 将文本框滚动条滚动到最底部
          return _this.text_elem.scrollTop = _this.text_elem.scrollHeight;
        };
      })(this));
      // 如果存在流 ID，则移除控制台日志流
      if (this.stream_id) {
        this.sidebar.wrapper.ws.cmd("consoleLogStreamRemove", {
          stream_id: this.stream_id
        });
      }
      // 通过 WebSocket 发送命令，创建控制台日志流
      return this.sidebar.wrapper.ws.cmd("consoleLogStream", {
        filter: this.filter
      }, (function(_this) {
        return function(res) {
          return _this.stream_id = res.stream_id;
        };
      })(this));
    };
    // 关闭控制台，将浏览器地址栏的 hash 设置为空，锁定侧边栏移动，开始停止拖动侧边栏
    Console.prototype.close = function() {
      window.top.location.hash = "";
      this.sidebar.move_lock = "y";
      this.sidebar.startDrag();
      return this.sidebar.stopDrag();
    };

    // 打开控制台，开始拖动侧边栏，移动侧边栏，设置固定按钮的目标位置，停止拖动侧边栏
    Console.prototype.open = function() {
      this.sidebar.startDrag();
      this.sidebar.moved("y");
      this.sidebar.fixbutton_targety = this.sidebar.page_height - this.sidebar.fixbutton_inity - 50;
      return this.sidebar.stopDrag();
    };

    // 控制台打开时的操作，关闭侧边栏，记录日志
    Console.prototype.onOpened = function() {
      this.sidebar.onClosed();
      return this.log("onOpened");
    };

    // 控制台关闭时的操作，移除控制台样式，如果有流 ID，则通知侧边栏移除控制台日志流
    Console.prototype.onClosed = function() {
      $(document.body).removeClass("body-console");
      if (this.stream_id) {
        return this.sidebar.wrapper.ws.cmd("consoleLogStreamRemove", {
          stream_id: this.stream_id
        });
      }
    };

    // 清理操作，如果存在容器，则移除容器并置空
    Console.prototype.cleanup = function() {
      if (this.container) {
        this.container.remove();
        return this.container = null;
      }
    };

    // 停止在 Y 轴上的拖动操作，根据固定按钮的目标位置进行操作，记录日志，如果控制台关闭则执行 onClosed 操作
    Console.prototype.stopDragY = function() {
      var targety;
      if (this.sidebar.fixbutton_targety === this.sidebar.fixbutton_inity) {
        targety = 0;
        this.opened = false;
      } else {
        targety = this.sidebar.fixbutton_targety - this.sidebar.fixbutton_inity;
        this.onOpened();
        this.opened = true;
      }
      if (this.tag) {
        this.tag.css("transition", "0.5s ease-out");
        this.tag.css("transform", "translateY(" + targety + "px)").one(transitionEnd, (function(_this) {
          return function() {
            _this.tag.css("transition", "");
            if (!_this.opened) {
              return _this.cleanup();
            }
          };
        })(this));
      }
      this.log("stopDragY", "opened:", this.opened, targety);
      if (!this.opened) {
        return this.onClosed();
      }
    };
    # 定义一个方法，用于改变控制台的过滤器
    Console.prototype.changeFilter = function(filter) {
      # 将传入的过滤器赋值给当前对象的过滤器属性
      this.filter = filter;
      # 如果过滤器为空字符串，则设置读取大小为32KB
      if (this.filter === "") {
        this.read_size = 32 * 1024;
      } else {
        # 如果过滤器不为空，则设置读取大小为5MB
        this.read_size = 5 * 1024 * 1024;
      }
      # 调用loadConsoleText方法加载控制台文本
      return this.loadConsoleText();
    };

    # 定义一个方法，用于处理标签点击事件
    Console.prototype.handleTabClick = function(e) {
      # 获取当前点击的元素
      var elem;
      elem = $(e.currentTarget);
      # 将当前点击的标签的过滤器设置为活动过滤器
      this.tab_active = elem.data("filter");
      # 移除所有标签的active类
      $("a", this.tabs).removeClass("active");
      # 给当前点击的标签添加active类
      elem.addClass("active");
      # 调用changeFilter方法，将当前标签的过滤器作为参数
      this.changeFilter(this.tab_active);
      # 设置浏览器地址栏的hash值
      window.top.location.hash = "#ZeroNet:Console:" + elem.data("title");
      # 阻止默认行为
      return false;
    };

    # 返回Console类
    return Console;

  })(Class);

  # 将Console类赋值给window对象的Console属性
  window.Console = Console;
}).call(this);

/* ---- Menu.coffee ---- */

// 创建一个匿名函数，用于定义 Menu 类
(function() {
  var Menu,
    slice = [].slice;

  // 定义 Menu 类
  Menu = (function() {
    // Menu 类的构造函数，接受一个按钮作为参数
    function Menu(button) {
      this.button = button;
      // 克隆并移除模板元素，将其添加到 body 中
      this.elem = $(".menu.template").clone().removeClass("template");
      this.elem.appendTo("body");
      this.items = [];
    }

    // 定义 Menu 类的 show 方法
    Menu.prototype.show = function() {
      var button_pos, left;
      // 如果已经有菜单显示，并且当前菜单与之前显示的菜单是同一个按钮，则隐藏当前菜单并返回
      if (window.visible_menu && window.visible_menu.button[0] === this.button[0]) {
        window.visible_menu.hide();
        return this.hide();
      } else {
        // 获取按钮的位置信息
        button_pos = this.button.offset();
        left = button_pos.left;
        // 设置菜单的位置
        this.elem.css({
          "top": button_pos.top + this.button.outerHeight(),
          "left": left
        });
        this.button.addClass("menu-active");
        this.elem.addClass("visible");
        // 如果菜单超出窗口右侧，则调整位置
        if (this.elem.position().left + this.elem.width() + 20 > window.innerWidth) {
          this.elem.css("left", window.innerWidth - this.elem.width() - 20);
        }
        // 如果已经有菜单显示，则隐藏之前的菜单
        if (window.visible_menu) {
          window.visible_menu.hide();
        }
        // 将当前菜单设置为可见菜单
        return window.visible_menu = this;
      }
    };

    // 定义 Menu 类的 hide 方法
    Menu.prototype.hide = function() {
      // 隐藏菜单并移除菜单按钮的激活样式
      this.elem.removeClass("visible");
      this.button.removeClass("menu-active");
      // 将可见菜单设置为 null
      return window.visible_menu = null;
    };

    // 定义 Menu 类的 addItem 方法
    Menu.prototype.addItem = function(title, cb) {
      var item;
      // 克隆并移除模板元素，设置菜单项的标题
      item = $(".menu-item.template", this.elem).clone().removeClass("template");
      item.html(title);
      // 绑定菜单项的点击事件
      item.on("click", (function(_this) {
        return function() {
          // 如果回调函数返回 false，则隐藏菜单
          if (!cb(item)) {
            _this.hide();
          }
          return false;
        };
      })(this));
      item.appendTo(this.elem);
      this.items.push(item);
      return item;
    };

    // 定义 Menu 类的 log 方法
    Menu.prototype.log = function() {
      var args;
      args = 1 <= arguments.length ? slice.call(arguments, 0) : [];
      // 打印日志
      return console.log.apply(console, ["[Menu]"].concat(slice.call(args)));
    };
    # 返回 Menu 类
    return Menu;

  })();

  # 将 Menu 类绑定到 window 对象上
  window.Menu = Menu;

  # 当点击页面时触发事件
  $("body").on("click", function(e) {
    # 如果菜单可见，并且点击的目标不是菜单按钮，并且点击的目标的父元素不是菜单元素，则隐藏菜单
    if (window.visible_menu && e.target !== window.visible_menu.button[0] && $(e.target).parent()[0] !== window.visible_menu.elem[0]) {
      return window.visible_menu.hide();
    }
  });
// 定义一个立即执行函数，用于扩展 String 对象的原型
(function() {
  // 在 String 对象的原型上添加 startsWith 方法，用于判断字符串是否以指定字符串开头
  String.prototype.startsWith = function(s) {
    return this.slice(0, s.length) === s;
  };
  // 在 String 对象的原型上添加 endsWith 方法，用于判断字符串是否以指定字符串结尾
  String.prototype.endsWith = function(s) {
    return s === '' || this.slice(-s.length) === s;
  };
  // 在 String 对象的原型上添加 capitalize 方法，用于将字符串首字母大写
  String.prototype.capitalize = function() {
    if (this.length) {
      return this[0].toUpperCase() + this.slice(1);
    } else {
      return "";
    }
  };
  // 在 String 对象的原型上添加 repeat 方法，用于重复字符串指定次数
  String.prototype.repeat = function(count) {
    return new Array(count + 1).join(this);
  };
  // 定义全局函数 isEmpty，用于判断对象是否为空
  window.isEmpty = function(obj) {
    var key;
    for (key in obj) {
      return false;
    }
    return true;
  };
}).call(this);

// 定义一个立即执行函数，用于实现限流函数 RateLimit
(function() {
  // 定义 limits 对象，用于存储函数的限流状态
  var limits = {};
  // 定义 call_after_interval 对象，用于存储函数的延迟调用状态
  var call_after_interval = {};
  // 定义全局函数 RateLimit，用于限流调用指定函数
  window.RateLimit = function(interval, fn) {
    // 如果函数未被限流
    if (!limits[fn]) {
      // 设置函数的延迟调用状态为 false
      call_after_interval[fn] = false;
      // 立即调用函数
      fn();
      // 设置函数的限流状态，并在指定时间间隔后清除限流状态和延迟调用状态
      return limits[fn] = setTimeout((function() {
        if (call_after_interval[fn]) {
          fn();
        }
        delete limits[fn];
        return delete call_after_interval[fn];
      }), interval);
    } else {
      // 如果函数已被限流，则设置延迟调用状态为 true
      return call_after_interval[fn] = true;
    }
  };
}).call(this);

// 初始化滚动容器和内容
window.initScrollable = function () {
  // 获取滚动容器、内容包裹器和内容元素
  var scrollContainer = document.querySelector('.scrollable'),
      scrollContentWrapper = document.querySelector('.scrollable .content-wrapper'),
      scrollContent = document.querySelector('.scrollable .content'),
      contentPosition = 0, // 内容位置
      scrollerBeingDragged = false, // 拖动滚动条的标志
      scroller, // 滚动条
      topPosition, // 顶部位置
      scrollerHeight; // 滚动条高度
};
    // 计算滚动条应该有多高
    function calculateScrollerHeight() {
        // 计算可见比例
        var visibleRatio = scrollContainer.offsetHeight / scrollContentWrapper.scrollHeight;
        // 如果可见比例为1，则隐藏滚动条
        if (visibleRatio == 1)
            scroller.style.display = "none";
        else
            scroller.style.display = "block";
        // 返回滚动条高度
        return visibleRatio * scrollContainer.offsetHeight;
    }

    // 移动滚动条
    function moveScroller(evt) {
        // 计算滚动百分比
        var scrollPercentage = evt.target.scrollTop / scrollContentWrapper.scrollHeight;
        // 计算滚动条的位置
        topPosition = scrollPercentage * (scrollContainer.offsetHeight - 5); // 5px arbitrary offset so scroll bar doesn't move too far beyond content wrapper bounding box
        scroller.style.top = topPosition + 'px';
    }

    // 开始拖动滚动条
    function startDrag(evt) {
        normalizedPosition = evt.pageY;
        contentPosition = scrollContentWrapper.scrollTop;
        scrollerBeingDragged = true;
        window.addEventListener('mousemove', scrollBarScroll);
        return false;
    }

    // 停止拖动滚动条
    function stopDrag(evt) {
        scrollerBeingDragged = false;
        window.removeEventListener('mousemove', scrollBarScroll);
    }

    // 滚动条滚动
    function scrollBarScroll(evt) {
        if (scrollerBeingDragged === true) {
            evt.preventDefault();
            var mouseDifferential = evt.pageY - normalizedPosition;
            var scrollEquivalent = mouseDifferential * (scrollContentWrapper.scrollHeight / scrollContainer.offsetHeight);
            scrollContentWrapper.scrollTop = contentPosition + scrollEquivalent;
        }
    }

    // 更新滚动条高度
    function updateHeight() {
        // 计算滚动条高度并减去10
        scrollerHeight = calculateScrollerHeight() - 10;
        scroller.style.height = scrollerHeight + 'px';
    }
    function createScroller() {
        // *Creates scroller element and appends to '.scrollable' div
        // 创建滚动条元素
        scroller = document.createElement("div");
        scroller.className = 'scroller';

        // 根据内容确定滚动条的大小
        scrollerHeight = calculateScrollerHeight() - 10;

        if (scrollerHeight / scrollContainer.offsetHeight < 1) {
            // *If there is a need to have scroll bar based on content size
            // 如果内容大小需要滚动条
            scroller.style.height = scrollerHeight + 'px';

            // 将滚动条添加到 scrollContainer div
            scrollContainer.appendChild(scroller);

            // 显示滚动路径
            scrollContainer.className += ' showScroll';

            // 附加相关的可拖动监听器
            scroller.addEventListener('mousedown', startDrag);
            window.addEventListener('mouseup', stopDrag);
        }

    }

    createScroller();


    // *** Listeners ***
    // 滚动内容包装器的滚动监听器
    scrollContentWrapper.addEventListener('scroll', moveScroller);

    // 返回更新高度的函数
    return updateHeight;
# 定义一个自执行函数，用于创建侧边栏
(function() {
  # 定义 Sidebar 类
  var Sidebar, wrapper,
    # 绑定函数的上下文
    bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; },
    # 继承父类的属性和方法
    extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
    # 判断对象是否包含指定属性
    hasProp = {}.hasOwnProperty,
    # 查找数组中指定元素的索引
    indexOf = [].indexOf || function(item) { for (var i = 0, l = this.length; i < l; i++) { if (i in this && this[i] === item) return i; } return -1; };
  # 定义 Sidebar 类，继承自 superClass
  Sidebar = (function(superClass) {
    # 继承 superClass
    extend(Sidebar, superClass);
    # 定义 Sidebar 类
    function Sidebar(wrapper1) {
      # 初始化 Sidebar 对象的属性
      this.wrapper = wrapper1;
      this.unloadGlobe = bind(this.unloadGlobe, this);
      this.displayGlobe = bind(this.displayGlobe, this);
      this.loadGlobe = bind(this.loadGlobe, this);
      this.animDrag = bind(this.animDrag, this);
      this.setHtmlTag = bind(this.setHtmlTag, this);
      this.waitMove = bind(this.waitMove, this);
      this.resized = bind(this.resized, this);
      this.tag = null;
      this.container = null;
      this.opened = false;
      this.width = 410;
      this.console = new Console(this);
      this.fixbutton = $(".fixbutton");
      this.fixbutton_addx = 0;
      this.fixbutton_addy = 0;
      this.fixbutton_initx = 0;
      this.fixbutton_inity = 15;
      this.fixbutton_targetx = 0;
      this.move_lock = null;
      this.page_width = $(window).width();
      this.page_height = $(window).height();
      this.frame = $("#inner-iframe");
      this.initFixbutton();
      this.dragStarted = 0;
      this.globe = null;
      this.preload_html = null;
      this.original_set_site_info = this.wrapper.setSiteInfo;
      # 如果当前页面的 URL 带有指定的 hash 标记，则执行以下操作
      if (window.top.location.hash === "#ZeroNet:OpenSidebar") {
        # 开始拖动 Sidebar
        this.startDrag();
        # 移动 Sidebar
        this.moved("x");
        # 设置 fixbutton 的目标位置
        this.fixbutton_targetx = this.fixbutton_initx - this.width;
        # 停止拖动 Sidebar
        this.stopDrag();
      }
    }
    # 初始化固定按钮
    Sidebar.prototype.initFixbutton = function() {
      # 绑定 mousedown 和 touchstart 事件处理函数
      this.fixbutton.on("mousedown touchstart", (function(_this) {
        return function(e) {
          # 如果不是鼠标左键点击，则返回
          if (e.button > 0) {
            return;
          }
          # 阻止默认事件
          e.preventDefault();
          # 解绑 click、touchend 和 touchcancel 事件处理函数
          _this.fixbutton.off("click touchend touchcancel");
          # 记录拖拽开始的时间
          _this.dragStarted = +(new Date);
          # 移除之前的拖拽背景
          $(".drag-bg").remove();
          # 创建新的拖拽背景并添加到 body 中
          $("<div class='drag-bg'></div>").appendTo(document.body);
          # 绑定一次 mousemove 和 touchmove 事件处理函数
          return $("body").one("mousemove touchmove", function(e) {
            var mousex, mousey;
            mousex = e.pageX;
            mousey = e.pageY;
            if (!mousex) {
              mousex = e.originalEvent.touches[0].pageX;
              mousey = e.originalEvent.touches[0].pageY;
            }
            # 计算鼠标位置和按钮位置的偏移量
            _this.fixbutton_addx = _this.fixbutton.offset().left - mousex;
            _this.fixbutton_addy = _this.fixbutton.offset().top - mousey;
            # 开始拖拽
            return _this.startDrag();
          });
        };
      })(this));
      # 绑定 click、touchend 和 touchcancel 事件处理函数
      this.fixbutton.parent().on("click touchend touchcancel", (function(_this) {
        return function(e) {
          # 如果拖拽时间小于 100 毫秒，则跳转到按钮链接指定的页面
          if ((+(new Date)) - _this.dragStarted < 100) {
            window.top.location = _this.fixbutton.find(".fixbutton-bg").attr("href");
          }
          # 停止拖拽
          return _this.stopDrag();
        };
      })(this));
      # 调用 resized 方法
      this.resized();
      # 绑定 resize 事件处理函数
      return $(window).on("resize", this.resized);
    };

    # 窗口大小改变时的处理函数
    Sidebar.prototype.resized = function() {
      # 获取窗口宽度和高度
      this.page_width = $(window).width();
      this.page_height = $(window).height();
      # 初始化固定按钮的 x 坐标
      this.fixbutton_initx = this.page_width - 75;
      # 如果侧边栏已打开，则设置固定按钮的左偏移量为初始值减去侧边栏宽度，否则为初始值
      if (this.opened) {
        return this.fixbutton.css({
          left: this.fixbutton_initx - this.width
        });
      } else {
        return this.fixbutton.css({
          left: this.fixbutton_initx
        });
      }
    };
    // 定义 Sidebar 原型对象的 startDrag 方法
    Sidebar.prototype.startDrag = function() {
      // 记录日志，记录初始固定按钮的位置
      this.log("startDrag", this.fixbutton_initx, this.fixbutton_inity);
      // 设置固定按钮的目标位置为初始位置
      this.fixbutton_targetx = this.fixbutton_initx;
      this.fixbutton_targety = this.fixbutton_inity;
      // 给固定按钮添加 dragging 类
      this.fixbutton.addClass("dragging");
      // 如果是 IE 浏览器，禁用指针事件
      if (navigator.userAgent.indexOf('MSIE') !== -1 || navigator.appVersion.indexOf('Trident/') > 0) {
        this.fixbutton.css("pointer-events", "none");
      }
      // 一次性绑定 click 事件处理函数
      this.fixbutton.one("click", (function(_this) {
        return function(e) {
          // 停止拖动
          _this.stopDrag();
          // 移除 dragging 类
          _this.fixbutton.removeClass("dragging");
          // 计算按钮移动的距离
          moved_x = Math.abs(_this.fixbutton.offset().left - _this.fixbutton_initx);
          moved_y = Math.abs(_this.fixbutton.offset().top - _this.fixbutton_inity);
          // 如果移动距离超过阈值，阻止默认事件
          if (moved_x > 5 || moved_y > 10) {
            return e.preventDefault();
          }
        };
      })(this));
      // 绑定父元素的 mousemove 和 touchmove 事件处理函数
      this.fixbutton.parents().on("mousemove touchmove", this.animDrag);
      // 绑定父元素的 mousemove 和 touchmove 事件处理函数
      this.fixbutton.parents().on("mousemove touchmove", this.waitMove);
      // 一次性绑定父元素的 mouseup、touchend 和 touchcancel 事件处理函数
      return this.fixbutton.parents().one("mouseup touchend touchcancel", (function(_this) {
        return function(e) {
          // 阻止默认事件
          e.preventDefault();
          // 停止拖动
          return _this.stopDrag();
        };
      })(this));
    };
    // 定义 Sidebar 对象的 waitMove 方法，处理鼠标移动事件
    Sidebar.prototype.waitMove = function(e) {
      // 设置整个页面的透视效果
      var moved_x, moved_y;
      document.body.style.perspective = "1000px";
      // 设置整个页面的高度为100%
      document.body.style.height = "100%";
      // 设置整个页面的透视效果将要发生改变
      document.body.style.willChange = "perspective";
      document.documentElement.style.height = "100%";
      // 计算横向和纵向移动的距离
      moved_x = Math.abs(parseInt(this.fixbutton[0].style.left) - this.fixbutton_targetx);
      moved_y = Math.abs(parseInt(this.fixbutton[0].style.top) - this.fixbutton_targety);
      // 如果横向移动距离大于5且时间超过50毫秒，则执行以下操作
      if (moved_x > 5 && (+(new Date)) - this.dragStarted + moved_x > 50) {
        // 执行横向移动操作
        this.moved("x");
        // 停止按钮动画，并在1秒内移动到初始位置
        this.fixbutton.stop().animate({
          "top": this.fixbutton_inity
        }, 1000);
        // 解除父元素的鼠标移动事件监听
        return this.fixbutton.parents().off("mousemove touchmove", this.waitMove);
      } 
      // 如果纵向移动距离大于5且时间超过50毫秒，则执行以下操作
      else if (moved_y > 5 && (+(new Date)) - this.dragStarted + moved_y > 50) {
        // 执行纵向移动操作
        this.moved("y");
        // 解除父元素的鼠标移动事件监听
        return this.fixbutton.parents().off("mousemove touchmove", this.waitMove);
      }
    };
    // 定义 Sidebar 对象的 moved 方法，用于处理侧边栏移动事件
    Sidebar.prototype.moved = function(direction) {
      var img;
      // 记录移动方向
      this.log("Moved", direction);
      // 锁定移动方向
      this.move_lock = direction;
      // 如果移动方向为 "y"
      if (direction === "y") {
        // 给 body 元素添加类名 "body-console"
        $(document.body).addClass("body-console");
        // 创建控制台的 HTML 标签
        return this.console.createHtmltag();
      }
      // 创建 HTML 标签
      this.createHtmltag();
      // 给 body 元素添加类名 "body-sidebar"
      $(document.body).addClass("body-sidebar");
      // 给容器元素绑定鼠标按下、触摸结束和触摸取消事件的处理函数
      this.container.on("mousedown touchend touchcancel", (function(_this) {
        return function(e) {
          if (e.target !== e.currentTarget) {
            return true;
          }
          _this.log("closing");
          if ($(document.body).hasClass("body-sidebar")) {
            _this.close();
            return true;
          }
        };
      })(this));
      // 移除窗口大小改变事件的处理函数
      $(window).off("resize");
      // 绑定窗口大小改变事件的处理函数
      $(window).on("resize", (function(_this) {
        return function() {
          // 设置 body 元素的高度为窗口高度
          $(document.body).css("height", $(window).height());
          // 调用 scrollable 方法
          _this.scrollable();
          // 调用 resized 方法
          return _this.resized();
        };
      })(this));
      // 设置 wrapper 对象的 setSiteInfo 方法
      this.wrapper.setSiteInfo = (function(_this) {
        return function(site_info) {
          _this.setSiteInfo(site_info);
          return _this.original_set_site_info.apply(_this.wrapper, arguments);
        };
      })(this);
      // 创建一个新的 Image 对象
      img = new Image();
      // 设置图片的 src 属性
      return img.src = "/uimedia/globe/world.jpg";
    };

    // 定义 Sidebar 对象的 setSiteInfo 方法，用于设置站点信息
    Sidebar.prototype.setSiteInfo = function(site_info) {
      // 限制调用频率为 1500 毫秒，调用 updateHtmlTag 方法
      RateLimit(1500, (function(_this) {
        return function() {
          return _this.updateHtmlTag();
        };
      })(this));
      // 限制调用频率为 30000 毫秒，调用 displayGlobe 方法
      return RateLimit(30000, (function(_this) {
        return function() {
          return _this.displayGlobe();
        };
      })(this));
    };
    # 创建 HTML 标签的方法
    Sidebar.prototype.createHtmltag = function() {
      # 创建一个延迟对象，用于处理异步加载
      this.when_loaded = $.Deferred();
      # 如果容器不存在，则创建容器并添加到文档中
      if (!this.container) {
        this.container = $("<div class=\"sidebar-container\"><div class=\"sidebar scrollable\"><div class=\"content-wrapper\"><div class=\"content\">\n</div></div></div></div>");
        this.container.appendTo(document.body);
        # 查找并设置标签
        this.tag = this.container.find(".sidebar");
        # 更新 HTML 标签
        this.updateHtmlTag();
        # 返回滚动条对象
        return this.scrollable = window.initScrollable();
      }
    };

    # 更新 HTML 标签的方法
    Sidebar.prototype.updateHtmlTag = function() {
      # 如果预加载的 HTML 存在，则设置 HTML 标签并清空预加载的 HTML
      if (this.preload_html) {
        this.setHtmlTag(this.preload_html);
        return this.preload_html = null;
      } else {
        # 否则，通过 WebSocket 发送请求获取 HTML 标签
        return this.wrapper.ws.cmd("sidebarGetHtmlTag", {}, this.setHtmlTag);
      }
    };
    // 设置 HTML 标签的内容
    Sidebar.prototype.setHtmlTag = function(res) {
      // 如果标签内的内容为空
      if (this.tag.find(".content").children().length === 0) {
        // 输出日志信息
        this.log("Creating content");
        // 给容器添加 loaded 类
        this.container.addClass("loaded");
        // 使用 morphdom 替换标签内的内容为 res
        morphdom(this.tag.find(".content")[0], '<div class="content">' + res + '</div>');
        // 解决 when_loaded Promise
        this.when_loaded.resolve();
      } else {
        // 使用 morphdom 替换标签内的内容为 res，同时传入回调函数
        morphdom(this.tag.find(".content")[0], '<div class="content">' + res + '</div>', {
          // 在替换前执行的回调函数
          onBeforeMorphEl: function(from_el, to_el) {
            // 如果原始元素的类名为 "globe" 或包含 "noupdate"
            if (from_el.className === "globe" || from_el.className.indexOf("noupdate") >= 0) {
              return false;
            } else {
              return true;
            }
          }
        });
      }
      // 绑定点击和触摸结束事件，显示输入框，保存私钥
      this.tag.find("#privatekey-add").off("click, touchend").on("click touchend", (function(_this) {
        return function(e) {
          _this.wrapper.displayPrompt("Enter your private key:", "password", "Save", "", function(privatekey) {
            return _this.wrapper.ws.cmd("userSetSitePrivatekey", [privatekey], function(res) {
              return _this.wrapper.notifications.add("privatekey", "done", "Private key saved for site signing", 5000);
            });
          });
          return false;
        };
      })(this));
      // 绑定点击和触摸结束事件，显示确认框，移除保存的私钥
      this.tag.find("#privatekey-forget").off("click, touchend").on("click touchend", (function(_this) {
        return function(e) {
          _this.wrapper.displayConfirm("Remove saved private key for this site?", "Forget", function(res) {
            if (!res) {
              return false;
            }
            return _this.wrapper.ws.cmd("userSetSitePrivatekey", [""], function(res) {
              return _this.wrapper.notifications.add("privatekey", "done", "Saved private key removed", 5000);
            });
          });
          return false;
        };
      })(this));
      // 设置 href 属性为当前路径的列表页面路径
      return this.tag.find("#browse-files").attr("href", document.location.pathname.replace(/(\/.*?(\/|$)).*$/, "/list$1"));
    };
    // 定义 Sidebar 对象的 animDrag 方法，处理拖动事件
    Sidebar.prototype.animDrag = function(e) {
      // 获取鼠标当前位置的 x 和 y 坐标
      var mousex, mousey, overdrag, overdrag_percent, targetx, targety;
      mousex = e.pageX;
      mousey = e.pageY;
      // 如果鼠标位置不存在且存在触摸事件，则获取触摸位置的 x 和 y 坐标
      if (!mousex && e.originalEvent.touches) {
        mousex = e.originalEvent.touches[0].pageX;
        mousey = e.originalEvent.touches[0].pageY;
      }
      // 计算超出拖动范围的距离
      overdrag = this.fixbutton_initx - this.width - mousex;
      // 如果超出拖动范围，则计算超出的百分比，并重新计算鼠标位置
      if (overdrag > 0) {
        overdrag_percent = 1 + overdrag / 300;
        mousex = (mousex + (this.fixbutton_initx - this.width) * overdrag_percent) / (1 + overdrag_percent);
      }
      // 计算目标 x 和 y 坐标
      targetx = this.fixbutton_initx - mousex - this.fixbutton_addx;
      targety = this.fixbutton_inity - mousey - this.fixbutton_addy;
      // 如果移动锁定在 x 轴上，则固定 y 坐标
      if (this.move_lock === "x") {
        targety = this.fixbutton_inity;
      } else if (this.move_lock === "y") {
        // 如果移动锁定在 y 轴上，则固定 x 坐标
        targetx = this.fixbutton_initx;
      }
      // 如果没有移动锁定或者移动锁定在 x 轴上，则设置固定按钮的 left 样式和标签的 transform 样式
      if (!this.move_lock || this.move_lock === "x") {
        this.fixbutton[0].style.left = (mousex + this.fixbutton_addx) + "px";
        if (this.tag) {
          this.tag[0].style.transform = "translateX(" + (0 - targetx) + "px)";
        }
      }
      // 如果没有移动锁定或者移动锁定在 y 轴上，则设置固定按钮的 top 样式和控制台标签的 transform 样式
      if (!this.move_lock || this.move_lock === "y") {
        this.fixbutton[0].style.top = (mousey + this.fixbutton_addy) + "px";
        if (this.console.tag) {
          this.console.tag[0].style.transform = "translateY(" + (0 - targety) + "px)";
        }
      }
      // 根据条件判断是否打开或关闭固定按钮
      if ((!this.opened && targetx > this.width / 3) || (this.opened && targetx > this.width * 0.9)) {
        this.fixbutton_targetx = this.fixbutton_initx - this.width;
      } else {
        this.fixbutton_targetx = this.fixbutton_initx;
      }
      // 根据条件判断是否打开或关闭控制台
      if ((!this.console.opened && 0 - targety > this.page_height / 10) || (this.console.opened && 0 - targety > this.page_height * 0.8)) {
        this.fixbutton_targety = this.page_height - this.fixbutton_inity - 50;
      } else {
        this.fixbutton_targety = this.fixbutton_inity;
      }
    };
    # 停止拖拽操作的方法
    Sidebar.prototype.stopDrag = function() {
      var left, top;
      # 解除父元素的鼠标移动事件和触摸移动事件的绑定
      this.fixbutton.parents().off("mousemove touchmove");
      # 解除当前元素的鼠标移动事件和触摸移动事件的绑定
      this.fixbutton.off("mousemove touchmove");
      # 恢复当前元素的鼠标事件
      this.fixbutton.css("pointer-events", "");
      # 移除拖拽背景
      $(".drag-bg").remove();
      # 如果当前元素没有拖拽状态，则返回
      if (!this.fixbutton.hasClass("dragging")) {
        return;
      }
      # 移除拖拽状态
      this.fixbutton.removeClass("dragging");
      # 如果目标位置和当前位置不一致
      if (this.fixbutton_targetx !== this.fixbutton.offset().left || this.fixbutton_targety !== this.fixbutton.offset().top) {
        # 根据移动锁定方向设置 top 和 left 的值
        if (this.move_lock === "y") {
          top = this.fixbutton_targety;
          left = this.fixbutton_initx;
        }
        if (this.move_lock === "x") {
          top = this.fixbutton_inity;
          left = this.fixbutton_targetx;
        }
        # 使用动画效果移动元素到目标位置
        this.fixbutton.stop().animate({
          "left": left,
          "top": top
        }, 500, "easeOutBack", (function(_this) {
          return function() {
            # 如果目标位置和初始位置一致，则重置 left 属性
            if (_this.fixbutton_targetx === _this.fixbutton_initx) {
              _this.fixbutton.css("left", "auto");
            } else {
              _this.fixbutton.css("left", left);
            }
            # 触发鼠标移出事件
            return $(".fixbutton-bg").trigger("mouseout");
          };
        })(this));
        # 停止 X 轴方向的拖拽
        this.stopDragX();
        # 停止控制台 Y 轴方向的拖拽
        this.console.stopDragY();
      }
      # 重置移动锁定状态
      return this.move_lock = null;
    };
    // 停止侧边栏在水平方向上的拖动
    Sidebar.prototype.stopDragX = function() {
      var targetx;
      // 如果按钮目标位置等于初始位置或者移动锁定在垂直方向上
      if (this.fixbutton_targetx === this.fixbutton_initx || this.move_lock === "y") {
        // 目标位置为0
        targetx = 0;
        // 侧边栏关闭状态
        this.opened = false;
      } else {
        // 目标位置为侧边栏的宽度
        targetx = this.width;
        // 如果侧边栏已经打开
        if (this.opened) {
          // 调用onOpened方法
          this.onOpened();
        } else {
          // 当加载完成后调用onOpened方法
          this.when_loaded.done((function(_this) {
            return function() {
              return _this.onOpened();
            };
          })(this));
        }
        // 侧边栏打开状态
        this.opened = true;
      }
      // 如果存在标签
      if (this.tag) {
        // 设置过渡效果
        this.tag.css("transition", "0.4s ease-out");
        // 设置transform属性，实现水平平移
        this.tag.css("transform", "translateX(-" + targetx + "px)").one(transitionEnd, (function(_this) {
          return function() {
            _this.tag.css("transition", "");
            // 如果侧边栏关闭
            if (!_this.opened) {
              _this.container.remove();
              _this.container = null;
              if (_this.tag) {
                _this.tag.remove();
                return _this.tag = null;
              }
            }
          };
        })(this));
      }
      // 输出日志
      this.log("stopdrag", "opened:", this.opened);
      // 如果侧边栏关闭
      if (!this.opened) {
        // 调用onClosed方法
        return this.onClosed();
      }
    };

    // 对指定路径进行签名
    Sidebar.prototype.sign = function(inner_path, privatekey) {
      // 显示签名进度
      this.wrapper.displayProgress("sign", "Signing: " + inner_path + "...", 0);
      // 调用网站签名命令
      return this.wrapper.ws.cmd("siteSign", {
        privatekey: privatekey,
        inner_path: inner_path,
        update_changed_files: true
      }, (function(_this) {
        return function(res) {
          // 如果签名成功
          if (res === "ok") {
            // 显示签名成功信息
            return _this.wrapper.displayProgress("sign", inner_path + " signed!", 100);
          } else {
            // 显示签名失败信息
            return _this.wrapper.displayProgress("sign", "Error signing " + inner_path, -1);
          }
        };
      })(this));
    };
    # 定义一个名为 publish 的方法，用于发布站点
    Sidebar.prototype.publish = function(inner_path, privatekey) {
      # 调用 wrapper 对象的 ws 属性的 cmd 方法，执行站点发布操作
      return this.wrapper.ws.cmd("sitePublish", {
        privatekey: privatekey,
        inner_path: inner_path,
        sign: true,
        update_changed_files: true
      }, (function(_this) {
        # 返回一个函数，用于处理发布结果
        return function(res) {
          # 如果发布成功，显示通知消息
          if (res === "ok") {
            return _this.wrapper.notifications.add("sign", "done", inner_path + " Signed and published!", 5000);
          }
        };
      })(this));
    };

    # 定义一个名为 handleSiteDeleteClick 的方法，用于处理站点删除点击事件
    Sidebar.prototype.handleSiteDeleteClick = function() {
      var options, question;
      # 如果站点信息中包含私钥，显示确认提示
      if (this.wrapper.site_info.privatekey) {
        question = "Are you sure?<br>This site has a saved private key";
        options = ["Forget private key and delete site"];
      } else {
        question = "Are you sure?";
        options = ["Delete this site", "Blacklist"];
      }
      # 显示确认对话框，根据用户选择执行相应操作
      return this.wrapper.displayConfirm(question, options, (function(_this) {
        return function(confirmed) {
          if (confirmed === 1) {
            # 如果确认删除站点，执行站点删除操作
            _this.tag.find("#button-delete").addClass("loading");
            return _this.wrapper.ws.cmd("siteDelete", _this.wrapper.site_info.address, function() {
              return document.location = $(".fixbutton-bg").attr("href");
            });
          } else if (confirmed === 2) {
            # 如果确认加入黑名单，执行站点加入黑名单和删除操作
            return _this.wrapper.displayPrompt("Blacklist this site", "text", "Delete and Blacklist", "Reason", function(reason) {
              _this.tag.find("#button-delete").addClass("loading");
              _this.wrapper.ws.cmd("siteblockAdd", [_this.wrapper.site_info.address, reason]);
              return _this.wrapper.ws.cmd("siteDelete", _this.wrapper.site_info.address, function() {
                return document.location = $(".fixbutton-bg").attr("href");
              });
            });
          }
        };
      })(this));
    };

    # 定义一个名为 close 的方法，用于关闭侧边栏
    Sidebar.prototype.close = function() {
      # 设置移动锁定为水平方向
      this.move_lock = "x";
      # 调用 startDrag 方法
      this.startDrag();
      # 调用 stopDrag 方法
      return this.stopDrag();
    // 定义 Sidebar 对象的 onClosed 方法
    Sidebar.prototype.onClosed = function() {
      // 移除窗口的 resize 事件监听器
      $(window).off("resize");
      // 添加窗口的 resize 事件监听器，并指定回调函数为 this.resized
      $(window).on("resize", this.resized);
      // 设置文档 body 的过渡效果和样式，并在过渡结束后执行回调函数
      $(document.body).css("transition", "0.6s ease-in-out").removeClass("body-sidebar").on(transitionEnd, (function(_this) {
        return function(e) {
          // 检查过渡结束后的条件，执行相应操作
          if (e.target === document.body && !$(document.body).hasClass("body-sidebar") && !$(document.body).hasClass("body-console")) {
            $(document.body).css("height", "auto").css("perspective", "").css("will-change", "").css("transition", "").off(transitionEnd);
            // 卸载 Globe
            return _this.unloadGlobe();
          }
        };
      })(this));
      // 设置 wrapper 的 setSiteInfo 方法为原始的 set_site_info 方法
      return this.wrapper.setSiteInfo = this.original_set_site_info;
    };

    // 定义 Sidebar 对象的 loadGlobe 方法
    Sidebar.prototype.loadGlobe = function() {
      // 如果 globe 元素有 loading 类，则延迟执行
      if (this.tag.find(".globe").hasClass("loading")) {
        return setTimeout(((function(_this) {
          return function() {
            // 如果 DAT 未定义，则动态创建并加载 all.js 脚本
            var script_tag;
            if (typeof DAT === "undefined") {
              script_tag = $("<script>");
              script_tag.attr("nonce", _this.wrapper.script_nonce);
              script_tag.attr("src", "/uimedia/globe/all.js");
              script_tag.on("load", _this.displayGlobe);
              return document.head.appendChild(script_tag[0]);
            } else {
              // 否则直接显示 Globe
              return _this.displayGlobe();
            }
          };
        })(this)), 600);
      }
    };
    # 定义 Sidebar 对象的 displayGlobe 方法
    Sidebar.prototype.displayGlobe = function() {
      # 创建一个新的图片对象
      var img;
      img = new Image();
      # 设置图片对象的源地址
      img.src = "/uimedia/globe/world.jpg";
      # 当图片加载完成时执行回调函数
      return img.onload = (function(_this) {
        return function() {
          # 通过 WebSocket 发送请求获取地球数据
          return _this.wrapper.ws.cmd("sidebarGetPeers", [], function(globe_data) {
            # 如果已经存在地球对象，则移除之前的点
            if (_this.globe) {
              _this.globe.scene.remove(_this.globe.points);
              # 添加新的地球数据并创建点
              _this.globe.addData(globe_data, {
                format: 'magnitude',
                name: "hello",
                animated: false
              });
              _this.globe.createPoints();
              # 移除加载状态
              return (ref = _this.tag) != null ? ref.find(".globe").removeClass("loading") : void 0;
            } else if (typeof DAT !== "undefined") {
              try {
                # 如果不存在地球对象但支持 WebGL，则创建新的地球对象
                _this.globe = new DAT.Globe(_this.tag.find(".globe")[0], {
                  "imgDir": "/uimedia/globe/"
                });
                # 添加新的地球数据并创建点
                _this.globe.addData(globe_data, {
                  format: 'magnitude',
                  name: "hello"
                });
                _this.globe.createPoints();
                # 开始动画
                _this.globe.animate();
              } catch (error) {
                e = error;
                console.log("WebGL error", e);
                # 如果出现 WebGL 错误，则显示错误信息
                if ((ref1 = _this.tag) != null) {
                  ref1.find(".globe").addClass("error").text("WebGL not supported");
                }
              }
              # 移除加载状态
              return (ref2 = _this.tag) != null ? ref2.find(".globe").removeClass("loading") : void 0;
            }
          });
        };
      })(this);
    };

    # 定义 Sidebar 对象的 unloadGlobe 方法
    Sidebar.prototype.unloadGlobe = function() {
      # 如果地球对象不存在，则返回 false
      if (!this.globe) {
        return false;
      }
      # 卸载地球对象并将其置为 null
      this.globe.unload();
      return this.globe = null;
    };

    # 返回 Sidebar 对象
    return Sidebar;

  })(Class);

  # 获取全局变量 window.wrapper
  wrapper = window.wrapper;

  # 延迟执行
  setTimeout((function() {
    # 返回一个新的侧边栏对象，并将其赋值给 window.sidebar
    return window.sidebar = new Sidebar(wrapper);
  }), 500);
  
  # 设置窗口过渡结束事件的名称，用于在不同浏览器中监听过渡结束事件
  window.transitionEnd = 'transitionend webkitTransitionEnd oTransitionEnd otransitionend';
// 将该函数作为参数传递给一个立即执行的函数，以便在不同的环境中使用
(function(f){
    // 如果是在 Node.js 环境中，则将模块导出
    if(typeof exports==="object"&&typeof module!=="undefined"){
        module.exports=f()
    }
    // 如果是在 AMD 环境中，则使用 define 来定义模块
    else if(typeof define==="function"&&define.amd){
        define([],f)
    }
    // 否则将函数绑定到全局对象上
    else{
        var g;
        if(typeof window!=="undefined"){
            g=window
        }else if(typeof global!=="undefined"){
            g=global
        }else if(typeof self!=="undefined"){
            g=self
        }else{
            g=this
        }
        g.morphdom = f()
    }
})(function(){
    // 定义一些变量和函数
    var define,module,exports;
    return (function e(t,n,r){
        function s(o,u){
            if(!n[o]){
                if(!t[o]){
                    var a=typeof require=="function"&&require;
                    if(!u&&a)return a(o,!0);
                    if(i)return i(o,!0);
                    var f=new Error("Cannot find module '"+o+"'");
                    throw f.code="MODULE_NOT_FOUND",f
                }
                var l=n[o]={exports:{}};
                t[o][0].call(l.exports,function(e){
                    var n=t[o][1][e];
                    return s(n?n:e)
                },l,l.exports,e,t,n,r)
            }
            return n[o].exports
        }
        var i=typeof require=="function"&&require;
        for(var o=0;o<r.length;o++)s(r[o]);
        return s
    })({
        1:[function(require,module,exports){
            // 定义一个特殊元素处理器对象
            var specialElHandlers = {
                // 处理 <option> 元素
                OPTION: function(fromEl, toEl) {
                    if ((fromEl.selected = toEl.selected)) {
                        fromEl.setAttribute('selected', '');
                    } else {
                        fromEl.removeAttribute('selected', '');
                    }
                },
                // 处理 <input> 元素
                /*INPUT: function(fromEl, toEl) {
                    fromEl.checked = toEl.checked;
                    fromEl.value = toEl.value;

                    if (!toEl.hasAttribute('checked')) {
                        fromEl.removeAttribute('checked');
                    }

                    if (!toEl.hasAttribute('value')) {
                        fromEl.removeAttribute('value');
                    }
                }*/
            };

            // 定义一个空函数
            function noop() {}
/**
 * 遍历目标节点上的所有属性，并确保原始 DOM 节点具有相同的属性。
 * 如果在原始节点上找到的属性在新节点上不存在，则从原始节点中删除它
 * @param  {HTMLElement} fromNode
 * @param  {HTMLElement} toNode
 */
function morphAttrs(fromNode, toNode) {
    // 获取目标节点的所有属性
    var attrs = toNode.attributes;
    var i;
    var attr;
    var attrName;
    var attrValue;
    var foundAttrs = {};

    // 遍历目标节点的属性
    for (i=attrs.length-1; i>=0; i--) {
        attr = attrs[i];
        if (attr.specified !== false) {
            attrName = attr.name;
            attrValue = attr.value;
            foundAttrs[attrName] = true;

            // 如果原始节点上的属性值与目标节点不同，则设置原始节点上的属性值为目标节点的属性值
            if (fromNode.getAttribute(attrName) !== attrValue) {
                fromNode.setAttribute(attrName, attrValue);
            }
        }
    }

    // 删除原始 DOM 元素上找到的任何额外属性，这些属性在目标元素上找不到
    attrs = fromNode.attributes;

    for (i=attrs.length-1; i>=0; i--) {
        attr = attrs[i];
        if (attr.specified !== false) {
            attrName = attr.name;
            if (!foundAttrs.hasOwnProperty(attrName)) {
                fromNode.removeAttribute(attrName);
            }
        }
    }
}

/**
 * 将一个 DOM 元素的子元素复制到另一个 DOM 元素
 */
function moveChildren(from, to) {
    var curChild = from.firstChild;
    while(curChild) {
        var nextChild = curChild.nextSibling;
        to.appendChild(curChild);
        curChild = nextChild;
    }
    return to;
}

function morphdom(fromNode, toNode, options) {
    if (!options) {
        options = {};
    }

    if (typeof toNode === 'string') {
        var newBodyEl = document.createElement('body');
        newBodyEl.innerHTML = toNode;
        toNode = newBodyEl.childNodes[0];
    }

    var savedEls = {}; // 用于保存具有 ID 的 DOM 元素
    var unmatchedEls = {};
    var onNodeDiscarded = options.onNodeDiscarded || noop;
    # 如果存在 options.onBeforeMorphEl，则使用它，否则使用空函数 noop
    var onBeforeMorphEl = options.onBeforeMorphEl || noop;
    # 如果存在 options.onBeforeMorphElChildren，则使用它，否则使用空函数 noop
    var onBeforeMorphElChildren = options.onBeforeMorphElChildren || noop;

    # 辅助函数，用于移除节点
    function removeNodeHelper(node, nestedInSavedEl) {
        # 获取节点的 ID
        var id = node.id;
        # 如果节点有 ID，则将其保存在 savedEls 中，以便在目标 DOM 树中重用
        if (id) {
            savedEls[id] = node;
        } else if (!nestedInSavedEl) {
            # 如果不是嵌套在已保存的元素中，则表示该节点已被完全丢弃，不会存在于最终的 DOM 中
            onNodeDiscarded(node);
        }

        # 如果节点类型为元素节点
        if (node.nodeType === 1) {
            # 遍历子节点
            var curChild = node.firstChild;
            while(curChild) {
                # 递归调用 removeNodeHelper 函数
                removeNodeHelper(curChild, nestedInSavedEl || id);
                curChild = curChild.nextSibling;
            }
        }
    }

    # 遍历丢弃的子节点
    function walkDiscardedChildNodes(node) {
        # 如果节点类型为元素节点
        if (node.nodeType === 1) {
            # 遍历子节点
            var curChild = node.firstChild;
            while(curChild) {
                # 如果子节点没有 ID，则处理丢弃的节点
                if (!curChild.id) {
                    onNodeDiscarded(curChild);
                    # 递归调用 walkDiscardedChildNodes 函数
                    walkDiscardedChildNodes(curChild);
                }
                curChild = curChild.nextSibling;
            }
        }
    }

    # 移除节点
    function removeNode(node, parentNode, alreadyVisited) {
        # 从父节点中移除节点
        parentNode.removeChild(node);

        # 如果已经访问过该节点
        if (alreadyVisited) {
            # 如果节点没有 ID，则处理丢弃的节点，并递归调用 walkDiscardedChildNodes 函数
            if (!node.id) {
                onNodeDiscarded(node);
                walkDiscardedChildNodes(node);
            }
        } else {
            # 否则调用 removeNodeHelper 函数
            removeNodeHelper(node);
        }
    }

    # 初始化变量 morphedNode 和 morphedNodeType
    var morphedNode = fromNode;
    var morphedNodeType = morphedNode.nodeType;
    var toNodeType = toNode.nodeType;
    // 处理给定的两个 DOM 节点不兼容的情况（例如 <div> --> <span> 或 <div> --> 文本）
    if (morphedNodeType === 1) { // 如果原节点是元素节点
        if (toNodeType === 1) { // 如果目标节点是元素节点
            if (morphedNode.tagName !== toNode.tagName) { // 如果原节点和目标节点的标签名不同
                onNodeDiscarded(fromNode); // 触发丢弃节点事件
                morphedNode = moveChildren(morphedNode, document.createElement(toNode.tagName)); // 将原节点的子节点移动到新创建的目标节点中
            }
        } else { // 如果目标节点是文本节点
            // 从元素节点转换为文本节点
            return toNode; // 返回目标文本节点
        }
    } else if (morphedNodeType === 3) { // 如果原节点是文本节点
        if (toNodeType === 3) { // 如果目标节点是文本节点
            morphedNode.nodeValue = toNode.nodeValue; // 更新原节点的文本值为目标节点的文本值
            return morphedNode; // 返回更新后的原文本节点
        } else { // 如果目标节点不是文本节点
            onNodeDiscarded(fromNode); // 触发丢弃节点事件
            // 文本节点转换为其他类型节点
            return toNode; // 返回目标节点
        }
    }

    morphEl(morphedNode, toNode, false); // 对原节点和目标节点进行进一步的 DOM 变换

    // 对于任何未找到新位置的保存元素，触发“onNodeDiscarded”事件
    for (var savedElId in savedEls) {
        if (savedEls.hasOwnProperty(savedElId)) {
            var savedEl = savedEls[savedElId];
            onNodeDiscarded(savedEl); // 触发丢弃节点事件
            walkDiscardedChildNodes(savedEl); // 遍历丢弃节点的子节点
        }
    }

    if (morphedNode !== fromNode && fromNode.parentNode) { // 如果原节点和目标节点不相同且原节点有父节点
        fromNode.parentNode.replaceChild(morphedNode, fromNode); // 用目标节点替换原节点
    }

    return morphedNode; // 返回变换后的原节点
# 导出模块中的 morphdom 函数
module.exports = morphdom;
# 结束模块定义
},{}]},{},[1])(1)
# 执行模块，传入参数 1
});
```