# `ZeroNet\plugins\UiFileManager\media\js\all.js`

```py
/* ---- lib/Animation.coffee ---- */

// 创建一个名为 Animation 的类
(function() {
  var Animation;

  Animation = (function() {
    function Animation() {}

    // 定义 slideDown 方法，用于元素的下滑动画
    Animation.prototype.slideDown = function(elem, props) {
      // 获取元素的当前高度和样式
      var cstyle, h, margin_bottom, margin_top, padding_bottom, padding_top, transition;
      // 如果元素的 offsetTop 大于 2000，则直接返回，不执行下滑动画
      if (elem.offsetTop > 2000) {
        return;
      }
      h = elem.offsetHeight;
      cstyle = window.getComputedStyle(elem);
      margin_top = cstyle.marginTop;
      margin_bottom = cstyle.marginBottom;
      padding_top = cstyle.paddingTop;
      padding_bottom = cstyle.paddingBottom;
      transition = cstyle.transition;
      // 设置元素的样式和属性，准备执行动画
      elem.style.boxSizing = "border-box";
      elem.style.overflow = "hidden";
      elem.style.transform = "scale(0.6)";
      elem.style.opacity = "0";
      elem.style.height = "0px";
      elem.style.marginTop = "0px";
      elem.style.marginBottom = "0px";
      elem.style.paddingTop = "0px";
      elem.style.paddingBottom = "0px";
      elem.style.transition = "none";
      // 延迟执行动画效果，避免立即执行导致无效
      setTimeout((function() {
        elem.className += " animate-inout";
        elem.style.height = h + "px";
        elem.style.transform = "scale(1)";
        elem.style.opacity = "1";
        elem.style.marginTop = margin_top;
        elem.style.marginBottom = margin_bottom;
        elem.style.paddingTop = padding_top;
        return elem.style.paddingBottom = padding_bottom;
      }), 1);
      // 监听动画结束事件，清除样式和属性，移除事件监听
      return elem.addEventListener("transitionend", function() {
        elem.classList.remove("animate-inout");
        elem.style.transition = elem.style.transform = elem.style.opacity = elem.style.height = null;
        elem.style.boxSizing = elem.style.marginTop = elem.style.marginBottom = null;
        elem.style.paddingTop = elem.style.paddingBottom = elem.style.overflow = null;
        return elem.removeEventListener("transitionend", arguments.callee, false);
      });
    };
    // 定义 Animation 对象的 slideUp 方法，用于向上滑动元素
    Animation.prototype.slideUp = function(elem, remove_func, props) {
      // 如果元素的上边距大于 1000，则执行 remove_func 函数并返回
      if (elem.offsetTop > 1000) {
        return remove_func();
      }
      // 为元素添加类名 "animate-back"
      elem.className += " animate-back";
      // 设置元素的盒模型为 border-box
      elem.style.boxSizing = "border-box";
      // 设置元素的高度为当前高度
      elem.style.height = elem.offsetHeight + "px";
      // 设置元素的 overflow 为 hidden
      elem.style.overflow = "hidden";
      // 设置元素的缩放为 1
      elem.style.transform = "scale(1)";
      // 设置元素的透明度为 1
      elem.style.opacity = "1";
      // 设置元素的指针事件为 none
      elem.style.pointerEvents = "none";
      // 延迟执行以下操作
      setTimeout((function() {
        // 设置元素的高度为 0
        elem.style.height = "0px";
        // 设置元素的上外边距为 0
        elem.style.marginTop = "0px";
        // 设置元素的下外边距为 0
        elem.style.marginBottom = "0px";
        // 设置元素的上内边距为 0
        elem.style.paddingTop = "0px";
        // 设置元素的下内边距为 0
        elem.style.paddingBottom = "0px";
        // 设置元素的缩放为 0.8
        elem.style.transform = "scale(0.8)";
        // 设置元素的上边框宽度为 0
        elem.style.borderTopWidth = "0px";
        // 设置元素的下边框宽度为 0
        elem.style.borderBottomWidth = "0px";
        // 设置元素的透明度为 0
        return elem.style.opacity = "0";
      }), 1);
      // 监听过渡结束事件
      return elem.addEventListener("transitionend", function(e) {
        // 如果属性名为 "opacity" 或经过时间大于等于 0.6，则执行以下操作
        if (e.propertyName === "opacity" || e.elapsedTime >= 0.6) {
          // 移除过渡结束事件监听
          elem.removeEventListener("transitionend", arguments.callee, false);
          // 执行 remove_func 函数
          return remove_func();
        }
      });
    };
    // 定义一个名为slideUpInout的方法，用于元素的滑动淡入淡出效果
    Animation.prototype.slideUpInout = function(elem, remove_func, props) {
      // 为元素添加CSS类名，实现动画效果
      elem.className += " animate-inout";
      // 设置元素的盒模型为border-box
      elem.style.boxSizing = "border-box";
      // 设置元素的高度为当前高度
      elem.style.height = elem.offsetHeight + "px";
      // 设置元素的overflow属性为hidden，隐藏溢出内容
      elem.style.overflow = "hidden";
      // 设置元素的缩放比例为1
      elem.style.transform = "scale(1)";
      // 设置元素的不透明度为1
      elem.style.opacity = "1";
      // 设置元素的指针事件为none，禁用鼠标事件
      elem.style.pointerEvents = "none";
      // 延迟1毫秒后执行以下操作
      setTimeout((function() {
        // 设置元素的高度为0
        elem.style.height = "0px";
        // 设置元素的上外边距为0
        elem.style.marginTop = "0px";
        // 设置元素的下外边距为0
        elem.style.marginBottom = "0px";
        // 设置元素的上内边距为0
        elem.style.paddingTop = "0px";
        // 设置元素的下内边距为0
        elem.style.paddingBottom = "0px";
        // 设置元素的缩放比例为0.8
        elem.style.transform = "scale(0.8)";
        // 设置元素的上边框宽度为0
        elem.style.borderTopWidth = "0px";
        // 设置元素的下边框宽度为0
        elem.style.borderBottomWidth = "0px";
        // 设置元素的不透明度为0
        return elem.style.opacity = "0";
      }), 1);
      // 监听过渡结束事件，执行回调函数
      return elem.addEventListener("transitionend", function(e) {
        // 如果属性名为opacity或者过渡时间超过0.6秒
        if (e.propertyName === "opacity" || e.elapsedTime >= 0.6) {
          // 移除过渡结束事件的监听器
          elem.removeEventListener("transitionend", arguments.callee, false);
          // 执行移除函数
          return remove_func();
        }
      });
    };

    // 定义一个名为showRight的方法，用于元素的右侧显示效果
    Animation.prototype.showRight = function(elem, props) {
      // 为元素添加CSS类名，实现动画效果
      elem.className += " animate";
      // 设置元素的不透明度为0
      elem.style.opacity = 0;
      // 设置元素的缩放和平移效果
      elem.style.transform = "TranslateX(-20px) Scale(1.01)";
      // 延迟1毫秒后执行以下操作
      setTimeout((function() {
        // 设置元素的不透明度为1
        elem.style.opacity = 1;
        // 设置元素的缩放和平移效果
        return elem.style.transform = "TranslateX(0px) Scale(1)";
      }), 1);
      // 监听过渡结束事件，执行回调函数
      return elem.addEventListener("transitionend", function() {
        // 移除CSS类名
        elem.classList.remove("animate");
        // 清空元素的变换和不透明度属性
        return elem.style.transform = elem.style.opacity = null;
      });
    };
    # 定义动画对象的显示方法，接受元素和属性作为参数
    Animation.prototype.show = function(elem, props) {
      # 声明延迟变量，根据参数中的延迟值或默认值设置延迟时间
      var delay, ref;
      delay = ((ref = arguments[arguments.length - 2]) != null ? ref.delay : void 0) * 1000 || 1;
      # 设置元素的不透明度为0
      elem.style.opacity = 0;
      # 在1毫秒后给元素添加"animate"类
      setTimeout((function() {
        return elem.className += " animate";
      }), 1);
      # 在延迟时间后将元素的不透明度设置为1
      setTimeout((function() {
        return elem.style.opacity = 1;
      }), delay);
      # 添加过渡结束事件监听器，当过渡结束时执行相应操作
      return elem.addEventListener("transitionend", function() {
        # 移除"animate"类，重置不透明度
        elem.classList.remove("animate");
        elem.style.opacity = null;
        # 移除过渡结束事件监听器
        return elem.removeEventListener("transitionend", arguments.callee, false);
      });
    };

    # 定义动画对象的隐藏方法，接受元素、移除函数和属性作为参数
    Animation.prototype.hide = function(elem, remove_func, props) {
      # 声明延迟变量，根据参数中的延迟值或默认值设置延迟时间
      var delay, ref;
      delay = ((ref = arguments[arguments.length - 2]) != null ? ref.delay : void 0) * 1000 || 1;
      # 给元素添加"animate"类
      elem.className += " animate";
      # 在延迟时间后将元素的不透明度设置为0
      setTimeout((function() {
        return elem.style.opacity = 0;
      }), delay);
      # 添加过渡结束事件监听器，当过渡结束时执行相应操作
      return elem.addEventListener("transitionend", function(e) {
        # 如果属性名为"opacity"，执行移除函数
        if (e.propertyName === "opacity") {
          return remove_func();
        }
      });
    };

    # 定义动画对象的添加可见类方法，接受元素和属性作为参数
    Animation.prototype.addVisibleClass = function(elem, props) {
      # 在下一个宏任务中给元素添加"visible"类
      return setTimeout(function() {
        return elem.classList.add("visible");
      });
    };

    # 导出动画对象
    return Animation;

  })();
  
  # 将动画对象绑定到全局对象window上
  window.Animation = new Animation();
# 将当前作用域设置为全局作用域
}).call(this);

# ---- lib/Class.coffee ----

# 定义一个匿名函数，该函数用于创建类
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
      # 在日志信息前添加类名，并打印日志
      args.unshift("[" + this.constructor.name + "]");
      console.log.apply(console, args);
      return this;
    };

    # 定义 logStart 方法，用于记录开始时间并打印日志
    Class.prototype.logStart = function() {
      var args, name;
      name = arguments[0], args = 2 <= arguments.length ? slice.call(arguments, 1) : [];
      # 如果 trace 为 false，则直接返回
      if (!this.trace) {
        return;
      }
      # 如果 logtimers 未定义，则创建一个空对象
      this.logtimers || (this.logtimers = {});
      # 记录开始时间
      this.logtimers[name] = +(new Date);
      # 如果有参数，则打印日志
      if (args.length > 0) {
        this.log.apply(this, ["" + name].concat(slice.call(args), ["(started)"]));
      }
      return this;
    };

    # 定义 logEnd 方法，用于记录结束时间并打印日志
    Class.prototype.logEnd = function() {
      var args, ms, name;
      name = arguments[0], args = 2 <= arguments.length ? slice.call(arguments, 1) : [];
      # 计算时间差
      ms = +(new Date) - this.logtimers[name];
      # 打印日志
      this.log.apply(this, ["" + name].concat(slice.call(args), ["(Done in " + ms + "ms)"]));
      return this;
    };

    return Class;

  })();

  # 将 Class 类绑定到全局对象 window 上
  window.Class = Class;

}).call(this);

# ---- lib/Dollar.coffee ----

# 定义一个匿名函数，该函数用于创建 $ 函数
(function() {
  # 将 $ 函数绑定到全局对象 window 上
  window.$ = function(selector) {
    # 如果选择器以 "#" 开头，则返回对应 id 的元素
    if (selector.startsWith("#")) {
      return document.getElementById(selector.replace("#", ""));
    }
  };

}).call(this);

# ---- lib/ItemList.coffee ----

# 定义 ItemList 类
(function() {
  var ItemList;

  ItemList = (function() {
    # 初始化 ItemList 类
    function ItemList(item_class1, key1) {
      this.item_class = item_class1;
      this.key = key1;
      this.items = [];
      this.items_bykey = {};
    }
    # 定义 ItemList 对象的同步方法，用于将传入的行数据同步到 ItemList 中
    ItemList.prototype.sync = function(rows, item_class, key) {
      var current_obj, i, item, len, results, row;
      # 清空当前 ItemList 中的所有项
      this.items.splice(0, this.items.length);
      results = [];
      # 遍历传入的行数据
      for (i = 0, len = rows.length; i < len; i++) {
        row = rows[i];
        # 获取当前行数据对应的 Item 对象
        current_obj = this.items_bykey[row[this.key]];
        # 如果存在对应的 Item 对象，则更新其行数据
        if (current_obj) {
          current_obj.row = row;
          results.push(this.items.push(current_obj));
        } else {
          # 如果不存在对应的 Item 对象，则创建一个新的 Item 对象，并添加到 ItemList 中
          item = new this.item_class(row, this);
          this.items_bykey[row[this.key]] = item;
          results.push(this.items.push(item));
        }
      }
      return results;
    };

    # 定义 ItemList 对象的删除方法，用于删除指定的 Item 对象
    ItemList.prototype.deleteItem = function(item) {
      var index;
      # 获取要删除的 Item 对象在 ItemList 中的索引
      index = this.items.indexOf(item);
      # 如果索引大于 -1，则从 ItemList 中删除该项
      if (index > -1) {
        this.items.splice(index, 1);
      } else {
        # 否则输出错误信息
        console.log("Can't delete item", item);
      }
      # 从 items_bykey 中删除对应的键值对
      return delete this.items_bykey[item.row[this.key]];
    };

    # 导出 ItemList 对象
    return ItemList;

  })();
  
  # 将 ItemList 对象绑定到 window 对象上
  window.ItemList = ItemList;
# 调用一个匿名函数，并将 this 绑定到当前上下文
}).call(this);

# 定义 Menu 类
# 包含一系列方法和属性
Menu = (function() {
  # 构造函数，初始化属性和绑定方法
  function Menu() {
    this.render = bind(this.render, this);
    this.getStyle = bind(this.getStyle, this);
    this.renderItem = bind(this.renderItem, this);
    this.handleClick = bind(this.handleClick, this);
    this.getDirection = bind(this.getDirection, this);
    this.storeNode = bind(this.storeNode, this);
    this.toggle = bind(this.toggle, this);
    this.hide = bind(this.hide, this);
    this.show = bind(this.show, this);
    # 初始化属性
    this.visible = false;
    this.items = [];
    this.node = null;
    this.height = 0;
    this.direction = "bottom";
  }

  # 显示菜单
  Menu.prototype.show = function() {
    # 如果已经有其他菜单显示，则隐藏
    var ref;
    if ((ref = window.visible_menu) != null) {
      ref.hide();
    }
    this.visible = true;
    window.visible_menu = this;
    return this.direction = this.getDirection();
  };

  # 隐藏菜单
  Menu.prototype.hide = function() {
    return this.visible = false;
  };

  # 切换菜单的显示状态
  Menu.prototype.toggle = function() {
    if (this.visible) {
      this.hide();
    } else {
      this.show();
    }
    return Page.projector.scheduleRender();
  };

  # 添加菜单项
  Menu.prototype.addItem = function(title, cb, selected) {
    if (selected == null) {
      selected = false;
    }
    return this.items.push([title, cb, selected]);
  };
    // 将节点存储到菜单对象中
    Menu.prototype.storeNode = function(node) {
      this.node = node;
      // 如果菜单可见
      if (this.visible) {
        // 移除节点的可见类名
        node.className = node.className.replace("visible", "");
        // 延迟20毫秒后添加可见类名，并设置样式
        setTimeout(((function(_this) {
          return function() {
            node.className += " visible";
            return node.attributes.style.value = _this.getStyle();
          };
        })(this)), 20);
        // 设置节点的最大高度为none，并记录节点的高度
        node.style.maxHeight = "none";
        this.height = node.offsetHeight;
        // 设置节点的最大高度为0px，并返回菜单弹出方向
        node.style.maxHeight = "0px";
        return this.direction = this.getDirection();
      }
    };

    // 获取菜单弹出方向
    Menu.prototype.getDirection = function() {
      if (this.node && this.node.parentNode.getBoundingClientRect().top + this.height + 60 > document.body.clientHeight && this.node.parentNode.getBoundingClientRect().top - this.height > 0) {
        return "top";
      } else {
        return "bottom";
      }
    };

    // 处理菜单点击事件
    Menu.prototype.handleClick = function(e) {
      var cb, i, item, keep_menu, len, ref, selected, title;
      keep_menu = false;
      ref = this.items;
      // 遍历菜单项
      for (i = 0, len = ref.length; i < len; i++) {
        item = ref[i];
        title = item[0], cb = item[1], selected = item[2];
        // 如果点击的菜单项与当前项匹配
        if (title === e.currentTarget.textContent || e.currentTarget["data-title"] === title) {
          // 执行回调函数，并记录是否保持菜单显示
          keep_menu = typeof cb === "function" ? cb(item) : void 0;
          break;
        }
      }
      // 如果不需要保持菜单显示且回调函数不为空，则隐藏菜单
      if (keep_menu !== true && cb !== null) {
        this.hide();
      }
      return false;
    };
    // 渲染菜单项
    Menu.prototype.renderItem = function(item) {
      var cb, classes, href, onclick, selected, title;
      // 解构赋值，获取菜单项的标题、回调函数、是否选中
      title = item[0], cb = item[1], selected = item[2];
      // 如果选中是一个函数，则调用该函数获取选中状态
      if (typeof selected === "function") {
        selected = selected();
      }
      // 如果标题是"---"，则返回一个菜单项分隔符
      if (title === "---") {
        return h("div.menu-item-separator", {
          key: Time.timestamp()
        });
      } else {
        // 根据不同情况设置 href 和 onclick
        if (cb === null) {
          href = void 0;
          onclick = this.handleClick;
        } else if (typeof cb === "string") {
          href = cb;
          onclick = true;
        } else {
          href = "#" + title;
          onclick = this.handleClick;
        }
        // 设置类名
        classes = {
          "selected": selected,
          "noaction": cb === null
        };
        // 返回一个菜单项链接
        return h("a.menu-item", {
          href: href,
          onclick: onclick,
          "data-title": title,
          key: title,
          classes: classes
        }, title);
      }
    };

    // 获取菜单样式
    Menu.prototype.getStyle = function() {
      var max_height, style;
      // 根据菜单是否可见设置最大高度
      if (this.visible) {
        max_height = this.height;
      } else {
        max_height = 0;
      }
      style = "max-height: " + max_height + "px";
      // 根据菜单弹出方向设置样式
      if (this.direction === "top") {
        style += ";margin-top: " + (0 - this.height - 50) + "px";
      } else {
        style += ";margin-top: 0px";
      }
      return style;
    };

    // 渲染菜单
    Menu.prototype.render = function(class_name) {
      if (class_name == null) {
        class_name = "";
      }
      // 如果菜单可见或者已经存在节点，则渲染菜单
      if (this.visible || this.node) {
        return h("div.menu" + class_name, {
          classes: {
            "visible": this.visible
          },
          style: this.getStyle(),
          afterCreate: this.storeNode
        }, this.items.map(this.renderItem));
      }
    };

    return Menu;

  })();

  // 将 Menu 对象挂载到全局对象 window 上
  window.Menu = Menu;

  // 监听鼠标在文档 body 上的mouseup事件
  document.body.addEventListener("mouseup", function(e) {
    var menu_node, menu_parents, ref, ref1;
    // ...
  });
    # 如果窗口中没有可见的菜单或者可见菜单没有节点，则返回 false
    if (!window.visible_menu || !window.visible_menu.node) {
      return false;
    }
    # 将可见菜单的节点赋值给 menu_node
    menu_node = window.visible_menu.node;
    # 将菜单节点和其父节点添加到 menu_parents 数组中
    menu_parents = [menu_node, menu_node.parentNode];
    # 如果事件目标的父节点不在 menu_parents 数组中，并且事件目标的父节点的父节点也不在 menu_parents 数组中
    if ((ref = e.target.parentNode, indexOf.call(menu_parents, ref) < 0) && (ref1 = e.target.parentNode.parentNode, indexOf.call(menu_parents, ref1) < 0)) {
      # 隐藏可见菜单
      window.visible_menu.hide();
      # 调度页面重新渲染
      return Page.projector.scheduleRender();
    }
  });
// 定义一个匿名函数，将其作为立即执行函数调用，并将 this 绑定到全局对象
}).call(this);

/* ---- lib/Promise.coffee ---- */

// 定义一个匿名函数
(function() {
  var Promise,  // 声明变量 Promise
    slice = [].slice;  // 声明变量 slice，并初始化为一个空数组的 slice 方法

  // 定义 Promise 类
  Promise = (function() {
    // 定义 Promise 类的静态方法 when
    Promise.when = function() {
      var args, fn, i, len, num_uncompleted, promise, task, task_id, tasks;  // 声明变量
      tasks = 1 <= arguments.length ? slice.call(arguments, 0) : [];  // 初始化 tasks 变量
      num_uncompleted = tasks.length;  // 获取任务数量
      args = new Array(num_uncompleted);  // 创建一个长度为任务数量的数组
      promise = new Promise();  // 创建一个 Promise 对象
      fn = function(task_id) {  // 定义函数 fn
        return task.then(function() {  // 返回任务的 then 方法
          args[task_id] = Array.prototype.slice.call(arguments);  // 将任务的结果存入 args 数组
          num_uncompleted--;  // 未完成任务数量减一
          if (num_uncompleted === 0) {  // 如果所有任务都已完成
            return promise.complete.apply(promise, args);  // 调用 promise 的 complete 方法
          }
        });
      };
      for (task_id = i = 0, len = tasks.length; i < len; task_id = ++i) {  // 遍历任务列表
        task = tasks[task_id];  // 获取任务
        fn(task_id);  // 调用 fn 函数
      }
      return promise;  // 返回 Promise 对象
    };

    // 定义 Promise 类的构造函数
    function Promise() {
      this.resolved = false;  // 初始化 resolved 属性为 false
      this.end_promise = null;  // 初始化 end_promise 属性为 null
      this.result = null;  // 初始化 result 属性为 null
      this.callbacks = [];  // 初始化 callbacks 属性为一个空数组
    }

    // 定义 Promise 类的 resolve 方法
    Promise.prototype.resolve = function() {
      var back, callback, i, len, ref;  // 声明变量
      if (this.resolved) {  // 如果已经 resolved
        return false;  // 返回 false
      }
      this.resolved = true;  // 设置 resolved 为 true
      this.data = arguments;  // 将参数存入 data 属性
      if (!arguments.length) {  // 如果参数长度为 0
        this.data = [true];  // 将 data 设置为 [true]
      }
      this.result = this.data[0];  // 将结果设置为 data 的第一个元素
      ref = this.callbacks;  // 将 callbacks 赋值给 ref
      for (i = 0, len = ref.length; i < len; i++) {  // 遍历 callbacks
        callback = ref[i];  // 获取回调函数
        back = callback.apply(callback, this.data);  // 调用回调函数
      }
      if (this.end_promise) {  // 如果存在 end_promise
        return this.end_promise.resolve(back);  // 调用 end_promise 的 resolve 方法
      }
    };

    // 定义 Promise 类的 fail 方法
    Promise.prototype.fail = function() {
      return this.resolve(false);  // 调用 resolve 方法，传入 false
    };

    // 定义 Promise 类的 then 方法
    Promise.prototype.then = function(callback) {
      if (this.resolved === true) {  // 如果已经 resolved
        callback.apply(callback, this.data);  // 调用回调函数
        return;  // 返回
      }
      this.callbacks.push(callback);  // 将回调函数添加到 callbacks 数组
      return this.end_promise = new Promise();  // 返回一个新的 Promise 对象
    };
  return Promise;
  // 返回 Promise 对象

})();

window.Promise = Promise;
// 将自定义的 Promise 对象赋值给全局对象 window 的 Promise 属性

/*
s = Date.now()
log = (text) ->
    console.log Date.now()-s, Array.prototype.slice.call(arguments).join(", ")

log "Started"

cmd = (query) ->
    p = new Promise()
    setTimeout ( ->
        p.resolve query+" Result"
    ), 100
    return p

back = cmd("SELECT * FROM message").then (res) ->
    log res
    return "Return from query"
.then (res) ->
    log "Back then", res

log "Query started", back
 */
// 注释部分为 JavaScript 代码，包括了定义变量、函数、Promise 对象的使用和链式调用，以及日志输出。
// 在全局环境中调用该函数
}).call(this);

/* ---- lib/Prototypes.coffee ---- */

// 定义 String 对象的 startsWith 方法
(function() {
  String.prototype.startsWith = function(s) {
    return this.slice(0, s.length) === s;
  };

  // 定义 String 对象的 endsWith 方法
  String.prototype.endsWith = function(s) {
    return s === '' || this.slice(-s.length) === s;
  };

  // 定义 String 对象的 repeat 方法
  String.prototype.repeat = function(count) {
    return new Array(count + 1).join(this);
  };

  // 定义全局函数 isEmpty
  window.isEmpty = function(obj) {
    var key;
    for (key in obj) {
      return false;
    }
    return true;
  };

}).call(this);

/* ---- lib/RateLimitCb.coffee ---- */

// 定义 RateLimitCb 函数
(function() {
  var call_after_interval, calling, calling_iterval, last_time,
    slice = [].slice;

  // 初始化变量
  last_time = {};
  calling = {};
  calling_iterval = {};
  call_after_interval = {};

  // 定义全局函数 RateLimitCb
  window.RateLimitCb = function(interval, fn, args) {
    var cb;
    if (args == null) {
      args = [];
    }
    cb = function() {
      var left;
      left = interval - (Date.now() - last_time[fn]);
      if (left <= 0) {
        delete last_time[fn];
        if (calling[fn]) {
          RateLimitCb(interval, fn, calling[fn]);
        }
        return delete calling[fn];
      } else {
        return setTimeout((function() {
          delete last_time[fn];
          if (calling[fn]) {
            RateLimitCb(interval, fn, calling[fn]);
          }
          return delete calling[fn];
        }), left);
      }
    };
    if (last_time[fn]) {
      return calling[fn] = args;
    } else {
      last_time[fn] = Date.now();
      return fn.apply(this, [cb].concat(slice.call(args)));
    }
  };

  // 定义全局函数 RateLimit
  window.RateLimit = function(interval, fn) {
    if (calling_iterval[fn] > interval) {
      clearInterval(calling[fn]);
      delete calling[fn];
    }
    # 如果函数 fn 没有被调用
    if (!calling[fn]) {
      # 设置 fn 的延迟调用标志为 false
      call_after_interval[fn] = false;
      # 立即调用函数 fn
      fn();
      # 设置 fn 的调用间隔
      calling_iterval[fn] = interval;
      # 返回调用 fn 的定时器 ID，并将 fn 标记为正在调用
      return calling[fn] = setTimeout((function() {
        # 如果在间隔时间内需要再次调用 fn，则再次调用
        if (call_after_interval[fn]) {
          fn();
        }
        # 删除 fn 的调用标记和延迟调用标记
        delete calling[fn];
        return delete call_after_interval[fn];
      }), interval);
    } else {
      # 如果 fn 已经在调用中，则设置延迟调用标志为 true
      return call_after_interval[fn] = true;
    }
  };


  /*
  window.s = Date.now()
  window.load = (done, num) ->
    console.log "Loading #{num}...", Date.now()-window.s
    setTimeout (-> done()), 1000
  
  RateLimit 500, window.load, [0] # Called instantly
  RateLimit 500, window.load, [1]
  setTimeout (-> RateLimit 500, window.load, [300]), 300
  setTimeout (-> RateLimit 500, window.load, [600]), 600 # Called after 1000ms
  setTimeout (-> RateLimit 500, window.load, [1000]), 1000
  setTimeout (-> RateLimit 500, window.load, [1200]), 1200  # Called after 2000ms
  setTimeout (-> RateLimit 500, window.load, [3000]), 3000  # Called after 3000ms
   */
# 定义一个匿名函数，将其作为方法调用
(function() {
  # 定义一个 Text 类
  var Text,
    # 定义一个 indexOf 方法，用于查找数组中指定元素的索引
    indexOf = [].indexOf || function(item) { for (var i = 0, l = this.length; i < l; i++) { if (i in this && this[i] === item) return i; } return -1; };

  # 定义 Text 类
  Text = (function() {
    # 定义 Text 类的构造函数
    function Text() {}

    # 将文本转换为颜色值
    Text.prototype.toColor = function(text, saturation, lightness) {
      # 计算文本的哈希值
      var hash, i, j, ref;
      if (saturation == null) {
        saturation = 30;
      }
      if (lightness == null) {
        lightness = 50;
      }
      hash = 0;
      for (i = j = 0, ref = text.length - 1; 0 <= ref ? j <= ref : j >= ref; i = 0 <= ref ? ++j : --j) {
        hash += text.charCodeAt(i) * i;
        hash = hash % 1777;
      }
      return "hsl(" + (hash % 360) + ("," + saturation + "%," + lightness + "%)");
    };

    # 渲染标记文本
    Text.prototype.renderMarked = function(text, options) {
      if (options == null) {
        options = {};
      }
      options["gfm"] = true;
      options["breaks"] = true;
      options["sanitize"] = true;
      options["renderer"] = marked_renderer;
      text = marked(text, options);
      return this.fixHtmlLinks(text);
    };

    # 将电子邮件地址转换为链接
    Text.prototype.emailLinks = function(text) {
      return text.replace(/([a-zA-Z0-9]+)@zeroid.bit/g, "<a href='?to=$1' onclick='return Page.message_create.show(\"$1\")'>$1@zeroid.bit</a>");
    };

    # 修复 HTML 链接
    Text.prototype.fixHtmlLinks = function(text) {
      if (window.is_proxy) {
        return text.replace(/href="http:\/\/(127.0.0.1|localhost):43110/g, 'href="http://zero');
      } else {
        return text.replace(/href="http:\/\/(127.0.0.1|localhost):43110/g, 'href="');
      }
    };

    # 修复链接
    Text.prototype.fixLink = function(link) {
      var back;
      if (window.is_proxy) {
        back = link.replace(/http:\/\/(127.0.0.1|localhost):43110/, 'http://zero');
        return back.replace(/http:\/\/zero\/([^\/]+\.bit)/, "http://$1");
      } else {
        return link.replace(/http:\/\/(127.0.0.1|localhost):43110/, '');
      }
    };
    # 将文本转换为 URL 格式，将非字母数字字符替换为 "+"
    Text.prototype.toUrl = function(text) {
      return text.replace(/[^A-Za-z0-9]/g, "+").replace(/[+]+/g, "+").replace(/[+]+$/, "");
    };
    
    # 根据地址获取站点的 URL
    Text.prototype.getSiteUrl = function(address) {
      # 如果是代理模式，并且地址中包含 "."，则返回以 "http://" 开头的地址
      if (window.is_proxy) {
        if (indexOf.call(address, ".") >= 0) {
          return "http://" + address + "/";
        } else {
          return "http://zero/" + address + "/";
        }
      } else {
        # 如果不是代理模式，则返回以 "/" 开头的地址
        return "/" + address + "/";
      }
    };
    
    # 修复回复文本的格式，将 ">.*\n" 后面不是换行符或 ">" 的字符替换为换行符
    Text.prototype.fixReply = function(text) {
      return text.replace(/(>.*\n)([^\n>])/gm, "$1\n$2");
    };
    
    # 将文本转换为比特币地址格式，将非字母数字字符替换为空
    Text.prototype.toBitcoinAddress = function(text) {
      return text.replace(/[^A-Za-z0-9]/g, "");
    };
    
    # 对象转换为 JSON 字符串，并进行 URI 编码
    Text.prototype.jsonEncode = function(obj) {
      return unescape(encodeURIComponent(JSON.stringify(obj)));
    };
    
    # JSON 字符串进行 URI 解码，并解析为对象
    Text.prototype.jsonDecode = function(obj) {
      return JSON.parse(decodeURIComponent(escape(obj)));
    };
    
    # 对象或字符串进行 base64 编码
    Text.prototype.fileEncode = function(obj) {
      if (typeof obj === "string") {
        return btoa(unescape(encodeURIComponent(obj)));
      } else {
        return btoa(unescape(encodeURIComponent(JSON.stringify(obj, void 0, '\t'))));
      }
    };
    
    # 对字符串进行 UTF-8 编码
    Text.prototype.utf8Encode = function(s) {
      return unescape(encodeURIComponent(s));
    };
    
    # 对字符串进行 UTF-8 解码
    Text.prototype.utf8Decode = function(s) {
      return decodeURIComponent(escape(s));
    };
    // 定义一个名为 distance 的方法，用于计算两个字符串之间的距离
    Text.prototype.distance = function(s1, s2) {
      // 将两个字符串转换为小写
      s1 = s1.toLocaleLowerCase();
      s2 = s2.toLocaleLowerCase();
      // 初始化变量
      next_find_i = 0;
      next_find = s2[0];
      match = true;
      extra_parts = {};
      // 遍历字符串 s1
      for (j = 0, len = s1.length; j < len; j++) {
        char = s1[j];
        // 判断当前字符是否与下一个要找的字符相同
        if (char !== next_find) {
          // 如果不同，将当前字符添加到额外部分中
          if (extra_parts[next_find_i]) {
            extra_parts[next_find_i] += char;
          } else {
            extra_parts[next_find_i] = char;
          }
        } else {
          // 如果相同，更新下一个要找的字符的索引和值
          next_find_i++;
          next_find = s2[next_find_i];
        }
      }
      // 如果额外部分中存在值，则将其置为空字符串
      if (extra_parts[next_find_i]) {
        extra_parts[next_find_i] = "";
      }
      // 将额外部分转换为数组
      extra_parts = (function() {
        var results;
        results = [];
        for (key in extra_parts) {
          val = extra_parts[key];
          results.push(val);
        }
        return results;
      })();
      // 如果下一个要找的字符的索引大于等于 s2 的长度，则返回额外部分的长度和连接后的字符串长度之和
      if (next_find_i >= s2.length) {
        return extra_parts.length + extra_parts.join("").length;
      } else {
        // 否则返回 false
        return false;
      }
    };

    // 定义一个名为 parseQuery 的方法，用于解析查询字符串
    Text.prototype.parseQuery = function(query) {
      // 初始化变量
      params = {};
      parts = query.split('&');
      // 遍历查询字符串的各个部分
      for (j = 0, len = parts.length; j < len; j++) {
        part = parts[j];
        // 将部分字符串按照 "=" 分割，获取键和值
        ref = part.split("="), key = ref[0], val = ref[1];
        // 如果值存在，则将键值对添加到 params 中；否则将键作为 url 的值添加到 params 中
        if (val) {
          params[decodeURIComponent(key)] = decodeURIComponent(val);
        } else {
          params["url"] = decodeURIComponent(key);
        }
      }
      // 返回解析后的参数对象
      return params;
    };
    # 将参数编码为查询字符串
    Text.prototype.encodeQuery = function(params) {
      var back, key, val;
      back = [];
      # 如果参数中包含 url，则将其添加到返回数组中
      if (params.url) {
        back.push(params.url);
      }
      # 遍历参数对象，将键值对编码并添加到返回数组中
      for (key in params) {
        val = params[key];
        if (!val || key === "url") {
          continue;
        }
        back.push((encodeURIComponent(key)) + "=" + (encodeURIComponent(val)));
      }
      # 将数组中的元素用 & 连接成字符串并返回
      return back.join("&");
    };

    # 对文本进行高亮处理
    Text.prototype.highlight = function(text, search) {
      var back, i, j, len, part, parts;
      # 如果文本为空，则返回空数组
      if (!text) {
        return [""];
      }
      # 使用正则表达式将文本分割成多个部分
      parts = text.split(RegExp(search, "i"));
      back = [];
      # 遍历分割后的部分，将其添加到返回数组中，并在需要的地方添加高亮标签
      for (i = j = 0, len = parts.length; j < len; i = ++j) {
        part = parts[i];
        back.push(part);
        if (i < parts.length - 1) {
          back.push(h("span.highlight", {
            key: i
          }, search));
        }
      }
      # 返回处理后的文本数组
      return back;
    };

    # 格式化文件大小
    Text.prototype.formatSize = function(size) {
      var size_mb;
      # 如果大小不是数字，则返回空字符串
      if (isNaN(parseInt(size))) {
        return "";
      }
      # 将文件大小转换为 MB
      size_mb = size / 1024 / 1024;
      # 根据文件大小的不同范围，返回不同格式的文件大小字符串
      if (size_mb >= 1000) {
        return (size_mb / 1024).toFixed(1) + " GB";
      } else if (size_mb >= 100) {
        return size_mb.toFixed(0) + " MB";
      } else if (size / 1024 >= 1000) {
        return size_mb.toFixed(2) + " MB";
      } else {
        return (parseInt(size) / 1024).toFixed(2) + " KB";
      }
    };

    # 返回 Text 对象
    return Text;

  })();

  # 判断是否为代理
  window.is_proxy = document.location.host === "zero" || window.location.pathname === "/";

  # 创建 Text 对象
  window.Text = new Text();
// 将整个代码块包裹在一个立即调用的匿名函数中，以避免变量污染全局作用域
(function() {
  // 定义 Time 类
  var Time;

  Time = (function() {
    function Time() {}

    // 定义 since 方法，计算时间差并返回相应的描述
    Time.prototype.since = function(timestamp) {
      var back, minutes, now, secs;
      // 获取当前时间戳（单位：秒）
      now = +(new Date) / 1000;
      // 如果传入的时间戳大于 1000000000000，将其转换为秒
      if (timestamp > 1000000000000) {
        timestamp = timestamp / 1000;
      }
      // 计算时间差（单位：秒）
      secs = now - timestamp;
      // 根据时间差返回相应的描述
      if (secs < 60) {
        back = "Just now";
      } else if (secs < 60 * 60) {
        minutes = Math.round(secs / 60);
        back = "" + minutes + " minutes ago";
      } else if (secs < 60 * 60 * 24) {
        back = (Math.round(secs / 60 / 60)) + " hours ago";
      } else if (secs < 60 * 60 * 24 * 3) {
        back = (Math.round(secs / 60 / 60 / 24)) + " days ago";
      } else {
        back = "on " + this.date(timestamp);
      }
      // 替换掉以 "1 [a-z]+s" 开头的字符串为 "1 [a-z]+"，用于处理单复数
      back = back.replace(/^1 ([a-z]+)s/, "1 $1");
      return back;
    };

    // 定义 dateIso 方法，返回 ISO 格式的日期字符串
    Time.prototype.dateIso = function(timestamp) {
      var tzoffset;
      // 如果未传入时间戳，则使用当前时间戳
      if (timestamp == null) {
        timestamp = null;
      }
      if (!timestamp) {
        timestamp = window.Time.timestamp();
      }
      // 如果传入的时间戳大于 1000000000000，将其转换为秒
      if (timestamp > 1000000000000) {
        timestamp = timestamp / 1000;
      }
      // 获取时区偏移量
      tzoffset = (new Date()).getTimezoneOffset() * 60;
      // 返回 ISO 格式的日期字符串
      return (new Date((timestamp - tzoffset) * 1000)).toISOString().split("T")[0];
    };
    # 在 Time 对象的原型上添加 date 方法，用于将时间戳转换为指定格式的日期字符串
    Time.prototype.date = function(timestamp, format) {
      var display, parts;
      # 如果时间戳为空，则设置为 null
      if (timestamp == null) {
        timestamp = null;
      }
      # 如果格式为空，则设置为 "short"
      if (format == null) {
        format = "short";
      }
      # 如果时间戳不存在，则获取当前时间戳
      if (!timestamp) {
        timestamp = window.Time.timestamp();
      }
      # 如果时间戳大于 1000000000000，则将其转换为秒级时间戳
      if (timestamp > 1000000000000) {
        timestamp = timestamp / 1000;
      }
      # 将时间戳转换为日期字符串，并根据格式进行处理
      parts = (new Date(timestamp * 1000)).toString().split(" ");
      if (format === "short") {
        display = parts.slice(1, 4);
      } else if (format === "day") {
        display = parts.slice(1, 3);
      } else if (format === "month") {
        display = [parts[1], parts[3]];
      } else if (format === "long") {
        display = parts.slice(1, 5);
      }
      # 返回处理后的日期字符串
      return display.join(" ").replace(/( [0-9]{4})/, ",$1");
    };
    
    # 在 Time 对象的原型上添加 weekDay 方法，用于获取时间戳对应的星期几
    Time.prototype.weekDay = function(timestamp) {
      # 如果时间戳大于 1000000000000，则将其转换为秒级时间戳
      if (timestamp > 1000000000000) {
        timestamp = timestamp / 1000;
      }
      # 返回时间戳对应的星期几
      return ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][(new Date(timestamp * 1000)).getDay()];
    };
    
    # 在 Time 对象上添加 timestamp 方法，用于将日期字符串转换为时间戳
    Time.prototype.timestamp = function(date) {
      # 如果日期为空，则设置为空字符串
      if (date == null) {
        date = "";
      }
      # 如果日期为 "now" 或为空，则返回当前时间戳
      if (date === "now" || date === "") {
        return parseInt(+(new Date) / 1000);
      } else {
        # 否则，将日期字符串转换为时间戳并返回
        return parseInt(Date.parse(date) / 1000);
      }
    };
    
    # 返回 Time 对象
    return Time;
    })();
    # 将 Time 对象赋值给 window.Time
    window.Time = new Time;
// 调用一个匿名函数，将 this 绑定到当前上下文
}).call(this);

// 定义 ZeroFrame 类
/* ---- lib/ZeroFrame.coffee ---- */
(function() {
  var ZeroFrame,
    // 定义 bind 函数，用于绑定函数的上下文
    bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; },
    // 定义 extend 函数，用于继承父类的属性和方法
    extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
    // 定义 hasProp 对象，用于检查对象是否包含特定属性
    hasProp = {}.hasOwnProperty;

  // ZeroFrame 类继承自 superClass
  ZeroFrame = (function(superClass) {
    // 继承 superClass 的属性和方法
    extend(ZeroFrame, superClass);

    // ZeroFrame 构造函数
    function ZeroFrame(url) {
      // 绑定 this 到 onCloseWebsocket 方法
      this.onCloseWebsocket = bind(this.onCloseWebsocket, this);
      // 绑定 this 到 onOpenWebsocket 方法
      this.onOpenWebsocket = bind(this.onOpenWebsocket, this);
      // 绑定 this 到 onRequest 方法
      this.onRequest = bind(this.onRequest, this);
      // 绑定 this 到 onMessage 方法
      this.onMessage = bind(this.onMessage, this);
      // 设置 ZeroFrame 实例的 url 属性
      this.url = url;
      // 初始化等待的回调函数
      this.waiting_cb = {};
      // 获取当前页面的 wrapper_nonce
      this.wrapper_nonce = document.location.href.replace(/.*wrapper_nonce=([A-Za-z0-9]+).*/, "$1");
      // 连接到服务器
      this.connect();
      // 设置下一个消息的 ID
      this.next_message_id = 1;
      // 初始化历史状态
      this.history_state = {};
      // 调用 init 方法
      this.init();
    }

    // ZeroFrame 类的 init 方法
    ZeroFrame.prototype.init = function() {
      // 返回 this
      return this;
    };
    // 连接到父窗口
    ZeroFrame.prototype.connect = function() {
      // 将目标设为父窗口
      this.target = window.parent;
      // 添加消息事件监听器，处理接收到的消息
      window.addEventListener("message", this.onMessage, false);
      // 发送命令通知内部准备就绪
      this.cmd("innerReady");
      // 在窗口即将关闭前保存滚动位置
      window.addEventListener("beforeunload", (function(_this) {
        return function(e) {
          // 记录滚动位置
          _this.log("save scrollTop", window.pageYOffset);
          _this.history_state["scrollTop"] = window.pageYOffset;
          // 通知 wrapper 替换状态
          return _this.cmd("wrapperReplaceState", [_this.history_state, null]);
        };
      })(this));
      // 获取当前状态
      return this.cmd("wrapperGetState", [], (function(_this) {
        return function(state) {
          // 如果存在状态，则更新历史状态
          if (state != null) {
            _this.history_state = state;
          }
          // 恢复滚动位置
          _this.log("restore scrollTop", state, window.pageYOffset);
          if (window.pageYOffset === 0 && state) {
            return window.scroll(window.pageXOffset, state.scrollTop);
          }
        };
      })(this));
    };

    // 处理接收到的消息
    ZeroFrame.prototype.onMessage = function(e) {
      var cmd, message;
      // 获取消息内容和命令
      message = e.data;
      cmd = message.cmd;
      // 根据命令类型进行相应处理
      if (cmd === "response") {
        if (this.waiting_cb[message.to] != null) {
          return this.waiting_cb[message.to](message.result);
        } else {
          return this.log("Websocket callback not found:", message);
        }
      } else if (cmd === "wrapperReady") {
        return this.cmd("innerReady");
      } else if (cmd === "ping") {
        return this.response(message.id, "pong");
      } else if (cmd === "wrapperOpenedWebsocket") {
        return this.onOpenWebsocket();
      } else if (cmd === "wrapperClosedWebsocket") {
        return this.onCloseWebsocket();
      } else {
        return this.onRequest(cmd, message.params);
      }
    };

    // 处理未知请求
    ZeroFrame.prototype.onRequest = function(cmd, message) {
      return this.log("Unknown request", message);
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
      if (params == null) {
        params = {};
      }
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
      if (cb == null) {
        cb = null;
      }
      # 设置消息的包装 nonce 和 id
      message.wrapper_nonce = this.wrapper_nonce;
      message.id = this.next_message_id;
      # 递增下一个消息的 id
      this.next_message_id += 1;
      # 发送消息到目标窗口
      this.target.postMessage(message, "*");
      # 如果有回调函数，将回调函数存储到等待回调的字典中
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
    
    # 将 ZeroFrame 对象绑定到 window 对象上
    window.ZeroFrame = ZeroFrame;
}).call(this);

/* ---- lib/maquette.js ---- */

// 使用立即执行函数表达式，将 maquette.js 封装为模块
(function (root, factory) {
    // 如果支持 AMD 规范，则注册为匿名模块
    if (typeof define === 'function' && define.amd) {
        define(['exports'], factory);
    } else if (typeof exports === 'object' && typeof exports.nodeName !== 'string') {
        // 如果支持 CommonJS 规范，则使用 factory 导出模块
        factory(exports);
    } else {
        // 否则将模块挂载到全局对象 root.maquette 上
        factory(root.maquette = {});
    }
}(this, function (exports) {
    'use strict';
    // 声明变量
    var NAMESPACE_W3 = 'http://www.w3.org/';
    var NAMESPACE_SVG = NAMESPACE_W3 + '2000/svg';
    var NAMESPACE_XLINK = NAMESPACE_W3 + '1999/xlink';
    // Utilities
    // 声明空数组
    var emptyArray = [];
    // 定义 extend 函数，用于合并对象
    var extend = function (base, overrides) {
        var result = {};
        // 将 base 对象的属性复制到 result 对象
        Object.keys(base).forEach(function (key) {
            result[key] = base[key];
        });
        // 如果有 overrides 对象，则将其属性也复制到 result 对象
        if (overrides) {
            Object.keys(overrides).forEach(function (key) {
                result[key] = overrides[key];
            });
        }
        return result;
    };
    // Hyperscript helper functions
    // 定义 same 函数，用于比较两个虚拟节点是否相同
    var same = function (vnode1, vnode2) {
        if (vnode1.vnodeSelector !== vnode2.vnodeSelector) {
            return false;
        }
        if (vnode1.properties && vnode2.properties) {
            if (vnode1.properties.key !== vnode2.properties.key) {
                return false;
            }
            return vnode1.properties.bind === vnode2.properties.bind;
        }
        return !vnode1.properties && !vnode2.properties;
    };
    // 定义 toTextVNode 函数，用于将数据转换为文本节点的虚拟节点
    var toTextVNode = function (data) {
        return {
            vnodeSelector: '',
            properties: undefined,
            children: undefined,
            text: data.toString(),
            domNode: null
        };
    };
    // 将子元素插入到父元素中
    var appendChildren = function (parentSelector, insertions, main) {
        for (var i = 0; i < insertions.length; i++) {
            var item = insertions[i];
            if (Array.isArray(item)) {
                // 递归调用，将子元素插入到父元素中
                appendChildren(parentSelector, item, main);
            } else {
                if (item !== null && item !== undefined) {
                    if (!item.hasOwnProperty('vnodeSelector')) {
                        // 如果子元素不包含 vnodeSelector 属性，则转换为文本节点
                        item = toTextVNode(item);
                    }
                    // 将子元素添加到主元素中
                    main.push(item);
                }
            }
        }
    };
    // 渲染辅助函数
    var missingTransition = function () {
        // 如果没有提供过渡对象，则抛出错误
        throw new Error('Provide a transitions object to the projectionOptions to do animations');
    };
    // 默认投影选项
    var DEFAULT_PROJECTION_OPTIONS = {
        namespace: undefined,
        eventHandlerInterceptor: undefined,
        styleApplyer: function (domNode, styleName, value) {
            // 提供一个钩子来为仍然需要它的浏览器添加供应商前缀
            domNode.style[styleName] = value;
        },
        transitions: {
            enter: missingTransition,
            exit: missingTransition
        }
    };
    // 应用默认投影选项
    var applyDefaultProjectionOptions = function (projectorOptions) {
        return extend(DEFAULT_PROJECTION_OPTIONS, projectorOptions);
    };
    // 检查样式值
    var checkStyleValue = function (styleValue) {
        if (typeof styleValue !== 'string') {
            // 样式值必须是字符串
            throw new Error('Style values must be strings');
        }
    };
    // 查找子元素的索引
    var findIndexOfChild = function (children, sameAs, start) {
        if (sameAs.vnodeSelector !== '') {
            // 从指定位置开始查找与指定元素相同的子元素索引
            for (var i = start; i < children.length; i++) {
                if (same(children[i], sameAs)) {
                    return i;
                }
            }
        }
        return -1;
    };
    # 定义一个函数，用于处理添加节点的逻辑
    var nodeAdded = function (vNode, transitions) {
        # 如果虚拟节点有属性
        if (vNode.properties) {
            # 获取进入动画
            var enterAnimation = vNode.properties.enterAnimation;
            # 如果存在进入动画
            if (enterAnimation) {
                # 如果进入动画是一个函数
                if (typeof enterAnimation === 'function') {
                    # 调用进入动画函数
                    enterAnimation(vNode.domNode, vNode.properties);
                } else {
                    # 调用过渡对象的进入方法
                    transitions.enter(vNode.domNode, vNode.properties, enterAnimation);
                }
            }
        }
    };
    # 定义一个函数，用于处理移除节点的逻辑
    var nodeToRemove = function (vNode, transitions) {
        # 获取节点的 DOM 元素
        var domNode = vNode.domNode;
        # 如果虚拟节点有属性
        if (vNode.properties) {
            # 获取退出动画
            var exitAnimation = vNode.properties.exitAnimation;
            # 如果存在退出动画
            if (exitAnimation) {
                # 设置节点的指针事件为无
                domNode.style.pointerEvents = 'none';
                # 定义一个函数，用于移除节点
                var removeDomNode = function () {
                    if (domNode.parentNode) {
                        domNode.parentNode.removeChild(domNode);
                    }
                };
                # 如果退出动画是一个函数
                if (typeof exitAnimation === 'function') {
                    # 调用退出动画函数
                    exitAnimation(domNode, removeDomNode, vNode.properties);
                    return;
                } else {
                    # 调用过渡对象的退出方法
                    transitions.exit(vNode.domNode, vNode.properties, exitAnimation, removeDomNode);
                    return;
                }
            }
        }
        # 如果节点的 DOM 元素有父节点，则移除节点
        if (domNode.parentNode) {
            domNode.parentNode.removeChild(domNode);
        }
    };
    // 检查子节点是否可区分
    var checkDistinguishable = function (childNodes, indexToCheck, parentVNode, operation) {
        // 获取要检查的子节点
        var childNode = childNodes[indexToCheck];
        // 如果子节点是文本节点，则无需区分
        if (childNode.vnodeSelector === '') {
            return;
        }
        // 获取子节点的属性和键
        var properties = childNode.properties;
        var key = properties ? (properties.key === undefined ? properties.bind : properties.key) : undefined;
        // 如果没有键，则遍历所有子节点，检查是否有重复的节点
        if (!key) {
            for (var i = 0; i < childNodes.length; i++) {
                if (i !== indexToCheck) {
                    var node = childNodes[i];
                    // 如果有重复的节点，则根据操作类型抛出错误
                    if (same(node, childNode)) {
                        if (operation === 'added') {
                            throw new Error(parentVNode.vnodeSelector + ' had a ' + childNode.vnodeSelector + ' child ' + 'added, but there is now more than one. You must add unique key properties to make them distinguishable.');
                        } else {
                            throw new Error(parentVNode.vnodeSelector + ' had a ' + childNode.vnodeSelector + ' child ' + 'removed, but there were more than one. You must add unique key properties to make them distinguishable.');
                        }
                    }
                }
            }
        }
    };
    var createDom;  // 未定义的变量
    var updateDom;  // 未定义的变量
    };
    // 添加子节点到 DOM 节点
    var addChildren = function (domNode, children, projectionOptions) {
        if (!children) {
            return;
        }
        // 遍历所有子节点，创建对应的 DOM 节点
        for (var i = 0; i < children.length; i++) {
            createDom(children[i], domNode, undefined, projectionOptions);
        }
    };
    # 初始化 DOM 节点的属性和子节点
    var initPropertiesAndChildren = function (domNode, vnode, projectionOptions) {
        # 添加子节点到 DOM 节点
        addChildren(domNode, vnode.children, projectionOptions);
        # 在设置属性之前设置子节点，这对于 <select> 元素的 value 属性是必要的
        if (vnode.text) {
            # 设置 DOM 节点的文本内容
            domNode.textContent = vnode.text;
        }
        # 设置 DOM 节点的属性
        setProperties(domNode, vnode.properties, projectionOptions);
        # 如果存在 afterCreate 属性，则调用该属性指定的函数
        if (vnode.properties && vnode.properties.afterCreate) {
            vnode.properties.afterCreate(domNode, projectionOptions, vnode.vnodeSelector, vnode.properties, vnode.children);
        }
    };
    # 创建 DOM 元素的函数，接受虚拟节点、父节点、插入位置和投影选项作为参数
    createDom = function (vnode, parentNode, insertBefore, projectionOptions) {
        var domNode, i, c, start = 0, type, found;
        var vnodeSelector = vnode.vnodeSelector;
        # 如果虚拟节点选择器为空
        if (vnodeSelector === '') {
            # 创建文本节点
            domNode = vnode.domNode = document.createTextNode(vnode.text);
            # 如果有插入位置，则在插入位置之前插入文本节点，否则追加到父节点
            if (insertBefore !== undefined) {
                parentNode.insertBefore(domNode, insertBefore);
            } else {
                parentNode.appendChild(domNode);
            }
        } else {
            # 遍历虚拟节点选择器
            for (i = 0; i <= vnodeSelector.length; ++i) {
                c = vnodeSelector.charAt(i);
                # 如果遍历到选择器末尾或者遇到 . 或 # 符号
                if (i === vnodeSelector.length || c === '.' || c === '#') {
                    type = vnodeSelector.charAt(start - 1);
                    found = vnodeSelector.slice(start, i);
                    # 根据类型添加类名或者 ID
                    if (type === '.') {
                        domNode.classList.add(found);
                    } else if (type === '#') {
                        domNode.id = found;
                    } else {
                        # 如果是 svg 元素，则设置命名空间为 SVG
                        if (found === 'svg') {
                            projectionOptions = extend(projectionOptions, { namespace: NAMESPACE_SVG });
                        }
                        # 根据命名空间创建元素
                        if (projectionOptions.namespace !== undefined) {
                            domNode = vnode.domNode = document.createElementNS(projectionOptions.namespace, found);
                        } else {
                            domNode = vnode.domNode = document.createElement(found);
                        }
                        # 如果有插入位置，则在插入位置之前插入元素，否则追加到父节点
                        if (insertBefore !== undefined) {
                            parentNode.insertBefore(domNode, insertBefore);
                        } else {
                            parentNode.appendChild(domNode);
                        }
                    }
                    start = i + 1;
                }
            }
            # 初始化属性和子元素
            initPropertiesAndChildren(domNode, vnode, projectionOptions);
        }
    };
    # 更新虚拟 DOM 对象到实际 DOM 节点
    updateDom = function (previous, vnode, projectionOptions) {
        # 获取之前的 DOM 节点
        var domNode = previous.domNode;
        var textUpdated = false;
        # 如果传入的虚拟节点和之前的虚拟节点相同，则不做任何修改
        if (previous === vnode) {
            return false;    # 根据约定，传入的 VNode 对象在传递给 maquette 后不得再修改
        }
        var updated = false;
        # 如果虚拟节点的选择器为空
        if (vnode.vnodeSelector === '') {
            # 如果虚拟节点的文本内容和之前的不同
            if (vnode.text !== previous.text) {
                # 创建新的文本节点
                var newVNode = document.createTextNode(vnode.text);
                # 用新的文本节点替换原来的 DOM 节点
                domNode.parentNode.replaceChild(newVNode, domNode);
                vnode.domNode = newVNode;
                textUpdated = true;
                return textUpdated;
            }
        } else {
            # 如果虚拟节点的选择器以 'svg' 开头
            if (vnode.vnodeSelector.lastIndexOf('svg', 0) === 0) {
                # 更新投影选项，设置命名空间为 SVG
                projectionOptions = extend(projectionOptions, { namespace: NAMESPACE_SVG });
            }
            # 如果之前的文本内容和现在的不同
            if (previous.text !== vnode.text) {
                updated = true;
                # 如果虚拟节点的文本内容为 undefined
                if (vnode.text === undefined) {
                    domNode.removeChild(domNode.firstChild);    # 可能是唯一的文本节点
                } else {
                    domNode.textContent = vnode.text;
                }
            }
            # 更新子节点
            updated = updateChildren(vnode, domNode, previous.children, vnode.children, projectionOptions) || updated;
            # 更新属性
            updated = updateProperties(domNode, previous.properties, vnode.properties, projectionOptions) || updated;
            # 如果虚拟节点有属性且有 afterUpdate 方法，则调用该方法
            if (vnode.properties && vnode.properties.afterUpdate) {
                vnode.properties.afterUpdate(domNode, projectionOptions, vnode.vnodeSelector, vnode.properties, vnode.children);
            }
        }
        # 如果更新了，并且虚拟节点有属性且有 updateAnimation 方法，则调用该方法
        if (updated && vnode.properties && vnode.properties.updateAnimation) {
            vnode.properties.updateAnimation(domNode, vnode.properties, previous.properties);
        }
        # 将虚拟节点的 DOM 节点设置为之前的 DOM 节点
        vnode.domNode = previous.domNode;
        return textUpdated;
    };
    // 创建投影，将虚拟 DOM 映射到真实 DOM
    var createProjection = function (vnode, projectionOptions) {
        // 返回一个对象，包含 update 方法和 domNode 属性
        return {
            // update 方法用于更新虚拟 DOM
            update: function (updatedVnode) {
                // 如果更新后的虚拟 DOM 的选择器与之前的不同，抛出错误
                if (vnode.vnodeSelector !== updatedVnode.vnodeSelector) {
                    throw new Error('The selector for the root VNode may not be changed. (consider using dom.merge and add one extra level to the virtual DOM)');
                }
                // 调用 updateDom 方法更新真实 DOM
                updateDom(vnode, updatedVnode, projectionOptions);
                // 更新 vnode 为 updatedVnode
                vnode = updatedVnode;
            },
            // 返回虚拟 DOM 对应的真实 DOM 节点
            domNode: vnode.domNode
        };
    };
    // 这里没有添加另外两个参数，因为 TypeScript 编译器会为解构 'children' 创建代理代码
    // 导出一个名为 h 的函数，接受一个选择器参数和一个属性参数
    exports.h = function (selector) {
        var properties = arguments[1];
        // 如果选择器不是字符串类型，抛出错误
        if (typeof selector !== 'string') {
            throw new Error();
        }
        var childIndex = 1;
        // 如果存在属性参数，并且属性参数不包含 vnodeSelector 属性，不是数组，并且是对象类型
        if (properties && !properties.hasOwnProperty('vnodeSelector') && !Array.isArray(properties) && typeof properties === 'object') {
            childIndex = 2;
        } else {
            // 可选的属性参数被省略
            properties = undefined;
        }
        var text = undefined;
        var children = undefined;
        var argsLength = arguments.length;
        // 识别一个常见的特殊情况，只有一个文本节点
        if (argsLength === childIndex + 1) {
            var onlyChild = arguments[childIndex];
            if (typeof onlyChild === 'string') {
                text = onlyChild;
            } else if (onlyChild !== undefined && onlyChild.length === 1 && typeof onlyChild[0] === 'string') {
                text = onlyChild[0];
            }
        }
        // 如果没有文本节点，处理子节点
        if (text === undefined) {
            children = [];
            for (; childIndex < arguments.length; childIndex++) {
                var child = arguments[childIndex];
                if (child === null || child === undefined) {
                    continue;
                } else if (Array.isArray(child)) {
                    appendChildren(selector, child, children);
                } else if (child.hasOwnProperty('vnodeSelector')) {
                    children.push(child);
                } else {
                    children.push(toTextVNode(child));
                }
            }
        }
        // 返回一个包含选择器、属性、子节点和文本的对象
        return {
            vnodeSelector: selector,
            properties: properties,
            children: children,
            text: text === '' ? undefined : text,
            domNode: null
        };
    };
    /**
     * 包含简单的低级实用函数，用于操作真实 DOM
     */
    };
    /**
     * 创建一个 CalculationCache 对象，用于缓存 VNode 树。
     * 实际上，几乎不需要缓存 VNode 树，因为几乎永远不会出现 60 帧每秒的问题。
     * 有关更多信息，请参阅 CalculationCache。
     *
     * @param <Result> 缓存值的类型。
     */
    exports.createCache = function () {
        var cachedInputs = undefined; // 缓存的输入
        var cachedOutcome = undefined; // 缓存的结果
        var result = {
            invalidate: function () { // 使缓存无效
                cachedOutcome = undefined; // 清空缓存的结果
                cachedInputs = undefined; // 清空缓存的输入
            },
            result: function (inputs, calculation) { // 计算结果
                if (cachedInputs) { // 如果有缓存的输入
                    for (var i = 0; i < inputs.length; i++) { // 遍历输入
                        if (cachedInputs[i] !== inputs[i]) { // 如果输入不一致
                            cachedOutcome = undefined; // 清空缓存的结果
                        }
                    }
                }
                if (!cachedOutcome) { // 如果没有缓存的结果
                    cachedOutcome = calculation(); // 计算结果
                    cachedInputs = inputs; // 缓存输入
                }
                return cachedOutcome; // 返回缓存的结果
            }
        };
        return result; // 返回结果对象
    };
    /**
     * 创建一个 Mapping 实例，用于将源对象数组与结果对象数组保持同步。
     * 参见 http://maquettejs.org/docs/arrays.html|Working with arrays。
     *
     * @param <Source>       源项目的类型。例如数据库记录。
     * @param <Target>       目标项目的类型。例如 Component。
     * @param getSourceKey   一个函数(source)，必须返回一个用于标识每个源对象的键。结果必须是字符串或数字。
     * @param createResult   一个函数(source, index)，必须从给定的源创建一个新的结果对象。此函数与 Array.map(callback) 中的 callback 参数相同。
     * @param updateResult   一个函数(source, target, index)，用于将结果更新为更新后的源对象。
     */
    // 创建一个名为 createMapping 的函数，接受三个参数：getSourceKey、createResult 和 updateResult
    exports.createMapping = function (getSourceKey, createResult, updateResult) {
        // 声明一个空数组 keys 用于存储源数据的键
        var keys = [];
        // 声明一个空数组 results 用于存储结果数据
        var results = [];
        // 返回一个对象，包含 results 和 map 两个属性
        return {
            results: results, // 将结果数组 results 作为对象的一个属性返回
            // map 方法用于映射新的数据源
            map: function (newSources) {
                // 将新数据源的键映射到一个新的数组 newKeys 中
                var newKeys = newSources.map(getSourceKey);
                // 复制结果数组 results 到 oldTargets 中
                var oldTargets = results.slice();
                // 声明一个变量 oldIndex 并初始化为 0
                var oldIndex = 0;
                // 遍历新数据源数组
                for (var i = 0; i < newSources.length; i++) {
                    // 获取当前源数据
                    var source = newSources[i];
                    // 获取当前源数据的键
                    var sourceKey = newKeys[i];
                    // 如果当前源数据的键等于 keys 数组中的某个值
                    if (sourceKey === keys[oldIndex]) {
                        // 将结果数组中对应位置的值设置为 oldTargets 中对应位置的值
                        results[i] = oldTargets[oldIndex];
                        // 调用 updateResult 方法，更新结果
                        updateResult(source, oldTargets[oldIndex], i);
                        // oldIndex 自增
                        oldIndex++;
                    } else {
                        // 如果当前源数据的键不在 keys 数组中
                        var found = false;
                        // 遍历 keys 数组
                        for (var j = 1; j < keys.length; j++) {
                            // 计算搜索索引
                            var searchIndex = (oldIndex + j) % keys.length;
                            // 如果 keys 数组中的某个值等于当前源数据的键
                            if (keys[searchIndex] === sourceKey) {
                                // 将结果数组中对应位置的值设置为 oldTargets 中对应位置的值
                                results[i] = oldTargets[searchIndex];
                                // 调用 updateResult 方法，更新结果
                                updateResult(newSources[i], oldTargets[searchIndex], i);
                                // 更新 oldIndex
                                oldIndex = searchIndex + 1;
                                // 设置 found 为 true
                                found = true;
                                // 跳出循环
                                break;
                            }
                        }
                        // 如果未找到匹配的键
                        if (!found) {
                            // 调用 createResult 方法，创建新的结果
                            results[i] = createResult(source, i);
                        }
                    }
                }
                // 将结果数组的长度设置为新数据源数组的长度
                results.length = newSources.length;
                // 更新 keys 数组为新的键数组
                keys = newKeys;
            }
        };
    };
    /**
     * 使用提供的 projectionOptions 创建一个 Projector 实例。
     *
     * 有关更多信息，请参阅 Projector。
     *
     * @param projectionOptions   影响 DOM 渲染和更新的选项。
     */
    };
// 定义一个立即执行函数，将 BINARY_EXTENSIONS 数组添加到 window 对象中
(function() {
  // 定义一个包含各种二进制文件扩展名的数组
  window.BINARY_EXTENSIONS = ["3dm", "3ds", "3g2", "3gp", "7z", "a", "aac", "adp", "ai", "aif", "aiff", "alz", "ape", "apk", "appimage", "ar", "arj", "asc", "asf", "au", "avi", "bak", "baml", "bh", "bin", "bk", "bmp", "btif", "bz2", "bzip2", "cab", "caf", "cgm", "class", "cmx", "cpio", "cr2", "cur", "dat", "dcm", "deb", "dex", "djvu", "dll", "dmg", "dng", "doc", "docm", "docx", "dot", "dotm", "dra", "DS_Store", "dsk", "dts", "dtshd", "dvb", "dwg", "dxf", "ecelp4800", "ecelp7470", "ecelp9600", "egg", "eol", "eot", "epub", "exe", "f4v", "fbs", "fh", "fla", "flac", "flatpak", "fli", "flv", "fpx", "fst", "fvt", "g3", "gh", "gif", "gpg", "graffle", "gz", "gzip", "h261", "h263", "h264", "icns", "ico", "ief", "img", "ipa", "iso", "jar", "jpeg", "jpg", "jpgv", "jpm", "jxr", "key", "ktx", "lha", "lib", "lvp", "lz", "lzh", "lzma", "lzo", "m3u", "m4a", "m4v", "mar", "mdi", "mht", "mid", "midi", "mj2", "mka", "mkv", "mmr", "mng", "mobi", "mov", "movie", "mp3", "mp4", "mp4a", "mpeg", "mpg", "mpga", "msgpack", "mxu", "nef", "npx", "numbers", "nupkg", "o", "oga", "ogg", "ogv", "otf", "pages", "pbm", "pcx", "pdb", "pdf", "pea", "pgm", "pic", "png", "pnm", "pot", "potm", "potx", "ppa", "ppam", "ppm", "pps", "ppsm", "ppsx", "ppt", "pptm", "pptx", "psd", "pya", "pyc", "pyo", "pyv", "qt", "rar", "ras", "raw", "resources", "rgb", "rip", "rlc", "rmf", "rmvb", "rpm", "rtf", "rz", "s3m", "s7z", "scpt", "sgi", "shar", "sig", "sil", "sketch", "slk", "smv", "snap", "snk", "so", "stl", "sub", "suo", "swf", "tar", "tbz2", "tbz", "tga", "tgz", "thmx", "tif", "tiff", "tlz", "ttc", "ttf", "txz", "udf", "uvh", "uvi", "uvm", "uvp", "uvs", "uvu", "viv", "vob", "war", "wav", "wax", "wbmp", "wdp", "weba", "webm", "webp", "whl", "wim", "wm", "wma", "wmv", "wmx", "woff2", "woff", "wrm", "wvx", "xbm", "xif", "xla", "xlam", "xls", "xlsb", "xlsm", "xlsx", "xlt", "xltm", "xltx", "xm", "xmind", "xpi", "xpm", "xwd", "xz", "z", "zip", "zipx"];
})();
// 将整个代码块作为一个匿名函数立即执行，保持作用域隔离
(function() {
  // 定义 FileEditor 类
  var FileEditor,
    // 定义 bind 函数，用于绑定函数的 this 上下文
    bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; },
    // 定义 extend 函数，用于继承父类的属性和方法
    extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
    // 定义 hasProp 对象，用于检查对象是否包含指定属性
    hasProp = {}.hasOwnProperty;
  
  // 定义 FileEditor 类，继承自 superClass
  FileEditor = (function(superClass) {
    // 继承 superClass 的属性和方法
    extend(FileEditor, superClass);
    
    // 定义 FileEditor 类的构造函数，接受 inner_path1 参数
    function FileEditor(inner_path1) {
      // 初始化 inner_path 属性
      this.inner_path = inner_path1;
      // 绑定 save 方法的 this 上下文
      this.save = bind(this.save, this);
      // 绑定 handleSaveClick 方法的 this 上下文
      this.handleSaveClick = bind(this.handleSaveClick, this);
      // 绑定 handleSidebarButtonClick 方法的 this 上下文
      this.handleSidebarButtonClick = bind(this.handleSidebarButtonClick, this);
      // 绑定 foldJson 方法的 this 上下文
      this.foldJson = bind(this.foldJson, this);
      // 绑定 storeCmNode 方法的 this 上下文
      this.storeCmNode = bind(this.storeCmNode, this);
      // 绑定 isModified 方法的 this 上下文
      this.isModified = bind(this.isModified, this);
      // 初始化 need_update 属性为 true
      this.need_update = true;
      // 初始化 on_loaded 属性为一个 Promise 对象
      this.on_loaded = new Promise();
      // 初始化 is_loading 属性为 false
      this.is_loading = false;
      // 初始化 content 属性为空字符串
      this.content = "";
      // 初始化 node_cm 属性为 null
      this.node_cm = null;
      // 初始化 cm 属性为 null
      this.cm = null;
      // 初始化 error 属性为 null
      this.error = null;
      // 初始化 is_loaded 属性为 false
      this.is_loaded = false;
      // 初始化 is_modified 属性为 false
      this.is_modified = false;
      // 初始化 is_saving 属性为 false
      this.is_saving = false;
      // 初始化 mode 属性为 "Loading"
      this.mode = "Loading";
    }
    # 更新文件内容的方法
    FileEditor.prototype.update = function() {
      # 定义变量，判断是否需要加载文件内容
      var is_required;
      is_required = Page.url_params.get("edit_mode") !== "new";
      # 调用 Page.cmd 方法，获取文件内容
      return Page.cmd("fileGet", {
        inner_path: this.inner_path,
        required: is_required
      }, (function(_this) {
        return function(res) {
          # 如果返回结果中包含错误信息，则将错误信息赋给当前对象的 error 属性，并将错误信息赋给内容
          if (res != null ? res.error : void 0) {
            _this.error = res.error;
            _this.content = res.error;
            _this.log("Error loading: " + _this.error);
          } else {
            # 如果返回结果存在
            if (res) {
              _this.content = res;
            } else {
              # 如果返回结果不存在，则将内容置为空字符串，并将模式设置为 "Create"
              _this.content = "";
              _this.mode = "Create";
            }
          }
          # 如果内容为空，则清除编辑器的历史记录
          if (!_this.content) {
            _this.cm.getDoc().clearHistory();
          }
          # 设置编辑器的内容为获取到的内容
          _this.cm.setValue(_this.content);
          # 如果没有错误，则将 is_loaded 属性设置为 true
          if (!_this.error) {
            _this.is_loaded = true;
          }
          # 调用 Page.projector.scheduleRender 方法，进行页面渲染
          return Page.projector.scheduleRender();
        };
      })(this));
    };

    # 判断文件内容是否被修改
    FileEditor.prototype.isModified = function() {
      return this.content !== this.cm.getValue();
    };

    # 存储 CodeMirror 编辑器节点的方法
    FileEditor.prototype.storeCmNode = function(node) {
      return this.node_cm = node;
    };

    # 获取文件类型的方法
    FileEditor.prototype.getMode = function(inner_path) {
      # 获取文件扩展名
      var ext, types;
      ext = inner_path.split(".").pop();
      # 定义文件类型与扩展名的对应关系
      types = {
        "py": "python",
        "json": "application/json",
        "js": "javascript",
        "coffee": "coffeescript",
        "html": "htmlmixed",
        "htm": "htmlmixed",
        "php": "htmlmixed",
        "rs": "rust",
        "css": "css",
        "md": "markdown",
        "xml": "xml",
        "svg": "xml"
      };
      # 返回对应的文件类型
      return types[ext];
    };
    // 定义 FileEditor 对象的 foldJson 方法，用于折叠 JSON 数据
    FileEditor.prototype.foldJson = function(from, to) {
      var count, e, endToken, internal, parsed, prevLine, startToken, toParse;
      this.log("foldJson", from, to);
      startToken = '{';  // 设置起始标记为 {
      endToken = '}';  // 设置结束标记为 }
      prevLine = this.cm.getLine(from.line);  // 获取折叠起始行的内容
      if (prevLine.lastIndexOf('[') > prevLine.lastIndexOf('{')) {  // 判断起始行中 [ 和 { 的位置，决定起始和结束标记
        startToken = '[';
        endToken = ']';
      }
      internal = this.cm.getRange(from, to);  // 获取折叠区域的内容
      toParse = startToken + internal + endToken;  // 拼接成完整的 JSON 字符串
      try {
        parsed = JSON.parse(toParse);  // 尝试解析 JSON 字符串
        count = Object.keys(parsed).length;  // 获取解析后的对象的属性数量
      } catch (error) {
        e = error;  // 捕获解析错误
        null;
      }
      if (count) {
        return "\u21A4" + count + "\u21A6";  // 如果有属性数量，返回箭头符号和属性数量
      } else {
        return "\u2194";  // 如果没有属性数量，返回双向箭头符号
      }
    };

    // 定义 FileEditor 对象的 createCodeMirror 方法，用于创建 CodeMirror 编辑器
    FileEditor.prototype.createCodeMirror = function() {
      var mode, options;
      mode = this.getMode(this.inner_path);  // 获取编辑器模式
      this.log("Creating CodeMirror", this.inner_path, mode);
      options = {
        value: "Loading...",  // 设置默认值为 "Loading..."
        mode: mode,  // 设置编辑器模式
        lineNumbers: true,  // 显示行号
        styleActiveLine: true,  // 高亮当前行
        matchBrackets: true,  // 匹配括号
        keyMap: "sublime",  // 设置键盘映射
        theme: "mdn-like",  // 设置主题样式
        extraKeys: {
          "Ctrl-Space": "autocomplete"  // 设置额外的快捷键
        },
        foldGutter: true,  // 启用折叠功能
        gutters: ["CodeMirror-linenumbers", "CodeMirror-foldgutter"]  // 设置编辑器 gutter
      };
      if (mode === "application/json") {  // 如果是 JSON 模式
        options.gutters.unshift("CodeMirror-lint-markers");  // 在 gutter 中添加 lint 标记
        options.lint = true;  // 启用 lint 功能
        options.foldOptions = {
          widget: this.foldJson  // 设置折叠选项的小部件为 foldJson 方法
        };
      }
      this.cm = CodeMirror(this.node_cm, options);  // 创建 CodeMirror 编辑器
      return this.cm.on("changes", (function(_this) {  // 监听编辑器内容变化事件
        return function(changes) {
          if (_this.is_loaded && !_this.is_modified) {  // 如果已加载且未修改
            _this.is_modified = true;  // 标记为已修改
            return Page.projector.scheduleRender();  // 调度重新渲染页面
          }
        };
      })(this));
    };
    # 加载编辑器的方法
    FileEditor.prototype.loadEditor = function() {
      var script;
      # 如果没有正在加载的情况下
      if (!this.is_loading) {
        # 在 head 标签末尾插入样式表链接
        document.getElementsByTagName("head")[0].insertAdjacentHTML("beforeend", "<link rel=\"stylesheet\" href=\"codemirror/all.css\" />");
        # 创建 script 元素
        script = document.createElement('script');
        # 设置 script 的 src 属性
        script.src = "codemirror/all.js";
        # 当 script 加载完成时执行回调函数
        script.onload = (function(_this) {
          return function() {
            # 创建 CodeMirror 编辑器
            _this.createCodeMirror();
            # 解决加载完成的 Promise
            return _this.on_loaded.resolve();
          };
        })(this);
        # 将 script 元素添加到 head 标签中
        document.head.appendChild(script);
      }
      # 返回加载完成的 Promise
      return this.on_loaded;
    };

    # 处理侧边栏按钮点击的方法
    FileEditor.prototype.handleSidebarButtonClick = function() {
      # 切换侧边栏状态
      Page.is_sidebar_closed = !Page.is_sidebar_closed;
      # 返回 false 阻止默认行为
      return false;
    };

    # 处理保存按钮点击的方法
    FileEditor.prototype.handleSaveClick = function() {
      var mark, num_errors;
      # 获取编辑器中的错误标记数量
      num_errors = ((function() {
        var i, len, ref, results;
        ref = Page.file_editor.cm.getAllMarks();
        results = [];
        for (i = 0, len = ref.length; i < len; i++) {
          mark = ref[i];
          if (mark.className === "CodeMirror-lint-mark-error") {
            results.push(mark);
          }
        }
        return results;
      })()).length;
      # 如果存在错误标记
      if (num_errors > 0) {
        # 弹出警告提示框，询问是否保存
        Page.cmd("wrapperConfirm", ["<b>Warning:</b> The file looks invalid.", "Save anyway"], this.save);
      } else {
        # 否则直接保存
        this.save();
      }
      # 返回 false 阻止默认行为
      return false;
    };
    # 定义文件编辑器的保存方法
    FileEditor.prototype.save = function() {
      # 调度页面重新渲染
      Page.projector.scheduleRender();
      # 设置正在保存标志为 true
      this.is_saving = true;
      # 调用 Page.cmd 方法执行文件写入操作
      return Page.cmd("fileWrite", [this.inner_path, Text.fileEncode(this.cm.getValue())], (function(_this) {
        return function(res) {
          # 保存完成后，将正在保存标志设置为 false
          _this.is_saving = false;
          # 如果保存出现错误，则显示错误通知
          if (res.error) {
            Page.cmd("wrapperNotification", ["error", "Error saving " + res.error]);
          } else {
            # 保存成功后，设置保存完成标志为 true
            _this.is_save_done = true;
            # 2秒后将保存完成标志设置为 false，并调度页面重新渲染
            setTimeout((function() {
              _this.is_save_done = false;
              return Page.projector.scheduleRender();
            }), 2000);
            # 更新文件内容，并将修改标志设置为 false
            _this.content = _this.cm.getValue();
            _this.is_modified = false;
            # 如果当前模式为"Create"，则将模式设置为"Edit"
            if (_this.mode === "Create") {
              _this.mode = "Edit";
            }
            # 设置文件列表需要更新的标志为 true
            Page.file_list.need_update = true;
          }
          # 调度页面重新渲染
          return Page.projector.scheduleRender();
        };
      })(this));
    };
    // 定义 FileEditor 对象的 render 方法
    FileEditor.prototype.render = function() {
      // 如果需要更新
      var ref;
      if (this.need_update) {
        // 加载编辑器并更新
        this.loadEditor().then((function(_this) {
          return function() {
            return _this.update();
          };
        })(this));
        // 将 need_update 标记设为 false
        this.need_update = false;
      }
      // 返回一个 div 元素，包含编辑器相关的属性和子元素
      return h("div.editor", {
        // 创建后执行 storeCmNode 方法
        afterCreate: this.storeCmNode,
        // 设置类名
        classes: {
          error: this.error,
          loaded: this.is_loaded
        }
      }, [
        // 创建一个侧边栏按钮
        h("a.sidebar-button", {
          href: "#Sidebar",
          onclick: this.handleSidebarButtonClick
        }, h("span", "\u2039")), 
        // 创建编辑器头部
        h("div.editor-head", [
          // 如果 mode 为 "Edit" 或 "Create"，则创建保存按钮
          (ref = this.mode) === "Edit" || ref === "Create" ? h("a.save.button", {
            href: "#Save",
            classes: {
              loading: this.is_saving,
              done: this.is_save_done,
              disabled: !this.is_modified
            },
            onclick: this.handleSaveClick
          }, this.is_save_done ? "Save: done!" : "Save") : void 0, 
          // 显示编辑器模式和路径
          h("span.title", this.mode, ": ", this.inner_path)
        ]), 
        // 如果有错误，则显示错误消息和查看链接
        this.error ? h("div.error-message", h("h2", "Unable to load the file: " + this.error), h("a", {
          href: Page.file_list.getHref(this.inner_path)
        }, "View in browser")) : void 0
      ]);
    };

    // 导出 FileEditor 对象
    return FileEditor;

  })(Class);

  // 将 FileEditor 对象赋值给 window 对象的属性
  window.FileEditor = FileEditor;
// 将整个代码块封装在一个立即执行函数表达式中，避免变量污染全局作用域
(function() {
  // 定义 FileItemList 类
  var FileItemList,
    // 定义 bind 函数，用于绑定函数的上下文
    bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; },
    // 定义 extend 函数，用于实现继承
    extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
    // 定义 hasProp 对象，用于检查对象是否包含特定属性
    hasProp = {}.hasOwnProperty;
  
  // FileItemList 类继承自 superClass
  FileItemList = (function(superClass) {
    // 使用 extend 函数实现继承
    extend(FileItemList, superClass);
    
    // FileItemList 类的构造函数
    function FileItemList(inner_path1) {
      // 初始化内部路径
      this.inner_path = inner_path1;
      // 绑定 sort 方法的上下文为当前对象
      this.sort = bind(this.sort, this);
      // 绑定 getOptionalInfo 方法的上下文为当前对象
      this.getOptionalInfo = bind(this.getOptionalInfo, this);
      // 绑定 hasPermissionDelete 方法的上下文为当前对象
      this.hasPermissionDelete = bind(this.hasPermissionDelete, this);
      // 绑定 isAdded 方法的上下文为当前对象
      this.isAdded = bind(this.isAdded, this);
      // 绑定 isModified 方法的上下文为当前对象
      this.isModified = bind(this.isModified, this);
      // 绑定 getFileType 方法的上下文为当前对象
      this.getFileType = bind(this.getFileType, this);
      // 绑定 addOptionalFilesToItems 方法的上下文为当前对象
      this.addOptionalFilesToItems = bind(this.addOptionalFilesToItems, this);
      // 绑定 updateOptionalFiles 方法的上下文为当前对象
      this.updateOptionalFiles = bind(this.updateOptionalFiles, this);
      // 绑定 updateAddedFiles 方法的上下文为当前对象
      this.updateAddedFiles = bind(this.updateAddedFiles, this);
      // 绑定 updateModifiedFiles 方法的上下文为当前对象
      this.updateModifiedFiles = bind(this.updateModifiedFiles, this);
      // 初始化 items 数组
      this.items = [];
      // 初始化 updating 为 false
      this.updating = false;
      // 初始化 files_modified 为空对象
      this.files_modified = {};
      // 初始化 dirs_modified 为空对象
      this.dirs_modified = {};
      // 初始化 files_added 为空对象
      this.files_added = {};
      // 初始化 dirs_added 为空对象
      this.dirs_added = {};
      // 初始化 files_optional 为空对象
      this.files_optional = {};
      // 初始化 items_by_name 为空对象
      this.items_by_name = {};
    }
    // 更新文件列表的方法
    FileItemList.prototype.update = function(cb) {
      // 标记为正在更新
      this.updating = true;
      // 记录更新目录列表的开始
      this.logStart("Updating dirlist");
      // 调用 Page.cmd 方法，请求获取目录列表
      return Page.cmd("dirList", {
        inner_path: this.inner_path,  // 内部路径
        stats: true  // 是否包含统计信息
      }, (function(_this) {
        return function(res) {
          var i, len, pattern_ignore, ref, ref1, ref2, ref3, row;
          // 如果返回结果中包含错误信息，则记录错误
          if (res.error) {
            _this.error = res.error;
          } else {
            _this.error = null;
            // 生成忽略文件的正则表达式
            pattern_ignore = RegExp("^" + ((ref = Page.site_info.content) != null ? ref.ignore : void 0));
            // 清空当前文件列表
            _this.items.splice(0, _this.items.length);
            _this.items_by_name = {};
            // 遍历返回的文件列表
            for (i = 0, len = res.length; i < len; i++) {
              row = res[i];
              // 获取文件类型
              row.type = _this.getFileType(row);
              // 获取文件的内部路径
              row.inner_path = _this.inner_path + row.name;
              // 如果存在需要忽略的文件，并且匹配忽略规则，则标记为被忽略
              if (((ref1 = Page.site_info.content) != null ? ref1.ignore : void 0) && row.inner_path.match(pattern_ignore)) {
                row.ignored = true;
              }
              // 将文件添加到文件列表中
              _this.items.push(row);
              // 根据文件名建立索引
              _this.items_by_name[row.name] = row;
            }
            // 对文件列表进行排序
            _this.sort();
          }
          // 如果存在站点信息，并且拥有自定义设置，则更新添加的文件
          if ((ref2 = Page.site_info) != null ? (ref3 = ref2.settings) != null ? ref3.own : void 0 : void 0) {
            _this.updateAddedFiles();
          }
          // 更新可选文件
          return _this.updateOptionalFiles(function() {
            // 标记更新结束
            _this.updating = false;
            // 如果存在回调函数，则执行回调
            if (typeof cb === "function") {
              cb();
            }
            // 记录更新目录列表的结束
            _this.logEnd("Updating dirlist", _this.inner_path);
            // 调度重新渲染页面
            Page.projector.scheduleRender();
            // 更新修改的文件，并调度重新渲染页面
            return _this.updateModifiedFiles(function() {
              return Page.projector.scheduleRender();
            });
          });
        };
      })(this));
    };
    # 更新修改过的文件列表
    FileItemList.prototype.updateModifiedFiles = function(cb) {
      # 调用 Page.cmd 方法，请求站点修改过的文件列表
      return Page.cmd("siteListModifiedFiles", [], (function(_this) {
        return function(res) {
          # 初始化修改过的文件和目录列表
          _this.files_modified = {};
          _this.dirs_modified = {};
          # 遍历返回的修改过的文件列表
          ref = res.modified_files;
          for (i = 0, len = ref.length; i < len; i++) {
            inner_path = ref[i];
            # 将文件路径添加到修改过的文件列表中
            _this.files_modified[inner_path] = true;
            dir_inner_path = "";
            dir_parts = inner_path.split("/");
            # 遍历文件路径的目录部分
            ref1 = dir_parts.slice(0, -1);
            for (j = 0, len1 = ref1.length; j < len1; j++) {
              dir_part = ref1[j];
              # 构建目录内部路径
              if (dir_inner_path) {
                dir_inner_path += "/" + dir_part;
              } else {
                dir_inner_path = dir_part;
              }
              # 将目录内部路径添加到修改过的目录列表中
              _this.dirs_modified[dir_inner_path] = true;
            }
          }
          # 执行回调函数
          return typeof cb === "function" ? cb() : void 0;
        };
      })(this));
    };
    // 更新已添加文件列表的方法
    FileItemList.prototype.updateAddedFiles = function() {
      // 调用 Page.cmd 方法，获取 content.json 文件的内容
      return Page.cmd("fileGet", "content.json", (function(_this) {
        return function(res) {
          var content, dirs_content, file, file_name, i, j, len, len1, match, pattern, ref, ref1, results;
          // 如果获取不到内容，则返回 false
          if (!res) {
            return false;
          }
          // 解析获取到的 JSON 内容
          content = JSON.parse(res);
          // 如果内容中没有 files 字段，则返回 false
          if (content.files == null) {
            return false;
          }
          // 初始化已添加文件列表为空对象
          _this.files_added = {};
          // 遍历文件列表
          ref = _this.items;
          for (i = 0, len = ref.length; i < len; i++) {
            file = ref[i];
            // 如果文件名为 content.json 或者是文件夹，则跳过
            if (file.name === "content.json" || file.is_dir) {
              continue;
            }
            // 如果文件列表中没有当前文件，则将其添加到已添加文件列表中
            if (!content.files[_this.inner_path + file.name]) {
              _this.files_added[_this.inner_path + file.name] = true;
            }
          }
          // 初始化已添加文件夹列表为空对象
          _this.dirs_added = {};
          dirs_content = {};
          // 遍历文件列表和可选文件列表中的文件名
          for (file_name in Object.assign({}, content.files, content.files_optional)) {
            // 如果文件名不以当前路径开头，则跳过
            if (!file_name.startsWith(_this.inner_path)) {
              continue;
            }
            // 使用正则表达式匹配文件名，获取文件夹名
            pattern = new RegExp(_this.inner_path + "(.*?)/");
            match = file_name.match(pattern);
            // 如果匹配不到，则跳过
            if (!match) {
              continue;
            }
            // 将文件夹名添加到文件夹内容列表中
            dirs_content[match[1]] = true;
          }
          // 遍历文件列表
          ref1 = _this.items;
          results = [];
          for (j = 0, len1 = ref1.length; j < len1; j++) {
            file = ref1[j];
            // 如果不是文件夹，则跳过
            if (!file.is_dir) {
              continue;
            }
            // 如果文件夹内容列表中没有当前文件夹，则将其添加到已添加文件夹列表中
            if (!dirs_content[file.name]) {
              results.push(_this.dirs_added[_this.inner_path + file.name] = true);
            } else {
              results.push(void 0);
            }
          }
          return results;
        };
      })(this));
    };
    # 更新可选文件列表的方法
    FileItemList.prototype.updateOptionalFiles = function(cb) {
      # 调用 Page.cmd 方法，请求可选文件列表
      return Page.cmd("optionalFileList", {
        filter: ""
      }, (function(_this) {
        return function(res) {
          # 初始化可选文件对象
          var i, len, optional_file;
          _this.files_optional = {};
          # 遍历返回的可选文件列表，将文件路径作为键，文件对象作为值存入文件对象中
          for (i = 0, len = res.length; i < len; i++) {
            optional_file = res[i];
            _this.files_optional[optional_file.inner_path] = optional_file;
          }
          # 将可选文件添加到文件项中
          _this.addOptionalFilesToItems();
          # 如果回调函数是一个函数，则执行回调函数
          return typeof cb === "function" ? cb() : void 0;
        };
      })(this));
    };
    # 将可选文件添加到项目列表中
    FileItemList.prototype.addOptionalFilesToItems = function() {
      # 初始化是否添加标志为 false
      var is_added = false;
      # 获取可选文件列表
      ref = this.files_optional;
      # 遍历可选文件列表
      for (inner_path in ref) {
        optional_file = ref[inner_path];
        # 判断可选文件的路径是否以当前项目路径开头
        if (optional_file.inner_path.startsWith(this.inner_path)) {
          # 如果可选文件是当前项目的子目录
          if (this.getDirectory(optional_file.inner_path) === this.inner_path) {
            # 获取文件名
            file_name = this.getFileName(optional_file.inner_path);
            # 如果项目中不存在该文件
            if (!this.items_by_name[file_name]) {
              # 创建文件对象并添加到项目列表中
              row = {
                "name": file_name,
                "type": "file",
                "optional_empty": true,
                "size": optional_file.size,
                "is_dir": false,
                "inner_path": optional_file.inner_path
              };
              this.items.push(row);
              this.items_by_name[file_name] = row;
              # 更新是否添加标志
              is_added = true;
            }
          } else {
            # 获取目录名
            dir_name = (ref1 = optional_file.inner_path.replace(this.inner_path, "").match(/(.*?)\//, "")) != null ? ref1[1] : void 0;
            # 如果项目中不存在该目录
            if (dir_name && !this.items_by_name[dir_name]) {
              # 创建目录对象并添加到项目列表中
              row = {
                "name": dir_name,
                "type": "dir",
                "optional_empty": true,
                "size": 0,
                "is_dir": true,
                "inner_path": optional_file.inner_path
              };
              this.items.push(row);
              this.items_by_name[dir_name] = row;
              # 更新是否添加标志
              is_added = true;
            }
          }
        }
      }
      # 如果有文件被添加，则进行排序
      if (is_added) {
        return this.sort();
      }
    };

    # 获取文件类型
    FileItemList.prototype.getFileType = function(file) {
      # 如果是目录，返回类型为 "dir"，否则返回 "unknown"
      if (file.is_dir) {
        return "dir";
      } else {
        return "unknown";
      }
    };
    # 定义 FileItemList 对象的 getDirectory 方法，用于获取内部路径的目录部分
    FileItemList.prototype.getDirectory = function(inner_path) {
      # 如果内部路径包含斜杠，则返回斜杠之前的部分作为目录
      if (inner_path.indexOf("/") !== -1) {
        return inner_path.replace(/^(.*\/)(.*?)$/, "$1");
      } else {
        return "";
      }
    };
    
    # 定义 FileItemList 对象的 getFileName 方法，用于获取内部路径的文件名部分
    FileItemList.prototype.getFileName = function(inner_path) {
      return inner_path.replace(/^(.*\/)(.*?)$/, "$2");
    };
    
    # 定义 FileItemList 对象的 isModified 方法，用于判断指定内部路径的文件或目录是否被修改过
    FileItemList.prototype.isModified = function(inner_path) {
      return this.files_modified[inner_path] || this.dirs_modified[inner_path];
    };
    
    # 定义 FileItemList 对象的 isAdded 方法，用于判断指定内部路径的文件或目录是否被添加过
    FileItemList.prototype.isAdded = function(inner_path) {
      return this.files_added[inner_path] || this.dirs_added[inner_path];
    };
    
    # 定义 FileItemList 对象的 hasPermissionDelete 方法，用于判断指定文件是否有删除权限
    FileItemList.prototype.hasPermissionDelete = function(file) {
      var optional_info, ref, ref1, ref2;
      # 如果文件类型为目录或父级目录，则无法删除
      if ((ref = file.type) === "dir" || ref === "parent") {
        return false;
      }
      # 如果文件内部路径为 "content.json"，则无法删除
      if (file.inner_path === "content.json") {
        return false;
      }
      # 获取文件的可选信息，如果已下载百分比大于 0，则有删除权限
      optional_info = this.getOptionalInfo(file.inner_path);
      if (optional_info && optional_info.downloaded_percent > 0) {
        return true;
      } else {
        # 否则，根据页面信息中的设置判断是否有删除权限
        return (ref1 = Page.site_info) != null ? (ref2 = ref1.settings) != null ? ref2.own : void 0 : void 0;
      }
    };
    
    # 定义 FileItemList 对象的 getOptionalInfo 方法，用于获取指定内部路径的可选信息
    FileItemList.prototype.getOptionalInfo = function(inner_path) {
      return this.files_optional[inner_path];
    };
    
    # 定义 FileItemList 对象的 sort 方法，用于对文件列表进行排序
    FileItemList.prototype.sort = function() {
      return this.items.sort(function(a, b) {
        return (b.is_dir - a.is_dir) || a.name.localeCompare(b.name);
      });
    };
    
    # 导出 FileItemList 对象
    return FileItemList;
    
    # 将 FileItemList 对象绑定到全局对象 window 上
    })(Class);
    
    window.FileItemList = FileItemList;
# 调用匿名函数，设置 this 指向全局对象
}).call(this);

# 定义 FileList 类
/* ---- FileList.coffee ---- */
(function() {
  var FileList,
    # 定义 bind 函数，用于绑定函数的 this 指向
    bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; },
    # 定义 extend 函数，用于实现继承
    extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
    # 定义 hasProp 变量，用于判断对象是否包含指定属性
    hasProp = {}.hasOwnProperty,
    # 定义 indexOf 函数，用于获取数组中指定元素的索引
    indexOf = [].indexOf || function(item) { for (var i = 0, l = this.length; i < l; i++) { if (i in this && this[i] === item) return i; } return -1; };

  # 定义 FileList 类
  FileList = (function(superClass) {
    # 继承 superClass
    extend(FileList, superClass);

    # 判断是否全部选中
    FileList.prototype.isSelectedAll = function() {
      return false;
    };

    # 更新文件列表
    FileList.prototype.update = function() {
      return this.item_list.update((function(_this) {
        return function() {
          return document.body.classList.add("loaded");
        };
      })(this));
    };

    # 获取文件链接
    FileList.prototype.getHref = function(inner_path) {
      return "/" + this.site + "/" + inner_path;
    };

    # 获取文件列表链接
    FileList.prototype.getListHref = function(inner_path) {
      return "/list/" + this.site + "/" + inner_path;
    };

    # 获取编辑链接
    FileList.prototype.getEditHref = function(inner_path, mode) {
      var href;
      if (mode == null) {
        mode = null;
      }
      href = this.url_root + "?file=" + inner_path;
      if (mode) {
        href += "&edit_mode=" + mode;
      }
      return href;
    };
    # 检查所选项目的数量和大小，并记录可选信息为空的数量
    FileList.prototype.checkSelectedItems = function() {
      var i, item, len, optional_info, ref, results;
      this.selected_items_num = 0;  # 初始化所选项目数量为0
      this.selected_items_size = 0;  # 初始化所选项目大小为0
      this.selected_optional_empty_num = 0;  # 初始化可选信息为空的数量为0
      ref = this.item_list.items;  # 获取项目列表
      results = [];  # 初始化结果数组
      for (i = 0, len = ref.length; i < len; i++) {  # 遍历项目列表
        item = ref[i];  # 获取当前项目
        if (this.selected[item.inner_path]) {  # 如果项目被选中
          this.selected_items_num += 1;  # 所选项目数量加1
          this.selected_items_size += item.size;  # 所选项目大小累加
          optional_info = this.item_list.getOptionalInfo(item.inner_path);  # 获取项目的可选信息
          if (optional_info && !optional_info.downloaded_percent > 0) {  # 如果可选信息存在且下载百分比大于0
            results.push(this.selected_optional_empty_num += 1);  # 可选信息为空的数量加1
          } else {
            results.push(void 0);  # 否则结果数组添加undefined
          }
        } else {
          results.push(void 0);  # 否则结果数组添加undefined
        }
      }
      return results;  # 返回结果数组
    };

    # 处理菜单创建点击事件
    FileList.prototype.handleMenuCreateClick = function() {
      this.menu_create.items = [];  # 清空菜单创建的项目
      this.menu_create.items.push(["File", this.handleNewFileClick]);  # 添加文件选项
      this.menu_create.items.push(["Directory", this.handleNewDirectoryClick]);  # 添加目录选项
      this.menu_create.toggle();  # 切换菜单显示状态
      return false;  # 返回false
    };

    # 处理新建文件点击事件
    FileList.prototype.handleNewFileClick = function() {
      Page.cmd("wrapperPrompt", "New file name:", (function(_this) {  # 调用wrapperPrompt方法，提示输入新文件名
        return function(file_name) {
          return window.top.location.href = _this.getEditHref(_this.inner_path + file_name, "new");  # 在顶层窗口中打开新文件编辑页面
        };
      })(this));
      return false;  # 返回false
    };

    # 处理新建目录点击事件
    FileList.prototype.handleNewDirectoryClick = function() {
      Page.cmd("wrapperPrompt", "New directory name:", (function(_this) {  # 调用wrapperPrompt方法，提示输入新目录名
        return function(res) {
          return alert("directory name " + res);  # 弹出新目录名
        };
      })(this));
      return false;  # 返回false
    };

    # 处理选择点击事件
    FileList.prototype.handleSelectClick = function(e) {
      return false;  # 返回false
    };
    # 处理选择结束事件的方法
    FileList.prototype.handleSelectEnd = function(e) {
      # 移除鼠标抬起事件监听器
      document.body.removeEventListener('mouseup', this.handleSelectEnd);
      # 清空选择操作
      return this.select_action = null;
    };
    
    # 处理鼠标按下事件的方法
    FileList.prototype.handleSelectMousedown = function(e) {
      var inner_path;
      # 获取当前元素的内部路径
      inner_path = e.currentTarget.attributes.inner_path.value;
      # 如果已经选择了该路径，则取消选择，否则进行选择
      if (this.selected[inner_path]) {
        delete this.selected[inner_path];
        this.select_action = "deselect";
      } else {
        this.selected[inner_path] = true;
        this.select_action = "select";
      }
      # 检查已选择的项目
      this.checkSelectedItems();
      # 添加鼠标抬起事件监听器
      document.body.addEventListener('mouseup', this.handleSelectEnd);
      # 阻止事件冒泡
      e.stopPropagation();
      # 调度页面重新渲染
      Page.projector.scheduleRender();
      return false;
    };
    
    # 处理鼠标进入行事件的方法
    FileList.prototype.handleRowMouseenter = function(e) {
      var inner_path;
      # 如果鼠标按下并且有选择操作，则根据选择操作进行选择或取消选择
      if (e.buttons && this.select_action) {
        inner_path = e.target.attributes.inner_path.value;
        if (this.select_action === "select") {
          this.selected[inner_path] = true;
        } else {
          delete this.selected[inner_path];
        }
        # 检查已选择的项目
        this.checkSelectedItems();
        # 调度页面重新渲染
        Page.projector.scheduleRender();
      }
      return false;
    };
    
    # 处理取消选择操作的方法
    FileList.prototype.handleSelectbarCancel = function() {
      # 清空已选择的项目
      this.selected = {};
      # 检查已选择的项目
      this.checkSelectedItems();
      # 调度页面重新渲染
      Page.projector.scheduleRender();
      return false;
    };
    // 处理选择栏中的删除操作，根据参数决定是否删除可选文件
    FileList.prototype.handleSelectbarDelete = function(e, remove_optional) {
      // 如果未指定是否删除可选文件，默认为不删除
      if (remove_optional == null) {
        remove_optional = false;
      }
      // 遍历选中的文件
      for (inner_path in this.selected) {
        // 获取文件的可选信息
        optional_info = this.item_list.getOptionalInfo(inner_path);
        // 从选中列表中删除文件
        delete this.selected[inner_path];
        // 如果文件有可选信息且不删除可选文件，则调用删除可选文件的命令
        if (optional_info && !remove_optional) {
          Page.cmd("optionalFileDelete", inner_path);
        } else {
          // 否则调用删除文件的命令
          Page.cmd("fileDelete", inner_path);
        }
      }
      // 设置需要更新标志为 true
      this.need_update = true;
      // 调度重新渲染页面
      Page.projector.scheduleRender();
      // 检查选中的文件项
      this.checkSelectedItems();
      // 返回 false，阻止默认行为
      return false;
    };

    // 处理选择栏中的删除可选文件操作
    FileList.prototype.handleSelectbarRemoveOptional = function(e) {
      // 调用 handleSelectbarDelete 方法，并指定删除可选文件
      return this.handleSelectbarDelete(e, true);
    };

    // 渲染选择栏
    FileList.prototype.renderSelectbar = function() {
      // 返回一个 div 元素，包含选中文件的信息和操作按钮
      return h("div.selectbar", {
        classes: {
          // 根据选中文件数量决定是否显示选择栏
          visible: this.selected_items_num > 0
        }
      }, [
        "Selected:", h("span.info", [h("span.num", this.selected_items_num + " files"), h("span.size", "(" + (Text.formatSize(this.selected_items_size)) + ")")]), h("div.actions", [
          // 如果有可选文件，显示删除并移除可选按钮，否则显示删除按钮
          this.selected_optional_empty_num > 0 ? h("a.action.delete.remove_optional", {
            href: "#",
            onclick: this.handleSelectbarRemoveOptional
          }, "Delete and remove optional") : h("a.action.delete", {
            href: "#",
            onclick: this.handleSelectbarDelete
          }, "Delete")
        ]),
        // 显示取消按钮
        h("a.cancel.link", {
          href: "#",
          onclick: this.handleSelectbarCancel
        }, "Cancel")
      ]);
    };
    // 渲染文件列表的表头
    FileList.prototype.renderHead = function() {
      var i, inner_path_parent, len, parent_dir, parent_links, ref;
      parent_links = [];
      inner_path_parent = "";
      ref = this.inner_path.split("/");
      // 遍历内部路径，生成父级目录链接
      for (i = 0, len = ref.length; i < len; i++) {
        parent_dir = ref[i];
        if (!parent_dir) {
          continue;
        }
        if (inner_path_parent) {
          inner_path_parent += "/";
        }
        inner_path_parent += "" + parent_dir;
        parent_links.push([
          " / ", h("a", {
            href: this.getListHref(inner_path_parent)
          }, parent_dir)
        ]);
      }
      // 返回包含父级目录链接的表头
      return h("div.tr.thead", h("div.td.full", h("a", {
        href: this.getListHref("")
      }, "root"), parent_links));
    };

    // 渲染文件项的复选框
    FileList.prototype.renderItemCheckbox = function(item) {
      // 如果没有删除权限，则返回空数组
      if (!this.item_list.hasPermissionDelete(item)) {
        return [" "];
      }
      // 返回包含复选框的链接
      return h("a.checkbox-outer", {
        href: "#Select",
        onmousedown: this.handleSelectMousedown,
        onclick: this.handleSelectClick,
        inner_path: item.inner_path
      }, h("span.checkbox"));
    };

    // 渲染文件列表的所有项
    FileList.prototype.renderItems = function() {
      return [
        // 如果存在错误并且没有项并且不在更新中，则返回包含错误信息的表格行
        this.item_list.error && !this.item_list.items.length && !this.item_list.updating ? [
          h("div.tr", {
            key: "error"
          }, h("div.td.full.error", this.item_list.error))
        ] : void 0, 
        // 如果存在内部路径，则渲染父级目录项
        this.inner_path ? this.renderItem({
          "name": "..",
          type: "parent",
          size: 0
        }) : void 0, 
        // 渲染所有文件项
        this.item_list.items.map(this.renderItem)
      ];
    };

    // 渲染整个文件列表
    FileList.prototype.render = function() {
      // 如果需要更新，则执行更新操作，并将需要更新标志置为false
      if (this.need_update) {
        this.update();
        this.need_update = false;
        // 如果没有项，则返回空数组
        if (!this.item_list.items) {
          return [];
        }
      }
      // 返回包含选择栏、表头、文件项和页脚的文件列表
      return h("div.files", [this.renderSelectbar(), this.renderHead(), h("div.tbody", this.renderItems()), this.renderFoot()]);
    };
    # 返回 FileList 变量
    return FileList;

  })(Class);
  
  # 将 Class 传入 FileList 函数中，并将结果赋值给 window.FileList
  window.FileList = FileList;
// 调用匿名函数并将 this 绑定到全局对象
}).call(this);

// 定义 UiFileManager 类
/* ---- UiFileManager.coffee ---- */
(function() {
  var UiFileManager,
    // 定义辅助函数 bind
    bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; },
    // 定义继承函数 extend
    extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
    // 定义 hasOwnProperty 函数
    hasProp = {}.hasOwnProperty;

  // 将 maquette.h 赋值给全局对象 window 的属性 h
  window.h = maquette.h;

  // 定义 UiFileManager 类
  UiFileManager = (function(superClass) {
    // 继承 superClass
    extend(UiFileManager, superClass);

    // 定义 UiFileManager 构造函数
    function UiFileManager() {
      // 绑定 this，并返回绑定后的函数
      this.render = bind(this.render, this);
      this.createProjector = bind(this.createProjector, this);
      this.onRequest = bind(this.onRequest, this);
      this.checkBodyWidth = bind(this.checkBodyWidth, this);
      // 调用 superClass 的构造函数
      return UiFileManager.__super__.constructor.apply(this, arguments);
    }
    // 初始化 UiFileManager 对象的方法
    UiFileManager.prototype.init = function() {
      // 从 URL 查询参数中获取 site、address、inner_path 和 file 参数
      this.url_params = new URLSearchParams(window.location.search);
      this.list_site = this.url_params.get("site");
      this.list_address = this.url_params.get("address");
      this.list_inner_path = this.url_params.get("inner_path");
      this.editor_inner_path = this.url_params.get("file");
      // 创建 FileList 对象
      this.file_list = new FileList(this.list_site, this.list_inner_path);
      this.site_info = null;
      this.server_info = null;
      this.is_sidebar_closed = false;
      // 如果存在 editor_inner_path，则创建 FileEditor 对象
      if (this.editor_inner_path) {
        this.file_editor = new FileEditor(this.editor_inner_path);
      }
      // 在窗口即将关闭时触发事件，检查文件编辑器是否有未保存的修改
      window.onbeforeunload = (function(_this) {
        return function() {
          var ref;
          if ((ref = _this.file_editor) != null ? ref.isModified() : void 0) {
            return true;
          } else {
            return null;
          }
        };
      })(this);
      // 在窗口大小改变时触发事件，检查页面宽度
      window.onresize = (function(_this) {
        return function() {
          return _this.checkBodyWidth();
        };
      })(this);
      // 检查页面宽度
      this.checkBodyWidth();
      // 设置视口的宽度和初始缩放比例
      this.cmd("wrapperSetViewport", "width=device-width, initial-scale=0.8");
      // 获取服务器信息并保存到 server_info
      this.cmd("serverInfo", {}, (function(_this) {
        return function(server_info) {
          return _this.server_info = server_info;
        };
      })(this));
      // 获取站点信息并保存到 site_info
      return this.cmd("siteInfo", {}, (function(_this) {
        return function(site_info) {
          // 设置页面标题，根据站点信息和文件路径
          _this.cmd("wrapperSetTitle", "List: /" + _this.list_inner_path + " - " + site_info.content.title + " - ZeroNet");
          _this.site_info = site_info;
          // 如果存在文件编辑器，则根据站点设置设置编辑器的只读状态和模式
          if (_this.file_editor) {
            _this.file_editor.on_loaded.then(function() {
              _this.file_editor.cm.setOption("readOnly", !site_info.settings.own);
              return _this.file_editor.mode = site_info.settings.own ? "Edit" : "View";
            });
          }
          // 调度渲染
          return _this.projector.scheduleRender();
        };
      })(this));
    };
    // 检查页面宽度是否小于960像素，如果没有文件编辑器则返回false
    UiFileManager.prototype.checkBodyWidth = function() {
      var ref, ref1;
      if (!this.file_editor) {
        return false;
      }
      // 如果页面宽度小于960像素且侧边栏未关闭，则关闭侧边栏并调度重新渲染
      if (document.body.offsetWidth < 960 && !this.is_sidebar_closed) {
        this.is_sidebar_closed = true;
        return (ref = this.projector) != null ? ref.scheduleRender() : void 0;
      } else if (document.body.offsetWidth > 960 && this.is_sidebar_closed) {
        // 如果页面宽度大于960像素且侧边栏已关闭，则打开侧边栏并调度重新渲染
        this.is_sidebar_closed = false;
        return (ref1 = this.projector) != null ? ref1.scheduleRender() : void 0;
      }
    };

    // 处理请求消息的方法
    UiFileManager.prototype.onRequest = function(cmd, message) {
      // 如果命令是"setSiteInfo"，则更新站点信息并调度重新渲染
      if (cmd === "setSiteInfo") {
        this.site_info = message;
        RateLimitCb(1000, (function(_this) {
          return function(cb_done) {
            return _this.file_list.update(cb_done);
          };
        })(this));
        return this.projector.scheduleRender();
      } else if (cmd === "setServerInfo") {
        // 如果命令是"setServerInfo"，则更新服务器信息并调度重新渲染
        this.server_info = message;
        return this.projector.scheduleRender();
      } else {
        // 如果命令不是已知的，则记录未知的传入消息
        return this.log("Unknown incoming message:", cmd);
      }
    };

    // 创建并返回maquette投影器
    UiFileManager.prototype.createProjector = function() {
      this.projector = maquette.createProjector();
      return this.projector.replace($("#content"), this.render);
    };

    // 渲染方法，返回一个div元素
    UiFileManager.prototype.render = function() {
      return h("div.content#content", [
        h("div.manager", {
          classes: {
            // 根据条件添加不同的类
            editing: this.file_editor,
            sidebar_closed: this.is_sidebar_closed
          }
        }, [this.file_list.render(), this.file_editor ? this.file_editor.render() : void 0])
      ]);
    };

    // 将UiFileManager继承自ZeroFrame
    return UiFileManager;

  })(ZeroFrame);

  // 创建UiFileManager实例并调用createProjector方法
  window.Page = new UiFileManager();

  window.Page.createProjector();
# 调用一个匿名函数，并将 this 作为参数传入
}).call(this);
```