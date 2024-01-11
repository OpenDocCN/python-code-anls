# `ZeroNet\src\Ui\media\all.js`

```
# 定义一个匿名函数，用于限制函数的调用频率
(function() {
  # 定义一个空对象，用于存储函数的调用限制
  var limits = {};
  # 定义一个空对象，用于存储函数的延迟调用
  var call_after_interval = {};
  # 在全局对象 window 上定义 RateLimit 函数
  window.RateLimit = function(interval, fn) {
    # 如果函数的调用限制不存在
    if (!limits[fn]) {
      # 将函数的延迟调用设置为 false
      call_after_interval[fn] = false;
      # 调用函数
      fn();
      # 设置函数的调用限制，超过指定时间后执行函数
      return limits[fn] = setTimeout((function() {
        # 如果函数的延迟调用为 true，则执行函数
        if (call_after_interval[fn]) {
          fn();
        }
        # 删除函数的调用限制
        delete limits[fn];
        # 删除函数的延迟调用
        return delete call_after_interval[fn];
      }), interval);
    } else {
      # 如果函数的调用限制存在，则将函数的延迟调用设置为 true
      return call_after_interval[fn] = true;
    }
  };
}).call(this);

# 定义一个匿名函数，用于在全局对象 window 上定义 _ 函数
(function() {
  # 在全局对象 window 上定义 _ 函数，用于返回传入的参数
  window._ = function(s) {
    return s;
  };
}).call(this);

# 定义一个 ZeroWebsocket 类
(function() {
  # 定义一个 ZeroWebsocket 类
  var ZeroWebsocket = (function() {
    # 初始化 ZeroWebsocket 类
    function ZeroWebsocket(url) {
      # 绑定函数的上下文
      this.onCloseWebsocket = bind(this.onCloseWebsocket, this);
      this.onErrorWebsocket = bind(this.onErrorWebsocket, this);
      this.onOpenWebsocket = bind(this.onOpenWebsocket, this);
      this.log = bind(this.log, this);
      this.response = bind(this.response, this);
      this.route = bind(this.route, this);
      this.onMessage = bind(this.onMessage, this);
      # 设置 ZeroWebsocket 类的属性 url
      this.url = url;
      # 设置 ZeroWebsocket 类的属性 next_message_id
      this.next_message_id = 1;
      # 定义一个空对象，用于存储等待的回调函数
      this.waiting_cb = {};
      # 初始化 ZeroWebsocket 类
      this.init();
    }
    # 定义 ZeroWebsocket 类的方法 init
    ZeroWebsocket.prototype.init = function() {
      return this;
    };
    # 定义 ZeroWebsocket 类的方法 connect
    ZeroWebsocket.prototype.connect = function() {
      # 创建 WebSocket 对象，并设置相关事件处理函数
      this.ws = new WebSocket(this.url);
      this.ws.onmessage = this.onMessage;
      this.ws.onopen = this.onOpenWebsocket;
      this.ws.onerror = this.onErrorWebsocket;
      this.ws.onclose = this.onCloseWebsocket;
      # 设置连接状态为未连接
      this.connected = false;
      # 定义一个空数组，用于存储消息队列
      return this.message_queue = [];
    };
    // 当接收到消息时的处理函数
    ZeroWebsocket.prototype.onMessage = function(e) {
      // 解析收到的消息
      var message = JSON.parse(e.data);
      // 获取消息中的命令
      var cmd = message.cmd;
      // 如果命令是 "response"
      if (cmd === "response") {
        // 如果等待回调中存在该消息的回调函数
        if (this.waiting_cb[message.to] != null) {
          // 调用回调函数，并传入结果
          return this.waiting_cb[message.to](message.result);
        } else {
          // 否则记录日志，表示未找到对应的回调函数
          return this.log("Websocket callback not found:", message);
        }
      } 
      // 如果命令是 "ping"
      else if (cmd === "ping") {
        // 发送 "pong" 响应
        return this.response(message.id, "pong");
      } 
      // 其他情况
      else {
        // 调用路由函数，处理命令和消息
        return this.route(cmd, message);
      }
    };

    // 处理未知命令的函数
    ZeroWebsocket.prototype.route = function(cmd, message) {
      return this.log("Unknown command", message);
    };

    // 发送响应消息的函数
    ZeroWebsocket.prototype.response = function(to, result) {
      return this.send({
        "cmd": "response",
        "to": to,
        "result": result
      });
    };

    // 发送命令消息的函数
    ZeroWebsocket.prototype.cmd = function(cmd, params, cb) {
      // 如果参数为空，则设置为空对象
      if (params == null) {
        params = {};
      }
      // 如果回调函数为空，则设置为null
      if (cb == null) {
        cb = null;
      }
      // 发送命令消息，并传入回调函数
      return this.send({
        "cmd": cmd,
        "params": params
      }, cb);
    };

    // 发送消息的函数
    ZeroWebsocket.prototype.send = function(message, cb) {
      // 如果回调函数为空，则设置为null
      if (cb == null) {
        cb = null;
      }
      // 如果消息中没有id字段，则设置为下一个消息id，并递增
      if (message.id == null) {
        message.id = this.next_message_id;
        this.next_message_id += 1;
      }
      // 如果已连接到websocket服务器
      if (this.connected) {
        // 发送消息到服务器
        this.ws.send(JSON.stringify(message));
      } else {
        // 否则记录日志，表示未连接到服务器，将消息加入消息队列
        this.log("Not connected, adding message to queue");
        this.message_queue.push(message);
      }
      // 如果存在回调函数，则将回调函数存入等待回调字典中
      if (cb) {
        return this.waiting_cb[message.id] = cb;
      }
    };

    // 记录日志的函数
    ZeroWebsocket.prototype.log = function() {
      // 将日志信息输出到控制台
      var args = 1 <= arguments.length ? slice.call(arguments, 0) : [];
      return console.log.apply(console, ["[ZeroWebsocket]"].concat(slice.call(args)));
    };
    // 定义 ZeroWebsocket 对象的 onOpenWebsocket 方法，处理 WebSocket 连接打开事件
    ZeroWebsocket.prototype.onOpenWebsocket = function(e) {
      // 打印日志，表示连接已打开
      this.log("Open");
      // 设置连接状态为已连接
      this.connected = true;
      // 遍历消息队列中的消息，发送到 WebSocket 服务器
      ref = this.message_queue;
      for (i = 0, len = ref.length; i < len; i++) {
        message = ref[i];
        this.ws.send(JSON.stringify(message));
      }
      // 清空消息队列
      this.message_queue = [];
      // 如果定义了 onOpen 回调函数，则执行该函数
      if (this.onOpen != null) {
        return this.onOpen(e);
      }
    };

    // 定义 ZeroWebsocket 对象的 onErrorWebsocket 方法，处理 WebSocket 错误事件
    ZeroWebsocket.prototype.onErrorWebsocket = function(e) {
      // 打印错误日志
      this.log("Error", e);
      // 如果定义了 onError 回调函数，则执行该函数
      if (this.onError != null) {
        return this.onError(e);
      }
    };

    // 定义 ZeroWebsocket 对象的 onCloseWebsocket 方法，处理 WebSocket 关闭事件
    ZeroWebsocket.prototype.onCloseWebsocket = function(e, reconnect) {
      // 如果未指定重新连接时间，则默认为 10000 毫秒
      if (reconnect == null) {
        reconnect = 10000;
      }
      // 打印日志，表示连接已关闭
      this.log("Closed", e);
      // 设置连接状态为未连接
      this.connected = false;
      // 如果存在错误并且错误码为 1000 且连接不干净，则打印错误日志
      if (e && e.code === 1000 && e.wasClean === false) {
        this.log("Server error, please reload the page", e.wasClean);
      } else {
        // 否则，延迟一段时间后重新连接
        setTimeout(((function(_this) {
          return function() {
            _this.log("Reconnecting...");
            return _this.connect();
          };
        })(this)), reconnect);
      }
      // 如果定义了 onClose 回调函数，则执行该函数
      if (this.onClose != null) {
        return this.onClose(e);
      }
    };

    // 返回 ZeroWebsocket 对象
    return ZeroWebsocket;

  })();
  
  // 将 ZeroWebsocket 对象暴露到全局作用域
  window.ZeroWebsocket = ZeroWebsocket;
// 将代码包装在一个匿名函数中，以避免全局变量污染
(function() {
  // 为 jQuery 添加一个名为 scale 的 CSS 钩子，用于处理元素的缩放效果
  jQuery.cssHooks['scale'] = {
    // 获取元素的缩放值
    get: function(elem, computed) {
      // 通过 window.getComputedStyle 获取元素的 transform 属性，并匹配出数字部分
      var match = window.getComputedStyle(elem)[transform_property].match("[0-9\.]+")
      if (match) {
        // 将匹配到的值转换为浮点数并返回
        var scale = parseFloat(match[0])
        return scale
      } else {
        // 如果没有匹配到值，则返回默认的缩放值 1.0
        return 1.0
      }
    },
    // 设置元素的缩放值
    set: function(elem, val) {
      // 通过 window.getComputedStyle 获取元素的 transform 属性，并匹配出所有数字部分
      var transforms = window.getComputedStyle(elem)[transform_property].match(/[0-9\.]+/g)
      if (transforms) {
        // 更新缩放值
        transforms[0] = val
        transforms[3] = val
        // 设置元素的 transform 属性为新的缩放值
        elem.style[transform_property] = 'matrix('+transforms.join(", ")+')'
      } else {
        // 如果没有获取到 transform 属性，则直接设置元素的缩放值
        elem.style[transform_property] = "scale("+val+")"
      }
    }
  }

  // 为 jQuery 添加一个名为 scale 的动画效果
  jQuery.fx.step.scale = function(fx) {
    // 调用之前定义的 scale 钩子的 set 方法来设置元素的缩放值
    jQuery.cssHooks['scale'].set(fx.elem, fx.now)
  };

  // 检测浏览器是否支持 transform 属性，根据支持情况设置 transform_property 变量
  if (window.getComputedStyle(document.body).transform) {
    transform_property = "transform"
  } else {
    transform_property = "webkitTransform"
  }

  // ...（以下代码未提供足够的信息，无法进行准确的注释）
})();
    # 如果时间参数为空，则设置默认时间为5
    if (time == null) {
      time = 5;
    }
    # 将当前元素赋值给elem
    elem = this;
    # 在延迟时间后添加class_name类名到当前元素
    setTimeout((function() {
      return elem.addClass(class_name);
    }), time);
    # 返回当前元素
    return this;
  };

  # 定义一个jQuery插件，用于在延迟时间后设置元素的CSS属性
  jQuery.fn.cssLater = function(name, val, time) {
    var elem;
    # 如果时间参数为空，则设置默认时间为500
    if (time == null) {
      time = 500;
    }
    # 将当前元素赋值给elem
    elem = this;
    # 在延迟时间后设置当前元素的CSS属性
    setTimeout((function() {
      return elem.css(name, val);
    }), time);
    # 返回当前元素
    return this;
  };
// 将整个代码包裹在一个匿名函数中，以避免污染全局命名空间
(function (factory) {
    // 如果支持 AMD 规范，则使用 define 来定义模块
    if (typeof define === "function" && define.amd) {
        define(['jquery'], function ($) {
            return factory($);
        });
    } 
    // 如果支持 CommonJS 规范，则使用 module.exports 导出模块
    else if (typeof module === "object" && typeof module.exports === "object") {
        exports = factory(require('jquery'));
    } 
    // 否则直接调用 factory 函数，并传入 jQuery 对象
    else {
        factory(jQuery);
    }
})(function($){

// 保留原始的 jQuery "swing" 缓动函数，命名为 "jswing"
if (typeof $.easing !== 'undefined') {
    $.easing['jswing'] = $.easing['swing'];
}

// 定义一些数学函数和常量
var pow = Math.pow,
    sqrt = Math.sqrt,
    sin = Math.sin,
    cos = Math.cos,
    PI = Math.PI,
    c1 = 1.70158,
    c2 = c1 * 1.525,
    c3 = c1 + 1,
    c4 = ( 2 * PI ) / 3,
    c5 = ( 2 * PI ) / 4.5;

// 定义一个自定义的缓动函数 "bounceOut"
function bounceOut(x) {
    var n1 = 7.5625,
        d1 = 2.75;
    if ( x < 1/d1 ) {
        return n1*x*x;
    } else if ( x < 2/d1 ) {
        return n1*(x-=(1.5/d1))*x + .75;
    } else if ( x < 2.5/d1 ) {
        return n1*(x-=(2.25/d1))*x + .9375;
    } else {
        return n1*(x-=(2.625/d1))*x + .984375;
    }
}

// 扩展 jQuery 缓动函数对象，添加一些自定义的缓动函数
$.extend( $.easing,
{
    def: 'easeOutQuad',
    swing: function (x) {
        return $.easing[$.easing.def](x);
    },
    easeInQuad: function (x) {
        return x * x;
    },
    easeOutQuad: function (x) {
        return 1 - ( 1 - x ) * ( 1 - x );
    },
    easeInOutQuad: function (x) {
        return x < 0.5 ?
            2 * x * x :
            1 - pow( -2 * x + 2, 2 ) / 2;
    },
    easeInCubic: function (x) {
        return x * x * x;
    },
    easeOutCubic: function (x) {
        return 1 - pow( 1 - x, 3 );
    },
    // ... 继续添加其他自定义的缓动函数
    easeInOutCubic: function (x) {
        // 使用三次方函数实现缓入缓出效果
        return x < 0.5 ?
            4 * x * x * x :
            1 - pow( -2 * x + 2, 3 ) / 2;
    },
    easeInQuart: function (x) {
        // 使用四次方函数实现缓入效果
        return x * x * x * x;
    },
    easeOutQuart: function (x) {
        // 使用四次方函数实现缓出效果
        return 1 - pow( 1 - x, 4 );
    },
    easeInOutQuart: function (x) {
        // 使用四次方函数实现缓入缓出效果
        return x < 0.5 ?
            8 * x * x * x * x :
            1 - pow( -2 * x + 2, 4 ) / 2;
    },
    easeInQuint: function (x) {
        // 使用五次方函数实现缓入效果
        return x * x * x * x * x;
    },
    easeOutQuint: function (x) {
        // 使用五次方函数实现缓出效果
        return 1 - pow( 1 - x, 5 );
    },
    easeInOutQuint: function (x) {
        // 使用五次方函数实现缓入缓出效果
        return x < 0.5 ?
            16 * x * x * x * x * x :
            1 - pow( -2 * x + 2, 5 ) / 2;
    },
    easeInSine: function (x) {
        // 使用正弦函数实现缓入效果
        return 1 - cos( x * PI/2 );
    },
    easeOutSine: function (x) {
        // 使用正弦函数实现缓出效果
        return sin( x * PI/2 );
    },
    easeInOutSine: function (x) {
        // 使用正弦函数实现缓入缓出效果
        return -( cos( PI * x ) - 1 ) / 2;
    },
    easeInExpo: function (x) {
        // 使用指数函数实现缓入效果
        return x === 0 ? 0 : pow( 2, 10 * x - 10 );
    },
    easeOutExpo: function (x) {
        // 使用指数函数实现缓出效果
        return x === 1 ? 1 : 1 - pow( 2, -10 * x );
    },
    easeInOutExpo: function (x) {
        // 使用指数函数实现缓入缓出效果
        return x === 0 ? 0 : x === 1 ? 1 : x < 0.5 ?
            pow( 2, 20 * x - 10 ) / 2 :
            ( 2 - pow( 2, -20 * x + 10 ) ) / 2;
    },
    easeInCirc: function (x) {
        // 使用圆形函数实现缓入效果
        return 1 - sqrt( 1 - pow( x, 2 ) );
    },
    easeOutCirc: function (x) {
        // 使用圆形函数实现缓出效果
        return sqrt( 1 - pow( x - 1, 2 ) );
    },
    easeInOutCirc: function (x) {
        // 使用圆形函数实现缓入缓出效果
        return x < 0.5 ?
            ( 1 - sqrt( 1 - pow( 2 * x, 2 ) ) ) / 2 :
            ( sqrt( 1 - pow( -2 * x + 2, 2 ) ) + 1 ) / 2;
    },
    easeInElastic: function (x) {
        // 使用弹性函数实现缓入效果
        return x === 0 ? 0 : x === 1 ? 1 :
            -pow( 2, 10 * x - 10 ) * sin( ( x * 10 - 10.75 ) * c4 );
    },
    easeOutElastic: function (x) {
        // 使用弹性函数实现缓出效果
        return x === 0 ? 0 : x === 1 ? 1 :
            pow( 2, -10 * x ) * sin( ( x * 10 - 0.75 ) * c4 ) + 1;
    },
    easeInOutElastic: function (x) {
        // 使用弹簧缓动函数实现在动画过程中速度的变化
        return x === 0 ? 0 : x === 1 ? 1 : x < 0.5 ?
            // 在前半段时间内，使用弹簧缓动函数计算速度变化
            -( pow( 2, 20 * x - 10 ) * sin( ( 20 * x - 11.125 ) * c5 )) / 2 :
            // 在后半段时间内，使用弹簧缓动函数计算速度变化
            pow( 2, -20 * x + 10 ) * sin( ( 20 * x - 11.125 ) * c5 ) / 2 + 1;
    },
    easeInBack: function (x) {
        // 使用后退缓动函数实现在动画开始时速度的变化
        return c3 * x * x * x - c1 * x * x;
    },
    easeOutBack: function (x) {
        // 使用后退缓动函数实现在动画结束时速度的变化
        return 1 + c3 * pow( x - 1, 3 ) + c1 * pow( x - 1, 2 );
    },
    easeInOutBack: function (x) {
        // 使用后退缓动函数实现在动画过程中速度的变化
        return x < 0.5 ?
            // 在前半段时间内，使用后退缓动函数计算速度变化
            ( pow( 2 * x, 2 ) * ( ( c2 + 1 ) * 2 * x - c2 ) ) / 2 :
            // 在后半段时间内，使用后退缓动函数计算速度变化
            ( pow( 2 * x - 2, 2 ) *( ( c2 + 1 ) * ( x * 2 - 2 ) + c2 ) + 2 ) / 2;
    },
    easeInBounce: function (x) {
        // 使用反弹缓动函数实现在动画开始时速度的变化
        return 1 - bounceOut( 1 - x );
    },
    easeOutBounce: bounceOut,
    easeInOutBounce: function (x) {
        // 使用反弹缓动函数实现在动画过程中速度的变化
        return x < 0.5 ?
            // 在前半段时间内，使用反弹缓动函数计算速度变化
            ( 1 - bounceOut( 1 - 2 * x ) ) / 2 :
            // 在后半段时间内，使用反弹缓动函数计算速度变化
            ( 1 + bounceOut( 2 * x - 1 ) ) / 2;
    }
(function() {
  // 创建 Fixbutton 类
  var Fixbutton;

  Fixbutton = (function() {
    function Fixbutton() {
      // 初始化拖拽状态为 false
      this.dragging = false;
      // 当鼠标移入 fixbutton-bg 元素时触发
      $(".fixbutton-bg").on("mouseover", function() {
        // 动画效果：缩放 fixbutton-bg 元素到 0.7 倍大小
        $(".fixbutton-bg").stop().animate({
          "scale": 0.7
        }, 800, "easeOutElastic");
        // 动画效果：显示 fixbutton-burger 元素，设置不透明度为 1.5，左移 0
        $(".fixbutton-burger").stop().animate({
          "opacity": 1.5,
          "left": 0
        }, 800, "easeOutElastic");
        // 动画效果：隐藏 fixbutton-text 元素，设置不透明度为 0，左移 20
        return $(".fixbutton-text").stop().animate({
          "opacity": 0,
          "left": 20
        }, 300, "easeOutCubic");
      });
      // 当鼠标移出 fixbutton-bg 元素时触发
      $(".fixbutton-bg").on("mouseout", function() {
        // 如果 fixbutton 元素有 dragging 类，则返回 true
        if ($(".fixbutton").hasClass("dragging")) {
          return true;
        }
        // 动画效果：缩放 fixbutton-bg 元素到 0.6 倍大小
        $(".fixbutton-bg").stop().animate({
          "scale": 0.6
        }, 300, "easeOutCubic");
        // 动画效果：隐藏 fixbutton-burger 元素，设置不透明度为 0，左移 -20
        $(".fixbutton-burger").stop().animate({
          "opacity": 0,
          "left": -20
        }, 300, "easeOutCubic");
        // 动画效果：显示 fixbutton-text 元素，设置不透明度为 0.9，左移 0
        return $(".fixbutton-text").stop().animate({
          "opacity": 0.9,
          "left": 0
        }, 300, "easeOutBack");
      });
      // 当 fixbutton-bg 元素被点击时触发
      /*$(".fixbutton-bg").on "click", ->
                  return false
       */
      // 当鼠标在 fixbutton-bg 元素上按下时触发
      $(".fixbutton-bg").on("mousedown", function() {});
      // 当鼠标在 fixbutton-bg 元素上释放时触发
      $(".fixbutton-bg").on("mouseup", function() {});
    }
    return Fixbutton;
  })();
  // 将 Fixbutton 类绑定到全局对象 window 上
  window.Fixbutton = Fixbutton;
}).call(this);

// 创建 Infopanel 类
(function() {
  var Infopanel,
    // 定义 bind 函数
    bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };
  // 定义 Infopanel 类
  Infopanel = (function() {
    # 定义一个名为 Infopanel 的构造函数，接受一个参数 elem
    function Infopanel(elem) {
      # 将传入的 elem 参数赋值给当前对象的 elem 属性
      this.elem = elem;
      # 将 this.setAction 方法绑定到当前对象上
      this.setAction = bind(this.setAction, this);
      # 将 this.setClosedNum 方法绑定到当前对象上
      this.setClosedNum = bind(this.setClosedNum, this);
      # 将 this.setTitle 方法绑定到当前对象上
      this.setTitle = bind(this.setTitle, this);
      # 将 this.open 方法绑定到当前对象上
      this.open = bind(this.open, this);
      # 将 this.close 方法绑定到当前对象上
      this.close = bind(this.close, this);
      # 将 this.hide 方法绑定到当前对象上
      this.hide = bind(this.hide, this);
      # 将 this.updateEvents 方法绑定到当前对象上
      this.updateEvents = bind(this.updateEvents, this);
      # 将 this.unfold 方法绑定到当前对象上
      this.unfold = bind(this.unfold, this);
      # 将 this.show 方法绑定到当前对象上
      this.show = bind(this.show, this);
      # 初始化 visible 属性为 false
      this.visible = false;
    }

    # 定义 Infopanel 对象的 show 方法，接受一个布尔类型的参数 closed，默认值为 false
    Infopanel.prototype.show = function(closed) {
      # 如果传入的 closed 参数为 null，则将其赋值为 false
      if (closed == null) {
        closed = false;
      }
      # 给当前对象的 elem 属性的父元素添加 "visible" 类
      this.elem.parent().addClass("visible");
      # 如果 closed 为 true，则调用 close 方法，否则调用 open 方法
      if (closed) {
        return this.close();
      } else {
        return this.open();
      }
    };

    # 定义 Infopanel 对象的 unfold 方法
    Infopanel.prototype.unfold = function() {
      # 切换当前对象的 elem 属性的类名为 "unfolded"
      this.elem.toggleClass("unfolded");
      # 返回 false
      return false;
    };

    # 定义 Infopanel 对象的 updateEvents 方法
    Infopanel.prototype.updateEvents = function() {
      # 移除当前对象 elem 的所有 "click" 事件
      this.elem.off("click");
      # 移除当前对象 elem 下所有 ".close" 元素的 "click" 事件
      this.elem.find(".close").off("click");
      # 移除当前对象 elem 下所有 ".line" 元素的 "click" 事件
      this.elem.find(".line").off("click");
      # 给当前对象 elem 下所有 ".line" 元素添加 "click" 事件，调用 unfold 方法
      this.elem.find(".line").on("click", this.unfold);
      # 如果当前对象 elem 有 "closed" 类，则给其添加 "click" 事件，调用 onOpened 方法并打开面板；否则给 ".close" 元素添加 "click" 事件，调用 onClosed 方法并关闭面板
      if (this.elem.hasClass("closed")) {
        return this.elem.on("click", (function(_this) {
          return function() {
            _this.onOpened();
            return _this.open();
          };
        })(this));
      } else {
        return this.elem.find(".close").on("click", (function(_this) {
          return function() {
            _this.onClosed();
            return _this.close();
          };
        })(this));
      }
    };

    # 定义 Infopanel 对象的 hide 方法
    Infopanel.prototype.hide = function() {
      # 给当前对象 elem 的父元素移除 "visible" 类
      return this.elem.parent().removeClass("visible");
    };

    # 定义 Infopanel 对象的 close 方法
    Infopanel.prototype.close = function() {
      # 给当前对象 elem 添加 "closed" 类
      this.elem.addClass("closed");
      # 调用 updateEvents 方法
      this.updateEvents();
      # 返回 false
      return false;
    };

    # 定义 Infopanel 对象的 open 方法
    Infopanel.prototype.open = function() {
      # 给当前对象 elem 移除 "closed" 类
      this.elem.removeClass("closed");
      # 调用 updateEvents 方法
      this.updateEvents();
      # 返回 false
      return false;
    };
    # 设置信息面板的标题，接受两行文本作为参数
    Infopanel.prototype.setTitle = function(line1, line2) {
      # 在信息面板元素中找到类名为"line-1"的元素，并设置其文本内容为line1
      this.elem.find(".line-1").text(line1);
      # 在信息面板元素中找到类名为"line-2"的元素，并设置其文本内容为line2，然后返回该元素
      return this.elem.find(".line-2").text(line2);
    };
    
    # 设置信息面板的关闭数量，接受一个数字作为参数
    Infopanel.prototype.setClosedNum = function(num) {
      # 在信息面板元素中找到类名为"closed-num"的元素，并设置其文本内容为num
      return this.elem.find(".closed-num").text(num);
    };
    
    # 设置信息面板的操作按钮，接受一个标题和一个函数作为参数
    Infopanel.prototype.setAction = function(title, func) {
      # 在信息面板元素中找到类名为"button"的元素，并设置其文本内容为title，然后解绑"click"事件并绑定新的"click"事件为func
      return this.elem.find(".button").text(title).off("click").on("click", func);
    };
    
    # 返回信息面板对象
    return Infopanel;
    
    })();  # 立即执行函数表达式
    
    # 将信息面板对象暴露给全局作用域
    window.Infopanel = Infopanel;
// 调用一个匿名函数，并将 this 绑定到全局对象
}).call(this);

/* ---- Loading.coffee ---- */

// 定义一个 Loading 类
(function() {
  var Loading,
    slice = [].slice;

  Loading = (function() {
    // Loading 类的构造函数，接受一个 wrapper 参数
    function Loading(wrapper) {
      this.wrapper = wrapper;
      // 如果全局对象中存在 show_loadingscreen 方法，则调用 showScreen 方法
      if (window.show_loadingscreen) {
        this.showScreen();
      }
      // 初始化计时器
      this.timer_hide = null;
      this.timer_set = null;
    }

    // 设置加载进度的方法，接受一个百分比参数
    Loading.prototype.setProgress = function(percent) {
      // 如果隐藏计时器存在，则清除
      if (this.timer_hide) {
        clearInterval(this.timer_hide);
      }
      // 使用 RateLimit 方法限制每 500 毫秒执行一次匿名函数
      return this.timer_set = RateLimit(500, function() {
        // 设置进度条的样式
        return $(".progressbar").css({
          "transform": "scaleX(" + (parseInt(percent * 100) / 100) + ")"
        }).css("opacity", "1").css("display", "block");
      });
    };

    // 隐藏加载进度的方法
    Loading.prototype.hideProgress = function() {
      this.log("hideProgress");
      // 如果设置计时器存在，则清除
      if (this.timer_set) {
        clearInterval(this.timer_set);
      }
      // 使用 setTimeout 方法延迟 300 毫秒执行匿名函数
      return this.timer_hide = setTimeout(((function(_this) {
        return function() {
          // 设置进度条的样式并隐藏
          return $(".progressbar").css({
            "transform": "scaleX(1)"
          }).css("opacity", "0").hideLater(1000);
        };
      })(this)), 300);
    };

    // 显示加载屏幕的方法
    Loading.prototype.showScreen = function() {
      // 设置加载屏幕的样式并添加类名 ready
      $(".loadingscreen").css("display", "block").addClassLater("ready");
      this.screen_visible = true;
      // 打印连接信息
      return this.printLine("&nbsp;&nbsp;&nbsp;Connecting...");
    };
    // 定义一个名为 showTooLarge 的方法，用于显示网站大小超出默认限制的确认信息
    Loading.prototype.showTooLarge = function(site_info) {
      // 记录日志，显示大型网站确认信息
      this.log("Displaying large site confirmation");
      // 如果页面中没有类名为 button-setlimit 的元素
      if ($(".console .button-setlimit").length === 0) {
        // 创建一行警告信息，显示网站大小和默认允许的大小
        line = this.printLine("Site size: <b>" + (parseInt(site_info.settings.size / 1024 / 1024)) + "MB</b> is larger than default allowed " + (parseInt(site_info.size_limit)) + "MB", "warning");
        // 创建一个设置限制的按钮，并绑定点击事件
        button = $("<a href='#Set+limit' class='button button-setlimit'>" + ("Open site and set size limit to " + site_info.next_size_limit + "MB") + "</a>");
        button.on("click", (function(_this) {
          return function() {
            // 点击按钮后添加加载状态，并调用 wrapper 对象的 setSizeLimit 方法
            button.addClass("loading");
            return _this.wrapper.setSizeLimit(site_info.next_size_limit);
          };
        })(this));
        // 在警告信息后插入按钮
        line.after(button);
        // 延迟 100 毫秒后显示 "Ready." 信息
        return setTimeout(((function(_this) {
          return function() {
            return _this.printLine('Ready.');
          };
        })(this)), 100);
      }
    };
    // 显示使用 Tor 桥接的跟踪器
    Loading.prototype.showTrackerTorBridge = function(server_info) {
      // 如果页面中没有按钮集合，并且服务器信息中没有使用 Tor meek 桥接
      var button, line;
      if ($(".console .button-settrackerbridge").length === 0 && !server_info.tor_use_meek_bridges) {
        // 打印错误信息
        line = this.printLine("Tracker connection error detected.", "error");
        // 创建按钮元素
        button = $("<a href='#Enable+Tor+bridges' class='button button-settrackerbridge'>" + "Use Tor meek bridges for tracker connections" + "</a>");
        // 点击按钮时执行的函数
        button.on("click", (function(_this) {
          return function() {
            // 按钮添加加载状态
            button.addClass("loading");
            // 设置 Tor 使用桥接
            _this.wrapper.ws.cmd("configSet", ["tor_use_bridges", ""]);
            // 设置跟踪器代理为 Tor
            _this.wrapper.ws.cmd("configSet", ["trackers_proxy", "tor"]);
            // 更新站点信息
            _this.wrapper.ws.cmd("siteUpdate", {
              address: _this.wrapper.site_info.address,
              announce: true
            });
            // 重新加载 iframe
            _this.wrapper.reloadIframe();
            return false;
          };
        })(this));
        // 在错误信息后插入按钮
        line.after(button);
        // 如果服务器信息中没有 Tor meek 桥接
        if (!server_info.tor_has_meek_bridges) {
          // 按钮添加禁用状态
          button.addClass("disabled");
          // 打印警告信息
          return this.printLine("No meek bridge support in your client, please <a href='https://github.com/HelloZeroNet/ZeroNet#how-to-join'>download the latest bundle</a>.", "warning");
        }
      }
    };

    // 隐藏加载屏幕
    Loading.prototype.hideScreen = function() {
      // 打印日志信息
      this.log("hideScreen");
      // 如果加载屏幕没有完成
      if (!$(".loadingscreen").hasClass("done")) {
        // 如果加载屏幕可见
        if (this.screen_visible) {
          // 添加完成类并在 2000 毫秒后移除加载屏幕
          $(".loadingscreen").addClass("done").removeLater(2000);
        } else {
          // 直接移除加载屏幕
          $(".loadingscreen").remove();
        }
      }
      // 加载屏幕不可见
      return this.screen_visible = false;
    };
    // 定义 Loading 对象的 print 方法，用于在加载屏幕上打印文本
    Loading.prototype.print = function(text, type) {
      // 声明变量 last_line
      var last_line;
      // 如果 type 未定义，则默认为 "normal"
      if (type == null) {
        type = "normal";
      }
      // 如果加载屏幕不可见，则返回 false
      if (!this.screen_visible) {
        return false;
      }
      // 移除控制台中的光标
      $(".loadingscreen .console .cursor").remove();
      // 获取控制台中最后一行的内容
      last_line = $(".loadingscreen .console .console-line:last-child");
      // 如果类型为 "error"，则将文本包裹在错误样式的 span 标签中
      if (type === "error") {
        text = "<span class='console-error'>" + text + "</span>";
      }
      // 将文本追加到最后一行的内容中
      return last_line.html(last_line.html() + text);
    };

    // 定义 Loading 对象的 printLine 方法，用于在加载屏幕上打印一行文本
    Loading.prototype.printLine = function(text, type) {
      // 声明变量 line
      var line;
      // 如果 type 未定义，则默认为 "normal"
      if (type == null) {
        type = "normal";
      }
      // 如果加载屏幕不可见，则返回 false
      if (!this.screen_visible) {
        return false;
      }
      // 移除控制台中的光标
      $(".loadingscreen .console .cursor").remove();
      // 如果类型为 "error"，则将文本包裹在错误样式的 span 标签中
      if (type === "error") {
        text = "<span class='console-error'>" + text + "</span>";
      } else {
        // 否则在文本后面添加一个光标
        text = text + "<span class='cursor'> </span>";
      }
      // 创建一个包含文本内容的新的控制台行，并追加到控制台中
      line = $("<div class='console-line'>" + text + "</div>").appendTo(".loadingscreen .console");
      // 如果类型为 "warning"，则添加警告样式
      if (type === "warning") {
        line.addClass("console-warning");
      }
      // 返回新创建的控制台行
      return line;
    };

    // 定义 Loading 对象的 log 方法，用于在控制台中打印日志
    Loading.prototype.log = function() {
      // 将参数转换为数组
      var args;
      args = 1 <= arguments.length ? slice.call(arguments, 0) : [];
      // 调用 console.log 方法，在日志前添加标识符 "[Loading]"
      return console.log.apply(console, ["[Loading]"].concat(slice.call(args)));
    };

    // 返回 Loading 对象
    return Loading;

  })();
  
  // 将 Loading 对象绑定到全局对象 window 上
  window.Loading = Loading;
// 将整个代码包装在一个匿名函数中，避免变量污染全局作用域
(function() {
  // 定义变量和函数
  var Notifications,
    slice = [].slice;

  // 创建 Notifications 类
  Notifications = (function() {
    // 构造函数，接受一个参数 elem1
    function Notifications(elem1) {
      // 将参数 elem1 赋值给实例的 elem 属性
      this.elem = elem1;
      // 返回实例
      this;
    }

    // 在 Notifications 类的原型上定义 test 方法
    Notifications.prototype.test = function() {
      // 设置定时器，在 1000 毫秒后执行
      setTimeout(((function(_this) {
        return function() {
          // 调用 add 方法，传入参数
          _this.add("connection", "error", "Connection lost to <b>UiServer</b> on <b>localhost</b>!");
          // 调用 add 方法，传入参数
          return _this.add("message-Anyone", "info", "New  from <b>Anyone</b>.");
        };
      })(this)), 1000);
      // 设置定时器，在 3000 毫秒后执行
      return setTimeout(((function(_this) {
        return function() {
          // 调用 add 方法，传入参数
          return _this.add("connection", "done", "<b>UiServer</b> connection recovered.", 5000);
        };
      })(this)), 3000);
    };

    // 在 Notifications 类的原型上定义 close 方法
    Notifications.prototype.close = function(elem) {
      // 使用动画效果改变元素的宽度和透明度
      elem.stop().animate({
        "width": 0,
        "opacity": 0
      }, 700, "easeInOutCubic");
      // 使用滑动效果隐藏元素，并在动画完成后移除元素
      return elem.slideUp(300, (function() {
        return elem.remove();
      }));
    };

    // 在 Notifications 类的原型上定义 log 方法
    Notifications.prototype.log = function() {
      // 将参数转换为数组
      var args;
      args = 1 <= arguments.length ? slice.call(arguments, 0) : [];
      // 在控制台输出日志
      return console.log.apply(console, ["[Notifications]"].concat(slice.call(args)));
    };

    // 返回 Notifications 类
    return Notifications;

  })();

  // 将 Notifications 类绑定到全局对象 window 上
  window.Notifications = Notifications;

}).call(this);

// 另一个匿名函数
(function() {
  // 定义变量和函数
  var Wrapper, origin, proto, ws_url,
    // 定义辅助函数 bind、indexOf 和 slice
    bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; },
    indexOf = [].indexOf || function(item) { for (var i = 0, l = this.length; i < l; i++) { if (i in this && this[i] === item) return i; } return -1; },
    slice = [].slice;

  // 创建 Wrapper 类
  Wrapper = (function() {
    // 返回 Wrapper 类
    }

// 代码未完整，无法提供完整的注释
    // 验证事件是否受信任，如果不受信任则抛出异常
    Wrapper.prototype.verifyEvent = function(allowed_target, e) {
      var ref;
      // 如果事件不受信任，则抛出异常
      if (!e.originalEvent.isTrusted) {
        throw "Event not trusted";
      }
      // 如果事件的构造函数不在允许的构造函数列表中，则抛出异常
      if (ref = e.originalEvent.constructor, indexOf.call(this.allowed_event_constructors, ref) < 0) {
        throw "Invalid event constructor: " + e.constructor + " not in " + (JSON.stringify(this.allowed_event_constructors));
      }
      // 如果事件的当前目标不在允许的目标列表中，则抛出异常
      if (e.originalEvent.currentTarget !== allowed_target[0]) {
        throw "Invalid event target: " + e.originalEvent.currentTarget + " != " + allowed_target[0];
      }
    };

    // 处理来自 WebSocket 的消息
    Wrapper.prototype.onMessageWebsocket = function(e) {
      var message;
      // 解析收到的消息
      message = JSON.parse(e.data);
      // 处理消息
      return this.handleMessageWebsocket(message);
    };

    // 处理来自内部消息的处理
    Wrapper.prototype.onMessageInner = function(e) {
      var message;
      // 如果没有启用窗口消息的安全性，并且还未测试过 opener，则进行测试
      if (!window.postmessage_nonce_security && this.opener_tested === false) {
        if (window.opener && window.opener !== window) {
          this.log("Opener present", window.opener);
          this.displayOpenerDialog();
          return false;
        } else {
          this.opener_tested = true;
        }
      }
      // 获取消息内容
      message = e.data;
      // 如果消息没有命令，则记录错误并返回
      if (!message.cmd) {
        this.log("Invalid message:", message);
        return false;
      }
      // 如果启用了窗口消息的安全性，并且消息的包装器随机数与窗口的随机数不匹配，则记录错误并返回
      if (window.postmessage_nonce_security && message.wrapper_nonce !== window.wrapper_nonce) {
        this.log("Message nonce error:", message.wrapper_nonce, '!=', window.wrapper_nonce);
        return;
      }
      // 处理消息
      return this.handleMessage(message);
    };
    // 定义 Wrapper 对象的 cmd 方法，用于发送命令
    Wrapper.prototype.cmd = function(cmd, params, cb) {
      // 如果参数为空，则设置为空对象
      if (params == null) {
        params = {};
      }
      // 如果回调函数为空，则设置为 null
      if (cb == null) {
        cb = null;
      }
      // 创建消息对象
      var message = {};
      // 设置消息的命令和参数
      message.cmd = cmd;
      message.params = params;
      // 设置消息的 id，并将回调函数存入 waiting_cb 中
      message.id = this.next_cmd_message_id;
      if (cb) {
        this.ws.waiting_cb[message.id] = cb;
      }
      // 更新下一个命令消息的 id
      this.next_cmd_message_id -= 1;
      // 处理消息
      return this.handleMessage(message);
    };

    // 定义 Wrapper 对象的 toRelativeQuery 方法，用于处理相对查询
    Wrapper.prototype.toRelativeQuery = function(query) {
      // 如果查询为空，则设置为 null
      var back;
      if (query == null) {
        query = null;
      }
      // 如果查询为 null，则使用当前页面的查询
      if (query === null) {
        query = window.location.search;
      }
      // 获取当前页面的路径
      back = window.location.pathname;
      // 如果路径匹配指定格式，则在末尾添加斜杠
      if (back.match(/^\/[^\/]+$/)) {
        back += "/";
      }
      // 处理查询字符串
      if (query.startsWith("#")) {
        back = query;
      } else if (query.replace("?", "")) {
        back += "?" + query.replace("?", "");
      }
      // 返回处理后的查询
      return back;
    };

    // 定义 Wrapper 对象的 displayOpenerDialog 方法，用于显示打开对话框
    Wrapper.prototype.displayOpenerDialog = function() {
      // 创建对话框元素，并绑定点击事件
      var elem = $("<div class='opener-overlay'><div class='dialog'>You have opened this page by clicking on a link. Please, confirm if you want to load this site.<a href='?' target='_blank' class='button'>Open site</a></div></div>");
      elem.find('a').on("click", function() {
        // 点击链接时打开新窗口并关闭当前窗口
        window.open("?", "_blank");
        window.close();
        return false;
      });
      // 在页面顶部插入对话框元素
      return $("body").prepend(elem);
    };

    // 定义 Wrapper 对象的 actionOpenWindow 方法，用于执行打开窗口操作
    Wrapper.prototype.actionOpenWindow = function(params) {
      // 如果参数为字符串，则打开新窗口并跳转到指定地址
      var w;
      if (typeof params === "string") {
        w = window.open();
        w.opener = null;
        return w.location = params;
      } else {
        // 如果参数为数组，则打开新窗口并跳转到指定地址，并设置 opener 为 null
        w = window.open(null, params[1], params[2]);
        w.opener = null;
        return w.location = params[0];
      }
    };
    // 定义在包装器对象原型上的方法，用于请求全屏显示
    Wrapper.prototype.actionRequestFullscreen = function() {
      // 获取内部 iframe 元素
      var elem = document.getElementById("inner-iframe");
      // 获取请求全屏显示的方法，兼容不同浏览器的前缀
      var request_fullscreen = elem.requestFullScreen || elem.webkitRequestFullscreen || elem.mozRequestFullScreen || elem.msRequestFullScreen;
      // 调用请求全屏显示的方法
      return request_fullscreen.call(elem);
    };

    // 定义在包装器对象原型上的方法，用于处理 Web 通知
    Wrapper.prototype.actionWebNotification = function(message) {
      // 当事件 site_info 完成后执行
      return $.when(this.event_site_info).done((function(_this) {
        return function() {
          var res;
          // 如果用户已经允许 Web 通知
          if (Notification.permission === "granted") {
            // 显示 Web 通知
            return _this.displayWebNotification(message);
          } 
          // 如果用户已经拒绝 Web 通知
          else if (Notification.permission === "denied") {
            // 返回错误信息
            res = {
              "error": "Web notifications are disabled by the user"
            };
            // 发送内部消息
            return _this.sendInner({
              "cmd": "response",
              "to": message.id,
              "result": res
            });
          } 
          // 如果用户还未做出选择
          else {
            // 请求用户允许 Web 通知，并在用户做出选择后执行相应操作
            return Notification.requestPermission().then(function(permission) {
              if (permission === "granted") {
                return _this.displayWebNotification(message);
              }
            });
          }
        };
      })(this));
    };

    // 定义在包装器对象原型上的方法，用于关闭 Web 通知
    Wrapper.prototype.actionCloseWebNotification = function(message) {
      // 当事件 site_info 完成后执行
      return $.when(this.event_site_info).done((function(_this) {
        return function() {
          var id;
          // 获取 Web 通知的 ID
          id = message.params[0];
          // 关闭指定 ID 的 Web 通知
          return _this.web_notifications[id].close();
        };
      })(this));
    };
    # 定义一个方法用于显示 web 通知
    Wrapper.prototype.displayWebNotification = function(message) {
      # 从消息参数中获取标题
      var title = message.params[0];
      # 从消息参数中获取 id
      var id = message.params[1];
      # 从消息参数中获取选项
      var options = message.params[2];
      # 创建一个新的通知对象
      var notification = new Notification(title, options);
      # 将通知对象存储到 web_notifications 对象中
      this.web_notifications[id] = notification;
      # 设置通知显示时的回调函数
      notification.onshow = (function(_this) {
        return function() {
          # 发送内部消息，通知消息已经显示
          return _this.sendInner({
            "cmd": "response",
            "to": message.id,
            "result": "ok"
          });
        };
      })(this);
      # 设置通知点击时的回调函数
      notification.onclick = (function(_this) {
        return function(e) {
          # 如果选项中没有指定焦点标签，则阻止默认行为
          if (!options.focus_tab) {
            e.preventDefault();
          }
          # 发送内部消息，通知 web 通知被点击
          return _this.sendInner({
            "cmd": "webNotificationClick",
            "params": {
              "id": id
            }
          });
        };
      })(this);
      # 设置通知关闭时的回调函数
      return notification.onclose = (function(_this) {
        return function() {
          # 发送内部消息，通知 web 通知被关闭
          _this.sendInner({
            "cmd": "webNotificationClose",
            "params": {
              "id": id
            }
          });
          # 从 web_notifications 对象中删除关闭的通知
          return delete _this.web_notifications[id];
        };
      })(this);
    };
    # 定义一个方法用于添加权限
    Wrapper.prototype.actionPermissionAdd = function(message) {
      # 从消息中获取权限
      var permission;
      permission = message.params;
      # 当事件网站信息准备好后执行
      return $.when(this.event_site_info).done((function(_this) {
        return function() {
          # 如果权限已存在于网站设置中，则返回false
          if (indexOf.call(_this.site_info.settings.permissions, permission) >= 0) {
            return false;
          }
          # 请求权限的详细信息并显示确认框
          return _this.ws.cmd("permissionDetails", permission, function(permission_details) {
            return _this.displayConfirm("This site requests permission:" + (" <b>" + (_this.toHtmlSafe(permission)) + "</b>") + ("<br><small style='color: #4F4F4F'>" + permission_details + "</small>"), "Grant", function() {
              # 向服务器请求添加权限，并发送响应消息
              return _this.ws.cmd("permissionAdd", permission, function(res) {
                return _this.sendInner({
                  "cmd": "response",
                  "to": message.id,
                  "result": res
                });
              });
            });
          });
        };
      })(this));
    };

    # 定义一个方法用于处理通知
    Wrapper.prototype.actionNotification = function(message) {
      # 将消息参数转换为HTML安全格式
      message.params = this.toHtmlSafe(message.params);
      # 创建通知内容并添加到通知列表中
      var body = $("<span class='message'>" + message.params[1] + "</span>");
      return this.notifications.add("notification-" + message.id, message.params[0], body, message.params[2]);
    };
    // 定义一个方法用于显示确认框，接受消息体、按钮标题和回调函数作为参数
    Wrapper.prototype.displayConfirm = function(body, captions, cb) {
      // 创建消息体的外层容器，并添加消息内容
      body = $("<span class='message-outer'><span class='message'>" + body + "</span></span>");
      // 创建按钮容器
      buttons = $("<span class='buttons'></span>");
      // 如果按钮标题不是数组，则转换为数组
      if (!(captions instanceof Array)) {
        captions = [captions];
      }
      // 定义一个函数，用于处理按钮点击事件
      fn = (function(_this) {
        return function(button) {
          return button.on("click", function(e) {
            _this.verifyEvent(button, e);
            // 调用回调函数，并传入按钮的值
            cb(parseInt(e.currentTarget.dataset.value));
            return false;
          });
        };
      })(this);
      // 遍历按钮标题数组，创建按钮并绑定点击事件
      for (i = j = 0, len = captions.length; j < len; i = ++j) {
        caption = captions[i];
        button = $("<a></a>", {
          href: "#" + caption,
          "class": "button button-confirm button-" + caption + " button-" + (i + 1),
          "data-value": i + 1
        });
        button.text(caption);
        fn(button);
        buttons.append(button);
      }
      // 将按钮添加到消息体中
      body.append(buttons);
      // 将消息体添加到通知中心
      this.notifications.add("notification-" + caption, "ask", body);
      // 让第一个按钮获得焦点
      buttons.first().focus();
      // 滚动通知中心到最左侧
      return $(".notification").scrollLeft(0);
    };

    // 定义一个方法用于处理确认操作，接受消息和回调函数作为参数
    Wrapper.prototype.actionConfirm = function(message, cb) {
      var caption;
      // 如果回调函数未定义，则设置为false
      if (cb == null) {
        cb = false;
      }
      // 将消息参数转换为HTML安全格式
      message.params = this.toHtmlSafe(message.params);
      // 如果消息参数中有第二个值，则使用该值作为按钮标题，否则使用默认标题"ok"
      if (message.params[1]) {
        caption = message.params[1];
      } else {
        caption = "ok";
      }
      // 调用显示确认框的方法，并传入消息内容、按钮标题和回调函数
      return this.displayConfirm(message.params[0], caption, (function(_this) {
        return function(res) {
          // 发送内部消息，包括响应命令、消息ID和结果
          _this.sendInner({
            "cmd": "response",
            "to": message.id,
            "result": res
          });
          return false;
        };
      })(this));
    };
    // 定义一个方法用于显示提示框，接受消息、类型、标题、占位符和回调函数作为参数
    Wrapper.prototype.displayPrompt = function(message, type, caption, placeholder, cb) {
      // 创建一个包含消息内容的 span 元素
      var body = $("<span class='message'></span>").html(message);
      // 如果占位符为空，则设置默认值为空字符串
      if (placeholder == null) {
        placeholder = "";
      }
      // 创建一个输入框元素，设置类型、类名和占位符
      var input = $("<input/>", {
        type: type,
        "class": "input button-" + type,
        placeholder: placeholder
      });
      // 绑定键盘按键事件，当按下回车键时触发回调函数
      input.on("keyup", (function(_this) {
        return function(e) {
          _this.verifyEvent(input, e);
          if (e.keyCode === 13) {
            return cb(input.val());
          }
        };
      })(this));
      // 将输入框添加到消息内容中
      body.append(input);
      // 创建一个按钮元素，设置链接和类名
      var button = $("<a></a>", {
        href: "#" + caption,
        "class": "button button-" + caption
      }).text(caption);
      // 绑定点击事件，触发回调函数并阻止默认行为
      button.on("click", (function(_this) {
        return function(e) {
          _this.verifyEvent(button, e);
          cb(input.val());
          return false;
        };
      })(this));
      // 将按钮添加到消息内容中
      body.append(button);
      // 将消息内容添加到通知中心
      this.notifications.add("notification-" + message.id, "ask", body);
      // 让输入框获得焦点
      input.focus();
      // 滚动通知中心到最左侧
      return $(".notification").scrollLeft(0);
    };

    // 定义一个方法用于处理提示框的动作，接受消息作为参数
    Wrapper.prototype.actionPrompt = function(message) {
      // 将消息内容转义为 HTML 安全格式
      message.params = this.toHtmlSafe(message.params);
      // 如果消息参数中包含类型，则使用该类型，否则默认为文本类型
      var type = message.params[1] ? message.params[1] : "text";
      // 如果消息参数中包含标题，则使用该标题，否则默认为"OK"
      var caption = message.params[2] ? message.params[2] : "OK";
      // 如果消息参数中包含占位符，则使用该占位符，否则默认为空字符串
      var placeholder = message.params[3] != null ? message.params[3] : "";
      // 调用显示提示框的方法，并传入相应参数和回调函数
      return this.displayPrompt(message.params[0], type, caption, placeholder, (function(_this) {
        return function(res) {
          // 发送内部消息，包括命令、接收者和结果
          return _this.sendInner({
            "cmd": "response",
            "to": message.id,
            "result": res
          });
        };
      })(this));
    };
    # 定义一个方法，用于处理进度消息，将消息参数转换为HTML安全的格式，然后显示进度
    Wrapper.prototype.actionProgress = function(message) {
      message.params = this.toHtmlSafe(message.params);
      return this.displayProgress(message.params[0], message.params[1], message.params[2]);
    };

    # 定义一个方法，用于设置视口，如果视口元素存在，则更新其内容属性，否则创建一个新的视口元素并添加到头部
    Wrapper.prototype.actionSetViewport = function(message) {
      this.log("actionSetViewport", message);
      if ($("#viewport").length > 0) {
        return $("#viewport").attr("content", this.toHtmlSafe(message.params));
      } else {
        return $('<meta name="viewport" id="viewport">').attr("content", this.toHtmlSafe(message.params)).appendTo("head");
      }
    };

    # 定义一个方法，用于重新加载页面
    Wrapper.prototype.actionReload = function(message) {
      return this.reload(message.params[0]);
    };

    # 定义一个方法，用于重新加载页面，根据传入的URL参数进行处理
    Wrapper.prototype.reload = function(url_post) {
      var current_url;
      if (url_post == null) {
        url_post = "";
      }
      this.log("Reload");
      current_url = window.location.toString().replace(/#.*/g, "");
      if (url_post) {
        if (current_url.indexOf("?") > 0) {
          return window.location = current_url + "&" + url_post;
        } else {
          return window.location = current_url + "?" + url_post;
        }
      } else {
        return window.location.reload();
      }
    };
    # 定义一个方法用于获取本地存储的数据
    Wrapper.prototype.actionGetLocalStorage = function(message) {
      # 当事件 site_info 完成后执行回调函数
      return $.when(this.event_site_info).done((function(_this) {
        return function() {
          var data;
          # 从本地存储中获取指定键的数据
          data = localStorage.getItem("site." + _this.site_info.address + "." + _this.site_info.auth_address);
          # 如果数据不存在
          if (!data) {
            # 从本地存储中获取另一个键的数据
            data = localStorage.getItem("site." + _this.site_info.address);
            # 如果数据存在
            if (data) {
              # 将数据存储到新的键中
              localStorage.setItem("site." + _this.site_info.address + "." + _this.site_info.auth_address, data);
              # 删除旧的键
              localStorage.removeItem("site." + _this.site_info.address);
              # 记录日志
              _this.log("Migrated LocalStorage from global to auth_address based");
            }
          }
          # 如果数据存在，则解析为 JSON 格式
          if (data) {
            data = JSON.parse(data);
          }
          # 发送消息给内部方法
          return _this.sendInner({
            "cmd": "response",
            "to": message.id,
            "result": data
          });
        };
      })(this));
    };

    # 定义一个方法用于设置本地存储的数据
    Wrapper.prototype.actionSetLocalStorage = function(message) {
      # 当事件 site_info 完成后执行回调函数
      return $.when(this.event_site_info).done((function(_this) {
        return function() {
          var back;
          # 将数据以 JSON 格式存储到本地存储中
          back = localStorage.setItem("site." + _this.site_info.address + "." + _this.site_info.auth_address, JSON.stringify(message.params));
          # 发送消息给内部方法
          return _this.sendInner({
            "cmd": "response",
            "to": message.id,
            "result": back
          });
        };
      })(this));
    };
    // 当 WebSocket 连接打开时执行的函数
    Wrapper.prototype.onOpenWebsocket = function(e) {
      // 如果存在加载屏幕函数，则加入指定频道
      if (window.show_loadingscreen) {
        this.ws.cmd("channelJoin", {
          "channels": ["siteChanged", "serverChanged", "announcerChanged"]
        });
      } else {
        this.ws.cmd("channelJoin", {
          "channels": ["siteChanged", "serverChanged"]
        });
      }
      // 如果封装 WebSocket 未初始化且内部准备就绪，则发送内部消息
      if (!this.wrapperWsInited && this.inner_ready) {
        this.sendInner({
          "cmd": "wrapperOpenedWebsocket"
        });
        this.wrapperWsInited = true;
      }
      // 如果存在加载屏幕函数，则获取服务器信息和通告者信息
      if (window.show_loadingscreen) {
        this.ws.cmd("serverInfo", [], (function(_this) {
          return function(server_info) {
            return _this.server_info = server_info;
          };
        })(this));
        this.ws.cmd("announcerInfo", [], (function(_this) {
          return function(announcer_info) {
            return _this.setAnnouncerInfo(announcer_info);
          };
        })(this));
      }
      // 如果内部已加载，则重新加载站点信息
      if (this.inner_loaded) {
        this.reloadSiteInfo();
      }
      // 延迟 2000 毫秒后执行函数，如果站点信息不存在，则重新加载站点信息
      setTimeout(((function(_this) {
        return function() {
          if (!_this.site_info) {
            return _this.reloadSiteInfo();
          }
        };
      })(this)), 2000);
      // 如果存在 WebSocket 错误，则添加通知并清除错误
      if (this.ws_error) {
        this.notifications.add("connection", "done", "Connection with <b>UiServer Websocket</b> recovered.", 6000);
        return this.ws_error = null;
      }
    };
    # 当 WebSocket 关闭时的处理函数
    Wrapper.prototype.onCloseWebsocket = function(e) {
      # 将 wrapperWsInited 置为 false
      this.wrapperWsInited = false;
      # 设置一个定时器，延迟执行内部函数
      return setTimeout(((function(_this) {
        return function() {
          # 发送内部消息，通知 WebSocket 已关闭
          _this.sendInner({
            "cmd": "wrapperClosedWebsocket"
          });
          # 根据 WebSocket 关闭的状态进行不同的处理
          if (e && e.code === 1000 && e.wasClean === false) {
            # 如果关闭码为 1000 且不是干净的关闭，则设置 ws_error 为错误通知
            return _this.ws_error = _this.notifications.add("connection", "error", "UiServer Websocket error, please reload the page.");
          } else if (e && e.code === 1001 && e.wasClean === true) {
            # 如果关闭码为 1001 且是干净的关闭，则不做处理
          } else if (!_this.ws_error) {
            # 如果没有错误通知，则设置 ws_error 为连接丢失的错误通知
            return _this.ws_error = _this.notifications.add("connection", "error", "Connection with <b>UiServer Websocket</b> was lost. Reconnecting...");
          }
        };
      })(this)), 1000);
    };

    # 页面加载时的处理函数
    Wrapper.prototype.onPageLoad = function(e) {
      var ref;
      # 记录页面加载
      this.log("onPageLoad");
      # 将 inner_loaded 置为 true
      this.inner_loaded = true;
      # 如果 inner_ready 为假，则发送内部消息，通知页面已准备好
      if (!this.inner_ready) {
        this.sendInner({
          "cmd": "wrapperReady"
        });
      }
      # 如果 WebSocket 状态为打开且 site_info 不存在，则重新加载站点信息
      if (this.ws.ws.readyState === 1 && !this.site_info) {
        return this.reloadSiteInfo();
      } else if (this.site_info && (((ref = this.site_info.content) != null ? ref.title : void 0) != null) && !this.is_title_changed) {
        # 如果 site_info 存在且包含标题且标题未改变，则设置页面标题
        window.document.title = this.site_info.content.title + " - ZeroNet";
        return this.log("Setting title to", window.document.title);
      }
    };

    # Wrapper 加载完成时的处理函数
    Wrapper.prototype.onWrapperLoad = function() {
      # 设置 script_nonce 和 wrapper_key
      this.script_nonce = window.script_nonce;
      this.wrapper_key = window.wrapper_key;
      # 删除全局变量
      delete window.wrapper;
      delete window.wrapper_key;
      delete window.script_nonce;
      # 移除指定的元素
      return $("#script_init").remove();
    };

    # 发送内部消息的函数
    Wrapper.prototype.sendInner = function(message) {
      return this.inner.postMessage(message, '*');
    };
    // 重新加载站点信息的方法
    Wrapper.prototype.reloadSiteInfo = function() {
      var params;
      // 如果加载屏幕可见，则设置参数为文件状态
      if (this.loading.screen_visible) {
        params = {
          "file_status": window.file_inner_path
        };
      } else {
        // 否则参数为空
        params = {};
      }
      // 调用 WebSocket 对象的 siteInfo 方法，传入参数和回调函数
      return this.ws.cmd("siteInfo", params, (function(_this) {
        return function(site_info) {
          var ref;
          // 设置地址为站点信息中的地址
          _this.address = site_info.address;
          // 调用 setSiteInfo 方法，传入站点信息
          _this.setSiteInfo(site_info);
          // 如果站点大小超过限制并且加载屏幕不可见，则显示确认框
          if (site_info.settings.size > site_info.size_limit * 1024 * 1024 && !_this.loading.screen_visible) {
            _this.displayConfirm("Site is larger than allowed: " + ((site_info.settings.size / 1024 / 1024).toFixed(1)) + "MB/" + site_info.size_limit + "MB", "Set limit to " + site_info.next_size_limit + "MB", function() {
              // 调用 WebSocket 对象的 siteSetLimit 方法，传入下一个大小限制，并设置回调函数
              return _this.ws.cmd("siteSetLimit", [site_info.next_size_limit], function(res) {
                if (res === "ok") {
                  // 如果返回结果为 "ok"，则添加通知
                  return _this.notifications.add("size_limit", "done", "Site storage limit modified!", 5000);
                }
              });
            });
          }
          // 如果站点内容中有标题并且标题未改变，则设置文档标题
          if ((((ref = site_info.content) != null ? ref.title : void 0) != null) && !_this.is_title_changed) {
            window.document.title = site_info.content.title + " - ZeroNet";
            // 记录日志
            return _this.log("Setting title to", window.document.title);
          }
        };
      })(this));
    };
    // 定义一个名为 siteSign 的方法，用于对站点进行签名
    Wrapper.prototype.siteSign = function(inner_path, cb) {
      // 如果站点信息中存在私钥
      if (this.site_info.privatekey) {
        // 在信息面板中的按钮添加加载样式
        this.infopanel.elem.find(".button").addClass("loading");
        // 调用 WebSocket 对象的 siteSign 方法进行签名操作
        return this.ws.cmd("siteSign", {
          privatekey: "stored",
          inner_path: inner_path,
          update_changed_files: true
        }, (function(_this) {
          return function(res) {
            // 如果返回结果为 "ok"，则执行回调函数并传入 true
            if (res === "ok") {
              if (typeof cb === "function") {
                cb(true);
              }
            } else {
              // 如果返回结果不为 "ok"，则执行回调函数并传入 false
              if (typeof cb === "function") {
                cb(false);
              }
            }
            // 移除信息面板中的按钮加载样式
            return _this.infopanel.elem.find(".button").removeClass("loading");
          };
        })(this));
      } else {
        // 如果站点信息中不存在私钥，则显示提示框要求输入私钥
        return this.displayPrompt("Enter your private key:", "password", "Sign", "", (function(_this) {
          return function(privatekey) {
            // 在信息面板中的按钮添加加载样式
            _this.infopanel.elem.find(".button").addClass("loading");
            // 调用 WebSocket 对象的 siteSign 方法进行签名操作
            return _this.ws.cmd("siteSign", {
              privatekey: privatekey,
              inner_path: inner_path,
              update_changed_files: true
            }, function(res) {
              // 如果返回结果为 "ok"，则执行回调函数并传入 true
              if (res === "ok") {
                if (typeof cb === "function") {
                  cb(true);
                }
              } else {
                // 如果返回结果不为 "ok"，则执行回调函数并传入 false
                if (typeof cb === "function") {
                  cb(false);
                }
              }
              // 移除信息面板中的按钮加载样式
              return _this.infopanel.elem.find(".button").removeClass("loading");
            });
          };
        })(this));
      }
    };

    // 定义一个名为 sitePublish 的方法，用于发布站点
    Wrapper.prototype.sitePublish = function(inner_path) {
      // 调用 WebSocket 对象的 sitePublish 方法进行站点发布操作
      return this.ws.cmd("sitePublish", {
        "inner_path": inner_path,
        "sign": false
      });
    };
    # 更新修改面板的方法
    Wrapper.prototype.updateModifiedPanel = function() {
      # 调用 ws 对象的 cmd 方法，请求站点修改文件列表
      return this.ws.cmd("siteListModifiedFiles", [], (function(_this) {
        return function(res) {
          # 获取修改文件的数量
          var closed, num, ref;
          num = (ref = res.modified_files) != null ? ref.length : void 0;
          # 如果有修改文件
          if (num > 0) {
            # 检查是否关闭了修改文件通知
            closed = _this.site_info.settings.modified_files_notification === false;
            # 显示信息面板
            _this.infopanel.show(closed);
          } else {
            # 隐藏信息面板
            _this.infopanel.hide();
          }
          # 如果有修改文件
          if (num > 0) {
            # 设置信息面板的标题和内容
            _this.infopanel.setTitle(res.modified_files.length + " modified file" + (num > 1 ? 's' : ''), res.modified_files.join(", "));
            # 设置关闭的文件数量
            _this.infopanel.setClosedNum(num);
            # 设置操作为"签名并发布"
            _this.infopanel.setAction("Sign & Publish", function() {
              # 签名站点内容文件
              _this.siteSign("content.json", function(res) {
                if (res) {
                  # 如果签名成功，添加通知并发布站点内容文件
                  _this.notifications.add("sign", "done", "content.json Signed!", 5000);
                  return _this.sitePublish("content.json");
                }
              });
              return false;
            });
          }
          # 记录日志
          return _this.log("siteListModifiedFiles", num, res);
        };
      })(this));
    };
    // 设置播音员信息
    Wrapper.prototype.setAnnouncerInfo = function(announcer_info) {
      var key, ref, status_db, status_line, val;
      // 初始化状态数据库
      status_db = {
        announcing: [],
        error: [],
        announced: []
      };
      // 遍历播音员信息的状态，将状态信息存入状态数据库
      ref = announcer_info.stats;
      for (key in ref) {
        val = ref[key];
        if (val.status) {
          status_db[val.status].push(val);
        }
      }
      // 生成状态行信息
      status_line = "Trackers announcing: " + status_db.announcing.length + ", error: " + status_db.error.length + ", done: " + status_db.announced.length;
      // 更新或创建状态行
      if (this.announcer_line) {
        this.announcer_line.text(status_line);
      } else {
        this.announcer_line = this.loading.printLine(status_line);
      }
      // 根据条件显示 Tor 桥接器
      if (status_db.error.length > (status_db.announced.length + status_db.announcing.length) && status_db.announced.length < 3) {
        return this.loading.showTrackerTorBridge(this.server_info);
      }
    };

    // 更新进度信息
    Wrapper.prototype.updateProgress = function(site_info) {
      // 根据站点信息更新进度条
      if (site_info.tasks > 0 && site_info.started_task_num > 0) {
        return this.loading.setProgress(1 - (Math.max(site_info.tasks, site_info.bad_files) / site_info.started_task_num));
      } else {
        return this.loading.hideProgress();
      }
    };

    // 将值转换为 HTML 安全格式
    Wrapper.prototype.toHtmlSafe = function(values) {
      var i, j, len, value;
      // 如果值不是数组，则转换为数组
      if (!(values instanceof Array)) {
        values = [values];
      }
      // 遍历值数组，将其中的特殊字符转换为 HTML 安全格式
      for (i = j = 0, len = values.length; j < len; i = ++j) {
        value = values[i];
        if (value instanceof Array) {
          value = this.toHtmlSafe(value);
        } else {
          value = String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&apos;');
          value = value.replace(/&lt;([\/]{0,1}(br|b|u|i|small))&gt;/g, "<$1>");
        }
        values[i] = value;
      }
      // 返回转换后的值数组
      return values;
    };
    # 设置尺寸限制，并重新加载页面
    Wrapper.prototype.setSizeLimit = function(size_limit, reload) {
      # 如果 reload 为 null，则设置为 true
      if (reload == null) {
        reload = true;
      }
      # 记录日志
      this.log("setSizeLimit: " + size_limit + ", reload: " + reload);
      # 将 inner_loaded 设置为 false
      this.inner_loaded = false;
      # 发送命令给服务器，设置尺寸限制
      this.ws.cmd("siteSetLimit", [size_limit], (function(_this) {
        return function(res) {
          # 如果返回结果不是 "ok"，则返回 false
          if (res !== "ok") {
            return false;
          }
          # 打印返回结果
          _this.loading.printLine(res);
          # 将 inner_loaded 设置为 false
          _this.inner_loaded = false;
          # 如果 reload 为 true，则重新加载 iframe
          if (reload) {
            return _this.reloadIframe();
          }
        };
      })(this));
      # 返回 false
      return false;
    };

    # 重新加载 iframe
    Wrapper.prototype.reloadIframe = function() {
      # 获取当前 iframe 的 src 属性
      var src;
      src = $("iframe").attr("src");
      # 发送命令给服务器，获取 wrapper_nonce
      return this.ws.cmd("serverGetWrapperNonce", [], (function(_this) {
        return function(wrapper_nonce) {
          # 替换 src 中的 wrapper_nonce
          src = src.replace(/wrapper_nonce=[A-Za-z0-9]+/, "wrapper_nonce=" + wrapper_nonce);
          # 记录日志
          _this.log("Reloading iframe using url", src);
          # 设置 iframe 的 src 属性
          return $("iframe").attr("src", src);
        };
      })(this));
    };

    # 记录日志
    Wrapper.prototype.log = function() {
      var args;
      args = 1 <= arguments.length ? slice.call(arguments, 0) : [];
      return console.log.apply(console, ["[Wrapper]"].concat(slice.call(args)));
    };

    # 创建 Wrapper 对象
    return Wrapper;

  })();

  # 获取服务器地址
  origin = window.server_url || window.location.href.replace(/(\:\/\/.*?)\/.*/, "$1");

  # 根据协议设置 ws 和 http 对象
  if (origin.indexOf("https:") === 0) {
    proto = {
      ws: 'wss',
      http: 'https'
    };
  } else {
    proto = {
      ws: 'ws',
      http: 'http'
    };
  }

  # 拼接 WebSocket 的 URL
  ws_url = proto.ws + ":" + origin.replace(proto.http + ":", "") + "/ZeroNet-Internal/Websocket?wrapper_key=" + window.wrapper_key;

  # 创建 Wrapper 对象
  window.wrapper = new Wrapper(ws_url);
# 匿名函数，用于包装 ZeroFrame 对象
(function() {
  # 定义 WrapperZeroFrame 类
  var WrapperZeroFrame,
    # 定义 bind 函数，用于绑定函数的上下文
    bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };

  # 实现 WrapperZeroFrame 类
  WrapperZeroFrame = (function() {
    # 构造函数，接受一个 wrapper 对象作为参数
    function WrapperZeroFrame(wrapper) {
      # 绑定 this.certSelectGotoSite 函数的上下文
      this.certSelectGotoSite = bind(this.certSelectGotoSite, this);
      # 绑定 this.response 函数的上下文
      this.response = bind(this.response, this);
      # 绑定 this.cmd 函数的上下文
      this.cmd = bind(this.cmd, this);
      # 获取 wrapper 对象的 cmd 属性，并赋值给 this.wrapperCmd
      this.wrapperCmd = wrapper.cmd;
      # 获取 wrapper 对象的 ws.response 属性，并赋值给 this.wrapperResponse
      this.wrapperResponse = wrapper.ws.response;
      # 打印日志
      console.log("WrapperZeroFrame", wrapper);
    }

    # 定义 cmd 方法
    WrapperZeroFrame.prototype.cmd = function(cmd, params, cb) {
      # 如果 params 为 null，则赋值为空对象
      if (params == null) {
        params = {};
      }
      # 如果 cb 为 null，则赋值为 null
      if (cb == null) {
        cb = null;
      }
      # 调用 wrapperCmd 方法，并返回结果
      return this.wrapperCmd(cmd, params, cb);
    };

    # 定义 response 方法
    WrapperZeroFrame.prototype.response = function(to, result) {
      # 调用 wrapperResponse 方法，并返回结果
      return this.wrapperResponse(to, result);
    };

    # 定义 isProxyRequest 方法
    WrapperZeroFrame.prototype.isProxyRequest = function() {
      # 判断当前页面路径是否为根路径
      return window.location.pathname === "/";
    };

    # 定义 certSelectGotoSite 方法
    WrapperZeroFrame.prototype.certSelectGotoSite = function(elem) {
      # 获取元素的 href 属性
      var href;
      href = $(elem).attr("href");
      # 如果是代理请求，则修改 href 属性
      if (this.isProxyRequest()) {
        return $(elem).attr("href", "http://zero" + href);
      }
    };

    # 返回 WrapperZeroFrame 类
    return WrapperZeroFrame;

  })();

  # 创建全局变量 zeroframe，并实例化 WrapperZeroFrame 类
  window.zeroframe = new WrapperZeroFrame(window.wrapper);

}).call(this);
  // 通过 zeroframe 命令获取用户的全局设置
  zeroframe.cmd("userGetGlobalSettings", [], function(user_settings) {
    // 如果用户设置的主题与当前主题不同
    if (user_settings.theme !== theme) {
      // 更新用户设置的主题为当前主题
      user_settings.theme = theme;
      // 通过 zeroframe 命令设置用户的全局设置
      zeroframe.cmd("userSetGlobalSettings", [user_settings], function(status) {
        // 如果设置成功
        if (status === "ok") {
          // 重新加载页面
          location.reload();
        }
      });
    }
  });

  // 显示通知
  displayNotification = function(arg) {
    var matches, media;
    matches = arg.matches, media = arg.media;
    // 如果匹配不到
    if (!matches) {
      return;
    }
    // 通过 zeroframe 命令获取站点信息
    zeroframe.cmd("siteInfo", [], function(site_info) {
      // 如果用户有管理员权限
      if (indexOf.call(site_info.settings.permissions, "ADMIN") >= 0) {
        // 发送管理员通知
        zeroframe.cmd("wrapperNotification", ["info", "Your system's theme has been changed.<br>Please reload site to use it."]);
      } else {
        // 发送普通用户通知
        zeroframe.cmd("wrapperNotification", ["info", "Your system's theme has been changed.<br>Please open ZeroHello to use it."]);
      }
    });
  };

  // 检测颜色方案
  detectColorScheme = function() {
    // 如果是暗色模式
    if (mqDark.matches) {
      // 切换为暗色方案
      changeColorScheme("dark");
    } else if (mqLight.matches) {
      // 如果是亮色模式，切换为亮色方案
      changeColorScheme("light");
    }
    // 监听暗色模式变化，触发显示通知
    mqDark.addListener(displayNotification);
    // 监听亮色模式变化，触发显示通知
    mqLight.addListener(displayNotification);
  };

  // 通过 zeroframe 命令获取用户的全局设置
  zeroframe.cmd("userGetGlobalSettings", [], function(user_settings) {
    // 如果用户设置为使用系统主题
    if (user_settings.use_system_theme === true) {
      // 检测颜色方案
      detectColorScheme();
    }
  });
# 调用一个匿名函数，并将 this 作为参数传入
}).call(this);
```