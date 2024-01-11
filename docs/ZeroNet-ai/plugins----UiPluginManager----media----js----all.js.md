# `ZeroNet\plugins\UiPluginManager\media\js\all.js`

```
# 定义一个 Class 类
(function() {
  var Class,
    slice = [].slice;

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
      # 在日志信息前添加类名，然后调用 console.log 打印日志
      args.unshift("[" + this.constructor.name + "]");
      console.log.apply(console, args);
      return this;
    };

    # 定义 logStart 方法，用于记录操作开始时间
    Class.prototype.logStart = function() {
      var args, name;
      name = arguments[0], args = 2 <= arguments.length ? slice.call(arguments, 1) : [];
      # 如果 trace 为 false，则直接返回
      if (!this.trace) {
        return;
      }
      # 如果 logtimers 未定义，则初始化为一个空对象
      this.logtimers || (this.logtimers = {});
      # 记录操作开始时间
      this.logtimers[name] = +(new Date);
      # 如果有参数，则调用 log 方法打印日志
      if (args.length > 0) {
        this.log.apply(this, ["" + name].concat(slice.call(args), ["(started)"]));
      }
      return this;
    };

    # 定义 logEnd 方法，用于记录操作结束时间
    Class.prototype.logEnd = function() {
      var args, ms, name;
      name = arguments[0], args = 2 <= arguments.length ? slice.call(arguments, 1) : [];
      # 计算操作耗时
      ms = +(new Date) - this.logtimers[name];
      # 调用 log 方法打印日志
      this.log.apply(this, ["" + name].concat(slice.call(args), ["(Done in " + ms + "ms)"]));
      return this;
    };

    return Class;

  })();

  # 将 Class 类绑定到全局对象 window 上
  window.Class = Class;

}).call(this);
    # 定义一个名为 when 的方法，用于并行执行多个任务并等待它们全部完成
    Promise.when = function() {
      # 初始化变量
      var args, fn, i, len, num_uncompleted, promise, task, task_id, tasks;
      # 获取传入的任务列表
      tasks = 1 <= arguments.length ? slice.call(arguments, 0) : [];
      # 初始化未完成任务的数量
      num_uncompleted = tasks.length;
      # 创建一个数组用于存储每个任务的结果
      args = new Array(num_uncompleted);
      # 创建一个新的 Promise 对象
      promise = new Promise();
      # 定义一个函数，用于处理每个任务的完成情况
      fn = function(task_id) {
        return task.then(function() {
          # 将任务的结果存入数组
          args[task_id] = Array.prototype.slice.call(arguments);
          # 未完成任务数量减一
          num_uncompleted--;
          # 如果所有任务都完成了，则调用 promise 的 complete 方法
          if (num_uncompleted === 0) {
            return promise.complete.apply(promise, args);
          }
        });
      };
      # 遍历任务列表，对每个任务调用 fn 函数
      for (task_id = i = 0, len = tasks.length; i < len; task_id = ++i) {
        task = tasks[task_id];
        fn(task_id);
      }
      # 返回 promise 对象
      return promise;
    };

    # 定义一个名为 Promise 的构造函数
    function Promise() {
      # 初始化 Promise 对象的属性
      this.resolved = false;
      this.end_promise = null;
      this.result = null;
      this.callbacks = [];
    }

    # 定义 Promise 对象的 resolve 方法
    Promise.prototype.resolve = function() {
      # 如果 Promise 对象已经被解决，则返回 false
      var back, callback, i, len, ref;
      if (this.resolved) {
        return false;
      }
      # 标记 Promise 对象为已解决状态
      this.resolved = true;
      # 存储传入的参数
      this.data = arguments;
      # 如果没有传入参数，则默认设置为 [true]
      if (!arguments.length) {
        this.data = [true];
      }
      # 获取结果值
      this.result = this.data[0];
      # 遍历回调函数列表，执行回调函数
      ref = this.callbacks;
      for (i = 0, len = ref.length; i < len; i++) {
        callback = ref[i];
        back = callback.apply(callback, this.data);
      }
      # 如果存在 end_promise，则调用其 resolve 方法
      if (this.end_promise) {
        return this.end_promise.resolve(back);
      }
    };

    # 定义 Promise 对象的 fail 方法
    Promise.prototype.fail = function() {
      # 调用 resolve 方法，传入 false
      return this.resolve(false);
    };

    # 定义 Promise 对象的 then 方法
    Promise.prototype.then = function(callback) {
      # 如果 Promise 对象已经解决，则直接执行回调函数
      if (this.resolved === true) {
        callback.apply(callback, this.data);
        return;
      }
      # 将回调函数添加到回调函数列表中，并返回一个新的 Promise 对象
      this.callbacks.push(callback);
      return this.end_promise = new Promise();
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
// 结束前面的函数调用链
}).call(this);

// 定义字符串原型的方法，用于判断字符串是否以指定字符串开头
String.prototype.startsWith = function(s) {
    return this.slice(0, s.length) === s;
};

// 定义字符串原型的方法，用于判断字符串是否以指定字符串结尾
String.prototype.endsWith = function(s) {
    return s === '' || this.slice(-s.length) === s;
};

// 定义字符串原型的方法，用于重复字符串
String.prototype.repeat = function(count) {
    return new Array(count + 1).join(this);
};

// 定义全局函数，用于判断对象是否为空
window.isEmpty = function(obj) {
    var key;
    for (key in obj) {
        return false;
    }
    return true;
};

// 匿名函数自调用，用于定义 maquette 模块
(function (root, factory) {
    if (typeof define === 'function' && define.amd) {
        // 如果支持 AMD 规范，则注册为匿名模块
        define(['exports'], factory);
    } else if (typeof exports === 'object' && typeof exports.nodeName !== 'string') {
        // 如果支持 CommonJS 规范，则注册为模块
        factory(exports);
    } else {
        // 否则注册为全局变量
        factory(root.maquette = {});
    }
}(this, function (exports) {
    'use strict';
    // 命名空间常量
    var NAMESPACE_W3 = 'http://www.w3.org/';
    var NAMESPACE_SVG = NAMESPACE_W3 + '2000/svg';
    var NAMESPACE_XLINK = NAMESPACE_W3 + '1999/xlink';
    // 实用工具
    var emptyArray = [];
    // 扩展对象的方法
    var extend = function (base, overrides) {
        var result = {};
        Object.keys(base).forEach(function (key) {
            result[key] = base[key];
        });
        if (overrides) {
            Object.keys(overrides).forEach(function (key) {
                result[key] = overrides[key];
            });
        }
        return result;
    };
    // 超文本标记语言辅助函数
    # 检查两个虚拟节点是否相同
    var same = function (vnode1, vnode2) {
        # 如果虚拟节点选择器不同，返回 false
        if (vnode1.vnodeSelector !== vnode2.vnodeSelector) {
            return false;
        }
        # 如果虚拟节点都有属性
        if (vnode1.properties && vnode2.properties) {
            # 如果属性中的 key 不同，返回 false
            if (vnode1.properties.key !== vnode2.properties.key) {
                return false;
            }
            # 返回属性中的 bind 是否相同
            return vnode1.properties.bind === vnode2.properties.bind;
        }
        # 如果虚拟节点都没有属性，返回 true
        return !vnode1.properties && !vnode2.properties;
    };
    # 将数据转换为文本虚拟节点
    var toTextVNode = function (data) {
        return {
            vnodeSelector: '',
            properties: undefined,
            children: undefined,
            text: data.toString(),
            domNode: null
        };
    };
    # 将子节点添加到父节点
    var appendChildren = function (parentSelector, insertions, main) {
        for (var i = 0; i < insertions.length; i++) {
            var item = insertions[i];
            # 如果是数组，递归调用 appendChildren
            if (Array.isArray(item)) {
                appendChildren(parentSelector, item, main);
            } else {
                # 如果不是 null 或 undefined
                if (item !== null && item !== undefined) {
                    # 如果 item 没有 vnodeSelector 属性，转换为文本虚拟节点
                    if (!item.hasOwnProperty('vnodeSelector')) {
                        item = toTextVNode(item);
                    }
                    # 将 item 添加到 main 数组中
                    main.push(item);
                }
            }
        }
    };
    # 抛出错误，提示提供转换对象以执行动画
    var missingTransition = function () {
        throw new Error('Provide a transitions object to the projectionOptions to do animations');
    };
    # 默认的投影选项
    var DEFAULT_PROJECTION_OPTIONS = {
        namespace: undefined,
        eventHandlerInterceptor: undefined,
        # 应用样式的函数
        styleApplyer: function (domNode, styleName, value) {
            # 提供一个钩子来为仍然需要它的浏览器添加供应商前缀
            domNode.style[styleName] = value;
        },
        # 过渡动画
        transitions: {
            enter: missingTransition,
            exit: missingTransition
        }
    };
    # 应用默认的投影选项
    var applyDefaultProjectionOptions = function (projectorOptions) {
        return extend(DEFAULT_PROJECTION_OPTIONS, projectorOptions);
    };
    // 定义一个函数，用于检查样式值是否为字符串，如果不是则抛出错误
    var checkStyleValue = function (styleValue) {
        if (typeof styleValue !== 'string') {
            throw new Error('Style values must be strings');
        }
    };

    // 定义一个函数，用于查找子节点中与指定节点相同的节点的索引
    var findIndexOfChild = function (children, sameAs, start) {
        if (sameAs.vnodeSelector !== '') {
            // 不要扫描文本节点
            for (var i = start; i < children.length; i++) {
                if (same(children[i], sameAs)) {
                    return i;
                }
            }
        }
        return -1;
    };

    // 定义一个函数，用于处理节点添加时的动画效果
    var nodeAdded = function (vNode, transitions) {
        if (vNode.properties) {
            var enterAnimation = vNode.properties.enterAnimation;
            if (enterAnimation) {
                if (typeof enterAnimation === 'function') {
                    enterAnimation(vNode.domNode, vNode.properties);
                } else {
                    transitions.enter(vNode.domNode, vNode.properties, enterAnimation);
                }
            }
        }
    };

    // 定义一个函数，用于处理节点移除时的动画效果
    var nodeToRemove = function (vNode, transitions) {
        var domNode = vNode.domNode;
        if (vNode.properties) {
            var exitAnimation = vNode.properties.exitAnimation;
            if (exitAnimation) {
                domNode.style.pointerEvents = 'none';
                var removeDomNode = function () {
                    if (domNode.parentNode) {
                        domNode.parentNode.removeChild(domNode);
                    }
                };
                if (typeof exitAnimation === 'function') {
                    exitAnimation(domNode, removeDomNode, vNode.properties);
                    return;
                } else {
                    transitions.exit(vNode.domNode, vNode.properties, exitAnimation, removeDomNode);
                    return;
                }
            }
        }
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
// 定义 Animation 类
(function() {
  var Animation;

  Animation = (function() {
    function Animation() {}

    // 定义 slideDown 方法，用于元素的下滑动画
    Animation.prototype.slideDown = function(elem, props) {
      // 获取元素的当前高度和样式
      var cstyle, h, margin_bottom, margin_top, padding_bottom, padding_top, transition;
      // 如果元素的 offsetTop 大于 2000，则直接返回，不执行动画
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
      // 延迟执行动画，避免立即执行导致动画失效
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
// 调用立即执行函数，将 this 绑定到全局对象
}).call(this);

// Dollar.coffee 文件定义
/* ---- utils/Dollar.coffee ---- */

// 定义立即执行函数
(function() {
  // 定义全局函数 $，根据选择器返回对应的 DOM 元素
  window.$ = function(selector) {
    // 如果选择器以 # 开头，则返回对应 id 的 DOM 元素
    if (selector.startsWith("#")) {
      return document.getElementById(selector.replace("#", ""));
    }
  };
}).call(this);

// ZeroFrame.coffee 文件定义
/* ---- utils/ZeroFrame.coffee ---- */

// 定义立即执行函数
(function() {
  // 定义 ZeroFrame 类
  var ZeroFrame,
    // 定义辅助函数 bind、extend、hasProp
    bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; },
    extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
    hasProp = {}.hasOwnProperty;

  // ZeroFrame 类继承自 superClass
  ZeroFrame = (function(superClass) {
    // 扩展 ZeroFrame 类
    extend(ZeroFrame, superClass);

    // ZeroFrame 类构造函数
    function ZeroFrame(url) {
      // 绑定 this 到 onCloseWebsocket 方法
      this.onCloseWebsocket = bind(this.onCloseWebsocket, this);
      // 绑定 this 到 onOpenWebsocket 方法
      this.onOpenWebsocket = bind(this.onOpenWebsocket, this);
      // 绑定 this 到 onRequest 方法
      this.onRequest = bind(this.onRequest, this);
      // 绑定 this 到 onMessage 方法
      this.onMessage = bind(this.onMessage, this);
      // 设置 ZeroFrame 对象的 url 属性
      this.url = url;
      // 初始化等待回调的对象
      this.waiting_cb = {};
      // 获取当前页面的 wrapper_nonce
      this.wrapper_nonce = document.location.href.replace(/.*wrapper_nonce=([A-Za-z0-9]+).*/, "$1");
      // 连接到 ZeroNet
      this.connect();
      // 设置下一个消息的 id
      this.next_message_id = 1;
      // 初始化历史状态对象
      this.history_state = {};
      // 调用 init 方法
      this.init();
    }

    // ZeroFrame 类的 init 方法
    ZeroFrame.prototype.init = function() {
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
// 调用立即执行函数，将 this 绑定到全局对象
}).call(this);

// 定义 PluginList 类
/* ---- PluginList.coffee ---- */
(function() {
  var PluginList,
    // 定义 bind 函数，用于绑定函数的 this 值
    bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; },
    // 定义 extend 函数，用于实现继承
    extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
    // 定义 hasProp 对象，用于判断对象是否有指定属性
    hasProp = {}.hasOwnProperty;

  // 定义 PluginList 类
  PluginList = (function(superClass) {
    // 继承 superClass
    extend(PluginList, superClass);

    // 定义 PluginList 构造函数
    function PluginList(plugins) {
      // 绑定各个方法的 this 值
      this.handleDeleteClick = bind(this.handleDeleteClick, this);
      this.handleUpdateClick = bind(this.handleUpdateClick, this);
      this.handleResetClick = bind(this.handleResetClick, this);
      this.handleCheckboxChange = bind(this.handleCheckboxChange, this);
      this.savePluginStatus = bind(this.savePluginStatus, this);
      // 初始化插件列表
      this.plugins = plugins;
    }

    // 定义保存插件状态的方法
    PluginList.prototype.savePluginStatus = function(plugin, is_enabled) {
      // 调用 Page.cmd 方法，设置插件配置
      Page.cmd("pluginConfigSet", [plugin.source, plugin.inner_path, "enabled", is_enabled], (function(_this) {
        return function(res) {
          // 根据返回结果进行处理
          if (res === "ok") {
            return Page.updatePlugins();
          } else {
            return Page.cmd("wrapperNotification", ["error", res.error]);
          }
        };
      })(this));
      // 调度重新渲染页面
      return Page.projector.scheduleRender();
    };

    // 处理复选框改变事件的方法
    PluginList.prototype.handleCheckboxChange = function(e) {
      var node, plugin, value;
      node = e.currentTarget;
      plugin = node["data-plugin"];
      // 切换复选框的选中状态
      node.classList.toggle("checked");
      value = node.classList.contains("checked");
      // 保存插件状态
      return this.savePluginStatus(plugin, value);
    };

    // 处理重置按钮点击事件的方法
    PluginList.prototype.handleResetClick = function(e) {
      var node, plugin;
      node = e.currentTarget;
      plugin = node["data-plugin"];
      // 重置插件状态
      return this.savePluginStatus(plugin, null);
    };
    # 处理更新按钮的点击事件
    PluginList.prototype.handleUpdateClick = function(e) {
      # 获取当前点击的节点
      var node, plugin;
      node = e.currentTarget;
      # 获取节点上的插件信息
      plugin = node["data-plugin"];
      # 给节点添加加载样式
      node.classList.add("loading");
      # 调用Page.cmd方法，请求更新插件
      Page.cmd("pluginUpdate", [plugin.source, plugin.inner_path], (function(_this) {
        return function(res) {
          # 如果更新成功
          if (res === "ok") {
            # 发送通知，提示插件已更新到最新版本
            Page.cmd("wrapperNotification", ["done", "Plugin " + plugin.name + " updated to latest version"]);
            # 更新插件列表
            Page.updatePlugins();
          } else {
            # 如果更新失败，发送错误通知
            Page.cmd("wrapperNotification", ["error", res.error]);
          }
          # 移除加载样式
          return node.classList.remove("loading");
        };
      })(this));
      # 阻止默认事件
      return false;
    };

    # 处理删除按钮的点击事件
    PluginList.prototype.handleDeleteClick = function(e) {
      # 获取当前点击的节点
      var node, plugin;
      node = e.currentTarget;
      # 获取节点上的插件信息
      plugin = node["data-plugin"];
      # 如果插件已加载，提示无法删除
      if (plugin.loaded) {
        Page.cmd("wrapperNotification", ["info", "You can only delete plugin that are not currently active"]);
        return false;
      }
      # 给节点添加加载样式
      node.classList.add("loading");
      # 弹出确认框，确认是否删除插件
      Page.cmd("wrapperConfirm", ["Delete " + plugin.name + " plugin?", "Delete"], (function(_this) {
        return function(res) {
          # 如果取消删除
          if (!res) {
            # 移除加载样式，返回
            node.classList.remove("loading");
            return false;
          }
          # 调用Page.cmd方法，请求删除插件
          return Page.cmd("pluginRemove", [plugin.source, plugin.inner_path], function(res) {
            # 如果删除成功
            if (res === "ok") {
              # 发送通知，提示插件已删除
              Page.cmd("wrapperNotification", ["done", "Plugin " + plugin.name + " deleted"]);
              # 更新插件列表
              Page.updatePlugins();
            } else {
              # 如果删除失败，发送错误通知
              Page.cmd("wrapperNotification", ["error", res.error]);
            }
            # 移除加载样式
            return node.classList.remove("loading");
          });
        };
      })(this));
      # 阻止默认事件
      return false;
    };

    # 返回插件列表
    return PluginList;

  })(Class);

  # 将PluginList赋值给window对象的属性
  window.PluginList = PluginList;
// 调用一个匿名函数，并将 this 绑定到当前对象
}).call(this);

// 定义 UiPluginManager 类
/* ---- UiPluginManager.coffee ---- */
(function() {
  var UiPluginManager,
    // 定义 bind 函数，用于绑定函数的 this
    bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; },
    // 定义 extend 函数，用于继承父类的属性和方法
    extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
    // 定义 hasProp 对象，用于判断属性是否存在
    hasProp = {}.hasOwnProperty;

  // 将 maquette.h 赋值给 window.h
  window.h = maquette.h;

  // 定义 UiPluginManager 类
  UiPluginManager = (function(superClass) {
    // 继承 superClass 的属性和方法
    extend(UiPluginManager, superClass);

    // 定义 UiPluginManager 类的构造函数
    function UiPluginManager() {
      // 绑定 this 到 renderBottomRestart 方法
      this.renderBottomRestart = bind(this.renderBottomRestart, this);
      // 绑定 this 到 handleRestartClick 方法
      this.handleRestartClick = bind(this.handleRestartClick, this);
      // 绑定 this 到 render 方法
      this.render = bind(this.render, this);
      // 绑定 this 到 createProjector 方法
      this.createProjector = bind(this.createProjector, this);
      // 绑定 this 到 updatePlugins 方法
      this.updatePlugins = bind(this.updatePlugins, this);
      // 绑定 this 到 onOpenWebsocket 方法
      this.onOpenWebsocket = bind(this.onOpenWebsocket, this);
      // 调用父类的构造函数
      return UiPluginManager.__super__.constructor.apply(this, arguments);
    }

    // 定义 UiPluginManager 类的 init 方法
    UiPluginManager.prototype.init = function() {
      // 初始化内置插件列表
      this.plugin_list_builtin = new PluginList();
      // 初始化自定义插件列表
      this.plugin_list_custom = new PluginList();
      // 插件列表是否发生变化
      this.plugins_changed = null;
      // 是否需要重新启动
      this.need_restart = null;
      return this;
    };

    // 定义 UiPluginManager 类的 onOpenWebsocket 方法
    UiPluginManager.prototype.onOpenWebsocket = function() {
      // 设置页面标题
      this.cmd("wrapperSetTitle", "Plugin manager - ZeroNet");
      // 获取服务器信息
      this.cmd("serverInfo", {}, (function(_this) {
        return function(server_info) {
          return _this.server_info = server_info;
        };
      })(this));
      // 更新插件列表
      return this.updatePlugins();
    };
    // 更新插件列表的方法
    UiPluginManager.prototype.updatePlugins = function(cb) {
      // 调用 cmd 方法发送 pluginList 命令，获取插件列表
      return this.cmd("pluginList", [], (function(_this) {
        // 返回一个函数，处理获取到的插件列表
        return function(res) {
          // 定义变量，存储插件列表中需要更新的插件
          var item, plugins_builtin, plugins_custom;
          // 根据条件筛选出需要更新的插件
          _this.plugins_changed = (function() {
            var i, len, ref, results;
            ref = res.plugins;
            results = [];
            for (i = 0, len = ref.length; i < len; i++) {
              item = ref[i];
              if (item.enabled !== item.loaded || item.updated) {
                results.push(item);
              }
            }
            return results;
          })();
          // 筛选出内置插件
          plugins_builtin = (function() {
            var i, len, ref, results;
            ref = res.plugins;
            results = [];
            for (i = 0, len = ref.length; i < len; i++) {
              item = ref[i];
              if (item.source === "builtin") {
                results.push(item);
              }
            }
            return results;
          })();
          // 对内置插件按名称排序
          _this.plugin_list_builtin.plugins = plugins_builtin.sort(function(a, b) {
            return a.name.localeCompare(b.name);
          });
          // 筛选出自定义插件
          plugins_custom = (function() {
            var i, len, ref, results;
            ref = res.plugins;
            results = [];
            for (i = 0, len = ref.length; i < len; i++) {
              item = ref[i];
              if (item.source !== "builtin") {
                results.push(item);
              }
            }
            return results;
          })();
          // 对自定义插件按名称排序
          _this.plugin_list_custom.plugins = plugins_custom.sort(function(a, b) {
            return a.name.localeCompare(b.name);
          });
          // 调用 projector 的 scheduleRender 方法，更新渲染
          _this.projector.scheduleRender();
          // 如果传入了回调函数，则执行回调函数
          return typeof cb === "function" ? cb() : void 0;
        };
      })(this));
    };
    // 创建投影仪，用于渲染 UI
    UiPluginManager.prototype.createProjector = function() {
      // 使用 maquette 库创建投影仪
      this.projector = maquette.createProjector();
      // 用投影仪替换指定的 DOM 元素内容为渲染结果
      this.projector.replace($("#content"), this.render);
      // 用投影仪替换指定的 DOM 元素内容为底部重新启动按钮的渲染结果
      return this.projector.replace($("#bottom-restart"), this.renderBottomRestart);
    };

    // 渲染函数，根据插件列表渲染 UI
    UiPluginManager.prototype.render = function() {
      var ref;
      // 如果没有内置插件，则返回一个空的 div
      if (!this.plugin_list_builtin.plugins) {
        return h("div.content");
      }
      // 根据插件列表渲染 UI
      return h("div.content", [h("div.section", [((ref = this.plugin_list_custom.plugins) != null ? ref.length : void 0) ? [h("h2", "Installed third-party plugins"), this.plugin_list_custom.render()] : void 0, h("h2", "Built-in plugins"), this.plugin_list_builtin.render()])]);
    };

    // 处理重新启动按钮点击事件
    UiPluginManager.prototype.handleRestartClick = function() {
      // 设置重新启动加载状态为 true
      this.restart_loading = true;
      // 在 300 毫秒后执行服务器关闭并重启操作
      setTimeout(((function(_this) {
        return function() {
          return Page.cmd("serverShutdown", {
            restart: true
          });
        };
      })(this)), 300);
      // 调度投影仪重新渲染
      Page.projector.scheduleRender();
      // 阻止默认行为
      return false;
    };

    // 渲染底部重新启动按钮
    UiPluginManager.prototype.renderBottomRestart = function() {
      var ref;
      // 返回底部重新启动按钮的渲染结果
      return h("div.bottom.bottom-restart", {
        classes: {
          // 根据插件是否改变设置按钮是否可见
          visible: (ref = this.plugins_changed) != null ? ref.length : void 0
        }
      }, h("div.bottom-content", [
        h("div.title", "Some plugins status has been changed"), h("a.button.button-submit.button-restart", {
          href: "#Restart",
          classes: {
            // 根据重新启动加载状态设置按钮样式
            loading: this.restart_loading
          },
          // 设置按钮点击事件处理函数
          onclick: this.handleRestartClick
        }, "Restart ZeroNet client")
      ]));
    };

    // 实例化 UiPluginManager 类，并调用创建投影仪方法
    return UiPluginManager;

  })(ZeroFrame);

  // 创建全局 Page 对象，并调用创建投影仪方法
  window.Page = new UiPluginManager();

  window.Page.createProjector();
# 调用匿名函数，并将当前上下文作为参数传入
}).call(this);
```