# `ZeroNet\plugins\UiFileManager\media\js\lib\maquette.js`

```
(function (root, factory) {
    if (typeof define === 'function' && define.amd) {
        // 如果支持 AMD 规范，则注册为匿名模块
        define(['exports'], factory);
    } else if (typeof exports === 'object' && typeof exports.nodeName !== 'string') {
        // 如果支持 CommonJS 规范
        factory(exports);
    } else {
        // 如果是浏览器环境
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
    // Hyperscript 辅助函数
    // 比较两个虚拟节点是否相同
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
    // 将文本数据转换为文本虚拟节点
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
# 代码结尾，可能是一个函数或者代码块的结束
```