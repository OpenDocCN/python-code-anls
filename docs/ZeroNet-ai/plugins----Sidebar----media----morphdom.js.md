# `ZeroNet\plugins\Sidebar\media\morphdom.js`

```py
// 定义一个立即执行函数，用于导出 morphdom 模块
(function(f){
    // 如果是 CommonJS 环境，则导出模块
    if(typeof exports==="object" && typeof module!=="undefined"){
        module.exports=f()
    }
    // 如果是 AMD 环境，则定义模块
    else if(typeof define==="function" && define.amd){
        define([],f)
    }
    // 否则将模块挂载到全局对象上
    else{
        var g;
        if(typeof window!=="undefined"){
            g=window
        }
        else if(typeof global!=="undefined"){
            g=global
        }
        else if(typeof self!=="undefined"){
            g=self
        }
        else{
            g=this
        }
        g.morphdom = f()
    }
})(function(){
    // 定义特殊元素处理器对象
    var specialElHandlers = {
        // 为了兼容 IE，处理 OPTION 元素的 selected 属性
        OPTION: function(fromEl, toEl) {
            if ((fromEl.selected = toEl.selected)) {
                fromEl.setAttribute('selected', '');
            } else {
                fromEl.removeAttribute('selected', '');
            }
        },
        // 处理 INPUT 元素的 checked 和 value 属性
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
});
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
    // 处理给定的两个 DOM 节点不兼容的情况（例如 <div> --> <span> 或 <div> --> TEXT）
    if (morphedNodeType === 1) {
        // 如果目标节点是元素节点
        if (toNodeType === 1) {
            // 如果变换后的节点的标签名与目标节点的标签名不同
            if (morphedNode.tagName !== toNode.tagName) {
                // 触发“onNodeDiscarded”事件，丢弃原节点
                onNodeDiscarded(fromNode);
                // 将变换后的节点的子节点移动到新创建的与目标节点标签名相同的元素节点中
                morphedNode = moveChildren(morphedNode, document.createElement(toNode.tagName));
            }
        } else {
            // 从元素节点变换为文本节点
            return toNode;
        }
    } else if (morphedNodeType === 3) { // 文本节点
        // 如果目标节点也是文本节点
        if (toNodeType === 3) {
            // 更新变换后的节点的文本内容为目标节点的文本内容
            morphedNode.nodeValue = toNode.nodeValue;
            return morphedNode;
        } else {
            // 触发“onNodeDiscarded”事件，丢弃原节点
            onNodeDiscarded(fromNode);
            // 文本节点变换为其他类型节点
            return toNode;
        }
    }

    // 对变换后的节点和目标节点进行进一步的处理
    morphEl(morphedNode, toNode, false);

    // 触发“onNodeDiscarded”事件，丢弃任何未找到新位置的保存的元素节点
    for (var savedElId in savedEls) {
        if (savedEls.hasOwnProperty(savedElId)) {
            var savedEl = savedEls[savedElId];
            onNodeDiscarded(savedEl);
            walkDiscardedChildNodes(savedEl);
        }
    }

    // 如果变换后的节点不等于原节点且原节点有父节点
    if (morphedNode !== fromNode && fromNode.parentNode) {
        // 用变换后的节点替换原节点
        fromNode.parentNode.replaceChild(morphedNode, fromNode);
    }

    // 返回变换后的节点
    return morphedNode;
# 导出模块中的 morphdom 函数
module.exports = morphdom;
# 结束模块定义
},{}]},{},[1])(1)
# 执行模块，传入参数 1
});
```