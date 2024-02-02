# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\simple.js`

```py
// 使用立即执行函数表达式（IIFE）来定义模块
(function(mod) {
  // 如果是 CommonJS 环境，使用 require 引入 CodeMirror 模块
  if (typeof exports == "object" && typeof module == "object") 
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境，使用 define 引入 CodeMirror 模块
  else if (typeof define == "function" && define.amd) 
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境，直接引入 CodeMirror 模块
  else 
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义一个简单的模式
  CodeMirror.defineSimpleMode = function(name, states) {
    // 定义一个模式，返回一个简单模式的配置
    CodeMirror.defineMode(name, function(config) {
      return CodeMirror.simpleMode(config, states);
    });
  };

  // 实现简单模式
  CodeMirror.simpleMode = function(config, states) {
    // 确保状态对象中包含 "start" 状态
    ensureState(states, "start");
    var states_ = {}, meta = states.meta || {}, hasIndentation = false;
    // 遍历状态对象
    for (var state in states) if (state != meta && states.hasOwnProperty(state)) {
      var list = states_[state] = [], orig = states[state];
      // 遍历状态对象中的规则
      for (var i = 0; i < orig.length; i++) {
        var data = orig[i];
        // 将规则转换为 Rule 对象，并添加到状态列表中
        list.push(new Rule(data, states));
        // 如果规则中包含缩进或取消缩进，则设置 hasIndentation 为 true
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
    # 如果 meta 存在，则将 meta 中的属性添加到 mode 对象中
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
  // 遍历数组 val，将每个元素经过替换操作后放入结果数组 result 中
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
  // 初始化属性计数器
  var props = 0;
  // 遍历对象 a 的属性
  for (var prop in a) if (a.hasOwnProperty(prop)) {
    // 如果 b 中不包含 a 的属性，或者 a[prop] 与 b[prop] 不相等，则返回 false
    if (!b.hasOwnProperty(prop) || !cmp(a[prop], b[prop])) return false;
    // 属性计数器加一
    props++;
  }
  // 遍历对象 b 的属性
  for (var prop in b) if (b.hasOwnProperty(prop)) props--;
  // 返回属性计数器是否为 0
  return props == 0;
}

// 进入本地模式的函数，接受 config、state、spec 和 token 作为参数
function enterLocalMode(config, state, spec, token) {
  var pers;
  // 如果 spec 中包含 persistent 属性，则在 state 中查找对应的状态
  if (spec.persistent) for (var p = state.persistentStates; p && !pers; p = p.next)
    if (spec.spec ? cmp(spec.spec, p.spec) : spec.mode == p.mode) pers = p;
  // 根据 pers 是否存在来确定 mode 和 lState 的值
  var mode = pers ? pers.mode : spec.mode || CodeMirror.getMode(config, spec.spec);
  var lState = pers ? pers.state : CodeMirror.startState(mode);
  // 如果 spec 中包含 persistent 属性，并且 pers 不存在，则将 mode、spec 和 lState 添加到 state 的 persistentStates 中
  if (spec.persistent && !pers)
    state.persistentStates = {mode: mode, spec: spec.spec, state: lState, next: state.persistentStates};

  // 设置 state 的 localState 和 local 属性
  state.localState = lState;
  state.local = {mode: mode,
                 end: spec.end && toRegex(spec.end),
                 endScan: spec.end && spec.forceEnd !== false && toRegex(spec.end, false),
                 endToken: token && token.join ? token[token.length - 1] : token};
}

// 查找函数 indexOf，用于在数组 arr 中查找 val
function indexOf(val, arr) {
  // 遍历数组 arr，如果找到与 val 相等的元素，则返回 true
  for (var i = 0; i < arr.length; i++) if (arr[i] === val) return true;
}

// 缩进函数 indentFunction，接受 states 和 meta 作为参数
    # 定义一个函数，用于处理缩进
    return function(state, textAfter, line) {
      # 如果存在本地状态并且本地状态有缩进模式，则使用本地状态的缩进模式
      if (state.local && state.local.mode.indent)
        return state.local.mode.indent(state.localState, textAfter, line);
      # 如果缩进值为空或者存在本地状态或者当前状态在不缩进状态列表中，则返回 CodeMirror.Pass
      if (state.indent == null || state.local || meta.dontIndentStates && indexOf(state.state, meta.dontIndentStates) > -1)
        return CodeMirror.Pass;

      # 初始化位置和规则
      var pos = state.indent.length - 1, rules = states[state.state];
      # 循环扫描规则
      scan: for (;;) {
        for (var i = 0; i < rules.length; i++) {
          var rule = rules[i];
          # 如果规则包含取消缩进并且不是行首取消缩进
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
      # 返回缩进值
      return pos < 0 ? 0 : state.indent[pos];
    };
  }
# 闭合了一个代码块或者函数的结束
```