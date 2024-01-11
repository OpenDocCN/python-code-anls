# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\search\searchcursor.js`

```
// 使用立即执行函数表达式（IIFE）将模块化的代码封装起来，避免变量污染全局作用域
(function(mod) {
  // 如果是 CommonJS 环境，使用 require 引入 CodeMirror 模块
  if (typeof exports == "object" && typeof module == "object") 
    mod(require("../../lib/codemirror"))
  // 如果是 AMD 环境，使用 define 引入 CodeMirror 模块
  else if (typeof define == "function" && define.amd) 
    define(["../../lib/codemirror"], mod)
  // 如果是普通的浏览器环境，直接引入 CodeMirror 模块
  else 
    mod(CodeMirror)
})(function(CodeMirror) {
  "use strict"
  // 定义变量 Pos 为 CodeMirror.Pos
  var Pos = CodeMirror.Pos

  // 获取正则表达式的标志位
  function regexpFlags(regexp) {
    var flags = regexp.flags
    return flags != null ? flags : (regexp.ignoreCase ? "i" : "")
      + (regexp.global ? "g" : "")
      + (regexp.multiline ? "m" : "")
  }

  // 确保正则表达式具有指定的标志位
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

  // 在文档中向前搜索匹配正则表达式的内容
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

  // 在文档中向前搜索匹配多行正则表达式的内容
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
        // 如果当前行末尾不是查询字符串的起始部分，则继续搜索
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
# 闭合了一个代码块或者函数的结束
```