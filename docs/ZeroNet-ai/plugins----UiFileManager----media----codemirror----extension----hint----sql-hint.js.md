# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\hint\sql-hint.js`

```py
// 使用立即执行函数表达式（IIFE）来创建一个模块化的作用域
(function(mod) {
  // 如果是 CommonJS 环境，则使用 require 导入依赖模块
  if (typeof exports == "object" && typeof module == "object") // CommonJS
    mod(require("../../lib/codemirror"), require("../../mode/sql/sql"));
  // 如果是 AMD 环境，则使用 define 来定义模块
  else if (typeof define == "function" && define.amd) // AMD
    define(["../../lib/codemirror", "../../mode/sql/sql"], mod);
  // 如果是普通的浏览器环境，则直接使用全局变量 CodeMirror
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 声明变量
  var tables;
  var defaultTable;
  var keywords;
  var identifierQuote;
  // 定义常量对象
  var CONS = {
    QUERY_DIV: ";",
    ALIAS_KEYWORD: "AS"
  };
  // 定义 Pos 函数和 cmpPos 函数
  var Pos = CodeMirror.Pos, cmpPos = CodeMirror.cmpPos;

  // 定义一个函数，用于判断一个值是否为数组
  function isArray(val) { return Object.prototype.toString.call(val) == "[object Array]" }

  // 获取编辑器中的关键字
  function getKeywords(editor) {
    var mode = editor.doc.modeOption;
    if (mode === "sql") mode = "text/x-sql";
    return CodeMirror.resolveMode(mode).keywords;
  }

  // 获取编辑器中的标识符引用符号
  function getIdentifierQuote(editor) {
    var mode = editor.doc.modeOption;
    if (mode === "sql") mode = "text/x-sql";
    return CodeMirror.resolveMode(mode).identifierQuote || "`";
  }

  // 获取 item 的文本内容
  function getText(item) {
    return typeof item == "string" ? item : item.text;
  }

  // 将表名和值进行包装
  function wrapTable(name, value) {
    if (isArray(value)) value = {columns: value}
    if (!value.text) value.text = name
    return value
  }

  // 解析输入的表格数据
  function parseTables(input) {
    var result = {}
    if (isArray(input)) {
      for (var i = input.length - 1; i >= 0; i--) {
        var item = input[i]
        result[getText(item).toUpperCase()] = wrapTable(getText(item), item)
      }
    } else if (input) {
      for (var name in input)
        result[name.toUpperCase()] = wrapTable(name, input[name])
    }
    return result
  }

  // 根据表名获取表格数据
  function getTable(name) {
    return tables[name.toUpperCase()]
  }

  // 浅拷贝对象
  function shallowClone(object) {
    var result = {};
  // 遍历对象的属性，将属性和属性值复制到新的对象中
  for (var key in object) if (object.hasOwnProperty(key))
    result[key] = object[key];
  // 返回复制后的对象
  return result;
}

// 匹配字符串和单词，忽略大小写
function match(string, word) {
  // 获取字符串的长度
  var len = string.length;
  // 获取单词对应的子字符串
  var sub = getText(word).substr(0, len);
  // 比较字符串和子字符串是否相等（忽略大小写）
  return string.toUpperCase() === sub.toUpperCase();
}

// 添加匹配结果到数组中
function addMatches(result, search, wordlist, formatter) {
  // 如果单词列表是数组
  if (isArray(wordlist)) {
    // 遍历单词列表
    for (var i = 0; i < wordlist.length; i++)
      // 如果匹配成功，将格式化后的单词添加到结果数组中
      if (match(search, wordlist[i])) result.push(formatter(wordlist[i]))
  } else {
    // 如果单词列表是对象
    for (var word in wordlist) if (wordlist.hasOwnProperty(word)) {
      var val = wordlist[word]
      // 如果值为空或为true，将值设置为单词本身
      if (!val || val === true)
        val = word
      else
        // 如果值有displayText属性，将值设置为包含text和displayText属性的对象，否则设置为text属性的值
        val = val.displayText ? {text: val.text, displayText: val.displayText} : val.text
      // 如果匹配成功，将格式化后的值添加到结果数组中
      if (match(search, val)) result.push(formatter(val))
    }
  }
}

// 清理名称，去除标识符引号和前导点号
function cleanName(name) {
  // 如果名称以点号开头，去除点号
  if (name.charAt(0) == ".") {
    name = name.substr(1);
  }
  // 替换重复的标识符引号为单个标识符引号，并移除单个标识符引号
  var nameParts = name.split(identifierQuote+identifierQuote);
  for (var i = 0; i < nameParts.length; i++)
    nameParts[i] = nameParts[i].replace(new RegExp(identifierQuote,"g"), "");
  return nameParts.join(identifierQuote);
}

// 插入标识符引号
function insertIdentifierQuotes(name) {
  var nameParts = getText(name).split(".");
  for (var i = 0; i < nameParts.length; i++)
    nameParts[i] = identifierQuote +
      // 重复标识符引号
      nameParts[i].replace(new RegExp(identifierQuote,"g"), identifierQuote+identifierQuote) +
      identifierQuote;
  var escaped = nameParts.join(".");
  // 如果名称是字符串，返回转义后的名称
  if (typeof name == "string") return escaped;
  // 克隆名称对象，设置text属性为转义后的名称
  name = shallowClone(name);
  name.text = escaped;
  return name;
}

// 名称完成
function nameCompletion(cur, token, result, editor) {
    // 尝试完成表格、列名，并返回完成的起始位置
    var useIdentifierQuotes = false;  // 是否使用标识符引号
    var nameParts = [];  // 存储表格和列名的数组
    var start = token.start;  // 记录起始位置
    var cont = true;  // 控制循环的条件
    while (cont) {
      cont = (token.string.charAt(0) == ".");  // 判断是否继续循环
      useIdentifierQuotes = useIdentifierQuotes || (token.string.charAt(0) == identifierQuote);  // 判断是否使用标识符引号

      start = token.start;  // 更新起始位置
      nameParts.unshift(cleanName(token.string));  // 将清理后的表格和列名加入数组

      token = editor.getTokenAt(Pos(cur.line, token.start));  // 获取下一个标记
      if (token.string == ".") {
        cont = true;  // 如果下一个标记是"."，继续循环
        token = editor.getTokenAt(Pos(cur.line, token.start));  // 获取下一个标记
      }
    }

    // 尝试完成表格名
    var string = nameParts.join(".");  // 将表格和列名数组连接成字符串
    addMatches(result, string, tables, function(w) {
      return useIdentifierQuotes ? insertIdentifierQuotes(w) : w;  // 如果使用标识符引号，则插入引号
    });

    // 尝试完成默认表格的列名
    addMatches(result, string, defaultTable, function(w) {
      return useIdentifierQuotes ? insertIdentifierQuotes(w) : w;  // 如果使用标识符引号，则插入引号
    });

    // 尝试完成列名
    string = nameParts.pop();  // 弹出最后一个元素，即列名
    var table = nameParts.join(".");  // 将剩余的元素连接成表格名

    var alias = false;  // 别名标识
    var aliasTable = table;  // 别名表格名
    // 检查表格是否可用，如果不可用，则通过别名找到表格
    if (!getTable(table)) {
      var oldTable = table;
      table = findTableByAlias(table, editor);  // 通过别名找到表格
      if (table !== oldTable) alias = true;  // 如果找到的表格和原表格不同，则设置别名标识为true
    }

    var columns = getTable(table);  // 获取表格的列
    if (columns && columns.columns)
      columns = columns.columns;  // 如果存在列，则获取列

    if (columns) {
      addMatches(result, string, columns, function(w) {
        var tableInsert = table;  // 插入的表格名
        if (alias == true) tableInsert = aliasTable;  // 如果有别名，则使用别名表格名
        if (typeof w == "string") {
          w = tableInsert + "." + w;  // 如果是字符串，则加上表格名
        } else {
          w = shallowClone(w);
          w.text = tableInsert + "." + w.text;  // 如果是对象，则修改对象的文本属性
        }
        return useIdentifierQuotes ? insertIdentifierQuotes(w) : w;  // 如果使用标识符引号，则插入引号
      });
    }

    return start;  // 返回起始位置
  }

  function eachWord(lineText, f) {
    // 将文本按空格分割成单词数组
    var words = lineText.split(/\s+/)
    // 遍历单词数组
    for (var i = 0; i < words.length; i++)
      // 如果单词不为空，则调用函数 f 处理去除逗号和分号后的单词
      if (words[i]) f(words[i].replace(/[,;]/g, ''))
  }

  // 根据别名在编辑器中查找表
  function findTableByAlias(alias, editor) {
    // 获取编辑器文档
    var doc = editor.doc;
    // 获取完整查询语句
    var fullQuery = doc.getValue();
    // 将别名转换为大写
    var aliasUpperCase = alias.toUpperCase();
    // 初始化前一个单词和表名
    var previousWord = "";
    var table = "";
    // 初始化分隔符数组
    var separator = [];
    // 初始化有效范围
    var validRange = {
      start: Pos(0, 0),
      end: Pos(editor.lastLine(), editor.getLineHandle(editor.lastLine()).length)
    };

    // 添加分隔符位置
    var indexOfSeparator = fullQuery.indexOf(CONS.QUERY_DIV);
    while(indexOfSeparator != -1) {
      separator.push(doc.posFromIndex(indexOfSeparator));
      indexOfSeparator = fullQuery.indexOf(CONS.QUERY_DIV, indexOfSeparator+1);
    }
    separator.unshift(Pos(0, 0));
    separator.push(Pos(editor.lastLine(), editor.getLineHandle(editor.lastLine()).text.length));

    // 查找有效范围
    var prevItem = null;
    var current = editor.getCursor()
    for (var i = 0; i < separator.length; i++) {
      if ((prevItem == null || cmpPos(current, prevItem) > 0) && cmpPos(current, separator[i]) <= 0) {
        validRange = {start: prevItem, end: separator[i]};
        break;
      }
      prevItem = separator[i];
    }

    if (validRange.start) {
      // 获取有效范围内的查询语句
      var query = doc.getRange(validRange.start, validRange.end, false);

      for (var i = 0; i < query.length; i++) {
        var lineText = query[i];
        // 遍历查询语句中的每个单词
        eachWord(lineText, function(word) {
          var wordUpperCase = word.toUpperCase();
          // 如果单词与别名相同且前一个单词是表名，则将表名赋值给 table
          if (wordUpperCase === aliasUpperCase && getTable(previousWord))
            table = previousWord;
          // 如果单词不是别名关键字，则更新前一个单词
          if (wordUpperCase !== CONS.ALIAS_KEYWORD)
            previousWord = word;
        });
        // 如果找到表名，则跳出循环
        if (table) break;
      }
    }
    // 返回表名
    return table;
  }

  // 注册 SQL 提示功能
  CodeMirror.registerHelper("hint", "sql", function(editor, options) {
    // 解析表名
    tables = parseTables(options && options.tables)
    // 获取默认表名
    var defaultTableName = options && options.defaultTable;
    # 检查是否存在禁用关键字选项，如果存在则赋值给 disableKeywords，否则为 undefined
    var disableKeywords = options && options.disableKeywords;
    # 如果存在默认表名，则获取对应的表格，赋值给 defaultTable，否则为 undefined
    defaultTable = defaultTableName && getTable(defaultTableName);
    # 获取编辑器中的关键字
    keywords = getKeywords(editor);
    # 获取编辑器中的标识符引用
    identifierQuote = getIdentifierQuote(editor);

    # 如果存在默认表名且默认表不存在，则通过别名查找表格
    if (defaultTableName && !defaultTable)
      defaultTable = findTableByAlias(defaultTableName, editor);

    # 如果默认表存在且包含列，则将列赋值给 defaultTable
    defaultTable = defaultTable || [];

    # 获取当前光标位置
    var cur = editor.getCursor();
    # 初始化结果数组
    var result = [];
    # 获取当前光标处的标记
    var token = editor.getTokenAt(cur), start, end, search;
    # 如果标记的结束位置大于光标位置，则修正结束位置和字符串
    if (token.end > cur.ch) {
      token.end = cur.ch;
      token.string = token.string.slice(0, cur.ch - token.start);
    }

    # 如果标记的字符串匹配指定模式，则赋值给搜索字符串、开始位置和结束位置
    if (token.string.match(/^[.`"'\w@][\w$#]*$/g)) {
      search = token.string;
      start = token.start;
      end = token.end;
    } else {
      start = end = cur.ch;
      search = "";
    }
    # 如果搜索字符串以"."或标识符引用字符开头，则调用 nameCompletion 函数
    if (search.charAt(0) == "." || search.charAt(0) == identifierQuote) {
      start = nameCompletion(cur, token, result, editor);
    } else {
      # 定义一个函数，用于设置对象或类名
      var objectOrClass = function(w, className) {
        if (typeof w === "object") {
          w.className = className;
        } else {
          w = { text: w, className: className };
        }
        return w;
      };
      # 将默认表格中匹配搜索字符串的结果添加到结果数组中
      addMatches(result, search, defaultTable, function(w) {
        return objectOrClass(w, "CodeMirror-hint-table CodeMirror-hint-default-table");
      });
      # 将所有表格中匹配搜索字符串的结果添加到结果数组中
      addMatches(
        result,
        search,
        tables, function(w) {
          return objectOrClass(w, "CodeMirror-hint-table");
        }
      );
      # 如果禁用关键字选项不存在，则将匹配搜索字符串的关键字结果添加到结果数组中
      if (!disableKeywords)
        addMatches(result, search, keywords, function(w) {
          return objectOrClass(w.toUpperCase(), "CodeMirror-hint-keyword");
        });
    }
    # 返回结果数组和光标位置范围
    return {list: result, from: Pos(cur.line, start), to: Pos(cur.line, end)};
  });
# 闭合了一个代码块或者函数的结束括号
```