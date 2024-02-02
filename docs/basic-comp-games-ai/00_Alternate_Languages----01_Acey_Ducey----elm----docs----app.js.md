# `basic-computer-games\00_Alternate_Languages\01_Acey_Ducey\elm\docs\app.js`

```py
// 在给定的参数、函数和包装器的基础上创建一个新的函数对象
function F(arity, fun, wrapper) {
  wrapper.a = arity; // 设置包装器的参数个数
  wrapper.f = fun; // 设置包装器的函数
  return wrapper; // 返回包装器
}

// 创建一个接受两个参数的函数对象
function F2(fun) {
  return F(2, fun, function(a) { return function(b) { return fun(a,b); }; }) // 返回一个包装器函数
}
// 创建一个接受三个参数的函数对象
function F3(fun) {
  return F(3, fun, function(a) {
    return function(b) { return function(c) { return fun(a, b, c); }; };
  });
}
// 创建一个接受四个参数的函数对象
function F4(fun) {
  return F(4, fun, function(a) { return function(b) { return function(c) {
    return function(d) { return fun(a, b, c, d); }; }; };
  });
}
// 创建一个接受五个参数的函数对象
function F5(fun) {
  return F(5, fun, function(a) { return function(b) { return function(c) {
    return function(d) { return function(e) { return fun(a, b, c, d, e); }; }; }; };
  });
}
// 创建一个接受六个参数的函数对象
function F6(fun) {
  return F(6, fun, function(a) { return function(b) { return function(c) {
    return function(d) { return function(e) { return function(f) {
    return fun(a, b, c, d, e, f); }; }; }; }; };
  });
}
// 创建一个接受七个参数的函数对象
function F7(fun) {
  return F(7, fun, function(a) { return function(b) { return function(c) {
    return function(d) { return function(e) { return function(f) {
    return function(g) { return fun(a, b, c, d, e, f, g); }; }; }; }; }; };
  });
}
// 创建一个接受八个参数的函数对象
function F8(fun) {
  return F(8, fun, function(a) { return function(b) { return function(c) {
    return function(d) { return function(e) { return function(f) {
    return function(g) { return function(h) {
    return fun(a, b, c, d, e, f, g, h); }; }; }; }; }; }; };
  });
}
// 创建一个接受九个参数的函数对象
function F9(fun) {
  return F(9, fun, function(a) { return function(b) { return function(c) {
    return function(d) { return function(e) { return function(f) {
    return function(g) { return function(h) { return function(i) {
    return fun(a, b, c, d, e, f, g, h, i); }; }; }; }; }; }; }; };
  });
}

// 调用接受两个参数的函数对象
function A2(fun, a, b) {
  return fun.a === 2 ? fun.f(a, b) : fun(a)(b);
}
// 调用接受三个参数的函数对象
function A3(fun, a, b, c) {
  return fun.a === 3 ? fun.f(a, b, c) : fun(a)(b)(c);
}
// 调用接受四个参数的函数对象
function A4(fun, a, b, c, d) {
  return fun.a === 4 ? fun.f(a, b, c, d) : fun(a)(b)(c)(d);
// 定义一个函数 A5，接受一个函数和五个参数，如果函数的属性 a 等于 5，则调用函数的 f 方法，否则依次调用函数传入参数的方法
function A5(fun, a, b, c, d, e) {
  return fun.a === 5 ? fun.f(a, b, c, d, e) : fun(a)(b)(c)(d)(e);
}

// 定义一个函数 A6，接受一个函数和六个参数，如果函数的属性 a 等于 6，则调用函数的 f 方法，否则依次调用函数传入参数的方法
function A6(fun, a, b, c, d, e, f) {
  return fun.a === 6 ? fun.f(a, b, c, d, e, f) : fun(a)(b)(c)(d)(e)(f);
}

// 定义一个函数 A7，接受一个函数和七个参数，如果函数的属性 a 等于 7，则调用函数的 f 方法，否则依次调用函数传入参数的方法
function A7(fun, a, b, c, d, e, f, g) {
  return fun.a === 7 ? fun.f(a, b, c, d, e, f, g) : fun(a)(b)(c)(d)(e)(f)(g);
}

// 定义一个函数 A8，接受一个函数和八个参数，如果函数的属性 a 等于 8，则调用函数的 f 方法，否则依次调用函数传入参数的方法
function A8(fun, a, b, c, d, e, f, g, h) {
  return fun.a === 8 ? fun.f(a, b, c, d, e, f, g, h) : fun(a)(b)(c)(d)(e)(f)(g)(h);
}

// 定义一个函数 A9，接受一个函数和九个参数，如果函数的属性 a 等于 9，则调用函数的 f 方法，否则依次调用函数传入参数的方法
function A9(fun, a, b, c, d, e, f, g, h, i) {
  return fun.a === 9 ? fun.f(a, b, c, d, e, f, g, h, i) : fun(a)(b)(c)(d)(e)(f)(g)(h)(i);
}

// 定义一个函数 _Utils_eq，用于比较两个值是否相等
function _Utils_eq(x, y)
{
    // 使用栈来存储比较过程中的值，初始值为两个参数的比较结果
    for (
        var pair, stack = [], isEqual = _Utils_eqHelp(x, y, 0, stack);
        isEqual && (pair = stack.pop());
        isEqual = _Utils_eqHelp(pair.a, pair.b, 0, stack)
        )
    {}

    return isEqual;
}

// 辅助函数 _Utils_eqHelp，用于递归比较两个值是否相等
function _Utils_eqHelp(x, y, depth, stack)
{
    // 如果两个值相等，则返回 true
    if (x === y)
    {
        return true;
    }

    // 如果其中一个值不是对象或者为 null，则返回 false
    if (typeof x !== 'object' || x === null || y === null)
    {
        typeof x === 'function' && _Debug_crash(5);
        return false;
    }

    // 如果递归深度超过 100，则返回 true
    if (depth > 100)
    {
        stack.push(_Utils_Tuple2(x,y));
        return true;
    }

    // 如果值为 Set_elm_builtin 或 RBNode_elm_builtin 或 RBEmpty_elm_builtin，则转换为列表
    if (x.$ < 0)
    {
        x = $elm$core$Dict$toList(x);
        y = $elm$core$Dict$toList(y);
    }

    // 递归比较对象的每个属性值是否相等
    for (var key in x)
    {
        if (!_Utils_eqHelp(x[key], y[key], depth + 1, stack))
        {
            return false;
        }
    }
    return true;
}

// 定义一个函数 _Utils_equal，用于比较两个值是否相等
var _Utils_equal = F2(_Utils_eq);

// 定义一个函数 _Utils_notEqual，用于比较两个值是否不相等
var _Utils_notEqual = F2(function(a, b) { return !_Utils_eq(a,b); });
// 定义一个函数，用于比较两个值的大小，返回特定的整数值LT、EQ和GT
function _Utils_cmp(x, y, ord)
{
    // 如果x不是对象类型
    if (typeof x !== 'object')
    {
        // 如果x等于y，则返回0，否则如果x小于y，则返回-1，否则返回1
        return x === y ? /*EQ*/ 0 : x < y ? /*LT*/ -1 : /*GT*/ 1;
    }

    /**_UNUSED/
    // 如果x是String类型的实例
    if (x instanceof String)
    {
        // 获取x和y的值
        var a = x.valueOf();
        var b = y.valueOf();
        // 如果a等于b，则返回0，否则如果a小于b，则返回-1，否则返回1
        return a === b ? 0 : a < b ? -1 : 1;
    }
    //*/

    /**/
    // 如果x的$属性未定义
    if (typeof x.$ === 'undefined')
    //*/
    /**_UNUSED/
    // 如果x的$属性的第一个元素是'#'
    if (x.$[0] === '#')
    //*/
    {
        // 递归比较x.a和y.a，如果不相等则返回，否则继续比较x.b和y.b，如果不相等则返回，否则继续比较x.c和y.c
        return (ord = _Utils_cmp(x.a, y.a))
            ? ord
            : (ord = _Utils_cmp(x.b, y.b))
                ? ord
                : _Utils_cmp(x.c, y.c);
    }

    // 遍历conses直到列表结束或不匹配
    for (; x.b && y.b && !(ord = _Utils_cmp(x.a, y.a)); x = x.b, y = y.b) {} // WHILE_CONSES
    // 如果ord为真，则返回ord，否则如果x.b为真，则返回1，否则如果y.b为真，则返回-1，否则返回0
    return ord || (x.b ? /*GT*/ 1 : y.b ? /*LT*/ -1 : /*EQ*/ 0);
}

// 定义一个函数，用于比较两个值的大小，返回布尔值
var _Utils_lt = F2(function(a, b) { return _Utils_cmp(a, b) < 0; });
var _Utils_le = F2(function(a, b) { return _Utils_cmp(a, b) < 1; });
var _Utils_gt = F2(function(a, b) { return _Utils_cmp(a, b) > 0; });
var _Utils_ge = F2(function(a, b) { return _Utils_cmp(a, b) >= 0; });

var _Utils_compare = F2(function(x, y)
{
    // 获取两个值的比较结果
    var n = _Utils_cmp(x, y);
    // 如果n小于0，则返回LT，否则如果n不为0，则返回GT，否则返回EQ
    return n < 0 ? $elm$core$Basics$LT : n ? $elm$core$Basics$GT : $elm$core$Basics$EQ;
});


// 常用值

// 定义一个值为0的元组
var _Utils_Tuple0 = 0;
// 定义一个未使用的值为{ $: '#0' }的元组
var _Utils_Tuple0_UNUSED = { $: '#0' };

// 定义一个包含两个值的元组的函数
function _Utils_Tuple2(a, b) { return { a: a, b: b }; }
// 定义一个未使用的包含两个值的元组的函数
function _Utils_Tuple2_UNUSED(a, b) { return { $: '#2', a: a, b: b }; }

// 定义一个包含三个值的元组的函数
function _Utils_Tuple3(a, b, c) { return { a: a, b: b, c: c }; }
// 定义一个未使用的包含三个值的元组的函数
function _Utils_Tuple3_UNUSED(a, b, c) { return { $: '#3', a: a, b: b, c: c }; }

// 定义一个返回字符的函数
function _Utils_chr(c) { return c; }
// 定义一个未使用的返回字符的函数
function _Utils_chr_UNUSED(c) { return new String(c); }


// 记录

// 更新记录的函数
function _Utils_update(oldRecord, updatedFields)
{
    // 创建一个新的记录
    var newRecord = {};

    // 遍历旧记录的每个属性，将其赋值给新记录
    for (var key in oldRecord)
    {
        newRecord[key] = oldRecord[key];
    }

    // 遍历更新的字段，将其赋值给新记录
    for (var key in updatedFields)
    # 将更新后的字段值更新到新记录中的对应字段
    newRecord[key] = updatedFields[key];
    # 返回更新后的新记录
    return newRecord;
// APPEND

// 定义一个函数，用于将两个参数连接起来
var _Utils_append = F2(_Utils_ap);

// 定义一个函数，用于连接字符串
function _Utils_ap(xs, ys)
{
    // 如果第一个参数是字符串，则直接将两个字符串连接起来
    if (typeof xs === 'string')
    {
        return xs + ys;
    }

    // 如果第一个参数是空列表，则直接返回第二个参数
    if (!xs.b)
    {
        return ys;
    }
    // 否则，将第一个参数的头部和第二个参数连接起来
    var root = _List_Cons(xs.a, ys);
    xs = xs.b
    // 遍历第一个参数，将其余部分和第二个参数连接起来
    for (var curr = root; xs.b; xs = xs.b) // WHILE_CONS
    {
        curr = curr.b = _List_Cons(xs.a, ys);
    }
    return root;
}

// 定义空列表
var _List_Nil = { $: 0 };
var _List_Nil_UNUSED = { $: '[]' };

// 定义一个函数，用于创建非空列表
function _List_Cons(hd, tl) { return { $: 1, a: hd, b: tl }; }
function _List_Cons_UNUSED(hd, tl) { return { $: '::', a: hd, b: tl }; }

// 定义一个函数，用于将数组转换为列表
var _List_cons = F2(_List_Cons);

function _List_fromArray(arr)
{
    var out = _List_Nil;
    // 遍历数组，将数组中的元素逐个添加到列表中
    for (var i = arr.length; i--; )
    {
        out = _List_Cons(arr[i], out);
    }
    return out;
}

// 定义一个函数，用于将列表转换为数组
function _List_toArray(xs)
{
    // 遍历列表，将列表中的元素逐个添加到数组中
    for (var out = []; xs.b; xs = xs.b) // WHILE_CONS
    {
        out.push(xs.a);
    }
    return out;
}

// 定义一个函数，用于对两个列表进行映射
var _List_map2 = F3(function(f, xs, ys)
{
    // 遍历两个列表，将对应位置的元素传入函数进行映射
    for (var arr = []; xs.b && ys.b; xs = xs.b, ys = ys.b) // WHILE_CONSES
    {
        arr.push(A2(f, xs.a, ys.a));
    }
    return _List_fromArray(arr);
});

// 定义一个函数，用于对三个列表进行映射
var _List_map3 = F4(function(f, xs, ys, zs)
{
    // 遍历三个列表，将对应位置的元素传入函数进行映射
    for (var arr = []; xs.b && ys.b && zs.b; xs = xs.b, ys = ys.b, zs = zs.b) // WHILE_CONSES
    {
        arr.push(A3(f, xs.a, ys.a, zs.a));
    }
    return _List_fromArray(arr);
});

// 定义一个函数，用于对四个列表进行映射
var _List_map4 = F5(function(f, ws, xs, ys, zs)
{
    // 遍历四个列表，将对应位置的元素传入函数进行映射
    for (var arr = []; ws.b && xs.b && ys.b && zs.b; ws = ws.b, xs = xs.b, ys = ys.b, zs = zs.b) // WHILE_CONSES
    {
        arr.push(A4(f, ws.a, xs.a, ys.a, zs.a));
    }
    return _List_fromArray(arr);
});

// 定义一个函数，用于对五个列表进行映射
var _List_map5 = F6(function(f, vs, ws, xs, ys, zs)
{
    // 遍历五个列表，将对应位置的元素传入函数进行映射
    for (var arr = []; vs.b && ws.b && xs.b && ys.b && zs.b; vs = vs.b, ws = ws.b, xs = xs.b, ys = ys.b, zs = zs.b) // WHILE_CONSES
    {
        arr.push(A5(f, vs.a, ws.a, xs.a, ys.a, zs.a));
    }
    return _List_fromArray(arr);
});

// 定义一个函数，用于对列表进行排序
var _List_sortBy = F2(function(f, xs)
{
    # 将数组转换为列表，对列表进行排序，并返回排序后的列表
    return _List_fromArray(
        # 将列表转换为数组，然后调用sort方法进行排序，排序规则是根据f(a)和f(b)的比较结果
        _List_toArray(xs).sort(function(a, b) {
            return _Utils_cmp(f(a), f(b));
        })
    );
});
// 定义一个函数，用于对列表进行排序
var _List_sortWith = F2(function(f, xs)
{
    // 将列表转换为数组，然后使用给定的比较函数进行排序，最后再转换回列表
    return _List_fromArray(_List_toArray(xs).sort(function(a, b) {
        // 使用给定的比较函数对元素进行比较，并返回排序后的结果
        var ord = A2(f, a, b);
        return ord === $elm$core$Basics$EQ ? 0 : ord === $elm$core$Basics$LT ? -1 : 1;
    }));
});

// 定义一个空的 JavaScript 数组
var _JsArray_empty = [];

// 定义一个函数，用于创建只包含一个元素的 JavaScript 数组
function _JsArray_singleton(value)
{
    return [value];
}

// 定义一个函数，用于获取 JavaScript 数组的长度
function _JsArray_length(array)
{
    return array.length;
}

// 定义一个函数，用于根据给定的大小、偏移量和函数来初始化 JavaScript 数组
var _JsArray_initialize = F3(function(size, offset, func)
{
    var result = new Array(size);

    for (var i = 0; i < size; i++)
    {
        result[i] = func(offset + i);
    }

    return result;
});

// 定义一个函数，用于根据给定的最大值和列表来初始化 JavaScript 数组
var _JsArray_initializeFromList = F2(function (max, ls)
{
    var result = new Array(max);

    for (var i = 0; i < max && ls.b; i++)
    {
        result[i] = ls.a;
        ls = ls.b;
    }

    result.length = i;
    return _Utils_Tuple2(result, ls);
});

// 定义一个函数，用于获取 JavaScript 数组中指定索引的元素
var _JsArray_unsafeGet = F2(function(index, array)
{
    return array[index];
});

// 定义一个函数，用于设置 JavaScript 数组中指定索引的元素值
var _JsArray_unsafeSet = F3(function(index, value, array)
{
    var length = array.length;
    var result = new Array(length);

    for (var i = 0; i < length; i++)
    {
        result[i] = array[i];
    }

    result[index] = value;
    return result;
});

// 定义一个函数，用于向 JavaScript 数组中添加一个元素
var _JsArray_push = F2(function(value, array)
{
    var length = array.length;
    var result = new Array(length + 1);

    for (var i = 0; i < length; i++)
    {
        result[i] = array[i];
    }

    result[length] = value;
    return result;
});

// 定义一个函数，用于对 JavaScript 数组进行左折叠操作
var _JsArray_foldl = F3(function(func, acc, array)
{
    var length = array.length;

    for (var i = 0; i < length; i++)
    {
        acc = A2(func, array[i], acc);
    }

    return acc;
});

// 定义一个函数，用于对 JavaScript 数组进行右折叠操作
var _JsArray_foldr = F3(function(func, acc, array)
{
    for (var i = array.length - 1; i >= 0; i--)
    {
        acc = A2(func, array[i], acc);
    }

    return acc;
});

// 定义一个函数，用于对 JavaScript 数组中的每个元素进行映射操作
var _JsArray_map = F2(function(func, array)
{
    var length = array.length;
    var result = new Array(length);

    for (var i = 0; i < length; i++)
    # 遍历数组中的每个元素，将经过函数处理后的结果存入结果数组中
    result[i] = func(array[i]);
    # 返回处理后的结果数组
    return result;
// 定义一个函数，将给定的函数应用到数组的每个元素上，并返回结果数组
var _JsArray_indexedMap = F3(function(func, offset, array)
{
    // 获取数组的长度
    var length = array.length;
    // 创建一个与原数组长度相同的空数组
    var result = new Array(length);

    // 遍历数组，对每个元素应用给定的函数，并将结果存入新数组
    for (var i = 0; i < length; i++)
    {
        result[i] = A2(func, offset + i, array[i]);
    }

    // 返回结果数组
    return result;
});

// 定义一个函数，返回数组的子数组
var _JsArray_slice = F3(function(from, to, array)
{
    return array.slice(from, to);
});

// 定义一个函数，将源数组的元素追加到目标数组中
var _JsArray_appendN = F3(function(n, dest, source)
{
    // 获取目标数组的长度
    var destLen = dest.length;
    // 计算需要复制的元素个数
    var itemsToCopy = n - destLen;

    // 如果需要复制的元素个数大于源数组的长度，则将需要复制的元素个数设为源数组的长度
    if (itemsToCopy > source.length)
    {
        itemsToCopy = source.length;
    }

    // 计算结果数组的长度
    var size = destLen + itemsToCopy;
    // 创建一个与结果数组长度相同的空数组
    var result = new Array(size);

    // 将目标数组的元素复制到结果数组中
    for (var i = 0; i < destLen; i++)
    {
        result[i] = dest[i];
    }

    // 将源数组的元素复制到结果数组中
    for (var i = 0; i < itemsToCopy; i++)
    {
        result[i + destLen] = source[i];
    }

    // 返回结果数组
    return result;
});

// 定义一个函数，用于记录日志
var _Debug_log = F2(function(tag, value)
{
    return value;
});

// 定义一个函数，用于记录未使用的日志
var _Debug_log_UNUSED = F2(function(tag, value)
{
    console.log(tag + ': ' + _Debug_toString(value));
    return value;
});

// 定义一个函数，用于抛出未实现的错误
function _Debug_todo(moduleName, region)
{
    return function(message) {
        _Debug_crash(8, moduleName, region, message);
    };
}

// 定义一个函数，用于抛出未实现的 case 错误
function _Debug_todoCase(moduleName, region, value)
{
    return function(message) {
        _Debug_crash(9, moduleName, region, value, message);
    };
}

// 定义一个函数，将值转换为字符串
function _Debug_toString(value)
{
    return '<internals>';
}

// 定义一个函数，将未使用的值转换为字符串
function _Debug_toString_UNUSED(value)
{
    return _Debug_toAnsiString(false, value);
}

// 定义一个函数，将值转换为 ANSI 字符串
function _Debug_toAnsiString(ansi, value)
{
    // 根据值的类型返回相应的颜色字符串
    if (typeof value === 'function')
    {
        return _Debug_internalColor(ansi, '<function>');
    }

    if (typeof value === 'boolean')
    {
        return _Debug_ctorColor(ansi, value ? 'True' : 'False');
    }

    if (typeof value === 'number')
    {
        return _Debug_numberColor(ansi, value + '');
    }

    if (value instanceof String)
    {
        return _Debug_charColor(ansi, "'" + _Debug_addSlashes(value, true) + "'");
    }
    // ...
    // 其他类型的值的处理
    // ...
});
    }
    # 如果值的类型是字符串，返回带有 ANSI 转义码的字符串
    if (typeof value === 'string')
    {
        return _Debug_stringColor(ansi, '"' + _Debug_addSlashes(value, false) + '"');
    }
    # 如果值的类型是对象并且包含 '$' 属性
    if (typeof value === 'object' && '$' in value)
    }
    # 如果 DataView 存在并且值是DataView类型
    if (typeof DataView === 'function' && value instanceof DataView)
    {
        return _Debug_stringColor(ansi, '<' + value.byteLength + ' bytes>');
    }
    # 如果 File 存在并且值是File类型
    if (typeof File !== 'undefined' && value instanceof File)
    {
        return _Debug_internalColor(ansi, '<' + value.name + '>');
    }
    # 如果值的类型是对象
    if (typeof value === 'object')
    {
        var output = [];
        # 遍历对象的属性，组成字符串数组
        for (var key in value)
        {
            var field = key[0] === '_' ? key.slice(1) : key;
            output.push(_Debug_fadeColor(ansi, field) + ' = ' + _Debug_toAnsiString(ansi, value[key]));
        }
        # 如果数组长度为0，返回空对象
        if (output.length === 0)
        {
            return '{}';
        }
        # 返回对象的字符串表示形式
        return '{ ' + output.join(', ') + ' }';
    }
    # 返回内部值的 ANSI 转义码
    return _Debug_internalColor(ansi, '<internals>');
// 添加转义字符到字符串中，用于调试目的
function _Debug_addSlashes(str, isChar)
{
    var s = str
        .replace(/\\/g, '\\\\')  // 替换反斜杠为两个反斜杠
        .replace(/\n/g, '\\n')   // 替换换行符为\n
        .replace(/\t/g, '\\t')   // 替换制表符为\t
        .replace(/\r/g, '\\r')   // 替换回车符为\r
        .replace(/\v/g, '\\v')   // 替换垂直制表符为\v
        .replace(/\0/g, '\\0');   // 替换空字符为\0

    if (isChar)
    {
        return s.replace(/\'/g, '\\\'');  // 如果是字符，替换单引号为\'
    }
    else
    {
        return s.replace(/\"/g, '\\"');  // 如果不是字符，替换双引号为\"
    }
}

// 为字符串添加 ANSI 颜色，用于调试目的
function _Debug_ctorColor(ansi, string)
{
    return ansi ? '\x1b[96m' + string + '\x1b[0m' : string;  // 如果启用 ANSI，则添加亮青色
}

function _Debug_numberColor(ansi, string)
{
    return ansi ? '\x1b[95m' + string + '\x1b[0m' : string;  // 如果启用 ANSI，则添加紫色
}

function _Debug_stringColor(ansi, string)
{
    return ansi ? '\x1b[93m' + string + '\x1b[0m' : string;  // 如果启用 ANSI，则添加黄色
}

function _Debug_charColor(ansi, string)
{
    return ansi ? '\x1b[92m' + string + '\x1b[0m' : string;  // 如果启用 ANSI，则添加绿色
}

function _Debug_fadeColor(ansi, string)
{
    return ansi ? '\x1b[37m' + string + '\x1b[0m' : string;  // 如果启用 ANSI，则添加淡灰色
}

function _Debug_internalColor(ansi, string)
{
    return ansi ? '\x1b[36m' + string + '\x1b[0m' : string;  // 如果启用 ANSI，则添加青色
}

function _Debug_toHexDigit(n)
{
    return String.fromCharCode(n < 10 ? 48 + n : 55 + n);  // 返回十六进制数字的字符表示
}

// 抛出一个包含链接的错误，用于调试目的
function _Debug_crash(identifier)
{
    throw new Error('https://github.com/elm/core/blob/1.0.0/hints/' + identifier + '.md');
}

// 未使用的调试功能，用于调试目的
function _Debug_crash_UNUSED(identifier, fact1, fact2, fact3, fact4)
{
    switch(identifier)  // 未使用的调试功能的标识符
    }
}

// 将区域对象转换为字符串，用于调试目的
function _Debug_regionToString(region)
{
    if (region.Q.H === region.V.H)
    {
        return 'on line ' + region.Q.H;  // 如果起始行等于结束行，返回单行信息
    }
    return 'on lines ' + region.Q.H + ' through ' + region.V.H;  // 否则返回多行信息
}

// 数学相关函数

var _Basics_add = F2(function(a, b) { return a + b; });  // 加法
var _Basics_sub = F2(function(a, b) { return a - b; });  // 减法
var _Basics_mul = F2(function(a, b) { return a * b; });  // 乘法
var _Basics_fdiv = F2(function(a, b) { return a / b; });  // 浮点数除法
var _Basics_idiv = F2(function(a, b) { return (a / b) | 0; });  // 整数除法
var _Basics_pow = F2(Math.pow);  // 幂运算

var _Basics_remainderBy = F2(function(b, a) { return a % b; });  // 取余数
// 定义一个函数，用于计算 x 对 modulus 取模的结果
var _Basics_modBy = F2(function(modulus, x)
{
    // 计算 x 对 modulus 取模的结果
    var answer = x % modulus;
    // 如果 modulus 为 0，则触发调试错误
    return modulus === 0
        ? _Debug_crash(11)
        :
    // 如果 answer 为正且 modulus 为负，或者 answer 为负且 modulus 为正，则返回 answer + modulus，否则返回 answer
    ((answer > 0 && modulus < 0) || (answer < 0 && modulus > 0))
        ? answer + modulus
        : answer;
});


// 三角函数

// 定义常量 _Basics_pi 为圆周率
var _Basics_pi = Math.PI;
// 定义常量 _Basics_e 为自然对数的底
var _Basics_e = Math.E;
// 定义函数 _Basics_cos 为余弦函数
var _Basics_cos = Math.cos;
// 定义函数 _Basics_sin 为正弦函数
var _Basics_sin = Math.sin;
// 定义函数 _Basics_tan 为正切函数
var _Basics_tan = Math.tan;
// 定义函数 _Basics_acos 为反余弦函数
var _Basics_acos = Math.acos;
// 定义函数 _Basics_asin 为反正弦函数
var _Basics_asin = Math.asin;
// 定义函数 _Basics_atan 为反正切函数
var _Basics_atan = Math.atan;
// 定义函数 _Basics_atan2 为反正切函数的双参数版本
var _Basics_atan2 = F2(Math.atan2);


// 更多数学函数

// 定义函数 _Basics_toFloat 用于将参数转换为浮点数
function _Basics_toFloat(x) { return x; }
// 定义函数 _Basics_truncate 用于将参数取整
function _Basics_truncate(n) { return n | 0; }
// 定义函数 _Basics_isInfinite 用于判断参数是否为无穷大
function _Basics_isInfinite(n) { return n === Infinity || n === -Infinity; }

// 定义函数 _Basics_ceiling 为向上取整函数
var _Basics_ceiling = Math.ceil;
// 定义函数 _Basics_floor 为向下取整函数
var _Basics_floor = Math.floor;
// 定义函数 _Basics_round 为四舍五入函数
var _Basics_round = Math.round;
// 定义函数 _Basics_sqrt 为平方根函数
var _Basics_sqrt = Math.sqrt;
// 定义函数 _Basics_log 为自然对数函数
var _Basics_log = Math.log;
// 定义函数 _Basics_isNaN 用于判断参数是否为 NaN

// 布尔值

// 定义函数 _Basics_not 用于取反
function _Basics_not(bool) { return !bool; }
// 定义函数 _Basics_and 为逻辑与函数
var _Basics_and = F2(function(a, b) { return a && b; });
// 定义函数 _Basics_or 为逻辑或函数
var _Basics_or  = F2(function(a, b) { return a || b; });
// 定义函数 _Basics_xor 为逻辑异或函数
var _Basics_xor = F2(function(a, b) { return a !== b; });

// 字符串操作

// 定义函数 _String_cons 用于在字符串前添加一个字符
var _String_cons = F2(function(chr, str)
{
    return chr + str;
});

// 定义函数 _String_uncons 用于从字符串中取出第一个字符
function _String_uncons(string)
{
    var word = string.charCodeAt(0);
    return !isNaN(word)
        ? $elm$core$Maybe$Just(
            0xD800 <= word && word <= 0xDBFF
                ? _Utils_Tuple2(_Utils_chr(string[0] + string[1]), string.slice(2))
                : _Utils_Tuple2(_Utils_chr(string[0]), string.slice(1))
        )
        : $elm$core$Maybe$Nothing;
}

// 定义函数 _String_append 用于连接两个字符串
var _String_append = F2(function(a, b)
{
    return a + b;
});

// 定义函数 _String_length 用于获取字符串的长度
function _String_length(str)
{
    return str.length;
}

// 定义函数 _String_map 用于对字符串中的每个字符应用函数
var _String_map = F2(function(func, string)
{
    var len = string.length;
    var array = new Array(len);
    var i = 0;
    while (i < len)
    {
        // 获取字符串中指定位置的字符的 Unicode 编码
        var word = string.charCodeAt(i);
        // 如果是代理对的高位码点
        if (0xD800 <= word && word <= 0xDBFF)
        {
            // 调用 func 函数处理代理对，并将结果存入数组
            array[i] = func(_Utils_chr(string[i] + string[i+1]));
            // 移动指针到下一个字符
            i += 2;
            // 继续循环
            continue;
        }
        // 如果不是代理对的高位码点，直接调用 func 函数处理字符，并将结果存入数组
        array[i] = func(_Utils_chr(string[i]));
        // 移动指针到下一个字符
        i++;
    }
    // 将数组中的字符连接成字符串并返回
    return array.join('');
// 定义一个函数，接受一个布尔函数和一个字符串作为参数，返回过滤后的字符串
var _String_filter = F2(function(isGood, str)
{
    // 创建一个空数组
    var arr = [];
    // 获取字符串的长度
    var len = str.length;
    // 初始化循环变量 i
    var i = 0;
    // 循环遍历字符串
    while (i < len)
    {
        // 获取当前字符
        var char = str[i];
        // 获取当前字符的 Unicode 编码
        var word = str.charCodeAt(i);
        // 递增 i
        i++;
        // 如果当前字符是高代理项
        if (0xD800 <= word && word <= 0xDBFF)
        {
            // 将下一个字符拼接到当前字符上
            char += str[i];
            // 递增 i
            i++;
        }

        // 如果当前字符通过 isGood 函数的检查
        if (isGood(_Utils_chr(char)))
        {
            // 将当前字符加入数组
            arr.push(char);
        }
    }
    // 将数组中的字符拼接成字符串并返回
    return arr.join('');
});

// 定义一个函数，接受一个字符串作为参数，返回反转后的字符串
function _String_reverse(str)
{
    // 获取字符串的长度
    var len = str.length;
    // 创建一个与字符串长度相等的数组
    var arr = new Array(len);
    // 初始化循环变量 i
    var i = 0;
    // 循环遍历字符串
    while (i < len)
    {
        // 获取当前字符的 Unicode 编码
        var word = str.charCodeAt(i);
        // 如果当前字符是高代理项
        if (0xD800 <= word && word <= 0xDBFF)
        {
            // 将下一个字符放在当前字符的位置
            arr[len - i] = str[i + 1];
            // 递增 i
            i++;
            // 将当前字符放在下一个位置
            arr[len - i] = str[i - 1];
            // 递增 i
            i++;
        }
        else
        {
            // 将当前字符放在对称位置
            arr[len - i] = str[i];
            // 递增 i
            i++;
        }
    }
    // 将数组中的字符拼接成字符串并返回
    return arr.join('');
}

// 定义一个函数，接受一个函数、一个初始状态和一个字符串作为参数，返回折叠后的状态
var _String_foldl = F3(function(func, state, string)
{
    // 获取字符串的长度
    var len = string.length;
    // 初始化循环变量 i
    var i = 0;
    // 循环遍历字符串
    while (i < len)
    {
        // 获取当前字符
        var char = string[i];
        // 获取当前字符的 Unicode 编码
        var word = string.charCodeAt(i);
        // 递增 i
        i++;
        // 如果当前字符是高代理项
        if (0xD800 <= word && word <= 0xDBFF)
        {
            // 将下一个字符拼接到当前字符上
            char += string[i];
            // 递增 i
            i++;
        }
        // 使用给定的函数和字符状态进行折叠
        state = A2(func, _Utils_chr(char), state);
    }
    // 返回折叠后的状态
    return state;
});

// 定义一个函数，接受一个函数、一个初始状态和一个字符串作为参数，返回反向折叠后的状态
var _String_foldr = F3(function(func, state, string)
{
    // 初始化循环变量 i
    var i = string.length;
    // 循环遍历字符串
    while (i--)
    {
        // 获取当前字符
        var char = string[i];
        // 获取当前字符的 Unicode 编码
        var word = string.charCodeAt(i);
        // 如果当前字符是低代理项
        if (0xDC00 <= word && word <= 0xDFFF)
        {
            // 递减 i
            i--;
            // 将前一个字符拼接到当前字符上
            char = string[i] + char;
        }
        // 使用给定的函数和字符状态进行折叠
        state = A2(func, _Utils_chr(char), state);
    }
    // 返回折叠后的状态
    return state;
});

// 定义一个函数，接受一个分隔符和一个字符串作为参数，返回分割后的数组
var _String_split = F2(function(sep, str)
{
    // 使用给定的分隔符对字符串进行分割并返回结果
    return str.split(sep);
});

// 定义一个函数，接受一个分隔符和一个字符串数组作为参数，返回连接后的字符串
var _String_join = F2(function(sep, strs)
{
    // 使用给定的分隔符对字符串数组进行连接并返回结果
    return strs.join(sep);
});

// 定义一个函数，接受一个起始位置、一个结束位置和一个字符串作为参数，返回切片后的字符串
var _String_slice = F3(function(start, end, str) {
    // 对字符串进行切片并返回结果
    return str.slice(start, end);
});

// 定义一个函数，接受一个字符串作为参数，返回去除两端空白后的字符串
function _String_trim(str)
{
    // 返回去除两端空白后的字符串
    return str.trim();
}
# 去除字符串左侧的空白字符
function _String_trimLeft(str)
{
    return str.replace(/^\s+/, '');
}

# 去除字符串右侧的空白字符
function _String_trimRight(str)
{
    return str.replace(/\s+$/, '');
}

# 将字符串转换为单词列表
function _String_words(str)
{
    return _List_fromArray(str.trim().split(/\s+/g));
}

# 将字符串按行分割成列表
function _String_lines(str)
{
    return _List_fromArray(str.split(/\r\n|\r|\n/g));
}

# 将字符串转换为大写
function _String_toUpper(str)
{
    return str.toUpperCase();
}

# 将字符串转换为小写
function _String_toLower(str)
{
    return str.toLowerCase();
}

# 判断字符串中是否存在满足条件的字符
var _String_any = F2(function(isGood, string)
{
    var i = string.length;
    while (i--)
    {
        var char = string[i];
        var word = string.charCodeAt(i);
        if (0xDC00 <= word && word <= 0xDFFF)
        {
            i--;
            char = string[i] + char;
        }
        if (isGood(_Utils_chr(char)))
        {
            return true;
        }
    }
    return false;
});

# 判断字符串中的所有字符是否都满足条件
var _String_all = F2(function(isGood, string)
{
    var i = string.length;
    while (i--)
    {
        var char = string[i];
        var word = string.charCodeAt(i);
        if (0xDC00 <= word && word <= 0xDFFF)
        {
            i--;
            char = string[i] + char;
        }
        if (!isGood(_Utils_chr(char)))
        {
            return false;
        }
    }
    return true;
});

# 判断字符串是否包含指定子字符串
var _String_contains = F2(function(sub, str)
{
    return str.indexOf(sub) > -1;
});

# 判断字符串是否以指定子字符串开头
var _String_startsWith = F2(function(sub, str)
{
    return str.indexOf(sub) === 0;
});

# 判断字符串是否以指定子字符串结尾
var _String_endsWith = F2(function(sub, str)
{
    return str.length >= sub.length &&
        str.lastIndexOf(sub) === str.length - sub.length;
});

# 获取字符串中指定子字符串的所有索引
var _String_indexes = F2(function(sub, str)
{
    var subLen = sub.length;

    if (subLen < 1)
    {
        return _List_Nil;
    }

    var i = 0;
    var is = [];

    while ((i = str.indexOf(sub, i)) > -1)
    {
        is.push(i);
        i = i + subLen;
    }

    return _List_fromArray(is);
});

# 将数字转换为字符串
function _String_fromNumber(number)
{
    return number + '';
}
# 将字符串转换为整数
function _String_toInt(str)
{
    # 初始化总和为0
    var total = 0;
    # 获取字符串的第一个字符的 Unicode 编码
    var code0 = str.charCodeAt(0);
    # 如果第一个字符是加号或减号，则从第二个字符开始计算
    var start = code0 == 0x2B /* + */ || code0 == 0x2D /* - */ ? 1 : 0;

    # 遍历字符串的每个字符
    for (var i = start; i < str.length; ++i)
    {
        # 获取当前字符的 Unicode 编码
        var code = str.charCodeAt(i);
        # 如果当前字符不是数字，则返回空值
        if (code < 0x30 || 0x39 < code)
        {
            return $elm$core$Maybe$Nothing;
        }
        # 将字符转换为数字并累加到总和中
        total = 10 * total + code - 0x30;
    }

    # 如果字符串只包含符号，则返回空值；否则返回包含符号的数字
    return i == start
        ? $elm$core$Maybe$Nothing
        : $elm$core$Maybe$Just(code0 == 0x2D ? -total : total);
}


# 将字符串转换为浮点数
function _String_toFloat(s)
{
    # 检查字符串是否是十六进制、八进制或二进制数
    if (s.length === 0 || /[\sxbo]/.test(s))
    {
        return $elm$core$Maybe$Nothing;
    }
    # 将字符串转换为数字
    var n = +s;
    # 更快的 isNaN 检查
    return n === n ? $elm$core$Maybe$Just(n) : $elm$core$Maybe$Nothing;
}

# 将字符列表转换为字符串
function _String_fromList(chars)
{
    return _List_toArray(chars).join('');
}

# 将字符转换为 Unicode 编码
function _Char_toCode(char)
{
    var code = char.charCodeAt(0);
    # 如果是代理对，则计算 Unicode 编码
    if (0xD800 <= code && code <= 0xDBFF)
    {
        return (code - 0xD800) * 0x400 + char.charCodeAt(1) - 0xDC00 + 0x10000
    }
    return code;
}

# 将 Unicode 编码转换为字符
function _Char_fromCode(code)
{
    return _Utils_chr(
        # 如果编码不在有效范围内，则返回替代字符
        (code < 0 || 0x10FFFF < code)
            ? '\uFFFD'
            :
        # 如果编码在基本多文种平面内，则直接转换为字符
        (code <= 0xFFFF)
            ? String.fromCharCode(code)
            :
        # 如果编码在辅助平面内，则计算代理对并转换为字符
        (code -= 0x10000,
            String.fromCharCode(Math.floor(code / 0x400) + 0xD800, code % 0x400 + 0xDC00)
        )
    );
}

# 将字符转换为大写
function _Char_toUpper(char)
{
    return _Utils_chr(char.toUpperCase());
}

# 将字符转换为小写
function _Char_toLower(char)
{
    return _Utils_chr(char.toLowerCase());
}

# 将字符转换为本地化大写
function _Char_toLocaleUpper(char)
{
    return _Utils_chr(char.toLocaleUpperCase());
}

# 将字符转换为本地化小写
function _Char_toLocaleLower(char)
{
    return _Utils_chr(char.toLocaleLowerCase());
}

# 将 JSON 错误转换为字符串
/**_UNUSED/
function _Json_errorToString(error)
{
    return $elm$json$Json$Decode$errorToString(error);
}
//*/

# 核心解码器，返回成功的消息
function _Json_succeed(msg)
{
    # 返回一个包含 $ 和 a 两个键值对的字典
    return {
        $: 0,  # 键为 $，值为 0
        a: msg  # 键为 a，值为 msg（假设 msg 是之前定义过的变量）
    };
// 定义一个函数，用于返回一个包含错误信息的对象
function _Json_fail(msg)
{
    return {
        $: 1,
        a: msg
    };
}

// 定义一个函数，用于返回一个包含解码器的对象
function _Json_decodePrim(decoder)
{
    return { $: 2, b: decoder };
}

// 定义一个解码整数的函数
var _Json_decodeInt = _Json_decodePrim(function(value) {
    return (typeof value !== 'number')
        ? _Json_expecting('an INT', value)
        :
    (-2147483647 < value && value < 2147483647 && (value | 0) === value)
        ? $elm$core$Result$Ok(value)
        :
    (isFinite(value) && !(value % 1))
        ? $elm$core$Result$Ok(value)
        : _Json_expecting('an INT', value);
});

// 定义一个解码布尔值的函数
var _Json_decodeBool = _Json_decodePrim(function(value) {
    return (typeof value === 'boolean')
        ? $elm$core$Result$Ok(value)
        : _Json_expecting('a BOOL', value);
});

// 定义一个解码浮点数的函数
var _Json_decodeFloat = _Json_decodePrim(function(value) {
    return (typeof value === 'number')
        ? $elm$core$Result$Ok(value)
        : _Json_expecting('a FLOAT', value);
});

// 定义一个解码任意值的函数
var _Json_decodeValue = _Json_decodePrim(function(value) {
    return $elm$core$Result$Ok(_Json_wrap(value));
});

// 定义一个解码字符串的函数
var _Json_decodeString = _Json_decodePrim(function(value) {
    return (typeof value === 'string')
        ? $elm$core$Result$Ok(value)
        : (value instanceof String)
            ? $elm$core$Result$Ok(value + '')
            : _Json_expecting('a STRING', value);
});

// 定义一个解码列表的函数
function _Json_decodeList(decoder) { return { $: 3, b: decoder }; }

// 定义一个解码数组的函数
function _Json_decodeArray(decoder) { return { $: 4, b: decoder }; }

// 定义一个解码空值的函数
function _Json_decodeNull(value) { return { $: 5, c: value }; }

// 定义一个带字段的解码器函数
var _Json_decodeField = F2(function(field, decoder)
{
    return {
        $: 6,
        d: field,
        b: decoder
    };
});

// 定义一个带索引的解码器函数
var _Json_decodeIndex = F2(function(index, decoder)
{
    return {
        $: 7,
        e: index,
        b: decoder
    };
});

// 定义一个解码键值对的函数
function _Json_decodeKeyValuePairs(decoder)
{
    return {
        $: 8,
        b: decoder
    };
}

// 定义一个映射多个解码器的函数
function _Json_mapMany(f, decoders)
{
    return {
        $: 9,
        f: f,
        g: decoders
    };
}
// 定义一个函数，接受一个回调函数和一个解码器，返回一个对象
var _Json_andThen = F2(function(callback, decoder)
{
    return {
        $: 10, // 标识符，表示这是一个 andThen 函数
        b: decoder, // 解码器
        h: callback // 回调函数
    };
});

// 定义一个函数，接受一个解码器数组，返回一个对象
function _Json_oneOf(decoders)
{
    return {
        $: 11, // 标识符，表示这是一个 oneOf 函数
        g: decoders // 解码器数组
    };
}


// 解码对象

// 定义一个函数，接受一个函数和一个解码器，返回一个对象
var _Json_map1 = F2(function(f, d1)
{
    return _Json_mapMany(f, [d1]);
});

// 定义一个函数，接受一个函数和两个解码器，返回一个对象
var _Json_map2 = F3(function(f, d1, d2)
{
    return _Json_mapMany(f, [d1, d2]);
});

// 定义一个函数，接受一个函数和三个解码器，返回一个对象
var _Json_map3 = F4(function(f, d1, d2, d3)
{
    return _Json_mapMany(f, [d1, d2, d3]);
});

// 定义一个函数，接受一个函数和四个解码器，返回一个对象
var _Json_map4 = F5(function(f, d1, d2, d3, d4)
{
    return _Json_mapMany(f, [d1, d2, d3, d4]);
});

// 定义一个函数，接受一个函数和五个解码器，返回一个对象
var _Json_map5 = F6(function(f, d1, d2, d3, d4, d5)
{
    return _Json_mapMany(f, [d1, d2, d3, d4, d5]);
});

// 定义一个函数，接受一个函数和六个解码器，返回一个对象
var _Json_map6 = F7(function(f, d1, d2, d3, d4, d5, d6)
{
    return _Json_mapMany(f, [d1, d2, d3, d4, d5, d6]);
});

// 定义一个函数，接受一个函数和七个解码器，返回一个对象
var _Json_map7 = F8(function(f, d1, d2, d3, d4, d5, d6, d7)
{
    return _Json_mapMany(f, [d1, d2, d3, d4, d5, d6, d7]);
});

// 定义一个函数，接受一个函数和八个解码器，返回一个对象
var _Json_map8 = F9(function(f, d1, d2, d3, d4, d5, d6, d7, d8)
{
    return _Json_mapMany(f, [d1, d2, d3, d4, d5, d6, d7, d8]);
});


// 解码

// 定义一个函数，接受一个解码器和一个字符串，尝试解析字符串为 JSON 对象，然后运行解码器
var _Json_runOnString = F2(function(decoder, string)
{
    try
    {
        var value = JSON.parse(string); // 尝试解析字符串为 JSON 对象
        return _Json_runHelp(decoder, value); // 运行解码器
    }
    catch (e)
    {
        return $elm$core$Result$Err(A2($elm$json$Json$Decode$Failure, 'This is not valid JSON! ' + e.message, _Json_wrap(string))); // 返回错误信息
    }
});

// 定义一个函数，接受一个解码器和一个值，运行解码器
var _Json_run = F2(function(decoder, value)
{
    return _Json_runHelp(decoder, _Json_unwrap(value)); // 运行解码器
});

// 定义一个辅助函数，接受一个解码器和一个值，根据解码器类型进行处理
function _Json_runHelp(decoder, value)
{
    switch (decoder.$) // 根据解码器类型进行处理
    }
}

// 定义一个函数，接受一个解码器、一个值和一个转换函数，对数组进行解码
function _Json_runArrayDecoder(decoder, value, toElmValue)
{
    var len = value.length; // 获取数组长度
    var array = new Array(len); // 创建一个与数组长度相同的新数组
    for (var i = 0; i < len; i++) // 遍历数组
    {
        // 对值数组中的每个元素进行 JSON 解码
        var result = _Json_runHelp(decoder, value[i]);
        // 如果解码结果不是成功的话
        if (!$elm$core$Result$isOk(result))
        {
            // 返回错误结果，包含索引和错误信息
            return $elm$core$Result$Err(A2($elm$json$Json$Decode$Index, i, result.a));
        }
        // 将解码成功的值存入数组
        array[i] = result.a;
    }
    // 返回成功的结果，将数组转换为 Elm 值
    return $elm$core$Result$Ok(toElmValue(array));
// 判断给定的值是否为数组或者文件列表
function _Json_isArray(value)
{
    return Array.isArray(value) || (typeof FileList !== 'undefined' && value instanceof FileList);
}

// 将 JavaScript 数组转换为 Elm 数组
function _Json_toElmArray(array)
{
    return A2($elm$core$Array$initialize, array.length, function(i) { return array[i]; });
}

// 返回一个包含期望类型错误信息的错误结果
function _Json_expecting(type, value)
{
    return $elm$core$Result$Err(A2($elm$json$Json$Decode$Failure, 'Expecting ' + type, _Json_wrap(value)));
}

// 判断两个值是否相等
function _Json_equality(x, y)
{
    if (x === y)
    {
        return true;
    }

    if (x.$ !== y.$)
    {
        return false;
    }

    switch (x.$)
    {
        case 0:
        case 1:
            return x.a === y.a;

        case 2:
            return x.b === y.b;

        case 5:
            return x.c === y.c;

        case 3:
        case 4:
        case 8:
            return _Json_equality(x.b, y.b);

        case 6:
            return x.d === y.d && _Json_equality(x.b, y.b);

        case 7:
            return x.e === y.e && _Json_equality(x.b, y.b);

        case 9:
            return x.f === y.f && _Json_listEquality(x.g, y.g);

        case 10:
            return x.h === y.h && _Json_equality(x.b, y.b);

        case 11:
            return _Json_listEquality(x.g, y.g);
    }
}

// 判断两个列表的解码器是否相等
function _Json_listEquality(aDecoders, bDecoders)
{
    var len = aDecoders.length;
    if (len !== bDecoders.length)
    {
        return false;
    }
    for (var i = 0; i < len; i++)
    {
        if (!_Json_equality(aDecoders[i], bDecoders[i]))
        {
            return false;
        }
    }
    return true;
}

// 将值编码为 JSON 字符串
var _Json_encode = F2(function(indentLevel, value)
{
    return JSON.stringify(_Json_unwrap(value), null, indentLevel) + '';
});

// 以下四个函数未被使用，可以忽略
function _Json_wrap_UNUSED(value) { return { $: 0, a: value }; }
function _Json_unwrap_UNUSED(value) { return value.a; }
function _Json_wrap(value) { return value; }
function _Json_unwrap(value) { return value; }
function _Json_emptyArray() { return []; }
// 定义一个函数，返回一个空的 JSON 对象
function _Json_emptyObject() { return {}; }

// 定义一个函数，用于向 JSON 对象添加字段
var _Json_addField = F3(function(key, value, object)
{
    object[key] = _Json_unwrap(value);
    return object;
});

// 定义一个函数，用于向 JSON 数组添加条目
function _Json_addEntry(func)
{
    return F2(function(entry, array)
    {
        array.push(_Json_unwrap(func(entry)));
        return array;
    });
}

// 定义一个变量，用于将 null 值进行 JSON 编码
var _Json_encodeNull = _Json_wrap(null);

// 以下为任务相关的函数和变量

// 定义一个函数，用于创建一个成功的任务
function _Scheduler_succeed(value)
{
    return {
        $: 0,
        a: value
    };
}

// 定义一个函数，用于创建一个失败的任务
function _Scheduler_fail(error)
{
    return {
        $: 1,
        a: error
    };
}

// 定义一个函数，用于创建一个绑定任务
function _Scheduler_binding(callback)
{
    return {
        $: 2,
        b: callback,
        c: null
    };
}

// 定义一个函数，用于创建一个串联任务
var _Scheduler_andThen = F2(function(callback, task)
{
    return {
        $: 3,
        b: callback,
        d: task
    };
});

// 定义一个函数，用于创建一个错误处理任务
var _Scheduler_onError = F2(function(callback, task)
{
    return {
        $: 4,
        b: callback,
        d: task
    };
});

// 定义一个函数，用于创建一个接收任务
function _Scheduler_receive(callback)
{
    return {
        $: 5,
        b: callback
    };
}

// 以下为进程相关的函数和变量

// 定义一个变量，用于生成进程的唯一标识
var _Scheduler_guid = 0;

// 定义一个函数，用于直接生成一个进程
function _Scheduler_rawSpawn(task)
{
    var proc = {
        $: 0,
        e: _Scheduler_guid++,
        f: task,
        g: null,
        h: []
    };

    _Scheduler_enqueue(proc);

    return proc;
}

// 定义一个函数，用于生成一个绑定任务，用于生成进程
function _Scheduler_spawn(task)
{
    return _Scheduler_binding(function(callback) {
        callback(_Scheduler_succeed(_Scheduler_rawSpawn(task)));
    });
}

// 定义一个函数，用于向进程发送消息
function _Scheduler_rawSend(proc, msg)
{
    proc.h.push(msg);
    _Scheduler_enqueue(proc);
}

// 定义一个函数，用于生成一个绑定任务，用于向进程发送消息
var _Scheduler_send = A2(function(proc, msg)
{
    return _Scheduler_binding(function(callback) {
        _Scheduler_rawSend(proc, msg);
        callback(_Scheduler_succeed(_Utils_Tuple0));
    });
});

// 定义一个函数，用于生成一个绑定任务，用于终止进程
function _Scheduler_kill(proc)
{
    return _Scheduler_binding(function(callback) {
        var task = proc.f;
        if (task.$ === 2 && task.c)
        {
            task.c();
        }

        proc.f = null;

        callback(_Scheduler_succeed(_Utils_Tuple0));
    });
    });
    # 结束一个代码块或函数的定义
// 定义了一个名为 Process 的类型别名，包含了进程的各个属性
type alias Process =
  { $ : tag
  , id : unique_id
  , root : Task
  , stack : null | { $: SUCCEED | FAIL, a: callback, b: stack }
  , mailbox : [msg]
  }

// 定义了一个变量 _Scheduler_working，用于标识调度器是否正在工作
var _Scheduler_working = false;
// 定义了一个数组变量 _Scheduler_queue，用于存储进程队列
var _Scheduler_queue = [];

// 定义了一个函数 _Scheduler_enqueue，用于将进程加入队列
function _Scheduler_enqueue(proc)
{
    // 将进程加入队列
    _Scheduler_queue.push(proc);
    // 如果调度器正在工作，则直接返回
    if (_Scheduler_working)
    {
        return;
    }
    // 标识调度器正在工作
    _Scheduler_working = true;
    // 从队列中取出进程并执行
    while (proc = _Scheduler_queue.shift())
    {
        _Scheduler_step(proc);
    }
    // 标识调度器工作结束
    _Scheduler_working = false;
}

// 定义了一个函数 _Scheduler_step，用于执行进程的下一步操作
function _Scheduler_step(proc)
{
    while (proc.f)
    {
        // 获取根任务的标签
        var rootTag = proc.f.$;
        // 如果根任务标签为 0 或 1
        if (rootTag === 0 || rootTag === 1)
        {
            // 循环直到找到与根任务标签相同的任务
            while (proc.g && proc.g.$ !== rootTag)
            {
                proc.g = proc.g.i;
            }
            // 如果没有找到相同标签的任务，则返回
            if (!proc.g)
            {
                return;
            }
            // 执行下一个任务
            proc.f = proc.g.b(proc.f.a);
            proc.g = proc.g.i;
        }
        // 如果根任务标签为 2
        else if (rootTag === 2)
        {
            // 执行回调函数，并将新的根任务加入队列
            proc.f.c = proc.f.b(function(newRoot) {
                proc.f = newRoot;
                _Scheduler_enqueue(proc);
            });
            return;
        }
        // 如果根任务标签为 5
        else if (rootTag === 5)
        {
            // 如果邮箱中没有消息，则返回
            if (proc.h.length === 0)
            {
                return;
            }
            // 执行下一个任务
            proc.f = proc.f.b(proc.h.shift());
        }
        // 如果根任务标签为 3 或 4
        else // if (rootTag === 3 || rootTag === 4)
        {
            // 将当前任务加入栈中，并执行下一个任务
            proc.g = {
                $: rootTag === 3 ? 0 : 1,
                b: proc.f.b,
                i: proc.g
            };
            proc.f = proc.f.d;
        }
    }
}

// 定义了一个函数 _Process_sleep，用于实现进程的休眠功能
function _Process_sleep(time)
{
    return _Scheduler_binding(function(callback) {
        // 设置定时器，当时间到达后执行回调函数
        var id = setTimeout(function() {
            callback(_Scheduler_succeed(_Utils_Tuple0));
        }, time);

        // 返回取消定时器的函数
        return function() { clearTimeout(id); };
    });
}

// 定义了一个变量 _Platform_worker，用于创建工作线程
var _Platform_worker = F4(function(impl, flagDecoder, debugMetadata, args)
{
    # 调用 _Platform_initialize 函数，并传入参数
    return _Platform_initialize(
        flagDecoder,  # 参数1：flagDecoder
        args,  # 参数2：args
        impl.aB,  # 参数3：impl.aB
        impl.aJ,  # 参数4：impl.aJ
        impl.aH,  # 参数5：impl.aH
        function() { return function() {} }  # 参数6：匿名函数
    );
// 初始化程序
function _Platform_initialize(flagDecoder, args, init, update, subscriptions, stepperBuilder)
{
    // 运行 JSON 解码器，将 flags 解码为 JSON 值
    var result = A2(_Json_run, flagDecoder, _Json_wrap(args ? args['flags'] : undefined));
    // 如果解码成功，则继续执行；否则抛出错误
    $elm$core$Result$isOk(result) || _Debug_crash(2 /**_UNUSED/, _Json_errorToString(result.a) /**/);
    // 创建空的管理器对象
    var managers = {};
    // 调用 init 函数，获取初始化模型和命令
    var initPair = init(result.a);
    // 获取初始化模型
    var model = initPair.a;
    // 根据模型创建 stepper 函数
    var stepper = stepperBuilder(sendToApp, model);
    // 设置效果管理器
    var ports = _Platform_setupEffects(managers, sendToApp);

    // 定义发送消息到应用的函数
    function sendToApp(msg, viewMetadata)
    {
        // 调用 update 函数，获取新的模型和命令
        var pair = A2(update, msg, model);
        // 更新模型
        stepper(model = pair.a, viewMetadata);
        // 将命令加入到效果队列中
        _Platform_enqueueEffects(managers, pair.b, subscriptions(model));
    }

    // 将初始化命令加入到效果队列中
    _Platform_enqueueEffects(managers, initPair.b, subscriptions(model));

    // 如果存在 ports，则返回包含 ports 的对象；否则返回空对象
    return ports ? { ports: ports } : {};
}

// 跟踪预加载
//
// 这是 elm/browser 和 elm/http 中用于注册由 init 触发的任何 HTTP 请求的代码。
var _Platform_preload;

// 注册预加载
function _Platform_registerPreload(url)
{
    _Platform_preload.add(url);
}

// 效果管理器
var _Platform_effectManagers = {};

// 设置效果
function _Platform_setupEffects(managers, sendToApp)
{
    var ports;

    // 设置所有必要的效果管理器
    for (var key in _Platform_effectManagers)
    {
        var manager = _Platform_effectManagers[key];

        if (manager.a)
        {
            ports = ports || {};
            ports[key] = manager.a(key, sendToApp);
        }

        managers[key] = _Platform_instantiateManager(manager, sendToApp);
    }

    return ports;
}

// 创建管理器
function _Platform_createManager(init, onEffects, onSelfMsg, cmdMap, subMap)
{
    return {
        b: init,
        c: onEffects,
        d: onSelfMsg,
        e: cmdMap,
        f: subMap
    };
}

// 实例化管理器
function _Platform_instantiateManager(info, sendToApp)
{
    var router = {
        g: sendToApp,
        h: undefined
    };

    var onEffects = info.c;
    var onSelfMsg = info.d;
    # 将info.e赋值给cmdMap变量
    var cmdMap = info.e;
    # 将info.f赋值给subMap变量
    var subMap = info.f;

    # 定义一个名为loop的函数，参数为state
    function loop(state)
    {
        # 返回一个新的任务，该任务在接收到消息后继续执行loop函数
        return A2(_Scheduler_andThen, loop, _Scheduler_receive(function(msg)
        {
            # 获取消息中的值
            var value = msg.a;

            # 如果消息类型为0
            if (msg.$ === 0)
            {
                # 调用onSelfMsg函数处理消息，并返回结果
                return A3(onSelfMsg, router, value, state);
            }

            # 如果cmdMap和subMap都存在
            if (cmdMap && subMap)
            {
                # 调用onEffects函数处理消息，并返回结果
                return A4(onEffects, router, value.i, value.j, state);
            }
            # 如果cmdMap存在但subMap不存在
            else
            {
                # 调用onEffects函数处理消息，并返回结果
                return A3(onEffects, router, cmdMap ? value.i : value.j, state);
            }
        }));
    }

    # 将loop函数的执行结果赋值给router.h，并使用_Scheduler_rawSpawn函数进行原始的任务调度
    return router.h = _Scheduler_rawSpawn(A2(_Scheduler_andThen, loop, info.b));
// ROUTING

// 定义一个函数，用于向应用发送消息
var _Platform_sendToApp = F2(function(router, msg)
{
    // 返回一个绑定了回调函数的调度器
    return _Scheduler_binding(function(callback)
    {
        // 调用路由器的消息处理函数
        router.g(msg);
        // 调用回调函数，传递成功的消息
        callback(_Scheduler_succeed(_Utils_Tuple0));
    });
});

// 定义一个函数，用于向自身发送消息
var _Platform_sendToSelf = F2(function(router, msg)
{
    // 使用调度器发送消息到路由器的处理函数
    return A2(_Scheduler_send, router.h, {
        $: 0,
        a: msg
    });
});

// BAGS

// 定义一个函数，用于创建一个包含值的叶子节点
function _Platform_leaf(home)
{
    return function(value)
    {
        return {
            $: 1,
            k: home,
            l: value
        };
    };
}

// 定义一个函数，用于批量处理包
function _Platform_batch(list)
{
    return {
        $: 2,
        m: list
    };
}

// 定义一个函数，用于映射标签和包
var _Platform_map = F2(function(tagger, bag)
{
    return {
        $: 3,
        n: tagger,
        o: bag
    }
});

// PIPE BAGS INTO EFFECT MANAGERS
//
// Effects must be queued!
//
// Say your init contains a synchronous command, like Time.now or Time.here
//
//   - This will produce a batch of effects (FX_1)
//   - The synchronous task triggers the subsequent `update` call
//   - This will produce a batch of effects (FX_2)
//
// If we just start dispatching FX_2, subscriptions from FX_2 can be processed
// before subscriptions from FX_1. No good! Earlier versions of this code had
// this problem, leading to these reports:
//
//   https://github.com/elm/core/issues/980
//   https://github.com/elm/core/pull/981
//   https://github.com/elm/compiler/issues/1776
//
// The queue is necessary to avoid ordering issues for synchronous commands.

// Why use true/false here? Why not just check the length of the queue?
// The goal is to detect "are we currently dispatching effects?" If we
// are, we need to bail and let the ongoing while loop handle things.
//
// Now say the queue has 1 element. When we dequeue the final element,
// the queue will be empty, but we are still actively dispatching effects.
// So you could get queue jumping in a really tricky category of cases.
//
// 定义一个队列，用于存储效果
var _Platform_effectsQueue = [];
// 定义一个标志，用于表示是否正在处理效果
var _Platform_effectsActive = false;
// 将效果添加到队列中，等待执行
function _Platform_enqueueEffects(managers, cmdBag, subBag)
{
    _Platform_effectsQueue.push({ p: managers, q: cmdBag, r: subBag });

    // 如果效果队列已经激活，则直接返回
    if (_Platform_effectsActive) return;

    // 标记效果队列为激活状态
    _Platform_effectsActive = true;
    // 依次执行效果队列中的效果
    for (var fx; fx = _Platform_effectsQueue.shift(); )
    {
        _Platform_dispatchEffects(fx.p, fx.q, fx.r);
    }
    // 标记效果队列为非激活状态
    _Platform_effectsActive = false;
}

// 执行效果
function _Platform_dispatchEffects(managers, cmdBag, subBag)
{
    // 创建一个空的效果字典
    var effectsDict = {};
    // 收集命令效果
    _Platform_gatherEffects(true, cmdBag, effectsDict, null);
    // 收集子效果
    _Platform_gatherEffects(false, subBag, effectsDict, null);

    // 遍历管理器，发送效果
    for (var home in managers)
    {
        _Scheduler_rawSend(managers[home], {
            $: 'fx',
            a: effectsDict[home] || { i: _List_Nil, j: _List_Nil }
        });
    }
}

// 收集效果
function _Platform_gatherEffects(isCmd, bag, effectsDict, taggers)
{
    switch (bag.$)
    {
        // 如果是单个效果
        case 1:
            var home = bag.k;
            var effect = _Platform_toEffect(isCmd, home, taggers, bag.l);
            effectsDict[home] = _Platform_insert(isCmd, effect, effectsDict[home]);
            return;

        // 如果是效果列表
        case 2:
            for (var list = bag.m; list.b; list = list.b) // WHILE_CONS
            {
                _Platform_gatherEffects(isCmd, list.a, effectsDict, taggers);
            }
            return;

        // 如果是嵌套效果
        case 3:
            _Platform_gatherEffects(isCmd, bag.o, effectsDict, {
                s: bag.n,
                t: taggers
            });
            return;
    }
}

// 将值应用到标签器
function _Platform_toEffect(isCmd, home, taggers, value)
{
    function applyTaggers(x)
    {
        for (var temp = taggers; temp; temp = temp.t)
        {
            x = temp.s(x);
        }
        return x;
    }

    // 获取对应的效果管理器
    var map = isCmd
        ? _Platform_effectManagers[home].e
        : _Platform_effectManagers[home].f;

    // 应用标签器并返回效果
    return A2(map, applyTaggers, value)
}

// 将新效果插入到效果列表中
function _Platform_insert(isCmd, newEffect, effects)
{
    // 如果效果列表为空，则创建一个空的效果列表
    effects = effects || { i: _List_Nil, j: _List_Nil };
    # 如果 isCmd 为真，则将 newEffect 添加到 effects.i 中
    # 否则将 newEffect 添加到 effects.j 中
    isCmd
        ? (effects.i = _List_Cons(newEffect, effects.i))
        : (effects.j = _List_Cons(newEffect, effects.j));
    # 返回 effects
    return effects;
// 检查端口名称是否已经存在于效果管理器中，如果存在则触发错误
function _Platform_checkPortName(name)
{
    if (_Platform_effectManagers[name])
    {
        _Debug_crash(3, name)
    }
}

// 创建一个输出端口，将其添加到效果管理器中
function _Platform_outgoingPort(name, converter)
{
    _Platform_checkPortName(name);
    _Platform_effectManagers[name] = {
        e: _Platform_outgoingPortMap,
        u: converter,
        a: _Platform_setupOutgoingPort
    };
    return _Platform_leaf(name);
}

// 创建一个输入端口，将其添加到效果管理器中
function _Platform_incomingPort(name, converter)
{
    _Platform_checkPortName(name);
    _Platform_effectManagers[name] = {
        f: _Platform_incomingPortMap,
        u: converter,
        a: _Platform_setupIncomingPort
    };
}
    # 返回一个平台叶子对象，根据给定的名称
    return _Platform_leaf(name);
}
// 定义一个名为 _Platform_incomingPortMap 的函数，接受两个标签器参数，并返回一个函数
var _Platform_incomingPortMap = F2(function(tagger, finalTagger)
{
    return function(value)
    {
        return tagger(finalTagger(value));
    };
});

// 定义一个名为 _Platform_setupIncomingPort 的函数，接受名称和 sendToApp 参数
function _Platform_setupIncomingPort(name, sendToApp)
{
    var subs = _List_Nil; // 初始化 subs 为空列表
    var converter = _Platform_effectManagers[name].u; // 获取 _Platform_effectManagers[name].u 的值并赋给 converter

    // 创建管理器
    var init = _Scheduler_succeed(null); // 调用 _Scheduler_succeed 函数并将 null 作为参数传入，将返回值赋给 init
    _Platform_effectManagers[name].b = init; // 将 init 赋给 _Platform_effectManagers[name].b
    _Platform_effectManagers[name].c = F3(function(router, subList, state) // 将一个匿名函数赋给 _Platform_effectManagers[name].c
    {
        subs = subList; // 将 subList 赋给 subs
        return init; // 返回 init
    });

    // 公共 API
    function send(incomingValue) // 定义名为 send 的函数，接受 incomingValue 参数
    {
        var result = A2(_Json_run, converter, _Json_wrap(incomingValue)); // 调用 _Json_run 函数，并将 converter 和 _Json_wrap(incomingValue) 作为参数传入，将返回值赋给 result
        $elm$core$Result$isOk(result) || _Debug_crash(4, name, result.a); // 如果 result 不是 OK，则调用 _Debug_crash 函数

        var value = result.a; // 获取 result.a 的值并赋给 value
        for (var temp = subs; temp.b; temp = temp.b) // 循环遍历 subs
        {
            sendToApp(temp.a(value)); // 调用 sendToApp 函数，并将 temp.a(value) 作为参数传入
        }
    }

    return { send: send }; // 返回一个对象，包含 send 函数
}

// 导出 Elm 模块
function _Platform_export(exports) // 定义名为 _Platform_export 的函数，接受 exports 参数
{
    scope['Elm'] // 如果 scope 中存在 'Elm'
        ? _Platform_mergeExportsProd(scope['Elm'], exports) // 调用 _Platform_mergeExportsProd 函数
        : scope['Elm'] = exports; // 否则将 exports 赋给 scope['Elm']
}

// 合并导出的 Elm 模块（生产环境）
function _Platform_mergeExportsProd(obj, exports) // 定义名为 _Platform_mergeExportsProd 的函数，接受 obj 和 exports 参数
{
    for (var name in exports) // 遍历 exports
    {
        (name in obj) // 如果 obj 中存在 name
            ? (name == 'init') // 如果 name 等于 'init'
                ? _Debug_crash(6) // 调用 _Debug_crash 函数
                : _Platform_mergeExportsProd(obj[name], exports[name]) // 否则递归调用 _Platform_mergeExportsProd 函数
            : (obj[name] = exports[name]); // 将 exports[name] 赋给 obj[name]
    }
}

// 导出未使用的 Elm 模块
function _Platform_export_UNUSED(exports) // 定义名为 _Platform_export_UNUSED 的函数，接受 exports 参数
{
    scope['Elm'] // 如果 scope 中存在 'Elm'
        ? _Platform_mergeExportsDebug('Elm', scope['Elm'], exports) // 调用 _Platform_mergeExportsDebug 函数
        : scope['Elm'] = exports; // 否则将 exports 赋给 scope['Elm']
}

// 合并导出的 Elm 模块（调试模式）
function _Platform_mergeExportsDebug(moduleName, obj, exports) // 定义名为 _Platform_mergeExportsDebug 的函数，接受 moduleName、obj 和 exports 参数
{
    for (var name in exports) // 遍历 exports
    {
        // 如果对象中存在指定的属性名
        (name in obj)
            // 如果属性名为'init'，则调用_Debug_crash函数，否则调用_Platform_mergeExportsDebug函数
            ? (name == 'init')
                ? _Debug_crash(6, moduleName)
                : _Platform_mergeExportsDebug(moduleName + '.' + name, obj[name], exports[name])
            // 如果对象中不存在指定的属性名，则将exports中对应的属性值赋给obj
            : (obj[name] = exports[name]);
    }
// HELPERS

// 定义一个变量，用于重定向 href 到应用程序
var _VirtualDom_divertHrefToApp;

// 定义一个变量，如果 document 存在则使用 document，否则使用空对象
var _VirtualDom_doc = typeof document !== 'undefined' ? document : {};

// 定义一个函数，用于向父节点添加子节点
function _VirtualDom_appendChild(parent, child)
{
    parent.appendChild(child);
}

// 定义一个初始化函数，接受四个参数，其中 virtualNode 是虚拟节点，flagDecoder 是标志解码器，debugMetadata 是调试元数据，args 是参数
var _VirtualDom_init = F4(function(virtualNode, flagDecoder, debugMetadata, args)
{
    // NOTE: this function needs _Platform_export available to work
    // 注意：此函数需要 _Platform_export 可用才能工作

    /**/
    var node = args['node'];
    //*/
    /**_UNUSED/
    var node = args && args['node'] ? args['node'] : _Debug_crash(0);
    //*/

    // 用虚拟节点渲染并替换节点的父节点
    node.parentNode.replaceChild(
        _VirtualDom_render(virtualNode, function() {}),
        node
    );

    return {};
});



// TEXT

// 定义一个函数，用于创建文本节点
function _VirtualDom_text(string)
{
    return {
        $: 0,
        a: string
    };
}



// NODE

// 定义一个函数，用于创建不带命名空间的节点
var _VirtualDom_nodeNS = F2(function(namespace, tag)
{
    return F2(function(factList, kidList)
    {
        for (var kids = [], descendantsCount = 0; kidList.b; kidList = kidList.b) // WHILE_CONS
        {
            var kid = kidList.a;
            descendantsCount += (kid.b || 0);
            kids.push(kid);
        }
        descendantsCount += kids.length;

        return {
            $: 1,
            c: tag,
            d: _VirtualDom_organizeFacts(factList),
            e: kids,
            f: namespace,
            b: descendantsCount
        };
    });
});

// 定义一个不带命名空间的节点
var _VirtualDom_node = _VirtualDom_nodeNS(undefined);



// KEYED NODE

// 定义一个函数，用于创建带命名空间的节点
var _VirtualDom_keyedNodeNS = F2(function(namespace, tag)
{
    return F2(function(factList, kidList)
    {
        for (var kids = [], descendantsCount = 0; kidList.b; kidList = kidList.b) // WHILE_CONS
        {
            var kid = kidList.a;
            descendantsCount += (kid.b.b || 0);
            kids.push(kid);
        }
        descendantsCount += kids.length;

        return {
            $: 2,
            c: tag,
            d: _VirtualDom_organizeFacts(factList),
            e: kids,
            f: namespace,
            b: descendantsCount
        };
    });
});
// 创建一个未命名的键控节点
var _VirtualDom_keyedNode = _VirtualDom_keyedNodeNS(undefined);

// 自定义虚拟节点，包含属性 $: 3，d: factList，g: model，h: render，i: diff
function _VirtualDom_custom(factList, model, render, diff)
{
    return {
        $: 3,
        d: _VirtualDom_organizeFacts(factList),
        g: model,
        h: render,
        i: diff
    };
}

// 映射函数，接受一个标签器和节点，返回一个包含属性 $: 4，j: tagger，k: node，b: 1 + (node.b || 0) 的对象
var _VirtualDom_map = F2(function(tagger, node)
{
    return {
        $: 4,
        j: tagger,
        k: node,
        b: 1 + (node.b || 0)
    };
});

// 惰性计算函数，接受一个引用和一个惰性计算函数，返回一个包含属性 $: 5，l: refs，m: thunk，k: undefined 的对象
function _VirtualDom_thunk(refs, thunk)
{
    return {
        $: 5,
        l: refs,
        m: thunk,
        k: undefined
    };
}

// 惰性计算函数，接受一个函数和一个参数，返回一个惰性计算对象
var _VirtualDom_lazy = F2(function(func, a)
{
    return _VirtualDom_thunk([func, a], function() {
        return func(a);
    });
});

// 以下是惰性计算函数的重载，分别接受不同数量的参数，返回相应的惰性计算对象
var _VirtualDom_lazy2 = F3(function(func, a, b) { /* ... */ });
var _VirtualDom_lazy3 = F4(function(func, a, b, c) { /* ... */ });
var _VirtualDom_lazy4 = F5(function(func, a, b, c, d) { /* ... */ });
var _VirtualDom_lazy5 = F6(function(func, a, b, c, d, e) { /* ... */ });
var _VirtualDom_lazy6 = F7(function(func, a, b, c, d, e, f) { /* ... */ });
var _VirtualDom_lazy7 = F8(function(func, a, b, c, d, e, f, g) { /* ... */ });
var _VirtualDom_lazy8 = F9(function(func, a, b, c, d, e, f, g, h) { /* ... */ });

// 空的 FACTS 部分
// 定义一个函数，接受两个参数，返回一个包含键和处理程序的对象
var _VirtualDom_on = F2(function(key, handler)
{
    return {
        $: 'a0',
        n: key,
        o: handler
    };
});

// 定义一个函数，接受两个参数，返回一个包含键和值的对象，表示样式
var _VirtualDom_style = F2(function(key, value)
{
    return {
        $: 'a1',
        n: key,
        o: value
    };
});

// 定义一个函数，接受两个参数，返回一个包含键和值的对象，表示属性
var _VirtualDom_property = F2(function(key, value)
{
    return {
        $: 'a2',
        n: key,
        o: value
    };
});

// 定义一个函数，接受两个参数，返回一个包含键和值的对象，表示属性
var _VirtualDom_attribute = F2(function(key, value)
{
    return {
        $: 'a3',
        n: key,
        o: value
    };
});

// 定义一个函数，接受三个参数，返回一个包含键、命名空间和值的对象，表示属性
var _VirtualDom_attributeNS = F3(function(namespace, key, value)
{
    return {
        $: 'a4',
        n: key,
        o: { f: namespace, o: value }
    };
});

// 检查是否存在 XSS 攻击向量的函数

// 如果标签是 'script'，则返回 'p'，否则返回原标签
function _VirtualDom_noScript(tag)
{
    return tag == 'script' ? 'p' : tag;
}

// 如果属性键以 'on' 或 'formAction' 开头，则在键前加上 'data-'，否则返回原键
function _VirtualDom_noOnOrFormAction(key)
{
    return /^(on|formAction$)/i.test(key) ? 'data-' + key : key;
}

// 如果属性键是 'innerHTML' 或 'formAction'，则在键前加上 'data-'，否则返回原键
function _VirtualDom_noInnerHtmlOrFormAction(key)
{
    return key == 'innerHTML' || key == 'formAction' ? 'data-' + key : key;
}

// 如果属性值以 'javascript:' 开头，则返回空字符串，否则返回原值
function _VirtualDom_noJavaScriptUri(value)
{
    return /^javascript:/i.test(value.replace(/\s/g,'')) ? '' : value;
}

// 如果属性值以 'javascript:' 开头，则返回一个警告信息，否则返回原值
function _VirtualDom_noJavaScriptUri_UNUSED(value)
{
    return /^javascript:/i.test(value.replace(/\s/g,''))
        ? 'javascript:alert("This is an XSS vector. Please use ports or web components instead.")'
        : value;
}

// 如果属性值以 'javascript:' 或 'data:text/html' 开头，则返回空字符串，否则返回原值
function _VirtualDom_noJavaScriptOrHtmlUri(value)
{
    return /^\s*(javascript:|data:text\/html)/i.test(value) ? '' : value;
}

// 如果属性值以 'javascript:' 或 'data:text/html' 开头，则返回一个警告信息，否则返回原值
function _VirtualDom_noJavaScriptOrHtmlUri_UNUSED(value)
{
    return /^\s*(javascript:|data:text\/html)/i.test(value)
        ? 'javascript:alert("This is an XSS vector. Please use ports or web components instead.")'
        : value;
}

// MAP FACTS

// 定义一个函数，接受两个参数，返回一个处理过的属性对象
var _VirtualDom_mapAttribute = F2(function(func, attr)
{
    return (attr.$ === 'a0')
        ? A2(_VirtualDom_on, attr.n, _VirtualDom_mapHandler(func, attr.o))
        : attr;
});

// 定义一个函数，接受两个参数，返回一个处理过的处理程序对象
function _VirtualDom_mapHandler(func, handler)
    // 将虚拟 DOM 的事件处理器转换为整数标签
    var tag = $elm$virtual_dom$VirtualDom$toHandlerInt(handler);

    // 0 = Normal
    // 1 = MayStopPropagation
    // 2 = MayPreventDefault
    // 3 = Custom

    // 返回一个对象，包含处理器的标签和处理器的映射函数
    return {
        // 处理器的标签
        $: handler.$,
        // 如果标签不存在，则使用 A2 函数映射处理器的数据
        a:
            !tag
                ? A2($elm$json$Json$Decode$map, func, handler.a)
                :
            // 如果标签存在，则使用 A3 函数映射处理器的数据
            A3($elm$json$Json$Decode$map2,
                // 根据标签值选择不同的映射函数
                tag < 3
                    ? _VirtualDom_mapEventTuple
                    : _VirtualDom_mapEventRecord,
                // 成功的映射函数
                $elm$json$Json$Decode$succeed(func),
                // 处理器的数据
                handler.a
            )
    };
}
// 定义一个函数，用于将事件处理函数应用到事件元组上
var _VirtualDom_mapEventTuple = F2(function(func, tuple)
{
    return _Utils_Tuple2(func(tuple.a), tuple.b);
});

// 定义一个函数，用于将事件处理函数应用到事件记录上
var _VirtualDom_mapEventRecord = F2(function(func, record)
{
    return {
        t: func(record.t),
        R: record.R,
        O: record.O
    }
});



// ORGANIZE FACTS

// 定义一个函数，用于整理虚拟 DOM 节点的属性和事件
function _VirtualDom_organizeFacts(factList)
{
    for (var facts = {}; factList.b; factList = factList.b) // WHILE_CONS
    {
        var entry = factList.a;

        var tag = entry.$;
        var key = entry.n;
        var value = entry.o;

        if (tag === 'a2')
        {
            (key === 'className')
                ? _VirtualDom_addClass(facts, key, _Json_unwrap(value))
                : facts[key] = _Json_unwrap(value);

            continue;
        }

        var subFacts = facts[tag] || (facts[tag] = {});
        (tag === 'a3' && key === 'class')
            ? _VirtualDom_addClass(subFacts, key, value)
            : subFacts[key] = value;
    }

    return facts;
}

// 定义一个函数，用于向对象添加类名
function _VirtualDom_addClass(object, key, newClass)
{
    var classes = object[key];
    object[key] = classes ? classes + ' ' + newClass : newClass;
}



// RENDER

// 定义一个函数，用于渲染虚拟 DOM 节点
function _VirtualDom_render(vNode, eventNode)
{
    var tag = vNode.$;

    if (tag === 5)
    {
        return _VirtualDom_render(vNode.k || (vNode.k = vNode.m()), eventNode);
    }

    if (tag === 0)
    {
        return _VirtualDom_doc.createTextNode(vNode.a);
    }

    if (tag === 4)
    {
        var subNode = vNode.k;
        var tagger = vNode.j;

        while (subNode.$ === 4)
        {
            typeof tagger !== 'object'
                ? tagger = [tagger, subNode.j]
                : tagger.push(subNode.j);

            subNode = subNode.k;
        }

        var subEventRoot = { j: tagger, p: eventNode };
        var domNode = _VirtualDom_render(subNode, subEventRoot);
        domNode.elm_event_node_ref = subEventRoot;
        return domNode;
    }

    if (tag === 3)
    // ...（此处省略部分代码）
    {
        // 根据虚拟节点的属性创建真实 DOM 节点
        var domNode = vNode.h(vNode.g);
        // 应用虚拟节点的事件和属性到真实 DOM 节点
        _VirtualDom_applyFacts(domNode, eventNode, vNode.d);
        // 返回创建的真实 DOM 节点
        return domNode;
    }
    
    // 在这一点上 `tag` 必须是 1 或 2
    
    // 根据虚拟节点的命名空间和标签名创建真实 DOM 节点
    var domNode = vNode.f
        ? _VirtualDom_doc.createElementNS(vNode.f, vNode.c)
        : _VirtualDom_doc.createElement(vNode.c);
    
    // 如果需要将 href 重定向到应用程序，并且虚拟节点的标签名是 'a'，则添加点击事件监听器
    if (_VirtualDom_divertHrefToApp && vNode.c == 'a')
    {
        domNode.addEventListener('click', _VirtualDom_divertHrefToApp(domNode));
    }
    
    // 应用虚拟节点的事件和属性到真实 DOM 节点
    _VirtualDom_applyFacts(domNode, eventNode, vNode.d);
    
    // 遍历虚拟节点的子节点，将其渲染为真实 DOM 节点并添加到父节点中
    for (var kids = vNode.e, i = 0; i < kids.length; i++)
    {
        _VirtualDom_appendChild(domNode, _VirtualDom_render(tag === 1 ? kids[i] : kids[i].b, eventNode));
    }
    
    // 返回创建的真实 DOM 节点
    return domNode;
    }
// 应用给定的属性到虚拟 DOM 节点上
function _VirtualDom_applyFacts(domNode, eventNode, facts)
{
    // 遍历属性对象
    for (var key in facts)
    {
        var value = facts[key];

        // 如果属性是样式
        key === 'a1'
            ? _VirtualDom_applyStyles(domNode, value)
            :
        // 如果属性是事件
        key === 'a0'
            ? _VirtualDom_applyEvents(domNode, eventNode, value)
            :
        // 如果属性是普通属性
        key === 'a3'
            ? _VirtualDom_applyAttrs(domNode, value)
            :
        // 如果属性是命名空间属性
        key === 'a4'
            ? _VirtualDom_applyAttrsNS(domNode, value)
            :
        // 如果属性不是特殊属性，且节点上的值与给定值不同，则更新节点上的值
        ((key !== 'value' && key !== 'checked') || domNode[key] !== value) && (domNode[key] = value);
    }
}

// 应用样式到虚拟 DOM 节点上
function _VirtualDom_applyStyles(domNode, styles)
{
    var domNodeStyle = domNode.style;

    // 遍历样式对象
    for (var key in styles)
    {
        // 更新节点的样式
        domNodeStyle[key] = styles[key];
    }
}

// 应用普通属性到虚拟 DOM 节点上
function _VirtualDom_applyAttrs(domNode, attrs)
{
    // 遍历属性对象
    for (var key in attrs)
    {
        var value = attrs[key];
        // 如果属性值不是 undefined，则设置属性；否则移除属性
        typeof value !== 'undefined'
            ? domNode.setAttribute(key, value)
            : domNode.removeAttribute(key);
    }
}

// 应用命名空间属性到虚拟 DOM 节点上
function _VirtualDom_applyAttrsNS(domNode, nsAttrs)
{
    // 遍历命名空间属性对象
    for (var key in nsAttrs)
    {
        var pair = nsAttrs[key];
        var namespace = pair.f;
        var value = pair.o;
        // 如果属性值不是 undefined，则设置命名空间属性；否则移除命名空间属性
        typeof value !== 'undefined'
            ? domNode.setAttributeNS(namespace, key, value)
            : domNode.removeAttributeNS(namespace, key);
    }
}

// 应用事件到虚拟 DOM 节点上
function _VirtualDom_applyEvents(domNode, eventNode, events)
{
    var allCallbacks = domNode.elmFs || (domNode.elmFs = {});

    // 遍历事件对象
    for (var key in events)
    {
        // 获取事件处理函数
        var newHandler = events[key];
        // 获取旧的回调函数
        var oldCallback = allCallbacks[key];
    
        // 如果没有新的事件处理函数，则移除旧的事件监听器
        if (!newHandler)
        {
            domNode.removeEventListener(key, oldCallback);
            allCallbacks[key] = undefined;
            continue;
        }
    
        // 如果存在旧的回调函数
        if (oldCallback)
        {
            // 获取旧的事件处理函数
            var oldHandler = oldCallback.q;
            // 如果旧的事件处理函数和新的事件处理函数相同，则更新旧的回调函数为新的事件处理函数
            if (oldHandler.$ === newHandler.$)
            {
                oldCallback.q = newHandler;
                continue;
            }
            // 否则移除旧的事件监听器
            domNode.removeEventListener(key, oldCallback);
        }
    
        // 创建新的回调函数
        oldCallback = _VirtualDom_makeCallback(eventNode, newHandler);
        // 添加新的事件监听器
        domNode.addEventListener(key, oldCallback,
            // 如果支持 passive 事件监听，则设置为 passive
            _VirtualDom_passiveSupported
            && { passive: $elm$virtual_dom$VirtualDom$toHandlerInt(newHandler) < 2 }
        );
        // 更新回调函数
        allCallbacks[key] = oldCallback;
    }
// PASSIVE EVENTS

// 检测浏览器是否支持passive事件监听
var _VirtualDom_passiveSupported;

try
{
    window.addEventListener('t', null, Object.defineProperty({}, 'passive', {
        get: function() { _VirtualDom_passiveSupported = true; }
    }));
}
catch(e) {}


// EVENT HANDLERS

// 创建事件处理函数的回调函数
function _VirtualDom_makeCallback(eventNode, initialHandler)
{
    function callback(event)
    {
        var handler = callback.q;
        var result = _Json_runHelp(handler.a, event);

        // 如果结果不是Ok，则返回
        if (!$elm$core$Result$isOk(result))
        {
            return;
        }

        var tag = $elm$virtual_dom$VirtualDom$toHandlerInt(handler);

        // 0 = Normal
        // 1 = MayStopPropagation
        // 2 = MayPreventDefault
        // 3 = Custom

        var value = result.a;
        var message = !tag ? value : tag < 3 ? value.a : value.t;
        var stopPropagation = tag == 1 ? value.b : tag == 3 && value.R;
        var currentEventNode = (
            stopPropagation && event.stopPropagation(),
            (tag == 2 ? value.b : tag == 3 && value.O) && event.preventDefault(),
            eventNode
        );
        var tagger;
        var i;
        while (tagger = currentEventNode.j)
        {
            if (typeof tagger == 'function')
            {
                message = tagger(message);
            }
            else
            {
                for (var i = tagger.length; i--; )
                {
                    message = tagger[i](message);
                }
            }
            currentEventNode = currentEventNode.p;
        }
        currentEventNode(message, stopPropagation); // stopPropagation implies isSync
    }

    callback.q = initialHandler;

    return callback;
}

// 判断两个事件是否相等
function _VirtualDom_equalEvents(x, y)
{
    return x.$ == y.$ && _Json_equality(x.a, y.a);
}

// DIFF

// 对比两个虚拟DOM节点的差异
function _VirtualDom_diff(x, y)
{
    # 创建一个空数组用于存储补丁
    var patches = [];
    # 调用_VirtualDom_diffHelp函数，传入参数x, y, patches, 0，并将返回值存储在patches数组中
    _VirtualDom_diffHelp(x, y, patches, 0);
    # 返回存储了补丁的数组
    return patches;
}
// 定义一个函数，用于向 patches 数组中添加一个新的 patch 对象，并返回该对象
function _VirtualDom_pushPatch(patches, type, index, data)
{
    var patch = {
        $: type, // patch 类型
        r: index, // 节点索引
        s: data, // 数据
        t: undefined, // 保留字段
        u: undefined // 保留字段
    };
    patches.push(patch); // 将 patch 对象添加到 patches 数组中
    return patch; // 返回新添加的 patch 对象
}

// 定义一个辅助函数，用于比较两个节点的差异并生成补丁
function _VirtualDom_diffHelp(x, y, patches, index)
{
    if (x === y) // 如果两个节点相同，则无需比较
    {
        return;
    }

    var xType = x.$; // 获取节点 x 的类型
    var yType = y.$; // 获取节点 y 的类型

    // 如果节点类型不同，直接添加替换的 patch
    if (xType !== yType)
    {
        if (xType === 1 && yType === 2) // 如果 x 是 1，y 是 2，执行特定操作
        {
            y = _VirtualDom_dekey(y); // 对节点 y 进行特定操作
            yType = 1; // 更新节点 y 的类型
        }
        else
        {
            _VirtualDom_pushPatch(patches, 0, index, y); // 添加替换的 patch
            return;
        }
    }

    // 现在我们知道两个节点的类型相同
    switch (yType) // 根据节点类型执行不同的操作
    }
}

// 假设传入的数组长度相同，用于比较两个数组是否相等
function _VirtualDom_pairwiseRefEqual(as, bs)
{
    for (var i = 0; i < as.length; i++) // 遍历数组
    {
        if (as[i] !== bs[i]) // 如果数组元素不相等
        {
            return false; // 返回 false
        }
    }

    return true; // 返回 true
}

// 用于比较两个节点的差异并生成补丁
function _VirtualDom_diffNodes(x, y, patches, index, diffKids)
{
    // 如果节点的重要指标发生变化，则直接添加替换的 patch
    if (x.c !== y.c || x.f !== y.f)
    {
        _VirtualDom_pushPatch(patches, 0, index, y); // 添加替换的 patch
        return;
    }

    var factsDiff = _VirtualDom_diffFacts(x.d, y.d); // 比较节点的属性差异
    factsDiff && _VirtualDom_pushPatch(patches, 4, index, factsDiff); // 如果有差异，则添加属性差异的 patch

    diffKids(x, y, patches, index); // 比较子节点的差异
}

// 比较节点的属性差异
function _VirtualDom_diffFacts(x, y, category)
{
    var diff;

    // 查找属性的变化和移除
    for (var xKey in x) // 遍历节点 x 的属性
    {
        // 遍历对象 x 的属性
        if (xKey === 'a1' || xKey === 'a0' || xKey === 'a3' || xKey === 'a4')
        {
            // 如果属性为特定值，则调用 _VirtualDom_diffFacts 函数进行比较
            var subDiff = _VirtualDom_diffFacts(x[xKey], y[xKey] || {}, xKey);
            // 如果有差异，则将差异存入 diff 对象中
            if (subDiff)
            {
                diff = diff || {};
                diff[xKey] = subDiff;
            }
            // 继续下一次循环
            continue;
        }
    
        // 如果属性不在新对象中，则将其从旧对象中移除
        if (!(xKey in y))
        {
            diff = diff || {};
            // 根据不同情况设置不同的值
            diff[xKey] =
                !category
                    ? (typeof x[xKey] === 'string' ? '' : null)
                    :
                (category === 'a1')
                    ? ''
                    :
                (category === 'a0' || category === 'a3')
                    ? undefined
                    :
                { f: x[xKey].f, o: undefined };
            // 继续下一次循环
            continue;
        }
    
        // 获取属性值
        var xValue = x[xKey];
        var yValue = y[xKey];
    
        // 如果属性值相等，则继续下一次循环
        if (xValue === yValue && xKey !== 'value' && xKey !== 'checked'
            || category === 'a0' && _VirtualDom_equalEvents(xValue, yValue))
        {
            continue;
        }
    
        // 如果属性值不相等，则将新值存入 diff 对象中
        diff = diff || {};
        diff[xKey] = yValue;
    }
    
    // 遍历对象 y，将新属性存入 diff 对象中
    for (var yKey in y)
    {
        if (!(yKey in x))
        {
            diff = diff || {};
            diff[yKey] = y[yKey];
        }
    }
    
    // 返回 diff 对象
    return diff;
// DIFF KIDS

// 对比两个父节点的子节点，生成补丁并添加到 patches 中
function _VirtualDom_diffKids(xParent, yParent, patches, index)
{
    var xKids = xParent.e; // 获取 xParent 的子节点列表
    var yKids = yParent.e; // 获取 yParent 的子节点列表

    var xLen = xKids.length; // 获取 xKids 的长度
    var yLen = yKids.length; // 获取 yKids 的长度

    // FIGURE OUT IF THERE ARE INSERTS OR REMOVALS
    // 判断是否有需要插入或移除的子节点

    if (xLen > yLen)
    {
        _VirtualDom_pushPatch(patches, 6, index, {
            v: yLen,
            i: xLen - yLen
        }); // 如果 xLen 大于 yLen，将插入移除的信息添加到 patches 中
    }
    else if (xLen < yLen)
    {
        _VirtualDom_pushPatch(patches, 7, index, {
            v: xLen,
            e: yKids
        }); // 如果 xLen 小于 yLen，将插入的信息添加到 patches 中
    }

    // PAIRWISE DIFF EVERYTHING ELSE
    // 逐对对比其他子节点

    for (var minLen = xLen < yLen ? xLen : yLen, i = 0; i < minLen; i++)
    {
        var xKid = xKids[i]; // 获取 xKids 中的子节点
        _VirtualDom_diffHelp(xKid, yKids[i], patches, ++index); // 对比两个子节点并生成补丁
        index += xKid.b || 0; // 更新 index
    }
}

// KEYED DIFF

// 对比具有 key 的子节点
function _VirtualDom_diffKeyedKids(xParent, yParent, patches, rootIndex)
{
    var localPatches = []; // 本地补丁列表

    var changes = {}; // Dict String Entry
    var inserts = []; // Array { index : Int, entry : Entry }
    // type Entry = { tag : String, vnode : VNode, index : Int, data : _ }

    var xKids = xParent.e; // 获取 xParent 的子节点列表
    var yKids = yParent.e; // 获取 yParent 的子节点列表
    var xLen = xKids.length; // 获取 xKids 的长度
    var yLen = yKids.length; // 获取 yKids 的长度
    var xIndex = 0; // xKids 的索引
    var yIndex = 0; // yKids 的索引

    var index = rootIndex; // 根索引

    while (xIndex < xLen && yIndex < yLen)
    {
        // 循环对比子节点
    }

    // eat up any remaining nodes with removeNode and insertNode
    // 移除剩余的节点并插入新节点

    while (xIndex < xLen)
    {
        index++;
        var x = xKids[xIndex]; // 获取 xKids 中的子节点
        var xNode = x.b; // 获取 xNode
        _VirtualDom_removeNode(changes, localPatches, x.a, xNode, index); // 移除节点并生成补丁
        index += xNode.b || 0; // 更新 index
        xIndex++; // 更新 xIndex
    }

    while (yIndex < yLen)
    {
        var endInserts = endInserts || []; // 结尾插入的节点列表
        var y = yKids[yIndex]; // 获取 yKids 中的子节点
        _VirtualDom_insertNode(changes, localPatches, y.a, y.b, undefined, endInserts); // 插入节点并生成补丁
        yIndex++; // 更新 yIndex
    }

    if (localPatches.length > 0 || inserts.length > 0 || endInserts)
    {
        // 如果存在本地补丁、插入或结尾插入的节点
    }
}
    {
        # 调用_VirtualDom_pushPatch函数，传入参数patches, 8, rootIndex, 以及一个包含w, x, y字段的对象
        _VirtualDom_pushPatch(patches, 8, rootIndex, {
            w: localPatches,  # w字段表示本地修补
            x: inserts,  # x字段表示插入
            y: endInserts  # y字段表示结束插入
        });
    }
// CHANGES FROM KEYED DIFF

// 定义虚拟 DOM 节点的后缀
var _VirtualDom_POSTFIX = '_elmW6BL';

// 插入节点的函数，用于处理虚拟 DOM 节点的插入
function _VirtualDom_insertNode(changes, localPatches, key, vnode, yIndex, inserts)
{
    // 获取指定 key 对应的 entry
    var entry = changes[key];

    // 如果之前没有见过这个 key
    if (!entry)
    {
        // 创建一个新的 entry，并添加到 inserts 数组中
        entry = {
            c: 0,
            z: vnode,
            r: yIndex,
            s: undefined
        };
        inserts.push({ r: yIndex, A: entry });
        changes[key] = entry;
        return;
    }

    // 如果之前已经移除了这个 key，表示找到了匹配的节点
    if (entry.c === 1)
    {
        // 将 entry 添加到 inserts 数组中
        inserts.push({ r: yIndex, A: entry });

        // 更新 entry 的状态和子补丁
        entry.c = 2;
        var subPatches = [];
        _VirtualDom_diffHelp(entry.z, vnode, subPatches, entry.r);
        entry.r = yIndex;
        entry.s.s = {
            w: subPatches,
            A: entry
        };
        return;
    }

    // 如果这个 key 已经被插入或移动过，表示是重复的节点
    _VirtualDom_insertNode(changes, localPatches, key + _VirtualDom_POSTFIX, vnode, yIndex, inserts);
}

// 移除节点的函数，用于处理虚拟 DOM 节点的移除
function _VirtualDom_removeNode(changes, localPatches, key, vnode, index)
{
    // 获取指定 key 对应的 entry
    var entry = changes[key];

    // 如果之前没有见过这个 key
    if (!entry)
    {
        // 创建一个新的 patch，并将 entry 添加到 changes 中
        var patch = _VirtualDom_pushPatch(localPatches, 9, index, undefined);
        changes[key] = {
            c: 1,
            z: vnode,
            r: index,
            s: patch
        };
        return;
    }

    // 如果这个 key 已经被插入过，表示找到了匹配的节点
    if (entry.c === 0)
    {
        // 更新 entry 的状态和子补丁
        entry.c = 2;
        var subPatches = [];
        _VirtualDom_diffHelp(vnode, entry.z, subPatches, index);
        _VirtualDom_pushPatch(localPatches, 9, index, {
            w: subPatches,
            A: entry
        });
        return;
    }

    // 如果这个 key 已经被移除或移动过，表示是重复的节点
    _VirtualDom_removeNode(changes, localPatches, key + _VirtualDom_POSTFIX, vnode, index);
}

// ADD DOM NODES
//
// Each DOM node has an "index" assigned in order of traversal. It is important
// 每个 DOM 节点都有一个按遍历顺序分配的“index”。这很重要
// 为了最小化对实际 DOM 的遍历，这些索引（以及虚拟节点的后代计数）让我们可以跳过整个子树的 DOM，如果我们知道那里没有补丁。

// 将虚拟节点添加到 DOM 节点中
function _VirtualDom_addDomNodes(domNode, vNode, patches, eventNode)
{
    // 调用辅助函数，传入初始索引和范围
    _VirtualDom_addDomNodesHelp(domNode, vNode, patches, 0, 0, vNode.b, eventNode);
}

// 假设 `patches` 不为空且索引单调递增。
function _VirtualDom_addDomNodesHelp(domNode, vNode, patches, i, low, high, eventNode)
{
    // 获取当前索引处的补丁
    var patch = patches[i];
    var index = patch.r;

    // 循环处理索引等于低位索引的情况
    while (index === low)
    {
        var patchType = patch.$;

        // 根据补丁类型执行相应操作
        if (patchType === 1)
        {
            // 递归调用添加 DOM 节点
            _VirtualDom_addDomNodes(domNode, vNode.k, patch.s, eventNode);
        }
        else if (patchType === 8)
        {
            // 设置补丁的 DOM 节点和事件节点
            patch.t = domNode;
            patch.u = eventNode;

            // 获取子补丁并递归调用添加 DOM 节点的辅助函数
            var subPatches = patch.s.w;
            if (subPatches.length > 0)
            {
                _VirtualDom_addDomNodesHelp(domNode, vNode, subPatches, 0, low, high, eventNode);
            }
        }
        else if (patchType === 9)
        {
            // 设置补丁的 DOM 节点和事件节点
            patch.t = domNode;
            patch.u = eventNode;

            // 获取数据并设置相关属性
            var data = patch.s;
            if (data)
            {
                data.A.s = domNode;
                // 获取子补丁并递归调用添加 DOM 节点的辅助函数
                var subPatches = data.w;
                if (subPatches.length > 0)
                {
                    _VirtualDom_addDomNodesHelp(domNode, vNode, subPatches, 0, low, high, eventNode);
                }
            }
        }
        else
        {
            // 设置补丁的 DOM 节点和事件节点
            patch.t = domNode;
            patch.u = eventNode;
        }

        i++;

        // 如果没有下一个补丁或者下一个补丁的索引大于高位索引，则返回当前索引
        if (!(patch = patches[i]) || (index = patch.r) > high)
        {
            return i;
        }
    }

    // 获取虚拟节点的标签
    var tag = vNode.$;

    if (tag === 4)
    {
        // 获取子节点
        var subNode = vNode.k;

        // 循环直到找到非 4 类型的子节点
        while (subNode.$ === 4)
        {
            subNode = subNode.k;
        }

        // 调用 _VirtualDom_addDomNodesHelp 方法，处理子节点
        return _VirtualDom_addDomNodesHelp(domNode, subNode, patches, i, low + 1, high, domNode.elm_event_node_ref);
    }

    // 在这一点上，标签必须是 1 或 2
    // 获取虚拟节点的子节点
    var vKids = vNode.e;
    // 获取 DOM 节点的子节点
    var childNodes = domNode.childNodes;
    // 遍历虚拟节点的子节点
    for (var j = 0; j < vKids.length; j++)
    {
        low++;
        // 获取当前子节点
        var vKid = tag === 1 ? vKids[j] : vKids[j].b;
        // 计算下一个子节点的位置
        var nextLow = low + (vKid.b || 0);
        // 如果当前位置在索引范围内，则调用 _VirtualDom_addDomNodesHelp 方法处理子节点
        if (low <= index && index <= nextLow)
        {
            i = _VirtualDom_addDomNodesHelp(childNodes[j], vKid, patches, i, low, nextLow, eventNode);
            // 如果不存在 patch 或者索引超出范围，则返回当前索引
            if (!(patch = patches[i]) || (index = patch.r) > high)
            {
                return i;
            }
        }
        low = nextLow;
    }
    return i;
// 应用补丁到虚拟 DOM 树上的实际 DOM 节点
function _VirtualDom_applyPatches(rootDomNode, oldVirtualNode, patches, eventNode)
{
    // 如果补丁列表为空，直接返回根节点
    if (patches.length === 0)
    {
        return rootDomNode;
    }
    // 将实际 DOM 节点添加到根节点上
    _VirtualDom_addDomNodes(rootDomNode, oldVirtualNode, patches, eventNode);
    // 应用补丁并返回更新后的根节点
    return _VirtualDom_applyPatchesHelp(rootDomNode, patches);
}

// 辅助函数，应用补丁到实际 DOM 节点
function _VirtualDom_applyPatchesHelp(rootDomNode, patches)
{
    // 遍历补丁列表
    for (var i = 0; i < patches.length; i++)
    {
        var patch = patches[i];
        var localDomNode = patch.t
        // 应用单个补丁并获取新节点
        var newNode = _VirtualDom_applyPatch(localDomNode, patch);
        // 如果当前节点是根节点，更新根节点
        if (localDomNode === rootDomNode)
        {
            rootDomNode = newNode;
        }
    }
    // 返回更新后的根节点
    return rootDomNode;
}

// 应用单个补丁到实际 DOM 节点
function _VirtualDom_applyPatch(domNode, patch)
{
    // 根据补丁类型进行相应操作
    switch (patch.$)
    {
        // 对应不同的 case，执行不同的操作并返回结果
        case 0:
            // 应用重绘补丁
            return _VirtualDom_applyPatchRedraw(domNode, patch.s, patch.u);
    
        case 4:
            // 应用属性变化补丁
            _VirtualDom_applyFacts(domNode, patch.u, patch.s);
            return domNode;
    
        case 3:
            // 替换节点数据
            domNode.replaceData(0, domNode.length, patch.s);
            return domNode;
    
        case 1:
            // 应用子补丁
            return _VirtualDom_applyPatchesHelp(domNode, patch.s);
    
        case 2:
            // 更新事件节点引用
            if (domNode.elm_event_node_ref)
            {
                domNode.elm_event_node_ref.j = patch.s;
            }
            else
            {
                domNode.elm_event_node_ref = { j: patch.s, p: patch.u };
            }
            return domNode;
    
        case 6:
            // 移除子节点
            var data = patch.s;
            for (var i = 0; i < data.i; i++)
            {
                domNode.removeChild(domNode.childNodes[data.v]);
            }
            return domNode;
    
        case 7:
            // 插入子节点
            var data = patch.s;
            var kids = data.e;
            var i = data.v;
            var theEnd = domNode.childNodes[i];
            for (; i < kids.length; i++)
            {
                domNode.insertBefore(_VirtualDom_render(kids[i], patch.u), theEnd);
            }
            return domNode;
    
        case 9:
            // 处理节点删除
            var data = patch.s;
            if (!data)
            {
                domNode.parentNode.removeChild(domNode);
                return domNode;
            }
            var entry = data.A;
            if (typeof entry.r !== 'undefined')
            {
                domNode.parentNode.removeChild(domNode);
            }
            entry.s = _VirtualDom_applyPatchesHelp(domNode, data.w);
            return domNode;
    
        case 8:
            // 应用重排序补丁
            return _VirtualDom_applyPatchReorder(domNode, patch);
    
        case 5:
            // 执行自定义补丁
            return patch.s(domNode);
    
        default:
            _Debug_crash(10); // 'Ran into an unknown patch!'
    }
{
    // 将新的虚拟节点应用到真实 DOM 上，并返回新的节点
    function _VirtualDom_applyPatchRedraw(domNode, vNode, eventNode)
    {
        // 获取父节点
        var parentNode = domNode.parentNode;
        // 渲染新的虚拟节点
        var newNode = _VirtualDom_render(vNode, eventNode);

        // 如果新节点没有事件节点引用，则将其设置为与旧节点相同的事件节点引用
        if (!newNode.elm_event_node_ref)
        {
            newNode.elm_event_node_ref = domNode.elm_event_node_ref;
        }

        // 如果存在父节点且新节点与旧节点不同，则用新节点替换旧节点
        if (parentNode && newNode !== domNode)
        {
            parentNode.replaceChild(newNode, domNode);
        }
        // 返回新节点
        return newNode;
    }

    // 重新排序 DOM 节点
    function _VirtualDom_applyPatchReorder(domNode, patch)
    {
        var data = patch.s;

        // 移除末尾插入的节点
        var frag = _VirtualDom_applyPatchReorderEndInsertsHelp(data.y, patch);

        // 移除节点
        domNode = _VirtualDom_applyPatchesHelp(domNode, data.w);

        // 插入节点
        var inserts = data.x;
        for (var i = 0; i < inserts.length; i++)
        {
            var insert = inserts[i];
            var entry = insert.A;
            var node = entry.c === 2
                ? entry.s
                : _VirtualDom_render(entry.z, patch.u);
            domNode.insertBefore(node, domNode.childNodes[insert.r]);
        }

        // 添加末尾插入的节点
        if (frag)
        {
            _VirtualDom_appendChild(domNode, frag);
        }

        return domNode;
    }

    // 帮助函数：创建文档片段并插入末尾插入的节点
    function _VirtualDom_applyPatchReorderEndInsertsHelp(endInserts, patch)
    {
        if (!endInserts)
        {
            return;
        }

        var frag = _VirtualDom_doc.createDocumentFragment();
        for (var i = 0; i < endInserts.length; i++)
        {
            var insert = endInserts[i];
            var entry = insert.A;
            _VirtualDom_appendChild(frag, entry.c === 2
                ? entry.s
                : _VirtualDom_render(entry.z, patch.u)
            );
        }
        return frag;
    }

    // 将真实 DOM 节点转换为虚拟节点
    function _VirtualDom_virtualize(node)
    {
        // 文本节点
        if (node.nodeType === 3)
        {
            return _VirtualDom_text(node.textContent);
        }

        // 奇怪的节点
        if (node.nodeType !== 1)
        {
            return _VirtualDom_text('');
        }

        // 元素节点
        var attrList = _List_Nil;
        var attrs = node.attributes;
        for (var i = attrs.length; i--; )
    # 遍历节点的属性列表
    {
        # 获取当前属性
        var attr = attrs[i];
        # 获取属性名
        var name = attr.name;
        # 获取属性值
        var value = attr.value;
        # 将属性名和属性值转换成 VirtualDom 属性，并添加到属性列表中
        attrList = _List_Cons( A2(_VirtualDom_attribute, name, value), attrList );
    }

    # 获取节点的标签名，并转换成小写
    var tag = node.tagName.toLowerCase();
    # 初始化子节点列表
    var kidList = _List_Nil;
    # 获取节点的子节点列表
    var kids = node.childNodes;

    # 遍历子节点列表，将子节点转换成 VirtualDom 节点，并添加到子节点列表中
    for (var i = kids.length; i--; )
    {
        kidList = _List_Cons(_VirtualDom_virtualize(kids[i]), kidList);
    }
    # 返回 VirtualDom 节点
    return A3(_VirtualDom_node, tag, attrList, kidList);
// 从带有键的节点中移除键，返回一个新的节点对象
function _VirtualDom_dekey(keyedNode)
{
    // 获取带有键的子节点数组
    var keyedKids = keyedNode.e;
    // 获取子节点数组的长度
    var len = keyedKids.length;
    // 创建一个新的子节点数组
    var kids = new Array(len);
    // 遍历带有键的子节点数组，获取每个子节点的内容
    for (var i = 0; i < len; i++)
    {
        kids[i] = keyedKids[i].b;
    }

    // 返回一个新的节点对象，移除了键
    return {
        $: 1,
        c: keyedNode.c,
        d: keyedNode.d,
        e: kids,
        f: keyedNode.f,
        b: keyedNode.b
    };
}

// 创建一个元素节点
var _Debugger_element;

// 创建一个元素节点，如果存在调试器则使用调试器，否则使用 F4 函数
var _Browser_element = _Debugger_element || F4(function(impl, flagDecoder, debugMetadata, args)
{
    // 初始化平台
    return _Platform_initialize(
        flagDecoder,
        args,
        impl.aB,
        impl.aJ,
        impl.aH,
        function(sendToApp, initialModel) {
            // 获取视图函数
            var view = impl.aK;
            /**/
            // 获取节点元素
            var domNode = args['node'];
            //*/
            /**_UNUSED/
            // 如果存在节点参数，则获取节点元素，否则抛出调试错误
            var domNode = args && args['node'] ? args['node'] : _Debug_crash(0);
            //*/
            // 将节点元素转换为虚拟节点
            var currNode = _VirtualDom_virtualize(domNode);

            // 创建浏览器动画器
            return _Browser_makeAnimator(initialModel, function(model)
            {
                // 获取下一个节点
                var nextNode = view(model);
                // 获取节点差异
                var patches = _VirtualDom_diff(currNode, nextNode);
                // 应用节点差异
                domNode = _VirtualDom_applyPatches(domNode, currNode, patches, sendToApp);
                // 更新当前节点
                currNode = nextNode;
            });
        }
    );
});

// 创建一个文档节点
var _Debugger_document;

// 创建一个文档节点，如果存在调试器则使用调试器，否则使用 F4 函数
var _Browser_document = _Debugger_document || F4(function(impl, flagDecoder, debugMetadata, args)
{
    # 调用 _Platform_initialize 函数，传入参数 flagDecoder, args, impl.aB, impl.aJ, impl.aH, 以及一个回调函数
    return _Platform_initialize(
        flagDecoder,
        args,
        impl.aB,
        impl.aJ,
        impl.aH,
        function(sendToApp, initialModel) {
            # 如果实现了 P 函数，则将 sendToApp 传入 P 函数，返回结果赋值给 divertHrefToApp
            var divertHrefToApp = impl.P && impl.P(sendToApp)
            # 将 impl.aK 赋值给 view
            var view = impl.aK;
            # 获取当前文档的标题
            var title = _VirtualDom_doc.title;
            # 获取当前文档的 body 节点
            var bodyNode = _VirtualDom_doc.body;
            # 将 bodyNode 虚拟化，赋值给 currNode
            var currNode = _VirtualDom_virtualize(bodyNode);
            # 调用 _Browser_makeAnimator 函数，传入 initialModel 和一个回调函数
            return _Browser_makeAnimator(initialModel, function(model)
            {
                # 将 divertHrefToApp 赋值给 _VirtualDom_divertHrefToApp
                _VirtualDom_divertHrefToApp = divertHrefToApp;
                # 调用 view 函数，传入 model，赋值给 doc
                var doc = view(model);
                # 创建一个新的 body 节点，赋值给 nextNode
                var nextNode = _VirtualDom_node('body')(_List_Nil)(doc.au);
                # 计算 currNode 和 nextNode 之间的差异，赋值给 patches
                var patches = _VirtualDom_diff(currNode, nextNode);
                # 将 patches 应用到 bodyNode 上，赋值给 bodyNode
                bodyNode = _VirtualDom_applyPatches(bodyNode, currNode, patches, sendToApp);
                # 将 currNode 更新为 nextNode
                currNode = nextNode;
                # 将 _VirtualDom_divertHrefToApp 置为 0
                _VirtualDom_divertHrefToApp = 0;
                # 如果标题发生变化，则更新文档的标题
                (title !== doc.aI) && (_VirtualDom_doc.title = title = doc.aI);
            });
        }
    );
// 取消动画帧的执行
var _Browser_cancelAnimationFrame =
    typeof cancelAnimationFrame !== 'undefined'
        ? cancelAnimationFrame
        : function(id) { clearTimeout(id); };

// 请求执行动画帧
var _Browser_requestAnimationFrame =
    typeof requestAnimationFrame !== 'undefined'
        ? requestAnimationFrame
        : function(callback) { return setTimeout(callback, 1000 / 60); };

// 创建动画函数
function _Browser_makeAnimator(model, draw)
{
    // 绘制模型
    draw(model);

    // 状态变量
    var state = 0;

    // 更新动画帧
    function updateIfNeeded()
    {
        state = state === 1
            ? 0
            : ( _Browser_requestAnimationFrame(updateIfNeeded), draw(model), 1 );
    }

    // 返回动画函数
    return function(nextModel, isSync)
    {
        // 更新模型
        model = nextModel;

        // 如果是同步更新
        isSync
            ? ( draw(model),
                state === 2 && (state = 1)
                )
            // 如果是异步更新
            : ( state === 0 && _Browser_requestAnimationFrame(updateIfNeeded),
                state = 2
                );
    };
}

// 应用程序
function _Browser_application(impl)
{
    // URL变化处理函数
    var onUrlChange = impl.aD;
    // URL请求处理函数
    var onUrlRequest = impl.aE;
    // 键盘事件处理函数
    var key = function() { key.a(onUrlChange(_Browser_getUrl())); };
}
    # 返回一个包含多个函数的对象，用于处理浏览器文档相关的操作
    return _Browser_document({
        # 定义 P 函数，用于处理发送到应用的操作
        P: function(sendToApp)
        {
            # 将 sendToApp 存储在 key.a 中
            key.a = sendToApp;
            # 添加 popstate 事件监听器，当浏览历史记录发生变化时触发 key 函数
            _Browser_window.addEventListener('popstate', key);
            # 如果浏览器不是 Trident 内核，添加 hashchange 事件监听器，当 URL 的片段标识符发生变化时触发 key 函数
            _Browser_window.navigator.userAgent.indexOf('Trident') < 0 || _Browser_window.addEventListener('hashchange', key);
    
            # 返回一个函数，用于处理 DOM 节点和事件
            return F2(function(domNode, event)
            {
                # 如果满足一系列条件，则阻止默认事件并发送 URL 请求到应用
                if (!event.ctrlKey && !event.metaKey && !event.shiftKey && event.button < 1 && !domNode.target && !domNode.hasAttribute('download'))
                {
                    event.preventDefault();
                    # 获取链接的 href 属性
                    var href = domNode.href;
                    # 获取当前 URL 和下一个 URL
                    var curr = _Browser_getUrl();
                    var next = $elm$url$Url$fromString(href).a;
                    # 发送 URL 请求到应用
                    sendToApp(onUrlRequest(
                        (next
                            && curr.ah === next.ah
                            && curr.Z === next.Z
                            && curr.ae.a === next.ae.a
                        )
                            ? $elm$browser$Browser$Internal(next)
                            : $elm$browser$Browser$External(href)
                    ));
                }
            });
        },
        # 定义 aB 函数，用于处理应用的标志
        aB: function(flags)
        {
            # 调用 impl 模块的 aB 函数，并传入标志、当前 URL 和 key 对象
            return A3(impl.aB, flags, _Browser_getUrl(), key);
        },
        # 将 impl 模块的 aK 函数赋值给 aK
        aK: impl.aK,
        # 将 impl 模块的 aJ 函数赋值给 aJ
        aJ: impl.aJ,
        # 将 impl 模块的 aH 函数赋值给 aH
        aH: impl.aH
    });
// 获取当前页面的 URL
function _Browser_getUrl()
{
    return $elm$url$Url$fromString(_VirtualDom_doc.location.href).a || _Debug_crash(1);
}

// 在浏览器历史记录中前进或后退
var _Browser_go = F2(function(key, n)
{
    return A2($elm$core$Task$perform, $elm$core$Basics$never, _Scheduler_binding(function() {
        n && history.go(n);
        key();
    }));
});

// 在浏览器历史记录中添加新的 URL 记录
var _Browser_pushUrl = F2(function(key, url)
{
    return A2($elm$core$Task$perform, $elm$core$Basics$never, _Scheduler_binding(function() {
        history.pushState({}, '', url);
        key();
    }));
});

// 在浏览器历史记录中替换当前的 URL 记录
var _Browser_replaceUrl = F2(function(key, url)
{
    return A2($elm$core$Task$perform, $elm$core$Basics$never, _Scheduler_binding(function() {
        history.replaceState({}, '', url);
        key();
    }));
});

// 全局事件

// 创建一个虚拟的节点对象，用于模拟事件监听和移除事件监听
var _Browser_fakeNode = { addEventListener: function() {}, removeEventListener: function() {} };
// 获取文档对象，如果浏览器环境中存在 document 对象则使用，否则使用虚拟节点对象
var _Browser_doc = typeof document !== 'undefined' ? document : _Browser_fakeNode;
// 获取窗口对象，如果浏览器环境中存在 window 对象则使用，否则使用虚拟节点对象
var _Browser_window = typeof window !== 'undefined' ? window : _Browser_fakeNode;

// 监听节点上的事件，并将事件发送给自身
var _Browser_on = F3(function(node, eventName, sendToSelf)
{
    return _Scheduler_spawn(_Scheduler_binding(function(callback)
    {
        function handler(event)    { _Scheduler_rawSpawn(sendToSelf(event)); }
        node.addEventListener(eventName, handler, _VirtualDom_passiveSupported && { passive: true });
        return function() { node.removeEventListener(eventName, handler); };
    }));
});

// 解码事件，使用给定的解码器对事件进行解码
var _Browser_decodeEvent = F2(function(decoder, event)
{
    var result = _Json_runHelp(decoder, event);
    return $elm$core$Result$isOk(result) ? $elm$core$Maybe$Just(result.a) : $elm$core$Maybe$Nothing;
});

// 页面可见性

// 获取页面可见性信息
function _Browser_visibilityInfo()
{
    return (typeof _VirtualDom_doc.hidden !== 'undefined')
        ? { az: 'hidden', av: 'visibilitychange' }
        :
    (typeof _VirtualDom_doc.mozHidden !== 'undefined')
        ? { az: 'mozHidden', av: 'mozvisibilitychange' }
        :
    # 检查是否存在 msHidden 属性，如果存在则返回 { az: 'msHidden', av: 'msvisibilitychange' }
    (typeof _VirtualDom_doc.msHidden !== 'undefined')
        ? { az: 'msHidden', av: 'msvisibilitychange' }
        :
    # 如果不存在 msHidden 属性，检查是否存在 webkitHidden 属性，如果存在则返回 { az: 'webkitHidden', av: 'webkitvisibilitychange' }
    (typeof _VirtualDom_doc.webkitHidden !== 'undefined')
        ? { az: 'webkitHidden', av: 'webkitvisibilitychange' }
    # 如果都不存在，则返回 { az: 'hidden', av: 'visibilitychange' }
        : { az: 'hidden', av: 'visibilitychange' };
// ANIMATION FRAMES

// 定义一个函数，返回一个绑定了 requestAnimationFrame 的调度器函数
function _Browser_rAF()
{
    return _Scheduler_binding(function(callback)
    {
        // 使用 requestAnimationFrame 注册一个回调函数，当动画帧可用时执行
        var id = _Browser_requestAnimationFrame(function() {
            // 执行回调函数，传入当前时间
            callback(_Scheduler_succeed(Date.now()));
        });

        // 返回一个函数，用于取消 requestAnimationFrame
        return function() {
            _Browser_cancelAnimationFrame(id);
        };
    });
}

// 返回一个绑定了 Date.now() 的调度器函数
function _Browser_now()
{
    return _Scheduler_binding(function(callback)
    {
        // 执行回调函数，传入当前时间
        callback(_Scheduler_succeed(Date.now()));
    });
}

// DOM STUFF

// 返回一个绑定了 document.getElementById 的调度器函数
function _Browser_withNode(id, doStuff)
{
    return _Scheduler_binding(function(callback)
    {
        // 使用 requestAnimationFrame 注册一个回调函数，当动画帧可用时执行
        _Browser_requestAnimationFrame(function() {
            // 获取指定 id 的 DOM 节点
            var node = document.getElementById(id);
            // 如果节点存在，则执行传入的 doStuff 函数并传入节点，否则返回未找到节点的错误
            callback(node
                ? _Scheduler_succeed(doStuff(node))
                : _Scheduler_fail($elm$browser$Browser$Dom$NotFound(id))
            );
        });
    });
}

// 返回一个绑定了 doStuff 函数的调度器函数
function _Browser_withWindow(doStuff)
{
    return _Scheduler_binding(function(callback)
    {
        // 使用 requestAnimationFrame 注册一个回调函数，当动画帧可用时执行
        _Browser_requestAnimationFrame(function() {
            // 执行传入的 doStuff 函数
            callback(_Scheduler_succeed(doStuff()));
        });
    });
}

// FOCUS and BLUR

// 定义一个函数，接受函数名和 id，返回一个绑定了节点的调度器函数
var _Browser_call = F2(function(functionName, id)
{
    return _Browser_withNode(id, function(node) {
        // 调用节点的指定函数
        node[functionName]();
        return _Utils_Tuple0;
    });
});

// WINDOW VIEWPORT

// 返回当前窗口的视口信息
function _Browser_getViewport()
{
    return {
        // 获取窗口的场景信息
        al: _Browser_getScene(),
        // 获取窗口的视口信息
        ao: {
            aq: _Browser_window.pageXOffset,
            ar: _Browser_window.pageYOffset,
            ap: _Browser_doc.documentElement.clientWidth,
            Y: _Browser_doc.documentElement.clientHeight
        }
    };
}

// 获取窗口的场景信息
function _Browser_getScene()
{
    var body = _Browser_doc.body;
    var elem = _Browser_doc.documentElement;
    return {
        // 获取页面的宽度
        ap: Math.max(body.scrollWidth, body.offsetWidth, elem.scrollWidth, elem.offsetWidth, elem.clientWidth),
        // 获取页面的高度
        Y: Math.max(body.scrollHeight, body.offsetHeight, elem.scrollHeight, elem.offsetHeight, elem.clientHeight)
    };

这是一个代码块的结束标记，表示一个函数或者一个条件语句的结束。
}



// 定义一个名为_Browser_setViewport的函数，接受两个参数x和y
var _Browser_setViewport = F2(function(x, y)
{
    // 调用_Browser_withWindow函数，在窗口中设置滚动条的位置为(x, y)
    return _Browser_withWindow(function()
    {
        _Browser_window.scroll(x, y);
        // 返回空元组
        return _Utils_Tuple0;
    });
});



// ELEMENT VIEWPORT

// 定义一个名为_Browser_getViewportOf的函数，接受一个参数id
function _Browser_getViewportOf(id)
{
    // 调用_Browser_withNode函数，获取指定id节点的视口信息
    return _Browser_withNode(id, function(node)
    {
        // 返回包含节点宽度、高度、滚动条位置等信息的对象
        return {
            al: {
                ap: node.scrollWidth,
                Y: node.scrollHeight
            },
            ao: {
                aq: node.scrollLeft,
                ar: node.scrollTop,
                ap: node.clientWidth,
                Y: node.clientHeight
            }
        };
    });
}



// 定义一个名为_Browser_setViewportOf的函数，接受三个参数id、x和y
var _Browser_setViewportOf = F3(function(id, x, y)
{
    // 调用_Browser_withNode函数，设置指定id节点的滚动条位置为(x, y)
    return _Browser_withNode(id, function(node)
    {
        node.scrollLeft = x;
        node.scrollTop = y;
        // 返回空元组
        return _Utils_Tuple0;
    });
});



// ELEMENT

// 定义一个名为_Browser_getElement的函数，接受一个参数id
function _Browser_getElement(id)
{
    // 调用_Browser_withNode函数，获取指定id节点的位置和大小信息
    return _Browser_withNode(id, function(node)
    {
        // 获取节点相对于视口的位置和大小信息，并返回包含这些信息的对象
        var rect = node.getBoundingClientRect();
        var x = _Browser_window.pageXOffset;
        var y = _Browser_window.pageYOffset;
        return {
            al: _Browser_getScene(),
            ao: {
                aq: x,
                ar: y,
                ap: _Browser_doc.documentElement.clientWidth,
                Y: _Browser_doc.documentElement.clientHeight
            },
            ax: {
                aq: x + rect.left,
                ar: y + rect.top,
                ap: rect.width,
                Y: rect.height
            }
        };
    });
}



// LOAD and RELOAD

// 定义一个名为_Browser_reload的函数，接受一个参数skipCache
function _Browser_reload(skipCache)
{
    // 执行一个任务，调用浏览器的location.reload方法来重新加载页面
    return A2($elm$core$Task$perform, $elm$core$Basics$never, _Scheduler_binding(function(callback)
    {
        _VirtualDom_doc.location.reload(skipCache);
    }));
}

// 定义一个名为_Browser_load的函数，接受一个参数url
function _Browser_load(url)
{
    // 执行一个任务，调用浏览器的location.href方法来加载指定url的页面
    return A2($elm$core$Task$perform, $elm$core$Basics$never, _Scheduler_binding(function(callback)
    {
        // 尝试在浏览器窗口中加载给定的 URL
        try
        {
            _Browser_window.location = url;
        }
        // 捕获可能的异常
        catch(err)
        {
            // 只有 Firefox 可能在这里抛出 NS_ERROR_MALFORMED_URI 异常
            // 其他浏览器会重新加载页面，因此让我们在这方面保持一致
            _VirtualDom_doc.location.reload(false);
        }
    }
# 定义位运算函数，计算两个数的按位与
var _Bitwise_and = F2(function(a, b)
{
    return a & b;
});

# 定义位运算函数，计算两个数的按位或
var _Bitwise_or = F2(function(a, b)
{
    return a | b;
});

# 定义位运算函数，计算两个数的按位异或
var _Bitwise_xor = F2(function(a, b)
{
    return a ^ b;
});

# 定义位运算函数，计算一个数的按位取反
function _Bitwise_complement(a)
{
    return ~a;
};

# 定义位运算函数，将一个数左移指定位数
var _Bitwise_shiftLeftBy = F2(function(offset, a)
{
    return a << offset;
});

# 定义位运算函数，将一个数右移指定位数
var _Bitwise_shiftRightBy = F2(function(offset, a)
{
    return a >> offset;
});

# 定义位运算函数，将一个数无符号右移指定位数
var _Bitwise_shiftRightZfBy = F2(function(offset, a)
{
    return a >>> offset;
});

# 定义获取当前时间的函数，将毫秒数转换为 POSIX 时间
function _Time_now(millisToPosix)
{
    return _Scheduler_binding(function(callback)
    {
        callback(_Scheduler_succeed(millisToPosix(Date.now())));
    });
}

# 定义设置定时器的函数，执行指定任务并返回清除定时器的函数
var _Time_setInterval = F2(function(interval, task)
{
    return _Scheduler_binding(function(callback)
    {
        var id = setInterval(function() { _Scheduler_rawSpawn(task); }, interval);
        return function() { clearInterval(id); };
    });
});

# 定义获取当前时区的函数
function _Time_here()
{
    return _Scheduler_binding(function(callback)
    {
        callback(_Scheduler_succeed(
            A2($elm$time$Time$customZone, -(new Date().getTimezoneOffset()), _List_Nil)
        ));
    });
}

# 定义获取时区名称的函数
function _Time_getZoneName()
{
    return _Scheduler_binding(function(callback)
    {
        try
        {
            var name = $elm$time$Time$Name(Intl.DateTimeFormat().resolvedOptions().timeZone);
        }
        catch (e)
        {
            var name = $elm$time$Time$Offset(new Date().getTimezoneOffset());
        }
        callback(_Scheduler_succeed(name));
    });
}

# 定义常量，表示相等关系
var $elm$core$Basics$EQ = 1;
# 定义常量，表示大于关系
var $elm$core$Basics$GT = 2;
# 定义常量，表示小于关系
var $elm$core$Basics$LT = 0;
# 定义列表操作函数，将元素添加到列表头部
var $elm$core$List$cons = _List_cons;
# 定义字典操作函数，从右向左遍历字典并执行指定函数
var $elm$core$Dict$foldr = F3(
    # 定义一个匿名函数，接受三个参数：func（函数）、acc（累加器）、t（字典）
    function (func, acc, t) {
        # 定义一个无限循环，用于遍历字典
        foldr:
        while (true) {
            # 如果字典为空，则返回累加器的值
            if (t.$ === -2) {
                return acc;
            } else {
                # 否则，获取字典中的键、值、左子树和右子树
                var key = t.b;
                var value = t.c;
                var left = t.d;
                var right = t.e;
                # 保存当前函数、累加器和左子树，然后递归处理右子树
                var $temp$func = func,
                    $temp$acc = A3(
                    func,
                    key,
                    value,
                    A3($elm$core$Dict$foldr, func, acc, right)),
                    $temp$t = left;
                func = $temp$func;
                acc = $temp$acc;
                t = $temp$t;
                # 继续执行循环
                continue foldr;
            }
        }
    });
# 将字典转换为列表，每个元素是键值对的元组
var $elm$core$Dict$toList = function (dict) {
    return A3(
        $elm$core$Dict$foldr,  # 使用 foldr 函数对字典进行折叠操作
        F3(
            function (key, value, list) {  # 对每个键值对执行函数，将其转换为元组并添加到列表中
                return A2(
                    $elm$core$List$cons,  # 将元组添加到列表的头部
                    _Utils_Tuple2(key, value),  # 创建键值对的元组
                    list);
            }),
        _List_Nil,  # 初始为空列表
        dict);
};

# 获取字典中的所有键，返回一个列表
var $elm$core$Dict$keys = function (dict) {
    return A3(
        $elm$core$Dict$foldr,  # 使用 foldr 函数对字典进行折叠操作
        F3(
            function (key, value, keyList) {  # 对每个键值对执行函数，将键添加到列表中
                return A2($elm$core$List$cons, key, keyList);  # 将键添加到列表的头部
            }),
        _List_Nil,  # 初始为空列表
        dict);
};

# 将集合转换为列表，只保留集合中的键
var $elm$core$Set$toList = function (_v0) {
    var dict = _v0;  # 将集合转换为字典
    return $elm$core$Dict$keys(dict);  # 调用 keys 函数获取字典中的所有键，返回一个列表
};

# 对数组进行右折叠操作
var $elm$core$Array$foldr = F3(
    function (func, baseCase, _v0) {
        var tree = _v0.c;  # 获取数组的树结构
        var tail = _v0.d;  # 获取数组的尾部
        var helper = F2(
            function (node, acc) {  # 定义辅助函数
                if (!node.$) {  # 如果节点不是叶子节点
                    var subTree = node.a;  # 获取子树
                    return A3($elm$core$Elm$JsArray$foldr, helper, acc, subTree);  # 对子树进行右折叠操作
                } else {  # 如果节点是叶子节点
                    var values = node.a;  # 获取节点的值
                    return A3($elm$core$Elm$JsArray$foldr, func, acc, values);  # 对节点的值进行右折叠操作
                }
            });
        return A3(
            $elm$core$Elm$JsArray$foldr,  # 对数组的树结构进行右折叠操作
            helper,  # 使用辅助函数进行折叠
            A3($elm$core$Elm$JsArray$foldr, func, baseCase, tail),  # 对数组的尾部进行右折叠操作
            tree);
    });

# 将数组转换为列表
var $elm$core$Array$toList = function (array) {
    return A3($elm$core$Array$foldr, $elm$core$List$cons, _List_Nil, array);  # 调用 foldr 函数对数组进行右折叠操作，将元素添加到列表中
};

# 表示一个失败的结果
var $elm$core$Result$Err = function (a) {
    return {$: 1, a: a};  # 返回一个包含错误信息的结果对象
};

# 表示一个 JSON 解码失败的结果
var $elm$json$Json$Decode$Failure = F2(
    function (a, b) {
        return {$: 3, a: a, b: b};  # 返回一个包含错误信息的 JSON 解码失败结果对象
    });

# 表示一个 JSON 字段
var $elm$json$Json$Decode$Field = F2(
    function (a, b) {
        return {$: 0, a: a, b: b};  # 返回一个包含字段名和字段值的 JSON 字段对象
    });

# 表示一个 JSON 索引
var $elm$json$Json$Decode$Index = F2(
    function (a, b) {
        return {$: 1, a: a, b: b};  # 返回一个包含索引和值的 JSON 索引对象
    });

# 表示一个成功的结果
var $elm$core$Result$Ok = function (a) {
    # 返回一个包含键值对的对象，其中 $ 的值为 0，a 的值为 a
    return {$: 0, a: a};
// 定义一个函数，用于创建一个包含多个选项的解码器
var $elm$json$Json$Decode$OneOf = function (a) {
    return {$: 2, a: a};
};
// 定义布尔类型的值 False
var $elm$core$Basics$False = 1;
// 定义加法函数
var $elm$core$Basics$add = _Basics_add;
// 定义一个函数，用于创建一个包含值的 Maybe 类型的 Just 构造器
var $elm$core$Maybe$Just = function (a) {
    return {$: 0, a: a};
};
// 定义一个不包含值的 Maybe 类型的 Nothing 构造器
var $elm$core$Maybe$Nothing = {$: 1};
// 判断字符串中的所有字符是否满足指定条件
var $elm$core$String$all = _String_all;
// 逻辑与运算
var $elm$core$Basics$and = _Basics_and;
// 字符串拼接
var $elm$core$Basics$append = _Utils_append;
// 将 JSON 值编码为字符串
var $elm$json$Json$Encode$encode = _Json_encode;
// 将整数转换为字符串
var $elm$core$String$fromInt = _String_fromNumber;
// 将字符串列表用指定分隔符连接起来
var $elm$core$String$join = F2(
    function (sep, chunks) {
        return A2(
            _String_join,
            sep,
            _List_toArray(chunks));
    });
// 将字符串根据指定分隔符拆分为字符串列表
var $elm$core$String$split = F2(
    function (sep, string) {
        return _List_fromArray(
            A2(_String_split, sep, string));
    });
// 将字符串按行缩进
var $elm$json$Json$Decode$indent = function (str) {
    return A2(
        $elm$core$String$join,
        '\n    ',
        A2($elm$core$String$split, '\n', str));
};
// 对列表进行左折叠
var $elm$core$List$foldl = F3(
    function (func, acc, list) {
        foldl:
        while (true) {
            if (!list.b) {
                return acc;
            } else {
                var x = list.a;
                var xs = list.b;
                var $temp$func = func,
                    $temp$acc = A2(func, x, acc),
                    $temp$list = xs;
                func = $temp$func;
                acc = $temp$acc;
                list = $temp$list;
                continue foldl;
            }
        }
    });
// 获取列表的长度
var $elm$core$List$length = function (xs) {
    return A3(
        $elm$core$List$foldl,
        F2(
            function (_v0, i) {
                return i + 1;
            }),
        0,
        xs);
};
// 将两个列表的元素一一对应应用函数，并返回结果列表
var $elm$core$List$map2 = _List_map2;
// 判断一个值是否小于等于另一个值
var $elm$core$Basics$le = _Utils_le;
// 计算两个数的差
var $elm$core$Basics$sub = _Basics_sub;
// 辅助函数，用于生成一个范围内的整数列表
var $elm$core$List$rangeHelp = F3(
    # 定义一个匿名函数，接受三个参数：lo, hi, list
    function (lo, hi, list) {
        # 定义一个标签，用于循环跳转
        rangeHelp:
        # 无限循环
        while (true) {
            # 如果 lo 小于等于 hi
            if (_Utils_cmp(lo, hi) < 1) {
                # 临时保存 lo, hi-1, 和在 list 前添加 hi 后的结果
                var $temp$lo = lo,
                    $temp$hi = hi - 1,
                    $temp$list = A2($elm$core$List$cons, hi, list);
                # 更新 lo, hi, list 的值
                lo = $temp$lo;
                hi = $temp$hi;
                list = $temp$list;
                # 跳转到 rangeHelp 标签处，继续循环
                continue rangeHelp;
            } else {
                # 如果 lo 大于 hi，则返回 list
                return list;
            }
        }
    });
# 定义一个函数，用于生成一个范围内的整数列表
var $elm$core$List$range = F2(
    function (lo, hi) {
        return A3($elm$core$List$rangeHelp, lo, hi, _List_Nil);
    });

# 定义一个函数，用于对列表中的元素进行索引映射
var $elm$core$List$indexedMap = F2(
    function (f, xs) {
        return A3(
            $elm$core$List$map2,
            f,
            A2(
                $elm$core$List$range,
                0,
                $elm$core$List$length(xs) - 1),
            xs);
    });

# 定义一个函数，用于将字符转换为其 Unicode 编码
var $elm$core$Char$toCode = _Char_toCode;

# 定义一个函数，用于判断字符是否为小写字母
var $elm$core$Char$isLower = function (_char) {
    var code = $elm$core$Char$toCode(_char);
    return (97 <= code) && (code <= 122);
};

# 定义一个函数，用于判断字符是否为大写字母
var $elm$core$Char$isUpper = function (_char) {
    var code = $elm$core$Char$toCode(_char);
    return (code <= 90) && (65 <= code);
};

# 定义一个函数，用于判断字符是否为字母
var $elm$core$Char$isAlpha = function (_char) {
    return $elm$core$Char$isLower(_char) || $elm$core$Char$isUpper(_char);
};

# 定义一个函数，用于判断字符是否为数字
var $elm$core$Char$isDigit = function (_char) {
    var code = $elm$core$Char$toCode(_char);
    return (code <= 57) && (48 <= code);
};

# 定义一个函数，用于判断字符是否为字母或数字
var $elm$core$Char$isAlphaNum = function (_char) {
    return $elm$core$Char$isLower(_char) || ($elm$core$Char$isUpper(_char) || $elm$core$Char$isDigit(_char));
};

# 定义一个函数，用于将列表反转
var $elm$core$List$reverse = function (list) {
    return A3($elm$core$List$foldl, $elm$core$List$cons, _List_Nil, list);
};

# 定义一个函数，用于从字符串中取出第一个字符
var $elm$core$String$uncons = _String_uncons;

# 定义一个函数，用于生成一个错误消息，表示多个可能的错误中的一个
var $elm$json$Json$Decode$errorOneOf = F2(
    function (i, error) {
        return '\n\n(' + ($elm$core$String$fromInt(i + 1) + (') ' + $elm$json$Json$Decode$indent(
            $elm$json$Json$Decode$errorToString(error))));
    });

# 定义一个函数，用于将错误转换为字符串
var $elm$json$Json$Decode$errorToString = function (error) {
    return A2($elm$json$Json$Decode$errorToStringHelp, error, _List_Nil);
};

# 定义一个函数，用于将错误转换为字符串的辅助函数
var $elm$json$Json$Decode$errorToStringHelp = F2(
    });

# 定义一个常量，表示数组的分支因子
var $elm$core$Array$branchFactor = 32;

# 定义一个函数，用于创建一个 Elm 内置数组
var $elm$core$Array$Array_elm_builtin = F4(
    function (a, b, c, d) {
        return {$: 0, a: a, b: b, c: c, d: d};
    });

# 定义一个常量，表示一个空的 JavaScript 数组
var $elm$core$Elm$JsArray$empty = _JsArray_empty;
# 定义向上取整函数
var $elm$core$Basics$ceiling = _Basics_ceiling;
# 定义浮点数除法函数
var $elm$core$Basics$fdiv = _Basics_fdiv;
# 定义对数函数，参数为底数和数字
var $elm$core$Basics$logBase = F2(
    function (base, number) {
        return _Basics_log(number) / _Basics_log(base);
    });
# 将参数转换为浮点数
var $elm$core$Basics$toFloat = _Basics_toFloat;
# 计算数组的分支因子
var $elm$core$Array$shiftStep = $elm$core$Basics$ceiling(
    A2($elm$core$Basics$logBase, 2, $elm$core$Array$branchFactor));
# 创建一个空数组
var $elm$core$Array$empty = A4($elm$core$Array$Array_elm_builtin, 0, $elm$core$Array$shiftStep, $elm$core$Elm$JsArray$empty, $elm$core$Elm$JsArray$empty);
# 从给定的初始化函数创建一个数组
var $elm$core$Elm$JsArray$initialize = _JsArray_initialize;
# 定义叶子节点构造函数
var $elm$core$Array$Leaf = function (a) {
    return {$: 1, a: a};
};
# 定义左侧函数
var $elm$core$Basics$apL = F2(
    function (f, x) {
        return f(x);
    });
# 定义右侧函数
var $elm$core$Basics$apR = F2(
    function (x, f) {
        return f(x);
    });
# 定义相等函数
var $elm$core$Basics$eq = _Utils_equal;
# 定义向下取整函数
var $elm$core$Basics$floor = _Basics_floor;
# 获取数组长度
var $elm$core$Elm$JsArray$length = _JsArray_length;
# 判断是否大于函数
var $elm$core$Basics$gt = _Utils_gt;
# 返回两个数中的较大值
var $elm$core$Basics$max = F2(
    function (x, y) {
        return (_Utils_cmp(x, y) > 0) ? x : y;
    });
# 乘法函数
var $elm$core$Basics$mul = _Basics_mul;
# 定义子树节点构造函数
var $elm$core$Array$SubTree = function (a) {
    return {$: 0, a: a};
};
# 从给定的列表创建一个数组
var $elm$core$Elm$JsArray$initializeFromList = _JsArray_initializeFromList;
# 压缩节点的函数
var $elm$core$Array$compressNodes = F2(
    # 定义一个匿名函数，接受 nodes 和 acc 两个参数
    function (nodes, acc) {
        # 创建一个标签，用于循环的跳转
        compressNodes:
        # 无限循环
        while (true) {
            # 从 nodes 中取出前 $elm$core$Array$branchFactor 个元素，创建一个新的数组
            var _v0 = A2($elm$core$Elm$JsArray$initializeFromList, $elm$core$Array$branchFactor, nodes);
            # 取出新数组的第一个元素作为 node，剩余的元素作为 remainingNodes
            var node = _v0.a;
            var remainingNodes = _v0.b;
            # 将 node 添加到 acc 中，创建一个新的列表 newAcc
            var newAcc = A2(
                $elm$core$List$cons,
                $elm$core$Array$SubTree(node),
                acc);
            # 如果 remainingNodes 为空，则返回 newAcc 的逆序列表
            if (!remainingNodes.b) {
                return $elm$core$List$reverse(newAcc);
            } else {
                # 否则，更新 nodes 和 acc 的值，继续循环
                var $temp$nodes = remainingNodes,
                    $temp$acc = newAcc;
                nodes = $temp$nodes;
                acc = $temp$acc;
                continue compressNodes;
            }
        }
    });
# 定义一个函数，用于从元组中获取第一个元素
var $elm$core$Tuple$first = function (_v0) {
    var x = _v0.a;
    return x;
};
# 定义一个函数，用于从构建器中创建树形数组
var $elm$core$Array$treeFromBuilder = F2(
    function (nodeList, nodeListSize) {
        treeFromBuilder:
        while (true) {
            # 计算新节点的大小
            var newNodeSize = $elm$core$Basics$ceiling(nodeListSize / $elm$core$Array$branchFactor);
            # 如果新节点大小为1，则从节点列表中初始化数组并返回
            if (newNodeSize === 1) {
                return A2($elm$core$Elm$JsArray$initializeFromList, $elm$core$Array$branchFactor, nodeList).a;
            } else {
                # 否则，压缩节点列表并更新节点列表和节点列表大小，继续循环
                var $temp$nodeList = A2($elm$core$Array$compressNodes, nodeList, _List_Nil),
                    $temp$nodeListSize = newNodeSize;
                nodeList = $temp$nodeList;
                nodeListSize = $temp$nodeListSize;
                continue treeFromBuilder;
            }
        }
    });
# 定义一个函数，用于将构建器转换为数组
var $elm$core$Array$builderToArray = F2(
    function (reverseNodeList, builder) {
        # 如果构建器为空，则直接返回空数组
        if (!builder.b) {
            return A4(
                $elm$core$Array$Array_elm_builtin,
                $elm$core$Elm$JsArray$length(builder.e),
                $elm$core$Array$shiftStep,
                $elm$core$Elm$JsArray$empty,
                builder.e);
        } else {
            # 否则，计算树的长度和深度，根据是否需要反转节点列表来创建树，并返回数组
            var treeLen = builder.b * $elm$core$Array$branchFactor;
            var depth = $elm$core$Basics$floor(
                A2($elm$core$Basics$logBase, $elm$core$Array$branchFactor, treeLen - 1));
            var correctNodeList = reverseNodeList ? $elm$core$List$reverse(builder.f) : builder.f;
            var tree = A2($elm$core$Array$treeFromBuilder, correctNodeList, builder.b);
            return A4(
                $elm$core$Array$Array_elm_builtin,
                $elm$core$Elm$JsArray$length(builder.e) + treeLen,
                A2($elm$core$Basics$max, 5, depth * $elm$core$Array$shiftStep),
                tree,
                builder.e);
        }
    });
# 定义一个整数除法函数
var $elm$core$Basics$idiv = _Basics_idiv;
# 定义一个小于比较函数
var $elm$core$Basics$lt = _Utils_lt;
# 定义一个初始化辅助函数，用于创建数组
var $elm$core$Array$initializeHelp = F5(
    function (fn, fromIndex, len, nodeList, tail) {
        initializeHelp:
        while (true) {
            if (fromIndex < 0) {
                // 如果 fromIndex 小于 0，则将 nodeList 和 tail 组成数组返回
                return A2(
                    $elm$core$Array$builderToArray,
                    false,
                    {f: nodeList, b: (len / $elm$core$Array$branchFactor) | 0, e: tail});
            } else {
                // 创建叶子节点 leaf，其中包含从 fromIndex 开始的 fn 函数生成的数组
                var leaf = $elm$core$Array$Leaf(
                    A3($elm$core$Elm$JsArray$initialize, $elm$core$Array$branchFactor, fromIndex, fn));
                // 更新循环变量
                var $temp$fn = fn,
                    $temp$fromIndex = fromIndex - $elm$core$Array$branchFactor,
                    $temp$len = len,
                    $temp$nodeList = A2($elm$core$List$cons, leaf, nodeList),
                    $temp$tail = tail;
                fn = $temp$fn;
                fromIndex = $temp$fromIndex;
                len = $temp$len;
                nodeList = $temp$nodeList;
                tail = $temp$tail;
                // 继续执行循环
                continue initializeHelp;
            }
        }
    });
# 定义取余函数
var $elm$core$Basics$remainderBy = _Basics_remainderBy;
# 定义初始化数组函数
var $elm$core$Array$initialize = F2(
    function (len, fn) {
        # 如果长度小于等于0，返回空数组
        if (len <= 0) {
            return $elm$core$Array$empty;
        } else {
            # 计算尾部长度
            var tailLen = len % $elm$core$Array$branchFactor;
            # 生成尾部数组
            var tail = A3($elm$core$Elm$JsArray$initialize, tailLen, len - tailLen, fn);
            # 计算初始索引
            var initialFromIndex = (len - tailLen) - $elm$core$Array$branchFactor;
            # 调用初始化辅助函数
            return A5($elm$core$Array$initializeHelp, fn, initialFromIndex, len, _List_Nil, tail);
        }
    });
# 定义布尔值True
var $elm$core$Basics$True = 0;
# 判断结果是否成功
var $elm$core$Result$isOk = function (result) {
    if (!result.$) {
        return true;
    } else {
        return false;
    }
};
# 对解码结果进行映射
var $elm$json$Json$Decode$map = _Json_map1;
# 对两个解码结果进行映射
var $elm$json$Json$Decode$map2 = _Json_map2;
# 返回成功的解码结果
var $elm$json$Json$Decode$succeed = _Json_succeed;
# 将虚拟 DOM 事件处理器转换为整数
var $elm$virtual_dom$VirtualDom$toHandlerInt = function (handler) {
    switch (handler.$) {
        case 0:
            return 0;
        case 1:
            return 1;
        case 2:
            return 2;
        default:
            return 3;
    }
};
# 定义外部浏览器
var $elm$browser$Browser$External = function (a) {
    return {$: 1, a: a};
};
# 定义内部浏览器
var $elm$browser$Browser$Internal = function (a) {
    return {$: 0, a: a};
};
# 返回输入值
var $elm$core$Basics$identity = function (x) {
    return x;
};
# 定义 DOM 未找到错误
var $elm$browser$Browser$Dom$NotFound = $elm$core$Basics$identity;
# 定义 HTTP 协议
var $elm$url$Url$Http = 0;
# 定义 HTTPS 协议
var $elm$url$Url$Https = 1;
# 定义 URL 类型
var $elm$url$Url$Url = F6(
    function (protocol, host, port_, path, query, fragment) {
        return {X: fragment, Z: host, ac: path, ae: port_, ah: protocol, ai: query};
    });
# 判断字符串是否包含指定子串
var $elm$core$String$contains = _String_contains;
# 返回字符串长度
var $elm$core$String$length = _String_length;
# 截取字符串
var $elm$core$String$slice = _String_slice;
# 删除字符串左侧指定长度的字符
var $elm$core$String$dropLeft = F2(
    function (n, string) {
        return (n < 1) ? string : A3(
            $elm$core$String$slice,
            n,
            $elm$core$String$length(string),
            string);
    });
    });


这是一个 JavaScript 代码块的结束标记，表示一个函数或者一个代码块的结束。
# 定义一个名为 indexes 的变量，指向 _String_indexes 函数
var $elm$core$String$indexes = _String_indexes;
# 定义一个名为 isEmpty 的函数，用于判断字符串是否为空
var $elm$core$String$isEmpty = function (string) {
    return string === '';
};
# 定义一个名为 left 的函数，用于获取字符串的前 n 个字符
var $elm$core$String$left = F2(
    function (n, string) {
        return (n < 1) ? '' : A3($elm$core$String$slice, 0, n, string);
    });
# 定义一个名为 toInt 的函数，用于将字符串转换为整数
var $elm$core$String$toInt = _String_toInt;
# 定义一个名为 chompBeforePath 的函数，用于解析 URL 中的路径部分
var $elm$url$Url$chompBeforePath = F5(
    function (protocol, path, params, frag, str) {
        if ($elm$core$String$isEmpty(str) || A2($elm$core$String$contains, '@', str)) {
            return $elm$core$Maybe$Nothing;
        } else {
            var _v0 = A2($elm$core$String$indexes, ':', str);
            if (!_v0.b) {
                return $elm$core$Maybe$Just(
                    A6($elm$url$Url$Url, protocol, str, $elm$core$Maybe$Nothing, path, params, frag));
            } else {
                if (!_v0.b.b) {
                    var i = _v0.a;
                    var _v1 = $elm$core$String$toInt(
                        A2($elm$core$String$dropLeft, i + 1, str));
                    if (_v1.$ === 1) {
                        return $elm$core$Maybe$Nothing;
                    } else {
                        var port_ = _v1;
                        return $elm$core$Maybe$Just(
                            A6(
                                $elm$url$Url$Url,
                                protocol,
                                A2($elm$core$String$left, i, str),
                                port_,
                                path,
                                params,
                                frag));
                    }
                } else {
                    return $elm$core$Maybe$Nothing;
                }
            }
        }
    });
# 定义一个名为 chompBeforeQuery 的函数，用于解析 URL 中的查询部分
var $elm$url$Url$chompBeforeQuery = F4(
    function (protocol, params, frag, str) {
        // 如果字符串为空，则返回 Nothing
        if ($elm$core$String$isEmpty(str)) {
            return $elm$core$Maybe$Nothing;
        } else {
            // 查找字符串中 '/' 的索引位置
            var _v0 = A2($elm$core$String$indexes, '/', str);
            // 如果没有找到 '/'，则调用 chompBeforePath 函数
            if (!_v0.b) {
                return A5($elm$url$Url$chompBeforePath, protocol, '/', params, frag, str);
            } else {
                // 如果找到了 '/'，则进行字符串分割并调用 chompBeforePath 函数
                var i = _v0.a;
                return A5(
                    $elm$url$Url$chompBeforePath,
                    protocol,
                    A2($elm$core$String$dropLeft, i, str),
                    params,
                    frag,
                    A2($elm$core$String$left, i, str));
            }
        }
    });
-- 定义一个函数，用于从 URL 字符串中截取协议部分之后的内容
var $elm$url$Url$chompBeforeFragment = F3(
    function (protocol, frag, str) {
        -- 如果字符串为空，则返回 Nothing
        if ($elm$core$String$isEmpty(str)) {
            return $elm$core$Maybe$Nothing;
        } else {
            -- 查找字符串中 '?' 的位置
            var _v0 = A2($elm$core$String$indexes, '?', str);
            -- 如果没有找到 '?'，则调用 chompBeforeQuery 函数
            if (!_v0.b) {
                return A4($elm$url$Url$chompBeforeQuery, protocol, $elm$core$Maybe$Nothing, frag, str);
            } else {
                -- 获取 '?' 的位置
                var i = _v0.a;
                -- 调用 chompBeforeQuery 函数，截取协议部分之后的内容和查询部分之前的内容
                return A4(
                    $elm$url$Url$chompBeforeQuery,
                    protocol,
                    $elm$core$Maybe$Just(
                        A2($elm$core$String$dropLeft, i + 1, str)),
                    frag,
                    A2($elm$core$String$left, i, str));
            }
        }
    });
-- 定义一个函数，用于从 URL 字符串中截取协议部分之后的内容
var $elm$url$Url$chompAfterProtocol = F2(
    function (protocol, str) {
        -- 如果字符串为空，则返回 Nothing
        if ($elm$core$String$isEmpty(str)) {
            return $elm$core$Maybe$Nothing;
        } else {
            -- 查找字符串中 '#' 的位置
            var _v0 = A2($elm$core$String$indexes, '#', str);
            -- 如果没有找到 '#'，则调用 chompBeforeFragment 函数
            if (!_v0.b) {
                return A3($elm$url$Url$chompBeforeFragment, protocol, $elm$core$Maybe$Nothing, str);
            } else {
                -- 获取 '#' 的位置
                var i = _v0.a;
                -- 调用 chompBeforeFragment 函数，截取协议部分之后的内容和片段部分之后的内容
                return A3(
                    $elm$url$Url$chompBeforeFragment,
                    protocol,
                    $elm$core$Maybe$Just(
                        A2($elm$core$String$dropLeft, i + 1, str)),
                    A2($elm$core$String$left, i, str));
            }
        }
    });
-- 判断一个字符串是否以指定的前缀开头
var $elm$core$String$startsWith = _String_startsWith;
-- 定义一个函数，用于从字符串中解析出 URL 对象
var $elm$url$Url$fromString = function (str) {
    -- 如果字符串以 'http://' 开头，则调用 chompAfterProtocol 函数，截取协议部分之后的内容
    return A2($elm$core$String$startsWith, 'http://', str) ? A2(
        $elm$url$Url$chompAfterProtocol,
        0,
        A2($elm$core$String$dropLeft, 7, str)) : (A2($elm$core$String$startsWith, 'https://', str) ? A2(
        $elm$url$Url$chompAfterProtocol,
        1,
        A2($elm$core$String$dropLeft, 8, str)) : $elm$core$Maybe$Nothing);
};
-- 定义一个永远不会返回的函数
var $elm$core$Basics$never = function (_v0) {
    # 永远不会执行的代码块
    never:
    # 当条件永远为真时，进入循环
    while (true) {
        # 将_v0赋值给nvr
        var nvr = _v0;
        # 将nvr赋值给$temp$_v0
        var $temp$_v0 = nvr;
        # 将$temp$_v0赋值给_v0
        _v0 = $temp$_v0;
        # 继续执行永远不会执行的代码块
        continue never;
    }
};
// 定义一个变量 $elm$core$Task$Perform，其值为 $elm$core$Basics$identity
var $elm$core$Task$Perform = $elm$core$Basics$identity;
// 定义一个函数 $elm$core$Task$succeed，其值为 _Scheduler_succeed
var $elm$core$Task$succeed = _Scheduler_succeed;
// 定义一个变量 $elm$core$Task$init，其值为 $elm$core$Task$succeed(0)
var $elm$core$Task$init = $elm$core$Task$succeed(0);
// 定义一个函数 $elm$core$List$foldrHelper，接受四个参数：fn, acc, ctr, ls
var $elm$core$List$foldrHelper = F4(
    function (fn, acc, ctr, ls) {
        // 如果 ls 为空列表，则返回 acc
        if (!ls.b) {
            return acc;
        } else {
            var a = ls.a;
            var r1 = ls.b;
            // 如果 r1 为空列表，则返回 A2(fn, a, acc)
            if (!r1.b) {
                return A2(fn, a, acc);
            } else {
                var b = r1.a;
                var r2 = r1.b;
                // 如果 r2 为空列表，则返回 A2(fn, a, A2(fn, b, acc))
                if (!r2.b) {
                    return A2(
                        fn,
                        a,
                        A2(fn, b, acc));
                } else {
                    var c = r2.a;
                    var r3 = r2.b;
                    // 如果 r3 为空列表，则返回 A2(fn, a, A2(fn, b, A2(fn, c, acc)))
                    if (!r3.b) {
                        return A2(
                            fn,
                            a,
                            A2(
                                fn,
                                b,
                                A2(fn, c, acc)));
                    } else {
                        var d = r3.a;
                        var r4 = r3.b;
                        // 如果 ctr 大于 500，则返回 A3($elm$core$List$foldl, fn, acc, $elm$core$List$reverse(r4))，否则返回 A4($elm$core$List$foldrHelper, fn, acc, ctr + 1, r4)
                        var res = (ctr > 500) ? A3(
                            $elm$core$List$foldl,
                            fn,
                            acc,
                            $elm$core$List$reverse(r4)) : A4($elm$core$List$foldrHelper, fn, acc, ctr + 1, r4);
                        // 返回 A2(fn, a, A2(fn, b, A2(fn, c, A2(fn, d, res))))
                        return A2(
                            fn,
                            a,
                            A2(
                                fn,
                                b,
                                A2(
                                    fn,
                                    c,
                                    A2(fn, d, res))));
                    }
                }
            }
        }
    });
// 定义一个函数 $elm$core$List$foldr，接受三个参数：fn, acc, ls
var $elm$core$List$foldr = F3(
    function (fn, acc, ls) {
        // 返回 A4($elm$core$List$foldrHelper, fn, acc, 0, ls)
        return A4($elm$core$List$foldrHelper, fn, acc, 0, ls);
    });
# 定义一个函数，将函数 f 应用到列表 xs 的每个元素上，并返回结果列表
var $elm$core$List$map = F2(
    function (f, xs) {
        # 使用 foldr 函数将 f 应用到列表 xs 的每个元素上，并将结果添加到累加器中
        return A3(
            $elm$core$List$foldr,
            F2(
                function (x, acc) {
                    # 将 f(x) 添加到累加器中
                    return A2(
                        $elm$core$List$cons,
                        f(x),
                        acc);
                }),
            _List_Nil,  # 初始累加器为空列表
            xs);
    });

# 定义一个函数，将 func 应用到 taskA 的结果上，并返回新的任务
var $elm$core$Task$map = F2(
    function (func, taskA) {
        # 使用 andThen 函数将 func 应用到 taskA 的结果上，并返回新的任务
        return A2(
            $elm$core$Task$andThen,
            function (a) {
                # 将 func(a) 封装成成功的任务
                return $elm$core$Task$succeed(
                    func(a));
            },
            taskA);
    });

# 定义一个函数，将 func 应用到 taskA 和 taskB 的结果上，并返回新的任务
var $elm$core$Task$map2 = F3(
    function (func, taskA, taskB) {
        # 使用 andThen 函数将 func 应用到 taskA 的结果上，并返回新的任务
        return A2(
            $elm$core$Task$andThen,
            function (a) {
                # 使用 andThen 函数将 func 应用到 taskB 的结果上，并返回新的任务
                return A2(
                    $elm$core$Task$andThen,
                    function (b) {
                        # 将 func(a, b) 封装成成功的任务
                        return $elm$core$Task$succeed(
                            A2(func, a, b));
                    },
                    taskB);
            },
            taskA);
    });

# 定义一个函数，将任务列表中的任务顺序执行，并返回结果列表的任务
var $elm$core$Task$sequence = function (tasks) {
    # 使用 foldr 函数将 map2(cons) 应用到任务列表中的每个任务上，并返回结果列表的任务
    return A3(
        $elm$core$List$foldr,
        $elm$core$Task$map2($elm$core$List$cons),  # 使用 map2(cons) 函数将任务结果添加到结果列表中
        $elm$core$Task$succeed(_List_Nil),  # 初始结果列表为空列表的任务
        tasks);
};

# 将消息发送给应用程序的函数
var $elm$core$Platform$sendToApp = _Platform_sendToApp;

# 定义一个函数，将任务作为命令发送给调度器
var $elm$core$Task$spawnCmd = F2(
    function (router, _v0) {
        var task = _v0;
        # 使用 andThen 函数将任务发送给应用程序，并返回新的任务
        return _Scheduler_spawn(
            A2(
                $elm$core$Task$andThen,
                $elm$core$Platform$sendToApp(router),  # 将任务发送给应用程序
                task));
    });

# 定义一个函数，处理任务的副作用
var $elm$core$Task$onEffects = F3(
    # 定义一个匿名函数，接受三个参数：router, commands, state
    function (router, commands, state) {
        # 返回一个任务，将任务的结果映射为0
        return A2(
            $elm$core$Task$map,
            function (_v0) {
                return 0;
            },
            # 将命令列表转换为任务列表，并执行这些任务
            $elm$core$Task$sequence(
                A2(
                    $elm$core$List$map,
                    # 将路由器和命令转换为任务
                    $elm$core$Task$spawnCmd(router),
                    commands)));
    });
var $elm$core$Task$onSelfMsg = F3(
    function (_v0, _v1, _v2) {
        return $elm$core$Task$succeed(0);
    });
# 定义一个接收三个参数的函数，返回一个成功的任务

var $elm$core$Task$cmdMap = F2(
    function (tagger, _v0) {
        var task = _v0;
        return A2($elm$core$Task$map, tagger, task);
    });
# 定义一个接收两个参数的函数，返回一个映射后的任务

_Platform_effectManagers['Task'] = _Platform_createManager($elm$core$Task$init, $elm$core$Task$onEffects, $elm$core$Task$onSelfMsg, $elm$core$Task$cmdMap);
# 将任务的初始化、效果处理、自身消息处理和映射函数传入平台管理器中

var $elm$core$Task$command = _Platform_leaf('Task');
# 定义一个命令，传入字符串'Task'作为参数

var $elm$core$Task$perform = F2(
    function (toMessage, task) {
        return $elm$core$Task$command(
            A2($elm$core$Task$map, toMessage, task));
    });
# 定义一个接收两个参数的函数，返回一个命令

var $elm$browser$Browser$element = _Browser_element;
# 将浏览器元素赋值给变量

var $author$project$Main$NewCard = function (a) {
    return {$: 2, a: a};
};
# 定义一个接收一个参数的函数，返回一个包含参数的对象

var $elm$random$Random$Generate = $elm$core$Basics$identity;
# 将基本函数库中的identity函数赋值给生成器

var $elm$random$Random$Seed = F2(
    function (a, b) {
        return {$: 0, a: a, b: b};
    });
# 定义一个接收两个参数的函数，返回一个包含参数的对象

var $elm$core$Bitwise$shiftRightZfBy = _Bitwise_shiftRightZfBy;
# 将位运算库中的shiftRightZfBy函数赋值给变量

var $elm$random$Random$next = function (_v0) {
    var state0 = _v0.a;
    var incr = _v0.b;
    return A2($elm$random$Random$Seed, ((state0 * 1664525) + incr) >>> 0, incr);
};
# 定义一个接收一个参数的函数，返回一个新的种子对象

var $elm$random$Random$initialSeed = function (x) {
    var _v0 = $elm$random$Random$next(
        A2($elm$random$Random$Seed, 0, 1013904223));
    var state1 = _v0.a;
    var incr = _v0.b;
    var state2 = (state1 + x) >>> 0;
    return $elm$random$Random$next(
        A2($elm$random$Random$Seed, state2, incr));
};
# 定义一个接收一个参数的函数，返回一个新的种子对象

var $elm$time$Time$Name = function (a) {
    return {$: 0, a: a};
};
# 定义一个接收一个参数的函数，返回一个包含参数的对象

var $elm$time$Time$Offset = function (a) {
    return {$: 1, a: a};
};
# 定义一个接收一个参数的函数，返回一个包含参数的对象

var $elm$time$Time$Zone = F2(
    function (a, b) {
        return {$: 0, a: a, b: b};
    });
# 定义一个接收两个参数的函数，返回一个包含参数的对象

var $elm$time$Time$customZone = $elm$time$Time$Zone;
# 将时间库中的Zone函数赋值给自定义区域

var $elm$time$Time$Posix = $elm$core$Basics$identity;
# 将基本函数库中的identity函数赋值给Posix

var $elm$time$Time$millisToPosix = $elm$core$Basics$identity;
# 将基本函数库中的identity函数赋值给millisToPosix

var $elm$time$Time$now = _Time_now($elm$time$Time$millisToPosix);
# 将时间库中的now函数和millisToPosix函数传入_Time_now函数中
var $elm$time$Time$posixToMillis = function (_v0) {
    // 将 POSIX 时间转换为毫秒
    var millis = _v0;
    return millis;
};
var $elm$random$Random$init = A2(
    $elm$core$Task$andThen,
    // 初始化随机数生成器
    function (time) {
        return $elm$core$Task$succeed(
            $elm$random$Random$initialSeed(
                $elm$time$Time$posixToMillis(time)));
    },
    $elm$time$Time$now);
var $elm$random$Random$step = F2(
    function (_v0, seed) {
        // 执行随机数生成器的一步操作
        var generator = _v0;
        return generator(seed);
    });
var $elm$random$Random$onEffects = F3(
    function (router, commands, seed) {
        if (!commands.b) {
            // 如果没有命令，直接返回种子
            return $elm$core$Task$succeed(seed);
        } else {
            var generator = commands.a;
            var rest = commands.b;
            var _v1 = A2($elm$random$Random$step, generator, seed);
            var value = _v1.a;
            var newSeed = _v1.b;
            return A2(
                $elm$core$Task$andThen,
                // 执行命令并递归处理剩余命令
                function (_v2) {
                    return A3($elm$random$Random$onEffects, router, rest, newSeed);
                },
                A2($elm$core$Platform$sendToApp, router, value));
        }
    });
var $elm$random$Random$onSelfMsg = F3(
    function (_v0, _v1, seed) {
        // 处理自身消息
        return $elm$core$Task$succeed(seed);
    });
var $elm$random$Random$Generator = $elm$core$Basics$identity;
var $elm$random$Random$map = F2(
    function (func, _v0) {
        var genA = _v0;
        return function (seed0) {
            var _v1 = genA(seed0);
            var a = _v1.a;
            var seed1 = _v1.b;
            return _Utils_Tuple2(
                // 对生成的随机数进行映射
                func(a),
                seed1);
        };
    });
var $elm$random$Random$cmdMap = F2(
    function (func, _v0) {
        var generator = _v0;
        return A2($elm$random$Random$map, func, generator);
    });
// 创建 Random 效果管理器
_Platform_effectManagers['Random'] = _Platform_createManager($elm$random$Random$init, $elm$random$Random$onEffects, $elm$random$Random$onSelfMsg, $elm$random$Random$cmdMap);
# 定义一个名为 $elm$random$Random$command 的变量，其值为 _Platform_leaf('Random')
var $elm$random$Random$command = _Platform_leaf('Random');
# 定义一个名为 $elm$random$Random$generate 的函数，接受一个标签函数和一个生成器函数作为参数
var $elm$random$Random$generate = F2(
    function (tagger, generator) {
        return $elm$random$Random$command(
            A2($elm$random$Random$map, tagger, generator));
    });
# 定义一个名为 $elm$core$Bitwise$and 的变量，其值为 _Bitwise_and
var $elm$core$Bitwise$and = _Bitwise_and;
# 定义一个名为 $elm$core$Basics$negate 的函数，接受一个参数 n，返回其相反数
var $elm$core$Basics$negate = function (n) {
    return -n;
};
# 定义一个名为 $elm$core$Bitwise$xor 的变量，其值为 _Bitwise_xor
var $elm$core$Bitwise$xor = _Bitwise_xor;
# 定义一个名为 $elm$random$Random$peel 的函数，接受一个参数 _v0
var $elm$random$Random$peel = function (_v0) {
    # 从参数 _v0 中获取属性 a 的值，赋给变量 state
    var state = _v0.a;
    # 计算 word 的值
    var word = (state ^ (state >>> ((state >>> 28) + 4))) * 277803737;
    # 返回 word 的右移位操作结果
    return ((word >>> 22) ^ word) >>> 0;
};
# 定义一个名为 $elm$random$Random$int 的函数，接受两个参数 a 和 b
var $elm$random$Random$int = F2(
    function (a, b) {
        return function (seed0) {
            # 判断 a 和 b 的大小关系，将较小的值赋给 lo，较大的值赋给 hi
            var _v0 = (_Utils_cmp(a, b) < 0) ? _Utils_Tuple2(a, b) : _Utils_Tuple2(b, a);
            var lo = _v0.a;
            var hi = _v0.b;
            # 计算范围 range
            var range = (hi - lo) + 1;
            # 如果范围 range 满足条件
            if (!((range - 1) & range)) {
                # 返回结果
                return _Utils_Tuple2(
                    (((range - 1) & $elm$random$Random$peel(seed0)) >>> 0) + lo,
                    $elm$random$Random$next(seed0));
            } else {
                # 计算 threshhold
                var threshhold = (((-range) >>> 0) % range) >>> 0;
                # 定义一个名为 accountForBias 的函数
                var accountForBias = function (seed) {
                    accountForBias:
                    while (true) {
                        # 获取随机数 x
                        var x = $elm$random$Random$peel(seed);
                        # 获取下一个种子值
                        var seedN = $elm$random$Random$next(seed);
                        # 如果 x 小于 threshhold
                        if (_Utils_cmp(x, threshhold) < 0) {
                            # 继续循环
                            var $temp$seed = seedN;
                            seed = $temp$seed;
                            continue accountForBias;
                        } else {
                            # 返回结果
                            return _Utils_Tuple2((x % range) + lo, seedN);
                        }
                    }
                };
                # 调用 accountForBias 函数
                return accountForBias(seed0);
            }
        };
    });
# 定义一个名为 $author$project$Main$newCard 的变量，其值为调用 $elm$random$Random$int 函数，传入参数 2 和 14
var $author$project$Main$newCard = A2($elm$random$Random$int, 2, 14);
# 定义一个名为 $author$project$Main$init 的函数，接受一个参数 _v0
var $author$project$Main$init = function (_v0) {
    # 返回一个包含两个元素的元组
    return _Utils_Tuple2(
        {
            # 第一个元素是一个包含键值对的对象
            a: {d: $elm$core$Maybe$Nothing, g: $elm$core$Maybe$Nothing, w: $elm$core$Maybe$Nothing},
            # 对象中的键C对应的值为$elm$core$Maybe$Nothing
            C: $elm$core$Maybe$Nothing,
            # 对象中的键D对应的值为$elm$core$Maybe$Nothing
            D: $elm$core$Maybe$Nothing,
            # 对象中的键i对应的值为100
            i: 100,
            # 对象中的键j对应的值为0
            j: 0
        },
        # 第二个元素是通过调用A2函数生成的值
        A2($elm$random$Random$generate, $author$project$Main$NewCard, $author$project$Main$newCard));
};
// 定义一个名为 $elm$core$Platform$Sub$batch 的变量，其值为 _Platform_batch
var $elm$core$Platform$Sub$batch = _Platform_batch;
// 定义一个名为 $elm$core$Platform$Sub$none 的变量，其值为 $elm$core$Platform$Sub$batch(_List_Nil)
var $elm$core$Platform$Sub$none = $elm$core$Platform$Sub$batch(_List_Nil);
// 定义一个名为 $author$project$Main$subscriptions 的函数，参数为 _v0，返回值为 $elm$core$Platform$Sub$none
var $author$project$Main$subscriptions = function (_v0) {
    return $elm$core$Platform$Sub$none;
};
// 定义一个名为 $author$project$Main$NewCardC 的函数，参数为 a，返回值为 {$: 3, a: a}
var $author$project$Main$NewCardC = function (a) {
    return {$: 3, a: a};
};
// 定义一个名为 $elm$core$Platform$Cmd$batch 的变量，其值为 _Platform_batch
var $elm$core$Platform$Cmd$batch = _Platform_batch;
// 定义一个名为 $elm$core$Platform$Cmd$none 的变量，其值为 $elm$core$Platform$Cmd$batch(_List_Nil)
var $elm$core$Platform$Cmd$none = $elm$core$Platform$Cmd$batch(_List_Nil);
// 定义一个名为 $author$project$Main$calculateNewState 的函数，参数为 F2( )
var $author$project$Main$calculateNewState = F2(
    });
// 定义一个名为 $author$project$Main$update 的函数，参数为 F2( )
var $author$project$Main$update = F2(
    });
// 定义一个名为 $elm$virtual_dom$VirtualDom$style 的变量，其值为 _VirtualDom_style
var $elm$virtual_dom$VirtualDom$style = _VirtualDom_style;
// 定义一个名为 $elm$html$Html$Attributes$style 的变量，其值为 $elm$virtual_dom$VirtualDom$style
var $elm$html$Html$Attributes$style = $elm$virtual_dom$VirtualDom$style;
// 定义一个名为 $author$project$Main$centerHeadlineStyle 的变量，其值为 _List_fromArray([A2($elm$html$Html$Attributes$style, 'display', 'grid'), A2($elm$html$Html$Attributes$style, 'place-items', 'center'), A2($elm$html$Html$Attributes$style, 'margin', '2rem')])
var $author$project$Main$centerHeadlineStyle = _List_fromArray(
    [
        A2($elm$html$Html$Attributes$style, 'display', 'grid'),
        A2($elm$html$Html$Attributes$style, 'place-items', 'center'),
        A2($elm$html$Html$Attributes$style, 'margin', '2rem')
    ]);
// 定义一个名为 $elm$html$Html$div 的变量，其值为 _VirtualDom_node('div')
var $elm$html$Html$div = _VirtualDom_node('div');
// 定义一个名为 $author$project$Main$NewGame 的变量，其值为 {$: 5}
var $author$project$Main$NewGame = {$: 5};
// 定义一个名为 $author$project$Main$Play 的变量，其值为 {$: 4}
var $author$project$Main$Play = {$: 4};
// 定义一个名为 $author$project$Main$UpdateBetValue 的函数，参数为 a，返回值为 {$: 1, a: a}
var $author$project$Main$UpdateBetValue = function (a) {
    return {$: 1, a: a};
};
// 定义一个名为 $elm$html$Html$article 的变量，其值为 _VirtualDom_node('article')
var $elm$html$Html$article = _VirtualDom_node('article');
// 定义一个名为 $elm$html$Html$button 的变量，其值为 _VirtualDom_node('button')
var $elm$html$Html$button = _VirtualDom_node('button');
// 定义一个名为 $author$project$Main$cardContentPStyle 的变量，其值为 _List_fromArray([A2($elm$html$Html$Attributes$style, 'font-size', '2rem')])
var $author$project$Main$cardContentPStyle = _List_fromArray(
    [
        A2($elm$html$Html$Attributes$style, 'font-size', '2rem')
    ]);
// 定义一个名为 $author$project$Main$cardToString 的函数，参数为 card，根据 card 的值返回相应的字符串
var $author$project$Main$cardToString = function (card) {
    if (!card.$) {
        var value = card.a;
        if (value < 11) {
            return $elm$core$String$fromInt(value);
        } else {
            switch (value) {
                case 11:
                    return 'Jack';
                case 12:
                    return 'Queen';
                case 13:
                    return 'King';
                case 14:
                    return 'Ace';
                default:
                    return 'impossible value';
            }
        }
    # 如果条件不成立，返回短横线
    } else {
        return '-';
    }
};
// 定义游戏样式，包含宽度和最大宽度
var $author$project$Main$gameStyle = _List_fromArray(
    [
        A2($elm$html$Html$Attributes$style, 'width', '100%'),
        A2($elm$html$Html$Attributes$style, 'max-width', '70rem')
    ]);
// 创建 input 元素
var $elm$html$Html$input = _VirtualDom_node('input');
// 将字符串封装成 JSON 字符串
var $elm$json$Json$Encode$string = _Json_wrap;
// 创建字符串属性
var $elm$html$Html$Attributes$stringProperty = F2(
    function (key, string) {
        return A2(
            _VirtualDom_property,
            key,
            $elm$json$Json$Encode$string(string));
    });
// 创建最大值属性
var $elm$html$Html$Attributes$max = $elm$html$Html$Attributes$stringProperty('max');
// 创建最小值属性
var $elm$html$Html$Attributes$min = $elm$html$Html$Attributes$stringProperty('min');
// 创建普通虚拟 DOM 节点
var $elm$virtual_dom$VirtualDom$Normal = function (a) {
    return {$: 0, a: a};
};
// 创建事件监听器
var $elm$virtual_dom$VirtualDom$on = _VirtualDom_on;
// 创建事件监听器，触发消息
var $elm$html$Html$Events$on = F2(
    function (event, decoder) {
        return A2(
            $elm$virtual_dom$VirtualDom$on,
            event,
            $elm$virtual_dom$VirtualDom$Normal(decoder));
    });
// 创建点击事件监听器
var $elm$html$Html$Events$onClick = function (msg) {
    return A2(
        $elm$html$Html$Events$on,
        'click',
        $elm$json$Json$Decode$succeed(msg));
};
// 创建始终阻止事件传播的事件监听器
var $elm$html$Html$Events$alwaysStop = function (x) {
    return _Utils_Tuple2(x, true);
};
// 创建可能阻止事件传播的事件监听器
var $elm$virtual_dom$VirtualDom$MayStopPropagation = function (a) {
    return {$: 1, a: a};
};
// 创建在特定事件上阻止事件传播的事件监听器
var $elm$html$Html$Events$stopPropagationOn = F2(
    function (event, decoder) {
        return A2(
            $elm$virtual_dom$VirtualDom$on,
            event,
            $elm$virtual_dom$VirtualDom$MayStopPropagation(decoder));
    });
// 解析 JSON 字段
var $elm$json$Json$Decode$field = _Json_decodeField;
// 从 JSON 对象中获取特定字段的解码器
var $elm$json$Json$Decode$at = F2(
    function (fields, decoder) {
        return A3($elm$core$List$foldr, $elm$json$Json$Decode$field, decoder, fields);
    });
// 解析 JSON 字符串
var $elm$json$Json$Decode$string = _Json_decodeString;
// 获取事件目标的值
var $elm$html$Html$Events$targetValue = A2(
    $elm$json$Json$Decode$at,
    # 将数组转换为列表
    _List_fromArray(
        ['target', 'value']),
    # 解码 JSON 字符串
    $elm$json$Json$Decode$string);
# 定义一个函数，用于处理输入事件，并调用指定的消息处理函数
var $elm$html$Html$Events$onInput = function (tagger) {
    # 返回一个事件处理器，用于阻止事件冒泡
    return A2(
        $elm$html$Html$Events$stopPropagationOn,
        'input',
        # 将目标值映射为指定的消息，并阻止事件冒泡
        A2(
            $elm$json$Json$Decode$map,
            $elm$html$Html$Events$alwaysStop,
            A2($elm$json$Json$Decode$map, tagger, $elm$html$Html$Events$targetValue)));
};
# 创建一个 p 标签
var $elm$html$Html$p = _VirtualDom_node('p');
# 定义一个样式，设置字体大小为 2rem
var $author$project$Main$standardFontSize = A2($elm$html$Html$Attributes$style, 'font-size', '2rem');
# 创建一个文本节点
var $elm$virtual_dom$VirtualDom$text = _VirtualDom_text;
# 创建一个包含文本的 Html 元素
var $elm$html$Html$text = $elm$virtual_dom$VirtualDom$text;
# 根据值显示错误信息
var $author$project$Main$showError = function (value) {
    if (!value.$) {
        # 如果值不为空，创建一个 p 标签，显示错误信息
        var string = value.a;
        return A2(
            $elm$html$Html$p,
            _List_fromArray(
                [$author$project$Main$standardFontSize]),
            _List_fromArray(
                [
                    $elm$html$Html$text(string)
                ]));
    } else {
        # 如果值为空，创建一个空的 div 元素
        return A2($elm$html$Html$div, _List_Nil, _List_Nil);
    }
};
# 根据三个卡片的比较结果，返回不同的消息
var $author$project$Main$getGameStateMessage = F3(
    function (cardA, cardB, cardC) {
        return ((_Utils_cmp(cardA, cardC) < 0) && (_Utils_cmp(cardB, cardC) > 0)) ? A2(
            $elm$html$Html$div,
            _List_fromArray(
                [$author$project$Main$standardFontSize]),
            _List_fromArray(
                [
                    $elm$html$Html$text('You won :)')
                ])) : A2(
            $elm$html$Html$div,
            _List_fromArray(
                [$author$project$Main$standardFontSize]),
            _List_fromArray(
                [
                    $elm$html$Html$text('You loose :(')
                ]));
    });
# 定义一个函数，用于将三个值映射为一个值
var $elm$core$Maybe$map3 = F4(
    # 定义一个函数，接受四个参数：func, ma, mb, mc
    function (func, ma, mb, mc) {
        # 如果 ma 的构造器是 Just，则返回一个 Nothing
        if (ma.$ === 1) {
            return $elm$core$Maybe$Nothing;
        } else {
            # 从 ma 中获取值并赋给变量 a
            var a = ma.a;
            # 如果 mb 的构造器是 Just，则返回一个 Nothing
            if (mb.$ === 1) {
                return $elm$core$Maybe$Nothing;
            } else {
                # 从 mb 中获取值并赋给变量 b
                var b = mb.a;
                # 如果 mc 的构造器是 Just，则返回一个 Nothing
                if (mc.$ === 1) {
                    return $elm$core$Maybe$Nothing;
                } else {
                    # 从 mc 中获取值并赋给变量 c，然后调用 func 函数，并将 a, b, c 作为参数传入，将结果包装成 Just 返回
                    var c = mc.a;
                    return $elm$core$Maybe$Just(
                        A3(func, a, b, c));
                }
            }
        }
    });
-- 定义一个函数，用于从 Maybe 类型中获取值，如果 Maybe 为空则返回默认值
var $elm$core$Maybe$withDefault = F2(
    function (_default, maybe) {
        if (!maybe.$) {
            var value = maybe.a;
            return value;
        } else {
            return _default;
        }
    });

-- 定义一个函数，用于展示最后一次游戏的胜负信息
var $author$project$Main$showLastWinLose = function (game) {
    return A2(
        $elm$core$Maybe$withDefault,
        $elm$html$Html$text('something is wrong'),
        A4($elm$core$Maybe$map3, $author$project$Main$getGameStateMessage, game.d, game.g, game.w));
};

-- 定义一个函数，用于展示最后一次游戏的信息
var $author$project$Main$showLastGame = function (game) {
    -- 如果游戏状态为空
    if (game.$ === 1) {
        -- 返回一个包含文本的 div 元素
        return A2(
            $elm$html$Html$div,
            _List_fromArray(
                [$author$project$Main$standardFontSize]),
            _List_fromArray(
                [
                    $elm$html$Html$text('This is your first game')
                ]));
    } else {
        // 从 game 对象中获取属性 a 的值
        var value = game.a;
        // 返回一个包含指定内容的 div 元素
        return A2(
            $elm$html$Html$div,
            _List_Nil,
            _List_fromArray(
                [
                    // 调用 Main 模块的 showLastWinLose 函数，并将 value 作为参数
                    $author$project$Main$showLastWinLose(value),
                    // 返回一个包含指定内容的 p 元素，显示 Card 1 的内容
                    A2(
                    $elm$html$Html$p,
                    $author$project$Main$cardContentPStyle,
                    _List_fromArray(
                        [
                            // 在 p 元素中显示 Card 1 的内容
                            $elm$html$Html$text(
                            'Card 1: ' + $author$project$Main$cardToString(value.d))
                        ])),
                    // 返回一个包含指定内容的 p 元素，显示 Card 2 的内容
                    A2(
                    $elm$html$Html$p,
                    $author$project$Main$cardContentPStyle,
                    _List_fromArray(
                        [
                            // 在 p 元素中显示 Card 2 的内容
                            $elm$html$Html$text(
                            'Card 2: ' + $author$project$Main$cardToString(value.g))
                        ])),
                    // 返回一个包含指定内容的 p 元素，显示 Drawn Card 的内容
                    A2(
                    $elm$html$Html$p,
                    $author$project$Main$cardContentPStyle,
                    _List_fromArray(
                        [
                            // 在 p 元素中显示 Drawn Card 的内容
                            $elm$html$Html$text(
                            'Drawn Card: ' + $author$project$Main$cardToString(value.w))
                        ]))
                ]));
    }
# 定义一个变量，用于表示 HTML 元素的属性 type
var $elm$html$Html$Attributes$type_ = $elm$html$Html$Attributes$stringProperty('type');
# 定义一个变量，用于表示 HTML 元素的属性 value
var $elm$html$Html$Attributes$value = $elm$html$Html$Attributes$stringProperty('value');
# 定义一个函数，用于显示游戏界面
var $author$project$Main$showGame = function (model) {
};
# 定义一个变量，用于表示 HTML 元素 h1
var $elm$html$Html$h1 = _VirtualDom_node('h1');
# 定义一个变量，用于表示标题样式
var $author$project$Main$headerStyle = _List_fromArray(
    [
        A2($elm$html$Html$Attributes$style, 'font-size', '2rem'),
        A2($elm$html$Html$Attributes$style, 'text-align', 'center')
    ]);
# 定义一个函数，用于显示标题
var $author$project$Main$showHeader = A2(
    $elm$html$Html$div,
    $author$project$Main$headerStyle,
    _List_fromArray(
        [
            A2(
            $elm$html$Html$h1,
            _List_fromArray(
                [
                    A2($elm$html$Html$Attributes$style, 'font-size', '4rem')
                ]),
            _List_fromArray(
                [
                    $elm$html$Html$text('ACEY DUCEY CARD GAME')
                ])),
            A2(
            $elm$html$Html$div,
            _List_Nil,
            _List_fromArray(
                [
                    $elm$html$Html$text('Creative Computing Morristown, New Jersey')
                ])),
            A2(
            $elm$html$Html$div,
            _List_Nil,
            _List_fromArray(
                [
                    $elm$html$Html$text('\n        Acey-Ducey is played in the following manner. The Dealer (Computer) deals two cards face up. \n        You have an option to bet or not bet depending on whether or not you feel the card will have a value between the first two.\n        If you do not want to bet, bet 0.\n        ')
                ]))
        ]));
# 定义一个函数，用于显示整个界面
var $author$project$Main$view = function (model) {
    return A2(
        $elm$html$Html$div,
        $author$project$Main$centerHeadlineStyle,
        _List_fromArray(
            [
                $author$project$Main$showHeader,
                $author$project$Main$showGame(model)
            ]));
};
# 定义 Elm 主程序的入口函数 main
var $author$project$Main$main = $elm$browser$Browser$element(
    {
        # 初始化函数
        aB: $author$project$Main$init, 
        # 订阅函数
        aH: $author$project$Main$subscriptions, 
        # 更新函数
        aJ: $author$project$Main$update, 
        # 视图函数
        aK: $author$project$Main$view
    }
);
# 导出 Elm 主程序的入口函数 main
_Platform_export(
    {
        'Main': {
            # 初始化函数
            'init': $author$project$Main$main(
                # 使用 Json 解码器成功解码 0
                $elm$json$Json$Decode$succeed(0)
            )(0)
        }
    }
);
# 执行上述代码
}(this));
```