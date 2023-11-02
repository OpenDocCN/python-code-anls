# ZeroNet源码解析 11

# `plugins/Trayicon/lib/winfolders.py`

这段代码的作用是获取Windows系统中的特殊文件夹，但不包括Python创建的文件夹。它使用`shell32.SHGetFolderPathW`函数获取Windows系统中的所有文件夹，并将它们存储在名为`_results`的Unicode字符串中。


import specialfolders
import ctypes

specialfolders.get(specialfolders.PROGRAMS)


这段代码使用`specialfolders`模块中的`get`函数获取Windows系统中的所有文件夹，并将它们存储在变量`_results`中。`PROGRAMS`是一个字符串，它包含Python安装目录中的所有文件夹的名称。


start_programs = specialfolders.get(specialfolders.PROGRAMS)


这段代码使用`get`函数获取所有文件夹的名称，并将它们存储在变量`start_programs`中。


       HWND hwnd = None
       PWSTR lpBuffer = _cub<PWSTR>(0)
       DWORD dwSize = 0
       _SHGetFolderPathW(None, _LPCWSTR(start_programs), &hwnd, None, 0, _MAX_PATH, &lpBuffer, &dwSize)

       if hwnd:
           _cvt_i32(hwnd, _HWND(hwnd), &dwSize, _cift_i32(0))
           _cvt_i32(hwnd, _HANDLE(hwnd), &dwSize, _cift_i32(0))
           _cvt_i32(hwnd, _DWORD(0), &dwSize, _cift_i32(0))
           _cvt_i32(hwnd, _LPCWSTR(start_programs), &lpBuffer, _cift_i32(0), _MAX_PATH)
           _cvt_i32(hwnd, _cift_i32(0), &dwSize, _cift_i32(0))
           _cvt_i32(hwnd, _HWND(0), &dwSize, _cift_i32(0))
           _cvt_i32(hwnd, _HANDLE(0), &dwSize, _cift_i32(0))
           _cvt_i32(hwnd, _DWORD(0), &dwSize, _cift_i32(0))
           _cvt_i32(hwnd, _LPCWSTR(0), &lpBuffer, _cift_i32(0), _MAX_PATH)

           specialfolders.result = True
       else:
           specialfolders.result = False


这段代码使用`shell32.SHGetFolderPathW`函数获取所有文件夹的名称，并将它们存储在变量`start_programs`中。然后，它使用`_cvt_i32`函数将这些名称转换为整数类型，以便将其存储在变量`start_programs_i32`中。


       if hwnd:
           _cvt_i32(hwnd, _HWND(hwnd), &dwSize, _cift_i32(0))
           _cvt_i32(hwnd, _HANDLE(hwnd), &dwSize, _cift_i32(0))
           _cvt_i32(hwnd, _DWORD(0), &dwSize, _cift_i32(0))
           _cvt_i32(hwnd, _LPCWSTR(start_programs), &lpBuffer, _cift_i32(0), _MAX_PATH)
           _cvt_i32(hwnd, _cift_i32(0), &dwSize, _cift_i32(0))
           _cvt_i32(hwnd, _HWND(0), &dwSize, _cift_i32(0))
           _cvt_i32(hwnd, _HANDLE(0), &dwSize, _cift_i32(0))
           _cvt_i32(hwnd, _DWORD(0), &dwSize, _cift_i32(0))
           _cvt_i32(hwnd, _LPCWSTR(0), &lpBuffer, _cift_i32(0), _MAX_PATH)

           specialfolders.result = True
       else:



```py
''' Get windows special folders without pythonwin
    Example:
            import specialfolders
            start_programs = specialfolders.get(specialfolders.PROGRAMS)

Code is public domain, do with it what you will.

Luke Pinner - Environment.gov.au, 2010 February 10
'''

#Imports use _syntax to mask them from autocomplete IDE's
import ctypes as _ctypes
from ctypes import create_unicode_buffer as _cub
from ctypes.wintypes import HWND as _HWND, HANDLE as _HANDLE,DWORD as _DWORD,LPCWSTR as _LPCWSTR,MAX_PATH as _MAX_PATH
_SHGetFolderPath = _ctypes.windll.shell32.SHGetFolderPathW

```

这段代码定义了一个名为 "constants" 的特殊文件夹，其中包含了一些常量。这些常量提供了关于桌面文件夹（DESKTOP）中不同文件的信息。

具体来说，常量包括：

- number 0: "DESKTOP"（桌面文件夹）
- number 2: "PROGRAMS"（程序文件夹）
- number 5: "MYDOCUMENTS"（个人文档文件夹）
- number 6: "FAVORITES"（喜好文件夹）
- number 7: "STARTUP"（启动文件夹）
- number 8: "RECENT"（最近访问文件夹）
- number 9: "SENDTO"（发送文件夹）
- number 11: "STARTMENU"（启动菜单文件夹）
- number 13: "MYMUSIC"（音乐文件夹）
- number 14: "MYVIDEOS"（视频文件夹）
- number 19: "NETHOOD"（网络文件夹）
- number 20: "FONTS"（字体文件夹）
- number 21: "TEMPLATES"（模板文件夹）
- number 22: "ALLUSERSSTARTMENU"（所有用户启动菜单文件夹）

这些常量的值在程序运行时由系统获取，然后用于设置用户桌面和文件系统的初始设置，如桌面文件夹包含哪些文件和文件夹的优先级。


```py
#public special folder constants
DESKTOP=                             0
PROGRAMS=                            2
MYDOCUMENTS=                         5
FAVORITES=                           6
STARTUP=                             7
RECENT=                              8
SENDTO=                              9
STARTMENU=                          11
MYMUSIC=                            13
MYVIDEOS=                           14
NETHOOD=                            19
FONTS=                              20
TEMPLATES=                          21
ALLUSERSSTARTMENU=                  22
```

这段代码定义了一个名为get的函数，它接受一个整型参数Folder，并返回以Folder为根目录的当前工作目录（即用户桌面）的路径。函数使用了_SHGetFolderPath函数，它是一个系统函数，可以在Windows平台上从指定目录开始获取指定路径。函数的参数包括：_HWND、_ctypes.c_int、_HANDLE、_DWORD、_LPCWSTR类型，分别表示窗口 handle、整数类型、文件 handle、无符号长整型和字符串类型。


```py
ALLUSERSPROGRAMS=                   23
ALLUSERSSTARTUP=                    24
ALLUSERSDESKTOP=                    25
APPLICATIONDATA=                    26
PRINTHOOD=                          27
LOCALSETTINGSAPPLICATIONDATA=       28
ALLUSERSFAVORITES=                  31
LOCALSETTINGSTEMPORARYINTERNETFILES=32
COOKIES=                            33
LOCALSETTINGSHISTORY=               34
ALLUSERSAPPLICATIONDATA=            35

def get(intFolder):
    _SHGetFolderPath.argtypes = [_HWND, _ctypes.c_int, _HANDLE, _DWORD, _LPCWSTR]
    auPathBuffer = _cub(_MAX_PATH)
    exit_code=_SHGetFolderPath(0, intFolder, 0, 0, auPathBuffer)
    return auPathBuffer.value


```

这段代码是一个Python程序，其主要作用是启动一个Python虚拟环境(即“STARTUP”)，并执行其中包含的一个名为“zeroNet.py”的脚本。

以下是程序的每一行解释：

1. `if __name__ == "__main__":`

这是一个条件语句，用于检查当前脚本是否是作为主程序运行的。如果它是一个主程序，那么程序下面的代码将被执行。

2. `import os`

这是Python的一个标准库函数，用于导入操作系统中的模块和函数。它将导入当前工作目录(即“os.getcwd()”)以及“os”模块中的所有定义。

3. `print(get(STARTUP))`

这是一行打印语句，用于打印获取到的启动文件路径。在这里，“get(STARTUP)”是一个Python库函数，用于从Python虚拟环境的所有启动文件中获取启动文件的一个列表。这个列表的第一个元素被打印出来。

4. `open(get(STARTUP)+"\\zeronet.cmd", "w").write("cd /D %s\r\nzeroNet.py" % os.getcwd())`

这是打开一个文件并写入文件内容的代码。在这里，“get(STARTUP)+"\\zeronet.cmd”是一个字符串，用于获取启动文件夹中包含“zeroNet.py”文件的路径。这个文件被打开并写入内容，其中包含当前工作目录的路径。

如果当前工作目录不是包含“zeroNet.py”文件的启动文件夹，那么将会抛出一个FileNotFoundError。


```py
if __name__ == "__main__":
	import os
	print(get(STARTUP))
	open(get(STARTUP)+"\\zeronet.cmd", "w").write("cd /D %s\r\nzeronet.py" % os.getcwd())
```

# `plugins/Trayicon/lib/__init__.py`

很抱歉，我无法解释该代码的作用，因为您没有提供代码。如果您能提供代码，我将非常乐意为您解释其作用。


```py

```

# `plugins/UiConfig/UiConfigPlugin.py`

这段代码的作用是加载一个名为"PluginManager"的插件，该插件可以用来管理插件。同时，它还加载了一个名为"Translate"的插件，该插件可以用来将文本翻译成多种语言。

具体来说，代码首先通过导入os和io模块，然后通过os.path.dirname(__file__)获取了插件目录的路径。接下来，代码通过判断_是否存在于locals()中来确定是否已经加载了插件。如果是，那么就加载了插件管理器插件，如果不是，那么就加载了Translate插件。

在加载插件管理器插件之后，代码就定义了一个名为"plugin_dir"的变量，该变量记录了插件目录的路径。接下来，代码通过os.walk(plugin_dir)遍历插件目录下的所有文件和子目录，并将插件的名称存储到一个名为"languages"的字典中。

最后，代码通过判断_是否存在于locals()中来确定是否已经加载了插件管理器插件。如果是，那么就加载了Translate插件，否则就插件管理器插件。


```py
import io
import os

from Plugin import PluginManager
from Config import config
from Translate import Translate
from util.Flag import flag


plugin_dir = os.path.dirname(__file__)

if "_" not in locals():
    _ = Translate(plugin_dir + "/languages/")


```

This is a class called `UiRequestPlugin` which is a plugin for the UiElement class in the Ui框架. This class extends the `UiRequestPlugin` class provided by the Ui framework.

The `actionWrapper` method is a wrapper method for the `action` method of the `UiRequestPlugin` class. It ensures that the `action` method is called on the `UiRequestPlugin` object, but any additional headers or arguments are passed to the `action` method.

The `actionUiMedia` method is a wrapper method for the `actionUiMedia` method of the `UiRequestPlugin` class. It ensures that the `actionUiMedia` method is called on the `UiRequestPlugin` object, but any additional arguments or headers are passed to the `actionUiMedia` method.

The `actionFile` method is a method that uses the `action` method to load the media file from the disk. It takes a file path as an argument and returns the `action` method, passing in the file path and an instance of `UiIO` class to read the file. If the file is a JavaScript file, it is assumed that it contains CSS stylesheets and assets and so is processed accordingly.


```py
@PluginManager.registerTo("UiRequest")
class UiRequestPlugin(object):
    def actionWrapper(self, path, extra_headers=None):
        if path.strip("/") != "Config":
            return super(UiRequestPlugin, self).actionWrapper(path, extra_headers)

        if not extra_headers:
            extra_headers = {}

        script_nonce = self.getScriptNonce()

        self.sendHeader(extra_headers=extra_headers, script_nonce=script_nonce)
        site = self.server.site_manager.get(config.homepage)
        return iter([super(UiRequestPlugin, self).renderWrapper(
            site, path, "uimedia/plugins/uiconfig/config.html",
            "Config", extra_headers, show_loadingscreen=False, script_nonce=script_nonce
        )])

    def actionUiMedia(self, path, *args, **kwargs):
        if path.startswith("/uimedia/plugins/uiconfig/"):
            file_path = path.replace("/uimedia/plugins/uiconfig/", plugin_dir + "/media/")
            if config.debug and (file_path.endswith("all.js") or file_path.endswith("all.css")):
                # If debugging merge *.css to all.css and *.js to all.js
                from Debug import DebugMedia
                DebugMedia.merge(file_path)

            if file_path.endswith("js"):
                data = _.translateData(open(file_path).read(), mode="js").encode("utf8")
            elif file_path.endswith("html"):
                data = _.translateData(open(file_path).read(), mode="html").encode("utf8")
            else:
                data = open(file_path, "rb").read()

            return self.actionFile(file_path, file_obj=io.BytesIO(data), file_size=len(data))
        else:
            return super(UiRequestPlugin, self).actionUiMedia(path)


```

这段代码定义了一个名为 `UiWebsocketPlugin` 的类，该类用于注册一个 WebSocket 插件到应用程序的管理员权限中。

具体来说，该类包含一个名为 `actionConfigList` 的方法，用于获取 WebSocket 插件的配置列表。在该方法中，首先定义了一个名为 `back` 的空字典，用于存储各个配置参数的备份值。然后，使用 `vars()` 函数获取 WebSocket 配置参数的变量，并更新 `config_values` 字典。接着，遍历 `config_values` 字典的每个键值对，判断该键是否属于 `config.keys_api_change_allowed` 数组，如果是，则执行以下操作：将参数值设置为 `config.parser.get_default(key)`，并设置为 `True` 的值，表示该参数值是待发送的，即 `pending` 状态。最后，将 `back` 字典的键值对添加到 `config_values` 中，并使用 `update()` 方法确保 `config_values` 中的键值对与 `config.keys_api_change_allowed` 数组中的键值对保持一致。

返回 `back` 字典，表示注册成功后 `back` 字典中存储的参数值。


```py
@PluginManager.registerTo("UiWebsocket")
class UiWebsocketPlugin(object):
    @flag.admin
    def actionConfigList(self, to):
        back = {}
        config_values = vars(config.arguments)
        config_values.update(config.pending_changes)
        for key, val in config_values.items():
            if key not in config.keys_api_change_allowed:
                continue
            is_pending = key in config.pending_changes
            if val is None and is_pending:
                val = config.parser.get_default(key)
            back[key] = {
                "value": val,
                "default": config.parser.get_default(key),
                "pending": is_pending
            }
        return back

```

# `plugins/UiConfig/__init__.py`

这段代码是一个Python脚本，它导入了名为"UiConfigPlugin"的模块。然而，它没有做任何其他事情，因为在该模块中没有定义任何函数或方法。所以，该代码的作用是导入了一个名为"UiConfigPlugin"的模块，但并不做任何其他事情。


```py
from . import UiConfigPlugin

```

# `plugins/UiConfig/media/js/all.js`

该代码定义了一个名为`Class`的JavaScript类，其作用是创建一个新的JavaScript类。该类实现了`trace`和`log`方法，用于输出调试信息。

具体来说，该类的`trace`方法是一个保护属性，在该方法中记录了类的调试信息，以便在需要时进行输出。`log`方法用于将日志信息输出到控制台或console。`logStart`和`logEnd`方法则是`log`方法的两种不同的实现方式，用于在日志信息中添加时间和启动时间戳。

类的实例可以通过调用`Class`构造函数来创建，并且`window.Class`是一个全局变量，用于将类的实例分配给`window`对象。


```py

/* ---- lib/Class.coffee ---- */


(function() {
  var Class,
    slice = [].slice;

  Class = (function() {
    function Class() {}

    Class.prototype.trace = true;

    Class.prototype.log = function() {
      var args;
      args = 1 <= arguments.length ? slice.call(arguments, 0) : [];
      if (!this.trace) {
        return;
      }
      if (typeof console === 'undefined') {
        return;
      }
      args.unshift("[" + this.constructor.name + "]");
      console.log.apply(console, args);
      return this;
    };

    Class.prototype.logStart = function() {
      var args, name;
      name = arguments[0], args = 2 <= arguments.length ? slice.call(arguments, 1) : [];
      if (!this.trace) {
        return;
      }
      this.logtimers || (this.logtimers = {});
      this.logtimers[name] = +(new Date);
      if (args.length > 0) {
        this.log.apply(this, ["" + name].concat(slice.call(args), ["(started)"]));
      }
      return this;
    };

    Class.prototype.logEnd = function() {
      var args, ms, name;
      name = arguments[0], args = 2 <= arguments.length ? slice.call(arguments, 1) : [];
      ms = +(new Date) - this.logtimers[name];
      this.log.apply(this, ["" + name].concat(slice.call(args), ["(Done in " + ms + "ms)"]));
      return this;
    };

    return Class;

  })();

  window.Class = Class;

}).call(this);

```

This is a JavaScript file that defines a `Promise` class. The `Promise` class is used to handle asynchronous operations in an efficient way. The class provides several methods and properties that make it easy to create and manage Promises.

The class has a `.resolve()` method that attempts to resolve the Promise with a truthy value. If the Promise is already resolved, the method returns `true`. If the Promise is not resolved, the method returns `false`. The example at the end of the file demonstrates how to use the `.resolve()` method to set the resolved value of a Promise.

The `.fail()` method fails the Promise with a falsy value. If the Promise is already resolved, the method returns `true`. If the Promise is not resolved, the method returns `false`. The example at the end of the file demonstrates how to use the `.fail()` method to set the resolved value of a Promise.

The `.then()` method is used to chain a callback function to the Promise. The callback function is passed an argument that will be passed to the callback function when the Promise is resolved. The example at the end of the file demonstrates how to use the `.then()` method to chain a callback function to a Promise.

The `.catch()` method is used to chain a callback function to the Promise. The callback function is passed a function that will be called with the Promise if the Promise is rejected. The example at the end of the file demonstrates how to use the `.catch()` method to chain a callback function to a Promise.

The `.end_promise()` method returns a Promise that resolves with the final Promise that is passed to it. The example at the end of the file demonstrates how to use the `.end_promise()` method to return a Promise that resolves with the final Promise.


```py
/* ---- lib/Promise.coffee ---- */


(function() {
  var Promise,
    slice = [].slice;

  Promise = (function() {
    Promise.when = function() {
      var args, fn, i, len, num_uncompleted, promise, task, task_id, tasks;
      tasks = 1 <= arguments.length ? slice.call(arguments, 0) : [];
      num_uncompleted = tasks.length;
      args = new Array(num_uncompleted);
      promise = new Promise();
      fn = function(task_id) {
        return task.then(function() {
          args[task_id] = Array.prototype.slice.call(arguments);
          num_uncompleted--;
          if (num_uncompleted === 0) {
            return promise.complete.apply(promise, args);
          }
        });
      };
      for (task_id = i = 0, len = tasks.length; i < len; task_id = ++i) {
        task = tasks[task_id];
        fn(task_id);
      }
      return promise;
    };

    function Promise() {
      this.resolved = false;
      this.end_promise = null;
      this.result = null;
      this.callbacks = [];
    }

    Promise.prototype.resolve = function() {
      var back, callback, i, len, ref;
      if (this.resolved) {
        return false;
      }
      this.resolved = true;
      this.data = arguments;
      if (!arguments.length) {
        this.data = [true];
      }
      this.result = this.data[0];
      ref = this.callbacks;
      for (i = 0, len = ref.length; i < len; i++) {
        callback = ref[i];
        back = callback.apply(callback, this.data);
      }
      if (this.end_promise) {
        return this.end_promise.resolve(back);
      }
    };

    Promise.prototype.fail = function() {
      return this.resolve(false);
    };

    Promise.prototype.then = function(callback) {
      if (this.resolved === true) {
        callback.apply(callback, this.data);
        return;
      }
      this.callbacks.push(callback);
      return this.end_promise = new Promise();
    };

    return Promise;

  })();

  window.Promise = Promise;


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

}).call(this);

```

该代码定义了几个方法，用于操作字符串。首先是 `startsWith` 方法，用于检查字符串是否以给定字符串开始。其次是 `endsWith` 方法，用于检查字符串是否以给定字符串结束。第三个方法是 `repeat` 方法，用于在给定次数下重复给定字符串。最后一个方法是 `isEmpty` 方法，用于判断对象或数组是否为空。

这些方法的实现主要依赖于对于字符串的基本操作，例如 `slice` 方法用于截取字符串的一部分，`Array.join` 方法用于将数组中的元素连接成一个字符串，以及 `Object.is` 方法用于判断对象是否为空。


```py
/* ---- lib/Prototypes.coffee ---- */


(function() {
  String.prototype.startsWith = function(s) {
    return this.slice(0, s.length) === s;
  };

  String.prototype.endsWith = function(s) {
    return s === '' || this.slice(-s.length) === s;
  };

  String.prototype.repeat = function(count) {
    return new Array(count + 1).join(this);
  };

  window.isEmpty = function(obj) {
    var key;
    for (key in obj) {
      return false;
    }
    return true;
  };

}).call(this);

```

This appears to be a JavaScript object that manages the rendering of a virtual DOM tree. It appears to have several methods for scheduling and canceling animations, stopping the animation, and stopping the animation of a specific animation. It also appears to have methods for creating and removing virtual DOM elements, and creating and removing projectmeta


```py
/* ---- lib/maquette.js ---- */


(function (root, factory) {
    if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define(['exports'], factory);
    } else if (typeof exports === 'object' && typeof exports.nodeName !== 'string') {
        // CommonJS
        factory(exports);
    } else {
        // Browser globals
        factory(root.maquette = {});
    }
}(this, function (exports) {
    'use strict';
    ;
    ;
    ;
    ;
    var NAMESPACE_W3 = 'http://www.w3.org/';
    var NAMESPACE_SVG = NAMESPACE_W3 + '2000/svg';
    var NAMESPACE_XLINK = NAMESPACE_W3 + '1999/xlink';
    // Utilities
    var emptyArray = [];
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
    // Hyperscript helper functions
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
    var toTextVNode = function (data) {
        return {
            vnodeSelector: '',
            properties: undefined,
            children: undefined,
            text: data.toString(),
            domNode: null
        };
    };
    var appendChildren = function (parentSelector, insertions, main) {
        for (var i = 0; i < insertions.length; i++) {
            var item = insertions[i];
            if (Array.isArray(item)) {
                appendChildren(parentSelector, item, main);
            } else {
                if (item !== null && item !== undefined) {
                    if (!item.hasOwnProperty('vnodeSelector')) {
                        item = toTextVNode(item);
                    }
                    main.push(item);
                }
            }
        }
    };
    // Render helper functions
    var missingTransition = function () {
        throw new Error('Provide a transitions object to the projectionOptions to do animations');
    };
    var DEFAULT_PROJECTION_OPTIONS = {
        namespace: undefined,
        eventHandlerInterceptor: undefined,
        styleApplyer: function (domNode, styleName, value) {
            // Provides a hook to add vendor prefixes for browsers that still need it.
            domNode.style[styleName] = value;
        },
        transitions: {
            enter: missingTransition,
            exit: missingTransition
        }
    };
    var applyDefaultProjectionOptions = function (projectorOptions) {
        return extend(DEFAULT_PROJECTION_OPTIONS, projectorOptions);
    };
    var checkStyleValue = function (styleValue) {
        if (typeof styleValue !== 'string') {
            throw new Error('Style values must be strings');
        }
    };
    var setProperties = function (domNode, properties, projectionOptions) {
        if (!properties) {
            return;
        }
        var eventHandlerInterceptor = projectionOptions.eventHandlerInterceptor;
        var propNames = Object.keys(properties);
        var propCount = propNames.length;
        for (var i = 0; i < propCount; i++) {
            var propName = propNames[i];
            /* tslint:disable:no-var-keyword: edge case */
            var propValue = properties[propName];
            /* tslint:enable:no-var-keyword */
            if (propName === 'className') {
                throw new Error('Property "className" is not supported, use "class".');
            } else if (propName === 'class') {
                if (domNode.className) {
                    // May happen if classes is specified before class
                    domNode.className += ' ' + propValue;
                } else {
                    domNode.className = propValue;
                }
            } else if (propName === 'classes') {
                // object with string keys and boolean values
                var classNames = Object.keys(propValue);
                var classNameCount = classNames.length;
                for (var j = 0; j < classNameCount; j++) {
                    var className = classNames[j];
                    if (propValue[className]) {
                        domNode.classList.add(className);
                    }
                }
            } else if (propName === 'styles') {
                // object with string keys and string (!) values
                var styleNames = Object.keys(propValue);
                var styleCount = styleNames.length;
                for (var j = 0; j < styleCount; j++) {
                    var styleName = styleNames[j];
                    var styleValue = propValue[styleName];
                    if (styleValue) {
                        checkStyleValue(styleValue);
                        projectionOptions.styleApplyer(domNode, styleName, styleValue);
                    }
                }
            } else if (propName === 'key') {
                continue;
            } else if (propValue === null || propValue === undefined) {
                continue;
            } else {
                var type = typeof propValue;
                if (type === 'function') {
                    if (propName.lastIndexOf('on', 0) === 0) {
                        if (eventHandlerInterceptor) {
                            propValue = eventHandlerInterceptor(propName, propValue, domNode, properties);    // intercept eventhandlers
                        }
                        if (propName === 'oninput') {
                            (function () {
                                // record the evt.target.value, because IE and Edge sometimes do a requestAnimationFrame between changing value and running oninput
                                var oldPropValue = propValue;
                                propValue = function (evt) {
                                    evt.target['oninput-value'] = evt.target.value;
                                    // may be HTMLTextAreaElement as well
                                    oldPropValue.apply(this, [evt]);
                                };
                            }());
                        }
                        domNode[propName] = propValue;
                    }
                } else if (type === 'string' && propName !== 'value' && propName !== 'innerHTML') {
                    if (projectionOptions.namespace === NAMESPACE_SVG && propName === 'href') {
                        domNode.setAttributeNS(NAMESPACE_XLINK, propName, propValue);
                    } else {
                        domNode.setAttribute(propName, propValue);
                    }
                } else {
                    domNode[propName] = propValue;
                }
            }
        }
    };
    var updateProperties = function (domNode, previousProperties, properties, projectionOptions) {
        if (!properties) {
            return;
        }
        var propertiesUpdated = false;
        var propNames = Object.keys(properties);
        var propCount = propNames.length;
        for (var i = 0; i < propCount; i++) {
            var propName = propNames[i];
            // assuming that properties will be nullified instead of missing is by design
            var propValue = properties[propName];
            var previousValue = previousProperties[propName];
            if (propName === 'class') {
                if (previousValue !== propValue) {
                    throw new Error('"class" property may not be updated. Use the "classes" property for conditional css classes.');
                }
            } else if (propName === 'classes') {
                var classList = domNode.classList;
                var classNames = Object.keys(propValue);
                var classNameCount = classNames.length;
                for (var j = 0; j < classNameCount; j++) {
                    var className = classNames[j];
                    var on = !!propValue[className];
                    var previousOn = !!previousValue[className];
                    if (on === previousOn) {
                        continue;
                    }
                    propertiesUpdated = true;
                    if (on) {
                        classList.add(className);
                    } else {
                        classList.remove(className);
                    }
                }
            } else if (propName === 'styles') {
                var styleNames = Object.keys(propValue);
                var styleCount = styleNames.length;
                for (var j = 0; j < styleCount; j++) {
                    var styleName = styleNames[j];
                    var newStyleValue = propValue[styleName];
                    var oldStyleValue = previousValue[styleName];
                    if (newStyleValue === oldStyleValue) {
                        continue;
                    }
                    propertiesUpdated = true;
                    if (newStyleValue) {
                        checkStyleValue(newStyleValue);
                        projectionOptions.styleApplyer(domNode, styleName, newStyleValue);
                    } else {
                        projectionOptions.styleApplyer(domNode, styleName, '');
                    }
                }
            } else {
                if (!propValue && typeof previousValue === 'string') {
                    propValue = '';
                }
                if (propName === 'value') {
                    if (domNode[propName] !== propValue && domNode['oninput-value'] !== propValue) {
                        domNode[propName] = propValue;
                        // Reset the value, even if the virtual DOM did not change
                        domNode['oninput-value'] = undefined;
                    }
                    // else do not update the domNode, otherwise the cursor position would be changed
                    if (propValue !== previousValue) {
                        propertiesUpdated = true;
                    }
                } else if (propValue !== previousValue) {
                    var type = typeof propValue;
                    if (type === 'function') {
                        throw new Error('Functions may not be updated on subsequent renders (property: ' + propName + '). Hint: declare event handler functions outside the render() function.');
                    }
                    if (type === 'string' && propName !== 'innerHTML') {
                        if (projectionOptions.namespace === NAMESPACE_SVG && propName === 'href') {
                            domNode.setAttributeNS(NAMESPACE_XLINK, propName, propValue);
                        } else {
                            domNode.setAttribute(propName, propValue);
                        }
                    } else {
                        if (domNode[propName] !== propValue) {
                            domNode[propName] = propValue;
                        }
                    }
                    propertiesUpdated = true;
                }
            }
        }
        return propertiesUpdated;
    };
    var findIndexOfChild = function (children, sameAs, start) {
        if (sameAs.vnodeSelector !== '') {
            // Never scan for text-nodes
            for (var i = start; i < children.length; i++) {
                if (same(children[i], sameAs)) {
                    return i;
                }
            }
        }
        return -1;
    };
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
    var checkDistinguishable = function (childNodes, indexToCheck, parentVNode, operation) {
        var childNode = childNodes[indexToCheck];
        if (childNode.vnodeSelector === '') {
            return;    // Text nodes need not be distinguishable
        }
        var properties = childNode.properties;
        var key = properties ? properties.key === undefined ? properties.bind : properties.key : undefined;
        if (!key) {
            for (var i = 0; i < childNodes.length; i++) {
                if (i !== indexToCheck) {
                    var node = childNodes[i];
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
    var createDom;
    var updateDom;
    var updateChildren = function (vnode, domNode, oldChildren, newChildren, projectionOptions) {
        if (oldChildren === newChildren) {
            return false;
        }
        oldChildren = oldChildren || emptyArray;
        newChildren = newChildren || emptyArray;
        var oldChildrenLength = oldChildren.length;
        var newChildrenLength = newChildren.length;
        var transitions = projectionOptions.transitions;
        var oldIndex = 0;
        var newIndex = 0;
        var i;
        var textUpdated = false;
        while (newIndex < newChildrenLength) {
            var oldChild = oldIndex < oldChildrenLength ? oldChildren[oldIndex] : undefined;
            var newChild = newChildren[newIndex];
            if (oldChild !== undefined && same(oldChild, newChild)) {
                textUpdated = updateDom(oldChild, newChild, projectionOptions) || textUpdated;
                oldIndex++;
            } else {
                var findOldIndex = findIndexOfChild(oldChildren, newChild, oldIndex + 1);
                if (findOldIndex >= 0) {
                    // Remove preceding missing children
                    for (i = oldIndex; i < findOldIndex; i++) {
                        nodeToRemove(oldChildren[i], transitions);
                        checkDistinguishable(oldChildren, i, vnode, 'removed');
                    }
                    textUpdated = updateDom(oldChildren[findOldIndex], newChild, projectionOptions) || textUpdated;
                    oldIndex = findOldIndex + 1;
                } else {
                    // New child
                    createDom(newChild, domNode, oldIndex < oldChildrenLength ? oldChildren[oldIndex].domNode : undefined, projectionOptions);
                    nodeAdded(newChild, transitions);
                    checkDistinguishable(newChildren, newIndex, vnode, 'added');
                }
            }
            newIndex++;
        }
        if (oldChildrenLength > oldIndex) {
            // Remove child fragments
            for (i = oldIndex; i < oldChildrenLength; i++) {
                nodeToRemove(oldChildren[i], transitions);
                checkDistinguishable(oldChildren, i, vnode, 'removed');
            }
        }
        return textUpdated;
    };
    var addChildren = function (domNode, children, projectionOptions) {
        if (!children) {
            return;
        }
        for (var i = 0; i < children.length; i++) {
            createDom(children[i], domNode, undefined, projectionOptions);
        }
    };
    var initPropertiesAndChildren = function (domNode, vnode, projectionOptions) {
        addChildren(domNode, vnode.children, projectionOptions);
        // children before properties, needed for value property of <select>.
        if (vnode.text) {
            domNode.textContent = vnode.text;
        }
        setProperties(domNode, vnode.properties, projectionOptions);
        if (vnode.properties && vnode.properties.afterCreate) {
            vnode.properties.afterCreate(domNode, projectionOptions, vnode.vnodeSelector, vnode.properties, vnode.children);
        }
    };
    createDom = function (vnode, parentNode, insertBefore, projectionOptions) {
        var domNode, i, c, start = 0, type, found;
        var vnodeSelector = vnode.vnodeSelector;
        if (vnodeSelector === '') {
            domNode = vnode.domNode = document.createTextNode(vnode.text);
            if (insertBefore !== undefined) {
                parentNode.insertBefore(domNode, insertBefore);
            } else {
                parentNode.appendChild(domNode);
            }
        } else {
            for (i = 0; i <= vnodeSelector.length; ++i) {
                c = vnodeSelector.charAt(i);
                if (i === vnodeSelector.length || c === '.' || c === '#') {
                    type = vnodeSelector.charAt(start - 1);
                    found = vnodeSelector.slice(start, i);
                    if (type === '.') {
                        domNode.classList.add(found);
                    } else if (type === '#') {
                        domNode.id = found;
                    } else {
                        if (found === 'svg') {
                            projectionOptions = extend(projectionOptions, { namespace: NAMESPACE_SVG });
                        }
                        if (projectionOptions.namespace !== undefined) {
                            domNode = vnode.domNode = document.createElementNS(projectionOptions.namespace, found);
                        } else {
                            domNode = vnode.domNode = document.createElement(found);
                        }
                        if (insertBefore !== undefined) {
                            parentNode.insertBefore(domNode, insertBefore);
                        } else {
                            parentNode.appendChild(domNode);
                        }
                    }
                    start = i + 1;
                }
            }
            initPropertiesAndChildren(domNode, vnode, projectionOptions);
        }
    };
    updateDom = function (previous, vnode, projectionOptions) {
        var domNode = previous.domNode;
        var textUpdated = false;
        if (previous === vnode) {
            return false;    // By contract, VNode objects may not be modified anymore after passing them to maquette
        }
        var updated = false;
        if (vnode.vnodeSelector === '') {
            if (vnode.text !== previous.text) {
                var newVNode = document.createTextNode(vnode.text);
                domNode.parentNode.replaceChild(newVNode, domNode);
                vnode.domNode = newVNode;
                textUpdated = true;
                return textUpdated;
            }
        } else {
            if (vnode.vnodeSelector.lastIndexOf('svg', 0) === 0) {
                projectionOptions = extend(projectionOptions, { namespace: NAMESPACE_SVG });
            }
            if (previous.text !== vnode.text) {
                updated = true;
                if (vnode.text === undefined) {
                    domNode.removeChild(domNode.firstChild);    // the only textnode presumably
                } else {
                    domNode.textContent = vnode.text;
                }
            }
            updated = updateChildren(vnode, domNode, previous.children, vnode.children, projectionOptions) || updated;
            updated = updateProperties(domNode, previous.properties, vnode.properties, projectionOptions) || updated;
            if (vnode.properties && vnode.properties.afterUpdate) {
                vnode.properties.afterUpdate(domNode, projectionOptions, vnode.vnodeSelector, vnode.properties, vnode.children);
            }
        }
        if (updated && vnode.properties && vnode.properties.updateAnimation) {
            vnode.properties.updateAnimation(domNode, vnode.properties, previous.properties);
        }
        vnode.domNode = previous.domNode;
        return textUpdated;
    };
    var createProjection = function (vnode, projectionOptions) {
        return {
            update: function (updatedVnode) {
                if (vnode.vnodeSelector !== updatedVnode.vnodeSelector) {
                    throw new Error('The selector for the root VNode may not be changed. (consider using dom.merge and add one extra level to the virtual DOM)');
                }
                updateDom(vnode, updatedVnode, projectionOptions);
                vnode = updatedVnode;
            },
            domNode: vnode.domNode
        };
    };
    ;
    // The other two parameters are not added here, because the Typescript compiler creates surrogate code for desctructuring 'children'.
    exports.h = function (selector) {
        var properties = arguments[1];
        if (typeof selector !== 'string') {
            throw new Error();
        }
        var childIndex = 1;
        if (properties && !properties.hasOwnProperty('vnodeSelector') && !Array.isArray(properties) && typeof properties === 'object') {
            childIndex = 2;
        } else {
            // Optional properties argument was omitted
            properties = undefined;
        }
        var text = undefined;
        var children = undefined;
        var argsLength = arguments.length;
        // Recognize a common special case where there is only a single text node
        if (argsLength === childIndex + 1) {
            var onlyChild = arguments[childIndex];
            if (typeof onlyChild === 'string') {
                text = onlyChild;
            } else if (onlyChild !== undefined && onlyChild.length === 1 && typeof onlyChild[0] === 'string') {
                text = onlyChild[0];
            }
        }
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
        return {
            vnodeSelector: selector,
            properties: properties,
            children: children,
            text: text === '' ? undefined : text,
            domNode: null
        };
    };
    /**
 * Contains simple low-level utility functions to manipulate the real DOM.
 */
    exports.dom = {
        /**
     * Creates a real DOM tree from `vnode`. The [[Projection]] object returned will contain the resulting DOM Node in
     * its [[Projection.domNode|domNode]] property.
     * This is a low-level method. Users wil typically use a [[Projector]] instead.
     * @param vnode - The root of the virtual DOM tree that was created using the [[h]] function. NOTE: [[VNode]]
     * objects may only be rendered once.
     * @param projectionOptions - Options to be used to create and update the projection.
     * @returns The [[Projection]] which also contains the DOM Node that was created.
     */
        create: function (vnode, projectionOptions) {
            projectionOptions = applyDefaultProjectionOptions(projectionOptions);
            createDom(vnode, document.createElement('div'), undefined, projectionOptions);
            return createProjection(vnode, projectionOptions);
        },
        /**
     * Appends a new childnode to the DOM which is generated from a [[VNode]].
     * This is a low-level method. Users wil typically use a [[Projector]] instead.
     * @param parentNode - The parent node for the new childNode.
     * @param vnode - The root of the virtual DOM tree that was created using the [[h]] function. NOTE: [[VNode]]
     * objects may only be rendered once.
     * @param projectionOptions - Options to be used to create and update the [[Projection]].
     * @returns The [[Projection]] that was created.
     */
        append: function (parentNode, vnode, projectionOptions) {
            projectionOptions = applyDefaultProjectionOptions(projectionOptions);
            createDom(vnode, parentNode, undefined, projectionOptions);
            return createProjection(vnode, projectionOptions);
        },
        /**
     * Inserts a new DOM node which is generated from a [[VNode]].
     * This is a low-level method. Users wil typically use a [[Projector]] instead.
     * @param beforeNode - The node that the DOM Node is inserted before.
     * @param vnode - The root of the virtual DOM tree that was created using the [[h]] function.
     * NOTE: [[VNode]] objects may only be rendered once.
     * @param projectionOptions - Options to be used to create and update the projection, see [[createProjector]].
     * @returns The [[Projection]] that was created.
     */
        insertBefore: function (beforeNode, vnode, projectionOptions) {
            projectionOptions = applyDefaultProjectionOptions(projectionOptions);
            createDom(vnode, beforeNode.parentNode, beforeNode, projectionOptions);
            return createProjection(vnode, projectionOptions);
        },
        /**
     * Merges a new DOM node which is generated from a [[VNode]] with an existing DOM Node.
     * This means that the virtual DOM and the real DOM will have one overlapping element.
     * Therefore the selector for the root [[VNode]] will be ignored, but its properties and children will be applied to the Element provided.
     * This is a low-level method. Users wil typically use a [[Projector]] instead.
     * @param domNode - The existing element to adopt as the root of the new virtual DOM. Existing attributes and childnodes are preserved.
     * @param vnode - The root of the virtual DOM tree that was created using the [[h]] function. NOTE: [[VNode]] objects
     * may only be rendered once.
     * @param projectionOptions - Options to be used to create and update the projection, see [[createProjector]].
     * @returns The [[Projection]] that was created.
     */
        merge: function (element, vnode, projectionOptions) {
            projectionOptions = applyDefaultProjectionOptions(projectionOptions);
            vnode.domNode = element;
            initPropertiesAndChildren(element, vnode, projectionOptions);
            return createProjection(vnode, projectionOptions);
        }
    };
    /**
 * Creates a [[CalculationCache]] object, useful for caching [[VNode]] trees.
 * In practice, caching of [[VNode]] trees is not needed, because achieving 60 frames per second is almost never a problem.
 * For more information, see [[CalculationCache]].
 *
 * @param <Result> The type of the value that is cached.
 */
    exports.createCache = function () {
        var cachedInputs = undefined;
        var cachedOutcome = undefined;
        var result = {
            invalidate: function () {
                cachedOutcome = undefined;
                cachedInputs = undefined;
            },
            result: function (inputs, calculation) {
                if (cachedInputs) {
                    for (var i = 0; i < inputs.length; i++) {
                        if (cachedInputs[i] !== inputs[i]) {
                            cachedOutcome = undefined;
                        }
                    }
                }
                if (!cachedOutcome) {
                    cachedOutcome = calculation();
                    cachedInputs = inputs;
                }
                return cachedOutcome;
            }
        };
        return result;
    };
    /**
 * Creates a {@link Mapping} instance that keeps an array of result objects synchronized with an array of source objects.
 * See {@link http://maquettejs.org/docs/arrays.html|Working with arrays}.
 *
 * @param <Source>       The type of source items. A database-record for instance.
 * @param <Target>       The type of target items. A [[Component]] for instance.
 * @param getSourceKey   `function(source)` that must return a key to identify each source object. The result must either be a string or a number.
 * @param createResult   `function(source, index)` that must create a new result object from a given source. This function is identical
 *                       to the `callback` argument in `Array.map(callback)`.
 * @param updateResult   `function(source, target, index)` that updates a result to an updated source.
 */
    exports.createMapping = function (getSourceKey, createResult, updateResult) {
        var keys = [];
        var results = [];
        return {
            results: results,
            map: function (newSources) {
                var newKeys = newSources.map(getSourceKey);
                var oldTargets = results.slice();
                var oldIndex = 0;
                for (var i = 0; i < newSources.length; i++) {
                    var source = newSources[i];
                    var sourceKey = newKeys[i];
                    if (sourceKey === keys[oldIndex]) {
                        results[i] = oldTargets[oldIndex];
                        updateResult(source, oldTargets[oldIndex], i);
                        oldIndex++;
                    } else {
                        var found = false;
                        for (var j = 1; j < keys.length; j++) {
                            var searchIndex = (oldIndex + j) % keys.length;
                            if (keys[searchIndex] === sourceKey) {
                                results[i] = oldTargets[searchIndex];
                                updateResult(newSources[i], oldTargets[searchIndex], i);
                                oldIndex = searchIndex + 1;
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            results[i] = createResult(source, i);
                        }
                    }
                }
                results.length = newSources.length;
                keys = newKeys;
            }
        };
    };
    /**
 * Creates a [[Projector]] instance using the provided projectionOptions.
 *
 * For more information, see [[Projector]].
 *
 * @param projectionOptions   Options that influence how the DOM is rendered and updated.
 */
    exports.createProjector = function (projectorOptions) {
        var projector;
        var projectionOptions = applyDefaultProjectionOptions(projectorOptions);
        projectionOptions.eventHandlerInterceptor = function (propertyName, eventHandler, domNode, properties) {
            return function () {
                // intercept function calls (event handlers) to do a render afterwards.
                projector.scheduleRender();
                return eventHandler.apply(properties.bind || this, arguments);
            };
        };
        var renderCompleted = true;
        var scheduled;
        var stopped = false;
        var projections = [];
        var renderFunctions = [];
        // matches the projections array
        var doRender = function () {
            scheduled = undefined;
            if (!renderCompleted) {
                return;    // The last render threw an error, it should be logged in the browser console.
            }
            renderCompleted = false;
            for (var i = 0; i < projections.length; i++) {
                var updatedVnode = renderFunctions[i]();
                projections[i].update(updatedVnode);
            }
            renderCompleted = true;
        };
        projector = {
            scheduleRender: function () {
                if (!scheduled && !stopped) {
                    scheduled = requestAnimationFrame(doRender);
                }
            },
            stop: function () {
                if (scheduled) {
                    cancelAnimationFrame(scheduled);
                    scheduled = undefined;
                }
                stopped = true;
            },
            resume: function () {
                stopped = false;
                renderCompleted = true;
                projector.scheduleRender();
            },
            append: function (parentNode, renderMaquetteFunction) {
                projections.push(exports.dom.append(parentNode, renderMaquetteFunction(), projectionOptions));
                renderFunctions.push(renderMaquetteFunction);
            },
            insertBefore: function (beforeNode, renderMaquetteFunction) {
                projections.push(exports.dom.insertBefore(beforeNode, renderMaquetteFunction(), projectionOptions));
                renderFunctions.push(renderMaquetteFunction);
            },
            merge: function (domNode, renderMaquetteFunction) {
                projections.push(exports.dom.merge(domNode, renderMaquetteFunction(), projectionOptions));
                renderFunctions.push(renderMaquetteFunction);
            },
            replace: function (domNode, renderMaquetteFunction) {
                var vnode = renderMaquetteFunction();
                createDom(vnode, domNode.parentNode, domNode, projectionOptions);
                domNode.parentNode.removeChild(domNode);
                projections.push(createProjection(vnode, projectionOptions));
                renderFunctions.push(renderMaquetteFunction);
            },
            detach: function (renderMaquetteFunction) {
                for (var i = 0; i < renderFunctions.length; i++) {
                    if (renderFunctions[i] === renderMaquetteFunction) {
                        renderFunctions.splice(i, 1);
                        return projections.splice(i, 1)[0];
                    }
                }
                throw new Error('renderMaquetteFunction was not found');
            }
        };
        return projector;
    };
}));


```

This looks like a JavaScript class called `Animation`. It appears to be a simple utility class for animating elements, and it includes several methods for adding and removing visible classes from elements.

The `Animation` class has a number of methods for adding visible classes to elements:

* `addVisibleClass(elem, props)`: adds the `visible` class to the element and applies any specified properties to it.
* `removeVisibleClass(elem, remove_func)`: removes the `visible` class from the element and applies any specified functions to it.

The `Animation` class also has methods for adding and removing the `animate` class from an element:

* `addEventListener(event, listener, false)`: adds an event listener to the element for the specified event, with the specified `listener` function.
* `removeEventListener(event, listener, false)`: removes an event listener from the element for the specified event.

It appears that the `Animation` class uses some basic CSS transitions to create animations when elements transition into the `visible` class. For example, when an element transitions into the `visible` class, the `transform` property is set to an anime-looking rotation, which is likely what creates the animation.

Overall, the `Animation` class seems like a simple and useful class for animating elements in the browser.


```py
/* ---- utils/Animation.coffee ---- */


(function() {
  var Animation;

  Animation = (function() {
    function Animation() {}

    Animation.prototype.slideDown = function(elem, props) {
      var cstyle, h, margin_bottom, margin_top, padding_bottom, padding_top, transition;
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
      return elem.addEventListener("transitionend", function() {
        elem.classList.remove("animate-inout");
        elem.style.transition = elem.style.transform = elem.style.opacity = elem.style.height = null;
        elem.style.boxSizing = elem.style.marginTop = elem.style.marginBottom = null;
        elem.style.paddingTop = elem.style.paddingBottom = elem.style.overflow = null;
        return elem.removeEventListener("transitionend", arguments.callee, false);
      });
    };

    Animation.prototype.slideUp = function(elem, remove_func, props) {
      if (elem.offsetTop > 1000) {
        return remove_func();
      }
      elem.className += " animate-back";
      elem.style.boxSizing = "border-box";
      elem.style.height = elem.offsetHeight + "px";
      elem.style.overflow = "hidden";
      elem.style.transform = "scale(1)";
      elem.style.opacity = "1";
      elem.style.pointerEvents = "none";
      setTimeout((function() {
        elem.style.height = "0px";
        elem.style.marginTop = "0px";
        elem.style.marginBottom = "0px";
        elem.style.paddingTop = "0px";
        elem.style.paddingBottom = "0px";
        elem.style.transform = "scale(0.8)";
        elem.style.borderTopWidth = "0px";
        elem.style.borderBottomWidth = "0px";
        return elem.style.opacity = "0";
      }), 1);
      return elem.addEventListener("transitionend", function(e) {
        if (e.propertyName === "opacity" || e.elapsedTime >= 0.6) {
          elem.removeEventListener("transitionend", arguments.callee, false);
          return remove_func();
        }
      });
    };

    Animation.prototype.slideUpInout = function(elem, remove_func, props) {
      elem.className += " animate-inout";
      elem.style.boxSizing = "border-box";
      elem.style.height = elem.offsetHeight + "px";
      elem.style.overflow = "hidden";
      elem.style.transform = "scale(1)";
      elem.style.opacity = "1";
      elem.style.pointerEvents = "none";
      setTimeout((function() {
        elem.style.height = "0px";
        elem.style.marginTop = "0px";
        elem.style.marginBottom = "0px";
        elem.style.paddingTop = "0px";
        elem.style.paddingBottom = "0px";
        elem.style.transform = "scale(0.8)";
        elem.style.borderTopWidth = "0px";
        elem.style.borderBottomWidth = "0px";
        return elem.style.opacity = "0";
      }), 1);
      return elem.addEventListener("transitionend", function(e) {
        if (e.propertyName === "opacity" || e.elapsedTime >= 0.6) {
          elem.removeEventListener("transitionend", arguments.callee, false);
          return remove_func();
        }
      });
    };

    Animation.prototype.showRight = function(elem, props) {
      elem.className += " animate";
      elem.style.opacity = 0;
      elem.style.transform = "TranslateX(-20px) Scale(1.01)";
      setTimeout((function() {
        elem.style.opacity = 1;
        return elem.style.transform = "TranslateX(0px) Scale(1)";
      }), 1);
      return elem.addEventListener("transitionend", function() {
        elem.classList.remove("animate");
        return elem.style.transform = elem.style.opacity = null;
      });
    };

    Animation.prototype.show = function(elem, props) {
      var delay, ref;
      delay = ((ref = arguments[arguments.length - 2]) != null ? ref.delay : void 0) * 1000 || 1;
      elem.style.opacity = 0;
      setTimeout((function() {
        return elem.className += " animate";
      }), 1);
      setTimeout((function() {
        return elem.style.opacity = 1;
      }), delay);
      return elem.addEventListener("transitionend", function() {
        elem.classList.remove("animate");
        elem.style.opacity = null;
        return elem.removeEventListener("transitionend", arguments.callee, false);
      });
    };

    Animation.prototype.hide = function(elem, remove_func, props) {
      var delay, ref;
      delay = ((ref = arguments[arguments.length - 2]) != null ? ref.delay : void 0) * 1000 || 1;
      elem.className += " animate";
      setTimeout((function() {
        return elem.style.opacity = 0;
      }), delay);
      return elem.addEventListener("transitionend", function(e) {
        if (e.propertyName === "opacity") {
          return remove_func();
        }
      });
    };

    Animation.prototype.addVisibleClass = function(elem, props) {
      return setTimeout(function() {
        return elem.classList.add("visible");
      });
    };

    return Animation;

  })();

  window.Animation = new Animation();

}).call(this);

```

This is a JavaScript class called `ZeroFrame` that implements a simple HTTP server using the Node.js `http` module. The server can handle a few different commands, including `cmd ping`, `cmd wrapOpenedWebsocket`, `cmd wrapClosedWebsocket`, `cmd response`, and `cmd send`.

The `ZeroFrame` class defines a number of methods, including `cmd` which takes a command and a block of parameters, and returns a message containing the result of the command. The `response` method returns a response to the specified command with the specified result.

The `onOpenWebsocket` and `onCloseWebsocket` methods are used to log a message if the websocket is opened or closed, respectively.

The `send` method sends a message with the specified `to` address and a callback function that will be called with the response. If a callback function is not provided, the message is sent as a push to the `ZeroFrame` instance's `queue`． The `queue` is a simple queue data structure that implements the `push` and `pop` methods.

The `wrapper_nonce` property is used to generate a unique nonce for each message, which is added to the `params` property before sending the message. The `wrapper_openssl_nonce` property is used to generate a nonce that is generated by an OpenSSL `secp256k1` hash.

The `onRequest` method is used to log a message if the server receives a `cmd` message from a client, and a `cmd` message with a `params` property.

The `cmd` method is used to handle the `cmd ping` command. It takes a `to` address and a callback function that is called with the response.

The `response` method is used to handle the `response` and `cmd response` commands. It takes a `result` object and returns a response to the specified `cmd` in the format `{ "status": "success", "message": "response message" }`.

The `cmd` method is used to handle the `cmd wrapOpenedWebsocket`, `cmd wrapClosedWebsocket`, and `cmd response` commands.

The `send` method sends a message with the specified `to` address and a callback function that will be called with the response. If a callback function is not provided, the message is sent as a push to the `ZeroFrame` instance's `queue`． The `queue` is a simple queue data structure that implements the `push` and `pop` methods.


```py
/* ---- utils/Dollar.coffee ---- */


(function() {
  window.$ = function(selector) {
    if (selector.startsWith("#")) {
      return document.getElementById(selector.replace("#", ""));
    }
  };

}).call(this);

/* ---- utils/ZeroFrame.coffee ---- */


(function() {
  var ZeroFrame,
    bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; },
    extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
    hasProp = {}.hasOwnProperty;

  ZeroFrame = (function(superClass) {
    extend(ZeroFrame, superClass);

    function ZeroFrame(url) {
      this.onCloseWebsocket = bind(this.onCloseWebsocket, this);
      this.onOpenWebsocket = bind(this.onOpenWebsocket, this);
      this.onRequest = bind(this.onRequest, this);
      this.onMessage = bind(this.onMessage, this);
      this.url = url;
      this.waiting_cb = {};
      this.wrapper_nonce = document.location.href.replace(/.*wrapper_nonce=([A-Za-z0-9]+).*/, "$1");
      this.connect();
      this.next_message_id = 1;
      this.history_state = {};
      this.init();
    }

    ZeroFrame.prototype.init = function() {
      return this;
    };

    ZeroFrame.prototype.connect = function() {
      this.target = window.parent;
      window.addEventListener("message", this.onMessage, false);
      this.cmd("innerReady");
      window.addEventListener("beforeunload", (function(_this) {
        return function(e) {
          _this.log("save scrollTop", window.pageYOffset);
          _this.history_state["scrollTop"] = window.pageYOffset;
          return _this.cmd("wrapperReplaceState", [_this.history_state, null]);
        };
      })(this));
      return this.cmd("wrapperGetState", [], (function(_this) {
        return function(state) {
          if (state != null) {
            _this.history_state = state;
          }
          _this.log("restore scrollTop", state, window.pageYOffset);
          if (window.pageYOffset === 0 && state) {
            return window.scroll(window.pageXOffset, state.scrollTop);
          }
        };
      })(this));
    };

    ZeroFrame.prototype.onMessage = function(e) {
      var cmd, message;
      message = e.data;
      cmd = message.cmd;
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

    ZeroFrame.prototype.onRequest = function(cmd, message) {
      return this.log("Unknown request", message);
    };

    ZeroFrame.prototype.response = function(to, result) {
      return this.send({
        "cmd": "response",
        "to": to,
        "result": result
      });
    };

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

    ZeroFrame.prototype.send = function(message, cb) {
      if (cb == null) {
        cb = null;
      }
      message.wrapper_nonce = this.wrapper_nonce;
      message.id = this.next_message_id;
      this.next_message_id += 1;
      this.target.postMessage(message, "*");
      if (cb) {
        return this.waiting_cb[message.id] = cb;
      }
    };

    ZeroFrame.prototype.onOpenWebsocket = function() {
      return this.log("Websocket open");
    };

    ZeroFrame.prototype.onCloseWebsocket = function() {
      return this.log("Websocket close");
    };

    return ZeroFrame;

  })(Class);

  window.ZeroFrame = ZeroFrame;

}).call(this);

```

This code appears to be setting up a config storage system for a web application. The `ConfigStorage` class is defined with a `createSection` method that takes a title as an argument and returns an object containing a `title` property and an array of `items` property. The `items` property is an array of objects, each representing a single configuration option with a `title` property and a `value` property.

The code also imports the `Orm` class from the `mysql2` package and the `IndexedObject` class from the `免疫系统` package.

It is unclear from this code snippet what the specific implementation of the config storage system is, but it appears to be setting up a simple configuration system that allows users to store and retrieve configuration options for the web application.


```py
/* ---- ConfigStorage.coffee ---- */


(function() {
  var ConfigStorage,
    bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; },
    extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
    hasProp = {}.hasOwnProperty;

  ConfigStorage = (function(superClass) {
    extend(ConfigStorage, superClass);

    function ConfigStorage(config) {
      this.config = config;
      this.createSection = bind(this.createSection, this);
      this.items = [];
      this.createSections();
      this.setValues(this.config);
    }

    ConfigStorage.prototype.setValues = function(values) {
      var i, item, len, ref, results, section;
      ref = this.items;
      results = [];
      for (i = 0, len = ref.length; i < len; i++) {
        section = ref[i];
        results.push((function() {
          var j, len1, ref1, results1;
          ref1 = section.items;
          results1 = [];
          for (j = 0, len1 = ref1.length; j < len1; j++) {
            item = ref1[j];
            if (!values[item.key]) {
              continue;
            }
            item.value = this.formatValue(values[item.key].value);
            item["default"] = this.formatValue(values[item.key]["default"]);
            item.pending = values[item.key].pending;
            results1.push(values[item.key].item = item);
          }
          return results1;
        }).call(this));
      }
      return results;
    };

    ConfigStorage.prototype.formatValue = function(value) {
      if (!value) {
        return false;
      } else if (typeof value === "object") {
        return value.join("\n");
      } else if (typeof value === "number") {
        return value.toString();
      } else {
        return value;
      }
    };

    ConfigStorage.prototype.deformatValue = function(value, type) {
      if (type === "object" && typeof value === "string") {
        if (!value.length) {
          return value = null;
        } else {
          return value.split("\n");
        }
      }
      if (type === "boolean" && !value) {
        return false;
      } else if (type === "number") {
        if (typeof value === "number") {
          return value.toString();
        } else if (!value) {
          return "0";
        } else {
          return value;
        }
      } else {
        return value;
      }
    };

    ConfigStorage.prototype.createSections = function() {
      var section;
      section = this.createSection("Web Interface");
      section.items.push({
        key: "open_browser",
        title: "Open web browser on ZeroNet startup",
        type: "checkbox"
      });
      section = this.createSection("Network");
      section.items.push({
        key: "offline",
        title: "Offline mode",
        type: "checkbox",
        description: "Disable network communication."
      });
      section.items.push({
        key: "fileserver_ip_type",
        title: "File server network",
        type: "select",
        options: [
          {
            title: "IPv4",
            value: "ipv4"
          }, {
            title: "IPv6",
            value: "ipv6"
          }, {
            title: "Dual (IPv4 & IPv6)",
            value: "dual"
          }
        ],
        description: "Accept incoming peers using IPv4 or IPv6 address. (default: dual)"
      });
      section.items.push({
        key: "fileserver_port",
        title: "File server port",
        type: "text",
        valid_pattern: /[0-9]*/,
        description: "Other peers will use this port to reach your served sites. (default: randomize)"
      });
      section.items.push({
        key: "ip_external",
        title: "File server external ip",
        type: "textarea",
        placeholder: "Detect automatically",
        description: "Your file server is accessible on these ips. (default: detect automatically)"
      });
      section.items.push({
        title: "Tor",
        key: "tor",
        type: "select",
        options: [
          {
            title: "Disable",
            value: "disable"
          }, {
            title: "Enable",
            value: "enable"
          }, {
            title: "Always",
            value: "always"
          }
        ],
        description: ["Disable: Don't connect to peers on Tor network", h("br"), "Enable: Only use Tor for Tor network peers", h("br"), "Always: Use Tor for every connections to hide your IP address (slower)"]
      });
      section.items.push({
        title: "Use Tor bridges",
        key: "tor_use_bridges",
        type: "checkbox",
        description: "Use obfuscated bridge relays to avoid network level Tor block (even slower)",
        isHidden: function() {
          return !Page.server_info.tor_has_meek_bridges;
        }
      });
      section.items.push({
        title: "Trackers",
        key: "trackers",
        type: "textarea",
        description: "Discover new peers using these adresses"
      });
      section.items.push({
        title: "Trackers files",
        key: "trackers_file",
        type: "textarea",
        description: "Load additional list of torrent trackers dynamically, from a file",
        placeholder: "Eg.: {data_dir}/trackers.json",
        value_pos: "fullwidth"
      });
      section.items.push({
        title: "Proxy for tracker connections",
        key: "trackers_proxy",
        type: "select",
        options: [
          {
            title: "Custom",
            value: ""
          }, {
            title: "Tor",
            value: "tor"
          }, {
            title: "Disable",
            value: "disable"
          }
        ],
        isHidden: function() {
          return Page.values["tor"] === "always";
        }
      });
      section.items.push({
        title: "Custom socks proxy address for trackers",
        key: "trackers_proxy",
        type: "text",
        placeholder: "Eg.: 127.0.0.1:1080",
        value_pos: "fullwidth",
        valid_pattern: /.+:[0-9]+/,
        isHidden: (function(_this) {
          return function() {
            var ref;
            return (ref = Page.values["trackers_proxy"]) === "tor" || ref === "disable";
          };
        })(this)
      });
      section = this.createSection("Performance");
      section.items.push({
        key: "log_level",
        title: "Level of logging to file",
        type: "select",
        options: [
          {
            title: "Everything",
            value: "DEBUG"
          }, {
            title: "Only important messages",
            value: "INFO"
          }, {
            title: "Only errors",
            value: "ERROR"
          }
        ]
      });
      section.items.push({
        key: "threads_fs_read",
        title: "Threads for async file system reads",
        type: "select",
        options: [
          {
            title: "Sync read",
            value: 0
          }, {
            title: "1 thread",
            value: 1
          }, {
            title: "2 threads",
            value: 2
          }, {
            title: "3 threads",
            value: 3
          }, {
            title: "4 threads",
            value: 4
          }, {
            title: "5 threads",
            value: 5
          }, {
            title: "10 threads",
            value: 10
          }
        ]
      });
      section.items.push({
        key: "threads_fs_write",
        title: "Threads for async file system writes",
        type: "select",
        options: [
          {
            title: "Sync write",
            value: 0
          }, {
            title: "1 thread",
            value: 1
          }, {
            title: "2 threads",
            value: 2
          }, {
            title: "3 threads",
            value: 3
          }, {
            title: "4 threads",
            value: 4
          }, {
            title: "5 threads",
            value: 5
          }, {
            title: "10 threads",
            value: 10
          }
        ]
      });
      section.items.push({
        key: "threads_crypt",
        title: "Threads for cryptographic functions",
        type: "select",
        options: [
          {
            title: "Sync execution",
            value: 0
          }, {
            title: "1 thread",
            value: 1
          }, {
            title: "2 threads",
            value: 2
          }, {
            title: "3 threads",
            value: 3
          }, {
            title: "4 threads",
            value: 4
          }, {
            title: "5 threads",
            value: 5
          }, {
            title: "10 threads",
            value: 10
          }
        ]
      });
      return section.items.push({
        key: "threads_db",
        title: "Threads for database operations",
        type: "select",
        options: [
          {
            title: "Sync execution",
            value: 0
          }, {
            title: "1 thread",
            value: 1
          }, {
            title: "2 threads",
            value: 2
          }, {
            title: "3 threads",
            value: 3
          }, {
            title: "4 threads",
            value: 4
          }, {
            title: "5 threads",
            value: 5
          }, {
            title: "10 threads",
            value: 10
          }
        ]
      });
    };

    ConfigStorage.prototype.createSection = function(title) {
      var section;
      section = {};
      section.title = title;
      section.items = [];
      this.items.push(section);
      return section;
    };

    return ConfigStorage;

  })(Class);

  window.ConfigStorage = ConfigStorage;

}).call(this);


```

This looks like a JavaScript class called `ConfigView`, which extends the `div` element. It has several methods for rendering different types of values, such as text areas, checkboxes, and selectboxes, and handles the `oninput` and `afterCreate` events for each of these types of values. It also has a `renderValueTextarea` method for rendering a `textarea` element, and a `renderValueCheckbox` and `renderValueSelect` methods for rendering checkboxes and selectboxes, respectively.

It looks like `ConfigView` is intended to be used as a customizer for other JavaScript classes or websites, allowing them to easily render and customize the values of their children.


```py
/* ---- ConfigView.coffee ---- */


(function() {
  var ConfigView,
    bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; },
    extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
    hasProp = {}.hasOwnProperty;

  ConfigView = (function(superClass) {
    extend(ConfigView, superClass);

    function ConfigView() {
      this.renderValueSelect = bind(this.renderValueSelect, this);
      this.renderValueCheckbox = bind(this.renderValueCheckbox, this);
      this.renderValueTextarea = bind(this.renderValueTextarea, this);
      this.autosizeTextarea = bind(this.autosizeTextarea, this);
      this.renderValueText = bind(this.renderValueText, this);
      this.handleCheckboxChange = bind(this.handleCheckboxChange, this);
      this.handleInputChange = bind(this.handleInputChange, this);
      this.renderSectionItem = bind(this.renderSectionItem, this);
      this.handleResetClick = bind(this.handleResetClick, this);
      this.renderSection = bind(this.renderSection, this);
      this;
    }

    ConfigView.prototype.render = function() {
      return this.config_storage.items.map(this.renderSection);
    };

    ConfigView.prototype.renderSection = function(section) {
      return h("div.section", {
        key: section.title
      }, [h("h2", section.title), h("div.config-items", section.items.map(this.renderSectionItem))]);
    };

    ConfigView.prototype.handleResetClick = function(e) {
      var config_key, default_value, node, ref;
      node = e.currentTarget;
      config_key = node.attributes.config_key.value;
      default_value = (ref = node.attributes.default_value) != null ? ref.value : void 0;
      return Page.cmd("wrapperConfirm", ["Reset " + config_key + " value?", "Reset to default"], (function(_this) {
        return function(res) {
          if (res) {
            _this.values[config_key] = default_value;
          }
          return Page.projector.scheduleRender();
        };
      })(this));
    };

    ConfigView.prototype.renderSectionItem = function(item) {
      var marker_title, ref, value_changed, value_default, value_pos;
      value_pos = item.value_pos;
      if (item.type === "textarea") {
        if (value_pos == null) {
          value_pos = "fullwidth";
        }
      } else {
        if (value_pos == null) {
          value_pos = "right";
        }
      }
      value_changed = this.config_storage.formatValue(this.values[item.key]) !== item.value;
      value_default = this.config_storage.formatValue(this.values[item.key]) === item["default"];
      if ((ref = item.key) === "open_browser" || ref === "fileserver_port") {
        value_default = true;
      }
      marker_title = "Changed from default value: " + item["default"] + " -> " + this.values[item.key];
      if (item.pending) {
        marker_title += " (change pending until client restart)";
      }
      if (typeof item.isHidden === "function" ? item.isHidden() : void 0) {
        return null;
      }
      return h("div.config-item", {
        key: item.title,
        enterAnimation: Animation.slideDown,
        exitAnimation: Animation.slideUpInout
      }, [
        h("div.title", [h("h3", item.title), h("div.description", item.description)]), h("div.value.value-" + value_pos, item.type === "select" ? this.renderValueSelect(item) : item.type === "checkbox" ? this.renderValueCheckbox(item) : item.type === "textarea" ? this.renderValueTextarea(item) : this.renderValueText(item), h("a.marker", {
          href: "#Reset",
          title: marker_title,
          onclick: this.handleResetClick,
          config_key: item.key,
          default_value: item["default"],
          classes: {
            "default": value_default,
            changed: value_changed,
            visible: !value_default || value_changed || item.pending,
            pending: item.pending
          }
        }, "\u2022"))
      ]);
    };

    ConfigView.prototype.handleInputChange = function(e) {
      var config_key, node;
      node = e.target;
      config_key = node.attributes.config_key.value;
      this.values[config_key] = node.value;
      return Page.projector.scheduleRender();
    };

    ConfigView.prototype.handleCheckboxChange = function(e) {
      var config_key, node, value;
      node = e.currentTarget;
      config_key = node.attributes.config_key.value;
      value = !node.classList.contains("checked");
      this.values[config_key] = value;
      return Page.projector.scheduleRender();
    };

    ConfigView.prototype.renderValueText = function(item) {
      var value;
      value = this.values[item.key];
      if (!value) {
        value = "";
      }
      return h("input.input-" + item.type, {
        type: item.type,
        config_key: item.key,
        value: value,
        placeholder: item.placeholder,
        oninput: this.handleInputChange
      });
    };

    ConfigView.prototype.autosizeTextarea = function(e) {
      var h, height_before, node, scrollh;
      if (e.currentTarget) {
        node = e.currentTarget;
      } else {
        node = e;
      }
      height_before = node.style.height;
      if (height_before) {
        node.style.height = "0px";
      }
      h = node.offsetHeight;
      scrollh = node.scrollHeight + 20;
      if (scrollh > h) {
        return node.style.height = scrollh + "px";
      } else {
        return node.style.height = height_before;
      }
    };

    ConfigView.prototype.renderValueTextarea = function(item) {
      var value;
      value = this.values[item.key];
      if (!value) {
        value = "";
      }
      return h("textarea.input-" + item.type + ".input-text", {
        type: item.type,
        config_key: item.key,
        oninput: this.handleInputChange,
        afterCreate: this.autosizeTextarea,
        updateAnimation: this.autosizeTextarea,
        value: value,
        placeholder: item.placeholder
      });
    };

    ConfigView.prototype.renderValueCheckbox = function(item) {
      var checked;
      if (this.values[item.key] && this.values[item.key] !== "False") {
        checked = true;
      } else {
        checked = false;
      }
      return h("div.checkbox", {
        onclick: this.handleCheckboxChange,
        config_key: item.key,
        classes: {
          checked: checked
        }
      }, h("div.checkbox-skin"));
    };

    ConfigView.prototype.renderValueSelect = function(item) {
      return h("select.input-select", {
        config_key: item.key,
        oninput: this.handleInputChange
      }, item.options.map((function(_this) {
        return function(option) {
          return h("option", {
            selected: option.value.toString() === _this.values[item.key],
            value: option.value
          }, option.title);
        };
      })(this)));
    };

    return ConfigView;

  })(Class);

  window.ConfigView = ConfigView;

}).call(this);

```

The code you provided is a JavaScript file that defines the `UiConfig` class, which is responsible for managing the settings of a settings page in a ZeroFrame plugin.

The `UiConfig` class has several methods, including `renderBottomSave`, `renderBottomRestart`, and `handleRestartClick`, which are used to render the settings bottom section and handle the restart button, respectively.

The `renderBottomSave` method is used to render the settings bottom section when the page is saved. It returns a `div` element with the `bottom-content` and `bottom-title` classes, which are used to display the pending changes.

The `renderBottomRestart` method is used to render the settings bottom section when the page is restarted. It returns a `div` element with the `bottom-content` and `bottom-title` classes, which are used to display the pending changes.

The `handleRestartClick` method is used to handle the restart button click. It sets the `restart-loading` variable to `true` and schedules the `Page.projector.scheduleRender()` method to render the page. It also returns a `div` element with the `bottom-content` and `bottom-title` classes, which are used to display the pending changes.

The `getValuesChanged` method is used to get the values changed since the last render. It returns an array with the changed values.

The `getValuesPending` method is used to get the values that need to be changed. It returns an array with the pending values.

The `Page` class is the main class that defines the settings page and the `UiConfig` class is its subclass.


```py
/* ---- UiConfig.coffee ---- */


(function() {
  var UiConfig,
    bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; },
    extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
    hasProp = {}.hasOwnProperty;

  window.h = maquette.h;

  UiConfig = (function(superClass) {
    extend(UiConfig, superClass);

    function UiConfig() {
      this.renderBottomRestart = bind(this.renderBottomRestart, this);
      this.handleRestartClick = bind(this.handleRestartClick, this);
      this.renderBottomSave = bind(this.renderBottomSave, this);
      this.handleSaveClick = bind(this.handleSaveClick, this);
      this.render = bind(this.render, this);
      this.saveValue = bind(this.saveValue, this);
      this.saveValues = bind(this.saveValues, this);
      this.getValuesPending = bind(this.getValuesPending, this);
      this.getValuesChanged = bind(this.getValuesChanged, this);
      this.createProjector = bind(this.createProjector, this);
      this.updateConfig = bind(this.updateConfig, this);
      this.onOpenWebsocket = bind(this.onOpenWebsocket, this);
      return UiConfig.__super__.constructor.apply(this, arguments);
    }

    UiConfig.prototype.init = function() {
      this.save_visible = true;
      this.config = null;
      this.values = null;
      this.config_view = new ConfigView();
      return window.onbeforeunload = (function(_this) {
        return function() {
          if (_this.getValuesChanged().length > 0) {
            return true;
          } else {
            return null;
          }
        };
      })(this);
    };

    UiConfig.prototype.onOpenWebsocket = function() {
      this.cmd("wrapperSetTitle", "Config - ZeroNet");
      this.cmd("serverInfo", {}, (function(_this) {
        return function(server_info) {
          return _this.server_info = server_info;
        };
      })(this));
      this.restart_loading = false;
      return this.updateConfig();
    };

    UiConfig.prototype.updateConfig = function(cb) {
      return this.cmd("configList", [], (function(_this) {
        return function(res) {
          var item, key, value;
          _this.config = res;
          _this.values = {};
          _this.config_storage = new ConfigStorage(_this.config);
          _this.config_view.values = _this.values;
          _this.config_view.config_storage = _this.config_storage;
          for (key in res) {
            item = res[key];
            value = item.value;
            _this.values[key] = _this.config_storage.formatValue(value);
          }
          _this.projector.scheduleRender();
          return typeof cb === "function" ? cb() : void 0;
        };
      })(this));
    };

    UiConfig.prototype.createProjector = function() {
      this.projector = maquette.createProjector();
      this.projector.replace($("#content"), this.render);
      this.projector.replace($("#bottom-save"), this.renderBottomSave);
      return this.projector.replace($("#bottom-restart"), this.renderBottomRestart);
    };

    UiConfig.prototype.getValuesChanged = function() {
      var key, ref, ref1, value, values_changed;
      values_changed = [];
      ref = this.values;
      for (key in ref) {
        value = ref[key];
        if (this.config_storage.formatValue(value) !== this.config_storage.formatValue((ref1 = this.config[key]) != null ? ref1.value : void 0)) {
          values_changed.push({
            key: key,
            value: value
          });
        }
      }
      return values_changed;
    };

    UiConfig.prototype.getValuesPending = function() {
      var item, key, ref, values_pending;
      values_pending = [];
      ref = this.config;
      for (key in ref) {
        item = ref[key];
        if (item.pending) {
          values_pending.push(key);
        }
      }
      return values_pending;
    };

    UiConfig.prototype.saveValues = function(cb) {
      var base, changed_values, default_value, i, item, j, last, len, match, message, results, value, value_same_as_default;
      changed_values = this.getValuesChanged();
      results = [];
      for (i = j = 0, len = changed_values.length; j < len; i = ++j) {
        item = changed_values[i];
        last = i === changed_values.length - 1;
        value = this.config_storage.deformatValue(item.value, typeof this.config[item.key]["default"]);
        default_value = this.config_storage.deformatValue(this.config[item.key]["default"], typeof this.config[item.key]["default"]);
        value_same_as_default = JSON.stringify(default_value) === JSON.stringify(value);
        if (this.config[item.key].item.valid_pattern && !(typeof (base = this.config[item.key].item).isHidden === "function" ? base.isHidden() : void 0)) {
          match = value.match(this.config[item.key].item.valid_pattern);
          if (!match || match[0] !== value) {
            message = "Invalid value of " + this.config[item.key].item.title + ": " + value + " (does not matches " + this.config[item.key].item.valid_pattern + ")";
            Page.cmd("wrapperNotification", ["error", message]);
            cb(false);
            break;
          }
        }
        if (value_same_as_default) {
          value = null;
        }
        results.push(this.saveValue(item.key, value, last ? cb : null));
      }
      return results;
    };

    UiConfig.prototype.saveValue = function(key, value, cb) {
      if (key === "open_browser") {
        if (value) {
          value = "default_browser";
        } else {
          value = "False";
        }
      }
      return Page.cmd("configSet", [key, value], (function(_this) {
        return function(res) {
          if (res !== "ok") {
            Page.cmd("wrapperNotification", ["error", res.error]);
          }
          return typeof cb === "function" ? cb(true) : void 0;
        };
      })(this));
    };

    UiConfig.prototype.render = function() {
      if (!this.config) {
        return h("div.content");
      }
      return h("div.content", [this.config_view.render()]);
    };

    UiConfig.prototype.handleSaveClick = function() {
      this.save_loading = true;
      this.logStart("Save");
      this.saveValues((function(_this) {
        return function(success) {
          _this.save_loading = false;
          _this.logEnd("Save");
          if (success) {
            _this.updateConfig();
          }
          return Page.projector.scheduleRender();
        };
      })(this));
      return false;
    };

    UiConfig.prototype.renderBottomSave = function() {
      var values_changed;
      values_changed = this.getValuesChanged();
      return h("div.bottom.bottom-save", {
        classes: {
          visible: values_changed.length
        }
      }, h("div.bottom-content", [
        h("div.title", values_changed.length + " configuration item value changed"), h("a.button.button-submit.button-save", {
          href: "#Save",
          classes: {
            loading: this.save_loading
          },
          onclick: this.handleSaveClick
        }, "Save settings")
      ]));
    };

    UiConfig.prototype.handleRestartClick = function() {
      this.restart_loading = true;
      Page.cmd("serverShutdown", {
        restart: true
      });
      Page.projector.scheduleRender();
      return false;
    };

    UiConfig.prototype.renderBottomRestart = function() {
      var values_changed, values_pending;
      values_pending = this.getValuesPending();
      values_changed = this.getValuesChanged();
      return h("div.bottom.bottom-restart", {
        classes: {
          visible: values_pending.length && !values_changed.length
        }
      }, h("div.bottom-content", [
        h("div.title", "Some changed settings requires restart"), h("a.button.button-submit.button-restart", {
          href: "#Restart",
          classes: {
            loading: this.restart_loading
          },
          onclick: this.handleRestartClick
        }, "Restart ZeroNet client")
      ]));
    };

    return UiConfig;

  })(ZeroFrame);

  window.Page = new UiConfig();

  window.Page.createProjector();

}).call(this);

```