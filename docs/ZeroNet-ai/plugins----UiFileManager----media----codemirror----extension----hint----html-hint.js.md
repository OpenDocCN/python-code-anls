# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\hint\html-hint.js`

```
// CodeMirror, 版权归 Marijn Haverbeke 和其他人所有
// 根据 MIT 许可证分发：https://codemirror.net/LICENSE

// 匿名函数，接受一个 mod 参数
(function(mod) {
  // 如果 exports 和 module 都存在，说明是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror"), require("./xml-hint"));
  // 如果 define 存在且是一个函数，说明是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror", "./xml-hint"], mod);
  // 否则是普通的浏览器环境
  else
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 语言列表
  var langs = "ab aa af ak sq am ar an hy as av ae ay az bm ba eu be bn bh bi bs br bg my ca ch ce ny zh cv kw co cr hr cs da dv nl dz en eo et ee fo fj fi fr ff gl ka de el gn gu ht ha he hz hi ho hu ia id ie ga ig ik io is it iu ja jv kl kn kr ks kk km ki rw ky kv kg ko ku kj la lb lg li ln lo lt lu lv gv mk mg ms ml mt mi mr mh mn na nv nb nd ne ng nn no ii nr oc oj cu om or os pa pi fa pl ps pt qu rm rn ro ru sa sc sd se sm sg sr gd sn si sk sl so st es su sw ss sv ta te tg th ti bo tk tl tn to tr ts tt tw ty ug uk ur uz ve vi vo wa cy wo fy xh yi yo za zu".split(" ");
  // 链接目标列表
  var targets = ["_blank", "_self", "_top", "_parent"];
  // 字符集列表
  var charsets = ["ascii", "utf-8", "utf-16", "latin1", "latin1"];
  // 请求方法列表
  var methods = ["get", "post", "put", "delete"];
  // 编码类型列表
  var encs = ["application/x-www-form-urlencoded", "multipart/form-data", "text/plain"];
  // 媒体类型列表
  var media = ["all", "screen", "print", "embossed", "braille", "handheld", "print", "projection", "screen", "tty", "tv", "speech",
               "3d-glasses", "resolution [>][<][=] [X]", "device-aspect-ratio: X/Y", "orientation:portrait",
               "orientation:landscape", "device-height: [X]", "device-width: [X]"];
  // 简单标签对象
  var s = { attrs: {} }; // Simple tag, reused for a whole lot of tags

  // 数据对象
  var data = {
    a: {
      attrs: {
        href: null, ping: null, type: null,
        media: media,
        target: targets,
        hreflang: langs
      }
    },
    abbr: s,
    acronym: s,
    address: s,
    applet: s,
    # 定义 HTML 元素 area 的属性
    area: {
      attrs: {
        alt: null, coords: null, href: null, target: null, ping: null,
        media: media, hreflang: langs, type: null,
        shape: ["default", "rect", "circle", "poly"]
      }
    },
    # 定义 HTML 元素 article 和 aside，它们没有特定的属性
    article: s,
    aside: s,
    # 定义 HTML 元素 audio 的属性
    audio: {
      attrs: {
        src: null, mediagroup: null,
        crossorigin: ["anonymous", "use-credentials"],
        preload: ["none", "metadata", "auto"],
        autoplay: ["", "autoplay"],
        loop: ["", "loop"],
        controls: ["", "controls"]
      }
    },
    # 定义 HTML 元素 b，它没有特定的属性
    b: s,
    # 定义 HTML 元素 base 的属性
    base: { attrs: { href: null, target: targets } },
    # 定义 HTML 元素 basefont，它没有特定的属性
    basefont: s,
    # 定义 HTML 元素 bdi，它没有特定的属性
    bdi: s,
    # 定义 HTML 元素 bdo，它没有特定的属性
    bdo: s,
    # 定义 HTML 元素 big，它没有特定的属性
    big: s,
    # 定义 HTML 元素 blockquote 的属性
    blockquote: { attrs: { cite: null } },
    # 定义 HTML 元素 body，它没有特定的属性
    body: s,
    # 定义 HTML 元素 br，它没有特定的属性
    br: s,
    # 定义 HTML 元素 button 的属性
    button: {
      attrs: {
        form: null, formaction: null, name: null, value: null,
        autofocus: ["", "autofocus"],
        disabled: ["", "autofocus"],
        formenctype: encs,
        formmethod: methods,
        formnovalidate: ["", "novalidate"],
        formtarget: targets,
        type: ["submit", "reset", "button"]
      }
    },
    # 定义 HTML 元素 canvas 的属性
    canvas: { attrs: { width: null, height: null } },
    # 定义 HTML 元素 caption，它没有特定的属性
    caption: s,
    # 定义 HTML 元素 center，它没有特定的属性
    center: s,
    # 定义 HTML 元素 cite，它没有特定的属性
    cite: s,
    # 定义 HTML 元素 code，它没有特定的属性
    code: s,
    # 定义 HTML 元素 col 的属性
    col: { attrs: { span: null } },
    # 定义 HTML 元素 colgroup 的属性
    colgroup: { attrs: { span: null } },
    # 定义 HTML 元素 command 的属性
    command: {
      attrs: {
        type: ["command", "checkbox", "radio"],
        label: null, icon: null, radiogroup: null, command: null, title: null,
        disabled: ["", "disabled"],
        checked: ["", "checked"]
      }
    },
    # 定义 HTML 元素 data 的属性
    data: { attrs: { value: null } },
    # 定义 HTML 元素 datagrid 的属性
    datagrid: { attrs: { disabled: ["", "disabled"], multiple: ["", "multiple"] } },
    # 定义 HTML 元素 datalist 的属性
    datalist: { attrs: { data: null } },
    # 定义 HTML 元素 dd，它没有特定的属性
    dd: s,
    # 定义 HTML 元素 del 的属性
    del: { attrs: { cite: null, datetime: null } },
    # 定义 HTML 元素 details 的属性
    details: { attrs: { open: ["", "open"] } },
    # 定义 HTML 元素 dfn，它没有特定的属性
    dfn: s,
    # 定义 HTML 元素 dir，它没有特定的属性
    dir: s,
    # 定义 HTML 元素 div，它没有特定的属性
    div: s,
    # 定义 HTML 元素 dl，它没有特定的属性
    dl: s,
    # 定义 HTML 元素 dt，它没有特定的属性
    dt: s,
    # 定义 HTML 元素 em，它没有特定的属性
    em: s,
    # 定义 HTML 元素 embed 的属性
    embed: { attrs: { src: null, type: null, width: null, height: null } },
    # 定义 HTML 元素 eventsource 的属性
    eventsource: { attrs: { src: null } },
    # 定义字段集合，包含禁用属性和表单属性
    fieldset: { attrs: { disabled: ["", "disabled"], form: null, name: null } },
    # 定义图例元素
    figcaption: s,
    # 定义图像和图形元素
    figure: s,
    # 定义字体元素
    font: s,
    # 定义页脚元素
    footer: s,
    # 定义表单元素，包含各种属性和值的集合
    form: {
      attrs: {
        action: null, name: null,
        "accept-charset": charsets,
        autocomplete: ["on", "off"],
        enctype: encs,
        method: methods,
        novalidate: ["", "novalidate"],
        target: targets
      }
    },
    # 定义框架元素
    frame: s,
    # 定义框架集元素
    frameset: s,
    # 定义标题元素，包括 h1 到 h6
    h1: s, h2: s, h3: s, h4: s, h5: s, h6: s,
    # 定义头部元素，包含属性和子元素
    head: {
      attrs: {},
      children: ["title", "base", "link", "style", "meta", "script", "noscript", "command"]
    },
    # 定义页眉元素
    header: s,
    # 定义标题组元素
    hgroup: s,
    # 定义水平线元素
    hr: s,
    # 定义 HTML 元素，包含属性和子元素
    html: {
      attrs: { manifest: null },
      children: ["head", "body"]
    },
    # 定义斜体文本元素
    i: s,
    # 定义内联框架元素，包含各种属性和值的集合
    iframe: {
      attrs: {
        src: null, srcdoc: null, name: null, width: null, height: null,
        sandbox: ["allow-top-navigation", "allow-same-origin", "allow-forms", "allow-scripts"],
        seamless: ["", "seamless"]
      }
    },
    # 定义图像元素，包含各种属性和值的集合
    img: {
      attrs: {
        alt: null, src: null, ismap: null, usemap: null, width: null, height: null,
        crossorigin: ["anonymous", "use-credentials"]
      }
    },
    # 定义输入对象，包含各种 HTML 元素的属性
    input: {
      attrs: {
        # 图像的替代文本
        alt: null, 
        # 提交表单时，提交的文件路径
        dirname: null, 
        # 关联表单元素
        form: null, 
        # 提交表单时的 URL
        formaction: null,
        # 图像的高度
        height: null, 
        # 与输入框关联的 datalist 元素的 id
        list: null, 
        # 允许的最大值
        max: null, 
        # 允许的最大字符数
        maxlength: null, 
        # 允许的最小值
        min: null,
        # 元素的名称
        name: null, 
        # 输入的模式
        pattern: null, 
        # 输入框的占位符
        placeholder: null, 
        # 输入框的宽度
        size: null, 
        # 图像的 URL
        src: null,
        # 步长
        step: null, 
        # 输入框的值
        value: null, 
        # 图像的宽度
        width: null,
        # 接受的文件类型
        accept: ["audio/*", "video/*", "image/*"],
        # 自动完成
        autocomplete: ["on", "off"],
        # 自动获取焦点
        autofocus: ["", "autofocus"],
        # 是否选中
        checked: ["", "checked"],
        # 是否禁用
        disabled: ["", "disabled"],
        # 表单数据的编码类型
        formenctype: encs,
        # 表单提交的 HTTP 方法
        formmethod: methods,
        # 是否跳过表单验证
        formnovalidate: ["", "novalidate"],
        # 表单提交的目标
        formtarget: targets,
        # 是否允许多个文件
        multiple: ["", "multiple"],
        # 只读
        readonly: ["", "readonly"],
        # 是否必填
        required: ["", "required"],
        # 输入类型
        type: ["hidden", "text", "search", "tel", "url", "email", "password", "datetime", "date", "month",
               "week", "time", "datetime-local", "number", "range", "color", "checkbox", "radio",
               "file", "submit", "image", "reset", "button"]
      }
    },
    # 插入标签
    ins: { attrs: { cite: null, datetime: null } },
    # 键盘输入
    kbd: s,
    # 密钥对生成器
    keygen: {
      attrs: {
        # 挑战字符串
        challenge: null, 
        # 关联的表单元素
        form: null, 
        # 元素的名称
        name: null,
        # 自动获取焦点
        autofocus: ["", "autofocus"],
        # 是否禁用
        disabled: ["", "disabled"],
        # 密钥类型
        keytype: ["RSA"]
      }
    },
    # 标签
    label: { attrs: { "for": null, form: null } },
    # 图例
    legend: s,
    # 列表项
    li: { attrs: { value: null } },
    # 链接
    link: {
      attrs: {
        # 链接的 URL
        href: null, 
        # 链接类型
        type: null,
        # 链接的语言
        hreflang: langs,
        # 媒体查询条件
        media: media,
        # 图像尺寸
        sizes: ["all", "16x16", "16x16 32x32", "16x16 32x32 64x64"]
      }
    },
    # 图像映射
    map: { attrs: { name: null } },
    # 高亮标记
    mark: s,
    # 菜单
    menu: { attrs: { label: null, type: ["list", "context", "toolbar"] } },
    # 元数据
    meta: {
      attrs: {
        # 元数据内容
        content: null,
        # 字符集
        charset: charsets,
        # 元数据名称
        name: ["viewport", "application-name", "author", "description", "generator", "keywords"],
        # HTTP 头部设置
        "http-equiv": ["content-language", "content-type", "default-style", "refresh"]
      }
    },
    # 定义 meter 元素，包含 value、min、low、high、max、optimum 属性
    meter: { attrs: { value: null, min: null, low: null, high: null, max: null, optimum: null } },
    # 定义 nav 元素，无属性
    nav: s,
    # 定义 noframes 元素，无属性
    noframes: s,
    # 定义 noscript 元素，无属性
    noscript: s,
    # 定义 object 元素，包含 data、type、name、usemap、form、width、height、typemustmatch 属性
    object: {
      attrs: {
        data: null, type: null, name: null, usemap: null, form: null, width: null, height: null,
        typemustmatch: ["", "typemustmatch"]
      }
    },
    # 定义 ol 元素，包含 reversed、start、type 属性
    ol: { attrs: { reversed: ["", "reversed"], start: null, type: ["1", "a", "A", "i", "I"] } },
    # 定义 optgroup 元素，包含 disabled、label 属性
    optgroup: { attrs: { disabled: ["", "disabled"], label: null } },
    # 定义 option 元素，包含 disabled、label、selected、value 属性
    option: { attrs: { disabled: ["", "disabled"], label: null, selected: ["", "selected"], value: null } },
    # 定义 output 元素，包含 for、form、name 属性
    output: { attrs: { "for": null, form: null, name: null } },
    # 定义 p 元素，无属性
    p: s,
    # 定义 param 元素，包含 name、value 属性
    param: { attrs: { name: null, value: null } },
    # 定义 pre 元素，无属性
    pre: s,
    # 定义 progress 元素，包含 value、max 属性
    progress: { attrs: { value: null, max: null } },
    # 定义 q 元素，包含 cite 属性
    q: { attrs: { cite: null } },
    # 定义 rp 元素，无属性
    rp: s,
    # 定义 rt 元素，无属性
    rt: s,
    # 定义 ruby 元素，无属性
    ruby: s,
    # 定义 s 元素，无属性
    s: s,
    # 定义 samp 元素，无属性
    samp: s,
    # 定义 script 元素，包含 type、src、async、defer、charset 属性
    script: {
      attrs: {
        type: ["text/javascript"],
        src: null,
        async: ["", "async"],
        defer: ["", "defer"],
        charset: charsets
      }
    },
    # 定义 section 元素，无属性
    section: s,
    # 定义 select 元素，包含 form、name、size、autofocus、disabled、multiple 属性
    select: {
      attrs: {
        form: null, name: null, size: null,
        autofocus: ["", "autofocus"],
        disabled: ["", "disabled"],
        multiple: ["", "multiple"]
      }
    },
    # 定义 small 元素，无属性
    small: s,
    # 定义 source 元素，包含 src、type、media 属性
    source: { attrs: { src: null, type: null, media: null } },
    # 定义 span 元素，无属性
    span: s,
    # 定义 strike 元素，无属性
    strike: s,
    # 定义 strong 元素，无属性
    strong: s,
    # 定义 style 元素，包含 type、media、scoped 属性
    style: {
      attrs: {
        type: ["text/css"],
        media: media,
        scoped: null
      }
    },
    # 定义 sub 元素，无属性
    sub: s,
    # 定义 summary 元素，无属性
    summary: s,
    # 定义 sup 元素，无属性
    sup: s,
    # 定义 table 元素，无属性
    table: s,
    # 定义 tbody 元素，无属性
    tbody: s,
    # 定义 td 元素，包含 colspan、rowspan、headers 属性
    td: { attrs: { colspan: null, rowspan: null, headers: null } },
    # 定义 textarea 元素，包含 dirname、form、maxlength、name、placeholder、rows、cols、autofocus、disabled、readonly、required、wrap 属性
    textarea: {
      attrs: {
        dirname: null, form: null, maxlength: null, name: null, placeholder: null,
        rows: null, cols: null,
        autofocus: ["", "autofocus"],
        disabled: ["", "disabled"],
        readonly: ["", "readonly"],
        required: ["", "required"],
        wrap: ["soft", "hard"]
      }
    },
  // 定义 HTML 元素的默认属性和值
  tfoot: s,
  th: { attrs: { colspan: null, rowspan: null, headers: null, scope: ["row", "col", "rowgroup", "colgroup"] } },
  thead: s,
  time: { attrs: { datetime: null } },
  title: s,
  tr: s,
  track: {
    attrs: {
      src: null, label: null, "default": null,
      kind: ["subtitles", "captions", "descriptions", "chapters", "metadata"],
      srclang: langs
    }
  },
  tt: s,
  u: s,
  ul: s,
  "var": s,
  video: {
    attrs: {
      src: null, poster: null, width: null, height: null,
      crossorigin: ["anonymous", "use-credentials"],
      preload: ["auto", "metadata", "none"],
      autoplay: ["", "autoplay"],
      mediagroup: ["movie"],
      muted: ["", "muted"],
      controls: ["", "controls"]
    }
  },
  wbr: s
};

// 定义全局属性和其可接受的值
var globalAttrs = {
  accesskey: ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
  "class": null,
  contenteditable: ["true", "false"],
  contextmenu: null,
  dir: ["ltr", "rtl", "auto"],
  draggable: ["true", "false", "auto"],
  dropzone: ["copy", "move", "link", "string:", "file:"],
  hidden: ["hidden"],
  id: null,
  inert: ["inert"],
  itemid: null,
  itemprop: null,
  itemref: null,
  itemscope: ["itemscope"],
  itemtype: null,
  lang: ["en", "es"],
  spellcheck: ["true", "false"],
  autocorrect: ["true", "false"],
  autocapitalize: ["true", "false"],
  style: null,
  tabindex: ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
  title: null,
  translate: ["yes", "no"],
  onclick: null,
  rel: ["stylesheet", "alternate", "author", "bookmark", "help", "license", "next", "nofollow", "noreferrer", "prefetch", "prev", "search", "tag"]
};
// 定义一个函数，用于填充对象的属性和值
function populate(obj) {
  # 遍历全局属性对象，将属性添加到目标对象的属性中
  for (var attr in globalAttrs) if (globalAttrs.hasOwnProperty(attr))
    obj.attrs[attr] = globalAttrs[attr];
}

# 使用全局属性对象填充目标对象
populate(s);
# 遍历数据对象，如果数据对象有该标签并且不等于s，则填充数据对象
for (var tag in data) if (data.hasOwnProperty(tag) && data[tag] != s)
  populate(data[tag]);

# 将数据对象赋值给CodeMirror.htmlSchema
CodeMirror.htmlSchema = data;
# 定义htmlHint函数，传入CodeMirror编辑器和选项参数
function htmlHint(cm, options) {
  # 定义本地变量，包含schemaInfo属性
  var local = {schemaInfo: data};
  # 如果有选项参数，将选项参数添加到本地变量中
  if (options) for (var opt in options) local[opt] = options[opt];
  # 返回xml提示
  return CodeMirror.hint.xml(cm, local);
}
# 注册htmlHint函数为html提示的辅助函数
CodeMirror.registerHelper("hint", "html", htmlHint);
# 闭合了一个代码块或者函数的结束
```