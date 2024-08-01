# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\pages\_app-d0edbdcb9ec8a0fc.js`

```py
    `]:Object.assign(Object.assign({},r(i)),{animationPlayState:"paused"}),[`${s}${e}-leave`]:Object.assign(Object.assign({},o(i)),{animationPlayState:"paused"}),[`
      ${s}${e}-enter${e}-enter-active,
      ${s}${e}-appear${e}-appear-active

这段代码使用了模板字符串来动态生成对象的属性名。具体解释如下：

- ``` `]: ```py：这是一个对象属性名的起始部分，后续将使用模板字符串来拼接属性名。
- ``` Object.assign(Object.assign({}, r(i)), { animationPlayState: "paused" }) ```py：这段代码通过 `Object.assign` 方法创建了一个新对象，包含了 `r(i)` 对象的所有属性，并添加了 `animationPlayState` 属性，并将其值设为 `"paused"`。
- ``` , ```py：逗号用于分隔不同的对象属性。
- ``` `${s}${e}-leave` ```py：使用模板字符串来拼接属性名，`${s}` 和 `${e}` 是变量或字符串的部分。
- ``` Object.assign(Object.assign({}, o(i)), { animationPlayState: "paused" }) ```py：同样使用 `Object.assign` 创建了一个新对象，包含了 `o(i)` 对象的所有属性，并添加了 `animationPlayState` 属性，并将其值设为 `"paused"`。
- ``` , ```py：再次用逗号分隔不同的对象属性。
- ``` `[``：结束该对象属性名的拼接。

这段代码的作用是生成一个复杂的对象，包含了多个动画状态的定义，并设置它们的 `animationPlayState` 属性为 `"paused"`，用于控制动画的播放状态。
/*
    'Noto Sans', sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol',
*/
    /*
        boxShadowSecondary:
          0 6px 16px 0 rgba(0, 0, 0, 0.08),
          0 3px 6px -4px rgba(0, 0, 0, 0.12),
          0 9px 28px 8px rgba(0, 0, 0, 0.05)
    */
    /*
        boxShadowTertiary:
          0 1px 2px 0 rgba(0, 0, 0, 0.03),
          0 1px 6px -1px rgba(0, 0, 0, 0.02),
          0 2px 4px 0 rgba(0, 0, 0, 0.02)
    */
    /*
        screenXS:480,screenXSMin:480,screenXSMax:575,screenSM:576,screenSMMin:576,screenSMMax:767,screenMD:768,screenMDMin:768,screenMDMax:991,screenLG:992,screenLGMin:992,screenLGMax:1199,screenXL:1200,screenXLMin:1200,screenXLMax:1599,screenXXL:1600,screenXXLMin:1600,boxShadowPopoverArrow:"2px 2px 5px rgba(0, 0, 0, 0.05)",boxShadowCard:
          0 1px 2px -2px rgba(0, 0, 0, 0.16),
          0 3px 6px 0 rgba(0, 0, 0, 0.12),
          0 5px 12px 4px rgba(0, 0, 0, 0.09)
    */
    /*
        boxShadowDrawerRight:
          -6px 0 16px 0 rgba(0, 0, 0, 0.08),
          -3px 0 6px -4px rgba(0, 0, 0, 0.12),
          -9px 0 28px 8px rgba(0, 0, 0, 0.05)
    */
    /*
        boxShadowDrawerLeft:
          6px 0 16px 0 rgba(0, 0, 0, 0.08),
          3px 0 6px -4px rgba(0, 0, 0, 0.12),
          9px 0 28px 8px rgba(0, 0, 0, 0.05)
    */
    /*
        boxShadowDrawerUp:
          0 6px 16px 0 rgba(0, 0, 0, 0.08),
          0 3px 6px -4px rgba(0, 0, 0, 0.12),
          0 9px 28px 8px rgba(0, 0, 0, 0.05)
    */
    /*
        boxShadowDrawerDown:
          0 -6px 16px 0 rgba(0, 0, 0, 0.08),
          0 -3px 6px -4px rgba(0, 0, 0, 0.12),
          0 -9px 28px 8px rgba(0, 0, 0, 0.05)
    */
try {
    /*
        var mode = localStorage.getItem('${o}') || '${t}';
        var colorScheme = '';
    */
    if (mode === 'system') {
        // handle system mode
        /*
            var mql = window.matchMedia('(prefers-color-scheme: dark)');
            if (mql.matches) {
                colorScheme = localStorage.getItem('${a}-dark') || '${r}';
            } else {
                colorScheme = localStorage.getItem('${a}-light') || '${n}';
            }
        */
    }
    if (mode === 'light') {
        /*
            colorScheme = localStorage.getItem('${a}-light') || '${n}';
        */
    }
    if (mode === 'dark') {
        /*
            colorScheme = localStorage.getItem('${a}-dark') || '${r}';
        */
    }
    if (colorScheme) {
        /*
            ${l}.setAttribute('${s}', colorScheme);
        */
    }
} catch (e) {
    /*
        |
        Copyright (c) 2018 Jed Watson.
        Licensed under the MIT License (MIT), see
        http://jedwatson.github.io/classnames
    */
}
```