# `ZeroNet\src\Ui\media\lib\jquery.cssanim.js`

```
# 将自定义的scale属性添加到jQuery的cssHooks对象中
jQuery.cssHooks['scale'] = {
    # 获取元素的scale属性值
    get: function(elem, computed) {
        # 通过window.getComputedStyle获取元素的transform属性值，并匹配出数字部分
        var match = window.getComputedStyle(elem)[transform_property].match("[0-9\.]+")
        # 如果匹配到数字部分
        if (match) {
            # 将匹配到的数字部分转换为浮点数，作为scale属性值返回
            var scale = parseFloat(match[0])
            return scale
        } else {
            # 如果未匹配到数字部分，则返回默认值1.0
            return 1.0
        }
    },
    # 设置元素的scale属性值
    set: function(elem, val) {
        # 通过window.getComputedStyle获取元素的transform属性值，并匹配出所有数字部分
        var transforms = window.getComputedStyle(elem)[transform_property].match(/[0-9\.]+/g)
        # 如果匹配到数字部分
        if (transforms) {
            # 更新transforms数组的第一个和第四个元素为val
            transforms[0] = val
            transforms[3] = val
            # 更新元素的transform属性值为新的matrix值
            elem.style[transform_property] = 'matrix('+transforms.join(", ")+')'
        } else {
            # 如果未匹配到数字部分，则直接设置元素的transform属性值为scale(val)
            elem.style[transform_property] = "scale("+val+")"
        }
    }
}

# 将自定义的scale属性添加到jQuery的fx.step对象中
jQuery.fx.step.scale = function(fx) {
    # 调用cssHooks中scale属性的set方法，设置元素的scale属性值为当前值fx.now
    jQuery.cssHooks['scale'].set(fx.elem, fx.now)
};

# 判断浏览器是否支持transform属性，根据支持情况设置transform_property变量
if (window.getComputedStyle(document.body).transform) {
    transform_property = "transform"
} else {
    transform_property = "webkitTransform"
}
```