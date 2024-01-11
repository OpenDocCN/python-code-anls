# `ZeroNet\src\Ui\media\lib\jquery.easing.js`

```
/*
 * jQuery Easing v1.4.1 - http://gsgd.co.uk/sandbox/jquery/easing/
 * Open source under the BSD License.
 * Copyright © 2008 George McGinley Smith
 * All rights reserved.
 * https://raw.github.com/gdsmith/jquery-easing/master/LICENSE
*/

(function (factory) {
    if (typeof define === "function" && define.amd) {
        define(['jquery'], function ($) {
            return factory($);
        });
    } else if (typeof module === "object" && typeof module.exports === "object") {
        exports = factory(require('jquery'));
    } else {
        factory(jQuery);
    }
})(function($){

// Preserve the original jQuery "swing" easing as "jswing"
if (typeof $.easing !== 'undefined') {
    $.easing['jswing'] = $.easing['swing'];
}

var pow = Math.pow,  // 定义 pow 函数为 Math.pow
    sqrt = Math.sqrt,  // 定义 sqrt 函数为 Math.sqrt
    sin = Math.sin,  // 定义 sin 函数为 Math.sin
    cos = Math.cos,  // 定义 cos 函数为 Math.cos
    PI = Math.PI,  // 定义 PI 为 Math.PI
    c1 = 1.70158,  // 定义 c1 为 1.70158
    c2 = c1 * 1.525,  // 定义 c2 为 c1 乘以 1.525
    c3 = c1 + 1,  // 定义 c3 为 c1 加 1
    c4 = ( 2 * PI ) / 3,  // 定义 c4 为 (2 * PI) 除以 3
    c5 = ( 2 * PI ) / 4.5;  // 定义 c5 为 (2 * PI) 除以 4.5

// x is the fraction of animation progress, in the range 0..1
function bounceOut(x) {  // 定义函数 bounceOut，参数为 x
    var n1 = 7.5625,  // 定义 n1 为 7.5625
        d1 = 2.75;  // 定义 d1 为 2.75
    if ( x < 1/d1 ) {  // 如果 x 小于 1/d1
        return n1*x*x;  // 返回 n1 乘以 x 乘以 x
    } else if ( x < 2/d1 ) {  // 否则如果 x 小于 2/d1
        return n1*(x-=(1.5/d1))*x + .75;  // 返回 n1 乘以 (x 减去 (1.5/d1)) 乘以 x 加上 0.75
    } else if ( x < 2.5/d1 ) {  // 否则如果 x 小于 2.5/d1
        return n1*(x-=(2.25/d1))*x + .9375;  // 返回 n1 乘以 (x 减去 (2.25/d1)) 乘以 x 加上 0.9375
    } else {  // 否则
        return n1*(x-=(2.625/d1))*x + .984375;  // 返回 n1 乘以 (x 减去 (2.625/d1)) 乘以 x 加上 0.984375
    }
}

$.extend( $.easing,  // 扩展 jQuery.easing 对象
{
    def: 'easeOutQuad',  // 默认动画效果为 easeOutQuad
    swing: function (x) {  // 定义 swing 函数，参数为 x
        return $.easing[$.easing.def](x);  // 返回 $.easing.def 对应的动画效果函数
    },
    easeInQuad: function (x) {  // 定义 easeInQuad 函数，参数为 x
        return x * x;  // 返回 x 的平方
    },
    easeOutQuad: function (x) {  // 定义 easeOutQuad 函数，参数为 x
        return 1 - ( 1 - x ) * ( 1 - x );  // 返回 1 减去 (1 减去 x) 乘以 (1 减去 x)
    },
    easeInOutQuad: function (x) {  // 定义 easeInOutQuad 函数，参数为 x
        return x < 0.5 ?  // 如果 x 小于 0.5
            2 * x * x :  // 返回 2 乘以 x 乘以 x
            1 - pow( -2 * x + 2, 2 ) / 2;  // 否则返回 1 减去 pow( -2 * x + 2, 2 ) 除以 2
    },
    easeInCubic: function (x) {  // 定义 easeInCubic 函数，参数为 x
        return x * x * x;  // 返回 x 的立方
    },
    easeOutCubic: function (x) {  // 定义 easeOutCubic 函数，参数为 x
        return 1 - pow( 1 - x, 3 );  // 返回 1 减去 pow( 1 减去 x, 3 )
    },
    easeInOutCubic: function (x) {  // 定义 easeInOutCubic 函数，参数为 x
        return x < 0.5 ?  // 如果 x 小于 0.5
            4 * x * x * x :  // 返回 4 乘以 x 乘以 x 乘以 x
            1 - pow( -2 * x + 2, 3 ) / 2;  // 否则返回 1 减去 pow( -2 * x + 2, 3 ) 除以 2
    easeInQuart: function (x) {
        // 使用四次方函数实现缓动效果，输入参数为x，返回x的四次方
        return x * x * x * x;
    },
    easeOutQuart: function (x) {
        // 使用四次方函数实现缓动效果，输入参数为x，返回1减去(1-x)的四次方
        return 1 - pow( 1 - x, 4 );
    },
    easeInOutQuart: function (x) {
        // 使用四次方函数实现缓动效果，输入参数为x，返回根据x的大小不同计算得到的值
        return x < 0.5 ?
            8 * x * x * x * x :
            1 - pow( -2 * x + 2, 4 ) / 2;
    },
    easeInQuint: function (x) {
        // 使用五次方函数实现缓动效果，输入参数为x，返回x的五次方
        return x * x * x * x * x;
    },
    easeOutQuint: function (x) {
        // 使用五次方函数实现缓动效果，输入参数为x，返回1减去(1-x)的五次方
        return 1 - pow( 1 - x, 5 );
    },
    easeInOutQuint: function (x) {
        // 使用五次方函数实现缓动效果，输入参数为x，返回根据x的大小不同计算得到的值
        return x < 0.5 ?
            16 * x * x * x * x * x :
            1 - pow( -2 * x + 2, 5 ) / 2;
    },
    easeInSine: function (x) {
        // 使用正弦函数实现缓动效果，输入参数为x，返回1减去x乘以π/2的余弦值
        return 1 - cos( x * PI/2 );
    },
    easeOutSine: function (x) {
        // 使用正弦函数实现缓动效果，输入参数为x，返回x乘以π/2的正弦值
        return sin( x * PI/2 );
    },
    easeInOutSine: function (x) {
        // 使用正弦函数实现缓动效果，输入参数为x，返回根据x计算得到的值
        return -( cos( PI * x ) - 1 ) / 2;
    },
    easeInExpo: function (x) {
        // 使用指数函数实现缓动效果，输入参数为x，返回根据x计算得到的值
        return x === 0 ? 0 : pow( 2, 10 * x - 10 );
    },
    easeOutExpo: function (x) {
        // 使用指数函数实现缓动效果，输入参数为x，返回根据x计算得到的值
        return x === 1 ? 1 : 1 - pow( 2, -10 * x );
    },
    easeInOutExpo: function (x) {
        // 使用指数函数实现缓动效果，输入参数为x，返回根据x的大小不同计算得到的值
        return x === 0 ? 0 : x === 1 ? 1 : x < 0.5 ?
            pow( 2, 20 * x - 10 ) / 2 :
            ( 2 - pow( 2, -20 * x + 10 ) ) / 2;
    },
    easeInCirc: function (x) {
        // 使用圆形函数实现缓动效果，输入参数为x，返回根据x计算得到的值
        return 1 - sqrt( 1 - pow( x, 2 ) );
    },
    easeOutCirc: function (x) {
        // 使用圆形函数实现缓动效果，输入参数为x，返回根据x计算得到的值
        return sqrt( 1 - pow( x - 1, 2 ) );
    },
    easeInOutCirc: function (x) {
        // 使用圆形函数实现缓动效果，输入参数为x，返回根据x的大小不同计算得到的值
        return x < 0.5 ?
            ( 1 - sqrt( 1 - pow( 2 * x, 2 ) ) ) / 2 :
            ( sqrt( 1 - pow( -2 * x + 2, 2 ) ) + 1 ) / 2;
    },
    easeInElastic: function (x) {
        // 使用弹性函数实现缓动效果，输入参数为x，返回根据x的大小不同计算得到的值
        return x === 0 ? 0 : x === 1 ? 1 :
            -pow( 2, 10 * x - 10 ) * sin( ( x * 10 - 10.75 ) * c4 );
    },
    easeOutElastic: function (x) {
        // 使用弹性函数实现缓动效果，输入参数为x，返回根据x的大小不同计算得到的值
        return x === 0 ? 0 : x === 1 ? 1 :
            pow( 2, -10 * x ) * sin( ( x * 10 - 0.75 ) * c4 ) + 1;
    },
    easeInOutElastic: function (x) {
        // 使用弹簧缓动函数实现在动画过程中速度的变化
        return x === 0 ? 0 : x === 1 ? 1 : x < 0.5 ?
            // 在前半段时间内，使用弹簧缓动函数计算速度变化
            -( pow( 2, 20 * x - 10 ) * sin( ( 20 * x - 11.125 ) * c5 )) / 2 :
            // 在后半段时间内，使用弹簧缓动函数计算速度变化
            pow( 2, -20 * x + 10 ) * sin( ( 20 * x - 11.125 ) * c5 ) / 2 + 1;
    },
    easeInBack: function (x) {
        // 使用后退缓动函数实现在动画开始时速度的变化
        return c3 * x * x * x - c1 * x * x;
    },
    easeOutBack: function (x) {
        // 使用后退缓动函数实现在动画结束时速度的变化
        return 1 + c3 * pow( x - 1, 3 ) + c1 * pow( x - 1, 2 );
    },
    easeInOutBack: function (x) {
        // 使用后退缓动函数实现在动画过程中速度的变化
        return x < 0.5 ?
            // 在前半段时间内，使用后退缓动函数计算速度变化
            ( pow( 2 * x, 2 ) * ( ( c2 + 1 ) * 2 * x - c2 ) ) / 2 :
            // 在后半段时间内，使用后退缓动函数计算速度变化
            ( pow( 2 * x - 2, 2 ) *( ( c2 + 1 ) * ( x * 2 - 2 ) + c2 ) + 2 ) / 2;
    },
    easeInBounce: function (x) {
        // 使用反弹缓动函数实现在动画开始时速度的变化
        return 1 - bounceOut( 1 - x );
    },
    easeOutBounce: bounceOut,
    easeInOutBounce: function (x) {
        // 使用反弹缓动函数实现在动画过程中速度的变化
        return x < 0.5 ?
            // 在前半段时间内，使用反弹缓动函数计算速度变化
            ( 1 - bounceOut( 1 - 2 * x ) ) / 2 :
            // 在后半段时间内，使用反弹缓动函数计算速度变化
            ( 1 + bounceOut( 2 * x - 1 ) ) / 2;
    }
# 闭合两个嵌套的匿名函数
});
```