# `ZeroNet\plugins\Sidebar\media_globe\Tween.js`

```
// 定义 TWEEN 对象，如果已经存在则使用已有的对象，否则创建一个新的空对象
var TWEEN=TWEEN||function(){
    // 定义局部变量 a, e, c, d, f，其中 f 为数组
    var a, e, c, d, f=[];
    // 返回一个对象，包含 start、stop、add、remove、update 方法
    return {
        // 定义 start 方法，用于启动动画循环
        start: function(g){
            c = setInterval(this.update, 1E3/(g||60));
        },
        // 定义 stop 方法，用于停止动画循环
        stop: function(){
            clearInterval(c);
        },
        // 定义 add 方法，用于向动画队列中添加动画
        add: function(g){
            f.push(g);
        },
        // 定义 remove 方法，用于从动画队列中移除动画
        remove: function(g){
            a = f.indexOf(g);
            a !== -1 && f.splice(a, 1);
        },
        // 定义 update 方法，用于更新动画队列中的动画状态
        update: function(){
            a = 0;
            e = f.length;
            // 获取当前时间
            for (d = (new Date).getTime(); a < e;){
                // 调用动画的 update 方法，更新动画状态
                if (f[a].update(d)) a++;
                else {
                    // 如果动画已结束，则从队列中移除
                    f.splice(a, 1);
                    e--;
                }
            }
        }
    };
}();
// 定义 TWEEN.Tween 构造函数
TWEEN.Tween = function(a){
    // 定义局部变量 e, c, d, f, g, j, n, k, l, m
    var e = {}, c = {}, d = {}, f = 1000, g = 0, j = null, n = TWEEN.Easing.Linear.EaseNone, k = null, l = null, m = null;
    // 定义 to 方法，用于设置动画的目标值和持续时间
    this.to = function(b, h){
        if (h !== null) f = h;
        for (var i in b)
            if (a[i] !== null) d[i] = b[i];
        return this;
    };
    // 定义 start 方法，用于启动动画
    this.start = function(){
        // 将当前动画添加到动画队列中
        TWEEN.add(this);
        // 设置动画开始时间
        j = (new Date).getTime() + g;
        for (var b in d)
            if (a[b] !== null){
                e[b] = a[b];
                c[b] = d[b] - a[b];
            }
        return this;
    };
    // 定义 stop 方法，用于停止动画
    this.stop = function(){
        // 从动画队列中移除当前动画
        TWEEN.remove(this);
        return this;
    };
    // 定义 delay 方法，用于设置动画延迟时间
    this.delay = function(b){
        g = b;
        return this;
    };
    // 定义 easing 方法，用于设置动画缓动函数
    this.easing = function(b){
        n = b;
        return this;
    };
    // 定义 chain 方法，用于设置动画链
    this.chain = function(b){
        k = b;
    };
    // 定义 onUpdate 方法，用于设置动画更新时的回调函数
    this.onUpdate = function(b){
        l = b;
        return this;
    };
    // 定义 onComplete 方法，用于设置动画完成时的回调函数
    this.onComplete = function(b){
        m = b;
        return this;
    };
    // 定义 update 方法，用于更新动画状态
    this.update = function(b){
        var h, i;
        if (b < j) return true;
        b = (b - j) / f;
        b = b > 1 ? 1 : b;
        i = n(b);
        for (h in c) a[h] = e[h] + c[h] * i;
        l !== null && l.call(a, i);
        if (b == 1){
            m !== null && m.call(a);
            k !== null && k.start();
            return false;
        }
        return true;
    };
};
// 定义 TWEEN.Easing 对象，包含各种缓动函数
TWEEN.Easing = {
    Linear: {},
    Quadratic: {},
    Cubic: {},
    Quartic: {},
    Quintic: {},
    Sinusoidal: {},
    Exponential: {},
    Circular: {},
    Elastic: {},
    Back: {},
    Bounce: {}
};
// 定义各种缓动函数
TWEEN.Easing.Linear.EaseNone = function(a){
    return a;
};
TWEEN.Easing.Quadratic.EaseIn = function(a){
    return a * a;
};
TWEEN.Easing.Quadratic.EaseOut = function(a){
    return -a * (a - 2);
};
TWEEN.Easing.Quadratic.EaseInOut = function(a){
    if ((a *= 2) < 1) return 0.5 * a * a;
    return -0.5 * (--a * (a - 2) - 1);
};
TWEEN.Easing.Cubic.EaseIn = function(a){
    return a * a * a;
};
TWEEN.Easing.Cubic.EaseOut = function(a){
    return --a * a * a + 1;
};
TWEEN.Easing.Cubic.EaseInOut = function(a){
    if ((a *= 2) < 1) return 0.5 * a * a * a;
    return 0.5 * ((a -= 2) * a * a + 2);
};
TWEEN.Easing.Quartic.EaseIn = function(a){
    return a * a * a * a;
};
# Quartic 缓动函数的 EaseOut 版本，根据输入参数计算缓动效果
TWEEN.Easing.Quartic.EaseOut=function(a){return-(--a*a*a*a-1)};
# Quartic 缓动函数的 EaseInOut 版本，根据输入参数计算缓动效果
TWEEN.Easing.Quartic.EaseInOut=function(a){if((a*=2)<1)return 0.5*a*a*a*a;return-0.5*((a-=2)*a*a*a-2)};
# Quintic 缓动函数的 EaseIn 版本，根据输入参数计算缓动效果
TWEEN.Easing.Quintic.EaseIn=function(a){return a*a*a*a*a};
# Quintic 缓动函数的 EaseOut 版本，根据输入参数计算缓动效果
TWEEN.Easing.Quintic.EaseOut=function(a){return(a-=1)*a*a*a*a+1};
# Quintic 缓动函数的 EaseInOut 版本，根据输入参数计算缓动效果
TWEEN.Easing.Quintic.EaseInOut=function(a){if((a*=2)<1)return 0.5*a*a*a*a*a;return 0.5*((a-=2)*a*a*a*a+2)};
# Sinusoidal 缓动函数的 EaseIn 版本，根据输入参数计算缓动效果
TWEEN.Easing.Sinusoidal.EaseIn=function(a){return-Math.cos(a*Math.PI/2)+1};
# Sinusoidal 缓动函数的 EaseOut 版本，根据输入参数计算缓动效果
TWEEN.Easing.Sinusoidal.EaseOut=function(a){return Math.sin(a*Math.PI/2)};
# Sinusoidal 缓动函数的 EaseInOut 版本，根据输入参数计算缓动效果
TWEEN.Easing.Sinusoidal.EaseInOut=function(a){return-0.5*(Math.cos(Math.PI*a)-1)};
# Exponential 缓动函数的 EaseIn 版本，根据输入参数计算缓动效果
TWEEN.Easing.Exponential.EaseIn=function(a){return a==0?0:Math.pow(2,10*(a-1))};
# Exponential 缓动函数的 EaseOut 版本，根据输入参数计算缓动效果
TWEEN.Easing.Exponential.EaseOut=function(a){return a==1?1:-Math.pow(2,-10*a)+1};
# Exponential 缓动函数的 EaseInOut 版本，根据输入参数计算缓动效果
TWEEN.Easing.Exponential.EaseInOut=function(a){if(a==0)return 0;if(a==1)return 1;if((a*=2)<1)return 0.5*Math.pow(2,10*(a-1));return 0.5*(-Math.pow(2,-10*(a-1))+2)};
# Circular 缓动函数的 EaseIn 版本，根据输入参数计算缓动效果
TWEEN.Easing.Circular.EaseIn=function(a){return-(Math.sqrt(1-a*a)-1)};
# Circular 缓动函数的 EaseOut 版本，根据输入参数计算缓动效果
TWEEN.Easing.Circular.EaseOut=function(a){return Math.sqrt(1- --a*a)};
# Circular 缓动函数的 EaseInOut 版本，根据输入参数计算缓动效果
TWEEN.Easing.Circular.EaseInOut=function(a){if((a/=0.5)<1)return-0.5*(Math.sqrt(1-a*a)-1);return 0.5*(Math.sqrt(1-(a-=2)*a)+1)};
# Elastic 缓动函数的 EaseIn 版本，根据输入参数计算缓动效果
TWEEN.Easing.Elastic.EaseIn=function(a){var e,c=0.1,d=0.4;if(a==0)return 0;if(a==1)return 1;d||(d=0.3);if(!c||c<1){c=1;e=d/4}else e=d/(2*Math.PI)*Math.asin(1/c);return-(c*Math.pow(2,10*(a-=1))*Math.sin((a-e)*2*Math.PI/d))};
# Elastic 缓动函数的 EaseOut 版本，根据输入参数计算缓动效果
TWEEN.Easing.Elastic.EaseOut=function(a){var e,c=0.1,d=0.4;if(a==0)return 0;if(a==1)return 1;d||(d=0.3);if(!c||c<1){c=1;e=d/4}else e=d/(2*Math.PI)*Math.asin(1/c);return c*Math.pow(2,-10*a)*Math.sin((a-e)*2*Math.PI/d)+1};
// 定义 Elastic 缓动函数的 EaseInOut 方法
TWEEN.Easing.Elastic.EaseInOut=function(a){
    var e,c=0.1,d=0.4;
    if(a==0) return 0;
    if(a==1) return 1;
    if(d==0) d=0.3;
    if(!c || c<1){
        c=1;
        e=d/4;
    } else {
        e=d/(2*Math.PI)*Math.asin(1/c);
    }
    if((a*=2)<1) return -0.5*c*Math.pow(2,10*(a-=1))*Math.sin((a-e)*2*Math.PI/d);
    return c*Math.pow(2,-10*(a-=1))*Math.sin((a-e)*2*Math.PI/d)*0.5+1;
};

// 定义 Back 缓动函数的 EaseIn 方法
TWEEN.Easing.Back.EaseIn=function(a){
    return a*a*(2.70158*a-1.70158);
};

// 定义 Back 缓动函数的 EaseOut 方法
TWEEN.Easing.Back.EaseOut=function(a){
    return (a-=1)*a*(2.70158*a+1.70158)+1;
};

// 定义 Back 缓动函数的 EaseInOut 方法
TWEEN.Easing.Back.EaseInOut=function(a){
    if((a*=2)<1) return 0.5*a*a*(3.5949095*a-2.5949095);
    return 0.5*((a-=2)*a*(3.5949095*a+2.5949095)+2);
};

// 定义 Bounce 缓动函数的 EaseIn 方法
TWEEN.Easing.Bounce.EaseIn=function(a){
    return 1-TWEEN.Easing.Bounce.EaseOut(1-a);
};

// 定义 Bounce 缓动函数的 EaseOut 方法
TWEEN.Easing.Bounce.EaseOut=function(a){
    return (a/=1)<1/2.75?7.5625*a*a:a<2/2.75?7.5625*(a-=1.5/2.75)*a+0.75:a<2.5/2.75?7.5625*(a-=2.25/2.75)*a+0.9375:7.5625*(a-=2.625/2.75)*a+0.984375;
};

// 定义 Bounce 缓动函数的 EaseInOut 方法
TWEEN.Easing.Bounce.EaseInOut=function(a){
    if(a<0.5) return TWEEN.Easing.Bounce.EaseIn(a*2)*0.5;
    return TWEEN.Easing.Bounce.EaseOut(a*2-1)*0.5+0.5;
};
```