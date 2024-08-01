# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\75fc9c18-f5c95b15762b1b2f.js`

```py
// 设置变量 t.version 为 "2.30.1"，并将 tr 函数赋给 V 变量，将 tj 函数赋给 t.fn 变量
// 定义 t.min 函数，接受任意数量参数，返回通过 tu("isBefore", ...) 调用的结果
t.version="2.30.1",V=tr,t.fn=tj,t.min=function(){var e=[].slice.call(arguments,0);return tu("isBefore",e)},
// 定义 t.max 函数，接受任意数量参数，返回通过 tu("isAfter", ...) 调用的结果
// 定义 t.now 函数，返回当前时间的时间戳，Date.now() 或 +new Date 的兼容写法
t.max=function(){var e=[].slice.call(arguments,0);return tu("isAfter",e)},t.now=function(){return Date.now?Date.now():+new Date},
// 将 d 函数赋给 t.utc
t.utc=d,
// 定义 t.unix 函数，接受一个参数 e，返回时间戳为 1000 * e 的日期对象
t.unix=function(e){return tr(1e3*e)},
// 定义 t.months 函数，接受两个参数 e 和 t，返回通过 tq(e, t, "months") 调用的结果
// 定义 t.isDate 函数，检查参数是否为日期对象
t.months=function(e,t){return tq(e,t,"months")},
t.isDate=u,
// 将 eJ 函数赋给 t.locale
t.locale=eJ,
// 将 m 函数赋给 t.invalid
t.invalid=m,
// 将 tk 函数赋给 t.duration
t.duration=tk,
// 将 k 函数赋给 t.isMoment
t.isMoment=k,
// 定义 t.weekdays 函数，接受三个参数 e、t、n，返回通过 tB(e, t, n, "weekdays") 调用的结果
// 将 eX 函数赋给 t.localeData
t.weekdays=function(e,t,n){return tB(e,t,n,"weekdays")},
t.parseZone=function(){return tr.apply(null,arguments).parseZone()},
// 将 eX 函数赋给 t.localeData
t.localeData=eX,
// 将 td 函数赋给 t.isDuration
t.isDuration=td,
// 定义 t.monthsShort 函数，接受两个参数 e 和 t，返回通过 tq(e, t, "monthsShort") 调用的结果
t.monthsShort=function(e,t){return tq(e,t,"monthsShort")},
// 定义 t.weekdaysMin 函数，接受三个参数 e、t、n，返回通过 tB(e, t, n, "weekdaysMin") 调用的结果
t.weekdaysMin=function(e,t,n){return tB(e,t,n,"weekdaysMin")},
// 将 eQ 函数赋给 t.defineLocale
t.defineLocale=eQ,
// 定义 t.updateLocale 函数，更新或删除指定语言环境的配置信息，并返回更新后的语言环境对象
t.updateLocale=function(e,t){
    // 如果传入 t 参数，则更新指定语言环境的配置信息
    if(null!=t){
        var n,s,i=eZ;
        // 如果当前语言环境已存在，并且有父语言环境，则使用其配置信息
        if(null!=ez[e] && null!=ez[e].parentLocale){
            ez[e].set(b(ez[e]._config,t));
        } else {
            // 否则，如果存在同名语言环境，则继承其配置信息
            null!=(s=eB(e)) && (i=s._config);
            t=b(i,t);
            // 如果不存在同名语言环境，则创建一个新的语言环境对象
            null==s && (t.abbr=e);
            n=new T(t);
            n.parentLocale=ez[e];
            ez[e]=n;
        }
        // 更新语言环境数据
        eJ(e);
    } else {
        // 如果没有传入 t 参数，则删除指定语言环境的配置信息
        null!=ez[e] && (null!=ez[e].parentLocale ? (ez[e]=ez[e].parentLocale,e===eJ() && eJ(e)) : null!=ez[e] && delete ez[e]);
    }
    // 返回更新后的语言环境对象
    return ez[e];
},
// 定义 t.locales 函数，返回所有已定义的语言环境对象列表
t.locales=function(){return A(ez)},
// 定义 t.relativeTimeRounding 函数，设置或返回相对时间的舍入方式
t.relativeTimeRounding=function(e){return void 0===e?nl:"function"==typeof e&&(nl=e,!0)},
// 定义 t.relativeTimeThreshold 函数，设置或返回相对时间的阈值
t.relativeTimeThreshold=function(e,t){return void 0!==nh[e]&&(void 0===t?nh[e]:(nh[e]=t,"s"===e&&(nh.ss=t-1),!0))},
// 定义 t.calendarFormat 函数，返回指定日期的日历格式字符串
t.calendarFormat=function(e,t){var n=e.diff(t,"days",!0);return n<-6?"sameElse":n<-1?"lastWeek":n<0?"lastDay":n<1?"sameDay":n<2?"nextDay":n<7?"nextWeek":"sameElse"},
// 将 tj 函数赋给 t.prototype
t.prototype=tj,
// 定义 t.HTML5_FMT 对象，包含一系列日期时间格式字符串
t.HTML5_FMT={DATETIME_LOCAL:"YYYY-MM-DDTHH:mm",DATETIME_LOCAL_SECONDS:"YYYY-MM-DDTHH:mm:ss",DATETIME_LOCAL_MS:"YYYY-MM-DDTHH:mm:ss.SSS",DATE:"YYYY-MM-DD",TIME:"HH:mm",TIME_SECONDS:"HH:mm:ss",TIME_MS:"HH:mm:ss.SSS",WEEK:"GGGG-[W]WW",MONTH:"YYYY-MM"}
// 返回 t 对象
t}()}}]);
```