# `.\lucidrains\liquid-conway\src\helpers.js`

```
# 从 lodash.clone 模块中导入 clone 函数
import clone from 'lodash.clone';

# 初始化一个包含 num 个元素的数组，每个元素都是 init 的克隆
function initArray(num, init) {
    return Array.from(Array(num)).map(() => clone(init));
}

# 生成一个从 low 到 high 的范围数组，步长为 step，默认为 1
function range(low, high, step = 1) {
    const arr = [];
    for (let i = low; i <= high; i += step) {
        arr.push(i);
    }
    return arr;
}

# 缓存函数的结果，使用 cacheObj 存储结果，deriveKeyFn 用于生成缓存的键
function cacheFn(fn, cacheObj, deriveKeyFn) {
    return (...args) => {
        let key;
        if (!deriveKeyFn) {
            key = JSON.stringify(args);
        } else {
            key = deriveKeyFn(...args);
        }

        if (cacheObj[key] !== undefined) {
            return cacheObj[key];
        }

        const ret = fn(...args);
        cacheObj[key] = ret;
        return ret;
    };
}

# 生成一个小于 num 的随机整数
function randInt(num) {
    return Math.floor(Math.random() * (num + 1));
}

# 导出包含 cacheFn、range、initArray、randInt 函数的对象
export default {
    cacheFn,
    range,
    initArray,
    randInt
};
```