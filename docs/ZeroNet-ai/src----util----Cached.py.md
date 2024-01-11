# `ZeroNet\src\util\Cached.py`

```
import time  # 导入时间模块


class Cached(object):  # 定义一个缓存类
    def __init__(self, timeout):  # 初始化方法，设置缓存字典和超时时间
        self.cache_db = {}  # 初始化缓存字典
        self.timeout = timeout  # 设置超时时间

    def __call__(self, func):  # 定义一个装饰器方法
        def wrapper(*args, **kwargs):  # 定义一个包装器方法
            key = "%s %s" % (args, kwargs)  # 生成缓存键值
            cached_value = None  # 初始化缓存值
            cache_hit = False  # 初始化缓存命中标志

            if key in self.cache_db:  # 如果缓存键值存在于缓存字典中
                cache_hit = True  # 设置缓存命中标志为True
                cached_value, time_cached_end = self.cache_db[key]  # 获取缓存值和缓存结束时间
                if time.time() > time_cached_end:  # 如果当前时间超过了缓存结束时间
                    self.cleanupExpired()  # 清理过期缓存
                    cached_value = None  # 重置缓存值
                    cache_hit = False  # 重置缓存命中标志

            if cache_hit:  # 如果缓存命中
                return cached_value  # 返回缓存值
            else:  # 如果缓存未命中
                cached_value = func(*args, **kwargs)  # 调用原始函数获取结果
                time_cached_end = time.time() + self.timeout  # 计算缓存结束时间
                self.cache_db[key] = (cached_value, time_cached_end)  # 将结果和缓存结束时间存入缓存字典
                return cached_value  # 返回结果

        wrapper.emptyCache = self.emptyCache  # 将清空缓存方法添加到包装器对象中

        return wrapper  # 返回包装器方法

    def cleanupExpired(self):  # 定义清理过期缓存的方法
        for key in list(self.cache_db.keys()):  # 遍历缓存字典的键列表
            cached_value, time_cached_end = self.cache_db[key]  # 获取缓存值和缓存结束时间
            if time.time() > time_cached_end:  # 如果当前时间超过了缓存结束时间
                del(self.cache_db[key])  # 从缓存字典中删除该缓存

    def emptyCache(self):  # 定义清空缓存的方法
        num = len(self.cache_db)  # 获取当前缓存字典的长度
        self.cache_db.clear()  # 清空缓存字典
        return num  # 返回清空的缓存数量


if __name__ == "__main__":  # 如果当前脚本被直接执行
    from gevent import monkey  # 导入monkey模块
    monkey.patch_all()  # 打补丁，使得gevent能够识别time.sleep

    @Cached(timeout=2)  # 使用装饰器缓存calcAdd函数的结果，超时时间为2秒
    def calcAdd(a, b):  # 定义一个加法函数
        print("CalcAdd", a, b)  # 打印调用信息
        return a + b  # 返回加法结果

    @Cached(timeout=1)  # 使用装饰器缓存calcMultiply函数的结果，超时时间为1秒
    def calcMultiply(a, b):  # 定义一个乘法函数
        print("calcMultiply", a, b)  # 打印调用信息
        return a * b  # 返回乘法结果

    for i in range(5):  # 循环5次
        print("---")  # 打印分隔线
        print("Emptied", calcAdd.emptyCache())  # 打印清空缓存前的缓存数量
        assert calcAdd(1, 2) == 3  # 断言调用calcAdd函数返回结果为3
        print("Emptied", calcAdd.emptyCache())  # 打印清空缓存后的缓存数量
        assert calcAdd(1, 2) == 3  # 断言调用calcAdd函数返回结果为3
        assert calcAdd(2, 3) == 5  # 断言调用calcAdd函数返回结果为5
        assert calcMultiply(2, 3) == 6  # 断言调用calcMultiply函数返回结果为6
        time.sleep(1)  # 休眠1秒
```