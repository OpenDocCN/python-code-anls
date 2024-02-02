# `ZeroNet\src\util\RateLimit.py`

```py
# 导入时间、gevent和日志模块
import time
import gevent
import logging

# 获取名为"RateLimit"的日志记录器
log = logging.getLogger("RateLimit")

# 用于存储事件的最后调用时间的字典
called_db = {}  
# 用于存储排队运行的命令的字典
queue_db = {}  

# 注册事件的调用，并设置惩罚时间
# 返回：无
def called(event, penalty=0):
    called_db[event] = time.time() + penalty

# 检查是否允许调用事件
# 返回：如果允许则返回True，否则返回False
def isAllowed(event, allowed_again=10):
    last_called = called_db.get(event)
    if not last_called:  # 如果之前没有调用过
        return True
    elif time.time() - last_called >= allowed_again:  # 如果距离上次调用已经超过了允许的时间间隔
        del called_db[event]  # 删除上次调用的时间，以节省内存
        return True
    else:
        return False

# 返回事件的延迟时间
def delayLeft(event, allowed_again=10):
    last_called = called_db.get(event)
    if not last_called:  # 如果之前没有调用过
        return 0
    else:
        return allowed_again - (time.time() - last_called)

# 调用排队中的事件
def callQueue(event):
    func, args, kwargs, thread = queue_db[event]
    log.debug("Calling: %s" % event)  # 记录调用事件的信息
    called(event)  # 标记事件已被调用
    del queue_db[event]  # 从排队中删除事件
    return func(*args, **kwargs)  # 调用事件对应的函数

# 限制调用频率并延迟函数调用（如果有必要）
# 如果在限制时间内再次调用函数，则之前排队的调用将被丢弃
# 返回：立即执行的gevent线程
def callAsync(event, allowed_again=10, func=None, *args, **kwargs):
    if isAllowed(event, allowed_again):  # 如果最近没有调用过，则立即调用
        called(event)  # 标记事件已被调用
        return gevent.spawn(func, *args, **kwargs)  # 创建并返回一个gevent线程
    # 如果函数调用最近被调用过，则将其安排在稍后执行
    time_left = allowed_again - max(0, time.time() - called_db[event])  # 计算距离下次执行还剩余的时间
    log.debug("Added to queue (%.2fs left): %s " % (time_left, event))  # 记录日志，将函数添加到队列中并显示剩余时间
    if not queue_db.get(event):  # 如果函数调用还没有被加入到队列中
        thread = gevent.spawn_later(time_left, lambda: callQueue(event))  # 使用gevent在一段时间后调用这个函数
        queue_db[event] = (func, args, kwargs, thread)  # 将函数调用信息加入到队列数据库中
        return thread  # 返回线程对象
    else:  # 如果函数调用已经在队列中，只需更新参数
        thread = queue_db[event][3]  # 获取已经在队列中的线程对象
        queue_db[event] = (func, args, kwargs, thread)  # 更新函数调用信息
        return thread  # 返回线程对象
# Rate limit and delay function call if needed
# Return: Wait for execution/delay then return value
# 如果需要，对函数调用进行速率限制和延迟
# 返回：等待执行/延迟然后返回值
def call(event, allowed_again=10, func=None, *args, **kwargs):
    # 如果允许调用
    if isAllowed(event):  # Not called recently, call it now
        called(event)
        # print "Calling now", allowed_again
        return func(*args, **kwargs)

    else:  # Called recently, schedule it for later
        # 计算剩余时间
        time_left = max(0, allowed_again - (time.time() - called_db[event]))
        # print "Time left: %s" % time_left, args, kwargs
        log.debug("Calling sync (%.2fs left): %s" % (time_left, event))
        called(event, time_left)
        time.sleep(time_left)
        back = func(*args, **kwargs)
        called(event)
        return back


# Cleanup expired events every 3 minutes
# 每3分钟清理过期事件
def rateLimitCleanup():
    while 1:
        expired = time.time() - 60 * 2  # Cleanup if older than 2 minutes
        for event in list(called_db.keys()):
            if called_db[event] < expired:
                del called_db[event]
        time.sleep(60 * 3)  # Every 3 minutes
gevent.spawn(rateLimitCleanup)


if __name__ == "__main__":
    from gevent import monkey
    monkey.patch_all()
    import random

    def publish(inner_path):
        print("Publishing %s..." % inner_path)
        return 1

    def cb(thread):
        print("Value:", thread.value)

    print("Testing async spam requests rate limit to 1/sec...")
    for i in range(3000):
        thread = callAsync("publish content.json", 1, publish, "content.json %s" % i)
        time.sleep(float(random.randint(1, 20)) / 100000)
    print(thread.link(cb))
    print("Done")

    time.sleep(2)

    print("Testing sync spam requests rate limit to 1/sec...")
    for i in range(5):
        call("publish data.json", 1, publish, "data.json %s" % i)
        time.sleep(float(random.randint(1, 100)) / 100)
    print("Done")

    print("Testing cleanup")
    thread = callAsync("publish content.json single", 1, publish, "content.json single")
    # 打印需要清理的数据库名称
    print("Needs to cleanup:", called_db, queue_db)
    # 打印等待3分钟进行清理过程
    print("Waiting 3min for cleanup process...")
    # 休眠3分钟
    time.sleep(60 * 3)
    # 打印已清理的数据库名称
    print("Cleaned up:", called_db, queue_db)
```