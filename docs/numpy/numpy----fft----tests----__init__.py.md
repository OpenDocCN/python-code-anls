# `.\numpy\numpy\fft\tests\__init__.py`

```
# 导入 datetime 模块中的 datetime 类
from datetime import datetime

# 定义一个名为 parse_timestamp 的函数，接收一个字符串参数 ts
def parse_timestamp(ts):
    # 使用 datetime 类的 strptime 方法解析时间戳字符串为 datetime 对象
    dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
    # 返回解析后的 datetime 对象
    return dt
```