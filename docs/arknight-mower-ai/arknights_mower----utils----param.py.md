# `arknights-mower\arknights_mower\utils\param.py`

```py
# 导入ParamArgs类型别名
from .typealias import ParamArgs

# 定义参数错误的异常类
class ParamError(ValueError):
    """ 参数错误 """

# 解析操作参数的函数
def parse_operation_params(args: ParamArgs = []):
    # 初始化变量
    level = None
    times = -1
    potion = 0
    originite = 0
    eliminate = 0

    try:
        # 遍历参数列表
        for p in args:
            # 判断参数是否以'-'开头
            if p[0] == '-':
                val = -1
                # 判断参数长度是否大于2
                if len(p) > 2:
                    val = int(p[2:])
                # 根据参数标识进行不同的处理
                if p[1] == 'r':
                    assert potion == 0
                    potion = val
                elif p[1] == 'R':
                    assert originite == 0
                    originite = val
                elif p[1] == 'e':
                    assert eliminate == 0
                    eliminate = 1
                elif p[1] == 'E':
                    assert eliminate == 0
                    eliminate = 2
            # 如果参数不包含'-'，则处理为次数
            elif p.find('-') == -1:
                assert times == -1
                times = int(p)
            # 否则处理为级别
            else:
                assert level is None
                level = p
    # 捕获异常并抛出参数错误异常
    except Exception:
        raise ParamError
    # 返回解析后的参数
    return level, times, potion, originite, eliminate

# 获取操作次数的函数
def operation_times(args: ParamArgs = []) -> int:
    # 调用解析参数函数获取次数
    _, times, _, _, _ = parse_operation_params(args)
    # 返回次数
    return times
```