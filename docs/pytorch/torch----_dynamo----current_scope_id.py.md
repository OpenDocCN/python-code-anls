# `.\pytorch\torch\_dynamo\current_scope_id.py`

```py
# 引入用于允许未标注的函数定义的类型检查配置
# 这里导入了contextlib和threading模块
import contextlib
import threading

# 全局变量，用于标识当前所在的子图追踪器（SubgraphTracer）
# 有时候很难找到适合使用的指令翻译器（InstructionTranslator）
_current_scope_id = threading.local()

# 定义一个函数current_scope_id，用于获取当前的作用域ID
def current_scope_id():
    global _current_scope_id
    # 如果_current_scope_id对象没有"value"属性，则初始化为1
    if not hasattr(_current_scope_id, "value"):
        _current_scope_id.value = 1
    return _current_scope_id.value

# 定义一个上下文管理器enter_new_scope，用于进入新的作用域
@contextlib.contextmanager
def enter_new_scope():
    global _current_scope_id
    try:
        # 将当前作用域ID增加1，进入新的作用域
        _current_scope_id.value = current_scope_id() + 1
        yield
    finally:
        # 退出作用域时，将当前作用域ID减少1
        _current_scope_id.value = current_scope_id() - 1
```