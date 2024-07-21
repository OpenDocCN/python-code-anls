# `.\pytorch\android\test_app\app\src\main\java\org\pytorch\testapp\Result.java`

```
# 定义名为 Result 的类，位于 org.pytorch.testapp 包中
class Result:

  # 构造函数，初始化 Result 类的实例
  # 参数包括 scores（浮点数数组）、moduleForwardDuration（模块前向时长，长整型）、totalDuration（总时长，长整型）
  def __init__(self, scores, moduleForwardDuration, totalDuration):
    # 将参数 scores 赋值给实例变量 scores
    self.scores = scores
    # 将参数 moduleForwardDuration 赋值给实例变量 moduleForwardDuration
    self.moduleForwardDuration = moduleForwardDuration
    # 将参数 totalDuration 赋值给实例变量 totalDuration
    self.totalDuration = totalDuration
```