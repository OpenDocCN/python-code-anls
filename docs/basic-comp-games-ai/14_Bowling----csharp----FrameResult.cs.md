# `14_Bowling\csharp\FrameResult.cs`

```
# 定义 FrameResult 类，用于表示一轮保龄球比赛的结果
class FrameResult:
    # 定义枚举类型 Points，表示得分情况
    class Points(Enum):
        None = 0  # 无得分
        Error = 1  # 出错
        Spare = 2  # 补中
        Strike = 3  # 全中

    # 定义属性 PinsBall1，表示第一球击倒的瓶数
    # 定义属性 PinsBall2，表示第二球击倒的瓶数
    # 定义属性 Score，表示该轮得分情况
    def __init__(self):
        self.PinsBall1 = 0
        self.PinsBall2 = 0
        self.Score = Points.None

    # 重置该轮比赛的结果，将击倒瓶数和得分情况都重置为初始值
    def reset(self):
        self.PinsBall1 = 0
        self.PinsBall2 = 0
        self.Score = Points.None
抱歉，给定的代码片段不完整，缺少了大部分内容，无法为其添加注释。如果您有完整的代码片段需要解释，请提供完整的代码片段。谢谢！
```