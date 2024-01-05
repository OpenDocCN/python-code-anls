# `14_Bowling\csharp\GameResults.cs`

```
# 创建一个名为 GameResults 的类
class GameResults:
    # 定义一个名为 FramesPerGame 的静态只读变量，值为 10
    FramesPerGame = 10
    # 定义一个名为 Results 的属性，用于存储 FrameResult 对象的数组
    Results = []

    # 定义一个构造函数，初始化 Results 数组，每个元素都是一个 FrameResult 对象
    def __init__(self):
        # 初始化 Results 数组，长度为 FramesPerGame，每个元素都是一个 FrameResult 对象
        self.Results = [FrameResult() for _ in range(self.FramesPerGame)]
抱歉，给定的代码片段不完整，无法为其添加注释。
```