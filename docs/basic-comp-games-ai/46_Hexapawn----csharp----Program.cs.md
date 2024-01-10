# `basic-computer-games\46_Hexapawn\csharp\Program.cs`

```
# 导入Games.Common.IO模块，用于处理输入输出
# 导入Games.Common.Randomness模块，用于生成随机数
# 导入Hexapawn模块，用于进行游戏
# 创建一个新的游戏系列对象，使用控制台输入输出和随机数生成器作为参数
# 开始游戏系列
new GameSeries(new ConsoleIO(), new RandomNumberGenerator()).Play()
```