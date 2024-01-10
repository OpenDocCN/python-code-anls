# `basic-computer-games\16_Bug\csharp\Program.cs`

```
# 导入BugGame模块
using BugGame;
# 导入Games.Common.IO模块
using Games.Common.IO;
# 导入Games.Common.Randomness模块
using Games.Common.Randomness;
# 创建一个新的游戏对象，并使用控制台输入输出和随机数生成器作为参数进行初始化
new Game(new ConsoleIO(), new RandomNumberGenerator()).Play();
```