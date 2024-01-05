# `d:/src/tocomm/basic-computer-games\16_Bug\csharp\Program.cs`

```
# 导入 BugGame 模块
import BugGame
# 导入 Games.Common.IO 模块
import Games.Common.IO
# 导入 Games.Common.Randomness 模块
import Games.Common.Randomness

# 创建一个新的游戏对象，使用控制台输入输出和随机数生成器作为参数
new Game(new ConsoleIO(), new RandomNumberGenerator()).Play()
```