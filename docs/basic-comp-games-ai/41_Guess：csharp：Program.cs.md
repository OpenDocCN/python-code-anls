# `d:/src/tocomm/basic-computer-games\41_Guess\csharp\Program.cs`

```
# 导入 Games.Common.IO 模块
global using Games.Common.IO;
# 导入 Games.Common.Randomness 模块
global using Games.Common.Randomness;
# 导入 Guess.Resources.Resource 模块中的所有静态成员
global using static Guess.Resources.Resource;  

# 导入 Guess 模块
using Guess;

# 创建一个新的游戏对象，使用 ConsoleIO 类处理输入输出，使用 RandomNumberGenerator 类生成随机数，并开始游戏
new Game(new ConsoleIO(), new RandomNumberGenerator()).Play();
```