# `basic-computer-games\30_Cube\csharp\Program.cs`

```py
# 引入 Games.Common.IO 命名空间
global using Games.Common.IO;
# 引入 Games.Common.Randomness 命名空间
global using Games.Common.Randomness;
# 引入 Cube.Resources.Resource 静态资源
global using static Cube.Resources.Resource;

# 引入 Cube 命名空间
using Cube;

# 根据命令行参数判断是否使用非随机数生成器，选择合适的随机数生成器
IRandom random = args.Contains("--non-random") ? new ZerosGenerator() : new RandomNumberGenerator();

# 创建游戏对象，使用控制台输入输出和随机数生成器
new Game(new ConsoleIO(), random).Play();
```