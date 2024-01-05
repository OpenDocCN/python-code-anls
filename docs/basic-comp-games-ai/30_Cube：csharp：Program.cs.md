# `d:/src/tocomm/basic-computer-games\30_Cube\csharp\Program.cs`

```
# 导入 Games.Common.IO 模块
global using Games.Common.IO;
# 导入 Games.Common.Randomness 模块
global using Games.Common.Randomness;
# 导入 Cube.Resources.Resource 模块中的所有静态成员
global using static Cube.Resources.Resource;

# 导入 Cube 模块
using Cube;

# 根据命令行参数判断是否使用非随机数生成器，选择合适的随机数生成器
IRandom random = args.Contains("--non-random") ? new ZerosGenerator() : new RandomNumberGenerator();

# 创建一个新的游戏对象，使用控制台输入输出和选择的随机数生成器
new Game(new ConsoleIO(), random).Play();
```