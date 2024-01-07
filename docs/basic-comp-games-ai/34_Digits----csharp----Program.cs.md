# `basic-computer-games\34_Digits\csharp\Program.cs`

```

# 使用 Digits 命名空间中的类和方法
global using Digits;
# 使用 Games.Common.IO 命名空间中的类和方法
global using Games.Common.IO;
# 使用 Games.Common.Randomness 命名空间中的类和方法
global using Games.Common.Randomness;
# 使用 Digits 命名空间中的资源文件
global using static Digits.Resources.Resource;

# 创建一个新的游戏系列对象，使用控制台输入输出和随机数生成器作为参数
new GameSeries(new ConsoleIO(), new RandomNumberGenerator()).Play();

```