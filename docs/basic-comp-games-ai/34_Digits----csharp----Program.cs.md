# `basic-computer-games\34_Digits\csharp\Program.cs`

```py
# 使用 Digits 命名空间中的全局成员
global using Digits;
# 使用 Games.Common.IO 命名空间中的全局成员
global using Games.Common.IO;
# 使用 Games.Common.Randomness 命名空间中的全局成员
global using Games.Common.Randomness;
# 使用 Digits.Resources.Resource 中的静态成员
global using static Digits.Resources.Resource;

# 创建一个新的 GameSeries 对象，并使用 ConsoleIO 和 RandomNumberGenerator 的实例进行初始化，然后开始游戏
new GameSeries(new ConsoleIO(), new RandomNumberGenerator()).Play();
```