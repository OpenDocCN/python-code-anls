# `34_Digits\csharp\Program.cs`

```
# 导入 Digits 命名空间
global using Digits;
# 导入 Games.Common.IO 命名空间
global using Games.Common.IO;
# 导入 Games.Common.Randomness 命名空间
global using Games.Common.Randomness;
# 导入 Digits.Resources.Resource 中的所有静态成员
global using static Digits.Resources.Resource;

# 创建一个新的 GameSeries 对象，使用 ConsoleIO 和 RandomNumberGenerator 的实例作为参数，然后调用 Play 方法开始游戏
new GameSeries(new ConsoleIO(), new RandomNumberGenerator()).Play();
```