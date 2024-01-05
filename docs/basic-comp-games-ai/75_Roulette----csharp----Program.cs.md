# `75_Roulette\csharp\Program.cs`

```
# 导入 Games.Common.IO 模块
global using Games.Common.IO;
# 导入 Games.Common.Randomness 模块
global using Games.Common.Randomness;
# 导入 Roulette.Resources.Resource 模块中的所有内容
global using static Roulette.Resources.Resource;
# 导入 Roulette 模块
using Roulette;

# 创建一个新的游戏对象，使用 ConsoleIO 对象和 RandomNumberGenerator 对象作为参数，然后开始游戏
new Game(new ConsoleIO(), new RandomNumberGenerator()).Play();
```