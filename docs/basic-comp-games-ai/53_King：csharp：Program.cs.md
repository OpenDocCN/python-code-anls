# `d:/src/tocomm/basic-computer-games\53_King\csharp\Program.cs`

```
# 导入 Games.Common.IO 模块
global using Games.Common.IO;
# 导入 Games.Common.Randomness 模块
global using Games.Common.Randomness;
# 导入 King.Resources 模块
global using King.Resources;
# 导入 King.Resources.Resource 模块中的所有静态成员
global using static King.Resources.Resource;
# 导入 King 模块
using King;

# 创建一个新的游戏对象，使用 ConsoleIO 类和 RandomNumberGenerator 类作为参数，然后开始游戏
new Game(new ConsoleIO(), new RandomNumberGenerator()).Play();
```