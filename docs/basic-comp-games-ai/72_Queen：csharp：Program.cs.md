# `72_Queen\csharp\Program.cs`

```
# 引入命名空间 Games.Common.IO
global using Games.Common.IO;
# 引入命名空间 Games.Common.Randomness
global using Games.Common.Randomness;
# 引入 Queen.Resources.Resource 中的所有静态成员
global using static Queen.Resources.Resource;

# 使用 Queen 命名空间
using Queen;

# 创建一个新的游戏对象，传入 ConsoleIO 和 RandomNumberGenerator 对象作为参数，然后开始游戏系列
new Game(new ConsoleIO(), new RandomNumberGenerator()).PlaySeries();
```