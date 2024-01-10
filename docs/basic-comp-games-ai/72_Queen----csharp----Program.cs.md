# `basic-computer-games\72_Queen\csharp\Program.cs`

```
# 引入 Games.Common.IO 命名空间
global using Games.Common.IO;
# 引入 Games.Common.Randomness 命名空间
global using Games.Common.Randomness;
# 引入 Queen.Resources.Resource 静态资源
global using static Queen.Resources.Resource;

# 使用 Queen 命名空间
using Queen;

# 创建一个新的游戏对象，并使用控制台输入输出和随机数生成器作为参数，开始游戏系列
new Game(new ConsoleIO(), new RandomNumberGenerator()).PlaySeries();
```