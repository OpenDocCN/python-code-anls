# `basic-computer-games\72_Queen\csharp\Program.cs`

```

# 引入 Games.Common.IO 命名空间
global using Games.Common.IO;
# 引入 Games.Common.Randomness 命名空间
global using Games.Common.Randomness;
# 引入 Queen.Resources.Resource 类的静态成员
global using static Queen.Resources.Resource;

# 引入 Queen 命名空间
using Queen;

# 创建一个新的游戏对象，使用 ConsoleIO 类处理输入输出，使用 RandomNumberGenerator 类生成随机数
new Game(new ConsoleIO(), new RandomNumberGenerator()).PlaySeries();

```