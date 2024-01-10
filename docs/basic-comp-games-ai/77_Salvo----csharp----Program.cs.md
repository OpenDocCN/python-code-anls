# `basic-computer-games\77_Salvo\csharp\Program.cs`

```
# 导入 System 命名空间
global using System;
# 导入 Games.Common.IO 命名空间
global using Games.Common.IO;
# 导入 Games.Common.Randomness 命名空间
global using Games.Common.Randomness;
# 导入 Salvo 命名空间
global using Salvo;
# 导入 Salvo.Ships 命名空间
global using Salvo.Ships;
# 导入 Salvo.Resources.Resource 中的所有静态成员
global using static Salvo.Resources.Resource;

# 创建一个新的游戏对象，并使用控制台输入输出和数据随机数生成器作为参数，然后开始游戏
# new Game(new ConsoleIO(), new RandomNumberGenerator()).Play();
new Game(new ConsoleIO(), new DataRandom()).Play();
```