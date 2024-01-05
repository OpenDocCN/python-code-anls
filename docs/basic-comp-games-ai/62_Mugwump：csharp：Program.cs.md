# `d:/src/tocomm/basic-computer-games\62_Mugwump\csharp\Program.cs`

```
# 导入 System 命名空间
global using System;
# 导入 Games.Common.IO 命名空间
global using Games.Common.IO;
# 导入 Games.Common.Randomness 命名空间
global using Games.Common.Randomness;

# 导入 Mugwump 命名空间
using Mugwump;

# 创建一个随机数生成器对象
var random = new RandomNumberGenerator();
# 创建一个控制台输入输出对象
var io = new ConsoleIO();

# 创建一个游戏对象，传入控制台输入输出对象和随机数生成器对象
var game = new Game(io, random);

# 开始游戏
game.Play();
```