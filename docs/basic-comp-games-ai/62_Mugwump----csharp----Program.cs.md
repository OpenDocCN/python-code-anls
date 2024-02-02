# `basic-computer-games\62_Mugwump\csharp\Program.cs`

```py
# 全局引用 System 命名空间
global using System;
# 全局引用 Games.Common.IO 命名空间
global using Games.Common.IO;
# 全局引用 Games.Common.Randomness 命名空间
global using Games.Common.Randomness;

# 引用 Mugwump 游戏库
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