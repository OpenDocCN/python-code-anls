# `basic-computer-games\07_Basketball\csharp\Program.cs`

```

# 导入篮球游戏模块
using Basketball;
# 导入通用输入输出模块
using Games.Common.IO;
# 导入通用随机数生成模块
using Games.Common.Randomness;

# 创建一个游戏对象，使用控制台输入输出和随机数生成器作为参数
var game = Game.Create(new ConsoleIO(), new RandomNumberGenerator());

# 开始游戏
game.Play();

```