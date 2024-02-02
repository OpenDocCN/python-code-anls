# `basic-computer-games\75_Roulette\csharp\Program.cs`

```py
# 引入 Games.Common.IO 命名空间
global using Games.Common.IO;
# 引入 Games.Common.Randomness 命名空间
global using Games.Common.Randomness;
# 引入 Roulette.Resources.Resource 静态资源
global using static Roulette.Resources.Resource;
# 引入 Roulette 命名空间
using Roulette;

# 创建一个新的游戏对象，使用控制台输入输出和随机数生成器作为参数，然后开始游戏
new Game(new ConsoleIO(), new RandomNumberGenerator()).Play();
```