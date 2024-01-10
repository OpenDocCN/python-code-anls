# `basic-computer-games\71_Poker\csharp\Program.cs`

```
# 引入 Games.Common.IO 模块
global using Games.Common.IO;
# 引入 Games.Common.Randomness 模块
global using Games.Common.Randomness;
# 引入 Poker 模块
global using Poker;

# 创建一个新的游戏对象，使用控制台输入输出和随机数生成器作为参数，然后开始游戏
new Game(new ConsoleIO(), new RandomNumberGenerator()).Play();
```