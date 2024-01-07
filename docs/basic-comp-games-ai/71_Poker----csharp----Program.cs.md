# `basic-computer-games\71_Poker\csharp\Program.cs`

```

# 引入 Games.Common.IO 命名空间
global using Games.Common.IO;
# 引入 Games.Common.Randomness 命名空间
global using Games.Common.Randomness;
# 引入 Poker 命名空间
global using Poker;

# 创建一个新的游戏对象，使用控制台输入输出和随机数生成器作为参数
new Game(new ConsoleIO(), new RandomNumberGenerator()).Play();

```