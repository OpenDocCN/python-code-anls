# `d:/src/tocomm/basic-computer-games\71_Poker\csharp\Program.cs`

```
# 导入 Games.Common.IO 模块
global using Games.Common.IO;
# 导入 Games.Common.Randomness 模块
global using Games.Common.Randomness;
# 导入 Poker 模块
global using Poker;

# 创建一个新的游戏对象，使用控制台输入输出和随机数生成器作为参数
new Game(new ConsoleIO(), new RandomNumberGenerator()).Play();
```