# `basic-computer-games\82_Stars\csharp\Program.cs`

```

# 导入所需的模块
using Games.Common.IO;
using Games.Common.Randomness;
using Stars;

# 创建一个新的游戏对象，使用控制台输入输出和随机数生成器作为参数，设置最大数字为100，最大猜测次数为7
var game = new Game(new ConsoleIO(), new RandomNumberGenerator(), maxNumber: 100, maxGuessCount: 7);

# 开始游戏，传入一个始终返回true的函数作为参数
game.Play(() => true);

```