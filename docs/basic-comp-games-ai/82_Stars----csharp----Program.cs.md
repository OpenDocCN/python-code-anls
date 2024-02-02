# `basic-computer-games\82_Stars\csharp\Program.cs`

```py
# 引入所需的命名空间和模块
using Games.Common.IO;
using Games.Common.Randomness;
using Stars;

# 创建一个新的游戏对象，使用控制台输入输出、随机数生成器，设置最大数字为100，最大猜测次数为7
var game = new Game(new ConsoleIO(), new RandomNumberGenerator(), maxNumber: 100, maxGuessCount: 7);

# 开始游戏，直到猜中为止
game.Play(() => true);
```