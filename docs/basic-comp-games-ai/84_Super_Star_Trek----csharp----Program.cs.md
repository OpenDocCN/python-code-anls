# `basic-computer-games\84_Super_Star_Trek\csharp\Program.cs`

```
// 引入所需的命名空间和模块
using Games.Common.IO;
using Games.Common.Randomness;
using SuperStarTrek;

// 创建控制台输入输出对象
var io = new ConsoleIO();
// 创建随机数生成器对象
var random = new RandomNumberGenerator();

// 创建游戏对象，传入输入输出对象和随机数生成器对象
var game = new Game(io, random);

// 执行游戏介绍
game.DoIntroduction();

// 循环执行游戏并询问是否重新开始
do
{
    game.Play();
} while (game.Replay());
```