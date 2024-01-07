# `basic-computer-games\41_Guess\csharp\Program.cs`

```

# 引入 Games.Common.IO 命名空间
global using Games.Common.IO;
# 引入 Games.Common.Randomness 命名空间
global using Games.Common.Randomness;
# 引入 Guess.Resources.Resource 类的静态成员
global using static Guess.Resources.Resource;  
# 使用 Guess 命名空间
using Guess;

# 创建一个新的游戏对象，并使用 ConsoleIO 和 RandomNumberGenerator 对象进行初始化，然后开始游戏
new Game(new ConsoleIO(), new RandomNumberGenerator()).Play();

```