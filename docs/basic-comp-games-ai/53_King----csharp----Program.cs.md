# `basic-computer-games\53_King\csharp\Program.cs`

```py
# 引入 Games.Common.IO 命名空间
global using Games.Common.IO;
# 引入 Games.Common.Randomness 命名空间
global using Games.Common.Randomness;
# 引入 King.Resources 命名空间
global using King.Resources;
# 引入 King.Resources.Resource 类的静态成员
global using static King.Resources.Resource;
# 引入 King 命名空间
using King;

# 创建一个新的游戏对象，并使用 ConsoleIO 和 RandomNumberGenerator 作为参数进行初始化，然后开始游戏
new Game(new ConsoleIO(), new RandomNumberGenerator()).Play();
```