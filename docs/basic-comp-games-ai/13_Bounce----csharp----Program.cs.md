# `basic-computer-games\13_Bounce\csharp\Program.cs`

```

# 引入 Games.Common.IO 和 Games.Common.Numbers 命名空间
global using Games.Common.IO;
global using Games.Common.Numbers;

# 引入 Bounce 命名空间
using Bounce;

# 创建一个新的游戏对象，使用控制台输入输出作为参数，然后开始游戏，直到返回值为 true
new Game(new ConsoleIO()).Play(() => true);

```