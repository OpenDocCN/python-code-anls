# `basic-computer-games\13_Bounce\csharp\Program.cs`

```
# 引入 Games.Common.IO 和 Games.Common.Numbers 模块
global using Games.Common.IO;
global using Games.Common.Numbers;

# 引入 Bounce 模块
using Bounce;

# 创建一个新的游戏对象，并使用控制台输入输出作为参数
new Game(new ConsoleIO()).Play(() => true);
```