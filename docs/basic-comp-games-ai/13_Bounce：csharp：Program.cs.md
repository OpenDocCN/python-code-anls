# `d:/src/tocomm/basic-computer-games\13_Bounce\csharp\Program.cs`

```
# 导入 Games.Common.IO 和 Games.Common.Numbers 模块
global using Games.Common.IO;
global using Games.Common.Numbers;

# 导入 Bounce 模块
using Bounce;

# 创建一个新的游戏对象，使用控制台输入输出作为参数，然后调用 Play 方法并传入一个始终返回 true 的 lambda 函数
new Game(new ConsoleIO()).Play(() => true);
```