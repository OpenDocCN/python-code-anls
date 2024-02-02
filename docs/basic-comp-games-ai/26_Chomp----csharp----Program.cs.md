# `basic-computer-games\26_Chomp\csharp\Program.cs`

```py
# 引入 Games.Common.IO 命名空间
global using Games.Common.IO;
# 引入 Chomp.Resources 命名空间
global using Chomp.Resources;
# 引入 Chomp 命名空间
using Chomp;

# 创建一个新的游戏对象，并使用控制台输入输出作为参数，开始游戏
new Game(new ConsoleIO()).Play();
```