# `26_Chomp\csharp\Program.cs`

```
# 引入 Games.Common.IO 命名空间
global using Games.Common.IO;
# 引入 Chomp.Resources 命名空间
global using Chomp.Resources;
# 引入 Chomp 命名空间
using Chomp;

# 创建一个新的游戏对象，使用控制台输入输出作为参数，然后开始游戏
new Game(new ConsoleIO()).Play();
```