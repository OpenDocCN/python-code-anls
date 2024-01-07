# `basic-computer-games\67_One_Check\csharp\Program.cs`

```

# 引入 Games.Common.IO 命名空间
global using Games.Common.IO;
# 引入 OneCheck.Resources.Resource 类的静态成员
global using static OneCheck.Resources.Resource;
# 引入 OneCheck 命名空间
using OneCheck;

# 创建一个新的游戏对象，使用控制台输入输出作为参数，然后开始游戏
new Game(new ConsoleIO()).Play();

```