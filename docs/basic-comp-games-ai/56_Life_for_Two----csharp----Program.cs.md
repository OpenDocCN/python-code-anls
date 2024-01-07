# `basic-computer-games\56_Life_for_Two\csharp\Program.cs`

```

# 引入 Games.Common.IO 命名空间
global using Games.Common.IO;
# 引入 LifeforTwo.Resources.Resource 命名空间
global using static LifeforTwo.Resources.Resource;
# 引入 LifeforTwo 命名空间
global using LifeforTwo;

# 创建一个新的游戏对象，并使用控制台输入输出作为参数，然后开始游戏
new Game(new ConsoleIO()).Play();

```