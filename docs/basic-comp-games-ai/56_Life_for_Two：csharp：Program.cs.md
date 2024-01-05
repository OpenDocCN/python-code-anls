# `56_Life_for_Two\csharp\Program.cs`

```
global using Games.Common.IO;  // 导入 Games.Common.IO 命名空间
global using static LifeforTwo.Resources.Resource;  // 导入 LifeforTwo.Resources.Resource 命名空间的静态成员
global using LifeforTwo;  // 导入 LifeforTwo 命名空间

new Game(new ConsoleIO()).Play();  // 创建一个新的 Game 对象，使用 ConsoleIO 作为参数，然后调用 Play 方法开始游戏
```