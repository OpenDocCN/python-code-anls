# `07_Basketball\csharp\Program.cs`

```
# 导入Basketball模块
import Basketball
# 导入Games.Common.IO模块
import Games.Common.IO
# 导入Games.Common.Randomness模块
import Games.Common.Randomness

# 创建游戏对象，使用ConsoleIO和RandomNumberGenerator作为参数
game = Game.Create(new ConsoleIO(), new RandomNumberGenerator())

# 开始游戏
game.Play()
```