# `46_Hexapawn\csharp\Program.cs`

```
# 导入 Games.Common.IO 模块中的所有内容
import Games.Common.IO
# 导入 Games.Common.Randomness 模块中的所有内容
import Games.Common.Randomness
# 导入 Hexapawn 模块中的所有内容
import Hexapawn

# 创建一个新的 GameSeries 对象，使用 ConsoleIO() 和 RandomNumberGenerator() 作为参数，然后调用 Play() 方法开始游戏
new GameSeries(new ConsoleIO(), new RandomNumberGenerator()).Play()
```