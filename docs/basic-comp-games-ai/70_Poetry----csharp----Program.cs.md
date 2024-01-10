# `basic-computer-games\70_Poetry\csharp\Program.cs`

```
# 引入 Games.Common.IO 模块
global using Games.Common.IO;
# 引入 Games.Common.Randomness 模块
global using Games.Common.Randomness;
# 引入 Poetry 模块
global using Poetry;

# 使用 ConsoleIO 和 RandomNumberGenerator 组合生成一首诗
Poem.Compose(new ConsoleIO(), new RandomNumberGenerator());
```