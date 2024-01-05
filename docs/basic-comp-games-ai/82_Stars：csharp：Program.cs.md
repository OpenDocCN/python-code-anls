# `d:/src/tocomm/basic-computer-games\82_Stars\csharp\Program.cs`

```
# 导入所需的模块
import Games.Common.IO
import Games.Common.Randomness
import Stars

# 创建游戏对象，传入控制台输入输出对象、随机数生成器对象，设置最大数字和最大猜测次数
game = Game(ConsoleIO(), RandomNumberGenerator(), maxNumber=100, maxGuessCount=7)

# 开始游戏，传入一个始终返回True的函数作为游戏继续条件
game.Play(lambda: True)
```