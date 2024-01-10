# `basic-computer-games\82_Stars\csharp\Game.cs`

```
namespace Stars;
# 创建名为 Stars 的命名空间

internal class Game
# 创建名为 Game 的内部类

private readonly TextIO _io;
private readonly IRandom _random;
private readonly int _maxNumber;
private readonly int _maxGuessCount;
# 声明私有变量 _io、_random、_maxNumber、_maxGuessCount，并初始化为只读

public Game(TextIO io, IRandom random, int maxNumber, int maxGuessCount)
# Game 类的构造函数，接受 TextIO 对象、IRandom 对象、最大数字和最大猜测次数作为参数

internal void Play(Func<bool> playAgain)
# Play 方法，接受一个返回布尔值的委托作为参数

private void DisplayIntroduction()
# DisplayIntroduction 方法，用于显示游戏介绍

private void Play()
# Play 方法，用于开始游戏

private void AcceptGuesses(int target)
# AcceptGuesses 方法，用于接受玩家的猜测

private void DisplayStars(int target, float guess)
# DisplayStars 方法，用于显示猜测结果
    {
        // 根据猜测和目标之间的差值确定星号的数量，用于显示猜测的准确性
        var stars = Math.Abs(guess - target) switch
        {
            // 如果差值大于等于64，显示1个星号
            >= 64 => "*",
            // 如果差值大于等于32，显示2个星号
            >= 32 => "**",
            // 如果差值大于等于16，显示3个星号
            >= 16 => "***",
            // 如果差值大于等于8，显示4个星号
            >= 8  => "****",
            // 如果差值大于等于4，显示5个星号
            >= 4  => "*****",
            // 如果差值大于等于2，显示6个星号
            >= 2  => "******",
            // 其他情况，显示7个星号
            _     => "*******"
        };

        // 输出星号
        _io.WriteLine(stars);
    }

    // 显示猜对的消息
    private void DisplayWin(int guessCount)
    {
        // 输出空行
        _io.WriteLine();
        // 输出79个星号
        _io.WriteLine(new string('*', 79));
        // 输出空行
        _io.WriteLine();
        // 输出猜对的消息和猜测次数
        _io.WriteLine($"You got it in {guessCount} guesses!!!  Let's play again...");
    }

    // 显示猜错的消息
    private void DisplayLoss(int target)
    {
        // 输出空行
        _io.WriteLine();
        // 输出猜错的消息和目标数字
        _io.WriteLine($"Sorry, that's {_maxGuessCount} guesses. The number was {target}.");
    }
# 闭合前面的函数定义
```