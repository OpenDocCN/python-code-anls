# `82_Stars\csharp\Game.cs`

```
    # 导入所需的模块
    import System
    import Games.Common.IO
    import Games.Common.Randomness
    import Stars.Resources

    # 定义 Game 类
    class Game:
        # 初始化 Game 类的属性
        def __init__(self, io, random, maxNumber, maxGuessCount):
            self._io = io  # 初始化 TextIO 类的实例
            self._random = random  # 初始化 IRandom 类的实例
            self._maxNumber = maxNumber  # 初始化最大数字
            self._maxGuessCount = maxGuessCount  # 初始化最大猜测次数
    }

    # 定义一个名为 Play 的内部方法，接受一个返回布尔值的委托 playAgain
    internal void Play(Func<bool> playAgain)
    {
        # 显示游戏介绍
        DisplayIntroduction();

        # 循环执行游戏直到 playAgain 方法返回 false
        do
        {
            Play();
        } while (playAgain.Invoke());
    }

    # 定义一个名为 DisplayIntroduction 的私有方法
    private void DisplayIntroduction()
    {
        # 使用 _io 对象的 Write 方法显示游戏标题
        _io.Write(Resource.Streams.Title);

        # 如果用户输入的字符串（不区分大小写）等于 "N"，则直接返回，不显示游戏说明
        if (_io.ReadString("Do you want instructions").Equals("N", StringComparison.InvariantCultureIgnoreCase))
        {
            return;
        }
        _io.WriteLine(Resource.Formats.Instructions, _maxNumber, _maxGuessCount);
    }
```
这行代码是调用 _io 对象的 WriteLine 方法，输出指定格式的字符串，其中 _maxNumber 和 _maxGuessCount 是参数。

```
    private void Play()
    {
        _io.WriteLine();
        _io.WriteLine();

        var target = _random.Next(_maxNumber) + 1;

        _io.WriteLine("Ok, I am thinking of a number.  Start guessing.");

        AcceptGuesses(target);
    }
```
这段代码定义了一个 Play 方法，其中通过 _random 对象生成一个随机数作为目标值，然后输出提示信息，最后调用 AcceptGuesses 方法。

```
    private void AcceptGuesses(int target)
    {
        for (int guessCount = 1; guessCount <= _maxGuessCount; guessCount++)
        {
```
这段代码定义了一个 AcceptGuesses 方法，其中使用 for 循环来接受用户的猜测。guessCount 从 1 开始，每次循环增加 1，直到达到 _maxGuessCount。
# 输出空行
_io.WriteLine()
# 读取用户输入的猜测数字
guess = _io.ReadNumber("Your guess")
# 如果猜测数字等于目标数字，显示胜利信息并返回
if guess == target:
    DisplayWin(guessCount)
    return
# 否则显示星号提示
DisplayStars(target, guess)
# 显示失败信息
DisplayLoss(target)
```

```python
# 根据猜测数字和目标数字的差值，显示对应数量的星号
private void DisplayStars(int target, float guess):
    stars = Math.Abs(guess - target) switch
    {
        case >= 64: "*",
            >= 32 => "**",  # 如果猜测次数大于等于32，则输出两个星号
            >= 16 => "***",  # 如果猜测次数大于等于16，则输出三个星号
            >= 8  => "****",  # 如果猜测次数大于等于8，则输出四个星号
            >= 4  => "*****",  # 如果猜测次数大于等于4，则输出五个星号
            >= 2  => "******",  # 如果猜测次数大于等于2，则输出六个星号
            _     => "*******"  # 其他情况输出七个星号
        };

        _io.WriteLine(stars);  # 输出星号
    }

    private void DisplayWin(int guessCount)
    {
        _io.WriteLine();  # 输出空行
        _io.WriteLine(new string('*', 79));  # 输出79个星号
        _io.WriteLine();  # 输出空行
        _io.WriteLine($"You got it in {guessCount} guesses!!!  Let's play again...");  # 输出猜测次数并提示再玩一次
    }

    private void DisplayLoss(int target)  # 显示失败信息
    {
        # 输出一行空白
        _io.WriteLine();
        # 输出猜测次数已达到最大值和目标数字的信息
        _io.WriteLine($"Sorry, that's {_maxGuessCount} guesses. The number was {target}.");
    }
}
```
在这段代码中，`_io.WriteLine();`用于输出一行空白，而`_io.WriteLine($"Sorry, that's {_maxGuessCount} guesses. The number was {target}.");`用于输出猜测次数已达到最大值和目标数字的信息。
```