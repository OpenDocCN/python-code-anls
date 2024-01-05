# `d:/src/tocomm/basic-computer-games\41_Guess\csharp\Game.cs`

```
namespace Guess;  # 命名空间声明

internal class Game  # 声明一个内部类 Game
{
    private readonly IReadWrite _io;  # 声明一个私有只读字段 _io，类型为 IReadWrite 接口
    private readonly IRandom _random;  # 声明一个私有只读字段 _random，类型为 IRandom 接口

    public Game(IReadWrite io, IRandom random)  # Game 类的构造函数，接受 IReadWrite 和 IRandom 接口类型的参数
    {
        _io = io;  # 将传入的 io 参数赋值给 _io 字段
        _random = random;  # 将传入的 random 参数赋值给 _random 字段
    }

    public void Play()  # Play 方法的声明
    {
        while (true)  # 进入无限循环
        {
            _io.Write(Streams.Introduction);  # 使用 _io 接口的 Write 方法输出 Streams.Introduction 的内容

            var limit = _io.ReadNumber(Prompts.Limit);  # 使用 _io 接口的 ReadNumber 方法读取用户输入的数字，并赋值给 limit 变量
            _io.WriteLine();  // 输出空行

            // 这里存在一个 bug。如果输入的限制值小于等于 0，程序将崩溃。
            var targetGuessCount = checked((int)Math.Log2(limit) + 1);  // 计算猜测的次数

            PlayGuessingRounds(limit, targetGuessCount);  // 调用 PlayGuessingRounds 方法进行猜数字游戏

            _io.Write(Streams.BlankLines);  // 输出空行
        }
    }

    private void PlayGuessingRounds(float limit, int targetGuessCount)
    {
        while (true)
        {
            _io.WriteLine(Formats.Thinking, limit);  // 输出提示信息，显示限制值

            // 这里存在一个 bug。如果输入的限制值为非整数，那么秘密数字可能会是大于限制值的下一个整数。
            var secretNumber = (int)_random.NextFloat(limit) + 1;  // 生成一个随机整数作为秘密数字，范围在1到limit之间

            var guessCount = 0;  // 初始化猜测次数为0

            while (true)  // 进入一个无限循环，直到猜对为止
            {
                var guess = _io.ReadNumber("");  // 从输入流中读取用户猜测的数字
                if (guess <= 0) { return; }  // 如果用户输入的数字小于等于0，则结束游戏
                guessCount++;  // 猜测次数加1
                if (IsGuessCorrect(guess, secretNumber)) { break; }  // 调用IsGuessCorrect函数判断猜测是否正确，如果正确则跳出循环
            }

            ReportResult(guessCount, targetGuessCount);  // 调用ReportResult函数报告猜测结果

            _io.Write(Streams.BlankLines);  // 在输出流中写入空行
        }
    }

    private bool IsGuessCorrect(float guess, int secretNumber)  // 定义一个函数用来判断猜测是否正确
    {
        if (guess < secretNumber) { _io.Write(Streams.TooLow); }  // 如果猜测的数字小于秘密数字，向输出流写入提示信息“太低”
        if (guess > secretNumber) { _io.Write(Streams.TooHigh); }  // 如果猜测的数字大于秘密数字，向输出流写入提示信息“太高”

        return guess == secretNumber;  // 返回猜测的数字是否等于秘密数字的布尔值
    }

    private void ReportResult(int guessCount, int targetGuessCount)
    {
        _io.WriteLine(Formats.ThatsIt, guessCount);  // 向输出流写入格式化的提示信息，表示猜测成功，并显示猜测次数
        _io.WriteLine(
            (guessCount - targetGuessCount) switch  // 使用 switch 语句根据猜测次数和目标猜测次数的差值进行不同的处理
            {
                < 0 => Strings.VeryGood,  // 如果差值小于 0，向输出流写入“非常好”的提示信息
                0 => Strings.Good,  // 如果差值等于 0，向输出流写入“好”的提示信息
                > 0 => string.Format(Formats.ShouldHave, targetGuessCount)  // 如果差值大于 0，向输出流写入格式化的提示信息，表示应该达到的猜测次数
            });
    }
}
```