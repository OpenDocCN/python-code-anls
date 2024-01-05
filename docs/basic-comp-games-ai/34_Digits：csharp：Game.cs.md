# `d:/src/tocomm/basic-computer-games\34_Digits\csharp\Game.cs`

```
namespace Digits;  # 命名空间声明

internal class GameSeries  # 内部类声明
{
    private readonly IReadOnlyList<int> _weights = new List<int> { 0, 1, 3 }.AsReadOnly();  # 声明只读整数列表_weights，并初始化为只读的列表

    private readonly IReadWrite _io;  # 声明只读的IReadWrite接口类型_io
    private readonly IRandom _random;  # 声明只读的IRandom接口类型_random

    public GameSeries(IReadWrite io, IRandom random)  # 类的构造函数，接受IReadWrite和IRandom类型的参数
    {
        _io = io;  # 将传入的io参数赋值给_io
        _random = random;  # 将传入的random参数赋值给_random
    }

    internal void Play()  # 内部方法Play
    {
        _io.Write(Streams.Introduction);  # 调用_io的Write方法，传入Streams.Introduction参数

        if (_io.ReadNumber(Prompts.ForInstructions) != 0)  # 调用_io的ReadNumber方法，传入Prompts.ForInstructions参数，并判断返回值是否不等于0
        {
            _io.Write(Streams.Instructions);  # 将指令流写入输入/输出接口
        }

        do
        {
            new Game(_io, _random).Play();  # 创建新的游戏实例并开始游戏
        } while (_io.ReadNumber(Prompts.WantToTryAgain) == 1);  # 当输入/输出接口读取的数字等于1时，继续循环

        _io.Write(Streams.Thanks);  # 将感谢流写入输入/输出接口
    }
}

internal class Game
{
    private readonly IReadWrite _io;  # 创建只读的输入/输出接口
    private readonly Guesser _guesser;  # 创建猜测者实例

    public Game(IReadWrite io, IRandom random)  # 游戏类的构造函数，接受输入/输出接口和随机数生成器实例作为参数
        _io = io;  # 将传入的io对象赋值给成员变量_io
        _guesser = new Guesser(random);  # 使用random对象创建一个新的Guesser对象并赋值给成员变量_guesser
    }

    public void Play()
    {
        var correctGuesses = 0;  # 初始化变量correctGuesses为0，用于记录猜对的次数

        for (int round = 0; round < 3; round++)  # 循环3次，每次代表一轮游戏
        {
            var digits = _io.Read10Digits(Prompts.TenNumbers, Streams.TryAgain);  # 调用_io对象的Read10Digits方法，传入提示语和再试一次的提示语，将返回的数字赋值给变量digits

            correctGuesses = GuessDigits(digits, correctGuesses);  # 调用GuessDigits方法，传入用户输入的数字和猜对的次数，将返回的猜对次数赋值给correctGuesses
        }

        _io.Write(correctGuesses switch  # 根据猜对的次数进行不同的输出
        {
            < 10 => Streams.YouWin,  # 如果猜对次数小于10，输出YouWin
            10 => Streams.ItsATie,  # 如果猜对次数等于10，输出ItsATie
            > 10 => Streams.IWin  # 如果猜对次数大于10，输出IWin
    private int GuessDigits(IEnumerable<int> digits, int correctGuesses)
    {
        _io.Write(Streams.Headings);  // 在输出流中写入标题信息

        foreach (var digit in digits)  // 遍历给定的数字集合
        {
            var guess = _guesser.GuessNextDigit();  // 使用猜测器猜测下一个数字
            if (guess == digit) { correctGuesses++; }  // 如果猜测正确，则增加正确猜测的次数

            _io.WriteLine(Formats.GuessResult, guess, digit, guess == digit ? "Right" : "Wrong", correctGuesses);  // 在输出流中写入猜测结果的格式化字符串

            _guesser.ObserveActualDigit(digit);  // 观察实际的数字
        }

        return correctGuesses;  // 返回正确猜测的次数
    }
}
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```