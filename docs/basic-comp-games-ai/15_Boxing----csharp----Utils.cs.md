# `basic-computer-games\15_Boxing\csharp\Utils.cs`

```
// 命名空间声明
namespace Boxing;
// 声明一个静态类 GameUtils
public static class GameUtils
{
    // 声明一个静态只读的 Random 对象 Rnd，使用当前时间的毫秒数作为种子
    private static readonly Random Rnd = new((int) DateTime.UtcNow.Ticks);
    // 打印不同拳击动作的描述
    public static void PrintPunchDescription() =>
        Console.WriteLine($"DIFFERENT PUNCHES ARE: {PunchDesc(Punch.FullSwing)}; {PunchDesc(Punch.Hook)}; {PunchDesc(Punch.Uppercut)}; {PunchDesc(Punch.Jab)}.");
    // 返回拳击动作的描述字符串
    private static string PunchDesc(Punch punch) => $"({(int)punch}) {punch.ToFriendlyString()}";
    // 获取用户输入的拳击动作
    public static Punch GetPunch(string prompt)
    {
        Console.WriteLine(prompt);
        Punch result;
        // 循环直到用户输入有效的拳击动作
        while (!Enum.TryParse(Console.ReadLine(), out result) || !Enum.IsDefined(typeof(Punch), result))
        {
            PrintPunchDescription();
        }
        return result;
    }
    // 声明一个属性 Roll，返回一个接受上限参数的函数，生成一个随机数
    public static Func<int, int> Roll { get;  } =  upperLimit => (int) (upperLimit * Rnd.NextSingle()) + 1;
    // 判断随机数是否满足条件
    public static bool RollSatisfies(int upperLimit, Predicate<int> predicate) => predicate(Roll(upperLimit));
    // 将拳击动作转换为友好的描述字符串
    public static string ToFriendlyString(this Punch punch)
        => punch switch
        {
            Punch.FullSwing => "FULL SWING",
            Punch.Hook => "HOOK",
            Punch.Uppercut => "UPPERCUT",
            Punch.Jab => "JAB",
            _ => throw new ArgumentOutOfRangeException(nameof(punch), punch, null)
        };
}
```