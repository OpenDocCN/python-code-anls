# `d:/src/tocomm/basic-computer-games\15_Boxing\csharp\Utils.cs`

```
// 命名空间声明
namespace Boxing;
// 定义静态类 GameUtils
public static class GameUtils
{
    // 声明私有静态只读字段 Rnd，用于生成随机数
    private static readonly Random Rnd = new((int) DateTime.UtcNow.Ticks);
    // 打印不同拳击动作的描述
    public static void PrintPunchDescription() =>
        Console.WriteLine($"DIFFERENT PUNCHES ARE: {PunchDesc(Punch.FullSwing)}; {PunchDesc(Punch.Hook)}; {PunchDesc(Punch.Uppercut)}; {PunchDesc(Punch.Jab)}.");
    
    // 返回指定拳击动作的描述
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
}
    public static Func<int, int> Roll { get;  } =  upperLimit => (int) (upperLimit * Rnd.NextSingle()) + 1;
    // 定义一个名为Roll的静态属性，其类型为Func<int, int>，用于生成一个随机数，范围为1到upperLimit

    public static bool RollSatisfies(int upperLimit, Predicate<int> predicate) => predicate(Roll(upperLimit));
    // 定义一个名为RollSatisfies的静态方法，用于检查生成的随机数是否满足给定的条件

    public static string ToFriendlyString(this Punch punch)
        => punch switch
        {
            Punch.FullSwing => "FULL SWING",
            Punch.Hook => "HOOK",
            Punch.Uppercut => "UPPERCUT",
            Punch.Jab => "JAB",
            _ => throw new ArgumentOutOfRangeException(nameof(punch), punch, null)
        };
    // 定义一个名为ToFriendlyString的扩展方法，用于将Punch枚举类型转换为友好的字符串表示形式
```