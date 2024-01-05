# `27_Civil_War\csharp\Battle.cs`

```
    // 声明一个枚举类型，表示战争的一方
    public enum Side { Confederate, Union, Both }
    // 声明一个枚举类型，表示用户的选项
    public enum Option { Battle, Replay, Quit }

    // 定义一个记录类型，表示一场战斗
    public record Battle(string Name, int[] Men, int[] Casualties, Side Offensive, string Description)
    {
        // 静态字段，存储历史战斗的列表
        public static readonly List<Battle> Historic = new()
        {
            // 初始化历史战斗列表
            new("Shiloh", new[] { 40000, 44894 }, new[] { 10699, 13047 }, Side.Both, "April 6-7, 1862.  The confederate surprise attack at Shiloh failed due to poor organization."),
            new("Antietam", new[] { 40000, 50000 }, new[] { 10000, 12000 }, Side.Both, "Sept 17, 1862.  The south failed to incorporate Maryland into the confederacy."),
            new("Murfreesboro", new[] { 38000, 45000 }, new[] { 11000, 12000 }, Side.Union, "Dec 31, 1862.  The south under Gen. Bragg won a close battle."),
            new("Chickamauga", new[] { 66000, 60000 }, new[] { 18000, 16000 }, Side.Confederate, "Sept. 15, 1863. Confusion in a forest near Chickamauga led to a costly southern victory."),
        };

        // 静态方法，用于选择一场战斗
        public static (Option, Battle?) SelectBattle()
        {
# 提示用户输入要模拟的战斗
Console.WriteLine("\n\n\nWhich battle do you wish to simulate?");
# 读取用户输入的内容并转换为整数
return int.Parse(Console.ReadLine() ?? "") switch
{
    # 如果用户输入为0，则返回(Option.Replay, null)
    0 => (Option.Replay, null),
    # 如果用户输入大于0且小于15，并且为整数n，则返回(Option.Battle, Historic[n-1])
    >0 and <15 and int n  => (Option.Battle, Historic[n-1]),
    # 如果用户输入不满足以上条件，则返回(Option.Quit, null)
    _ => (Option.Quit, null)
};
```