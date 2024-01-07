# `basic-computer-games\84_Super_Star_Trek\csharp\Commands\Command.cs`

```

// 声明一个枚举类型，表示游戏中的各种指令
namespace SuperStarTrek.Commands;

// 声明一个枚举类型，表示游戏中的各种指令
internal enum Command
{
    // 设置航向指令
    [Description("To set course")]
    NAV,

    // 进行短程传感器扫描指令
    [Description("For short range sensor scan")]
    SRS,

    // 进行长程传感器扫描指令
    [Description("For long range sensor scan")]
    LRS,

    // 发射相位炮指令
    [Description("To fire phasers")]
    PHA,

    // 发射光子鱼雷指令
    [Description("To fire photon torpedoes")]
    TOR,

    // 升降护盾指令
    [Description("To raise or lower shields")]
    SHE,

    // 损伤控制报告指令
    [Description("For damage control reports")]
    DAM,

    // 调用图书计算机指令
    [Description("To call on library-computer")]
    COM,

    // 辞职指令
    [Description("To resign your command")]
    XXX
}

```