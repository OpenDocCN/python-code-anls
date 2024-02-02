# `basic-computer-games\84_Super_Star_Trek\csharp\Commands\Command.cs`

```py
// 声明一个枚举类型，表示游戏中可用的指令
namespace SuperStarTrek.Commands;

// 定义一个枚举类型，包含不同的指令
internal enum Command
{
    // 设置航向的指令
    [Description("To set course")]
    NAV,

    // 进行短程传感器扫描的指令
    [Description("For short range sensor scan")]
    SRS,

    // 进行长程传感器扫描的指令
    [Description("For long range sensor scan")]
    LRS,

    // 发射相位炮的指令
    [Description("To fire phasers")]
    PHA,

    // 发射光子鱼雷的指令
    [Description("To fire photon torpedoes")]
    TOR,

    // 升降护盾的指令
    [Description("To raise or lower shields")]
    SHE,

    // 损伤控制报告的指令
    [Description("For damage control reports")]
    DAM,

    // 调用图书计算机的指令
    [Description("To call on library-computer")]
    COM,

    // 辞去指挥权的指令
    [Description("To resign your command")]
    XXX
}
```