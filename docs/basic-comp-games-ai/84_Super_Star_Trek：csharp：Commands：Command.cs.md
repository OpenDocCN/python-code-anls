# `d:/src/tocomm/basic-computer-games\84_Super_Star_Trek\csharp\Commands\Command.cs`

```
using System.ComponentModel;  # 导入 System.ComponentModel 模块
namespace SuperStarTrek.Commands;  # 定义 SuperStarTrek.Commands 命名空间
internal enum Command  # 定义一个枚举类型 Command
{
    [Description("To set course")]  # 为枚举值 NAV 添加描述信息
    NAV,

    [Description("For short range sensor scan")]  # 为枚举值 SRS 添加描述信息
    SRS,

    [Description("For long range sensor scan")]  # 为枚举值 LRS 添加描述信息
    LRS,

    [Description("To fire phasers")]  # 为枚举值 PHA 添加描述信息
    PHA,

    [Description("To fire photon torpedoes")]  # 为枚举值 TOR 添加描述信息
    TOR,
```
```python
# 创建一个枚举类型，每个枚举值都有一个描述
class Commands(Enum):
    # 用于升起或降低护盾
    SHE,
    # 用于损坏控制报告
    DAM,
    # 用于调用图书馆计算机
    COM,
    # 用于辞去指挥权
    XXX
}
```