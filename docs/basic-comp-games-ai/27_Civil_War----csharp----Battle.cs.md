# `basic-computer-games\27_Civil_War\csharp\Battle.cs`

```py
# 引入命名空间 System 和 System.Collections.Generic
using System;
using System.Collections.Generic;

# 定义枚举类型 Side，表示阵营，包括 Confederate（南方联盟）、Union（北方联邦）、Both（双方）
namespace CivilWar
{
    # 定义枚举类型 Option，表示选项，包括 Battle（战斗）、Replay（重播）、Quit（退出）
    public enum Side { Confederate, Union, Both }
    public enum Option { Battle, Replay, Quit }

    # 定义记录类型 Battle，包括名称、士兵数量、伤亡人数、进攻方阵营、描述
    public record Battle(string Name, int[] Men, int[] Casualties, Side Offensive, string Description)
    }
}
```