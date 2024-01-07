# `basic-computer-games\27_Civil_War\csharp\Battle.cs`

```

# 引入 System 和 System.Collections.Generic 命名空间
using System;
using System.Collections.Generic;

# 定义枚举类型 Side 和 Option
namespace CivilWar
{
    public enum Side { Confederate, Union, Both }
    public enum Option { Battle, Replay, Quit }

    # 定义记录类型 Battle，包含战斗名称、人数、伤亡人数、进攻方、描述等属性
    public record Battle(string Name, int[] Men, int[] Casualties, Side Offensive, string Description)
    {
        # 定义静态只读的 Historic 列表，包含多个战斗的信息
        public static readonly List<Battle> Historic = new()
        {
            # 初始化多个战斗的信息
            new("Bull Run", new[] { 18000, 18500 }, new[] { 1967, 2708 }, Side.Union, "July 21, 1861.  Gen. Beauregard, commanding the south, met Union forces with Gen. McDowell in a premature battle at Bull Run. Gen. Jackson helped push back the union attack."),
            new("Shiloh", new[] { 40000, 44894 }, new[] { 10699, 13047 }, Side.Both, "April 6-7, 1862.  The confederate surprise attack at Shiloh failed due to poor organization."),
            ...
            # 其他战斗信息的初始化
            ...
        };

        # 定义静态方法 SelectBattle，用于选择战斗
        public static (Option, Battle?) SelectBattle()
        {
            # 输出提示信息
            Console.WriteLine("\n\n\nWhich battle do you wish to simulate?");
            # 读取用户输入的数字，根据不同的情况返回不同的选项和战斗信息
            return int.Parse(Console.ReadLine() ?? "") switch
            {
                0 => (Option.Replay, null),
                >0 and <15 and int n  => (Option.Battle, Historic[n-1]),
                _ => (Option.Quit, null)
            };
        }
    }
}

```