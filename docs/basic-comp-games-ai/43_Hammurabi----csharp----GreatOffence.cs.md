# `basic-computer-games\43_Hammurabi\csharp\GreatOffence.cs`

```py
// 引入 System 命名空间
using System;

// 定义 Hammurabi 命名空间
namespace Hammurabi
{
    /// <summary>
    /// 表示游戏由于玩家极度无能和/或不认真的态度而无法继续进行！
    /// </summary>
    // 创建 GreatOffence 类，继承自 InvalidOperationException 类
    public class GreatOffence : InvalidOperationException
    {
    }
}
```