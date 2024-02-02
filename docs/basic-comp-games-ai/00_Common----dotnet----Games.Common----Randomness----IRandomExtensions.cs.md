# `basic-computer-games\00_Common\dotnet\Games.Common\Randomness\IRandomExtensions.cs`

```py
// 提供对 <see cref="IRandom" /> 的扩展方法，以在给定范围内提供随机数
public static class IRandomExtensions
{
    /// <summary>
    /// 获取一个随机的 <see cref="float" />，使得 0 &lt;= n &lt; exclusiveMaximum。
    /// </summary>
    /// <returns>随机数。</returns>
    public static float NextFloat(this IRandom random, float exclusiveMaximum) =>
        Scale(random.NextFloat(), exclusiveMaximum);

    /// <summary>
    /// 获取一个随机的 <see cref="float" />，使得 inclusiveMinimum &lt;= n &lt; exclusiveMaximum。
    /// </summary>
    /// <returns>随机数。</returns>
    public static float NextFloat(this IRandom random, float inclusiveMinimum, float exclusiveMaximum) =>
        Scale(random.NextFloat(), inclusiveMinimum, exclusiveMaximum);

    /// <summary>
    /// 获取一个随机的 <see cref="int" />，使得 0 &lt;= n &lt; exclusiveMaximum。
    /// </summary>
    /// <returns>随机数。</returns>
    public static int Next(this IRandom random, int exclusiveMaximum) => ToInt(random.NextFloat(exclusiveMaximum));

    /// <summary>
    /// 获取一个随机的 <see cref="int" />，使得 inclusiveMinimum &lt;= n &lt; exclusiveMaximum。
    /// </summary>
    /// <returns>随机数。</returns>
    public static int Next(this IRandom random, int inclusiveMinimum, int exclusiveMaximum) =>
        ToInt(random.NextFloat(inclusiveMinimum, exclusiveMaximum));

    /// <summary>
    /// 获取前一个未缩放的 <see cref="float" />（介于 0 和 1 之间），缩放到一个新的范围：
    /// 0 &lt;= x &lt; <paramref name="exclusiveMaximum" />。
    /// </summary>
    /// <returns>随机数。</returns>
    public static float PreviousFloat(this IRandom random, float exclusiveMaximum) =>
        Scale(random.PreviousFloat(), exclusiveMaximum);
}
    /// <summary>
    /// 将前一个未缩放的 <see cref="float" />（介于 0 和 1 之间）缩放到一个新的范围：
    /// <paramref name="inclusiveMinimum" /> &lt;= n &lt; <paramref name="exclusiveMaximum" />。
    /// </summary>
    /// <returns>随机数。</returns>
    public static float PreviousFloat(this IRandom random, float inclusiveMinimum, float exclusiveMaximum) =>
        Scale(random.PreviousFloat(), inclusiveMinimum, exclusiveMaximum);

    /// <summary>
    /// 将前一个未缩放的 <see cref="float" />（介于 0 和 1 之间）缩放到一个新的范围：0 &lt;= n &lt; <paramref name="exclusiveMaximum" />。
    /// </summary>
    /// <returns>随机数。</returns>
    public static int Previous(this IRandom random, int exclusiveMaximum) =>
        ToInt(random.PreviousFloat(exclusiveMaximum));

    /// <summary>
    /// 将前一个未缩放的 <see cref="float" />（介于 0 和 1 之间）缩放到一个新的范围：
    /// <paramref name="inclusiveMinimum" /> &lt;= n &lt; <paramref name="exclusiveMaximum" />。
    /// <returns>随机数。</returns>
    public static int Previous(this IRandom random, int inclusiveMinimum, int exclusiveMaximum) =>
        ToInt(random.PreviousFloat(inclusiveMinimum, exclusiveMaximum));

    /// <summary>
    /// 将 zeroToOne 缩放到 exclusiveMaximum。
    /// 如果 exclusiveMaximum 小于等于 0，则抛出 ArgumentOutOfRangeException 异常。
    /// </summary>
    private static float Scale(float zeroToOne, float exclusiveMaximum)
    {
        if (exclusiveMaximum <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(exclusiveMaximum), "必须大于 0。");
        }

        return Scale(zeroToOne, 0, exclusiveMaximum);
    }

    /// <summary>
    /// 将 zeroToOne 缩放到 inclusiveMinimum 和 exclusiveMaximum 之间。
    /// 如果 exclusiveMaximum 小于等于 inclusiveMinimum，则抛出 ArgumentOutOfRangeException 异常。
    /// </summary>
    private static float Scale(float zeroToOne, float inclusiveMinimum, float exclusiveMaximum)
    {
        if (exclusiveMaximum <= inclusiveMinimum)
        {
            throw new ArgumentOutOfRangeException(nameof(exclusiveMaximum), "必须大于 inclusiveMinimum。");
        }

        var range = exclusiveMaximum - inclusiveMinimum;
        return zeroToOne * range + inclusiveMinimum;
    }
    # 结束当前的代码块
    }

    # 将浮点数值转换为整数，取其向下取整的整数部分
    private static int ToInt(float value) => (int)Math.Floor(value);
# 闭合前面的函数定义
```