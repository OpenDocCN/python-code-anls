# `00_Common\dotnet\Games.Common\Randomness\IRandomExtensions.cs`

```
using System; // 导入 System 命名空间

namespace Games.Common.Randomness; // 声明 Games.Common.Randomness 命名空间

/// <summary>
/// Provides extension methods to <see cref="IRandom" /> providing random numbers in a given range.
/// </summary>
/// <value></value>
public static class IRandomExtensions // 声明一个静态类 IRandomExtensions，提供对 IRandom 接口的扩展方法

{
    /// <summary>
    /// Gets a random <see cref="float" /> such that 0 &lt;= n &lt; exclusiveMaximum.
    /// </summary>
    /// <returns>The random number.</returns>
    public static float NextFloat(this IRandom random, float exclusiveMaximum) =>
        Scale(random.NextFloat(), exclusiveMaximum); // 调用 Scale 方法，返回一个在指定范围内的随机浮点数

    /// <summary>
    /// Gets a random <see cref="float" /> such that inclusiveMinimum &lt;= n &lt; exclusiveMaximum.
    /// </summary>
```
在这段代码中，注释的作用是解释每个方法的功能和返回值，以及对命名空间和类的说明。同时，注释还可以帮助其他开发人员理解代码的用途和实现细节。
    /// <returns>The random number.</returns>
    public static float NextFloat(this IRandom random, float inclusiveMinimum, float exclusiveMaximum) =>
        Scale(random.NextFloat(), inclusiveMinimum, exclusiveMaximum);
    // 返回一个随机浮点数，范围在[inclusiveMinimum, exclusiveMaximum)之间

    /// <summary>
    /// Gets a random <see cref="int" /> such that 0 &lt;= n &lt; exclusiveMaximum.
    /// </summary>
    /// <returns>The random number.</returns>
    public static int Next(this IRandom random, int exclusiveMaximum) => ToInt(random.NextFloat(exclusiveMaximum));
    // 返回一个随机整数，范围在[0, exclusiveMaximum)之间

    /// <summary>
    /// Gets a random <see cref="int" /> such that inclusiveMinimum &lt;= n &lt; exclusiveMaximum.
    /// </summary>
    /// <returns>The random number.</returns>
    public static int Next(this IRandom random, int inclusiveMinimum, int exclusiveMaximum) =>
        ToInt(random.NextFloat(inclusiveMinimum, exclusiveMaximum));
    // 返回一个随机整数，范围在[inclusiveMinimum, exclusiveMaximum)之间

    /// <summary>
    /// Gets the previous unscaled <see cref="float" /> (between 0 and 1) scaled to a new range:
    /// 0 &lt;= x &lt; <paramref name="exclusiveMaximum" />.
    /// </summary>
```
这些代码是C#中的扩展方法，用于生成随机数。每个方法都有注释说明其作用和返回值。
    /// <summary>
    /// 返回前一个未缩放的随机浮点数（介于0和1之间），缩放到一个新的范围：
    /// <paramref name="inclusiveMinimum" /> &lt;= n &lt; <paramref name="exclusiveMaximum" />。
    /// </summary>
    /// <returns>随机数。</returns>
    public static float PreviousFloat(this IRandom random, float inclusiveMinimum, float exclusiveMaximum) =>
        Scale(random.PreviousFloat(), inclusiveMinimum, exclusiveMaximum);

    /// <summary>
    /// 返回前一个未缩放的随机浮点数（介于0和1之间），缩放到一个新的范围：
    /// 0 &lt;= n &lt; <paramref name="exclusiveMaximum" />。
    /// </summary>
    /// <returns>随机数。</returns>
    public static int Previous(this IRandom random, int exclusiveMaximum) =>
        ToInt(random.PreviousFloat(exclusiveMaximum));
```
在这段代码中，我们定义了三个扩展方法，用于生成随机数并进行缩放。每个方法都有注释来解释其作用和返回值。
    # 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
    def read_zip(fname):
        # 根据 ZIP 文件名读取其二进制，封装成字节流
        bio = BytesIO(open(fname, 'rb').read())
        # 使用字节流里面内容创建 ZIP 对象
        zip = zipfile.ZipFile(bio, 'r')
        # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
        fdict = {n:zip.read(n) for n in zip.namelist()}
        # 关闭 ZIP 对象
        zip.close()
        # 返回结果字典
        return fdict
        if (exclusiveMaximum <= inclusiveMinimum)
        {
            // 如果exclusiveMaximum小于等于inclusiveMinimum，则抛出参数超出范围的异常
            throw new ArgumentOutOfRangeException(nameof(exclusiveMaximum), "Must be greater than inclusiveMinimum.");
        }

        // 计算范围
        var range = exclusiveMaximum - inclusiveMinimum;
        // 计算zeroToOne在范围内的值
        return zeroToOne * range + inclusiveMinimum;
    }

    // 将浮点数转换为整数
    private static int ToInt(float value) => (int)Math.Floor(value);
}
```