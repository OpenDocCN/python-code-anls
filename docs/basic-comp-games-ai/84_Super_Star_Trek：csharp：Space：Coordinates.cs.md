# `84_Super_Star_Trek\csharp\Space\Coordinates.cs`

```
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制数据，并封装成字节流
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    zip.close()  # 关闭 ZIP 对象
    return fdict  # 返回结果字典
    // 定义一个只读属性 Y，用于获取 Y 坐标值
    internal int Y { get; }

    // 定义一个只读属性 RegionIndex，用于获取区域索引值
    internal int RegionIndex { get; }

    // 定义一个只读属性 SubRegionIndex，用于获取子区域索引值
    internal int SubRegionIndex { get; }

    // 静态方法 Validated，用于验证传入的值是否在 0 到 7 之间，并返回验证后的值
    private static int Validated(int value, string argumentName)
    {
        if (value >= 0 && value <= 7) { return value; }
        // 如果传入的值不在 0 到 7 之间，则抛出 ArgumentOutOfRangeException 异常
        throw new ArgumentOutOfRangeException(argumentName, value, "Must be 0 to 7 inclusive");
    }

    // 静态方法 IsValid，用于验证传入的值是否在 0 到 7 之间，并返回验证结果
    private static bool IsValid(int value) => value >= 0 && value <= 7;

    // 重写 ToString 方法，返回 X 和 Y 坐标值加 1 后的字符串
    public override string ToString() => $"{X+1} , {Y+1}";

    // 解构方法，用于将 X 和 Y 坐标值分解出来
    internal void Deconstruct(out int x, out int y)
    {
        x = X;
        y = Y;  # 将变量 Y 的值赋给变量 y

    }  # 结束代码块

    internal static bool TryCreate(float x, float y, out Coordinates coordinates)
    {
        var roundedX = Round(x);  # 将 x 值四舍五入并赋给变量 roundedX
        var roundedY = Round(y);  # 将 y 值四舍五入并赋给变量 roundedY

        if (IsValid(roundedX) && IsValid(roundedY))  # 如果 roundedX 和 roundedY 都是有效的
        {
            coordinates = new Coordinates(roundedX, roundedY);  # 创建一个新的 Coordinates 对象，并赋给 coordinates
            return true;  # 返回 true
        }

        coordinates = default;  # 将默认值赋给 coordinates
        return false;  # 返回 false

        static int Round(float value) => (int)Math.Round(value, MidpointRounding.AwayFromZero);  # 定义一个局部函数 Round，用于将浮点数四舍五入为整数
    }
    // 定义一个内部方法，用于获取当前坐标到目标坐标的方向和距离，并返回一个包含方向和距离的元组
    internal (float Direction, float Distance) GetDirectionAndDistanceTo(Coordinates destination) =>
        DirectionAndDistance.From(this).To(destination);

    // 定义一个内部方法，用于获取当前坐标到目标坐标的距离，并返回距离值
    internal float GetDistanceTo(Coordinates destination)
    {
        // 调用 GetDirectionAndDistanceTo 方法获取方向和距离，使用占位符 _ 忽略方向，只获取距离
        var (_, distance) = GetDirectionAndDistanceTo(destination);
        // 返回距离值
        return distance;
    }
}
```