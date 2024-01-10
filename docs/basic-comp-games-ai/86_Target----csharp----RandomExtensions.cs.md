# `basic-computer-games\86_Target\csharp\RandomExtensions.cs`

```
# 导入Games.Common.Randomness命名空间中的IRandom接口
using Games.Common.Randomness;

# 定义名为Target的命名空间
namespace Target
{
    # 定义名为RandomExtensions的静态类
    internal static class RandomExtensions
    {
        # 定义名为NextPosition的扩展方法，接收一个IRandom类型的参数，返回一个Point类型的实例
        public static Point NextPosition(this IRandom rnd) => new (
            # 使用IRandom对象的NextFloat方法生成一个随机角度，并转换成弧度表示
            Angle.InRotations(rnd.NextFloat()),
            # 使用IRandom对象的NextFloat方法生成一个随机角度，并转换成弧度表示
            Angle.InRotations(rnd.NextFloat()),
            # 使用IRandom对象的NextFloat方法生成一个随机数，并进行数学运算得到最终的位置值
            100000 * rnd.NextFloat() + rnd.NextFloat());
    }
}
```