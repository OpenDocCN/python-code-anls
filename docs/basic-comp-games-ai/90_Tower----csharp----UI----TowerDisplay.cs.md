# `basic-computer-games\90_Tower\csharp\UI\TowerDisplay.cs`

```py
using System;
using System.Text;
using Tower.Models;

namespace Tower.UI
{
    internal class TowerDisplay
    {
        private readonly Towers _towers; // 声明私有字段 _towers，用于存储 Towers 对象

        public TowerDisplay(Towers towers) // 构造函数，接受 Towers 对象作为参数
        {
            _towers = towers; // 将传入的 Towers 对象赋值给 _towers
        }

        public override string ToString() // 重写 ToString 方法
        {
            var builder = new StringBuilder(); // 创建 StringBuilder 对象

            foreach (var row in _towers) // 遍历 _towers 对象
            {
                AppendTower(row.Item1); // 调用 AppendTower 方法，传入第一个元素
                AppendTower(row.Item2); // 调用 AppendTower 方法，传入第二个元素
                AppendTower(row.Item3); // 调用 AppendTower 方法，传入第三个元素
                builder.AppendLine(); // 在 StringBuilder 对象中添加换行符
            }

            return builder.ToString(); // 返回 StringBuilder 对象转换为字符串后的结果

            void AppendTower(int size) // 定义局部方法 AppendTower，接受一个整数参数
            {
                var padding = 10 - size / 2; // 计算填充空格的数量
                builder.Append(' ', padding).Append('*', Math.Max(1, size)).Append(' ', padding); // 在 StringBuilder 对象中添加填充空格和星号
            }
        }
    }
}
```