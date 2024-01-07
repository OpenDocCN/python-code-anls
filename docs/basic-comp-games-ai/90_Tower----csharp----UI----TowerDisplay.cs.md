# `basic-computer-games\90_Tower\csharp\UI\TowerDisplay.cs`

```

// 引入命名空间
using System;
using System.Text;
using Tower.Models;

// 声明 TowerDisplay 类
namespace Tower.UI
{
    // 声明 TowerDisplay 类为内部类
    internal class TowerDisplay
    {
        // 声明私有字段 _towers，类型为 Towers
        private readonly Towers _towers;

        // TowerDisplay 类的构造函数，接受 Towers 类型的参数 towers
        public TowerDisplay(Towers towers)
        {
            // 将参数 towers 赋值给私有字段 _towers
            _towers = towers;
        }

        // 重写 ToString 方法
        public override string ToString()
        {
            // 创建 StringBuilder 对象
            var builder = new StringBuilder();

            // 遍历 _towers 中的每一行
            foreach (var row in _towers)
            {
                // 调用 AppendTower 方法，将第一、第二、第三个元素添加到 StringBuilder 中
                AppendTower(row.Item1);
                AppendTower(row.Item2);
                AppendTower(row.Item3);
                // 在 StringBuilder 中添加换行符
                builder.AppendLine();
            }

            // 返回 StringBuilder 的字符串表示形式
            return builder.ToString();

            // 定义局部方法 AppendTower，用于向 StringBuilder 中添加塔的表示
            void AppendTower(int size)
            {
                // 计算填充空格的数量
                var padding = 10 - size / 2;
                // 向 StringBuilder 中添加填充空格、*、填充空格
                builder.Append(' ', padding).Append('*', Math.Max(1, size)).Append(' ', padding);
            }
        }
    }
}

```