# `basic-computer-games\60_Mastermind\csharp\Code.cs`

```

// 引入命名空间
using System;
using System.Collections.Generic;
using System.Linq;

// 命名空间 Game
namespace Game
{
    /// <summary>
    /// 表示游戏中的秘密代码。
    /// </summary>
    public class Code
    {
        // 私有只读整型数组，存储颜色
        private readonly int[] m_colors;

        /// <summary>
        /// 从给定的位置集合初始化 Code 类的新实例。
        /// </summary>
        /// <param name="colors">
        /// 包含每个位置的颜色。
        /// </param>
        public Code(IEnumerable<int> colors)
        {
            // 将颜色转换为数组
            m_colors = colors.ToArray();
            // 如果颜色数组长度为0，则抛出参数异常
            if (m_colors.Length == 0)
                throw new ArgumentException("A code must contain at least one position");
        }

        /// <summary>
        /// 将此代码与给定的代码进行比较。
        /// </summary>
        /// <param name="other">
        /// 要比较的代码。
        /// </param>
        /// <returns>
        /// 黑色标记的数量和白色标记的数量。黑色标记的数量是两个代码中包含相同颜色的位置的数量。白色标记的数量是两个代码中出现的颜色数量，但位置不对。
        /// </returns>
        public (int blacks, int whites) Compare(Code other)
        {
            // 以下是原始BASIC程序中的O(N^2)（其中N是代码中的位置数）。请注意，还有一个O(N)的算法。（留给读者作为练习）
            if (other.m_colors.Length != m_colors.Length)
                throw new ArgumentException("Only codes of the same length can be compared");

            // 跟踪其他代码中已标记为精确匹配或近似匹配的位置。
            var consumed = new bool[m_colors.Length];

            var blacks = 0;
            var whites = 0;

            for (var i = 0; i < m_colors.Length; ++i)
            {
                if (m_colors[i] == other.m_colors[i])
                {
                    ++blacks;
                    consumed[i] = true;
                }
                else
                {
                    // 检查当前颜色是否在其他代码中的其他位置出现。我们必须小心，不要考虑也是精确匹配的位置。
                    for (var j = 0; j < m_colors.Length; ++j)
                    {
                        if (!consumed[j] &&
                            m_colors[i] == other.m_colors[j] &&
                            m_colors[j] != other.m_colors[j])
                        {
                            ++whites;
                            consumed[j] = true;
                            break;
                        }
                    }
                }
            }

            return (blacks, whites);
        }

        /// <summary>
        /// 获取代码的字符串表示形式。
        /// </summary>
        public override string ToString() =>
            new (m_colors.Select(index => Colors.List[index].ShortName).ToArray());
    }
}

```