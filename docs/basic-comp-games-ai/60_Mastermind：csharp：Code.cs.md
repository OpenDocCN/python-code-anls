# `60_Mastermind\csharp\Code.cs`

```
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
        public Code(IEnumerable<int> colors)
        {
            // 将传入的颜色数组转换为数组并赋值给成员变量 m_colors
            m_colors = colors.ToArray();
            // 如果颜色数组长度为0，则抛出参数异常
            if (m_colors.Length == 0)
                throw new ArgumentException("A code must contain at least one position");
        }

        /// <summary>
        /// Compares this code with the given code.
        /// </summary>
        /// <param name="other">
        /// The code to compare.
        /// </param>
        /// <returns>
        /// A number of black pegs and a number of white pegs.  The number
        /// of black pegs is the number of positions that contain the same
        /// color in both codes.  The number of white pegs is the number of
        /// colors that appear in both codes, but in the wrong positions.
        /// </returns>
        public (int blacks, int whites) Compare(Code other)
            // What follows is the O(N^2) from the original BASIC program
            // (where N is the number of positions in the code).  Note that
            // there is an O(N) algorithm.  (Finding it is left as an
            // exercise for the reader.)
            // 以下是原始BASIC程序中的O(N^2)算法（其中N是代码中位置的数量）。请注意，还有一个O(N)算法。（找到它留给读者作为练习。）
            if (other.m_colors.Length != m_colors.Length)
                throw new ArgumentException("Only codes of the same length can be compared");
            // 如果其他代码的颜色数量与当前代码的颜色数量不同，抛出参数异常

            // Keeps track of which positions in the other code have already
            // been marked as exact or close matches.
            // 跟踪其他代码中已经标记为完全匹配或接近匹配的位置。
            var consumed = new bool[m_colors.Length];
            // 创建一个布尔数组，用于跟踪其他代码中已经标记的位置

            var blacks = 0;
            var whites = 0;

            for (var i = 0; i < m_colors.Length; ++i)
            {
                if (m_colors[i] == other.m_colors[i])
                {
                    ++blacks;
                    // 如果当前位置的颜色与其他代码中相同位置的颜色相同，黑色匹配数加1
                consumed[i] = true;  // 将当前颜色在当前代码中的位置标记为已使用
                }
                else
                {
                    // 检查当前颜色是否在另一段代码中出现。我们必须小心，不要考虑也是完全匹配的位置。
                    for (var j = 0; j < m_colors.Length; ++j)
                    {
                        if (!consumed[j] &&  // 如果另一段代码中的颜色位置未被使用
                            m_colors[i] == other.m_colors[j] &&  // 当前颜色与另一段代码中的颜色相同
                            m_colors[j] != other.m_colors[j])  // 当前颜色在另一段代码中的位置不是完全匹配
                        {
                            ++whites;  // 白色数量加一
                            consumed[j] = true;  // 将另一段代码中的颜色位置标记为已使用
                            break;
                        }
                    }
                }
            }
        /// <summary>
        /// 返回黑白棋中黑子和白子的数量
        /// </summary>
        private (int, int) CountStones()
        {
            int blacks = m_colors.Count(c => c == StoneColor.Black);
            int whites = m_colors.Count(c => c == StoneColor.White);
            return (blacks, whites);
        }

        /// <summary>
        /// 获取代码的字符串表示形式
        /// </summary>
        public override string ToString() =>
            string.Join(", ", m_colors.Select(index => Colors.List[index].ShortName).ToArray());
    }
}
```