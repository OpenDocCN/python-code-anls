# `d:/src/tocomm/basic-computer-games\33_Dice\csharp\RollGenerator.cs`

```
// 引入命名空间
using System;
using System.Collections.Generic;

namespace BasicComputerGames.Dice
{
    // 定义 RollGenerator 类
    public class RollGenerator
    {
        // 创建 Random 对象
        static Random _rnd = new Random();

        // 重新设置随机数生成器的种子
        public static void ReseedRNG(int seed) => _rnd = new Random(seed);

        // 生成骰子点数的序列
        public IEnumerable<(int die1, int die2)> Rolls()
        {
            // 无限循环
            while (true)
            {
                // 返回两个骰子的点数
                yield return (_rnd.Next(1, 7), _rnd.Next(1, 7));
            }
        }
    }
}
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```