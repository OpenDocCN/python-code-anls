# `83_Stock_Market\csharp\Assets.cs`

```
// 引入不可变数组的命名空间
using System.Collections.Immutable;

namespace Game
{
    /// <summary>
    /// 存储玩家的资产。
    /// </summary>
    public record Assets
    {
        /// <summary>
        /// 获取玩家现金金额。
        /// </summary>
        public double Cash { get; init; } // 初始化现金金额

        /// <summary>
        /// 获取每家公司拥有的股票数量。
        /// </summary>
        public ImmutableArray<int> Portfolio { get; init; } // 初始化不可变数组
    }
}
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```