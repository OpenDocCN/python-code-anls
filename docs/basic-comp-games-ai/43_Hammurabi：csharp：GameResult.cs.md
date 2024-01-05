# `43_Hammurabi\csharp\GameResult.cs`

```
// 存储最终游戏结果的记录
public record GameResult
{
    // 获取玩家的表现评级
    public PerformanceRating Rating { get; init; }

    // 获取城市每人的土地面积
    public int AcresPerPerson { get; init; }

    // 获取最后一年在任期内挨饿的人数
        public int FinalStarvation { get; init; } // 最终饥饿人数

        /// <summary>
        /// Gets the total number of people who starved.
        /// </summary>
        public int TotalStarvation { get; init; } // 饥饿人数总计

        /// <summary>
        /// Gets the average starvation rate per year (as a percentage
        /// of population).
        /// </summary>
        public int AverageStarvationRate { get; init; } // 平均每年饥饿率（作为人口的百分比）

        /// <summary>
        /// Gets the number of people who want to assassinate the player.
        /// </summary>
        public int Assassins { get; init; } // 想要刺杀玩家的人数

        /// <summary>
        /// Gets a flag indicating whether the player was impeached for
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制数据，并封装成字节流对象
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建一个 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名列表，读取每个文件的数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
```