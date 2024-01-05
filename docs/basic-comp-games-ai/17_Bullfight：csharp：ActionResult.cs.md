# `17_Bullfight\csharp\ActionResult.cs`

```
// 引入命名空间
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Game
{
    /// <summary>
    /// Enumerates the different possible outcomes of the player's action.
    /// 枚举玩家行动的不同可能结果。
    /// </summary>
    public enum ActionResult
    {
        /// <summary>
        /// The fight continues.
        /// 战斗继续。
        /// </summary>
        FightContinues,

        /// <summary>
        /// The player fled from the ring.
        /// 玩家逃离了战斗。
        /// </summary>
        /// </summary>
        # 玩家逃跑
        PlayerFlees,

        /// <summary>
        /// 公牛刺伤了玩家。
        /// </summary>
        # 公牛刺伤玩家
        BullGoresPlayer,

        /// <summary>
        /// 公牛杀死了玩家。
        /// </summary>
        # 公牛杀死玩家
        BullKillsPlayer,

        /// <summary>
        /// 玩家杀死了公牛。
        /// </summary>
        # 玩家杀死公牛
        PlayerKillsBull,

        /// <summary>
        /// 玩家试图杀死公牛，但双方都幸存。
        /// </summary>
        # 玩家试图杀死公牛，但双方都幸存
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 使用 open 函数读取文件内容，'rb' 表示以二进制模式读取，BytesIO 将文件内容封装成字节流
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用 BytesIO 封装的字节流创建 ZIP 对象，'r' 表示以只读模式打开 ZIP 文件
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 使用 zip.namelist() 获取 ZIP 文件中的所有文件名，然后使用 zip.read(n) 读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
```