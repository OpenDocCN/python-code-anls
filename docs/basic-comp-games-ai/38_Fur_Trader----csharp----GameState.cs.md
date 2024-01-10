# `basic-computer-games\38_Fur_Trader\csharp\GameState.cs`

```
// 引入命名空间
using System;
using System.Collections.Generic;
using System.Text;

// 定义内部类 GameState
namespace FurTrader
{
    internal class GameState
    {
        // 内部属性，表示游戏是否结束
        internal bool GameOver { get; set; }

        // 内部属性，表示玩家的存款
        internal double Savings { get; set; }

        // 内部属性，表示远征次数
        internal int ExpeditionCount { get; set; }

        // 内部属性，表示未分配的毛皮数量
        internal int UnasignedFurCount { get; set; }

        // 内部属性，表示各种动物皮毛的数量
        internal int[] Pelts { get; private set; }

        // 内部属性，表示貂皮的数量
        internal int MinkPelts { get { return this.Pelts[0]; } set { this.Pelts[0] = value; } }

        // 内部属性，表示海狸皮的数量
        internal int BeaverPelts { get { return this.Pelts[1]; } set { this.Pelts[1] = value; } }

        // 内部属性，表示鼬皮的数量
        internal int ErminePelts { get { return this.Pelts[2]; } set { this.Pelts[2] = value; } }

        // 内部属性，表示狐狸皮的数量
        internal int FoxPelts { get { return this.Pelts[3]; } set { this.Pelts[3] = value; } }

        // 内部属性，表示选择的堡垒
        internal int SelectedFort { get; set; }

        // 构造函数，初始化游戏状态
        internal GameState()
        {
            this.Savings = 600;
            this.ExpeditionCount = 0;
            this.UnasignedFurCount = 190;
            this.Pelts = new int[4];
            this.SelectedFort = 0;
        }

        // 内部方法，开始新的回合
        internal void StartTurn()
        {
            this.SelectedFort = 0;              // 重置为默认值
            this.UnasignedFurCount = 190;       // 每个回合开始时有190个毛皮
            this.Pelts = new int[4];            // 每个回合重置毛皮数量
        }
    }
}
```