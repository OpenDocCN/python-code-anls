# `d:/src/tocomm/basic-computer-games\43_Hammurabi\csharp\GameState.cs`

```
        public int PopulationIncrease { get; init; }

        /// <summary>
        /// Gets the amount of bushels of grain in storage.
        /// </summary>
        public int BushelsInStorage { get; init; }

        /// <summary>
        /// Gets the amount of bushels of grain harvested per acre.
        /// </summary>
        public int BushelsPerAcre { get; init; }

        /// <summary>
        /// Gets the number of acres of land owned by the city.
        /// </summary>
        public int AcresOwned { get; init; }

        /// <summary>
        /// Gets the price of land per acre.
        /// </summary>
        public int PricePerAcre { get; init; }

        /// <summary>
        /// Gets the amount of bushels of grain eaten by rats.
        /// </summary>
        public int BushelsEatenByRats { get; init; }

        /// <summary>
        /// Gets the amount of bushels of grain given to the people.
        /// </summary>
        public int BushelsGivenToPeople { get; init; }

        /// <summary>
        /// Gets the amount of bushels of grain sold to the people.
        /// </summary>
        public int BushelsSoldToPeople { get; init; }

        /// <summary>
        /// Gets the amount of bushels of grain fed to the people's horses.
        /// </summary>
        public int BushelsFedToHorses { get; init; }

        /// <summary>
        /// Gets the amount of people who starved this year.
        /// </summary>
        public int PeopleStarved { get; init; }

        /// <summary>
        /// Gets the amount of people who immigrated to the city this year.
        /// </summary>
        public int Immigrants { get; init; }

        /// <summary>
        /// Gets the amount of acres of land bought this year.
        /// </summary>
        public int AcresBought { get; init; }

        /// <summary>
        /// Gets the amount of acres of land sold this year.
        /// </summary>
        public int AcresSold { get; init; }

        /// <summary>
        /// Gets the amount of bushels of grain planted this year.
        /// </summary>
        public int BushelsPlanted { get; init; }

        /// <summary>
        /// Gets the amount of acres of land harvested this year.
        /// </summary>
        public int AcresHarvested { get; init; }

        /// <summary>
        /// Gets the amount of bushels of grain harvested this year.
        /// </summary>
        public int BushelsHarvested { get; init; }

        /// <summary>
        /// Gets the amount of bushels of grain eaten by the people.
        /// </summary>
        public int BushelsEatenByPeople { get; init; }
    }
}
        // 表示人口增长的属性
        public int PopulationIncrease { get; init; }

        /// <summary>
        /// 获取饥饿人口的数量。
        /// </summary>
        public int Starvation { get; init; }

        /// <summary>
        /// 获取城市的面积（单位：英亩）。
        /// </summary>
        public int Acres { get; init; }

        /// <summary>
        /// 获取每英亩土地的价格（以蒲式耳为单位）。
        /// </summary>
        public int LandPrice { get; init; }

        /// <summary>
        /// 获取城市仓库中的粮食数量（以蒲式耳为单位）。
        /// </summary>
        public int Stores { get; init; }  # 定义一个属性，表示存储的数量

        /// <summary>
        /// Gets the amount of food distributed to the people.
        /// </summary>
        public int FoodDistributed { get; init; }  # 定义一个属性，表示分发给人们的食物数量

        /// <summary>
        /// Gets the number of acres that were planted.
        /// </summary>
        public int AcresPlanted { get; init; }  # 定义一个属性，表示种植的土地面积

        /// <summary>
        /// Gets the number of bushels produced per acre.
        /// </summary>
        public int Productivity { get; init; }  # 定义一个属性，表示每英亩产出的蒲式耳数量

        /// <summary>
        /// Gets the amount of food lost to rats.
        /// </summary>
        public int Spoilage { get; init; }  // 定义一个公共整数属性Spoilage，用于表示损坏程度

        /// <summary>
        /// Gets a flag indicating whether the current year is a plague year.
        /// </summary>
        public bool IsPlagueYear { get; init; }  // 定义一个公共布尔属性IsPlagueYear，用于表示当前年份是否为瘟疫年

        /// <summary>
        /// Gets a flag indicating whether the player has been impeached.
        /// </summary>
        public bool IsPlayerImpeached { get; init; }  // 定义一个公共布尔属性IsPlayerImpeached，用于表示玩家是否被弹劾
    }
}
```