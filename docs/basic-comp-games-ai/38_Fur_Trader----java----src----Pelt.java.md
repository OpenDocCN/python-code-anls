# `basic-computer-games\38_Fur_Trader\java\src\Pelt.java`

```
/**
 * Pelt object - tracks the name and number of pelts the player has for this pelt type
 * Pelt对象 - 跟踪玩家拥有的这种皮毛类型的名称和数量
 */
public class Pelt {

    private final String name; // 皮毛的名称
    private int number; // 皮毛的数量

    public Pelt(String name, int number) { // 构造函数，初始化皮毛的名称和数量
        this.name = name; // 设置皮毛的名称
        this.number = number; // 设置皮毛的数量
    }

    public void setPeltCount(int pelts) { // 设置皮毛的数量
        this.number = pelts; // 更新皮毛的数量
    }

    public int getNumber() { // 获取皮毛的数量
        return this.number; // 返回皮毛的数量
    }

    public String getName() { // 获取皮毛的名称
        return this.name; // 返回皮毛的名称
    }

    public void lostPelts() { // 丢失所有皮毛
        this.number = 0; // 将皮毛数量设置为0
    }
}
*/
```