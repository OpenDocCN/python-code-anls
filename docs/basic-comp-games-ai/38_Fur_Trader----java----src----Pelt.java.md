# `basic-computer-games\38_Fur_Trader\java\src\Pelt.java`

```
/**
 * Pelt object - tracks the name and number of pelts the player has for this pelt type
 */
public class Pelt {

    // 声明私有变量，用于存储皮毛的名称
    private final String name;
    // 声明私有变量，用于存储皮毛的数量
    private int number;

    // 构造函数，初始化皮毛的名称和数量
    public Pelt(String name, int number) {
        this.name = name;
        this.number = number;
    }

    // 设置皮毛的数量
    public void setPeltCount(int pelts) {
        this.number = pelts;
    }

    // 获取皮毛的数量
    public int getNumber() {
        return this.number;
    }

    // 获取皮毛的名称
    public String getName() {
        return this.name;
    }

    // 重置皮毛的数量为0
    public void lostPelts() {
        this.number = 0;
    }
}
```