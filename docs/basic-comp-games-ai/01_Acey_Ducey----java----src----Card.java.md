# `basic-computer-games\01_Acey_Ducey\java\src\Card.java`

```
/**
 * 一副牌中的一张牌 - 值在2-14之间，覆盖了2-9的牌和J、Q、K、A
 */
public class Card {
    private int value; // 牌的值
    private String name; // 牌的名称

    Card(int value) { // 构造函数，初始化牌的值
        init(value);
    }

    private void init(int value) { // 初始化函数，根据值确定牌的名称
        this.value = value; // 设置牌的值
        if (value < 11) { // 如果值小于11，直接用数字作为名称
            this.name = String.valueOf(value);
        } else { // 否则根据值选择对应的名称
            switch (value) {
                case 11:
                    this.name = "Jack";
                    break;
                case 12:
                    this.name = "Queen";
                    break;
                case 13:
                    this.name = "King";
                    break;
                case 14:
                    this.name = "Ace";
                    break;
                default:
                    this.name = "Unknown"; // 默认为未知
            }
        }
    }

    public int getValue() { // 获取牌的值
        return value;
    }

    public String getName() { // 获取牌的名称
        return name;
    }
}
```