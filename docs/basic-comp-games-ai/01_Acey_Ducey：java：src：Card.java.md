# `d:/src/tocomm/basic-computer-games\01_Acey_Ducey\java\src\Card.java`

```
/**
 * A card from a deck - the value is between 2-14 to cover
 * cards with a face value of 2-9 and then a Jack, Queen, King, and Ace
 */
public class Card {
    private int value; // 用于存储卡片的值，范围在2-14之间，代表2-9以及J、Q、K、A
    private String name; // 用于存储卡片的名称

    Card(int value) { // 构造函数，初始化卡片的值
        init(value);
    }

    private void init(int value) { // 初始化函数，根据值设置卡片的名称
        this.value = value; // 将传入的值赋给卡片的值
        if (value < 11) { // 如果值小于11
            this.name = String.valueOf(value); // 将值转换为字符串作为卡片的名称
        } else { // 如果值大于等于11
            switch (value) { // 根据值的不同情况设置不同的名称
                case 11:
                    this.name = "Jack"; // 如果值为11，名称为"Jack"
                    break;  # 终止当前的循环或者 switch 语句
                case 12:  # 如果 value 的值为 12
                    this.name = "Queen";  # 将 this.name 的值设为 "Queen"
                    break;  # 终止当前的循环或者 switch 语句
                case 13:  # 如果 value 的值为 13
                    this.name = "King";  # 将 this.name 的值设为 "King"
                    break;  # 终止当前的循环或者 switch 语句
                case 14:  # 如果 value 的值为 14
                    this.name = "Ace";  # 将 this.name 的值设为 "Ace"
                    break;  # 终止当前的循环或者 switch 语句

                default:  # 如果 value 的值不是以上任何一个 case
                    this.name = "Unknown";  # 将 this.name 的值设为 "Unknown"
            }
        }
    }

    public int getValue() {  # 定义一个公共方法 getValue，返回值为整数类型
        return value;  # 返回 value 的值
    }
# 定义一个公共方法，用于获取对象的名称
def getName():
    return name
```
这段代码定义了一个公共方法getName()，用于返回对象的名称。在Python中，方法的定义使用def关键字，方法体内的代码需要缩进。getName()方法没有参数，使用return关键字返回对象的名称。
```