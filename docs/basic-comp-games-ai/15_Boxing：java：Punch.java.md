# `d:/src/tocomm/basic-computer-games\15_Boxing\java\Punch.java`

```
import java.util.Arrays;  # 导入 java.util.Arrays 包，用于操作数组

/**
 * Types of Punches  # Punch 枚举类型的定义
 */
public enum Punch {  # 定义 Punch 枚举类型
    FULL_SWING(1),  # 枚举值 FULL_SWING，对应的 code 值为 1
    HOOK(2),  # 枚举值 HOOK，对应的 code 值为 2
    UPPERCUT(3),  # 枚举值 UPPERCUT，对应的 code 值为 3
    JAB(4);  # 枚举值 JAB，对应的 code 值为 4

    private final int code;  # 定义私有属性 code，用于存储枚举值对应的 code 值

    Punch(int code) {  # Punch 类的构造函数，用于初始化枚举值对应的 code 值
        this.code = code;  # 将传入的 code 值赋给私有属性 code
    }

    int getCode() { return  code;}  # 定义方法 getCode，用于获取枚举值对应的 code 值

    public static Punch fromCode(int code) {  # 定义静态方法 fromCode，用于根据 code 值获取对应的枚举值
        return Arrays.stream(Punch.values())  # 从 Punch 枚举类型中创建一个流
                .filter(p->p.code == code)  # 使用过滤器筛选出 code 值与给定值相等的 Punch 对象
                .findAny()  # 返回任意一个符合条件的元素
                .orElse(null);  # 如果没有找到符合条件的元素，则返回 null
    }

    public static Punch random() {
        return Punch.fromCode(Basic.randomOf(4));  # 返回一个随机生成的 Punch 对象，通过调用 fromCode 方法传入一个随机生成的 code 值
    }
}
```