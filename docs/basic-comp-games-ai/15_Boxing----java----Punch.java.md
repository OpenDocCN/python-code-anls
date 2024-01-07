# `basic-computer-games\15_Boxing\java\Punch.java`

```

// 导入 Arrays 类
import java.util.Arrays;

/**
 * Punch 枚举类型
 */
public enum Punch {
    // 定义 FULL_SWING、HOOK、UPPERCUT、JAB 四种 Punch 枚举值
    FULL_SWING(1),
    HOOK(2),
    UPPERCUT(3),
    JAB(4);

    // 定义私有属性 code
    private final int code;

    // Punch 枚举值的构造函数，初始化 code 属性
    Punch(int code) {
        this.code = code;
    }

    // 获取 code 属性的方法
    int getCode() { return  code;}

    // 根据 code 获取对应的 Punch 枚举值
    public static Punch fromCode(int code) {
        // 使用流式操作，根据 code 过滤出对应的 Punch 枚举值
        return Arrays.stream(Punch.values()).filter(p->p.code == code).findAny().orElse(null);
    }

    // 随机获取一个 Punch 枚举值
    public static Punch random() {
        // 调用 Basic 类的 randomOf 方法，传入 4 作为参数，获取一个随机数作为 code，再根据 code 获取对应的 Punch 枚举值
        return Punch.fromCode(Basic.randomOf(4));
    }
}

```