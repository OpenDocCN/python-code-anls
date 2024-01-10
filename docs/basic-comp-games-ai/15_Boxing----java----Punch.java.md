# `basic-computer-games\15_Boxing\java\Punch.java`

```
# 导入 java.util.Arrays 包
import java.util.Arrays;

# Punch 枚举类型，定义了不同类型的拳击动作
/**
 * Types of Punches
 */
public enum Punch {
    # 定义了四种不同的拳击动作，每个动作对应一个整数值
    FULL_SWING(1),
    HOOK(2),
    UPPERCUT(3),
    JAB(4);

    # 每种拳击动作对应的整数值
    private final int code;

    # 构造函数，初始化每种拳击动作对应的整数值
    Punch(int code) {
        this.code = code;
    }

    # 获取拳击动作对应的整数值
    int getCode() { return  code;}

    # 根据整数值获取对应的拳击动作
    public static Punch fromCode(int code) {
        return Arrays.stream(Punch.values()).filter(p->p.code == code).findAny().orElse(null);
    }

    # 随机获取一种拳击动作
    public static Punch random() {
        return Punch.fromCode(Basic.randomOf(4));
    }
}
```