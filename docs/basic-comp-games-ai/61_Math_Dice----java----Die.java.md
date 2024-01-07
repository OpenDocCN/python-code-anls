# `basic-computer-games\61_Math_Dice\java\Die.java`

```

// 导入 java.util.Random 类
import java.util.Random;

// 定义 Die 类
public class Die {
    // 定义默认的骰子面数
    private static final int DEFAULT_SIDES = 6;
    // 定义骰子的当前面值和面数
    private int faceValue;
    private int sides;
    // 创建 Random 对象用于生成随机数
    private Random generator = new Random();

    /**
     * 构造一个具有默认面数的新骰子
     */
    public Die() {
        this.sides = DEFAULT_SIDES;
        // 生成一个介于 1 和 sides 之间的随机数，并存储在 faceValue 中
        this.faceValue = 1 + generator.nextInt(sides);
    }

    /**
     * 生成一个介于 1 和 sides 之间的随机数，并存储在 faceValue 中
     */
    private void throwDie() {
        this.faceValue = 1 + generator.nextInt(sides);
    }

    /**
     * 返回当前骰子的面值
     * @return the faceValue
     */
    public int getFaceValue() {
        return faceValue;
    }

    // 打印骰子的图形表示
    public void printDie() {
        // 抛骰子，获取当前面值
        throwDie();
        int x = this.getFaceValue();

        System.out.println(" ----- ");

        // 根据面值打印不同的图形表示
        if(x==4||x==5||x==6) {
            printTwo();
        } else if(x==2||x==3) {
            System.out.println("| *   |");
        } else {
            printZero();
        }

        if(x==1||x==3||x==5) {
            System.out.println("|  *  |");
        } else if(x==2||x==4) {
            printZero();
        } else {
            printTwo();
        }

        if(x==4||x==5||x==6) {
            printTwo();
        } else if(x==2||x==3) {
            System.out.println("|   * |");
        } else {
            printZero();
        }

        System.out.println(" ----- ");
    }

    // 打印空白图形
    private void printZero() {
        System.out.println("|     |");
    }

    // 打印带有 * 的图形
    private void printTwo() {
        System.out.println("| * * |");
    }
}

```