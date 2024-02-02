# `basic-computer-games\61_Math_Dice\java\Die.java`

```py
import java.util.Random;

public class Die {
    private static final int DEFAULT_SIDES = 6;  // 默认骰子的面数
    private int faceValue;  // 骰子当前的点数
    private int sides;  // 骰子的面数
    private Random generator = new Random();  // 生成随机数的对象

    /**
     * Construct a new Die with default sides
     */
    public Die() {
        this.sides = DEFAULT_SIDES;  // 使用默认的面数初始化骰子
        this.faceValue = 1 + generator.nextInt(sides);  // 生成随机点数
    }

    /**
     * Generate a new random number between 1 and sides to be stored in faceValue
     */
    private void throwDie() {
        this.faceValue = 1 + generator.nextInt(sides);  // 生成新的随机点数
    }


    /**
     * @return the faceValue
     */
    public int getFaceValue() {
        return faceValue;  // 返回当前骰子的点数
    }


    public void printDie() {
        throwDie();  // 抛骰子，生成新的点数
        int x = this.getFaceValue();  // 获取当前骰子的点数

        System.out.println(" ----- ");  // 打印骰子的上边框

        if(x==4||x==5||x==6) {
            printTwo();  // 打印包含两个点的行
        } else if(x==2||x==3) {
            System.out.println("| *   |");  // 打印包含一个点的行
        } else {
            printZero();  // 打印空行
        }

        if(x==1||x==3||x==5) {
            System.out.println("|  *  |");  // 打印包含一个点的行
        } else if(x==2||x==4) {
            printZero();  // 打印空行
        } else {
            printTwo();  // 打印包含两个点的行
        }

        if(x==4||x==5||x==6) {
            printTwo();  // 打印包含两个点的行
        } else if(x==2||x==3) {
            System.out.println("|   * |");  // 打印包含一个点的行
        } else {
            printZero();  // 打印空行
        }

        System.out.println(" ----- ");  // 打印骰子的下边框
    }

    private void printZero() {
        System.out.println("|     |");  // 打印空行
    }

    private void printTwo() {
        System.out.println("| * * |");  // 打印包含两个点的行
    }
}
```