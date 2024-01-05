# `75_Roulette\java\oop\Bet.java`

```
/* A bet has a target (the code entered, which is 1-36, or special values for
 * the various groups, zero and double-zero), and an amount in dollars
 */
// 创建一个名为Bet的类，包含两个公共整数变量target和amount

public class Bet {
    public int target;
    public int amount;

    /* bet on a target, of an amount */
    // 创建一个名为Bet的构造函数，接受两个整数参数on和of，并将它们分别赋值给target和amount

    public Bet(int on, int of) {
        target = on; amount = of;
    }

    /* check if this is a valid bet - on a real target and of a valid amount */
    // 创建一个名为isValid的公共布尔类型方法，用于检查该赌注是否有效 - 是否是真实的目标和有效的金额

    public boolean isValid() {
        return ((target > 0) && (target <= 50) &&
                (amount >= 5) && (amount <= 500));
    }

    /* utility to return either the odds amount in the case of a win, or zero for a loss */
    // 创建一个实用方法，如果赢了则返回赔率金额，如果输了则返回零
    private int m(boolean isWon, int odds) {
        // 如果赢了，返回赔率；如果没赢，返回0
        return isWon? odds: 0;
    }

    /* 查看轮盘结果是否赢得了这个赌注。
     * 如果没有赢，返回0；如果赢了，返回赔率
     */
    public int winsOn(Wheel w) {
        if (target < 37) {
            // 如果是数字赌注1-36，并且是准确的数字，以35倍赔率赢得
            return m(w.isNumber() && (w.number() == target), 35);
        } else
            switch (target) {
            case 37:   // 1-12，以2倍赔率赢得
                return m(w.isNumber() && (w.number() <= 12), 2);
            case 38:   // 13-24，以2倍赔率赢得
                return m(w.isNumber() && (w.number() > 12) && (w.number() <= 24), 2);
            case 39:   // 25-36，以2倍赔率赢得
                return m(w.isNumber() && (w.number() > 24), 2);
            case 40:   // 第一列，以2倍赔率赢得
            case 49: // 单零，赔率为35
                return m(w.value().equals("0"), 35);
            case 50: // 双零，赔率为35
                return m(w.value().equals("00"), 35);
```

这段代码是一个 switch 语句，根据不同的情况返回不同的赔率。每个 case 后面的注释解释了该情况对应的赔率和条件。例如，case 49 对应的是单零，赔率为35，条件是判断轮盘的值是否为 "0"。
                return m(w.value().equals("00"), 35);
```
这行代码是一个条件返回语句，根据条件w.value().equals("00")的结果来返回不同的值。如果条件成立，返回35，否则返回m的值。

```
            }
```
这行代码是if语句的结束标记。

```
        throw new RuntimeException("Program Error - invalid bet");
```
这行代码是一个异常抛出语句，当条件不成立时抛出一个RuntimeException异常，异常信息为"Program Error - invalid bet"。

```
    }
```
这行代码是方法的结束标记。
```