# `basic-computer-games\75_Roulette\java\oop\Bet.java`

```
/* A bet has a target (the code entered, which is 1-36, or special values for
 * the various groups, zero and double-zero), and an amount in dollars
 */
// 创建一个赌注对象，包括一个目标值（输入的代码，范围为1-36，或者特殊值代表各种组合，包括零和双零），以及赌注金额

public class Bet {
    public int target; // 目标值
    public int amount; // 金额

    /* bet on a target, of an amount */
    // 对一个目标值进行赌注，指定赌注金额
    public Bet(int on, int of) {
        target = on; amount = of;
    }

    /* check if this is a valid bet - on a real target and of a valid amount */
    // 检查这个赌注是否有效 - 目标值合法且赌注金额有效
    public boolean isValid() {
        return ((target > 0) && (target <= 50) &&
                (amount >= 5) && (amount <= 500));
    }

    /* utility to return either the odds amount in the case of a win, or zero for a loss */
    // 返回赢得情况下的赔率金额，或者输掉情况下的零
    private int m(boolean isWon, int odds) {
        return isWon? odds: 0;
    }

    /* look at the wheel to see if this bet won.
     * returns 0 if it didn't, or the odds if it did
     */
    // 查看轮盘结果，判断这个赌注是否赢了
    // 如果没有赢，返回0；如果赢了，返回赔率
    // 根据轮盘上的结果判断赢得的赌注
    public int winsOn(Wheel w) {
        // 如果目标数字小于37
        if (target < 37) {
            // 如果是数字1-36，并且是目标数字，返回35倍的赌注
            return m(w.isNumber() && (w.number() == target), 35);
        } else
            // 否则根据目标数字进行判断
            switch (target) {
            case 37:   // 1-12, 赔率为2
                return m(w.isNumber() && (w.number() <= 12), 2);
            case 38:   // 13-24, 赔率为2
                return m(w.isNumber() && (w.number() > 12) && (w.number() <= 24), 2);
            case 39:   // 25-36, 赔率为2
                return m(w.isNumber() && (w.number() > 24), 2);
            case 40:   // 第一列, 赔率为2
                return m(w.isNumber() && ((w.number() % 3) == 1), 2);
            case 41:   // 第二列, 赔率为2
                return m(w.isNumber() && ((w.number() % 3) == 2), 2);
            case 42:   // 第三列, 赔率为2
                return m(w.isNumber() && ((w.number() % 3) == 0), 2);
            case 43:   // 1-18, 赔率为1
                return m(w.isNumber() && (w.number() <= 18), 1);
            case 44:   // 19-36, 赔率为1
                return m(w.isNumber() && (w.number() > 18), 1);
            case 45:   // 偶数, 赔率为1
                return m(w.isNumber() && ((w.number() %2) == 0), 1);
            case 46:   // 奇数, 赔率为1
                return m(w.isNumber() && ((w.number() %2) == 1), 1);
            case 47:   // 红色, 赔率为1
                return m(w.isNumber() && (w.color() == Wheel.BLACK), 1);
            case 48:   // 黑色, 赔率为1
                return m(w.isNumber() && (w.color() == Wheel.RED), 1);
            case 49: // 单零, 赔率为35
                return m(w.value().equals("0"), 35);
            case 50: // 双零, 赔率为35
                return m(w.value().equals("00"), 35);
            }
        // 如果都不符合，则抛出异常
        throw new RuntimeException("Program Error - invalid bet");
    }
# 闭合前面的函数定义
```