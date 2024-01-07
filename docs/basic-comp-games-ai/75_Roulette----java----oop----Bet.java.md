# `basic-computer-games\75_Roulette\java\oop\Bet.java`

```

/* A bet has a target (the code entered, which is 1-36, or special values for
 * the various groups, zero and double-zero), and an amount in dollars
 */
// 创建一个赌注类，包含一个目标（输入的代码，即1-36，或各种特殊组合，零和双零），以及一个金额（美元）

public class Bet {
    public int target; // 目标数字
    public int amount; // 金额

    /* bet on a target, of an amount */
    // 在一个目标上下注，下注金额
    public Bet(int on, int of) {
        target = on; amount = of;
    }

    /* check if this is a valid bet - on a real target and of a valid amount */
    // 检查这是否是一个有效的赌注 - 在一个真实的目标上下注，并且下注金额有效
    public boolean isValid() {
        return ((target > 0) && (target <= 50) &&
                (amount >= 5) && (amount <= 500));
    }

    /* utility to return either the odds amount in the case of a win, or zero for a loss */
    // 实用程序，在赢的情况下返回赔率金额，或者输的情况下返回零
    private int m(boolean isWon, int odds) {
        return isWon? odds: 0;
    }

    /* look at the wheel to see if this bet won.
     * returns 0 if it didn't, or the odds if it did
     */
    // 查看轮盘以查看这个赌注是否赢了。如果没有赢，则返回0，如果赢了则返回赔率
    public int winsOn(Wheel w) {
        if (target < 37) {
            // A number bet 1-36 wins at odds of 35 if it is the exact number
            // 1-36的数字赌注如果是确切的数字，则以35的赔率赢得
            return m(w.isNumber() && (w.number() == target), 35);
        } else
            switch (target) {
            case 37:   // 1-12, odds of 2
                return m(w.isNumber() && (w.number() <= 12), 2);
            case 38:   // 13-24, odds of 2
                return m(w.isNumber() && (w.number() > 12) && (w.number() <= 24), 2);
            case 39:   // 25-36, odds of 2
                return m(w.isNumber() && (w.number() > 24), 2);
            case 40:   // Column 1, odds of 2
                return m(w.isNumber() && ((w.number() % 3) == 1), 2);
            case 41:   // Column 2, odds of 2
                return m(w.isNumber() && ((w.number() % 3) == 2), 2);
            case 42:   // Column 3, odds of 2
                return m(w.isNumber() && ((w.number() % 3) == 0), 2);
            case 43:   // 1-18, odds of 1
                return m(w.isNumber() && (w.number() <= 18), 1);
            case 44:   // 19-36, odds of 1
                return m(w.isNumber() && (w.number() > 18), 1);
            case 45:   // even, odds of 1
                return m(w.isNumber() && ((w.number() %2) == 0), 1);
            case 46:   // odd, odds of 1
                return m(w.isNumber() && ((w.number() %2) == 1), 1);
            case 47:   // red, odds of 1
                return m(w.isNumber() && (w.color() == Wheel.BLACK), 1);
            case 48:   // black, odds of 1
                return m(w.isNumber() && (w.color() == Wheel.RED), 1);
            case 49: // single zero, odds of 35
                return m(w.value().equals("0"), 35);
            case 50: // double zero, odds of 35
                return m(w.value().equals("00"), 35);
            }
        throw new RuntimeException("Program Error - invalid bet");
    }
}

```