# `10_Blackjack\java\src\ScoringUtils.java`

```
import java.util.List;  // 导入 List 类

public final class ScoringUtils {  // 创建名为 ScoringUtils 的公共最终类

	/**
	 * 计算手牌的价值。当手牌包含 A 时，如果这不导致爆牌，将其中一个 A 计为 11。
	 * 
	 * @param hand 要评估的手牌
	 * @return 手牌的数值。超过 21 表示爆牌。
	 */
	public static final int scoreHand(List<Card> hand) {  // 创建名为 scoreHand 的公共静态方法，接受一个名为 hand 的 Card 类型列表，返回一个整数
		int nAces = (int) hand.stream().filter(c -> c.value() == 1).count();  // 计算手牌中 A 的数量
		int value = hand.stream()
				.mapToInt(Card::value)  // 将手牌中每张牌的点数映射为整数
				.filter(v -> v != 1)  // 过滤掉 A
				.map(v -> v > 10 ? 10 : v)  // 所有花牌的点数为 10。'expr ? a : b' 语法称为 '三元运算符'
				.sum();  // 计算手牌中非 A 牌的点数总和
		value += nAces;  // 将 A 计为 1 的情况加到总点数中
		if (nAces > 0 && value <= 11) {
			value += 10; // We can use one of the aces to an 11
			// 如果手牌中有A且当前总点数小于等于11，则将A的点数视为11
			// 不能使用多于一个A作为11点，因为那样总点数会超过21，就会爆牌
		}
		return value;
	}

	/**
	 * Compares two hands accounting for natural blackjacks and busting using the
	 * java.lang.Comparable convention of returning positive or negative integers
	 * 
	 * @param handA hand to compare
	 * @param handB other hand to compare
	 * @return a negative integer, zero, or a positive integer as handA is less
	 *         than, equal to, or greater than handB.
	 */
	public static final int compareHands(List<Card> handA, List<Card> handB) {
		int scoreA = scoreHand(handA); // 计算手牌A的点数
		int scoreB = scoreHand(handB); // 计算手牌B的点数
		if (scoreA == 21 && scoreB == 21) { // 如果手牌A和手牌B都是21点
			if (handA.size() == 2 && handB.size() != 2) {
				return 1; // 如果手牌A有两张牌且手牌B不是两张牌，则手牌A赢得自然的21点
			} else if (handA.size() != 2 && handB.size() == 2) {
				return -1; // 如果手牌A不是两张牌且手牌B有两张牌，则手牌B赢得自然的21点
			} else {
				return 0; // 平局
			}
		} else if (scoreA > 21 || scoreB > 21) {
			if (scoreA > 21 && scoreB > 21) {
				return 0; // 平局，双方都爆牌
			} else if (scoreB > 21) {
				return 1; // A赢，B爆牌
			} else {
				return -1; // B赢，A爆牌
			}
		} else {
			return Integer.compare(scoreA, scoreB); // 返回分数比较结果
		}
	}
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，避免内存泄漏。
```