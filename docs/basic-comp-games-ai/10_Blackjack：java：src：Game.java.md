# `d:/src/tocomm/basic-computer-games\10_Blackjack\java\src\Game.java`

```
import java.util.ArrayList;  // 导入 ArrayList 类
import java.util.Collection;  // 导入 Collection 类
import java.util.List;  // 导入 List 类
import java.text.DecimalFormat;  // 导入 DecimalFormat 类

/**
 * This is the primary class that runs the game itself.
 */
public class Game {
    
    private Deck deck;  // 创建一个私有的 Deck 对象变量
    private UserIo userIo;  // 创建一个私有的 UserIo 对象变量

    public Game(Deck deck, UserIo userIo) {  // 创建一个构造函数，接受 Deck 和 UserIo 对象作为参数
        this.deck = deck;  // 将传入的 Deck 对象赋值给私有的 deck 变量
        this.userIo = userIo;  // 将传入的 UserIo 对象赋值给私有的 userIo 变量
    }

	/**
	 * Run the game, running rounds until ended with CTRL+D/CTRL+Z or CTRL+C
     */
	 */
    public void run() {
		// 打印游戏标题
		userIo.println("BLACK JACK", 31);
		// 打印游戏信息
		userIo.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n", 15);
		// 如果用户想要游戏说明，则打印游戏规则
		if(userIo.promptBoolean("DO YOU WANT INSTRUCTIONS")){
			userIo.println("THIS IS THE GAME OF 21. AS MANY AS 7 PLAYERS MAY PLAY THE");
			userIo.println("GAME. ON EACH DEAL, BETS WILL BE ASKED FOR, AND THE");
			userIo.println("PLAYERS' BETS SHOULD BE TYPED IN. THE CARDS WILL THEN BE");
			userIo.println("DEALT, AND EACH PLAYER IN TURN PLAYS HIS HAND. THE");
			userIo.println("FIRST RESPONSE SHOULD BE EITHER 'D', INDICATING THAT THE");
			userIo.println("PLAYER IS DOUBLING DOWN, 'S', INDICATING THAT HE IS");
			userIo.println("STANDING, 'H', INDICATING HE WANTS ANOTHER CARD, OR '/',");
			userIo.println("INDICATING THAT HE WANTS TO SPLIT HIS CARDS. AFTER THE");
			userIo.println("INITIAL RESPONSE, ALL FURTHER RESPONSES SHOULD BE 'S' OR");
			userIo.println("'H', UNLESS THE CARDS WERE SPLIT, IN WHICH CASE DOUBLING");
			userIo.println("DOWN IS AGAIN PERMITTED. IN ORDER TO COLLECT FOR");
			userIo.println("BLACKJACK, THE INITIAL RESPONSE SHOULD BE 'S'.");
		}

		// 初始化玩家数量
		int nPlayers = 0;
		while(nPlayers < 1 || nPlayers > 7) {  # 当玩家数量小于1或大于7时，循环提示用户输入玩家数量
			nPlayers = userIo.promptInt("NUMBER OF PLAYERS");  # 通过用户输入获取玩家数量
		}

		deck.reshuffle();  # 重新洗牌

		Player dealer = new Player(0); //Dealer is Player 0  # 创建一个代表庄家的玩家对象，编号为0

		List<Player> players = new ArrayList<>();  # 创建一个玩家列表

		for(int i = 0; i < nPlayers; i++) {  # 循环创建玩家对象，编号从1到nPlayers
			players.add(new Player(i + 1));
		}

		while(true) {  # 无限循环
			while(!betsAreValid(players)){  # 当玩家的下注不合法时，循环执行以下操作
				userIo.println("BETS:");  # 输出提示信息
				for(int i = 0; i < nPlayers; i++) {  # 循环获取每个玩家的下注金额
					double bet = userIo.promptDouble("#" + (i + 1));  # 通过用户输入获取玩家的下注金额
					players.get(i).setCurrentBet(bet);  # 设置玩家的当前下注金额
				}
			}

			// It doesn't *really* matter whether we deal two cards at once to each player
			// or one card to each and then a second card to each, but this technically
			// mimics the way a deal works in real life.
			for(int i = 0; i < 2; i++){
				for(Player player : players){
					player.dealCard(deck.deal());  // 为每个玩家发牌
				}
				dealer.dealCard(deck.deal());  // 庄家发牌
			}

			printInitialDeal(players, dealer);  // 打印初始发牌情况

			if(dealer.getHand().get(0).value() == 1) {  // 如果庄家的第一张牌是A，则进行保险操作
				collectInsurance(players);  // 收取保险
			}

			if(ScoringUtils.scoreHand(dealer.getHand()) == 21) {  // 如果庄家的手牌总点数为21
				userIo.println("DEALER HAS " + dealer.getHand().get(1).toProseString() + " IN THE HOLE");  // 打印庄家手牌中第二张牌的信息
# 如果玩家选择放弃，则打印"FOR BLACKJACK"
userIo.println("FOR BLACKJACK");
# 否则，获取庄家的第一张牌
Card dealerFirstCard = dealer.getHand().get(0);
# 如果庄家的第一张牌点数为1或大于9，则打印"NO DEALER BLACKJACK."
if(dealerFirstCard.value() == 1 || dealerFirstCard.value() > 9) {
    userIo.println("");
    userIo.println("NO DEALER BLACKJACK.");
} 
# 遍历玩家列表，为每个玩家进行游戏
for(Player player : players){
    play(player);
}
# 如果庄家应该继续游戏，则为庄家进行游戏
if(shouldPlayDealer(players)){
    playDealer(dealer);
# 否则，打印庄家的第二张牌
} else {
    userIo.println("DEALER HAD " + dealer.getHand().get(1).toProseString() + " CONCEALED.");
}
# 评估本轮游戏结果
evaluateRound(players, dealer);
	protected void collectInsurance(Iterable<Player> players) {
		// 提示用户是否购买保险
		boolean isInsurance = userIo.promptBoolean("ANY INSURANCE");
		// 如果购买了保险
		if(isInsurance) {
			// 输出保险赌注
			userIo.println("INSURANCE BETS");
			// 遍历玩家集合
			for(Player player : players) {
				// 循环直到输入合法的保险赌注
				while(true) {
					// 提示用户输入玩家编号对应的保险赌注
					double insuranceBet = userIo.promptDouble("# " + player.getPlayerNumber() + " ");
					// 如果输入的赌注大于等于0且小于等于玩家当前赌注的一半
					if(insuranceBet >= 0 && insuranceBet <= (player.getCurrentBet() / 2)) {
						// 设置玩家的保险赌注并结束循环
						player.setInsuranceBet(insuranceBet);
						break;
					}
				}
			}
		}
	}
	 * Print the cards for each player and the up card for the dealer.
	 * Prints the initial deal in the following format:
	 *		
	 *	PLAYER 1     2    DEALER
     *         7    10     4   
     *         2     A   
	 */
	private void printInitialDeal(List<Player> players, Player dealer) {
	
        StringBuilder output = new StringBuilder(); 
		output.append("PLAYERS ");  // 添加 "PLAYERS " 到输出字符串
		for (Player player : players) {
			output.append(player.getPlayerNumber() + "\t");  // 添加每个玩家的编号和制表符到输出字符串
		}
		output.append("DEALER\n");  // 添加 "DEALER" 和换行符到输出字符串
		//Loop through two rows of cards		
        for (int j = 0; j < 2; j++) {  // 循环两次，表示两行牌
			output.append("\t");  // 添加制表符到输出字符串
			for (Player player : players) {
				output.append(player.getHand().get(j).toString()).append("\t");  // 添加每个玩家手中的牌到输出字符串，并加上制表符
			}
			# 如果玩家手中的第一张牌是庄家的明牌，则将其添加到输出列表中
			if(j == 0 ){
				output.append(dealer.getHand().get(j).toString());
			}
			# 在输出列表中添加换行符
			output.append("\n");
		}
		# 使用用户输入输出对象打印输出列表的内容
		userIo.print(output.toString());
	}

	/**
	 * Plays the players turn. Prompts the user to hit (H), stay (S), or if
	 * appropriate, split (/) or double down (D), and then performs those
	 * actions. On a hit, prints "RECEIVED A  [x]  HIT? "
	 * 
	 * @param player
	 */
	# 执行玩家的回合，提示玩家进行hit（H）、stay（S），如果适用，还可以进行split（/）或double down（D）操作
	# 在进行hit操作时，打印"RECEIVED A  [x]  HIT? "
	protected void play(Player player) {
		# 调用重载的play方法，传入玩家对象和1作为参数
		play(player, 1);
	}
	private void play(Player player, int handNumber) { // 定义一个名为play的方法，接受一个Player对象和一个整数作为参数
		String action; // 声明一个名为action的字符串变量
		if(player.isSplit()){ // 如果玩家已经分牌
			action = userIo.prompt("HAND " + handNumber); // 从用户输入中获取动作
		} else {
			action = userIo.prompt("PLAYER " + player.getPlayerNumber() + " "); // 从用户输入中获取动作
		}
		while(true){ // 进入无限循环
			if(action.equalsIgnoreCase("H")){ // 如果动作是“H”（要牌）
				Card c = deck.deal(); // 从牌堆中发一张牌
				player.dealCard(c, handNumber); // 玩家手中的牌加入这张牌
				if(ScoringUtils.scoreHand(player.getHand(handNumber)) > 21){ // 如果玩家手中的牌点数大于21
					userIo.println("RECEIVED " + c.toProseString() + "  ...BUSTED"); // 输出收到的牌和“爆牌”提示
					break; // 结束循环
				}
				action = userIo.prompt("RECEIVED " + c.toProseString() + " HIT"); // 从用户输入中获取动作
			} else if(action.equalsIgnoreCase("S")){ // 如果动作是“S”（停牌）
				break; // 结束循环
			} else if(action.equalsIgnoreCase("D") && player.canDoubleDown(handNumber)) { // 如果动作是“D”（加倍），并且玩家可以加倍
				Card c = deck.deal(); // 从牌堆中发一张牌
				# 玩家选择双倍下注，传入参数为c和handNumber
				player.doubleDown(c, handNumber);
				# 如果玩家手中的牌点数大于21，则打印“RECEIVED”和牌面点数，并且结束当前循环
				if(ScoringUtils.scoreHand(player.getHand(handNumber)) > 21){
					userIo.println("RECEIVED " + c.toProseString() + "  ...BUSTED");
					break;
				}
				# 打印“RECEIVED”和牌面点数
				userIo.println("RECEIVED " + c.toProseString());
				# 结束当前循环
				break;
			} else if(action.equalsIgnoreCase("/")) { # 如果玩家选择分牌
				# 如果玩家已经分牌，则提示玩家输入“H, S OR D”
				if(player.isSplit()) {
					action = userIo.prompt("TYPE H, S OR D, PLEASE");
				# 如果玩家可以分牌
				} else if(player.canSplit()) {
					# 玩家分牌
					player.split();
					# 从牌堆中发一张牌给第一手
					Card card = deck.deal();
					player.dealCard(card, 1);
					# 打印“FIRST HAND RECEIVES”和牌面点数
					userIo.println("FIRST HAND RECEIVES " + card.toProseString());
					# 从牌堆中发一张牌给第二手
					card = deck.deal();
					player.dealCard(card, 2);
					// 打印玩家收到的第二张牌的信息
					userIo.println("SECOND HAND RECEIVES " + card.toProseString());					
					// 如果玩家手中第一张牌的点数大于1，则不能在分牌后继续游戏
					if(player.getHand().get(0).value() > 1){ //Can't play after splitting aces
						// 分别对分牌后的两手牌进行游戏
						play(player, 1);
						play(player, 2);
					}
					// 返回，不跳出循环并打印另一个总数
					return; 
				} else {
					// 打印不允许分牌的信息
					userIo.println("SPLITTING NOT ALLOWED");
					// 等待玩家输入下一步操作
					action = userIo.prompt("PLAYER " + player.getPlayerNumber() + " ");
				}
			} else {
				// 如果玩家手中某一手牌的牌数为2，则提示玩家输入操作
				if(player.getHand(handNumber).size() == 2) {
					action = userIo.prompt("TYPE H,S,D, OR /, PLEASE");
				} else {
					// 如果玩家手中某一手牌的牌数不为2，则提示玩家输入操作
					action = userIo.prompt("TYPE H, OR S, PLEASE");
				}
			}
		}
		// 计算玩家某一手牌的总点数
		int total = ScoringUtils.scoreHand(player.getHand(handNumber));
		// 如果总点数为21
		if(total == 21) {
			userIo.println("BLACKJACK");
		} else {
			userIo.println("TOTAL IS " + total);
		}
	}
```
这段代码是一个条件语句，根据条件输出不同的信息。

```
	/**
	 * Check the Dealer's hand should be played out. If every player has either busted or won with natural Blackjack,
	 * the Dealer doesn't need to play.
	 * 
	 * @param players
	 * @return boolean whether the dealer should play
	 */
	protected boolean shouldPlayDealer(List<Player> players){
		for(Player player : players){
			int score = ScoringUtils.scoreHand(player.getHand());
			if(score < 21 || (score == 21 && player.getHand().size() > 2)){
				return true;
			}
			if(player.isSplit()){				
```
这段代码是一个方法的注释，解释了方法的作用和参数，以及返回值的含义。

```java
			userIo.println("BLACKJACK");
		} else {
			userIo.println("TOTAL IS " + total);
		}
	}
```
这段代码是一个条件语句，根据条件输出不同的信息。
				// 计算分数，判断是否需要继续抽牌
				int splitScore = ScoringUtils.scoreHand(player.getHand(2));
				// 如果分数小于21或者等于21但是手牌数量大于2，则返回true
				if(splitScore < 21 || (splitScore == 21 && player.getHand(2).size() > 2)){
					return true;
				}
			}
		}
		// 如果以上条件都不满足，则返回false
		return false;
	}	

	/**
	 * Play the dealer's hand. The dealer draws until they have >=17 or busts. Prints each draw as in the following example:
	 * 
	 * DEALER HAS A  5 CONCEALED FOR A TOTAL OF 11 
	 * DRAWS 10   ---TOTAL IS 21
	 *  
	 * @param dealerHand
	 */
	protected void playDealer(Player dealer) {
		// 计算庄家手牌的分数
		int score = ScoringUtils.scoreHand(dealer.getHand());
		// 打印庄家的手牌信息
		userIo.println("DEALER HAS " + dealer.getHand().get(1).toProseString() + " CONCEALED FOR A TOTAL OF " + score);
		if(score < 17):  # 如果分数小于17
			userIo.print("DRAWS")  # 输出"DRAWS"
		while(score < 17):  # 当分数小于17时
			Card dealtCard = deck.deal()  # 从牌堆中发一张牌
			dealer.dealCard(dealtCard)  # 庄家发牌
			score = ScoringUtils.scoreHand(dealer.getHand())  # 计算庄家手牌的分数
			userIo.print("  " + String.format("%-4s", dealtCard.toString()))  # 输出发的牌
		if(score > 21):  # 如果分数大于21
			userIo.println("...BUSTED\n")  # 输出"...BUSTED\n"
		else:  # 否则
			userIo.println("---TOTAL IS " + score + "\n")  # 输出"---TOTAL IS "和分数
	}

	/**
	 * Evaluates the result of the round, prints the results, and updates player/dealer totals.
	 * 评估每一轮游戏的结果
	 * 
	 * @param players 参与游戏的玩家列表
	 * @param dealerHand 庄家的手牌
	 */
	protected void evaluateRound(List<Player> players, Player dealer) {
		DecimalFormat formatter = new DecimalFormat("0.#"); // 创建一个格式化数字的对象，用于去除小数部分的末尾零
		for(Player player : players){ // 遍历玩家列表
			int result = ScoringUtils.compareHands(player.getHand(), dealer.getHand()); // 比较玩家手牌和庄家手牌的大小，返回结果
			double totalBet = 0; // 初始化总下注金额
			if(result > 0) { // 如果玩家赢了
				totalBet += player.getCurrentBet(); // 总下注金额增加玩家当前下注金额
			} else if(result < 0){ // 如果玩家输了
				totalBet -= player.getCurrentBet(); // 总下注金额减去玩家当前下注金额
			}
			if(player.isSplit()) { // 如果玩家进行了分牌
				int splitResult = ScoringUtils.compareHands(player.getHand(2), dealer.getHand()); // 比较玩家分牌后的手牌和庄家手牌的大小
				if(splitResult > 0):  # 如果分牌结果大于0
					totalBet += player.getSplitBet()  # 总下注金额加上玩家的分牌下注金额
				elif(splitResult < 0):  # 如果分牌结果小于0
					totalBet -= player.getSplitBet()  # 总下注金额减去玩家的分牌下注金额
			}
			if(player.getInsuranceBet() != 0):  # 如果玩家购买了保险
				dealerResult = ScoringUtils.scoreHand(dealer.getHand())  # 计算庄家手牌的得分
				if(dealerResult == 21 and len(dealer.getHand()) == 2):  # 如果庄家手牌得分为21且手牌数量为2
					totalBet += (player.getInsuranceBet() * 2)  # 总下注金额加上玩家保险下注金额的两倍
				else:
					totalBet -= player.getInsuranceBet()  # 总下注金额减去玩家的保险下注金额
			userIo.print("PLAYER " + player.getPlayerNumber())  # 打印玩家编号
			if(totalBet < 0):  # 如果总下注金额小于0
				userIo.print(" LOSES " + String.format("%6s", formatter.format(abs(totalBet))))  # 打印玩家输掉的金额
			elif(totalBet > 0):  # 如果总下注金额大于0
				userIo.print("  WINS " + String.format("%6s", formatter.format(totalBet))  # 打印玩家赢得的金额
			} else {
				userIo.print(" PUSHES      ");  # 如果玩家和庄家点数相同，则打印"PUSHES"
			}
			player.recordRound(totalBet);  # 记录玩家这一轮的下注金额
			dealer.recordRound(totalBet * (-1));  # 记录庄家这一轮的下注金额，因为是庄家输了，所以金额为负数
			userIo.println(" TOTAL= " + formatter.format(player.getTotal()));  # 打印玩家当前总金额
			player.resetHand();  # 重置玩家手中的牌
		}
		userIo.println("DEALER'S TOTAL= " + formatter.format(dealer.getTotal()) + "\n");  # 打印庄家当前总金额
		dealer.resetHand();  # 重置庄家手中的牌
	}

	/**
	 * Validates that all bets are between 0 (exclusive) and 500 (inclusive). Fractional bets are valid.
	 * 
	 * @param players The players with their current bet set.
	 * @return true if all bets are valid, false otherwise.
	 */
	public boolean betsAreValid(Collection<Player> players) {  # 定义一个方法用来验证玩家下注是否有效
		return players.stream()  # 使用流式操作遍历玩家集合
			// 使用流的 map 方法，获取每个玩家的当前下注金额
			.map(Player::getCurrentBet)
			// 使用流的 allMatch 方法，检查所有玩家的下注金额是否都大于0且小于等于500
			.allMatch(bet -> bet > 0 && bet <= 500);
	}
}
```