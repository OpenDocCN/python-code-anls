# `d:/src/tocomm/basic-computer-games\83_Stock_Market\java\StockMarket.java`

```
import java.util.ArrayList;  // 导入 ArrayList 类
import java.util.InputMismatchException;  // 导入 InputMismatchException 类
import java.util.List;  // 导入 List 接口
import java.util.Random;  // 导入 Random 类
import java.util.Scanner;  // 导入 Scanner 类

/**
 * Stock Market Simulation
 *
 * Some of the original program's variables' documentation and their equivalent in this program:
 * A-MRKT TRND SLP;             marketTrendSlope  // 市场趋势斜率
 * B5-BRKRGE FEE;               brokerageFee  // 佣金费用
 * C-TTL CSH ASSTS;             cashAssets  // 现金资产总额
 * C5-TTL CSH ASSTS (TEMP);     tmpCashAssets  // 临时现金资产总额
 * C(I)-CHNG IN STK VAL;        changeStockValue  // 股票价值变化
 * D-TTL ASSTS;                 assets  // 总资产
 * E1,E2-LRG CHNG MISC;         largeChange1, largeChange2  // 大幅变动1、2
 * I1,I2-STCKS W LRG CHNG;      randomStockIndex1, randomStockIndex2  // 随机股票索引1、2
 * N1,N2-LRG CHNG DAY CNTS;     largeChangeNumberDays1, largeChangeNumberDays2  // 大幅变动天数1、2
 * P5-TTL DAYS PRCHSS;          totalDaysPurchases  // 总购买天数
 */
# P(I)-PRTFL CNTNTS;           portfolioContents
# 定义变量 portfolioContents，用于存储投资组合内容

# Q9-NEW CYCL?;                newCycle
# 定义变量 newCycle，用于表示是否进入新的周期

# S4-SGN OF A;                 slopeSign
# 定义变量 slopeSign，用于表示斜率的符号

# S5-TTL DYS SLS;              totalDaysSales
# 定义变量 totalDaysSales，用于表示总销售天数

# S(I)-VALUE/SHR;              stockValue
# 定义变量 stockValue，用于表示每股股票价值

# T-TTL STCK ASSTS;            totalStockAssets
# 定义变量 totalStockAssets，用于表示总股票资产

# T5-TTL VAL OF TRNSCTNS;      totalValueOfTransactions
# 定义变量 totalValueOfTransactions，用于表示总交易价值

# W3-LRG CHNG;                 bigChange
# 定义变量 bigChange，用于表示大幅变动

# X1-SMLL CHNG(<$1);           smallChange
# 定义变量 smallChange，用于表示小幅变动

# Z4,Z5,Z6-NYSE AVE.;          tmpNyseAverage, nyseAverage, nyseAverageChange
# 定义变量 tmpNyseAverage, nyseAverage, nyseAverageChange，用于表示纽约证券交易所的平均值和变化

# Z(I)-TRNSCT                  transactionQuantity
# 定义变量 transactionQuantity，用于表示交易数量

# new price = old price + (trend x old price) + (small random price
# change) + (possible large price change)
# 计算新的股票价格，根据趋势、随机小价格变动和可能的大价格变动来计算

# Converted from BASIC to Java by Aldrin Misquitta (@aldrinm)
# 代码由Aldrin Misquitta (@aldrinm)从BASIC转换为Java
```
```java
public class StockMarket {

    private static final Random random = new Random();
    // 创建一个静态的Random对象，用于生成随机数
		// 主程序入口，接受用户输入
		public static void main(String[] args) {

			// 创建一个 Scanner 对象，用于接收用户输入
			Scanner scan = new Scanner(System.in);

			// 打印游戏介绍
			printIntro();
			// 打印游戏帮助信息
			printGameHelp(scan);

			// 初始化股票列表
			final List<Stock> stocks = initStocks();

			// 随机生成市场趋势斜率
			double marketTrendSlope = Math.floor((random.nextFloat() / 10) * 100 + 0.5)/100f;
			double totalValueOfTransactions;
			int largeChangeNumberDays1 = 0;
			int largeChangeNumberDays2 = 0;

			// 为第一个趋势斜率 (A) 随机生成天数
			var t8 = randomNumber(1, 6);

			// 随机确定第一个趋势斜率 (A) 的正负
			if (random.nextFloat() <= 0.5) {
			marketTrendSlope = -marketTrendSlope; // 反转市场趋势斜率

		}

		// 初始化现金资产：C
		double cashAssets = 10000; // 初始现金资产为10000
		boolean largeChange1 = false; // 第一个股票是否有大变动
		boolean largeChange2 = false; // 第二个股票是否有大变动
		double tmpNyseAverage; // 临时纽约证券交易所平均值
		double nyseAverage = 0; // 纽约证券交易所平均值
		boolean inProgress = true; // 是否还在进行中
		var firstRound = true; // 是否是第一轮

		while (inProgress) {

			/* 原始文档：
			根据前一天的股票值随机产生新的股票值
			N1，N2是随机天数，分别决定何时股票I1将增加10点，股票I2将减少10点。
			如果N1天已经过去，选择I1，设置E1，确定新的N1
			*/
			// 初始化随机股票索引
			int randomStockIndex1 = 0;
			int randomStockIndex2 = 0;

			// 如果大幅变动天数1小于等于0，则随机选择一个股票索引，并设置大幅变动天数1和大幅变动1标志位
			if (largeChangeNumberDays1 <= 0) {
				randomStockIndex1 = randomNumber(0, stocks.size());
				largeChangeNumberDays1 = randomNumber(1, 6);
				largeChange1 = true;
			}
			// 如果大幅变动天数2小于等于0，则随机选择一个股票索引，并设置大幅变动天数2和大幅变动2标志位
			if (largeChangeNumberDays2 <= 0) {
				randomStockIndex2 = randomNumber(0, stocks.size());
				largeChangeNumberDays2 = randomNumber(1, 6);
				largeChange2 = true;
			}
			// 调整所有股票的价值，根据大幅变动标志位和随机选择的股票索引
			adjustAllStockValues(stocks, largeChange1, largeChange2, marketTrendSlope, stocks.get(randomStockIndex1), stocks.get(randomStockIndex2));

			// 重置大幅变动标志位
			largeChange1 = false;
			largeChange2 = false;
			// 大幅变动天数1减1
			largeChangeNumberDays1--;
			largeChangeNumberDays2--;

			//AFTER T8 DAYS RANDOMLY CHANGE TREND SIGN AND SLOPE
			// 在经过T8天后，随机改变趋势符号和斜率
			t8 = t8 - 1;
			if (t8 < 1) {
				// 重新生成市场趋势斜率
				marketTrendSlope = newMarketTrendSlope();
				// 重新生成T8的随机值
				t8 = randomNumber(1, 6);
			}

			//PRINT PORTFOLIO
			// 打印投资组合
			printPortfolio(firstRound, stocks);

			tmpNyseAverage = nyseAverage;
			nyseAverage = 0;
			double totalStockAssets = 0;
			for (Stock stock : stocks) {
				// 计算纽约证券交易所平均值
				nyseAverage = nyseAverage + stock.getStockValue();
				// 计算总股票资产
				totalStockAssets = totalStockAssets + stock.getStockValue() * stock.getPortfolioContents();
			}
			// 对纽约证券交易所平均值进行四舍五入
			nyseAverage = Math.floor(100 * (nyseAverage / 5) + .5) / 100f;
			// 计算纽约证券交易所平均变化
			double nyseAverageChange = Math.floor((nyseAverage - tmpNyseAverage) * 100 + .5) / 100f;

			// 计算总资产
			double assets = totalStockAssets + cashAssets;
			// 如果是第一轮，则打印纽约证券交易所平均值
			if (firstRound) {
				System.out.printf("\n\nNEW YORK STOCK EXCHANGE AVERAGE: %.2f", nyseAverage);
			} else {
				// 如果不是第一轮，则打印纽约证券交易所平均值和变化值
				System.out.printf("\n\nNEW YORK STOCK EXCHANGE AVERAGE: %.2f NET CHANGE %.2f", nyseAverage, nyseAverageChange);
			}

			// 计算并打印总股票资产
			totalStockAssets = Math.floor(100 * totalStockAssets + 0.5) / 100d;
			System.out.printf("\n\nTOTAL STOCK ASSETS ARE   $ %.2f", totalStockAssets);
			// 计算并打印总现金资产
			cashAssets = Math.floor(100 * cashAssets + 0.5) / 100d;
			System.out.printf("\nTOTAL CASH ASSETS ARE    $ %.2f", cashAssets);
			// 计算并打印总资产
			assets = Math.floor(100 * assets + .5) / 100d;
			System.out.printf("\nTOTAL ASSETS ARE         $ %.2f\n", assets);

			// 如果不是第一轮，则询问用户是否继续
			if (!firstRound) {
				System.out.print("\nDO YOU WISH TO CONTINUE (YES-TYPE 1, NO-TYPE 0)? ");
				// 读取用户输入的数字
				var newCycle = readANumber(scan);
				if (newCycle < 1) {  # 如果新周期小于1
					System.out.println("HOPE YOU HAD FUN!!");  # 打印消息“希望你玩得开心！”
					inProgress = false;  # 将inProgress标记为false，表示进程结束
				}
			}

			if (inProgress) {  # 如果进程仍在进行中
				boolean validTransaction = false;  # 声明并初始化validTransaction变量为false
				//    TOTAL DAY'S PURCHASES IN $:P5  # 注释：P5位置的总购买金额
				double totalDaysPurchases = 0;  # 声明并初始化totalDaysPurchases变量为0
				//    TOTAL DAY'S SALES IN $:S5  # 注释：S5位置的总销售金额
				double totalDaysSales = 0;  # 声明并初始化totalDaysSales变量为0
				double tmpCashAssets;  # 声明tmpCashAssets变量
				while (!validTransaction) {  # 当validTransaction为false时循环
					//INPUT TRANSACTIONS  # 输入交易
					readStockTransactions(stocks, scan);  # 调用readStockTransactions函数，传入stocks和scan参数
					totalDaysPurchases = 0;  # 将totalDaysPurchases重置为0
					totalDaysSales = 0;  # 将totalDaysSales重置为0

					validTransaction = true;  # 将validTransaction标记为true，表示有效交易
# 遍历股票列表中的每一支股票
for (Stock stock : stocks) {
    # 设置交易数量为向下取整的值
    stock.setTransactionQuantity(Math.floor(stock.getTransactionQuantity() + 0.5));
    # 如果交易数量大于0
    if (stock.getTransactionQuantity() > 0) {
        # 计算总购买金额
        totalDaysPurchases = totalDaysPurchases + stock.getTransactionQuantity() * stock.getStockValue();
    } else {
        # 计算总销售金额
        totalDaysSales = totalDaysSales - stock.getTransactionQuantity() * stock.getStockValue();
        # 如果卖出数量大于持有数量
        if (-stock.getTransactionQuantity() > stock.getPortfolioContents()) {
            # 输出错误信息并设置交易为无效
            System.out.println("YOU HAVE OVERSOLD A STOCK; TRY AGAIN.");
            validTransaction = false;
            break;
        }
    }
}

// 计算总交易价值
totalValueOfTransactions = totalDaysPurchases + totalDaysSales;
// 计算佣金费用
var brokerageFee = Math.floor(0.01 * totalValueOfTransactions * 100 + .5) / 100d;
// 计算现金资产
// 旧现金资产 - 总购买金额 - 佣金费用 + 总销售金额
					# 计算临时现金资产，减去总购买金额和佣金费用，再加上总销售金额
					tmpCashAssets = cashAssets - totalDaysPurchases - brokerageFee + totalDaysSales;
					# 如果临时现金资产小于0，打印出超支的金额，并将validTransaction设置为false
					if (tmpCashAssets < 0) {
						System.out.printf("\nYOU HAVE USED $%.2f MORE THAN YOU HAVE.", -tmpCashAssets);
						validTransaction = false;
					} else {
						# 否则，更新现金资产为临时现金资产
						cashAssets = tmpCashAssets;
					}
				}

				// 计算新的投资组合
				for (Stock stock : stocks) {
					# 更新每支股票的投资组合内容为原内容加上交易数量
					stock.setPortfolioContents(stock.getPortfolioContents() + stock.getTransactionQuantity());
				}

				# 将firstRound设置为false，表示已经不是第一轮交易了
				firstRound = false;
			}

		}
	}
	/**
	 * 生成介于 lowerBound（包括）和 upperBound（不包括）之间的随机整数
	 */
	private static int randomNumber(int lowerBound, int upperBound) {
		return random.nextInt((upperBound - lowerBound)) + lowerBound;
	}

	/**
	 * 生成新的市场趋势斜率
	 */
	private static double newMarketTrendSlope() {
		return randomlyChangeTrendSignAndSlopeAndDuration();
	}

	/**
	 * 打印投资组合信息
	 * @param firstRound 是否第一轮
	 * @param stocks 股票列表
	 */
	private static void printPortfolio(boolean firstRound, List<Stock> stocks) {
		// 如果是第一轮，则打印股票信息
		if (firstRound) {
			System.out.printf("%n%-30s\t%12s\t%12s", "STOCK", "INITIALS", "PRICE/SHARE");
			// 遍历股票列表，打印每只股票的信息
			for (Stock stock : stocks) {
				System.out.printf("%n%-30s\t%12s\t%12.2f ------ %12.2f", stock.getStockName(), stock.getStockCode(),
						stock.getStockValue(), stock.getChangeStockValue());
			}
			System.out.println("");
		} else {
			// 打印当天交易结束的提示信息
			System.out.println("\n**********     END OF DAY'S TRADING     **********\n\n");
			// 打印表头
			System.out.printf("%n%-12s\t%-12s\t%-12s\t%-12s\t%-20s", "STOCK", "PRICE/SHARE",
					"HOLDINGS", "VALUE", "NET PRICE CHANGE");
			// 遍历股票列表，打印每只股票的信息
			for (Stock stock : stocks) {
				System.out.printf("%n%-12s\t%-12.2f\t%-12.0f\t%-12.2f\t%-20.2f",
						stock.getStockCode(), stock.getStockValue(), stock.getPortfolioContents(),
						stock.getStockValue() * stock.getPortfolioContents(), stock.getChangeStockValue());
			}
		}
	}

	// 读取股票交易信息
	private static void readStockTransactions(List<Stock> stocks, Scanner scan) {
		// 提示用户输入交易信息
		System.out.println("\n\nWHAT IS YOUR TRANSACTION IN");
		// 遍历股票列表，让用户输入每只股票的交易数量
		for (Stock stock : stocks) {
			System.out.printf("%s? ", stock.getStockCode());

			stock.setTransactionQuantity(readANumber(scan));
		}
	}
# 从 Scanner 对象中读取一个整数
private static int readANumber(Scanner scan) {
    # 初始化选择变量
    int choice = 0;

    # 初始化输入有效性标志
    boolean validInput = false;
    # 循环直到输入有效
    while (!validInput) {
        try:
            # 尝试从输入中读取一个整数
            choice = scan.nextInt();
            # 输入有效，设置标志为真
            validInput = true;
        except (InputMismatchException ex):
            # 捕获输入不是整数的异常，打印错误信息
            System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE");
        finally:
            # 无论是否捕获到异常，都清空输入行
            scan.nextLine();
    }

    # 返回读取到的整数
    return choice;
}

# 调整所有股票的值
private static void adjustAllStockValues(List<Stock> stocks, boolean largeChange1,
			boolean largeChange2,  // 定义一个布尔类型变量 largeChange2
			double marketTrendSlope,  // 定义一个双精度浮点数变量 marketTrendSlope
			Stock stockForLargeChange1, Stock stockForLargeChange2  // 定义两个 Stock 类型的变量 stockForLargeChange1 和 stockForLargeChange2
	) {
		//LOOP THROUGH ALL STOCKS  // 循环遍历所有股票
		for (Stock stock : stocks) {  // 对 stocks 列表中的每个股票进行循环
			double smallChange = random.nextFloat();  // 生成一个随机浮点数作为 smallChange

			if (smallChange <= 0.25) {  // 如果 smallChange 小于等于 0.25
				smallChange = 0.25;  // 将 smallChange 设置为 0.25
			} else if (smallChange <= 0.5) {  // 如果 smallChange 小于等于 0.5
				smallChange = 0.5;  // 将 smallChange 设置为 0.5
			} else if (smallChange <= 0.75) {  // 如果 smallChange 小于等于 0.75
				smallChange = 0.75;  // 将 smallChange 设置为 0.75
			} else {
				smallChange = 0;  // 否则将 smallChange 设置为 0
			}

			//BIG CHANGE CONSTANT:W3  (SET TO ZERO INITIALLY)  // 定义一个名为 bigChange 的常量，并初始化为 0
			var bigChange = 0;  // 声明并初始化 bigChange 变量为 0
			if (largeChange1) { // 如果 largeChange1 为真
				if (stock.getStockCode().equals(stockForLargeChange1.getStockCode())) { // 如果当前股票代码与 largeChange1 对应的股票代码相同
					//ADD 10 PTS. TO THIS STOCK;  RESET E1
					bigChange = 10; // 将 bigChange 设置为 10
				}
			}

			if (largeChange2) { // 如果 largeChange2 为真
				if (stock.getStockCode().equals(stockForLargeChange2.getStockCode())) { // 如果当前股票代码与 largeChange2 对应的股票代码相同
					//SUBTRACT 10 PTS. FROM THIS STOCK;  RESET E2
					bigChange = bigChange - 10; // 从 bigChange 减去 10
				}
			}

			stock.setChangeStockValue(Math.floor(marketTrendSlope * stock.stockValue) + smallChange +
					Math.floor(3 - 6 * random.nextFloat() + .5) + bigChange); // 设置股票变化值
			stock.setChangeStockValue(Math.floor(100 * stock.getChangeStockValue() + .5) / 100d); // 将股票变化值保留两位小数
			stock.stockValue += stock.getChangeStockValue(); // 更新股票价值

			if (stock.stockValue > 0) { // 如果股票价值大于 0
				// 将股票价格取整到小数点后两位
				stock.stockValue = Math.floor(100 * stock.stockValue + 0.5) / 100d;
			} else {
				// 如果股票价格小于等于0，则将变化值设为0，股票价格设为0
				stock.setChangeStockValue(0);
				stock.stockValue = 0;
			}
		}
	}

	private static double randomlyChangeTrendSignAndSlopeAndDuration() {
		// 随机改变趋势符号和斜率（A），以及持续时间
		var newTrend = Math.floor((random.nextFloat() / 10) * 100 + .5) / 100d;
		var slopeSign = random.nextFloat();
		// 如果斜率符号大于0.5，则将趋势取反
		if (slopeSign > 0.5) {
			newTrend = -newTrend;
		}
		return newTrend;
	}

	private static List<Stock> initStocks() {
		// 初始化股票列表
		List<Stock> stocks = new ArrayList<>();
		# 创建一个名为stocks的空列表
		List<Stock> stocks = new ArrayList<>();
		# 向stocks列表中添加一个新的Stock对象，参数为100, "INT. BALLISTIC MISSILES", "IBM"
		stocks.add(new Stock(100, "INT. BALLISTIC MISSILES", "IBM"));
		# 向stocks列表中添加一个新的Stock对象，参数为85, "RED CROSS OF AMERICA", "RCA"
		stocks.add(new Stock(85, "RED CROSS OF AMERICA", "RCA"));
		# 向stocks列表中添加一个新的Stock对象，参数为150, "LICHTENSTEIN, BUMRAP & JOKE", "LBJ"
		stocks.add(new Stock(150, "LICHTENSTEIN, BUMRAP & JOKE", "LBJ"));
		# 向stocks列表中添加一个新的Stock对象，参数为140, "AMERICAN BANKRUPT CO.", "ABC"
		stocks.add(new Stock(140, "AMERICAN BANKRUPT CO.", "ABC"));
		# 向stocks列表中添加一个新的Stock对象，参数为110, "CENSURED BOOKS STORE", "CBS"
		stocks.add(new Stock(110, "CENSURED BOOKS STORE", "CBS"));
		# 返回stocks列表
		return stocks;
	}

	# 定义一个名为printGameHelp的静态方法，参数为Scanner对象scan
	private static void printGameHelp(Scanner scan) {
		# 打印提示信息，要求用户输入指令
		System.out.print("DO YOU WANT THE INSTRUCTIONS (YES-TYPE 1, NO-TYPE 0) ? ");
		# 从用户输入中获取整数值
		int choice = scan.nextInt();
		# 如果用户输入的值大于等于1
		if (choice >= 1) {
			# 打印游戏说明
			System.out.println("");
			System.out.println("THIS PROGRAM PLAYS THE STOCK MARKET.  YOU WILL BE GIVEN");
			System.out.println("$10,000 AND MAY BUY OR SELL STOCKS.  THE STOCK PRICES WILL");
			System.out.println("BE GENERATED RANDOMLY AND THEREFORE THIS MODEL DOES NOT");
			System.out.println("REPRESENT EXACTLY WHAT HAPPENS ON THE EXCHANGE.  A TABLE");
			System.out.println("OF AVAILABLE STOCKS, THEIR PRICES, AND THE NUMBER OF SHARES");
			System.out.println("IN YOUR PORTFOLIO WILL BE PRINTED.  FOLLOWING THIS, THE");
			System.out.println("INITIALS OF EACH STOCK WILL BE PRINTED WITH A QUESTION");
			// 打印交易指示信息
			System.out.println("MARK.  HERE YOU INDICATE A TRANSACTION.  TO BUY A STOCK");
			System.out.println("TYPE +NNN, TO SELL A STOCK TYPE -NNN, WHERE NNN IS THE");
			System.out.println("NUMBER OF SHARES.  A BROKERAGE FEE OF 1% WILL BE CHARGED");
			System.out.println("ON ALL TRANSACTIONS.  NOTE THAT IF A STOCK'S VALUE DROPS");
			System.out.println("TO ZERO IT MAY REBOUND TO A POSITIVE VALUE AGAIN.  YOU");
			System.out.println("HAVE $10,000 TO INVEST.  USE INTEGERS FOR ALL YOUR INPUTS.");
			System.out.println("(NOTE:  TO GET A 'FEEL' FOR THE MARKET RUN FOR AT LEAST");
			System.out.println("10 DAYS)");
			System.out.println("-----GOOD LUCK!-----");
		}
		// 打印空行
		System.out.println("\n\n");
	}

	// 打印游戏介绍
	private static void printIntro() {
		System.out.println("                                STOCK MARKET");
		System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
		System.out.println("\n\n");
	}

	/**
	 * Stock class also storing the stock information and other related information for simplicity
	 * Stock类还存储股票信息和其他相关信息，以便简化处理
	 */
	private static class Stock {

		private final String stockName;  // 股票名称
		private final String stockCode;  // 股票代码
		private double stockValue;  // 股票价值
		private double portfolioContents = 0;  // 投资组合内容
		private double transactionQuantity = 0;  // 交易数量
		private double changeStockValue = 0;  // 股票价值变化

		public Stock(double stockValue, String stockName, String stockCode) {
			this.stockValue = stockValue;  // 初始化股票价值
			this.stockName = stockName;  // 初始化股票名称
			this.stockCode = stockCode;  // 初始化股票代码
		}

		public String getStockName() {
			return stockName;  // 返回股票名称
		}
		// 返回股票代码
		public String getStockCode() {
			return stockCode;
		}

		// 返回股票价值
		public double getStockValue() {
			return stockValue;
		}

		// 返回投资组合内容
		public double getPortfolioContents() {
			return portfolioContents;
		}

		// 设置投资组合内容
		public void setPortfolioContents(double portfolioContents) {
			this.portfolioContents = portfolioContents;
		}

		// 返回交易数量
		public double getTransactionQuantity() {
			return transactionQuantity;
		}
		// 设置交易数量
		public void setTransactionQuantity(double transactionQuantity) {
			this.transactionQuantity = transactionQuantity;
		}

		// 获取股票变动价值
		public double getChangeStockValue() {
			return changeStockValue;
		}

		// 设置股票变动价值
		public void setChangeStockValue(double changeStockValue) {
			this.changeStockValue = changeStockValue;
		}

		// 重写 toString 方法，返回股票信息的字符串表示
		@Override
		public String toString() {
			return "Stock{" +
					"stockValue=" + stockValue +
					", stockCode='" + stockCode + '\'' +
					", portfolioContents=" + portfolioContents +
					", transactionQuantity=" + transactionQuantity +
```
```


# 定义一个名为 Stock 的类，用于表示股票
public class Stock {
    # 定义私有属性 symbol，用于存储股票的代码
    private String symbol;
    # 定义私有属性 name，用于存储股票的名称
    private String name;
    # 定义私有属性 price，用于存储股票的价格
    private double price;
    
    # 定义构造函数，用于初始化 Stock 对象
    public Stock(String symbol, String name, double price) {
        this.symbol = symbol;
        this.name = name;
        this.price = price;
    }
    
    # 定义一个方法，用于获取股票代码
    public String getSymbol() {
        return symbol;
    }
    
    # 定义一个方法，用于获取股票名称
    public String getName() {
        return name;
    }
    
    # 定义一个方法，用于获取股票价格
    public double getPrice() {
        return price;
    }
    
    # 定义一个方法，用于设置股票价格
    public void setPrice(double price) {
        this.price = price;
    }
    
    # 重写 toString 方法，用于返回 Stock 对象的字符串表示
    @Override
    public String toString() {
        return "Stock{" +
                "symbol='" + symbol + '\'' +
                ", name='" + name + '\'' +
                ", price=" + price +
                '}';
    }
}
```