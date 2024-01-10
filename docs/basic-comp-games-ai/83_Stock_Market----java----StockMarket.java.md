# `basic-computer-games\83_Stock_Market\java\StockMarket.java`

```
# 导入所需的 Java 类
import java.util.ArrayList;
import java.util.InputMismatchException;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

/**
 * 股票市场模拟
 *
 * 一些原始程序变量的文档和它们在此程序中的等价物：
 * A-MRKT TRND SLP;             marketTrendSlope
 * B5-BRKRGE FEE;               brokerageFee
 * C-TTL CSH ASSTS;             cashAssets
 * C5-TTL CSH ASSTS (TEMP);     tmpCashAssets
 * C(I)-CHNG IN STK VAL;        changeStockValue
 * D-TTL ASSTS;                 assets
 * E1,E2-LRG CHNG MISC;         largeChange1, largeChange2
 * I1,I2-STCKS W LRG CHNG;      randomStockIndex1, randomStockIndex2
 * N1,N2-LRG CHNG DAY CNTS;     largeChangeNumberDays1, largeChangeNumberDays2
 * P5-TTL DAYS PRCHSS;          totalDaysPurchases
 * P(I)-PRTFL CNTNTS;           portfolioContents
 * Q9-NEW CYCL?;                newCycle
 * S4-SGN OF A;                 slopeSign
 * S5-TTL DYS SLS;              totalDaysSales
 * S(I)-VALUE/SHR;              stockValue
 * T-TTL STCK ASSTS;            totalStockAssets
 * T5-TTL VAL OF TRNSCTNS;      totalValueOfTransactions
 * W3-LRG CHNG;                 bigChange
 * X1-SMLL CHNG(<$1);           smallChange
 * Z4,Z5,Z6-NYSE AVE.;          tmpNyseAverage, nyseAverage, nyseAverageChange
 * Z(I)-TRNSCT                  transactionQuantity
 *
 * 新价格 = 旧价格 + (趋势 x 旧价格) + (小的随机价格变动) + (可能的大价格变动)
 *
 * 由 Aldrin Misquitta (@aldrinm) 从 BASIC 转换为 Java
 */
public class StockMarket {

    private static final Random random = new Random();

    }

    /**
     * 在 lowerBound（包括）和 upperBound（不包括）之间生成随机整数
     */
    private static int randomNumber(int lowerBound, int upperBound) {
        return random.nextInt((upperBound - lowerBound)) + lowerBound;
    }

    # 生成新的市场趋势斜率
    private static double newMarketTrendSlope() {
        return randomlyChangeTrendSignAndSlopeAndDuration();
    }
    // 打印投资组合信息，包括股票名称、股票代码、股价/股份
    private static void printPortfolio(boolean firstRound, List<Stock> stocks) {
        // 如果是第一轮交易
        if (firstRound) {
            // 打印表头
            System.out.printf("%n%-30s\t%12s\t%12s", "STOCK", "INITIALS", "PRICE/SHARE");
            // 遍历股票列表，打印每只股票的信息
            for (Stock stock : stocks) {
                System.out.printf("%n%-30s\t%12s\t%12.2f ------ %12.2f", stock.getStockName(), stock.getStockCode(),
                        stock.getStockValue(), stock.getChangeStockValue());
            }
            System.out.println("");
        } else {
            // 如果不是第一轮交易，打印交易结束信息
            System.out.println("\n**********     END OF DAY'S TRADING     **********\n\n");
            // 打印持仓股票的信息，包括股票代码、股价、持仓数量、价值、价格变动
            System.out.printf("%n%-12s\t%-12s\t%-12s\t%-12s\t%-20s", "STOCK", "PRICE/SHARE",
                    "HOLDINGS", "VALUE", "NET PRICE CHANGE");
            for (Stock stock : stocks) {
                System.out.printf("%n%-12s\t%-12.2f\t%-12.0f\t%-12.2f\t%-20.2f",
                        stock.getStockCode(), stock.getStockValue(), stock.getPortfolioContents(),
                        stock.getStockValue() * stock.getPortfolioContents(), stock.getChangeStockValue());
            }
        }
    }

    // 读取股票交易信息
    private static void readStockTransactions(List<Stock> stocks, Scanner scan) {
        System.out.println("\n\nWHAT IS YOUR TRANSACTION IN");
        // 遍历股票列表，询问用户每只股票的交易数量
        for (Stock stock : stocks) {
            System.out.printf("%s? ", stock.getStockCode());

            stock.setTransactionQuantity(readANumber(scan));
        }
    }

    // 读取一个数字
    private static int readANumber(Scanner scan) {
        int choice = 0;

        boolean validInput = false;
        // 循环直到输入合法的数字
        while (!validInput) {
            try {
                choice = scan.nextInt();
                validInput = true;
            } catch (InputMismatchException ex) {
                System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE");
            } finally {
                scan.nextLine();
            }
        }

        return choice;
    }
    // 调整所有股票的价值
    private static void adjustAllStockValues(List<Stock> stocks, boolean largeChange1,
            boolean largeChange2,
            double marketTrendSlope,
            Stock stockForLargeChange1, Stock stockForLargeChange2
    ) {
        // 遍历所有股票
        for (Stock stock : stocks) {
            // 生成一个小的变化值
            double smallChange = random.nextFloat();

            // 根据小的变化值进行不同的赋值
            if (smallChange <= 0.25) {
                smallChange = 0.25;
            } else if (smallChange <= 0.5) {
                smallChange = 0.5;
            } else if (smallChange <= 0.75) {
                smallChange = 0.75;
            } else {
                smallChange = 0;
            }

            // 设置大的变化值为0
            var bigChange = 0;
            // 如果有大的变化1
            if (largeChange1) {
                // 如果股票代码与大的变化1的股票代码相同
                if (stock.getStockCode().equals(stockForLargeChange1.getStockCode())) {
                    // 增加10点到这个股票的价值，重置E1
                    bigChange = 10;
                }
            }

            // 如果有大的变化2
            if (largeChange2) {
                // 如果股票代码与大的变化2的股票代码相同
                if (stock.getStockCode().equals(stockForLargeChange2.getStockCode())) {
                    // 从这个股票减去10点，重置E2
                    bigChange = bigChange - 10;
                }
            }

            // 计算股票的变化值
            stock.setChangeStockValue(Math.floor(marketTrendSlope * stock.stockValue) + smallChange +
                    Math.floor(3 - 6 * random.nextFloat() + .5) + bigChange);
            // 将股票的变化值保留两位小数
            stock.setChangeStockValue(Math.floor(100 * stock.getChangeStockValue() + .5) / 100d);
            // 更新股票的价值
            stock.stockValue += stock.getChangeStockValue();

            // 如果股票价值大于0
            if (stock.stockValue > 0) {
                // 将股票价值保留两位小数
                stock.stockValue = Math.floor(100 * stock.stockValue + 0.5) / 100d;
            } else {
                // 如果股票价值小于等于0，重置变化值和股票价值为0
                stock.setChangeStockValue(0);
                stock.stockValue = 0;
            }
        }
    }
    // 随机改变趋势符号、斜率和持续时间
    private static double randomlyChangeTrendSignAndSlopeAndDuration() {
        // 随机生成新的趋势值
        var newTrend = Math.floor((random.nextFloat() / 10) * 100 + .5) / 100d;
        // 随机生成斜率符号
        var slopeSign = random.nextFloat();
        // 如果斜率符号大于0.5，则将新趋势值取反
        if (slopeSign > 0.5) {
            newTrend = -newTrend;
        }
        // 返回新的趋势值
        return newTrend;
    }

    // 初始化股票列表
    private static List<Stock> initStocks() {
        // 创建股票列表
        List<Stock> stocks = new ArrayList<>();
        // 添加股票对象到列表中
        stocks.add(new Stock(100, "INT. BALLISTIC MISSILES", "IBM"));
        stocks.add(new Stock(85, "RED CROSS OF AMERICA", "RCA"));
        stocks.add(new Stock(150, "LICHTENSTEIN, BUMRAP & JOKE", "LBJ"));
        stocks.add(new Stock(140, "AMERICAN BANKRUPT CO.", "ABC"));
        stocks.add(new Stock(110, "CENSURED BOOKS STORE", "CBS"));
        // 返回股票列表
        return stocks;
    }
    // 打印游戏帮助信息
    private static void printGameHelp(Scanner scan) {
        // 提示用户是否需要游戏说明
        System.out.print("DO YOU WANT THE INSTRUCTIONS (YES-TYPE 1, NO-TYPE 0) ? ");
        // 获取用户选择
        int choice = scan.nextInt();
        // 如果用户选择大于等于1
        if (choice >= 1) {
            // 打印游戏说明
            System.out.println("");
            System.out.println("THIS PROGRAM PLAYS THE STOCK MARKET.  YOU WILL BE GIVEN");
            System.out.println("$10,000 AND MAY BUY OR SELL STOCKS.  THE STOCK PRICES WILL");
            System.out.println("BE GENERATED RANDOMLY AND THEREFORE THIS MODEL DOES NOT");
            System.out.println("REPRESENT EXACTLY WHAT HAPPENS ON THE EXCHANGE.  A TABLE");
            System.out.println("OF AVAILABLE STOCKS, THEIR PRICES, AND THE NUMBER OF SHARES");
            System.out.println("IN YOUR PORTFOLIO WILL BE PRINTED.  FOLLOWING THIS, THE");
            System.out.println("INITIALS OF EACH STOCK WILL BE PRINTED WITH A QUESTION");
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
     */
    private static class Stock {

        // 股票名称
        private final String stockName;
        // 股票代码
        private final String stockCode;
        // 股票价值
        private double stockValue;
        // 投资组合内容
        private double portfolioContents = 0;
        // 交易数量
        private double transactionQuantity = 0;
        // 股票价值变化
        private double changeStockValue = 0;

        // 构造函数，初始化股票价值、名称和代码
        public Stock(double stockValue, String stockName, String stockCode) {
            this.stockValue = stockValue;
            this.stockName = stockName;
            this.stockCode = stockCode;
        }

        // 获取股票名称
        public String getStockName() {
            return stockName;
        }

        // 获取股票代码
        public String getStockCode() {
            return stockCode;
        }

        // 获取股票价值
        public double getStockValue() {
            return stockValue;
        }

        // 获取投资组合内容
        public double getPortfolioContents() {
            return portfolioContents;
        }

        // 设置投资组合内容
        public void setPortfolioContents(double portfolioContents) {
            this.portfolioContents = portfolioContents;
        }

        // 获取交易数量
        public double getTransactionQuantity() {
            return transactionQuantity;
        }

        // 设置交易数量
        public void setTransactionQuantity(double transactionQuantity) {
            this.transactionQuantity = transactionQuantity;
        }

        // 获取股票价值变化
        public double getChangeStockValue() {
            return changeStockValue;
        }

        // 设置股票价值变化
        public void setChangeStockValue(double changeStockValue) {
            this.changeStockValue = changeStockValue;
        }

        // 重写 toString 方法，返回股票信息
        @Override
        public String toString() {
            return "Stock{" +
                    "stockValue=" + stockValue +
                    ", stockCode='" + stockCode + '\'' +
                    ", portfolioContents=" + portfolioContents +
                    ", transactionQuantity=" + transactionQuantity +
                    ", changeStockValue=" + changeStockValue +
                    '}';
        }
    }
# 闭合前面的函数定义
```