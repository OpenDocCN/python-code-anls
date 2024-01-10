# `basic-computer-games\83_Stock_Market\python\Stock_Market.py`

```
import random  # 导入 random 模块
from typing import Any, Dict, List  # 导入类型提示模块


class Stock_Market:  # 定义 Stock_Market 类
    def __init__(self) -> None:  # 初始化方法
        # Hard Coded Names
        short_names = ["IBM", "RCA", "LBJ", "ABC", "CBS"]  # 股票简称列表
        full_names = [  # 股票全称列表
            "INT. BALLISTIC MISSLES",
            "RED CROSS OF AMERICA",
            "LICHTENSTEIN, BUMRAP & JOKE",
            "AMERICAN BANKRUPT CO.",
            "CENSURED BOOKS STORE",
        ]

        # Initializing Dictionary to hold all the information systematically
        self.data: Dict[str, Any] = {}  # 初始化存储股票信息的字典
        for sn, fn in zip(short_names, full_names):  # 遍历简称和全称列表
            # A dictionary for each stock
            temp = {"Name": fn, "Price": None, "Holdings": 0}  # 每支股票的信息字典
            # Nested outer dictionary for all stocks
            self.data[sn] = temp  # 将每支股票的信息字典添加到总字典中

        # Initializing Randomly generated initial prices
        for stock in self.data.values():  # 遍历股票信息字典的值
            stock["Price"] = round(random.uniform(80, 120), 2)  # 生成随机初始价格，范围在 80 到 120 之间

        # Initialize Assets
        self.cash_assets = 10000  # 初始化现金资产
        self.stock_assets = 0  # 初始化股票资产

    def total_assets(self) -> float:  # 计算总资产的方法
        return self.cash_assets + self.stock_assets  # 返回现金资产和股票资产之和

    def _generate_day_change(self) -> None:  # 生成每日变化的方法
        self.changes = []  # 初始化变化列表
        for _ in range(len(self.data)):  # 循环股票数量次数
            self.changes.append(
                round(random.uniform(-5, 5), 2)
            )  # 生成随机变化百分比，范围在 -5 到 5 之间

    def update_prices(self) -> None:  # 更新股票价格的方法
        self._generate_day_change()  # 调用生成每日变化的方法
        for stock, change in zip(self.data.values(), self.changes):  # 遍历股票信息字典的值和变化列表
            stock["Price"] = round(stock["Price"] + (change / 100) * stock["Price", 2)  # 更新股票价格

    def print_exchange_average(self) -> None:  # 打印交易所平均价格的方法
        sum = 0  # 初始化总和
        for stock in self.data.values():  # 遍历股票信息字典的值
            sum += stock["Price"]  # 累加股票价格

        print(f"\nNEW YORK STOCK EXCHANGE AVERAGE: ${sum / 5:.2f}")  # 打印纽约证券交易所平均价格

    def get_average_change(self) -> float:  # 获取平均变化的方法
        sum: float = 0  # 初始化总和
        for change in self.changes:  # 遍历变化列表
            sum += change  # 累加变化值

        return round(sum / 5, 2)  # 返回平均变化值
    # 打印第一天的股票信息
    def print_first_day(self) -> None:
        # 打印表头
        print("\nSTOCK\t\t\t\t\tINITIALS\tPRICE/SHARE($)")
        # 遍历股票数据字典
        for stock, data in self.data.items():
            # 如果股票不是"LBJ"，按照格式打印股票信息
            if stock != "LBJ":
                print("{}\t\t\t{}\t\t{}".format(data["Name"], stock, data["Price"]))
            # 如果股票是"LBJ"，按照格式打印股票信息
            else:
                print("{}\t\t{}\t\t{}".format(data["Name"], stock, data["Price"]))

        # 调用打印交易平均值的方法
        self.print_exchange_average()
        # 调用打印资产的方法
        self.print_assets()

    # 获取用户输入
    def take_inputs(self) -> List[str]:
        # 打印提示信息
        print("\nWHAT IS YOUR TRANSACTION IN")
        flag = False
        # 循环直到用户输入有效数据
        while not flag:
            new_holdings = []
            # 遍历股票数据字典的键
            for stock in self.data.keys():
                try:
                    # 获取用户输入的股票持有量
                    new_holdings.append(int(input(f"{stock}? ")))
                except Exception:
                    # 如果输入无效，打印提示信息并跳出循环
                    print("\nINVALID ENTRY, TRY AGAIN\n")
                    break
            # 如果用户输入了所有股票的持有量，调用检查交易的方法
            if len(new_holdings) == 5:
                flag = self._check_transaction(new_holdings)

        # 返回用户输入的股票持有量列表
        return new_holdings  # type: ignore

    # 打印交易日的股票信息
    def print_trading_day(self) -> None:
        # 打印表头
        print("STOCK\tPRICE/SHARE\tHOLDINGS\tNET. Value\tPRICE CHANGE")
        # 遍历股票数据字典的键、值和价格变化列表
        for stock, data, change in zip(
            self.data.keys(), self.data.values(), self.changes
        ):
            # 计算股票持有量的价值
            value = data["Price"] * data["Holdings"]
            # 按照格式打印股票信息
            print(
                "{}\t{}\t\t{}\t\t{:.2f}\t\t{}".format(
                    stock, data["Price"], data["Holdings"], value, change
                )
            )

    # 更新现金资产
    def update_cash_assets(self, new_holdings) -> None:
        sell = 0
        buy = 0
        # 遍历股票数据字典的值和用户输入的股票持有量
        for stock, holding in zip(self.data.values(), new_holdings):
            # 如果持有量大于0，计算买入的价值
            if holding > 0:
                buy += stock["Price"] * holding
            # 如果持有量小于0，计算卖出的价值
            elif holding < 0:
                sell += stock["Price"] * abs(holding)

        # 更新现金资产
        self.cash_assets = self.cash_assets + sell - buy
    # 更新股票资产
    def update_stock_assets(self) -> None:
        # 初始化资产总额
        sum = 0
        # 遍历数据字典中的每一项
        for data in self.data.values():
            # 计算每支股票的市值并累加到总额中
            sum += data["Price"] * data["Holdings"]

        # 更新股票资产总额
        self.stock_assets = round(sum, 2)

    # 打印资产信息
    def print_assets(self) -> None:
        # 打印股票资产总额
        print(f"\nTOTAL STOCK ASSETS ARE: ${self.stock_assets:.2f}")
        # 打印现金资产总额
        print(f"TOTAL CASH ASSETS ARE: ${self.cash_assets:.2f}")
        # 打印总资产总额
        print(f"TOTAL ASSETS ARE: ${self.total_assets():.2f}")

    # 检查交易是否合法
    def _check_transaction(self, new_holdings) -> bool:
        # 初始化交易总额
        sum = 0
        # 遍历数据字典中的每一项和新持股数量
        for stock, holding in zip(self.data.values(), new_holdings):
            # 如果持股数量大于0，计算交易总额
            if holding > 0:
                sum += stock["Price"] * holding
            # 如果持股数量小于0，检查是否卖出超过持有数量的股票
            elif holding < 0:
                if abs(holding) > stock["Holdings"]:
                    print("\nYOU HAVE OVERSOLD SOME STOCKS, TRY AGAIN\n")
                    return False

        # 检查交易总额是否超过现金资产总额
        if sum > self.cash_assets:
            print(
                "\nYOU HAVE USED ${:.2f} MORE THAN YOU HAVE, TRY AGAIN\n".format(
                    sum - self.cash_assets
                )
            )
            return False

        return True

    # 更新持股数量
    def update_holdings(self, new_holdings) -> None:
        # 遍历数据字典中的每一项和新持股数量，更新持股数量
        for stock, new_holding in zip(self.data.values(), new_holdings):
            stock["Holdings"] += new_holding
# 定义打印说明的函数，没有返回值
def print_instruction() -> None:

    # 打印游戏说明
    print(
        """
THIS PROGRAM PLAYS THE STOCK MARKET.  YOU WILL BE GIVEN
$10,000 AND MAY BUY OR SELL STOCKS.  THE STOCK PRICES WILL
BE GENERATED RANDOMLY AND THEREFORE THIS MODEL DOES NOT
REPRESENT EXACTLY WHAT HAPPENS ON THE EXCHANGE.  A TABLE
OF AVAILABLE STOCKS, THEIR PRICES, AND THE NUMBER OF SHARES
IN YOUR PORTFOLIO WILL BE PRINTED.  FOLLOWING THIS, THE
INITIALS OF EACH STOCK WILL BE PRINTED WITH A QUESTION
MARK.  HERE YOU INDICATE A TRANSACTION.  TO BUY A STOCK
TYPE +NNN, TO SELL A STOCK TYPE -NNN, WHERE NNN IS THE
NUMBER OF SHARES.  A BROKERAGE FEE OF 1% WILL BE CHARGED
ON ALL TRANSACTIONS.  NOTE THAT IF A STOCK'S VALUE DROPS
TO ZERO IT MAY REBOUND TO A POSITIVE VALUE AGAIN.  YOU
HAVE $10,000 TO INVEST.  USE INTEGERS FOR ALL YOUR INPUTS.
(NOTE:  TO GET A 'FEEL' FOR THE MARKET RUN FOR AT LEAST
10 DAYS)
          ------------GOOD LUCK!------------\n
    """
    )


# 定义主函数，没有返回值
def main() -> None:
    # 打印标题
    print("\t\t      STOCK MARKET")
    # 询问是否需要打印游戏说明
    help = input("\nDO YOU WANT INSTRUCTIONS(YES OR NO)? ")

    # 如果需要打印游戏说明
    if help.lower() == "yes":
        # 调用打印说明的函数
        print_instruction()

    # 初始化游戏
    Game = Stock_Market()

    # 进行第一天交易
    Game.print_first_day()
    new_holdings = Game.take_inputs()
    Game.update_holdings(new_holdings)
    Game.update_cash_assets(new_holdings)
    print("\n------------END OF TRADING DAY--------------\n")

    response = 1
    while response == 1:

        # 模拟一天的交易
        Game.update_prices()
        Game.print_trading_day()
        Game.print_exchange_average()
        Game.update_stock_assets()
        Game.print_assets()

        # 询问是否继续游戏
        response = int(input("\nDO YOU WISH TO CONTINUE (YES-TYPE 1, NO-TYPE 0)? "))
        if response == 0:
            break

        new_holdings = Game.take_inputs()
        Game.update_holdings(new_holdings)
        Game.update_cash_assets(new_holdings)
        print("\n------------END OF TRADING DAY--------------\n")

    print("\nHOPE YOU HAD FUN!!!!")
    # 从用户输入中获取数据
    input()
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```