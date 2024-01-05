# `d:/src/tocomm/basic-computer-games\83_Stock_Market\python\Stock_Market.py`

```
import random  # 导入 random 模块
from typing import Any, Dict, List  # 从 typing 模块导入 Any, Dict, List 类型

class Stock_Market:  # 定义 Stock_Market 类
    def __init__(self) -> None:  # 初始化方法
        # Hard Coded Names
        short_names = ["IBM", "RCA", "LBJ", "ABC", "CBS"]  # 股票的简称列表
        full_names = [  # 股票的全称列表
            "INT. BALLISTIC MISSLES",
            "RED CROSS OF AMERICA",
            "LICHTENSTEIN, BUMRAP & JOKE",
            "AMERICAN BANKRUPT CO.",
            "CENSURED BOOKS STORE",
        ]

        # Initializing Dictionary to hold all the information systematically
        self.data: Dict[str, Any] = {}  # 初始化一个字典来存储所有信息
        for sn, fn in zip(short_names, full_names):  # 遍历简称和全称列表
            # A dictionary for each stock
            temp = {"Name": fn, "Price": None, "Holdings": 0}
            # 创建一个临时字典，包含股票名称、价格和持有数量，用于存储每支股票的信息
            self.data[sn] = temp
            # 将临时字典存入外部字典中，以股票名称作为键

        # 初始化随机生成的初始股票价格
        for stock in self.data.values():
            stock["Price"] = round(random.uniform(80, 120), 2)  # 价格在60到120之间随机生成

        # 初始化资产
        self.cash_assets = 10000
        self.stock_assets = 0

    def total_assets(self) -> float:
        # 返回现金资产和股票资产的总和
        return self.cash_assets + self.stock_assets

    def _generate_day_change(self) -> None:
        self.changes = []
        for _ in range(len(self.data)):
            # 生成每支股票当日价格变动
            self.changes.append(
                round(random.uniform(-5, 5), 2)
            )  # Random % Change b/w -5 and 5  # 生成一个随机的百分比变化，范围在-5和5之间

    def update_prices(self) -> None:  # 更新股票价格
        self._generate_day_change()  # 调用_generate_day_change方法生成当天的价格变化
        for stock, change in zip(self.data.values(), self.changes):  # 遍历股票数据和价格变化
            stock["Price"] = round(stock["Price"] + (change / 100) * stock["Price"], 2)  # 根据价格变化更新股票价格

    def print_exchange_average(self) -> None:  # 打印交易所平均价格
        sum = 0  # 初始化总和
        for stock in self.data.values():  # 遍历股票数据
            sum += stock["Price"]  # 计算总价格

        print(f"\nNEW YORK STOCK EXCHANGE AVERAGE: ${sum / 5:.2f}")  # 打印纽约证券交易所平均价格

    def get_average_change(self) -> float:  # 获取平均变化率
        sum: float = 0  # 初始化总和
        for change in self.changes:  # 遍历价格变化
            sum += change  # 计算总变化率
        return round(sum / 5, 2)  # 返回一个数值的四舍五入结果，保留两位小数

    def print_first_day(self) -> None:  # 定义一个方法，打印第一天的股票交易数据

        print("\nSTOCK\t\t\t\t\tINITIALS\tPRICE/SHARE($)")  # 打印表头
        for stock, data in self.data.items():  # 遍历股票数据字典
            if stock != "LBJ":  # 如果股票不是"LBJ"
                print("{}\t\t\t{}\t\t{}".format(data["Name"], stock, data["Price"]))  # 打印股票名称、股票代码、股价
            else:  # 如果股票是"LBJ"
                print("{}\t\t{}\t\t{}".format(data["Name"], stock, data["Price"]))  # 打印股票名称、股票代码、股价

        self.print_exchange_average()  # 调用打印交易平均值的方法
        self.print_assets()  # 调用打印资产的方法

    def take_inputs(self) -> List[str]:  # 定义一个方法，获取用户输入的交易信息，并返回一个字符串列表
        print("\nWHAT IS YOUR TRANSACTION IN")  # 打印提示信息
        flag = False  # 初始化标志变量为False
        while not flag:  # 当标志变量为False时循环执行以下操作
            new_holdings = []  # 初始化一个空列表
            for stock in self.data.keys():  # 遍历股票数据字典的键
                try:
                    new_holdings.append(int(input(f"{stock}? ")))  # 尝试从用户输入中获取整数并添加到new_holdings列表中
                except Exception:  # 捕获任何异常
                    print("\nINVALID ENTRY, TRY AGAIN\n")  # 打印错误消息
                    break  # 跳出循环
            if len(new_holdings) == 5:  # 如果new_holdings列表长度为5
                flag = self._check_transaction(new_holdings)  # 调用_check_transaction方法检查交易

        return new_holdings  # type: ignore  # 返回new_holdings列表，忽略类型检查

    def print_trading_day(self) -> None:  # 定义一个没有返回值的print_trading_day方法

        print("STOCK\tPRICE/SHARE\tHOLDINGS\tNET. Value\tPRICE CHANGE")  # 打印表头
        for stock, data, change in zip(  # 遍历self.data中的股票、数据和变化
            self.data.keys(), self.data.values(), self.changes
        ):
            value = data["Price"] * data["Holdings"]  # 计算股票价值
            print(  # 打印股票信息
                "{}\t{}\t\t{}\t\t{:.2f}\t\t{}".format(
                    stock, data["Price"], data["Holdings"], value, change
                )
    def update_cash_assets(self, new_holdings) -> None:  # 定义一个方法用于更新现金资产，参数为新的持有量
        sell = 0  # 初始化卖出金额为0
        buy = 0  # 初始化买入金额为0
        for stock, holding in zip(self.data.values(), new_holdings):  # 遍历股票数据和新的持有量
            if holding > 0:  # 如果持有量大于0
                buy += stock["Price"] * holding  # 计算买入金额

            elif holding < 0:  # 如果持有量小于0
                sell += stock["Price"] * abs(holding)  # 计算卖出金额

        self.cash_assets = self.cash_assets + sell - buy  # 更新现金资产

    def update_stock_assets(self) -> None:  # 定义一个方法用于更新股票资产
        sum = 0  # 初始化总价值为0
        for data in self.data.values():  # 遍历股票数据
            sum += data["Price"] * data["Holdings"]  # 计算总价值
        self.stock_assets = round(sum, 2)  # 将计算得到的股票资产总额保留两位小数并赋值给self.stock_assets

    def print_assets(self) -> None:  # 定义一个打印资产信息的方法
        print(f"\nTOTAL STOCK ASSETS ARE: ${self.stock_assets:.2f}")  # 打印股票资产总额
        print(f"TOTAL CASH ASSETS ARE: ${self.cash_assets:.2f}")  # 打印现金资产总额
        print(f"TOTAL ASSETS ARE: ${self.total_assets():.2f}")  # 打印总资产总额

    def _check_transaction(self, new_holdings) -> bool:  # 定义一个检查交易的私有方法，接受新的持股信息作为参数，返回布尔值
        sum = 0  # 初始化一个变量用于计算总价值
        for stock, holding in zip(self.data.values(), new_holdings):  # 遍历当前持股信息和新的持股信息
            if holding > 0:  # 如果新的持股数量大于0
                sum += stock["Price"] * holding  # 计算该股票的总价值并加到sum上

            elif holding < 0:  # 如果新的持股数量小于0
                if abs(holding) > stock["Holdings"]:  # 如果卖出的数量大于当前持股数量
                    print("\nYOU HAVE OVERSOLD SOME STOCKS, TRY AGAIN\n")  # 打印错误信息
                    return False  # 返回False表示交易不合法

        if sum > self.cash_assets:  # 如果计算得到的总价值大于现金资产总额
            print(  # 打印
                "\nYOU HAVE USED ${:.2f} MORE THAN YOU HAVE, TRY AGAIN\n".format(
                    sum - self.cash_assets
                )
            )
            return False
```
这段代码是一个条件语句，如果用户使用的金额超过了他们拥有的金额，就会返回False。

```
        return True
```
这段代码是一个条件语句，如果用户使用的金额没有超过他们拥有的金额，就会返回True。

```
    def update_holdings(self, new_holdings) -> None:
        for stock, new_holding in zip(self.data.values(), new_holdings):
            stock["Holdings"] += new_holding
```
这段代码定义了一个函数，用于更新用户的持股情况。

```
def print_instruction() -> None:

    print(
        """
THIS PROGRAM PLAYS THE STOCK MARKET.  YOU WILL BE GIVEN
$10,000 AND MAY BUY OR SELL STOCKS.  THE STOCK PRICES WILL
BE GENERATED RANDOMLY AND THEREFORE THIS MODEL DOES NOT
```
这段代码定义了一个函数，用于打印程序的使用说明。
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制数据，并封装成字节流
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
    # 如果用户输入的帮助选项是“yes”，则打印游戏说明
    if help.lower() == "yes":
        print_instruction()

    # 初始化股票市场游戏
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
        Game.print_trading_day()  # 调用Game对象的print_trading_day方法，打印交易日信息
        Game.print_exchange_average()  # 调用Game对象的print_exchange_average方法，打印交易所平均值信息
        Game.update_stock_assets()  # 调用Game对象的update_stock_assets方法，更新股票资产信息
        Game.print_assets()  # 调用Game对象的print_assets方法，打印资产信息

        response = int(input("\nDO YOU WISH TO CONTINUE (YES-TYPE 1, NO-TYPE 0)? "))  # 获取用户输入的是否继续交易的信息
        if response == 0:  # 如果用户输入为0
            break  # 跳出循环，结束交易

        new_holdings = Game.take_inputs()  # 调用Game对象的take_inputs方法，获取新的持仓信息
        Game.update_holdings(new_holdings)  # 调用Game对象的update_holdings方法，更新持仓信息
        Game.update_cash_assets(new_holdings)  # 调用Game对象的update_cash_assets方法，更新现金资产信息
        print("\n------------END OF TRADING DAY--------------\n")  # 打印交易日结束信息

    print("\nHOPE YOU HAD FUN!!!!")  # 打印祝福信息
    input()  # 等待用户输入


if __name__ == "__main__":  # 如果当前文件作为主程序运行
    main()  # 调用main函数，开始执行程序
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```