# `83_Stock_Market\csharp\Controller.cs`

```
// 引入必要的命名空间
using System;
using System.Collections.Generic;
using System.Linq;

namespace Game
{
    // 创建静态类 Controller
    public static class Controller
    {
        /// <summary>
        /// 管理与用户的初始交互
        /// </summary>
        public static void StartGame()
        {
            // 显示游戏横幅
            View.ShowBanner();

            // 获取用户是否要显示游戏说明
            var showInstructions = GetYesOrNo(View.PromptShowInstructions);
            // 显示分隔线
            View.ShowSeparator();
            // 如果用户选择显示游戏说明
            if (showInstructions)
                // 显示游戏说明
                View.ShowInstructions();
            View.ShowSeparator();
        }
        // 显示分隔符

        /// <summary>
        /// 从用户获取是或否的答案。
        /// </summary>
        /// <param name="prompt">
        /// 显示提示信息。
        /// </param>
        /// <returns>
        /// 如果用户回答是，则返回 true，如果回答否，则返回 false。
        /// </returns>
        public static bool GetYesOrNo(Action prompt)
        {
            prompt(); // 调用传入的 prompt 函数

            var response = default(char); // 定义一个字符变量 response，并初始化为默认值
            do
            {
                response = Console.ReadKey(intercept: true).KeyChar; // 从控制台获取用户输入的字符，不显示在屏幕上
            }
            while (response != '0' && response != '1');  // 循环直到用户输入为0或1

            View.ShowChar(response);  // 在视图中显示用户输入的字符
            return response == '1';  // 返回用户输入是否为1的布尔值
        }

        /// <summary>
        /// 获取给定公司集合中每家公司的交易金额，并返回更新后的资产。
        /// </summary>
        /// <param name="assets">
        /// 要更新的资产。
        /// </param>
        /// <param name="companies">
        /// 公司集合。
        /// </param>
        /// <returns>
        /// 更新后的资产。
        /// </returns>
        # 定义一个名为 UpdateAssets 的函数，接受两个参数 assets 和 companies，并返回一个 Assets 对象
        public static Assets UpdateAssets(Assets assets, IEnumerable<Company> companies)
        {
            # 创建一个无限循环，直到遇到 break 或 return 语句
            while (true)
            {
                # 调用 View 类的 PromptEnterTransactions 方法，提示用户输入交易信息
                View.PromptEnterTransactions();

                # 调用 Broker 类的 Apply 方法，传入 assets 对象、companies 列表中每个公司的交易金额和 companies 列表
                var result = Broker.Apply (
                    assets,
                    companies.Select(GetTransactionAmount).ToList(),
                    companies);

                # 使用 switch 语句根据 result 的不同情况进行处理
                switch (result)
                {
                    # 如果 result 匹配 (Assets newAssets, TransactionResult.Ok) 的情况
                    case (Assets newAssets, TransactionResult.Ok):
                        # 返回新的资产对象 newAssets
                        return newAssets;
                    # 如果 result 匹配 (_, TransactionResult.Oversold) 的情况
                    case (_, TransactionResult.Oversold):
                        # 调用 View 类的 ShowOversold 方法，显示提示信息
                        View.ShowOversold();
                        # 跳出当前 case 分支
                        break;
                    # 如果 result 匹配 (Assets newAssets, TransactionResult.Overspent) 的情况
                    case (Assets newAssets, TransactionResult.Overspent):
                        # 调用 View 类的 ShowOverspent 方法，显示提示信息，并传入参数 -newAssets.Cash
                        View.ShowOverspent(-newAssets.Cash);
        /// <summary>
        /// Gets a transaction amount for the given company.
        /// </summary>
        /// <param name="company">
        /// The company to buy or sell.
        /// </param>
        /// <returns>
        /// The number of shares to buy or sell.
        /// </returns>
        public static int GetTransactionAmount(Company company)
        {
            // 无限循环，直到条件满足退出循环
            while (true)
            {
                // 调用 View 类的 PromptBuySellCompany 方法，提示用户买入或卖出指定的公司
                View.PromptBuySellCompany(company);
# 从控制台读取用户输入的内容
var input = Console.ReadLine();
# 如果输入为空，则退出程序
if (input is null)
    Environment.Exit(0);
# 如果输入无法转换为整数，则提示用户输入有效的整数
else if (!Int32.TryParse(input, out var amount))
    View.PromptValidInteger();
# 如果输入能够成功转换为整数，则返回该整数
else
    return amount;
```