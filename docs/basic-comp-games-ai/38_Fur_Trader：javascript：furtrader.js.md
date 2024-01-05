# `d:/src/tocomm/basic-computer-games\38_Fur_Trader\javascript\furtrader.js`

```
// 创建一个新的 Promise 对象，用于处理输入操作
// 创建一个 INPUT 元素，用于接收用户输入
// 在页面上打印提示符 "? "
// 设置 INPUT 元素的类型为文本输入
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 添加键盘按下事件监听器，当按下回车键时执行相应操作
input_element.addEventListener("keydown", function (event) {
    # 如果按下的键是回车键
    if (event.keyCode == 13) {
        # 将输入元素的值赋给输入字符串
        input_str = input_element.value;
        # 从 id 为 "output" 的元素中移除输入元素
        document.getElementById("output").removeChild(input_element);
        # 打印输入字符串
        print(input_str);
        # 打印换行符
        print("\n");
        # 解析并返回输入字符串
        resolve(input_str);
    }
});
# 结束键盘按下事件监听器的定义
});
# 结束函数定义

# 定义一个名为 tab 的函数，接受一个参数 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回修改后的字符串
}

var f = [];  // 声明一个空数组
var bs = [, "MINK", "BEAVER", "ERMINE", "FOX"];  // 声明一个包含四个元素的数组

function reset_stats()  // 定义一个名为 reset_stats 的函数
{
    for (var j = 1; j <= 4; j++)  // 循环遍历 j 从 1 到 4
        f[j] = 0;  // 将数组 f 的每个元素初始化为 0
}

// Main program
async function main()  // 定义一个名为 main 的异步函数
{
    print(tab(31) + "FUR TRADER\n");  // 调用 tab 函数并打印结果
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 调用 tab 函数并打印结果
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  # 打印空行
    first_time = true;  # 初始化第一次循环标志为真
    while (1):  # 进入无限循环
        if (first_time):  # 如果是第一次循环
            print("YOU ARE THE LEADER OF A FRENCH FUR TRADING EXPEDITION IN \n");  # 打印提示信息
            print("1776 LEAVING THE LAKE ONTARIO AREA TO SELL FURS AND GET\n");  # 打印提示信息
            print("SUPPLIES FOR THE NEXT YEAR.  YOU HAVE A CHOICE OF THREE\n");  # 打印提示信息
            print("FORTS AT WHICH YOU MAY TRADE.  THE COST OF SUPPLIES\n");  # 打印提示信息
            print("AND THE AMOUNT YOU RECEIVE FOR YOUR FURS WILL DEPEND\n");  # 打印提示信息
            print("ON THE FORT THAT YOU CHOOSE.\n");  # 打印提示信息
            i = 600;  # 初始化变量 i 为 600
            print("DO YOU WISH TO TRADE FURS?\n");  # 打印提示信息
            first_time = false;  # 将第一次循环标志设为假
        print("ANSWER YES OR NO\t");  # 打印提示信息
        str = await input();  # 等待用户输入
        if (str == "NO"):  # 如果用户输入为 "NO"
            break;  # 退出循环
        print("\n");  # 打印空行
        print("YOU HAVE $" + i + " SAVINGS.\n");  # 打印提示信息
        # 打印文本信息
        print("AND 190 FURS TO BEGIN THE EXPEDITION.\n");
        # 计算随机数并赋值给 e1
        e1 = Math.floor((0.15 * Math.random() + 0.95) * Math.pow(10, 2) + 0.5) / Math.pow(10, 2);
        # 计算随机数并赋值给 b1
        b1 = Math.floor((0.25 * Math.random() + 1.00) * Math.pow(10, 2) + 0.5) / Math.pow(10, 2);
        # 打印文本信息
        print("\n");
        # 打印文本信息
        print("YOUR 190 FURS ARE DISTRIBUTED AMONG THE FOLLOWING\n");
        # 打印文本信息
        print("KINDS OF PELTS: MINK, BEAVER, ERMINE AND FOX.\n");
        # 重置统计数据
        reset_stats();
        # 循环4次
        for (j = 1; j <= 4; j++) {
            # 打印文本信息
            print("\n");
            # 打印文本信息
            print("HOW MANY " + bs[j] + " PELTS DO YOU HAVE\n");
            # 将输入的值转换为整数并赋值给 f[j]
            f[j] = parseInt(await input());
            # 计算所有皮毛的总数并赋值给 f[0]
            f[0] = f[1] + f[2] + f[3] + f[4];
            # 如果总数等于190，则跳出循环
            if (f[0] == 190)
                break;
            # 如果总数大于190，则打印文本信息并跳出循环
            if (f[0] > 190) {
                print("\n");
                print("YOU MAY NOT HAVE THAT MANY FURS.\n");
                print("DO NOT TRY TO CHEAT.  I CAN ADD.\n");
                print("YOU MUST START AGAIN.\n");
                break;
        }
        }
        if (f[0] > 190) {  # 如果f的第一个元素大于190
            first_time = true;  # 设置first_time为true
            continue;  # 继续下一次循环
        }
        print("YOU MAY TRADE YOUR FURS AT FORT 1, FORT 2,\n");  # 打印交易毛皮的提示信息
        print("OR FORT 3.  FORT 1 IS FORT HOCHELAGA (MONTREAL)\n");  # 打印关于FORT 1的信息
        print("AND IS UNDER THE PROTECTION OF THE FRENCH ARMY.\n");  # 打印受法国军队保护的信息
        print("FORT 2 IS FORT STADACONA (QUEBEC) AND IS UNDER THE\n");  # 打印关于FORT 2的信息
        print("PROTECTION OF THE FRENCH ARMY.  HOWEVER, YOU MUST\n");  # 打印受法国军队保护的信息
        print("MAKE A PORTAGE AND CROSS THE LACHINE RAPIDS.\n");  # 打印过LACHINE RAPIDS的信息
        print("FORT 3 IS FORT NEW YORK AND IS UNDER DUTCH CONTROL.\n");  # 打印关于FORT 3的信息
        print("YOU MUST CROSS THROUGH IROQUOIS LAND.\n");  # 打印穿越伊罗quois土地的信息
        do {
            print("ANSWER 1, 2, OR 3.\n");  # 打印选择1、2或3的提示信息
            b = parseInt(await input());  # 将输入的值转换为整数并赋给b
            if (b == 1) {  # 如果b等于1
                print("YOU HAVE CHOSEN THE EASIEST ROUTE.  HOWEVER, THE FORT\n");  # 打印选择了最容易的路线的信息
                print("IS FAR FROM ANY SEAPORT.  THE VALUE\n");  # 打印离任何海港很远的信息
                print("YOU RECEIVE FOR YOUR FURS WILL BE LOW AND THE COST\n");
                # 打印消息，指示选择的路线将导致毛皮的价值较低，供应品的成本较高
                print("OF SUPPLIES HIGHER THAN AT FORTS STADACONA OR NEW YORK.\n");
                # 打印消息，指示在Stadacona或纽约堡的供应品成本较高
            } else if (b == 2) {
                print("YOU HAVE CHOSEN A HARD ROUTE.  IT IS, IN COMPARSION,\n");
                # 打印消息，指示选择了一条困难的路线
                print("HARDER THAN THE ROUTE TO HOCHELAGA BUT EASIER THAN\n");
                # 打印消息，指示比去Hochelaga的路线更困难，但比去纽约的路线更容易
                print("THE ROUTE TO NEW YORK.  YOU WILL RECEIVE AN AVERAGE VALUE\n");
                # 打印消息，指示将获得平均价值的毛皮
                print("FOR YOUR FURS AND THE COST OF YOUR SUPPLIES WILL BE AVERAGE.\n");
                # 打印消息，指示供应品的成本将是平均水平
            } else {
                print("YOU HAVE CHOSEN THE MOST DIFFICULT ROUTE.  AT\n");
                # 打印消息，指示选择了最困难的路线
                print("FORT NEW YORK YOU WILL RECEIVE THE HIGHEST VALUE\n");
                # 打印消息，指示在纽约堡将获得最高价值的毛皮
                print("FOR YOUR FURS.  THE COST OF YOUR SUPPLIES\n");
                # 打印消息，指示供应品的成本将低于其他所有堡垒
                print("WILL BE LOWER THAN AT ALL THE OTHER FORTS.\n");
            }
            if (b >= 1 && b <= 3) {
                # 如果选择的路线在1到3之间
                print("DO YOU WANT TO TRADE AT ANOTHER FORT?\n");
                # 打印消息，询问是否想在另一个堡垒交易
                print("ANSWER YES OR NO\t");
                # 打印消息，提示回答是或否
                str = await input();
                # 等待用户输入
                if (str == "YES") {
                    b = 0;
                    # 如果用户输入是，则将b重置为0
                }
        } while (b < 1 || b > 3) ;
        // 设置显示海狸和所有信息的标志为真
        show_beaver = true;
        show_all = true;
        // 如果用户选择了1号选项
        if (b == 1) {
            // 减去160
            i -= 160;
            // 打印换行符
            print("\n");
            // 计算并赋值m1, e1, b1, d1
            m1 = Math.floor((0.2 * Math.random() + 0.7) * Math.pow(10, 2) + 0.5) / Math.pow(10, 2);
            e1 = Math.floor((0.2 * Math.random() + 0.65) * Math.pow(10, 2) + 0.5) / Math.pow(10, 2);
            b1 = Math.floor((0.2 * Math.random() + 0.75) * Math.pow(10, 2) + 0.5) / Math.pow(10, 2);
            d1 = Math.floor((0.2 * Math.random() + 0.8) * Math.pow(10, 2) + 0.5) / Math.pow(10, 2);
            // 打印费用信息
            print("SUPPLIES AT FORT HOCHELAGA COST $150.00.\n");
            print("YOUR TRAVEL EXPENSES TO HOCHELAGA WERE $10.00.\n");
        } 
        // 如果用户选择了2号选项
        else if (b == 2) {
            // 减去140
            i -= 140;
            // 打印换行符
            print("\n");
            // 计算并赋值m1, e1, b1, p
            m1 = Math.floor((0.3 * Math.random() + 0.85) * Math.pow(10, 2) + 0.5) / Math.pow(10, 2);
            e1 = Math.floor((0.15 * Math.random() + 0.8) * Math.pow(10, 2) + 0.5) / Math.pow(10, 2);
            b1 = Math.floor((0.2 * Math.random() + 0.9) * Math.pow(10, 2) + 0.5) / Math.pow(10, 2);
            p = Math.floor(10 * Math.random()) + 1;
# 如果p小于等于2，则执行以下操作
if (p <= 2):
    # 将f[2]设为0
    f[2] = 0
    # 打印以下信息
    print("YOUR BEAVER WERE TOO HEAVY TO CARRY ACROSS\n")
    print("THE PORTAGE.  YOU HAD TO LEAVE THE PELTS, BUT FOUND\n")
    print("THEM STOLEN WHEN YOU RETURNED.\n")
    # 将show_beaver设为false
    show_beaver = false
# 如果p大于2且小于等于6，则执行以下操作
elif (p <= 6):
    # 打印以下信息
    print("YOU ARRIVED SAFELY AT FORT STADACONA.\n")
# 如果p大于6且小于等于8，则执行以下操作
elif (p <= 8):
    # 重置统计数据
    reset_stats()
    # 打印以下信息
    print("YOUR CANOE UPSET IN THE LACHINE RAPIDS.  YOU\n")
    print("LOST ALL YOUR FURS.\n")
    # 将show_all设为false
    show_all = false
# 如果p大于8且小于等于10，则执行以下操作
elif (p <= 10):
    # 将f[4]设为0
    f[4] = 0
    # 打印以下信息
    print("YOUR FOX PELTS WERE NOT CURED PROPERLY.\n")
    print("NO ONE WILL BUY THEM.\n")
# 打印以下信息
print("SUPPLIES AT FORT STADACONA COST $125.00.\n")
print("YOUR TRAVEL EXPENSES TO STADACONA WERE $15.00.\n")
# 生成一个0.8到1.0之间的随机数，并保留两位小数
d1 = Math.floor((0.2 * Math.random() + 0.8) * Math.pow(10, 2) + 0.5) / Math.pow(10, 2);
# 如果b等于3
} else if (b == 3) {
    # 减去105
    i -= 105;
    # 打印换行符
    print("\n");
    # 生成一个1.05到1.20之间的随机数，并保留两位小数
    m1 = Math.floor((0.15 * Math.random() + 1.05) * Math.pow(10, 2) + 0.5) / Math.pow(10, 2);
    # 生成一个1.10到1.35之间的随机数，并保留两位小数
    d1 = Math.floor((0.25 * Math.random() + 1.1) * Math.pow(10, 2) + 0.5) / Math.pow(10, 2);
    # 生成一个1到10之间的随机整数
    p = Math.floor(10 * Math.random()) + 1;
    # 如果p小于等于2
    if (p <= 2) {
        # 打印信息并结束游戏
        print("YOU WERE ATTACKED BY A PARTY OF IROQUOIS.\n");
        print("ALL PEOPLE IN YOUR TRADING GROUP WERE\n");
        print("KILLED.  THIS ENDS THE GAME.\n");
        break;
    # 如果p小于等于6
    } else if (p <= 6) {
        # 打印信息
        print("YOU WERE LUCKY.  YOU ARRIVED SAFELY\n");
        print("AT FORT NEW YORK.\n");
    # 如果p小于等于8
    } else if (p <= 8) {
        # 重置状态
        reset_stats();
        # 打印信息
        print("YOU NARROWLY ESCAPED AN IROQUOIS RAIDING PARTY.\n");
        print("HOWEVER, YOU HAD TO LEAVE ALL YOUR FURS BEHIND.\n");
                show_all = false;  # 设置变量show_all为false
            } else if (p <= 10) {  # 如果p小于等于10
                b1 /= 2;  # b1除以2
                m1 /= 2;  # m1除以2
                print("YOUR MINK AND BEAVER WERE DAMAGED ON YOUR TRIP.\n");  # 打印旅行中水貂和海狸受损的信息
                print("YOU RECEIVE ONLY HALF THE CURRENT PRICE FOR THESE FURS.\n");  # 打印只能得到当前价格一半的信息
            }
            print("SUPPLIES AT NEW YORK COST $80.00.\n");  # 打印纽约的供应品价格
            print("YOUR TRAVEL EXPENSES TO NEW YORK WERE $25.00.\n");  # 打印到纽约的旅行费用
        }
        print("\n");  # 打印空行
        if (show_all) {  # 如果show_all为真
            if (show_beaver)  # 如果show_beaver为真
                print("YOUR BEAVER SOLD FOR $" + b1 * f[2] + " ");  # 打印卖出海狸的价格
            print("YOUR FOX SOLD FOR $" + d1 * f[4] + "\n");  # 打印卖出狐狸的价格
            print("YOUR ERMINE SOLD FOR $" + e1 * f[3] + " ");  # 打印卖出貂的价格
            print("YOUR MINK SOLD FOR $" + m1 * f[1] + "\n");  # 打印卖出水貂的价格
        }
        i += m1 * f[1] + b1 * f[2] + e1 * f[3] + d1 * f[4];  # 计算总收入
        print("\n");  # 打印空行
        print("YOU NOW HAVE $" + i + " INCLUDING YOUR PREVIOUS SAVINGS\n");  # 打印当前拥有的金额，包括之前的储蓄
        print("\n");  # 打印空行
        print("DO YOU WANT TO TRADE FURS NEXT YEAR?\n");  # 打印询问是否想要在下一年交易毛皮
    }
}

main();  # 调用主函数
```