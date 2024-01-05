# `d:/src/tocomm/basic-computer-games\95_Weekday\javascript\weekday.js`

```
// WEEKDAY
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

/**
 * Print given string to the end of the "output" element.
 * @param str - The string to be printed
 */
function print(str) {
    document.getElementById("output").appendChild(document.createTextNode(str));
}

/**
 * Obtain user input
 * @returns {Promise<String>} - A promise that resolves to the user input as a string
 */
function input() {
    return new Promise(function (resolve) {
        // Create an input element
        const input_element = document.createElement("INPUT");
        print("? ");  # 打印问号提示用户输入
        input_element.setAttribute("type", "text");  # 设置输入元素的类型为文本
        input_element.setAttribute("length", "50");  # 设置输入元素的长度为50
        document.getElementById("output").appendChild(input_element);  # 将输入元素添加到输出元素中
        input_element.focus();  # 让输入元素获得焦点
        input_element.addEventListener("keydown", function (event) {  # 添加键盘按下事件监听器
            if (event.keyCode === 13) {  # 如果按下的键是回车键
                const input_str = input_element.value;  # 获取输入元素的值
                document.getElementById("output").removeChild(input_element);  # 从输出元素中移除输入元素
                print(input_str);  # 打印输入的字符串
                print("\n");  # 打印换行符
                resolve(input_str);  # 返回输入的字符串
            }
        });
    });
}

/**
 * Create a string consisting of the given number of spaces
# 定义一个函数，根据给定的空格数量生成对应数量的空格字符串
def tab(spaceCount):
    str = ""
    while spaceCount > 0:  # 当空格数量大于0时
        str += " "  # 将空格添加到字符串中
        spaceCount -= 1  # 减少空格数量
    return str  # 返回生成的空格字符串

# 定义一些常量
MONTHS_PER_YEAR = 12  # 一年有12个月
DAYS_PER_COMMON_YEAR = 365  # 普通年份有365天
DAYS_PER_IDEALISED_MONTH = 30  # 理想月份有30天
MAXIMUM_DAYS_PER_MONTH = 31  # 一个月最多有31天
# 在普通年份中，每个月第一天的星期几的偏移量
COMMON_YEAR_MONTH_OFFSET = [0, 3, 3, 6, 1, 4, 6, 2, 5, 0, 3, 5]

# 日期表示
class DateStruct {
    #year;  // 私有变量，用于存储年份
    #month;  // 私有变量，用于存储月份
    #day;  // 私有变量，用于存储日期

    /**
     * Build a DateStruct
     * @param {number} year  // 参数，表示年份
     * @param {number} month  // 参数，表示月份
     * @param {number} day  // 参数，表示日期
     */
    constructor(year, month, day) {
        this.#year = year;  // 初始化年份
        this.#month = month;  // 初始化月份
        this.#day = day;  // 初始化日期
    }

    get year() {  // 获取年份的方法
        return this.#year;  // 返回年份
    }
```
在这段代码中，我们定义了一个名为DateStruct的类，其中包含了私有变量#year、#month和#day，用于存储年、月和日。构造函数constructor用于初始化这些私有变量，而get year方法用于获取年份的值。
    # 获取月份属性的值
    get month() {
        return this.#month;
    }

    # 获取日期属性的值
    get day() {
        return this.#day;
    }

    /**
     * 确定日期是否为公历日期。
     * 请注意，公历并非一下子在所有地方都被引入，
     * 请参考 https://en.wikipedia.org/wiki/Gregorian_calendar
     * @returns {boolean} 如果日期可能是公历日期，则返回 true；否则返回 false。
     */
    isGregorianDate() {
        let result = false;
        # 如果年份大于1582，则日期为公历日期
        if (this.#year > 1582) {
            result = true;
        } else if (this.#year === 1582) {
            if (this.#month > 10) {  # 如果月份大于10
                result = true;  # 返回true
            } else if (this.#month === 10 && this.#day >= 15) {  # 否则如果月份等于10且日期大于等于15
                result = true;  # 返回true
            }
        }
        return result;  # 返回结果
    }

    /**
     * The following performs a hash on the day parts which guarantees that
     * 1. different days will return different numbers
     * 2. the numbers returned are ordered.
     * @returns {number}  # 返回一个数字
     */
    getNormalisedDay() {  # 定义一个名为getNormalisedDay的函数
        return (this.year * MONTHS_PER_YEAR + this.month) * MAXIMUM_DAYS_PER_MONTH + this.day;  # 返回一个经过计算的数字
    }

    /**
     * Determine the day of the week.
     * This calculation returns a number between 1 and 7 where Sunday=1, Monday=2, ..., Saturday=7.
     * @returns {number} Value between 1 and 7 representing Sunday to Saturday.
     */
    getDayOfWeek() {
        // Calculate an offset based on the century part of the year.
        const centuriesSince1500 = Math.floor((this.year - 1500) / 100);
        let centuryOffset = centuriesSince1500 * 5 + (centuriesSince1500 + 3) / 4;
        centuryOffset = Math.floor(centuryOffset % 7); // 计算世纪部分的偏移量，并取模得到星期的偏移量

        // Calculate an offset based on the shortened two digit year.
        // January 1st moves forward by approximately 1.25 days per year
        const yearInCentury = this.year % 100;
        const yearInCenturyOffsets = yearInCentury / 4 + yearInCentury; // 计算年份部分的偏移量

        // combine offsets with day and month
        let dayOfWeek = centuryOffset + yearInCenturyOffsets + this.day + COMMON_YEAR_MONTH_OFFSET[this.month - 1]; // 结合世纪偏移量、年份偏移量、日期和月份

        dayOfWeek = Math.floor(dayOfWeek % 7) + 1; // 取模得到星期的值，并加1，得到最终的星期值
        if (this.month <= 2 && this.isLeapYear()) { // 如果月份小于等于2且是闰年
            dayOfWeek--;
        } // 减少星期几的值，以便与数组索引对应

        if (dayOfWeek === 0) { // 如果星期几的值为0，即为星期日
            dayOfWeek = 7; // 将星期几的值设为7，即为星期日的索引值
        }
        return dayOfWeek; // 返回星期几的索引值

    }

    /**
     * Determine if the given year is a leap year.
     * @returns {boolean} - 返回布尔值，表示给定年份是否为闰年
     */
    isLeapYear() {
        if ((this.year % 4) !== 0) { // 如果年份不能被4整除
            return false; // 返回false，表示不是闰年
        } else if ((this.year % 100) !== 0) { // 如果年份能被4整除但不能被100整除
            return true; // 返回true，表示是闰年
        } else if ((this.year % 400) !== 0) { // 如果年份能被100整除但不能被400整除
            return false; // 返回false，表示不是闰年
        }
        return true;
    }
    # 返回 true

    /**
     * Returns a US formatted date, i.e. Month/Day/Year.
     * @returns {string}
     */
    toString() {
        return this.#month + "/" + this.#day + "/" + this.#year;
    }
    # 返回一个美国格式的日期，即 月/日/年

}

/**
 * Duration representation.
 * Note: this class only handles positive durations well
 */
class Duration {
    #years;
    #months;
    #days;
```

这段代码中的注释并不完整，需要进一步解释每个语句的作用。
    /**
     * 构建一个持续时间
     * @param {number} years - 年份
     * @param {number} months - 月份
     * @param {number} days - 天数
     */
    constructor(years, months, days) {
        this.#years = years; // 设置年份
        this.#months = months; // 设置月份
        this.#days = days; // 设置天数
        this.#fixRanges(); // 调用修正范围的方法
    }

    get years() {
        return this.#years; // 获取年份
    }

    get months() {
        return this.#months; // 获取月份
    }
    }

    get days() {  # 定义一个名为days的属性，用于获取私有属性#days的值
        return this.#days;  # 返回私有属性#days的值
    }

    clone() {  # 定义一个名为clone的方法
        return new Duration(this.#years, this.#months, this.#days);  # 返回一个新的Duration对象，使用当前对象的#years、#months和#days属性值
    }

    /**
     * Adjust Duration by removing years, months and days from supplied Duration.
     * This is a naive calculation which assumes all months are 30 days.
     * @param {Duration} timeToRemove  # 参数timeToRemove是一个Duration对象
     */
    remove(timeToRemove) {  # 定义一个名为remove的方法，接受一个参数timeToRemove
        this.#years -= timeToRemove.years;  # 从当前对象的#years属性中减去参数timeToRemove的years属性值
        this.#months -= timeToRemove.months;  # 从当前对象的#months属性中减去参数timeToRemove的months属性值
        this.#days -= timeToRemove.days;  # 从当前对象的#days属性中减去参数timeToRemove的days属性值
        this.#fixRanges();  # 调用私有方法#fixRanges()，用于修正属性值的范围
    }

    /**
     * 将天数和月份移动到预期范围内。
     */
    #fixRanges() {
        if (this.#days < 0) {  // 如果天数小于0
            this.#days += DAYS_PER_IDEALISED_MONTH;  // 将天数增加IDEALISED_MONTH的天数
            this.#months--;  // 月份减1
        }
        if (this.#months < 0) {  // 如果月份小于0
            this.#months += MONTHS_PER_YEAR;  // 将月份增加一年的月份
            this.#years--;  // 年份减1
        }
    }

    /**
     * 计算持续时间所涵盖的天数的近似值。
     * 计算假设所有年份为365天，每个月为30天，并在经过的月份越多时增加额外的部分。
     */
     * @returns {number}
     */
    getApproximateDays() {
        // 计算近似天数，包括年份乘以每年的天数，月份乘以理想月份的天数，以及天数本身
        // 还包括对月份除以2取整的修正
        return (
            (this.#years * DAYS_PER_COMMON_YEAR)
            + (this.#months * DAYS_PER_IDEALISED_MONTH)
            + this.#days
            + Math.floor(this.#months / 2)
        );
    }

    /**
     * Returns a formatted duration with tab separated values, i.e. Years\tMonths\tDays.
     * @returns {string}
     */
    toString() {
        // 返回格式化后的持续时间，使用制表符分隔年、月、日
        return this.#years + "\t" + this.#months + "\t" + this.#days;
    }

    /**
    /**
     * Determine approximate Duration between two dates.
     * This is a naive calculation which assumes all months are 30 days.
     * @param {DateStruct} date1 - 第一个日期
     * @param {DateStruct} date2 - 第二个日期
     * @returns {Duration} - 返回两个日期之间的持续时间
     */
    static between(date1, date2) {
        let years = date1.year - date2.year;  // 计算年份差
        let months = date1.month - date2.month;  // 计算月份差
        let days = date1.day - date2.day;  // 计算天数差
        return new Duration(years, months, days);  // 返回持续时间对象
    }

    /**
     * Calculate years, months and days as factor of days.
     * This is a naive calculation which assumes all months are 30 days.
     * @param dayCount Total day to convert to a duration - 要转换为持续时间的总天数
     * @param factor   Factor to apply when calculating the duration - 计算持续时间时要应用的因子
     * @returns {Duration} - 返回计算得到的持续时间
     */
    static fromDays(dayCount, factor) {  // 从天数和因子计算出年、月、日的持续时间
        let totalDays = Math.floor(factor * dayCount);  // 计算总天数
        const years = Math.floor(totalDays / DAYS_PER_COMMON_YEAR);  // 计算年数
        totalDays -= years * DAYS_PER_COMMON_YEAR;  // 减去年数后剩余的天数
        const months = Math.floor(totalDays / DAYS_PER_IDEALISED_MONTH);  // 计算月数
        const days = totalDays - (months * DAYS_PER_IDEALISED_MONTH);  // 计算剩余的天数
        return new Duration(years, months, days);  // 返回持续时间对象
    }
}

// 主控制部分
async function main() {
    /**
     * 读取日期并提取日期信息。
     * 期望日期部分以逗号分隔，使用美国日期顺序，即月，日，年。
     * @returns {Promise<DateStruct>} 返回一个包含日期信息的 Promise 对象
     */
    async function inputDate() {
        let dateString = await input();  // 从用户输入中获取日期字符串
        const month = parseInt(dateString);  // 将日期字符串转换为整数表示的月份
        const day = parseInt(dateString.substr(dateString.indexOf(",") + 1));  // 从日期字符串中获取逗号后面的部分并转换为整数表示的日期
        const year = parseInt(dateString.substr(dateString.lastIndexOf(",") + 1));  // 从日期字符串中获取最后一个逗号后面的部分并转换为整数表示的年份
        return new DateStruct(year, month, day);  // 返回一个新的 DateStruct 对象，表示给定的年、月、日
    }

    /**
     * Obtain text for the day of the week.
     * @param {DateStruct} date
     * @returns {string}
     */
    function getDayOfWeekText(date) {
        const dayOfWeek = date.getDayOfWeek();  // 获取给定日期的星期几
        let dayOfWeekText = "";  // 初始化一个空字符串用于存储星期几的文本
        switch (dayOfWeek) {  // 根据星期几的值进行不同的处理
            case 1:
                dayOfWeekText = "SUNDAY.";  // 如果是星期一，则将 dayOfWeekText 设置为 "SUNDAY."
                break;
            case 2:
                dayOfWeekText = "MONDAY.";  // 如果是星期二，则将 dayOfWeekText 设置为 "MONDAY."
            case 3:  // 如果星期几是3，表示星期二
                dayOfWeekText = "TUESDAY.";  // 将dayOfWeekText设置为"TUESDAY."
                break;  // 跳出switch语句
            case 4:  // 如果星期几是4，表示星期三
                dayOfWeekText = "WEDNESDAY.";  // 将dayOfWeekText设置为"WEDNESDAY."
                break;  // 跳出switch语句
            case 5:  // 如果星期几是5，表示星期四
                dayOfWeekText = "THURSDAY.";  // 将dayOfWeekText设置为"THURSDAY."
                break;  // 跳出switch语句
            case 6:  // 如果星期几是6，表示星期五
                if (date.day === 13) {  // 如果日期是13号
                    dayOfWeekText = "FRIDAY THE THIRTEENTH---BEWARE!";  // 将dayOfWeekText设置为"FRIDAY THE THIRTEENTH---BEWARE!"
                } else {
                    dayOfWeekText = "FRIDAY.";  // 否则将dayOfWeekText设置为"FRIDAY."
                }
                break;  // 跳出switch语句
            case 7:  // 如果星期几是7，表示星期六
                dayOfWeekText = "SATURDAY.";  // 将dayOfWeekText设置为"SATURDAY."
                break;  // 跳出switch语句
    }
    return dayOfWeekText;
}

print(tab(32) + "WEEKDAY\n");  # 打印"WEEKDAY"，使用制表符进行格式化
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  # 打印"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，使用制表符进行格式化
print("\n");  # 打印空行
print("\n");  # 打印空行
print("\n");  # 打印空行
print("WEEKDAY IS A COMPUTER DEMONSTRATION THAT\n");  # 打印"WEEKDAY IS A COMPUTER DEMONSTRATION THAT"
print("GIVES FACTS ABOUT A DATE OF INTEREST TO YOU.\n");  # 打印"GIVES FACTS ABOUT A DATE OF INTEREST TO YOU."
print("\n");  # 打印空行
print("ENTER TODAY'S DATE IN THE FORM: 3,24,1979  ");  # 打印"ENTER TODAY'S DATE IN THE FORM: 3,24,1979  "
const today = await inputDate();  # 使用inputDate()函数获取用户输入的今天的日期
// This program determines the day of the week
//  for a date after 1582
print("ENTER DAY OF BIRTH (OR OTHER DAY OF INTEREST)");  # 打印"ENTER DAY OF BIRTH (OR OTHER DAY OF INTEREST)"
const dateOfBirth = await inputDate();  # 使用inputDate()函数获取用户输入的出生日期或其他感兴趣的日期
print("\n");  # 打印空行
// Test for date before current calendar.
    # 如果生日不是公历日期，则打印相应的提示信息
    if (!dateOfBirth.isGregorianDate()) {
        print("NOT PREPARED TO GIVE DAY OF WEEK PRIOR TO X.XV.MDLXXXII.\n");
    } else {
        # 获取今天和生日的标准化日期
        const normalisedToday = today.getNormalisedDay();
        const normalisedDob = dateOfBirth.getNormalisedDay();

        # 获取生日对应的星期几文本
        let dayOfWeekText = getDayOfWeekText(dateOfBirth);
        # 根据标准化日期比较今天和生日的关系，并打印相应的信息
        if (normalisedToday < normalisedDob) {
            print(dateOfBirth + " WILL BE A " + dayOfWeekText + "\n");
        } else if (normalisedToday === normalisedDob) {
            print(dateOfBirth + " IS A " + dayOfWeekText + "\n");
        } else {
            print(dateOfBirth + " WAS A " + dayOfWeekText + "\n");
        }

        # 如果今天和生日的标准化日期不相等，则打印换行符
        if (normalisedToday !== normalisedDob) {
            print("\n");
            # 计算今天和生日之间的时间差
            let differenceBetweenDates = Duration.between(today, dateOfBirth);
            # 如果时间差中包含年份，则继续判断天数和月数
            if (differenceBetweenDates.years >= 0) {
                if (differenceBetweenDates.days === 0 && differenceBetweenDates.months === 0) {
                    print("***HAPPY BIRTHDAY***\n");  // 打印生日祝福语

                }
                print("                        \tYEARS\tMONTHS\tDAYS\n");  // 打印表头
                print("                        \t-----\t------\t----\n");  // 打印分隔线
                print("YOUR AGE (IF BIRTHDATE) \t" + differenceBetweenDates + "\n");  // 打印年龄

                const approximateDaysBetween = differenceBetweenDates.getApproximateDays();  // 获取两个日期之间的大致天数
                const unaccountedTime = differenceBetweenDates.clone();  // 克隆未计算的时间段

                // 35% sleeping
                const sleepTimeSpent = Duration.fromDays(approximateDaysBetween, 0.35);  // 计算睡眠时间
                print("YOU HAVE SLEPT \t\t\t" + sleepTimeSpent + "\n");  // 打印睡眠时间
                unaccountedTime.remove(sleepTimeSpent);  // 从未计算的时间中移除睡眠时间

                // 17% eating
                const eatenTimeSpent = Duration.fromDays(approximateDaysBetween, 0.17);  // 计算进食时间
                print("YOU HAVE EATEN \t\t\t" + eatenTimeSpent + "\n");  // 打印进食时间
                unaccountedTime.remove(eatenTimeSpent);  // 从未计算的时间中移除进食时间

                // 23% working, studying or playing
// 计算工作/学习时间所占用的时长
const workPlayTimeSpent = Duration.fromDays(approximateDaysBetween, 0.23);
// 如果未计算的时间小于等于3年
if (unaccountedTime.years <= 3) {
    // 打印工作/玩耍时间所占用的时长
    print("YOU HAVE PLAYED \t\t" + workPlayTimeSpent + "\n");
} 
// 如果未计算的时间小于等于9年
else if (unaccountedTime.years <= 9) {
    // 打印工作/学习时间所占用的时长
    print("YOU HAVE PLAYED/STUDIED \t" + workPlayTimeSpent + "\n");
} 
// 如果未计算的时间大于9年
else {
    // 打印工作/玩耍时间所占用的时长
    print("YOU HAVE WORKED/PLAYED \t\t" + workPlayTimeSpent + "\n");
}
// 从未计算的时间中减去工作/玩耍时间
unaccountedTime.remove(workPlayTimeSpent);

// 打印剩余的放松时间
print("YOU HAVE RELAXED \t\t" + unaccountedTime + "\n");

// 计算退休年份
const retirementYear = dateOfBirth.year + 65;
// 打印退休年份
print("\n");
print(tab(16) + "***  YOU MAY RETIRE IN " + retirementYear + " ***\n");
print("\n");
    # 打印空行
    print("\n");
    print("\n");
    print("\n");
    print("\n");
    print("\n");
}

# 调用主函数
main();
```