# `.\DB-GPT-src\web\components\chart\autoChart\types.ts`

```py
# 导入必要的模块和类型定义
import { Advice, AdvisorConfig, ChartId, Datum, FieldInfo, PureChartKnowledge } from '@antv/ava';

# 定义图表类型
export type ChartType = ChartId | string;

# 定义建议规范和规则配置类型
export type Specification = Advice['spec'] | any;
export type RuleConfig = AdvisorConfig['ruleCfg'];

# 定义自动生成图表的属性类型
export type AutoChartProps = {
  data: Datum[];  # 数据数组
  chartType: ChartType[];  # 建议的图表类型数组
  scopeOfCharts?: {  # 图表范围，可包含或排除特定图表类型
    exclude?: string[];  # 要排除的图表类型数组
    include?: string[];  # 要包含的图表类型数组
  };
  ruleConfig?: RuleConfig;  # 自定义规则配置
};

# 定义图表知识类型
export type ChartKnowledge = PureChartKnowledge & { toSpec?: any };

# 定义自定义图表类型
export type CustomChart = {
  chartType: ChartType;  # 图表类型ID，唯一
  chartKnowledge: ChartKnowledge;  # 图表知识
  chineseName?: string;  # 图表中文名称
};

# 定义获取图表配置属性类型
export type GetChartConfigProps = {
  data: Datum[];  # 数据数组
  spec: Specification;  # 规范
  dataProps: FieldInfo[];  # 数据属性
  chartType?: ChartType;  # 图表类型
};

# 定义自定义顾问配置类型
export type CustomAdvisorConfig = {
  charts?: CustomChart[];  # 自定义图表数组
  scopeOfCharts?: {  # 图表范围，可包含或排除特定图表类型
    exclude?: string[];  # 要排除的图表类型数组
    include?: string[];  # 要包含的图表类型数组
  };
  ruleConfig?: RuleConfig;  # 自定义规则配置
};
```