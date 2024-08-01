# `.\DB-GPT-src\web\components\chart\autoChart\index.tsx`

```py
import { Empty, Row, Col, Select, Tooltip, Button, Space } from 'antd';
import { Advice, Advisor, Datum } from '@antv/ava';
import { Chart, ChartRef } from '@berryv/g2-react';
import i18n, { I18nKeys } from '@/app/i18n';
import { customizeAdvisor, getVisAdvices } from './advisor/pipeline';
import { useContext, useEffect, useMemo, useRef, useState } from 'react';
import { defaultAdvicesFilter } from './advisor/utils';
import { AutoChartProps, ChartType, CustomAdvisorConfig, CustomChart, Specification } from './types';
import { customCharts } from './charts';
import { ChatContext } from '@/app/chat-context';
import { compact, concat, uniq } from 'lodash';
import { processNilData, sortData } from './charts/util';
import { downloadImage } from '../helpers/downloadChartImage';;
import { DownloadOutlined } from '@ant-design/icons';

const { Option } = Select;

export const AutoChart = (props: AutoChartProps) => {
  // 解构 props 中的属性
  const { chartType, scopeOfCharts, ruleConfig, data: originalData } = props;

  // 处理空值数据 (为'-'的数据)，返回处理后的数据作为 Datum 数组
  const data = processNilData(originalData) as Datum[];
  // 获取当前聊天模式
  const { mode } = useContext(ChatContext);

  // 状态变量定义
  const [advisor, setAdvisor] = useState<Advisor>();
  const [advices, setAdvices] = useState<Advice[]>([]);
  const [renderChartType, setRenderChartType] = useState<ChartType>();
  const chartRef = useRef<ChartRef>();

  // 在组件加载或 ruleConfig、scopeOfCharts 更新时执行
  useEffect(() => {
    // 获取自定义图表配置
    const input_charts: CustomChart[] = customCharts;
    // 设置 Advisor 的自定义配置
    const advisorConfig: CustomAdvisorConfig = {
      charts: input_charts,
      scopeOfCharts: {
        // 指定需要排除的面积图类型
        exclude: ['area_chart', 'stacked_area_chart', 'percent_stacked_area_chart'],
      },
      ruleConfig,
    };
    // 使用自定义配置定制 Advisor 对象
    setAdvisor(customizeAdvisor(advisorConfig));
  }, [ruleConfig, scopeOfCharts]);

  /** 将 AVA 得到的图表推荐结果和模型的合并 */
  const getMergedAdvices = (avaAdvices: Advice[]) => {
    // 如果 Advisor 未定义，返回空数组
    if (!advisor) return [];
    // 对 AVA 推荐结果进行默认过滤
    const filteredAdvices = defaultAdvicesFilter({
      advices: avaAdvices,
    });
    // 合并所有图表类型并去重
    const allChartTypes = uniq(
      compact(
        concat(
          chartType,
          avaAdvices.map((item) => item.type),
        ),
      ),
    );
    // 根据所有图表类型生成建议列表
    const allAdvices = allChartTypes
      .map((chartTypeItem) => {
        const avaAdvice = filteredAdvices.find((item) => item.type === chartTypeItem);
        // 如果在 AVA 推荐列表中找到对应图表类型的推荐，直接采用该推荐
        if (avaAdvice) {
          return avaAdvice;
        }
        // 如果未在 AVA 推荐列表中找到对应图表类型的推荐，生成新的图表规格
        const dataAnalyzerOutput = advisor.dataAnalyzer.execute({ data });
        if ('data' in dataAnalyzerOutput) {
          const specGeneratorOutput = advisor.specGenerator.execute({
            data: dataAnalyzerOutput.data,
            dataProps: dataAnalyzerOutput.dataProps,
            chartTypeRecommendations: [{ chartType: chartTypeItem, score: 1 }],
          });
          // 返回生成的建议
          if ('advices' in specGeneratorOutput) return specGeneratorOutput.advices?.[0];
        }
      })
      .filter((advice) => advice?.spec) as Advice[];
    // 返回所有合并后的建议列表
    return allAdvices;
  };

  // 当 ruleConfig 或 scopeOfCharts 更新时重新计算建议
  useEffect(() => {
    // 获取 AVA 生成的图表推荐列表
    const avaAdvices = getVisAdvices({
      data,
      mode,
      advisor,
      chartType,
    });
    // 将 AVA 推荐结果与模型的建议合并
    const mergedAdvices = getMergedAdvices(avaAdvices);
    // 更新状态中的建议列表
    setAdvices(mergedAdvices);
  }, [advisor, chartType, mode, ruleConfig, data]);

  // 返回组件模板
  return (
    // 组件的 JSX 结构
    <div>
      {/* 选择图表类型的下拉菜单 */}
      <Select defaultValue={chartType} style={{ width: 120 }}>
        {advices.map((advice, index) => (
          <Option key={index} value={advice.type}>
            {i18n.t(I18nKeys[advice.type])}
          </Option>
        ))}
      </Select>
      {/* 图表组件 */}
      <Chart
        ref={chartRef}
        data={data}
        advisor={advisor}
        chartType={renderChartType || chartType}
      />
      {/* 下载图表的按钮 */}
      <Button
        type="primary"
        icon={<DownloadOutlined />}
        onClick={() => downloadImage(chartRef.current?.getChartInstance())}
      >
        {i18n.t(I18nKeys.download_chart)}
      </Button>
    </div>
  );
};
  if (data && advisor) {
    const avaAdvices = getVisAdvices({
      data,
      myChartAdvisor: advisor,
    });
    // 合并模型推荐的图表类型和 ava 推荐的图表类型
    const allAdvices = getMergedAdvices(avaAdvices);
    setAdvices(allAdvices);
    setRenderChartType(allAdvices[0]?.type as ChartType);
  }
}, [JSON.stringify(data), advisor, chartType]);

const visComponent = useMemo(() => {
  /* Advices exist, render the chart. */
  if (advices?.length > 0) {
    const chartTypeInput = renderChartType ?? advices[0].type;
    const spec: Specification = advices?.find((item: Advice) => item.type === chartTypeInput)?.spec ?? undefined;
    if (spec) {
      if (spec.data && ['line_chart', 'step_line_chart'].includes(chartTypeInput)) {
        // 处理 ava 内置折线图的排序问题
        const dataAnalyzerOutput = advisor?.dataAnalyzer.execute({ data })
        if (dataAnalyzerOutput && 'dataProps' in dataAnalyzerOutput) {
          spec.data = sortData({ data: spec.data, xField: dataAnalyzerOutput.dataProps?.find(field => field.recommendation === 'date'), chartType: chartTypeInput });
        }
      }
      if (chartTypeInput === 'pie_chart' && spec?.encode?.color) {
        // 补充饼图的 tooltip title 展示
        spec.tooltip = { title: { field: spec.encode.color } }
      }
      return (
        <Chart
          key={chartTypeInput}
          options={{
            ...spec,
            theme: mode,
            autoFit: true,
            height: 300,
          }}
          ref={chartRef}
        />
      );
    }
  }
}, [advices, renderChartType]);

if (renderChartType) {
  return (
    <div>
      <Row justify='space-between' className="mb-2">
        <Col>
          <Space>
            <span>{i18n.t('Advices')}</span>
            <Select
              className="w-52"
              value={renderChartType}
              placeholder={'Chart Switcher'}
              onChange={(value) => setRenderChartType(value)}
              size={'small'}
            >
              {advices?.map((item) => {
                const name = i18n.t(item.type as I18nKeys);
                return (
                  <Option key={item.type} value={item.type}>
                    <Tooltip title={name} placement={'right'}>
                      <div>{name}</div>
                    </Tooltip>
                  </Option>
                );
              })}
            </Select>
          </Space>
        </Col>
        <Col>
          <Tooltip title={i18n.t('Download')}>
            <Button
              onClick={() => downloadImage(chartRef.current, i18n.t(renderChartType as I18nKeys))}
              icon={<DownloadOutlined />}
              type='text'
            />
          </Tooltip>
        </Col>
      </Row>
      <div className="auto-chart-content">{visComponent}</div>
    </div>
  );
}


注释：


// 如果数据和顾问存在，则执行以下代码块
if (data && advisor) {
  // 获取可视化建议，使用当前数据和图表顾问对象
  const avaAdvices = getVisAdvices({
    data,
    myChartAdvisor: advisor,
  });
  // 合并模型推荐的图表类型和 ava 推荐的图表类型
  const allAdvices = getMergedAdvices(avaAdvices);
  // 设置所有推荐的图表类型
  setAdvices(allAdvices);
  // 设置当前渲染的图表类型为推荐列表的第一个类型
  setRenderChartType(allAdvices[0]?.type as ChartType);
}

const visComponent = useMemo(() => {
  /* 如果存在推荐建议，则渲染图表 */
  if (advices?.length > 0) {
    // 确定要渲染的图表类型，默认为 renderChartType 或者推荐列表的第一个类型
    const chartTypeInput = renderChartType ?? advices[0].type;
    // 获取对应图表类型的规格
    const spec: Specification = advices?.find((item: Advice) => item.type === chartTypeInput)?.spec ?? undefined;
    if (spec) {
      // 如果规格中包含数据并且是线形图或步进线形图类型，则处理数据排序
      if (spec.data && ['line_chart', 'step_line_chart'].includes(chartTypeInput)) {
        // 执行顾问的数据分析器，处理数据排序问题
        const dataAnalyzerOutput = advisor?.dataAnalyzer.execute({ data })
        if (dataAnalyzerOutput && 'dataProps' in dataAnalyzerOutput) {
          // 根据日期字段推荐对数据进行排序
          spec.data = sortData({ data: spec.data, xField: dataAnalyzerOutput.dataProps?.find(field => field.recommendation === 'date'), chartType: chartTypeInput });
        }
      }
      // 如果是饼图类型，并且规格中包含颜色编码字段，则补充饼图的 tooltip title 展示
      if (chartTypeInput === 'pie_chart' && spec?.encode?.color) {
        spec.tooltip = { title: { field: spec.encode.color } }
      }
      // 返回渲染的图表组件
      return (
        <Chart
          key={chartTypeInput}
          options={{
            ...spec,
            theme: mode,
            autoFit: true,
            height: 300,
          }}
          ref={chartRef}
        />
      );
    }
  }
}, [advices, renderChartType]);

// 如果 renderChartType 存在，则渲染图表切换器和下载按钮
if (renderChartType) {
  return (
    <div>
      <Row justify='space-between' className="mb-2">
        <Col>
          <Space>
            {/* 显示推荐的图表类型 */}
            <span>{i18n.t('Advices')}</span>
            {/* 选择框，用于切换当前渲染的图表类型 */}
            <Select
              className="w-52"
              value={renderChartType}
              placeholder={'Chart Switcher'}
              onChange={(value) => setRenderChartType(value)}
              size={'small'}
            >
              {/* 遍历并渲染所有推荐的图表类型选项 */}
              {advices?.map((item) => {
                const name = i18n.t(item.type as I18nKeys);
                return (
                  <Option key={item.type} value={item.type}>
                    {/* 鼠标悬停显示图表类型的名称 */}
                    <Tooltip title={name} placement={'right'}>
                      <div>{name}</div>
                    </Tooltip>
                  </Option>
                );
              })}
            </Select>
          </Space>
        </Col>
        <Col>
          {/* 下载按钮，用于下载当前图表的图像 */}
          <Tooltip title={i18n.t('Download')}>
            <Button
              onClick={() => downloadImage(chartRef.current, i18n.t(renderChartType as I18nKeys))}
              icon={<DownloadOutlined />}
              type='text'
            />
          </Tooltip>
        </Col>
      </Row>
      {/* 自动适应的图表内容 */}
      <div className="auto-chart-content">{visComponent}</div>
    </div>
  );
}
    );
  }



    ); 
  }
};

export * from './helpers';
```