# `.\DB-GPT-src\web\components\flow\add-nodes.tsx`

```py
import { apiInterceptors, getFlowNodes } from '@/client/api';
import { IFlowNode } from '@/types/flow';
import { PlusOutlined } from '@ant-design/icons';
import { Badge, Button, Collapse, CollapseProps, Input, Popover } from 'antd';
import React, { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { FLOW_NODES_KEY } from '@/utils';
import StaticNodes from './static-nodes';

// 输入框组件，来自antd库
const { Search } = Input;

// 定义分组数据的类型
type GroupType = { category: string; categoryLabel: string; nodes: IFlowNode[] };

// 添加节点的组件，React函数组件
const AddNodes: React.FC = () => {
  // 多语言翻译函数
  const { t } = useTranslation();

  // 状态管理：运算符节点、资源节点、运算符节点分组、资源节点分组、搜索值
  const [operators, setOperators] = useState<Array<IFlowNode>>([]);
  const [resources, setResources] = useState<Array<IFlowNode>>([]);
  const [operatorsGroup, setOperatorsGroup] = useState<GroupType[]>([]);
  const [resourcesGroup, setResourcesGroup] = useState<GroupType[]>([]);
  const [searchValue, setSearchValue] = useState<string>('');

  // 组件加载后获取节点数据
  useEffect(() => {
    getNodes();
  }, []);

  // 异步函数：获取节点数据
  async function getNodes() {
    // 发起API请求，获取流程节点数据
    const [_, data] = await apiInterceptors(getFlowNodes());
    if (data && data.length > 0) {
      // 将节点数据存储在本地存储中
      localStorage.setItem(FLOW_NODES_KEY, JSON.stringify(data));
      // 根据节点类型筛选运算符节点和资源节点
      const operatorNodes = data.filter((node) => node.flow_type === 'operator');
      const resourceNodes = data.filter((node) => node.flow_type === 'resource');
      // 更新状态：设置运算符节点和资源节点
      setOperators(operatorNodes);
      setResources(resourceNodes);
      // 更新状态：设置运算符节点分组和资源节点分组
      setOperatorsGroup(groupNodes(operatorNodes));
      setResourcesGroup(groupNodes(resourceNodes));
    }
  }

  // 函数：根据类别分组节点数据
  function groupNodes(data: IFlowNode[]) {
    const groups: GroupType[] = [];
    // 用于存储不同类别节点的映射表
    const categoryMap: Record<string, { category: string; categoryLabel: string; nodes: IFlowNode[] }> = {};
    data.forEach((item) => {
      const { category, category_label } = item;
      // 如果映射表中没有当前类别的节点，创建新的类别节点组，并加入到groups数组中
      if (!categoryMap[category]) {
        categoryMap[category] = { category, categoryLabel: category_label, nodes: [] };
        groups.push(categoryMap[category]);
      }
      // 将节点加入对应类别的节点数组中
      categoryMap[category].nodes.push(item);
    });
    return groups;
  }

  // useMemo钩子：计算运算符节点的折叠面板项
  const operatorItems: CollapseProps['items'] = useMemo(() => {
    if (!searchValue) {
      // 如果没有搜索值，返回全部运算符节点的折叠面板项
      return operatorsGroup.map(({ category, categoryLabel, nodes }) => ({
        key: category,
        label: categoryLabel,
        children: <StaticNodes nodes={nodes} />,
        extra: <Badge showZero count={nodes.length || 0} style={{ backgroundColor: nodes.length > 0 ? '#52c41a' : '#7f9474' }} />,
      }));
    } else {
      // 如果有搜索值，根据搜索值过滤节点并返回分组后的折叠面板项
      const searchedNodes = operators.filter((node) => node.label.toLowerCase().includes(searchValue.toLowerCase()));
      return groupNodes(searchedNodes).map(({ category, categoryLabel, nodes }) => ({
        key: category,
        label: categoryLabel,
        children: <StaticNodes nodes={nodes} />,
        extra: <Badge showZero count={nodes.length || 0} style={{ backgroundColor: nodes.length > 0 ? '#52c41a' : '#7f9474' }} />,
      }));
    }
  }, [operatorsGroup, operators, searchValue]);
  }, [operatorsGroup, searchValue]);

  const resourceItems: CollapseProps['items'] = useMemo(() => {
    // 如果搜索值为空，则展示所有资源分组
    if (!searchValue) {
      return resourcesGroup.map(({ category, categoryLabel, nodes }) => ({
        key: category,
        label: categoryLabel,
        children: <StaticNodes nodes={nodes} />,  // 显示静态节点列表
        extra: <Badge showZero count={nodes.length || 0} style={{ backgroundColor: nodes.length > 0 ? '#52c41a' : '#7f9474' }} />,  // 显示节点数量的徽章
      }));
    } else {
      // 如果有搜索值，则筛选出包含搜索值的资源节点，并按分组重新组织
      const searchedNodes = resources.filter((node) => node.label.toLowerCase().includes(searchValue.toLowerCase()));
      return groupNodes(searchedNodes).map(({ category, categoryLabel, nodes }) => ({
        key: category,
        label: categoryLabel,
        children: <StaticNodes nodes={nodes} />,  // 显示静态节点列表
        extra: <Badge showZero count={nodes.length || 0} style={{ backgroundColor: nodes.length > 0 ? '#52c41a' : '#7f9474' }} />,  // 显示节点数量的徽章
      }));
    }
  }, [resourcesGroup, searchValue]);

  function searchNode(val: string) {
    setSearchValue(val);  // 设置搜索值状态
  }

  return (
    <Popover
      placement="bottom"
      trigger={['click']}
      content={
        <div className="w-[320px] overflow-hidden overflow-y-auto scrollbar-default">
          <p className="my-2 font-bold">{t('add_node')}</p>  // 显示“添加节点”标题
          <Search placeholder="Search node" onSearch={searchNode} />  // 显示搜索框，并调用 searchNode 函数处理搜索
          <h2 className="my-2 ml-2 font-semibold">{t('operators')}</h2>  // 显示“操作符”标题
          <Collapse
            className="max-h-[300px] overflow-hidden overflow-y-auto scrollbar-default"
            size="small"
            defaultActiveKey={['']}
            items={operatorItems}  // 显示操作符的折叠面板
          />
          <h2 className="my-2 ml-2 font-semibold">{t('resource')}</h2>  // 显示“资源”标题
          <Collapse
            className="max-h-[300px] overflow-hidden overflow-y-auto scrollbar-default"
            size="small"
            defaultActiveKey={['']}
            items={resourceItems}  // 显示资源的折叠面板
          />
        </div>
      }
    >
      <Button
        type="primary"
        className="flex items-center justify-center rounded-full left-4 top-4"
        style={{ zIndex: 1050 }}
        icon={<PlusOutlined />}  // 显示一个带加号图标的按钮
      ></Button>
    </Popover>
  );
};

export default AddNodes;


注释：


// 结束 AddNodes 组件定义的主体部分
};

// 导出 AddNodes 组件作为默认导出
export default AddNodes;


这段代码片段定义了一个名为 `AddNodes` 的组件，并将其作为默认导出。在 JavaScript 中，`export default` 语句用于导出一个默认的函数、对象或类，以便在其他文件中导入使用。
```