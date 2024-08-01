# `.\DB-GPT-src\web\components\chat\db-editor.tsx`

```py
import React, { ChangeEvent, Key, useEffect, useMemo, useState } from 'react';
import { useRequest } from 'ahooks';
import { Button, Select, Table, Tooltip } from 'antd';
import { Input, Tree } from 'antd';
import Icon from '@ant-design/icons';
import type { DataNode } from 'antd/es/tree';
import MonacoEditor, { ISession } from './monaco-editor';
import { sendGetRequest, sendSpacePostRequest } from '@/utils/request';
import { useSearchParams } from 'next/navigation';
import { OnChange } from '@monaco-editor/react';
import Header from './header';
import Chart from '../chart';
import { CaretRightOutlined, LeftOutlined, RightOutlined, SaveFilled } from '@ant-design/icons';
import { ColumnType } from 'antd/es/table';
import Database from '../icons/database';
import TableIcon from '../icons/table';
import Field from '../icons/field';
import classNames from 'classnames';
import MyEmpty from '../common/MyEmpty';
import SplitScreenWeight from '@/components/icons/split-screen-width';
import SplitScreenHeight from '@/components/icons/split-screen-height';

const { Search } = Input;

type ITableData = {
  columns: string[];
  values: (string | number)[][];
};

interface EditorValueProps {
  sql?: string;
  thoughts?: string;
  title?: string;
  showcase?: string;
}

interface RoundProps {
  db_name: string;
  round: number;
  round_name: string;
}

interface IProps {
  editorValue?: EditorValueProps;
  chartData?: any;
  tableData?: ITableData;
  layout?: 'TB' | 'LR';
  tables?: any;
  handleChange: OnChange;
}

interface ITableTreeItem {
  title: string;
  key: string;
  type: string;
  default_value: string | null;
  can_null: string;
  comment: string | null;
  children: Array<ITableTreeItem>;
}

// 组件 DbEditorContent，接收多个 props 包括 editorValue, chartData, tableData, tables 和 handleChange
function DbEditorContent({ layout = 'LR', editorValue, chartData, tableData, tables, handleChange }: IProps) {
  // 使用 useMemo 钩子来创建 chartWrapper 元素，只有在 chartData 更新时重新计算
  const chartWrapper = useMemo(() => {
    if (!chartData) return null;

    // 返回一个包含 Chart 组件的 div 元素，用来显示图表数据
    return (
      <div className="flex-1 overflow-auto p-2" style={{ flexShrink: 0, overflow: 'hidden' }}>
        <Chart chartsData={[chartData]} />
      </div>
    );
  }, [chartData]);

  // 使用 useMemo 钩子来创建 columns 和 dataSource 对象，只有在 tableData 更新时重新计算
  const { columns, dataSource } = useMemo<{ columns: ColumnType<any>[]; dataSource: Record<string, string | number>[] }>(() => {
    // 从 tableData 中解构出 columns 和 values
    const { columns: cols = [], values: vals = [] } = tableData ?? {};
    // 将 columns 转换为 Ant Design Table 组件所需的格式
    const tbCols = cols.map<ColumnType<any>>((item) => ({
      key: item,
      dataIndex: item,
      title: item,
    }));
    // 将 values 转换为 dataSource 数组，每个元素是一个对象表示表格行数据
    const tbDatas = vals.map((row) => {
      return row.reduce<Record<string, string | number>>((acc, item, index) => {
        acc[cols[index]] = item;
        return acc;
      }, {});
    });

    return {
      columns: tbCols,
      dataSource: tbDatas,
    };
  }, [tableData]);

  // 使用 useMemo 钩子创建 session 对象，只有在 tables 更新时重新计算
  const session: ISession = useMemo(() => {
    // 创建一个空的映射表 map
    const map: Record<string, { columnName: string; columnType: string; }[]> = {};
    // 从 tables 中获取数据库信息
    const db = tables?.data;
    // 获取数据库中的表列表
    const tableList = db?.children;
    tableList?.forEach((table: ITableTreeItem) => {
      // 遍历 tableList 数组，对每个表格对象生成一个映射
      map[table.title] = table.children.map((column: ITableTreeItem) => {
        // 对每个表格的子节点（列）进行映射，生成包含列名和列类型的对象
        return {
          columnName: column.title,
          columnType: column.type,
        };
      })
    });
    return {
      async getTableList(schemaName) {
        // 异步方法：获取表格列表
        if (schemaName && schemaName !== db?.title) {
          // 如果提供了 schemaName，并且不等于当前数据库标题，则返回空数组
          return [];
        }
        // 返回所有表格的标题组成的数组，或空数组（如果 tableList 不存在）
        return tableList?.map((table: ITableTreeItem) => table.title) || [];
      },
      async getTableColumns(tableName) {
        // 异步方法：根据表格名获取列信息
        return map[tableName] || [];
      },
      async getSchemaList() {
        // 异步方法：获取数据库架构列表，只返回当前数据库标题或空数组
        return db?.title ? [db?.title] : [];
      }
    };
  }, [tables])
  return (
    <div
      className={classNames('flex w-full flex-1 h-full gap-2 overflow-hidden', {
        'flex-col': layout === 'TB',
        'flex-row': layout === 'LR',
      })}
    >
      <div className="flex-1 flex overflow-hidden rounded">
        <MonacoEditor value={editorValue?.sql || ''} language="mysql" onChange={handleChange} thoughts={editorValue?.thoughts || ''} session={session} />
      </div>
      <div className="flex-1 h-full overflow-auto bg-white dark:bg-theme-dark-container rounded p-4">
        {!!tableData?.values.length ? (
          <Table bordered scroll={{ x: 'auto' }} rowKey={columns[0].key} columns={columns} dataSource={dataSource} />
        ) : (
          <div className="h-full flex justify-center items-center">
            <MyEmpty />
          </div>
        )}
        {chartWrapper}
      </div>
    </div>
  );
}

function DbEditor() {
  // 状态管理：展开的树节点的键集合
  const [expandedKeys, setExpandedKeys] = useState<Key[]>([]);
  // 状态管理：搜索框中的数值
  const [searchValue, setSearchValue] = useState('');
  // 状态管理：当前轮次
  const [currentRound, setCurrentRound] = useState<null | string | number>();
  // 状态管理：是否自动展开父节点
  const [autoExpandParent, setAutoExpandParent] = useState(true);
  // 状态管理：图表数据
  const [chartData, setChartData] = useState();
  // 状态管理：编辑器的值
  const [editorValue, setEditorValue] = useState<EditorValueProps | EditorValueProps[]>();
  // 状态管理：新的编辑器值
  const [newEditorValue, setNewEditorValue] = useState<EditorValueProps>();
  // 状态管理：表格数据
  const [tableData, setTableData] = useState<{ columns: string[]; values: (string | number)[] }>();
  // 状态管理：当前选项卡索引
  const [currentTabIndex, setCurrentTabIndex] = useState<number>();
  // 状态管理：菜单展开状态
  const [isMenuExpand, setIsMenuExpand] = useState<boolean>(false);
  // 状态管理：布局方向，TB 表示上下布局，LR 表示左右布局
  const [layout, setLayout] = useState<'TB' | 'LR'>('TB');

  // 从 URL 查询参数中获取 id 和 scene
  const searchParams = useSearchParams();
  const id = searchParams?.get('id');
  const scene = searchParams?.get('scene');

  // 发送请求获取编辑器轮次数据
  const { data: rounds, loading: roundsLoading } = useRequest(
    async () =>
      await sendGetRequest('/v1/editor/sql/rounds', {
        con_uid: id,
      }),
    {
      // 成功时更新当前轮次
      onSuccess: (res) => {
        const lastItem = res?.data?.[res?.data?.length - 1];
        if (lastItem) {
          setCurrentRound(lastItem?.round);
        }
      },
    },
  );

  // 发送 SQL 运行请求
  const { run: runSql, loading: runLoading } = useRequest(
    async () => {
      // 获取当前轮次对应的数据库名称
      const db_name = rounds?.data?.find((item) => item.round === currentRound)?.db_name;
      return await sendSpacePostRequest(`/api/v1/editor/sql/run`, {
        db_name,
        sql: newEditorValue?.sql,
      });
    },
    {
      manual: true,
      // 成功时更新表格数据
      onSuccess: (res) => {
        setTableData({
          columns: res?.data?.colunms,
          values: res?.data?.values,
        });
      },
    },
  );

  // 发送图表运行请求
  const { run: runCharts, loading: runChartsLoading } = useRequest(
    async () => {
      // 获取当前轮次对应的数据库名称
      const db_name = rounds?.data?.find((item) => item.round === currentRound)?.db_name;
      const params: {
        db_name: string;
        sql?: string;
        chart_type?: string;
      } = {
        db_name,
        sql: newEditorValue?.sql,
      };
      // 如果场景为 chat_dashboard，则添加图表类型参数
      if (scene === 'chat_dashboard') {
        params['chart_type'] = newEditorValue?.showcase;
      }
      return await sendSpacePostRequest(`/api/v1/editor/chart/run`, params);
    },
    {
      manual: true,
      ready: !!newEditorValue?.sql,
      // 成功时更新表格和图表数据
      onSuccess: (res) => {
        if (res?.success) {
          setTableData({
            columns: res?.data?.sql_data?.colunms || [],
            values: res?.data?.sql_data?.values || [],
          });
          if (!res?.data?.chart_values) {
            setChartData(undefined);
          } else {
            setChartData({
              type: res?.data?.chart_type,
              values: res?.data?.chart_values,
              title: newEditorValue?.title,
              description: newEditorValue?.thoughts,
            });
          }
        }
      },
  },
);

const { run: submitSql, loading: submitLoading } = useRequest(
  async () => {
    // 获取当前轮次对应的数据库名称
    const db_name = rounds?.data?.find((item: RoundProps) => item.round === currentRound)?.db_name;
    // 发送提交 SQL 请求，并返回结果
    return await sendSpacePostRequest(`/api/v1/sql/editor/submit`, {
      conv_uid: id,
      db_name,
      conv_round: currentRound,
      old_sql: editorValue?.sql,
      old_speak: editorValue?.thoughts,
      new_sql: newEditorValue?.sql,
      // 提取注释内容或使用原始思考文本
      new_speak: newEditorValue?.thoughts?.match(/^\n--(.*)\n\n$/)?.[1]?.trim() || newEditorValue?.thoughts,
    });
  },
  {
    manual: true,
    // 成功响应时运行 runSql 函数
    onSuccess: (res) => {
      if (res?.success) {
        runSql();
      }
    },
  },
);

const { run: submitChart, loading: submitChartLoading } = useRequest(
  async () => {
    // 获取当前轮次对应的数据库名称
    const db_name = rounds?.data?.find((item) => item.round === currentRound)?.db_name;
    // 发送提交图表请求，并返回结果
    return await sendSpacePostRequest(`/api/v1/chart/editor/submit`, {
      conv_uid: id,
      chart_title: newEditorValue?.title,
      db_name,
      old_sql: editorValue?.[currentTabIndex]?.sql,
      new_chart_type: newEditorValue?.showcase,
      new_sql: newEditorValue?.sql,
      // 提取注释内容或使用原始思考文本
      new_comment: newEditorValue?.thoughts?.match(/^\n--(.*)\n\n$/)?.[1]?.trim() || newEditorValue?.thoughts,
      gmt_create: new Date().getTime(),
    });
  },
  {
    manual: true,
    // 成功响应时运行 runCharts 函数
    onSuccess: (res) => {
      if (res?.success) {
        runCharts();
      }
    },
  },
);

const { data: tables } = useRequest(
  async () => {
    // 获取当前轮次对应的数据库名称
    const db_name = rounds?.data?.find((item: RoundProps) => item.round === currentRound)?.db_name;
    // 发送获取表格数据请求，并返回结果
    return await sendGetRequest('/v1/editor/db/tables', {
      db_name,
      page_index: 1,
      page_size: 200,
    });
  },
  {
    // 当 rounds 数据中存在当前轮次的数据库名称时，准备就绪
    ready: !!rounds?.data?.find((item: RoundProps) => item.round === currentRound)?.db_name,
    // 刷新依赖于当前轮次的数据库名称
    refreshDeps: [rounds?.data?.find((item: RoundProps) => item.round === currentRound)?.db_name],
  },
);

const { run: handleGetEditorSql } = useRequest(
  async (round) =>
    // 发送获取编辑器 SQL 请求，并返回结果
    await sendGetRequest('/v1/editor/sql', {
      con_uid: id,
      round,
    }),
  {
    manual: true,
    // 成功响应时处理返回的 SQL 数据
    onSuccess: (res) => {
      let sql = undefined;
      try {
        if (Array.isArray(res?.data)) {
          sql = res?.data;
          setCurrentTabIndex(0);
        } else if (typeof res?.data === 'string') {
          const d = JSON.parse(res?.data);
          sql = d;
        } else {
          sql = res?.data;
        }
      } catch (e) {
        console.log(e);
      } finally {
        // 设置编辑器的值为获取到的 SQL 数据
        setEditorValue(sql);
        if (Array.isArray(sql)) {
          // 如果 SQL 是数组，设置新的编辑器值为当前标签页的 SQL 数据
          setNewEditorValue(sql?.[Number(currentTabIndex || 0)]);
        } else {
          // 否则设置新的编辑器值为整体 SQL 数据
          setNewEditorValue(sql);
        }
      }
    },
  },
);

const treeData = useMemo(() => {
    const loop = (data: Array<ITableTreeItem>, parentKey?: string | number): DataNode[] =>
      // 对给定的数据进行映射处理，返回处理后的节点数组
      data.map((item: ITableTreeItem) => {
        const strTitle = item.title; // 提取当前项的标题文本
        const index = strTitle.indexOf(searchValue); // 查找搜索关键词在标题文本中的位置
        const beforeStr = strTitle.substring(0, index); // 获取搜索关键词之前的文本部分
        const afterStr = strTitle.slice(index + searchValue.length); // 获取搜索关键词之后的文本部分
    
        const renderIcon = (type: string) => {
          // 根据类型渲染对应的图标组件
          switch (type) {
            case 'db':
              return <Database />;
            case 'table':
              return <TableIcon />;
            default:
              return <Field />;
          }
        };
    
        const showTitle =
          index > -1 ? ( // 如果找到了搜索关键词
            <Tooltip title={(item?.comment || item?.title) + (item?.can_null === 'YES' ? '(can null)' : `(can't null)`)}>
              <div className="flex items-center">
                {renderIcon(item.type)}&nbsp;&nbsp;&nbsp;
                {beforeStr}
                <span className="text-[#1677ff]">{searchValue}</span>
                {afterStr}&nbsp;
                {item?.type && <div className="text-gray-400">{item?.type}</div>}
              </div>
            </Tooltip>
          ) : ( // 如果没有找到搜索关键词
            <Tooltip title={(item?.comment || item?.title) + (item?.can_null === 'YES' ? '(can null)' : `(can't null)`)}>
              <div className="flex items-center">
                {renderIcon(item.type)}&nbsp;&nbsp;&nbsp;
                {strTitle}&nbsp;
                {item?.type && <div className="text-gray-400">{item?.type}</div>}
              </div>
            </Tooltip>
          );
    
        if (item.children) { // 如果当前项有子节点
          const itemKey = parentKey ? String(parentKey) + '_' + item.key : item.key; // 计算子节点的键值
          return { title: strTitle, showTitle, key: itemKey, children: loop(item.children, itemKey) }; // 返回包含子节点的节点对象
        }
    
        return {
          title: strTitle, // 返回不包含子节点的节点对象
          showTitle,
          key: item.key,
        };
      });
    
    if (tables?.data) {
      // 设置默认展开第一个节点
      setExpandedKeys([tables?.data.key]);
      return loop([tables?.data]); // 调用循环函数处理树形数据，并返回处理后的结果
    }
    
    return []; // 如果没有数据，则返回空数组
    
    // 使用 useMemo 钩子函数生成数据列表
    const dataList = useMemo(() => {
      let res: { key: string | number; title: string; parentKey?: string | number }[] = [];
    
      const generateList = (data: DataNode[], parentKey?: string | number) => {
        if (!data || data?.length <= 0) return; // 如果数据为空或长度为0，则直接返回
    
        for (let i = 0; i < data.length; i++) {
          const node = data[i];
          const { key, title } = node;
    
          res.push({ key, title: title as string, parentKey }); // 将节点信息添加到结果数组中
    
          if (node.children) {
            generateList(node.children, key); // 递归处理子节点
          }
        }
      };
    
      if (treeData) {
        generateList(treeData); // 如果树形数据存在，则调用生成列表函数处理
      }
    
      return res; // 返回处理后的结果数组
    }, [treeData]);
    
    // 定义获取父节点键值的函数
    const getParentKey = (key: Key, tree: DataNode[]): Key => {
    // 遍历树结构中的每个节点
    for (let i = 0; i < tree.length; i++) {
      // 获取当前节点
      const node = tree[i];
      // 如果节点有子节点
      if (node.children) {
        // 检查子节点中是否存在指定 key 的项
        if (node.children.some((item) => item.key === key)) {
          // 如果存在，将当前节点的 key 设为 parentKey
          parentKey = node.key;
        } else if (getParentKey(key, node.children)) {
          // 否则，递归查找指定 key 的父节点的 key
          parentKey = getParentKey(key, node.children);
        }
      }
    }
    // 返回找到的 parentKey
    return parentKey!;
  };

  // 处理输入框内容变化的回调函数
  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { value } = e.target;
    // 如果 tables.data 存在
    if (tables?.data) {
      // 如果输入值为空字符串
      if (!value) {
        // 清空展开的节点列表
        setExpandedKeys([]);
      } else {
        // 根据输入值筛选匹配的节点的父节点的 key 组成的数组
        const newExpandedKeys = dataList
          .map((item) => {
            // 如果节点的标题包含输入值
            if (item.title.indexOf(value) > -1) {
              // 返回该节点的父节点的 key
              return getParentKey(item.key, treeData);
            }
            return null;
          })
          // 过滤掉空值和重复值
          .filter((item, i, self) => item && self.indexOf(item) === i);
        // 设置新的展开节点列表
        setExpandedKeys(newExpandedKeys as Key[]);
      }
      // 设置搜索值为当前输入值
      setSearchValue(value);
      // 自动展开父节点
      setAutoExpandParent(true);
    }
  };

  // 当 currentRound 改变时执行的副作用函数
  useEffect(() => {
    if (currentRound) {
      // 根据当前回合获取编辑器中的 SQL
      handleGetEditorSql(currentRound);
    }
  }, [handleGetEditorSql, currentRound]);

  // 当 editorValue、scene 和 currentTabIndex 改变时执行的副作用函数
  useEffect(() => {
    if (editorValue && scene === 'chat_dashboard' && currentTabIndex) {
      // 运行图表展示函数
      runCharts();
    }
  }, [currentTabIndex, scene, editorValue, runCharts]);

  // 当 editorValue 和 scene 改变且 scene 不是 'chat_dashboard' 时执行的副作用函数
  useEffect(() => {
    if (editorValue && scene !== 'chat_dashboard') {
      // 运行 SQL 执行函数
      runSql();
    }
  }, [scene, editorValue, runSql]);

  // 解析 SQL 和备注的函数
  function resolveSqlAndThoughts(value: string | undefined) {
    // 如果输入值为空，则返回空的 SQL 和备注对象
    if (!value) {
      return { sql: '', thoughts: '' };
    }
    // 匹配 SQL 注释和 SQL 语句
    const match = value && value.match(/(--.*)\n([\s\S]*)/);
    let thoughts = '';
    let sql;
    // 如果匹配成功并且分组数大于等于 3
    if (match && match.length >= 3) {
      // 第一个分组为注释
      thoughts = match[1];
      // 第二个分组为 SQL 语句
      sql = match[2];
    }
    // 返回解析后的 SQL 和备注对象
    return { sql, thoughts };
  }

  // 返回 JSX 结构的代码块
  return (
    </div>
  );
}

export default DbEditor;
```