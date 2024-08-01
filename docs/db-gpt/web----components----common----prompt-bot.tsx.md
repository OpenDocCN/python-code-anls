# `.\DB-GPT-src\web\components\common\prompt-bot.tsx`

```py
import { useState } from 'react';
import { List, FloatButton, Popover, Tooltip, Form, message, Select, ConfigProvider } from 'antd';
import { useRequest } from 'ahooks';
import { sendSpacePostRequest } from '@/utils/request';
import { useTranslation } from 'react-i18next';

type SelectTableProps = {
  data: any;
  loading: boolean;
  submit: (prompt: string) => void;
  close: () => void;
};

// 选择表格组件，用于展示数据列表并允许选择
const SelectTable: React.FC<SelectTableProps> = ({ data, loading, submit, close }) => {
  const { t } = useTranslation();

  // 点击事件处理函数，接收内容并触发提交和关闭操作
  const handleClick = (content: string) => () => {
    submit(content);
    close();
  };

  return (
    <div
      style={{
        maxHeight: 400,
        overflow: 'auto',
      }}
    >
      {/* 列表组件，展示数据源中的每一项 */}
      <List
        dataSource={data?.data}  // 数据源
        loading={loading}  // 加载状态
        rowKey={(record: any) => record.prompt_name}  // 行标识函数
        renderItem={(item) => (
          <List.Item key={item.prompt_name} onClick={handleClick(item.content)}>
            {/* 提示框，展示具体内容 */}
            <Tooltip title={item.content}>
              <List.Item.Meta
                style={{ cursor: 'copy' }}
                title={item.prompt_name}  // 标题
                // 描述信息，包含场景信息和子场景信息
                description={t('Prompt_Info_Scene') + `：${item.chat_scene}，` + t('Prompt_Info_Sub_Scene') + `：${item.sub_chat_scene}`}
              />
            </Tooltip>
          </List.Item>
        )}
      />
    </div>
  );
};

type PromptBotProps = {
  submit: (prompt: string) => void;
};

// 提示机器人组件，用于管理提示的展示和交互
const PromptBot: React.FC<PromptBotProps> = ({ submit }) => {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);  // 控制弹出框状态
  const [current, setCurrent] = useState('common');  // 当前选择的提示类型

  // 使用请求钩子发送异步请求，获取提示数据
  const { data, loading } = useRequest(
    () => {
      const body = {
        prompt_type: current,  // 请求体，包含当前提示类型
      };
      return sendSpacePostRequest('/prompt/list', body);  // 发送POST请求
    },
    {
      refreshDeps: [current],  // 刷新依赖项
      onError: (err) => {
        message.error(err?.message);  // 错误处理
      },
    },
  );

  // 关闭弹出框
  const close = () => {
    setOpen(false);
  };

  // 处理弹出框打开状态变化
  const handleOpenChange = (newOpen: boolean) => {
    setOpen(newOpen);
  };

  // 处理选择变化
  const handleChange = (value: string) => {
    setCurrent(value);
  };

  return (
    <ConfigProvider
      theme={{
        components: {
          Popover: {
            minWidth: 250,  // 设置弹出框最小宽度
          },
        },
      }}
    // 使用 Popover 组件显示一个弹出框，包含标题、表单项和下拉选择框
    <Popover
      title={
        // 标题部分包含一个表单项，标签为固定文本加国际化字符串，用于选择类型
        <Form.Item label={'Prompt ' + t('Type')}>
          // 下拉选择框，显示当前选项值并绑定 handleChange 函数处理变化事件
          <Select
            style={{ width: 150 }}
            value={current}
            onChange={handleChange}
            // 选项列表包括两个选项，每个包含一个显示文本和一个值
            options={[
              {
                label: t('Public') + ' Prompts',
                value: 'common',
              },
              {
                label: t('Private') + ' Prompts',
                value: 'private',
              },
            ]}
          />
        </Form.Item>
      }
      // 弹出框内容部分使用 SelectTable 组件，传递数据、加载状态、提交函数和关闭函数作为属性
      content={<SelectTable {...{ data, loading, submit, close }} />}
      // 弹出框在右上方显示
      placement="topRight"
      // 触发方式设定为点击触发
      trigger="click"
      // 控制弹出框的打开状态
      open={open}
      // 处理弹出框打开状态变化的函数
      onOpenChange={handleOpenChange}
    >
      // 悬浮按钮使用 Tooltip 组件包裹，显示国际化字符串和 "Prompt" 文本
      <Tooltip title={t('Click_Select') + ' Prompt'}>
        // 自定义悬浮按钮样式，位于页面底部 30% 处
        <FloatButton className="bottom-[30%]" />
      </Tooltip>
    </Popover>
};

export default PromptBot;
```