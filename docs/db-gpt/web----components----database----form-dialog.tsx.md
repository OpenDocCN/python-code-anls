# `.\DB-GPT-src\web\components\database\form-dialog.tsx`

```py
/* eslint-disable react-hooks/exhaustive-deps */
// 从 antd 库中导入必要的组件和方法
import { Button, Form, Input, InputNumber, Modal, Select, message } from 'antd';
// 从 react 库中导入 useEffect, useMemo, useState 方法
import { useEffect, useMemo, useState } from 'react';
// 从 '@/client/api' 中导入相关的 API 方法
import { apiInterceptors, postDbAdd, postDbEdit, postDbTestConnect } from '@/client/api';
// 从 '@/types/db' 中导入类型定义
import { DBOption, DBType, DbListResponse, PostDbParams } from '@/types/db';
// 从 '@/pages/database' 中导入 isFileDb 方法
import { isFileDb } from '@/pages/database';
// 从 'react-i18next' 中导入 useTranslation 方法
import { useTranslation } from 'react-i18next';

// 定义 DBItem 类型，表示数据库项
type DBItem = DbListResponse[0];

// 定义 Props 接口，包含组件需要的各种属性
interface Props {
  dbTypeList: DBOption[]; // 数据库类型列表
  open: boolean; // 对话框是否打开
  choiceDBType?: DBType; // 选定的数据库类型
  editValue?: DBItem; // 编辑中的数据库项
  dbNames: string[]; // 数据库名称列表
  onSuccess?: () => void; // 操作成功时的回调函数
  onClose?: () => void; // 关闭对话框时的回调函数
}

// FormDialog 组件，用于显示数据库表单对话框
function FormDialog({ open, choiceDBType, dbTypeList, editValue, dbNames, onClose, onSuccess }: Props) {
  // 定义 loading 状态，用于显示加载状态
  const [loading, setLoading] = useState(false);
  // 使用 useTranslation 获取翻译函数 t
  const { t } = useTranslation();
  // 使用 Form.useForm 创建表单实例 form
  const [form] = Form.useForm<DBItem>();
  // 使用 Form.useWatch 监听 db_type 字段的变化，并将结果存储在 dbType 变量中
  const dbType = Form.useWatch('db_type', form);

  // 使用 useMemo 计算 fileDb 变量，判断是否为文件型数据库
  const fileDb = useMemo(() => isFileDb(dbTypeList, dbType), [dbTypeList, dbType]);

  // useEffect 用于在组件加载后设置表单的初始值，根据 choiceDBType
  useEffect(() => {
    if (choiceDBType) {
      form.setFieldValue('db_type', choiceDBType);
    }
  }, [choiceDBType]);

  // useEffect 用于在编辑值发生变化时更新表单的字段值
  useEffect(() => {
    if (editValue) {
      form.setFieldsValue({ ...editValue });
    }
  }, [editValue]);

  // useEffect 用于在对话框关闭时重置表单的字段值
  useEffect(() => {
    if (!open) {
      form.resetFields();
    }
  }, [open]);

  // onFinish 函数处理表单提交事件
  const onFinish = async (val: DBItem) => {
    // 解构表单数据，获取必要的参数
    const { db_host, db_path, db_port, ...params } = val;
    // 如果不是编辑模式且数据库名称已存在于列表中，则提示错误信息并返回
    if (!editValue && dbNames.some((item) => item === params.db_name)) {
      message.error('The database already exists!');
      return;
    }
    // 构造 PostDbParams 对象，根据文件型数据库和非文件型数据库分别设置参数
    const data: PostDbParams = {
      db_host: fileDb ? undefined : db_host,
      db_port: fileDb ? undefined : db_port,
      file_path: fileDb ? db_path : undefined,
      ...params,
    };
    // 设置加载状态为 true
    setLoading(true);
    try {
      // 发起测试数据库连接的 API 请求，并处理返回的错误
      const [testErr] = await apiInterceptors(postDbTestConnect(data));
      if (testErr) return;
      // 根据是否编辑状态选择调用添加或编辑数据库的 API 请求，并处理返回的错误
      const [err] = await apiInterceptors((editValue ? postDbEdit : postDbAdd)(data));
      if (err) {
        message.error(err.message);
        return;
      }
      // 显示操作成功的消息提示，并执行成功回调函数
      message.success('success');
      onSuccess?.();
    } catch (e: any) {
      // 捕获异常并显示错误消息
      message.error(e.message);
    } finally {
      // 最终设置加载状态为 false
      setLoading(false);
    }
  };

  // 使用 useMemo 计算 lockDBType 变量，用于锁定数据库类型选择
  const lockDBType = useMemo(() => !!editValue || !!choiceDBType, [editValue, choiceDBType]);

  // 返回组件 JSX 结构
  return (
    <Modal open={open} width={400} title={editValue ? t('Edit') : t('create_database')} maskClosable={false} footer={null} onCancel={onClose}>
      {/* Modal组件，用于显示一个模态对话框，根据open状态控制是否显示，设置宽度为400像素，根据editValue的值显示编辑或创建数据库的标题 */}
      <Form form={form} className="pt-2" labelCol={{ span: 6 }} labelAlign="left" onFinish={onFinish}>
        {/* 表单组件，使用form对象来管理表单数据，定义表单样式为pt-2，设置标签列布局为每行标签占据6个栅格，标签左对齐，定义表单提交成功后的回调函数为onFinish */}
        <Form.Item name="db_type" label="DB Type" className="mb-3" rules={[{ required: true }]}>
          {/* 数据库类型选择框，name属性指定字段名为db_type，标签显示为"DB Type"，设置样式为mb-3，设置必填规则 */}
          <Select aria-readonly={lockDBType} disabled={lockDBType} options={dbTypeList} />
          {/* 下拉选择框，根据lockDBType属性设置是否只读和禁用状态，选项列表来源于dbTypeList */}
        </Form.Item>
        <Form.Item name="db_name" label="DB Name" className="mb-3" rules={[{ required: true }]}>
          {/* 数据库名称输入框，name属性指定字段名为db_name，标签显示为"DB Name"，设置样式为mb-3，设置必填规则 */}
          <Input readOnly={!!editValue} disabled={!!editValue} />
          {/* 输入框，根据editValue属性设置只读和禁用状态 */}
        </Form.Item>
        {fileDb === true && (
          /* 如果fileDb为true，则显示以下内容 */
          <Form.Item name="db_path" label="Path" className="mb-3" rules={[{ required: true }]}>
            {/* 路径输入框，name属性指定字段名为db_path，标签显示为"Path"，设置样式为mb-3，设置必填规则 */}
            <Input />
            {/* 输入框用于输入路径 */}
          </Form.Item>
        )}
        {fileDb === false && (
          /* 如果fileDb为false，则显示以下内容 */
          <>
            <Form.Item name="db_user" label="Username" className="mb-3" rules={[{ required: true }]}>
              {/* 用户名输入框，name属性指定字段名为db_user，标签显示为"Username"，设置样式为mb-3，设置必填规则 */}
              <Input />
              {/* 输入框用于输入用户名 */}
            </Form.Item>
            <Form.Item name="db_pwd" label="Password" className="mb-3" rules={[{ required: false }]}>
              {/* 密码输入框，name属性指定字段名为db_pwd，标签显示为"Password"，设置样式为mb-3，设置非必填规则 */}
              <Input type="password" />
              {/* 输入框用于输入密码，类型为密码 */}
            </Form.Item>
            <Form.Item name="db_host" label="Host" className="mb-3" rules={[{ required: true }]}>
              {/* 主机地址输入框，name属性指定字段名为db_host，标签显示为"Host"，设置样式为mb-3，设置必填规则 */}
              <Input />
              {/* 输入框用于输入主机地址 */}
            </Form.Item>
            <Form.Item name="db_port" label="Port" className="mb-3" rules={[{ required: true }]}>
              {/* 端口号输入框，name属性指定字段名为db_port，标签显示为"Port"，设置样式为mb-3，设置必填规则 */}
              <InputNumber min={1} step={1} max={65535} />
              {/* 数字输入框，用于输入端口号，设置最小值为1，步长为1，最大值为65535 */}
            </Form.Item>
          </>
        )}

        <Form.Item name="comment" label="Remark" className="mb-3">
          {/* 备注输入框，name属性指定字段名为comment，标签显示为"Remark"，设置样式为mb-3 */}
          <Input />
          {/* 输入框用于输入备注信息 */}
        </Form.Item>
        <Form.Item className="flex flex-row-reverse pt-1 mb-0">
          {/* 表单项，设置样式为右侧弹性布局和顶部填充，底部边距为0 */}
          <Button htmlType="submit" type="primary" size="middle" className="mr-1" loading={loading}>
            {/* 提交按钮，设置类型为primary，大小为middle，设置样式为mr-1，根据loading属性设置加载状态 */}
            Save
          </Button>
          <Button size="middle" onClick={onClose}>
            {/* 取消按钮，大小为middle，点击事件为onClose */}
            Cancel
          </Button>
        </Form.Item>
      </Form>
    </Modal>
}

export default FormDialog;
```