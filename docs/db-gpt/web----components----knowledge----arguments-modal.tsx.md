# `.\DB-GPT-src\web\components\knowledge\arguments-modal.tsx`

```py
import React, { useEffect, useState } from 'react';
import { Modal, Tabs, Button, Input, Form, Col, Row, Spin } from 'antd';
import { useTranslation } from 'react-i18next';

import { AlertFilled, BookOutlined, FileSearchOutlined } from '@ant-design/icons';
import { apiInterceptors, getArguments, saveArguments } from '@/client/api';
import { IArguments, ISpace } from '@/types/knowledge';

const { TextArea } = Input;

interface IProps {
  space: ISpace; // 接收一个名为 space 的 ISpace 接口类型的参数
  argumentsShow: boolean; // 接收一个布尔类型的参数 argumentsShow，表示是否展示参数
  setArgumentsShow: (argumentsShow: boolean) => void; // 一个函数，用来设置 argumentsShow 的状态
}

export default function ArgumentsModal({ space, argumentsShow, setArgumentsShow }: IProps) {
  const { t } = useTranslation(); // 使用 i18n 库中的 useTranslation 钩子函数获取 t 函数，用于国际化文本

  const [newSpaceArguments, setNewSpaceArguments] = useState<IArguments | null>(); // 状态钩子，用来存储新的 space 参数或者 null
  const [spinning, setSpinning] = useState<boolean>(false); // 状态钩子，用来控制加载中的旋转状态，默认为 false

  // 异步函数，用于从 API 获取参数数据
  const fetchArguments = async () => {
    const [_, data] = await apiInterceptors(getArguments(space.name)); // 调用 apiInterceptors 函数发送请求，获取参数数据
    setNewSpaceArguments(data); // 将获取到的参数数据存储在状态钩子中
  };

  // 当 space.name 发生变化时，调用 fetchArguments 函数重新获取参数数据
  useEffect(() => {
    fetchArguments();
  }, [space.name]);

  // 渲染嵌入表单的函数
  const renderEmbeddingForm = () => {
    return (
      <Row gutter={24}> {/* 使用 Ant Design 的 Row 组件，设置间距为 24 */}
        <Col span={12} offset={0}> {/* Ant Design 的 Col 组件，占据12列，偏移量为0 */}
          <Form.Item<IArguments> tooltip={t(`the_top_k_vectors`)} rules={[{ required: true }]} label={t('topk')} name={['embedding', 'topk']}>
            {/* 表单项，提示信息为 t(`the_top_k_vectors`) 的翻译文本，设置必填规则，标签文本为 t('topk') 的翻译文本，字段名为 ['embedding', 'topk'] */}
            <Input className="mb-5 h-12" /> {/* Ant Design 的输入框组件，样式类名为 mb-5 h-12 */}
          </Form.Item>
        </Col>
        <Col span={12}>
          <Form.Item<IArguments>
            tooltip={t(`Set_a_threshold_score`)}
            rules={[{ required: true }]}
            label={t('recall_score')}
            name={['embedding', 'recall_score']}
          >
            <Input className="mb-5 h-12" placeholder={t('Please_input_the_owner')} />
          </Form.Item>
        </Col>
        <Col span={12}>
          <Form.Item<IArguments> tooltip={t(`recall_type`)} rules={[{ required: true }]} label={t('recall_type')} name={['embedding', 'recall_type']}>
            <Input className="mb-5 h-12" />
          </Form.Item>
        </Col>
        <Col span={12}>
          <Form.Item<IArguments> tooltip={t(`A_model_used`)} rules={[{ required: true }]} label={t('model')} name={['embedding', 'model']}>
            <Input className="mb-5 h-12" />
          </Form.Item>
        </Col>
        <Col span={12}>
          <Form.Item<IArguments>
            tooltip={t(`The_size_of_the_data_chunks`)}
            rules={[{ required: true }]}
            label={t('chunk_size')}
            name={['embedding', 'chunk_size']}
          >
            <Input className="mb-5 h-12" />
          </Form.Item>
        </Col>
        <Col span={12}>
          <Form.Item<IArguments>
            tooltip={t(`The_amount_of_overlap`)}
            rules={[{ required: true }]}
            label={t('chunk_overlap')}
            name={['embedding', 'chunk_overlap']}
          >
            <Input className="mb-5 h-12" placeholder={t('Please_input_the_description')} />
          </Form.Item>
        </Col>
      </Row>
    );
  };

  // 返回组件的 JSX
  return (
    <Modal
      visible={argumentsShow} // 根据 argumentsShow 参数决定是否显示弹窗
      title={t('Embedding Arguments')} // 设置弹窗标题，使用 t 函数获取国际化文本
      onCancel={() => setArgumentsShow(false)} // 点击取消按钮时调用 setArgumentsShow 函数，关闭弹窗
      footer={[
        <Button key="back" onClick={() => setArgumentsShow(false)}> {/* 弹窗底部的取消按钮，点击时调用 setArgumentsShow 函数，关闭弹窗 */}
          {t('Cancel')} {/* 按钮文本使用 t 函数获取国际化文本 */}
        </Button>,
        <Button key="submit" type="primary" onClick={() => saveArguments(space.name, newSpaceArguments)}>{t('Save')}</Button> {/* 弹窗底部的保存按钮，点击时调用 saveArguments 函数保存参数 */}
      ]}
    >
      {renderEmbeddingForm()} {/* 渲染嵌入表单 */}
    </Modal>
  );
}
    const renderPromptForm = () => {
        // 渲染提示表单的函数，返回包含场景、模板和最大令牌数输入框的表单
        return (
          <>
            // 场景输入框，显示4行文本域，带有提示信息
            <Form.Item<IArguments> tooltip={t(`A_contextual_parameter`)} label={t('scene')} name={['prompt', 'scene']}>
              <TextArea rows={4} className="mb-2" />
            </Form.Item>
            // 模板输入框，显示7行文本域，带有提示信息
            <Form.Item<IArguments> tooltip={t(`structure_or_format`)} label={t('template')} name={['prompt', 'template']}>
              <TextArea rows={7} className="mb-2" />
            </Form.Item>
            // 最大令牌数输入框，带有提示信息
            <Form.Item<IArguments> tooltip={t(`The_maximum_number_of_tokens`)} label={t('max_token')} name={['prompt', 'max_token']}>
              <Input className="mb-2" />
            </Form.Item>
          </>
        );
      };
    
      const renderSummary = () => {
        // 渲染摘要表单的函数，返回包含最大迭代次数和并发限制输入框的表单
        return (
          <>
            // 最大迭代次数输入框，必填项
            <Form.Item<IArguments> rules={[{ required: true }]} label={t('max_iteration')} name={['summary', 'max_iteration']}>
              <Input className="mb-2" />
            </Form.Item>
            // 并发限制输入框，必填项
            <Form.Item<IArguments> rules={[{ required: true }]} label={t('concurrency_limit')} name={['summary', 'concurrency_limit']}>
              <Input className="mb-2" />
            </Form.Item>
          </>
        );
      };
    
      const items = [
        {
          key: 'Embedding',
          label: (
            <div>
              <FileSearchOutlined />
              {t('Embedding')}
            </div>
          ),
          children: renderEmbeddingForm(),
        },
        {
          key: 'Prompt',
          label: (
            <div>
              <AlertFilled />
              {t('Prompt')}
            </div>
          ),
          children: renderPromptForm(), // 使用renderPromptForm函数渲染的表单部分
        },
        {
          key: 'Summary',
          label: (
            <div>
              <BookOutlined />
              {t('Summary')}
            </div>
          ),
          children: renderSummary(), // 使用renderSummary函数渲染的表单部分
        },
      ];
    
      const handleSubmit = async (fieldsValue: IArguments) => {
        // 提交表单处理函数，设置加载状态为true
        setSpinning(true);
        // 发送API请求，保存表单数据到数据库
        const [_, data, res] = await apiInterceptors(
          saveArguments(space.name, {
            argument: JSON.stringify(fieldsValue), // 将表单数据转换为JSON字符串格式
          }),
        );
        // 设置加载状态为false
        setSpinning(false);
        // 如果保存成功，隐藏参数框
        res?.success && setArgumentsShow(false);
      };
    
      return (
        <Modal
          width={850}
          open={argumentsShow} // 显示参数框的状态
          onCancel={() => {
            setArgumentsShow(false); // 取消按钮点击事件，隐藏参数框
          }}
          footer={null} // 不显示底部按钮
        >
          <Spin spinning={spinning}> // 加载状态指示器
            <Form
              size="large"
              className="mt-4"
              layout="vertical"
              name="basic"
              initialValues={{ ...newSpaceArguments }} // 表单初始值
              autoComplete="off"
              onFinish={handleSubmit} // 表单提交处理函数
            >
              <Tabs items={items}></Tabs> // 使用Tabs组件展示items数组中的表单项
              <div className="mt-3 mb-3">
                <Button htmlType="submit" type="primary" className="mr-6">
                  {t('Submit')} // 提交按钮，触发表单提交
                </Button>
                <Button
                  onClick={() => {
                    setArgumentsShow(false); // 关闭按钮点击事件，隐藏参数框
                  }}
                >
                  {t('close')} // 关闭按钮文字
                </Button>
              </div>
            </Form>
          </Spin>
        </Modal>
      );
}



# 这行代码关闭了一个代码块。在 Python 中，花括号 {} 通常用于表示代码块的起始和结束，但在示例中没有花括号，因为 Python 使用缩进来表示代码块。这行代码可能是在一种不常见的情况下使用，例如在另一种语言或上下文中。
```