# `.\DB-GPT-src\web\components\knowledge\doc-upload-form.tsx`

```py
import { Button, Form, Input, Upload, Spin, message } from 'antd';
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { InboxOutlined } from '@ant-design/icons';
import { apiInterceptors, addDocument, uploadDocument } from '@/client/api';
import { RcFile, UploadChangeParam } from 'antd/es/upload';
import { File, StepChangeParams } from '@/types/knowledge';
import { UploadRequestOption as RcCustomRequestOptions } from 'rc-upload/lib/interface';
import classNames from 'classnames';

type FileParams = {
  file: RcFile;
  fileList: FileList;
};

type IProps = {
  className: string;
  handleStepChange: (params: StepChangeParams) => void;
  spaceName: string;
  docType: string;
};

type FieldType = {
  docName: string;
  textSource: string;
  originFileObj: FileParams;
  text: string;
  webPageUrl: string;
};

const { Dragger } = Upload;
const { TextArea } = Input;

// React 组件，用于文档上传表单
export default function DocUploadForm(props: IProps) {
  // 解构传入的 props
  const { className, handleStepChange, spaceName, docType } = props;
  // 使用国际化钩子函数
  const { t } = useTranslation();
  // 创建表单实例
  const [form] = Form.useForm();
  // 状态钩子函数，用于显示上传状态的加载动画
  const [spinning, setSpinning] = useState<boolean>(false);
  // 状态钩子函数，用于存储上传的文件列表
  const [files, setFiles] = useState<Array<File>>([]);

  // 处理文档上传的函数
  const upload = async (data: FieldType) => {
    const { docName, textSource, text, webPageUrl } = data;
    let docId;
    // 设置加载动画为 true，表示开始上传
    setSpinning(true);
    switch (docType) {
      // 根据文档类型不同进行不同的上传处理
      case 'URL':
        // 调用 API 发送 URL 类型文档的上传请求
        [, docId] = await apiInterceptors(
          addDocument(spaceName as string, {
            doc_name: docName,
            content: webPageUrl,
            doc_type: 'URL',
          }),
        );
        break;
      case 'TEXT':
        // 调用 API 发送 TEXT 类型文档的上传请求
        [, docId] = await apiInterceptors(
          addDocument(spaceName as string, {
            doc_name: docName,
            source: textSource,
            content: text,
            doc_type: 'TEXT',
          }),
        );
        break;
    }
    // 设置加载动画为 false，表示上传结束
    setSpinning(false);
    // 根据上传结果处理不同的逻辑
    if (docType === 'DOCUMENT' && files.length < 1) {
      // 如果是普通文档类型且文件列表为空，则提示上传失败
      return message.error('Upload failed, please re-upload.');
    } else if (docType !== 'DOCUMENT' && !docId) {
      // 如果不是普通文档类型且文档 ID 不存在，则提示上传失败
      return message.error('Upload failed, please re-upload.');
    }
    // 调用父组件传入的处理步骤变化的函数
    handleStepChange({
      label: 'forward',
      files:
        docType === 'DOCUMENT'
          ? files
          : [
              {
                name: docName,
                doc_id: docId || -1,
              },
            ],
    });
  };

  // 处理文件变化的回调函数
  const handleFileChange = ({ file, fileList }: UploadChangeParam) => {
    // 如果文件列表为空，清空表单中的原始文件对象字段
    if (fileList.length === 0) {
      form.setFieldValue('originFileObj', null);
    }
  };

  // 上传文件的函数
  const uploadFile = async (options: RcCustomRequestOptions) => {
    const { onSuccess, onError, file } = options;
    const formData = new FormData();
    const filename = file?.name;
    // 将文件信息添加到 FormData 对象中
    formData.append('doc_name', filename);
    formData.append('doc_file', file);
    formData.append('doc_type', 'DOCUMENT');
    // 调用 API 发送文件上传请求
    const [, docId] = await apiInterceptors(uploadDocument(spaceName, formData));
    onSuccess?.({}, file);
  };
  if (Number.isInteger(docId)) {
    // 检查 docId 是否为整数，如果是，则执行成功回调并设置文件列表
    onSuccess && onSuccess(docId || 0);
    // 向文件列表添加新文件信息
    setFiles((files) => {
      files.push({
        name: filename,
        doc_id: docId || -1,
      });
      return files;
    });
  } else {
    // 如果 docId 不是整数，则执行错误回调
    onError && onError({ name: '', message: '' });
  }
};

const renderText = () => {
  // 渲染文本输入表单部分
  return (
    <>
      <Form.Item<FieldType> label={`${t('Name')}:`} name="docName" rules={[{ required: true, message: t('Please_input_the_name') }]}>
        <Input className="mb-5 h-12" placeholder={t('Please_input_the_name')} />
      </Form.Item>
      <Form.Item<FieldType>
        label={`${t('Text_Source')}:`}
        name="textSource"
        rules={[{ required: true, message: t('Please_input_the_text_source') }]}
      >
        <Input className="mb-5  h-12" placeholder={t('Please_input_the_text_source')} />
      </Form.Item>
      <Form.Item<FieldType> label={`${t('Text')}:`} name="text" rules={[{ required: true, message: t('Please_input_the_description') }]}>
        <TextArea rows={4} />
      </Form.Item>
    </>
  );
};

const renderWebPage = () => {
  // 渲染网页链接输入表单部分
  return (
    <>
      <Form.Item<FieldType> label={`${t('Name')}:`} name="docName" rules={[{ required: true, message: t('Please_input_the_name') }]}>
        <Input className="mb-5 h-12" placeholder={t('Please_input_the_name')} />
      </Form.Item>
      <Form.Item<FieldType>
        label={`${t('Web_Page_URL')}:`}
        name="webPageUrl"
        rules={[{ required: true, message: t('Please_input_the_Web_Page_URL') }]}
      >
        <Input className="mb-5  h-12" placeholder={t('Please_input_the_Web_Page_URL')} />
      </Form.Item>
    </>
  );
};

const renderDocument = () => {
  // 渲染文件上传表单部分
  return (
    <>
      <Form.Item<FieldType> name="originFileObj" rules={[{ required: true, message: t('Please_select_file') }]}>
        <Dragger
          multiple
          onChange={handleFileChange}
          maxCount={10}
          accept=".pdf,.ppt,.pptx,.xls,.xlsx,.doc,.docx,.txt,.md"
          customRequest={uploadFile}
        >
          <p className="ant-upload-drag-icon">
            <InboxOutlined />
          </p>
          <p style={{ color: 'rgb(22, 108, 255)', fontSize: '20px' }}>{t('Select_or_Drop_file')}</p>
          <p className="ant-upload-hint" style={{ color: 'rgb(22, 108, 255)' }}>
            PDF, PowerPoint, Excel, Word, Text, Markdown,
          </p>
        </Dragger>
      </Form.Item>
    </>
  );
};

const renderFormContainer = () => {
  // 根据不同的文档类型选择渲染对应的表单部分
  switch (docType) {
    case 'URL':
      return renderWebPage();
    case 'DOCUMENT':
      return renderDocument();
    default:
      return renderText();
  }
};

return (
    <Spin spinning={spinning}>
      <Form
        form={form}
        size="large"
        className={classNames('mt-4', className)}
        layout="vertical"
        name="basic"
        initialValues={{ remember: true }}
        autoComplete="off"
        onFinish={upload}
      >
        {renderFormContainer()}
        <Form.Item>
          <Button
            onClick={() => {
              handleStepChange({ label: 'back' });
            }}
            className="mr-4"
          >{`${t('Back')}`}</Button>
          <Button type="primary" loading={spinning} htmlType="submit">
            {t('Next')}
          </Button>
        </Form.Item>
      </Form>
    </Spin>
    
    
    
    # 在一个加载状态的旋转器（Spin）中显示一个表单（Form）
    <Spin spinning={spinning}>
      # 设置表单的各种属性和事件处理程序
      <Form
        form={form}  # 使用指定的表单实例
        size="large"  # 设置表单的大小为大号
        className={classNames('mt-4', className)}  # 添加样式类名，包括' mt-4'和传入的className
        layout="vertical"  # 设置表单布局为垂直布局
        name="basic"  # 设置表单的名称
        initialValues={{ remember: true }}  # 设置表单的初始值，包括记住选项为真
        autoComplete="off"  # 关闭表单的自动完成
        onFinish={upload}  # 设置表单提交成功后的回调函数
      >
        {renderFormContainer()}  # 渲染表单内部的容器，可能是动态生成的表单项
        <Form.Item>  # 表单项组件
          <Button
            onClick={() => {
              handleStepChange({ label: 'back' });  # 点击按钮时触发的函数，用于处理步骤的变化为返回
            }}
            className="mr-4"  # 添加样式类名，包括'mr-4'
          >{`${t('Back')}`}</Button>  # 显示“返回”按钮，按钮文本根据国际化函数t进行翻译
          <Button type="primary" loading={spinning} htmlType="submit">  # 主要的提交按钮，可能处于加载状态
            {t('Next')}  # 显示“下一步”按钮文本，按钮文本根据国际化函数t进行翻译
          </Button>
        </Form.Item>
      </Form>
    </Spin>
}


注释：


# 这行代码表示一个函数或控制语句的结束
```