# 知识库问答助手

基于 LangChain 框架和 RAG（检索增强生成）技术的本地私有知识库问答系统。

## 功能特点

-   多种文档格式支持：PDF / TXT / Markdown / DOCX / CSV
-   本地向量存储：基于 ChromaDB，数据完全本地化
-   中英文双语嵌入：使用 bge-small-zh-v1.5 模型
-   双 LLM 后端：支持 DeepSeek API 和本地 Ollama 模型
-   流式回答：实时生成答案
-   来源引用：每条回答均标明参考文档
-   Web 界面：基于 Streamlit，开箱即用

## 环境要求

-   Python 3.10+
-   至少 4GB 可用内存（建议 8GB 以使用本地模型）
-   （可选）Ollama — 使用本地 LLM 时安装

## 快速开始

### 1. 安装依赖

pip install -r requirements.txt

### 2. 配置 API Key

复制 .env.example 为 .env，填入 DeepSeek API Key：

DEEPSEEK_API_KEY=sk-your-key-here

注册 DeepSeek 开发者账号获取 API Key：platform.deepseek.com

### 3. 放入文档

将需要索引的文档放入 data/documents/ 目录。

### 4. 启动应用

streamlit run src/app.py

### 5. 索引文档

打开浏览器访问 Streamlit 界面，在左侧点击「索引文档」按钮。

### 6. 开始提问

在聊天输入框中输入问题，系统将基于文档内容生成回答。

## 配置说明

编辑 config/settings.yaml 可调整参数。

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| llm.provider | LLM 后端 | deepseek |
| llm.deepseek.model | DeepSeek 模型名 | deepseek-chat |
| llm.ollama.model | Ollama 模型名 | qwen2:7b |
| embedding.model_name | 嵌入模型 | BAAI/bge-small-zh-v1.5 |
| vector_store.top_k | 检索返回文档数 | 5 |
| text_splitter.chunk_size | 文本分块大小 | 512 |

## 切换为 Ollama 本地模型

1. 安装 Ollama（ollama.com）并拉取模型：
   ollama pull qwen2:7b
2. 修改 config/settings.yaml 中 llm.provider 为 ollama
3. 重启应用

## 技术栈

| 组件 | 技术 |
|------|------|
| RAG 框架 | LangChain |
| 嵌入模型 | bge-small-zh-v1.5 |
| 向量数据库 | ChromaDB |
| LLM 后端 | DeepSeek API / Ollama |
| 文档解析 | LangChain Loaders |
| Web 界面 | Streamlit |

## 许可

MIT
