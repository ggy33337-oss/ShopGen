# 基于多模态大模型的电商海报生成与文案协作系统

一个面向电商运营和视觉设计场景的 AI 图文生成系统。系统支持商品文案生成、图片生成、基于上一张图继续修改、参考资料解析、海报生成和本地会话记忆，目标是把“电商图片 + 文案 + 多轮修改”做成一条可演示、可落地的完整业务链路。

## 项目亮点

- 多模型协同：将意图识别、文案生成、多模态分析、图片生成、图片编辑拆成独立职责，避免所有逻辑堆在一个模型调用里。
- 会话级记忆：使用 SQLite 持久化会话、消息和绘画历史，支持“上一张图”“刚才那版”“继续优化”等上下文表达。
- 图片连续编辑：支持基于最近生成图片继续换背景、换色调、局部调整，并在模型超时或断连时使用兜底编辑提示词继续链路。
- Schema 驱动海报生成：上传参考资料后，先抽取结构化海报信息，再生成文案和图片提示词，降低模块耦合。
- 工程化落地：项目按 `api / services / poster / llm / infra / core / static` 分层，包含接口 DTO、日志、数据库、文件解析和前端交互。

## 核心功能

| 功能 | 说明 |
| --- | --- |
| 普通聊天 | 支持电商运营建议、商品卖点、标题和营销文案生成 |
| 图片生成 | 根据用户需求生成图片提示词，再调用图片模型生成图片 |
| 图文混合 | 同时返回用户可读文案和生成图片 |
| 图片连续修改 | 基于当前会话最近图片继续换背景、换颜色或优化风格 |
| 参考资料上传 | 支持 `png`、`jpg`、`jpeg`、`pdf`、`docx` |
| 粘贴图片 | 支持在输入框中 `Ctrl + V` 粘贴图片作为本次参考图 |
| 海报生成 | 使用 Poster Schema 串联分析、文案、Prompt 组装和生图 |
| 会话管理 | 支持新建、切换、搜索、删除会话 |
| 实时耗时 | 前端在生成中实时显示耗时，最终只展示耗时，不暴露模型名 |

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 前端 | HTML、CSS、原生 JavaScript |
| 后端 | FastAPI、Uvicorn |
| 数据校验 | Pydantic |
| 数据存储 | SQLite |
| 文档解析 | pypdf、python-docx |
| 日志 | JSONL |
| 模型接口 | OpenAI Compatible API、阿里云百炼兼容接口 |
| 接口文档 | FastAPI OpenAPI / Swagger UI |

## 系统架构

```text
浏览器前端
  ↓
FastAPI 路由层
  ↓
业务服务层 services
  ↓
模型编排层 llm / poster
  ↓
SQLite 会话存储 + JSONL 日志
  ↓
外部文本模型 / 多模态模型 / 图片模型
```

## 普通图文生成流程

```text
用户输入
  ↓
读取 SQLite 会话历史和最近绘画历史
  ↓
意图识别：text / image / mixed
  ↓
构造上下文：系统提示词 + 历史消息 + 绘画历史 + 当前输入
  ↓
文本模型生成文案或 image_prompt
  ↓
如需图片，调用图片生成或图片编辑接口
  ↓
返回结果，并写入 messages / visual_history
```

## 海报生成流程

```text
用户文本 + 可选参考资料
  ↓
File Parser 解析图片、PDF、DOCX
  ↓
MultiModal Analyzer 分析参考资料和用户需求
  ↓
Schema Builder 生成 Poster Schema
  ↓
Copywriting Agent 生成标题、副标题、CTA 和结构化图片提示词
  ↓
Prompt Builder 组装 final_prompt
  ↓
Image Generator 调用图片模型
  ↓
Response Builder 返回海报文案、图片和元数据
```

## 关键工程设计

### 1. 为什么要保存 visual_history

电商图片生成不是单轮任务。用户经常会说：

```text
把上面的图片换成蓝色背景
刚才那张更高级一点
上一版主体不变，只改色调
```

因此系统单独设计了 `visual_history`，保存图片 URL、用户需求、助手说明和图片提示词。这样后续修改可以选择最近图片作为参考，而不是重新从零生成，降低跑偏概率。

### 2. 为什么使用 Poster Schema

海报生成链路包含分析、文案、Prompt 和生图多个步骤。如果每一步直接传自然语言，很容易耦合和失控。项目使用 Poster Schema 作为中间协议，把商品、风格、版式、配色、卖点、人群等信息结构化，方便后续模块稳定消费。

### 3. 如何处理模型超时和断连

图片生成和图片编辑通常耗时较长，外部模型服务可能出现 timeout 或连接重置。系统在图片编辑场景中增加了兜底策略：

- 文本模型超时后，不直接让整条链路失败。
- 如果当前需求是基于上一张图修改，则使用本地兜底编辑提示词。
- 继续调用图片编辑接口，尽量完成用户的背景色、色调等修改需求。
- 对配置错误、模型不存在、鉴权失败等非临时错误，不伪装成成功。

### 4. 为什么前端不显示模型名

模型名称属于内部实现细节。面向用户时只展示生成结果和耗时，避免用户被模型名干扰。后端普通聊天接口也不再返回 `model` 字段。

## 数据库设计

当前使用 SQLite，数据库文件为：

```text
data/app.db
```

核心表：

| 表 | 作用 |
| --- | --- |
| conversations | 保存会话 ID、标题、创建时间、更新时间 |
| messages | 保存用户和助手消息，以及助手消息关联图片 |
| visual_history | 保存生成图片、图片提示词和后续编辑所需上下文 |

`visual_history` 当前最多保留最近 20 条记录；基于上一张图修改时默认读取最近 3 条绘画历史参与判断。

## 后端模块结构

```text
api/                       接口层：路由和 DTO
services/                  服务层：聊天、图片编辑、海报生成编排
poster/                    海报领域模块：解析、分析、Schema、文案、Prompt、响应
llm/                       模型调用和普通聊天 Prompt 构造
infra/                     SQLite 存储和日志
core/                      配置读取
static/                    前端页面、样式和交互逻辑
```

## 接口说明

FastAPI 自动生成接口文档：

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/openapi.json
```

核心接口：

| 接口 | 方法 | 说明 |
| --- | --- | --- |
| `/api/chat` | POST | 普通聊天、文案生成、图片生成、图片编辑 |
| `/api/poster/generate` | POST | 上传参考资料并生成海报 |
| `/api/conversations` | GET | 获取会话列表 |
| `/api/conversations` | POST | 新建会话 |
| `/api/conversations/{conversation_id}` | GET | 获取会话详情 |
| `/api/conversations/{conversation_id}` | DELETE | 删除会话 |

普通聊天响应示例：

```json
{
  "text": "已根据你的需求处理。",
  "image_url": "https://example.com/image.png",
  "latency_ms": 12345,
  "conversation_id": "default"
}
```

## 本地运行

安装依赖：

```bash
pip install -r requirements.txt
```

启动服务：

```bash
python main.py
```

访问地址：

```text
http://127.0.0.1:8000
```

指定端口：

```bash
python main.py --host 127.0.0.1 --port 8000
```

命令行模式：

```bash
python main.py --cli
```

## 环境变量

`.env` 中配置模型接口和密钥。示例：

```env
OPENAI_API_KEY="your_text_model_key"
IMAGE_API_KEY="your_image_model_key"
DASHSCOPE_API_KEY="your_dashscope_key"
MODEL_PROXY_URL=""
OPENAI_BASE_URL="https://example.com/v1/chat/completions"
IMAGE_BASE_URL="https://example.com/v1/images/generations"
MODEL_NAME="your_text_model"
IMAGE_MODEL_NAME="your_image_model"
INTENT_MODEL_NAME="qwen-plus"
POSTER_ANALYZER_MODEL_NAME="qwen-vl-plus"
```

安全说明：

- `.env` 不提交 Git。
- `data/` 和 `logs/` 是本地运行产物。
- 对外部署时建议使用服务器环境变量或密钥管理服务。

## 部署建议

当前版本适合单机部署：

```text
Nginx / HTTPS
  ↓
FastAPI + Uvicorn
  ↓
SQLite + 本地日志
  ↓
外部模型服务
```

后续多人使用时建议升级：

- SQLite 升级为 PostgreSQL。
- 上传文件和生成图片接入 OSS/COS/S3。
- 引入 Redis 保存任务状态和短期缓存。
- 生图和图片编辑改成异步任务队列。
- 增加用户系统、作品库、调用次数统计和成本统计。

## 项目边界

- 当前没有用户登录系统，会话通过本地 `conversation_id` 区分。
- 当前没有对象存储，图片主要依赖外部模型返回 URL。
- 上传参考资料只服务于本次生成，不作为长期知识库保存。
- 当前没有异步队列，长耗时生图请求仍是同步等待。
- 图片批量编辑需要用户明确指定单张图片，避免系统误选。

## 可扩展方向

- 用户账号和个人作品库。
- 品牌资产库：Logo、字体、品牌色、商品图。
- 海报模板管理。
- 图片批量生成和批量编辑。
- 异步任务进度条。
- 历史作品搜索。
- 多模型供应商容灾切换。
- 成本统计和限流策略。

