# 基于多模态大模型的电商海报生成与文案协作系统

一个面向电商运营和视觉设计场景的 AI 图文生成系统。系统支持商品文案生成、图片生成、基于上一张图继续修改、参考资料解析、海报生成和本地会话记忆，目标是把“电商图片 + 文案 + 多轮修改”做成一条可演示、可落地的完整业务链路。

## 项目亮点

- 单一编排核心：`runtime/orchestrator.py` 统一负责意图判断、链路路由、模型调用和状态轨迹。
- 会话级记忆：使用 MySQL 持久化会话、消息和绘画历史，使用 Redis 保存图片二进制。
- 三链路路由：历史图修改走链路一，本轮上传图走链路二，无图新请求进入链路三占位。
- 文本与意图使用 `qwen3.8-max`，视觉规划使用 `qwen3-vl-plus`，图片生成与编辑支持 `qwen-image-3.0`、Wanx 2.1 和 GPT Image。
- 向量知识库：文档先清理低信息内容并按完整语义边界分块，使用 `qwen3-vl-embedding` 向量化写入 Qdrant；查询时召回 Top 20，再由 `qwen3-rerank` 重排、阈值过滤并最多保留 3 条。
- 显式占位：无参考图的链路三仍返回 `placeholder`，暂不调用生图模型。

## 核心功能

| 功能 | 说明 |
| --- | --- |
| 普通聊天 | 支持电商运营建议、商品卖点、标题和营销文案生成 |
| 图片生成 | 链路一和链路二支持切换 Qwen Image、Wanx 2.1 和 GPT Image |
| 图文混合 | 同时返回用户可读文案和生成图片 |
| 图片连续修改 | 基于当前会话最近图片继续换背景、换颜色或优化风格 |
| 参考资料上传 | 支持 `png`、`jpg`、`jpeg`、`pdf`、`docx` |
| 粘贴图片 | 支持在输入框中 `Ctrl + V` 粘贴图片作为本次参考图 |
| 海报生成 | 上传参考图后进入统一编排器的链路二 |
| 会话管理 | 支持新建、切换、搜索、删除会话 |
| 实时耗时 | 前端在生成中实时显示耗时，最终只展示耗时，不暴露模型名 |

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 前端 | HTML、CSS、原生 JavaScript |
| 后端 | FastAPI、Uvicorn |
| 数据校验 | Pydantic |
| 数据存储 | MySQL 8+、Redis、Qdrant |
| 文档解析 | PyMuPDF、pypdf、python-docx、Pillow |
| 日志 | JSONL |
| 模型接口 | 阿里云百炼千问兼容接口与多模态生成原生接口 |
| 接口文档 | FastAPI OpenAPI / Swagger UI |

## 系统架构

```text
浏览器前端
  ↓
FastAPI 路由层
  ↓
业务服务适配层 services
  ↓
唯一编排核心 runtime/orchestrator.py
  ↓
千问网关 + Qdrant 文本/图片向量库
  ↓
MySQL 会话存储 + Redis 图片缓存 + JSONL 轨迹日志
```

生成任务的逐阶段诊断日志写入 `logs/generation.jsonl`。每条记录都包含统一的 `task_id`，可查看路由选择、知识检索、提示词规划、图片模型每次请求、重试、超时、HTTP 状态、服务端请求编号、缓存及最终失败阶段。请求正文中的 Base64 图片、API Key 和 Authorization 会自动脱敏；`logs/llm.jsonl` 继续保存成功任务摘要。

## 普通图文生成流程

```text
用户输入
  ↓
读取 MySQL 会话历史和最近绘画历史
  ↓
千问意图识别：text / image / mixed + 是否修改历史图
  ↓
历史图修改？是 → 链路一（历史图片 + 当前要求）
  ↓
否：本轮有上传图片？是 → 链路二（上传图片 + 知识库检索上下文）
  ↓
否 → 链路三占位（暂不调用生图模型）
  ↓
千问视觉规划 + Wanx 2.1 图片生成/编辑 → 写入 messages / visual_history
```

## 海报生成流程

```text
用户文本 + 可选参考资料
  ↓
File Parser 解析图片、PDF、DOCX 中的文本和图片
  ↓
统一编排器优先判断是否修改历史图
  ↓
有上传图片时进入链路二
  ↓
知识库向量检索 + 千问视觉规划
  ↓
Wanx 2.1 图片生成/编辑 → 返回海报文案、图片和链路元数据
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

### 2. 为什么使用单一编排器

聊天和海报接口只负责输入适配、会话持久化与响应转换。链路优先级、知识库调用、提示词规划和图片模型调用全部集中在 `runtime/orchestrator.py`，避免两个入口各自维护一套流程。

### 3. 如何处理模型超时和断连

图片生成和图片编辑通常耗时较长。Wanx 2.1 通过异步任务提交与状态轮询获取结果；配置错误、鉴权失败或模型无有效图片输出时明确返回失败，不伪装成成功。

### 4. 链路二如何整合提示词优先级

本轮上传图进入链路二后，视觉规划上下文按固定优先级组织：用户执行目标决定修改内容，上传图片决定未修改部分的视觉事实，业务参数只影响未明确指定的表现方式，知识库只补充空白信息，通用美化和负面约束优先级最低。低优先级信息不得覆盖高优先级信息。

每次上传新参考图都会创建一条新的图片编辑链并生成第 1 版。系统将最初上传图单独保存为根参考图；用户第一次针对结果提出优化时，视觉规划模型会同时参考第 1 版结果与根参考图，用第 1 版结果定位明确的优化方向和范围，但最终图片模型只以根参考图为生成基准，未明确要求的区域按根图保持。如果用户继续针对第 2 版提出第 3 次修改，系统不再调用图片生成模型，直接返回“请重新上传参考图片并仔细规划提示词。”；重新上传图片后修订次数重置。

知识检索不会再无条件采用向量 Top 1。Qdrant 先召回 20 条候选，系统过滤纯编号、乱码和空信息，`qwen3-rerank` 重排后只保留得分最高的 3 条，再使用 `KNOWLEDGE_RERANK_MIN_SCORE` 拒绝低相关结果。所有候选低于阈值时返回 `no_match`，不向视觉规划提示词注入知识。

当前默认重排阈值 `0.55` 是使用项目内置的 10 类电商视觉查询、41 条人工标注候选进行初始校准的结果，不是通用常量。运行 `python -m scripts.calibrate_reranker` 可以使用当前模型和固定任务指令重新测量；上线前仍应使用真实业务反馈扩充标注集并重新校准。

现有索引可以先使用 `python -m scripts.cleanup_knowledge_quality` 干运行审计，再使用 `python -m scripts.cleanup_knowledge_quality --apply` 删除确定无意义的向量点。查询不相关但内容本身有效的资料不会被永久删除，只会在本次检索中被 Reranker 拒绝。

### 5. 为什么前端不显示模型名

模型名称属于内部实现细节。面向用户时只展示生成结果和耗时，避免用户被模型名干扰。后端普通聊天接口也不再返回 `model` 字段。

## 数据库设计

当前使用 MySQL 保存结构化会话数据，数据库名由 `ASYNC_DATABASE_URL` 决定（当前配置为 `news_app`）；图片二进制存储在 Redis，通过 `/api/images/{image_id}` 访问。

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
services/                  输入、会话和响应适配层
runtime/                   唯一编排器、任务状态和链路三占位
knowledge/                 结构化文档解析、语义分块、Qdrant 文本/图片向量库
poster/                    文件解析和海报领域数据模型
llm/                       千问网关和 Prompt 构造
infra/                     MySQL 会话存储、Redis 图片缓存和 JSONL 日志
core/                      配置读取
static/                    前端页面、样式和交互逻辑
```

## 接口说明

FastAPI 自动生成接口文档：

```text
http://127.0.0.1:8002/docs
http://127.0.0.1:8002/openapi.json
```

核心接口：

| 接口 | 方法 | 说明 |
| --- | --- | --- |
| `/api/chat` | POST | 普通聊天、文案生成、图片生成、图片编辑 |
| `/api/poster/generate` | POST | 上传参考资料并提交海报生成任务，立即返回 `task_id` |
| `/api/poster/tasks/{task_id}` | GET | 查询海报任务状态，完成后返回结果 |
| `/api/images/{image_id}` | GET | 从 Redis 读取图片二进制 |
| `/api/knowledge/upload` | POST | 上传文档或图片并写入向量库 |
| `/api/knowledge/search` | POST | 向量化用户请求并检索相关资料 |
| `/api/knowledge/sources` | GET | 查看已入库资料 |
| `/api/conversations` | GET | 获取会话列表 |
| `/api/conversations` | POST | 新建会话 |
| `/api/conversations/{conversation_id}` | GET | 获取会话详情 |
| `/api/conversations/{conversation_id}` | DELETE | 删除会话 |

海报生成接口采用异步任务模式。`POST /api/poster/generate` 在上传完成后立即返回 `task_id`；前端或调用方轮询 `GET /api/poster/tasks/{task_id}`，当 `status` 为 `completed` 时读取 `result`，当 `status` 为 `failed` 时读取 `error`。

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

Windows PowerShell 推荐使用 UTF-8 启动脚本：

```powershell
.\start.ps1
```

访问地址：

```text
http://127.0.0.1:8002
```

指定端口：

```bash
python main.py --host 127.0.0.1 --port 8002
```

命令行模式：

```bash
python main.py --cli
```

## 环境变量

`.env` 中配置模型接口和密钥。示例：

```env
DASHSCOPE_API_KEY="your_dashscope_key"
MODEL_PROXY_URL=""
QWEN_TEXT_MODEL="qwen3.8-max"
QWEN_VL_MODEL="qwen3-vl-plus"
QWEN_IMAGE_MODEL="wanx2.1-imageedit"
QWEN_IMAGE_TOTAL_TIMEOUT="420"
OPENAI_IMAGE_API_KEY="your_openai_compatible_key"
OPENAI_IMAGE_BASE_URL="https://api.openai.com/v1"
OPENAI_IMAGE_MODEL="gpt-image-2"
POSTER_TOTAL_TIMEOUT_SECONDS="480"
QWEN_EMBEDDING_MODEL="qwen3-vl-embedding"
QWEN_EMBEDDING_DIMENSION="1024"
QWEN_EMBEDDING_TIMEOUT="120"
QWEN_EMBEDDING_ATTEMPTS="2"
QWEN_RERANK_MODEL="qwen3-rerank"
QWEN_RERANK_ENDPOINT="https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
QWEN_RERANK_TIMEOUT="60"
QWEN_RERANK_ATTEMPTS="2"
DASHSCOPE_COMPAT_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_API_BASE_URL="https://dashscope.aliyuncs.com/api/v1"
QWEN_IMAGE_SIZE="2048*2048"
ASYNC_DATABASE_URL="mysql+aiomysql://user:password@localhost:3306/news_app?charset=utf8mb4"
MYSQL_ECHO="false"
MYSQL_POOL_SIZE="10"
MYSQL_MAX_OVERFLOW="20"
MYSQL_POOL_RECYCLE="1800"
MYSQL_POOL_TIMEOUT="30"
REDIS_HOST="127.0.0.1"
REDIS_PORT="6379"
REDIS_DB="0"
REDIS_PASSWORD=""
REDIS_IMAGE_TTL_SECONDS="604800"
REDIS_MAX_IMAGE_BYTES="20971520"
REDIS_IMAGE_KEY_PREFIX="ecommerce:image"
QDRANT_URL="http://127.0.0.1:6333"
QDRANT_TRUST_ENV="false"
QDRANT_TEXT_COLLECTION="ecommerce_kb_text"
QDRANT_IMAGE_COLLECTION="ecommerce_kb_image"
KNOWLEDGE_STORAGE_DIR="data/knowledge"
KNOWLEDGE_MAX_FILE_MB="100"
KNOWLEDGE_MAX_DOCUMENT_IMAGES="6"
KNOWLEDGE_MAX_DOCUMENT_PAGES="300"
KNOWLEDGE_MAX_TEXT_CHARACTERS="500000"
KNOWLEDGE_EMBEDDING_WORKERS="3"
KNOWLEDGE_RECALL_LIMIT="20"
KNOWLEDGE_RERANK_TOP_N="3"
KNOWLEDGE_RERANK_MIN_SCORE="0.55"
```

只启动 Redis（需要 Docker Desktop）：

```powershell
docker compose -p ecommerce-assistant -f compose.redis.yml up -d
```

启动 Qdrant（需要 Docker Desktop）：

```powershell
docker compose -p ecommerce-assistant-rag -f compose.qdrant.yml up -d
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
MySQL + Redis + Qdrant + 本地日志
  ↓
外部模型服务
```

后续多人使用时建议升级：

- 为 MySQL 配置主从复制、备份和连接池监控。
- 上传文件和生成图片接入 OSS/COS/S3。
- 使用 Redis Sentinel 或 Cluster 提高图片缓存可用性。
- 将进程内海报任务管理器替换为 Redis/Celery 等持久化异步队列。
- 增加用户系统、作品库、调用次数统计和成本统计。

## 项目边界

- 当前没有用户登录系统，会话通过本地 `conversation_id` 区分。
- 当前没有对象存储，图片缓存在 Redis，并受 `REDIS_IMAGE_TTL_SECONDS` 控制。
- 聊天框上传的参考资料只服务于本次生成；长期资料通过 `/api/knowledge/upload` 单独入库。
- 当前使用进程内任务管理器，服务重启后未完成任务不会恢复；生产环境可替换为 Redis/Celery 等持久化队列。
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

