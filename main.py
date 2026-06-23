import argparse

import uvicorn

from services.chat_service import ask


def run_cli():
    user_input = input("请输入你的问题：")
    response, model_name, latency_ms, conversation_id = ask(user_input)
    print("AI助手的回答：", response)


def run_web(host, port):
    uvicorn.run("api.web_app:app", host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description="电商文案与图片生成助手")
    parser.add_argument("--cli", action="store_true", help="使用命令行单轮问答模式")
    parser.add_argument("--host", default="127.0.0.1", help="Web 服务监听地址")
    parser.add_argument("--port", type=int, default=8000, help="Web 服务监听端口")
    args = parser.parse_args()

    if args.cli:
        run_cli()
        return

    run_web(args.host, args.port)


if __name__ == "__main__":
    main()
