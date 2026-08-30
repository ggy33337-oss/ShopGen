# -*- coding: utf-8 -*-

import argparse
import asyncio
import os
import sys

import uvicorn

from services.chat_service import ask


def run_cli():
    user_input = input("请输入你的问题：")
    response, model_name, latency_ms, conversation_id = asyncio.run(ask(user_input))
    if isinstance(response, dict):
        response = response.get("text", "")
    print("AI助手的回答：", response)


def run_web(host, port):
    uvicorn.run("api.web_app:app", host=host, port=port)


def main():
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("LANG", "en_US.UTF-8")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
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
