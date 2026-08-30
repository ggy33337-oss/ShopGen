# -*- coding: utf-8 -*-

import asyncio

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from core.config import read_config


_async_engine: AsyncEngine | None = None
_engine_url = ""
_engine_lock = asyncio.Lock()


def get_database_url(values=None):
    values = values or read_config(".env")
    database_url = str(values.get("ASYNC_DATABASE_URL") or "").strip()
    if not database_url:
        raise RuntimeError("缺少 MySQL 配置：ASYNC_DATABASE_URL")
    if not database_url.startswith("mysql+aiomysql://"):
        raise RuntimeError("ASYNC_DATABASE_URL 必须使用 mysql+aiomysql:// 驱动。")
    return database_url


async def get_async_engine(values=None):
    global _async_engine, _engine_url

    values = values or read_config(".env")
    database_url = get_database_url(values)
    if _async_engine is not None and _engine_url == database_url:
        return _async_engine
    async with _engine_lock:
        if _async_engine is not None and _engine_url == database_url:
            return _async_engine
        if _async_engine is not None:
            await _async_engine.dispose()
        _async_engine = create_async_engine(
            database_url,
            echo=parse_bool(values.get("MYSQL_ECHO"), default=False),
            pool_size=int(values.get("MYSQL_POOL_SIZE") or 10),
            max_overflow=int(values.get("MYSQL_MAX_OVERFLOW") or 20),
            pool_pre_ping=True,
            pool_recycle=int(values.get("MYSQL_POOL_RECYCLE") or 1800),
            pool_timeout=int(values.get("MYSQL_POOL_TIMEOUT") or 30),
        )
        _engine_url = database_url
        return _async_engine


async def dispose_async_engine():
    global _async_engine, _engine_url

    async with _engine_lock:
        if _async_engine is not None:
            await _async_engine.dispose()
        _async_engine = None
        _engine_url = ""


def parse_bool(value, default=False):
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
