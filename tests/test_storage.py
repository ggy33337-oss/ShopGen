# -*- coding: utf-8 -*-

import unittest

from infra.conversation_store import SCHEMA_STATEMENTS
from infra.database import get_database_url
from infra.redis_image_cache import RedisImageCache, extract_image_id


class FakePipeline:
    def __init__(self, client):
        self.client = client
        self.commands = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def hset(self, key, mapping):
        self.commands.append(("hset", key, mapping))
        return self

    def expire(self, key, seconds):
        self.commands.append(("expire", key, seconds))
        return self

    def execute(self):
        for command in self.commands:
            if command[0] == "hset":
                _, key, mapping = command
                self.client.values[key] = mapping
        return [True] * len(self.commands)


class FakeRedis:
    def __init__(self):
        self.values = {}

    def pipeline(self, transaction=True):
        self.transaction = transaction
        return FakePipeline(self)

    def hmget(self, key, *fields):
        mapping = self.values.get(key, {})
        values = []
        for field in fields:
            value = mapping.get(field)
            if isinstance(value, str):
                value = value.encode("utf-8")
            values.append(value)
        return values

    def delete(self, *keys):
        deleted = 0
        for key in keys:
            deleted += int(self.values.pop(key, None) is not None)
        return deleted

    def ping(self):
        return True


class MySqlSchemaTests(unittest.TestCase):
    def test_tables_use_utf8mb4_unicode_ci(self):
        ddl = "\n".join(SCHEMA_STATEMENTS)
        self.assertEqual(3, len(SCHEMA_STATEMENTS))
        self.assertEqual(3, ddl.count("DEFAULT CHARSET=utf8mb4"))
        self.assertEqual(3, ddl.count("COLLATE=utf8mb4_unicode_ci"))
        self.assertIn("FOREIGN KEY (conversation_id)", ddl)

    def test_database_url_requires_aiomysql_driver(self):
        url = get_database_url(
            {"ASYNC_DATABASE_URL": "mysql+aiomysql://user:pass@localhost/db?charset=utf8mb4"}
        )
        self.assertTrue(url.startswith("mysql+aiomysql://"))
        with self.assertRaisesRegex(RuntimeError, r"mysql\+aiomysql"):
            get_database_url({"ASYNC_DATABASE_URL": "mysql://localhost/db"})


class RedisImageCacheTests(unittest.TestCase):
    def test_image_round_trip_uses_internal_url(self):
        client = FakeRedis()
        cache = RedisImageCache(
            values={"REDIS_IMAGE_TTL_SECONDS": "60"},
            client=client,
        )

        stored = cache.put_image(b"image-bytes", "image/png")
        loaded = cache.get_image(stored.image_id)

        self.assertTrue(stored.public_url.startswith("/api/images/"))
        self.assertEqual(stored.image_id, extract_image_id(stored.public_url))
        self.assertEqual(b"image-bytes", loaded.content)
        self.assertEqual("image/png", loaded.content_type)

    def test_multiple_images_can_be_deleted_with_one_redis_call(self):
        client = FakeRedis()
        cache = RedisImageCache(values={"REDIS_IMAGE_TTL_SECONDS": "60"}, client=client)
        first = cache.put_image(b"first", "image/png")
        second = cache.put_image(b"second", "image/png")

        deleted = cache.delete_images([first.image_id, second.image_id, "invalid"])

        self.assertEqual(2, deleted)
        self.assertIsNone(cache.get_image(first.image_id))
        self.assertIsNone(cache.get_image(second.image_id))


if __name__ == "__main__":
    unittest.main()
