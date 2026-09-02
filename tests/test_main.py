# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch

import main


class MainConfigurationTests(unittest.TestCase):
    def test_web_server_defaults_to_port_8002(self):
        with patch("sys.argv", ["main.py"]), patch("main.run_web") as mocked_run_web:
            main.main()

        mocked_run_web.assert_called_once_with("127.0.0.1", 8002)

    def test_command_line_port_overrides_default(self):
        with patch("sys.argv", ["main.py", "--port", "9000"]), patch(
            "main.run_web"
        ) as mocked_run_web:
            main.main()

        mocked_run_web.assert_called_once_with("127.0.0.1", 9000)


if __name__ == "__main__":
    unittest.main()
