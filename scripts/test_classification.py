"""分类解析与全量替换的离线回归测试，不访问真实模型或正式数据库。"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import db


class FakeRouter:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)

    def chat(self, messages, max_tokens=1500):
        return next(self.responses)


def favorite(aweme_id: str, title: str) -> dict:
    return {
        "aweme_id": aweme_id, "title": title, "tags": "[]", "author": "tester",
        "share_url": "", "play_url": "", "fav_time": 0,
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        db.DB_PATH = Path(directory) / "classification-test.db"
        from app import web

        db.save_favorites([favorite("1001", "Python 入门"), favorite("1002", "晚餐做法")])
        old_categories = ["旧知识", "旧生活", "旧娱乐", "其他"]
        db.set_categories({"1001": "旧知识", "1002": "旧生活"})
        db.set_meta("categories", json.dumps(old_categories, ensure_ascii=False))

        wrapped = '<think>{"draft": true}</think>\n```json\n{"1001":"知识学习","1002":"生活日常"}\n```'
        rows = db.get_classification_rows(10, only_unclassified=False)
        parsed = web._parse_category_mapping(
            wrapped, rows, ["知识学习", "生活日常", "休闲娱乐", "其他"],
        )
        assert parsed == {"1001": "知识学习", "1002": "生活日常"}
        category_text = (
            '<think>["错误一类","错误二类","错误三类","其他"]</think>\n'
            '["知识学习","生活日常","休闲娱乐","其他"]'
        )
        assert web._normalize_categories(web._extract_json_value(category_text, list)) == [
            "知识学习", "生活日常", "休闲娱乐", "其他",
        ]

        failing = FakeRouter([
            '["知识学习","生活日常","休闲娱乐","其他"]', "", "not json",
        ])
        try:
            web.run_classification(
                failing, reclassify_all=True, progress=lambda **fields: None,
                should_stop=lambda: False,
            )
            raise AssertionError("invalid model responses should fail")
        except RuntimeError:
            pass
        assert db.get_video("1001")["category"] == "旧知识"
        assert db.get_video("1002")["category"] == "旧生活"
        assert json.loads(db.get_meta("categories")) == old_categories

        successful = FakeRouter([
            '分类建议：\n["知识学习","生活日常","休闲娱乐","其他"]', wrapped,
        ])
        done = web.run_classification(
            successful, reclassify_all=True, progress=lambda **fields: None,
            should_stop=lambda: False,
        )
        assert done == 2
        assert db.get_video("1001")["category"] == "知识学习"
        assert db.get_video("1002")["category"] == "生活日常"

    print("Classification tests: 4/4 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
