"""分类解析与全量替换的离线回归测试，不访问真实模型或正式数据库。"""
from __future__ import annotations

import json
import re
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

    def chat_stream(self, messages, on_delta, max_tokens=1500):
        response = next(self.responses)
        if response:
            on_delta(response)
        return response


class DeadRouter:
    """始终失败的模型池替身，用于验证熔断与降级行为。"""

    def __init__(self):
        self.calls = 0

    def _fail(self):
        self.calls += 1
        raise RuntimeError("模型返回了空内容")

    def chat(self, messages, max_tokens=1500):
        return self._fail()

    def chat_stream(self, messages, on_delta, max_tokens=1500):
        return self._fail()


class ScriptedRouter:
    """按提示内容判定批次的替身：命中 fail_ids 的那批三档全失败，其余批正常。

    提示文本每行是「id|标题|作者|标签」，按行解析出本批 ID 再生成答案，
    不依赖调用顺序，避免响应错位导致测试假失败。
    """

    def __init__(self, fail_ids: set[str]):
        self.fail_ids = fail_ids
        self.attempt = 0

    def _answer(self, messages) -> str:
        text = messages[-1]["content"]
        ids = re.findall(r"^(\d{4,})\|", text, flags=re.MULTILINE)
        if not ids:
            raise AssertionError("提示中未找到视频 ID")
        if self.fail_ids.intersection(ids):
            self.attempt += 1
            return ["", "not json", "still not json"][min(self.attempt - 1, 2)]
        return "{" + ",".join(f'"{i}":"知识学习"' for i in ids) + "}"

    def chat(self, messages, max_tokens=1500):
        return self._answer(messages)

    def chat_stream(self, messages, on_delta, max_tokens=1500):
        return self._answer(messages)


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

        # —— 场景 1：唯一一批彻底失败 → 降级比例 100% 超阈值 → 必须中止保留旧分类。
        failing = ScriptedRouter(fail_ids={"1001"})
        try:
            web.run_classification(
                failing, reclassify_all=True, progress=lambda **fields: None,
                should_stop=lambda: False,
            )
            raise AssertionError("invalid model responses should fail")
        except RuntimeError as exc:
            assert "保留原有分类" in str(exc), str(exc)
        assert db.get_video("1001")["category"] == "旧知识"
        assert db.get_video("1002")["category"] == "旧生活"
        assert json.loads(db.get_meta("categories")) == old_categories

        # —— 场景 2：多批中偶发一批失败 → 降级为「其他」继续，不该整任务报废。
        # 6 批里失败 1 批 ≈ 16.7%，低于 20% 阈值。
        extra = [favorite(f"20{i:02d}", f"视频{i}") for i in range(200, 330)]
        db.save_favorites(extra)
        total = len(extra) + 2
        assert total > 125, total

        degraded = ScriptedRouter(fail_ids={"1001"})
        done = web.run_classification(
            degraded, reclassify_all=True, progress=lambda **fields: None,
            should_stop=lambda: False,
        )
        assert done == total, (done, total)
        assert web._degraded_count() == 25, web._degraded_count()
        # 被降级的那批是前 25 条（含 1001），应归入「其他」；其余批保持正常分类
        assert db.get_video("1001")["category"] == "其他"
        assert db.get_video("1002")["category"] == "其他"
        assert db.get_video("20299")["category"] == "知识学习"

        # —— 场景 3：模型整体不可用 → 降级比例迅速超阈值 → 熔断保留旧分类
        db.set_categories({"1001": "旧知识", "1002": "旧生活"})
        db.set_meta("categories", json.dumps(old_categories, ensure_ascii=False))
        dead = DeadRouter()
        try:
            web.run_classification(
                dead, reclassify_all=True, progress=lambda **fields: None,
                should_stop=lambda: False,
            )
            raise AssertionError("降级比例超阈值必须熔断")
        except RuntimeError as exc:
            assert "保留原有分类" in str(exc), str(exc)
        assert db.get_video("1001")["category"] == "旧知识"
        assert db.get_video("1002")["category"] == "旧生活"
        assert json.loads(db.get_meta("categories")) == old_categories
        # 每批 3 档尝试，25 条/批：第 6 批累计 150/332 ≈ 45% 超阈值时熔断
        assert dead.calls < total, dead.calls

        # —— 场景 4：全部批次成功 → 正常原子替换（无降级）
        db.set_categories({"1001": "旧知识", "1002": "旧生活"})
        db.set_meta("categories", json.dumps(old_categories, ensure_ascii=False))
        healthy = ScriptedRouter(fail_ids=set())
        done = web.run_classification(
            healthy, reclassify_all=True, progress=lambda **fields: None,
            should_stop=lambda: False,
        )
        assert done == total, (done, total)
        assert web._degraded_count() == 0
        assert db.get_video("1001")["category"] == "知识学习"
        assert db.get_video("1002")["category"] == "知识学习"
        assert db.get_video("20299")["category"] == "知识学习"

    print("Classification tests: 6/6 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
