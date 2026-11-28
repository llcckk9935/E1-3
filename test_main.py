"""Mini NPU 시뮬레이터(main.py)의 기능을 자동으로 검사하는 테스트입니다.

실행: python -m unittest -v
"""

import contextlib  # print 출력을 가로채 화면을 깔끔하게 유지
import copy  # 데이터를 통째로 복사(원본을 바꾸지 않기 위해)
import io
import json
import subprocess  # main.py를 실제 프로그램처럼 실행
import sys
import tempfile  # 테스트용 임시 폴더
import unittest  # 파이썬 기본 테스트 도구
from pathlib import Path

import main  # 검사할 대상 프로그램


class SimulatorTests(unittest.TestCase):
    """MAC 계산·판정·입력 검사·오류 처리를 확인하는 테스트 묶음입니다."""

    def test_integer_overflow_isolated_in_batch(self):
        """계산이 넘쳐도 그 패턴만 실패하고 나머지는 계속되는지 봅니다."""
        # 10의 200제곱처럼 엄청 큰 수로 5×5 행렬을 만듭니다.
        huge = [[10**200] * 5 for _ in range(5)]
        # 2D/1D MAC 모두 넘침을 알기 쉬운 오류로 알려 주는지 확인합니다.
        for function, pattern in [
            (main.mac, huge),
            (main.mac_flat, main.flatten(huge)),
        ]:
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(ValueError, "MAC.*유한 범위"):
                    function(pattern, pattern)

        # 넘치는 패턴 1개 + 정상 패턴 1개가 섞인 문서를 만듭니다.
        cross, x = main.generate_patterns(3)
        document = {
            "filters": {
                "size_5": {"cross": huge, "x": huge},
                "size_3": {"cross": cross, "x": x},
            },
            "patterns": {
                "size_5_1": {"input": huge, "expected": "+"},
                "size_3_1": {"input": cross, "expected": "+"},
            },
        }
        with contextlib.redirect_stdout(io.StringIO()):
            results = main.analyze(document)
        self.assertFalse(results[0]["passed"])  # 넘친 패턴은 실패
        self.assertIn("MAC", results[0]["reason"])
        self.assertTrue(results[1]["passed"])  # 정상 패턴은 통과
        # 같은 문서를 파일로 저장한 뒤 실제 프로그램으로도 실행해 봅니다.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overflow.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            run = subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    "main.py",
                    "--mode",
                    "2",
                    "--data",
                    str(path),
                    "--repeats",
                    "10",
                ],
                capture_output=True,
                encoding="utf-8",
            )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("size_5_1: FAIL | MAC", run.stdout)
        self.assertRegex(run.stdout, r"size_3_1:.*PASS")
        self.assertIn("총 테스트: 2개 | 통과: 1개 | 실패: 1개", run.stdout)
        # 프로그램이 죽을 때 나오는 Traceback이 없어야 합니다.
        self.assertNotIn("Traceback", run.stderr)

    def test_scores_and_flatten(self):
        """MAC 점수가 계산대로 나오고 2D·1D 결과가 같은지 확인합니다."""
        for n in (3, 5, 13, 25):
            c, x = main.generate_patterns(n)
            # Cross끼리 곱하면 1인 칸 수(가로 n + 세로 n - 겹친 1칸)
            self.assertEqual(main.mac(c, c), 2 * n - 1)
            # Cross와 X는 가운데 한 칸만 겹칩니다.
            self.assertEqual(main.mac(c, x), 1)
            self.assertEqual(
                main.mac(c, x), main.mac_flat(main.flatten(c), main.flatten(x))
            )

    def test_epsilon_and_labels(self):
        """아주 작은 차이는 동점 처리하고 라벨을 통일하는지 확인합니다."""
        self.assertEqual(main.decide(0.9, 0.8999999999999999), "UNDECIDED")
        self.assertEqual(main.decide(2e-9, 0), "Cross")
        for value, expected in [
            ("+", "Cross"),
            ("cross", "Cross"),
            ("x", "X"),
            ("X", "X"),
        ]:
            self.assertEqual(main.normalize_label(value), expected)
        with self.assertRaises(ValueError):
            main.normalize_label("?")

    def test_schema_and_nonfinite(self):
        """크기가 틀리거나 NaN·True·글자가 든 행렬을 거부하는지 확인합니다."""
        for matrix in (
            [[1]],  # 크기가 1×1
            [[1] * 3] * 2,  # 행이 2개뿐
            [[float("nan")] * 3] * 3,  # 숫자가 아닌 값(NaN)
            [[True] * 3] * 3,  # True/False
            [["x"] * 3] * 3,  # 글자
        ):
            with self.assertRaises(ValueError):
                main.validate_matrix(matrix, 3)

    def test_original_and_isolation(self):
        """data.json 결과와, 잘못된 패턴이 남에게 영향 없는지 봅니다."""
        doc = json.loads(Path("data.json").read_text(encoding="utf-8-sig"))
        with contextlib.redirect_stdout(io.StringIO()):
            results = main.analyze(doc)
        # 원본 데이터에서 실패해야 하는 패턴 3개를 확인합니다.
        self.assertEqual(
            [r["id"] for r in results if not r["passed"]],
            ["size_5_1", "size_13_2", "size_25_1"],
        )
        # 복사본에 일부러 잘못된 입력과 이상한 키를 넣어 봅니다.
        bad = copy.deepcopy(doc)
        bad["patterns"]["size_5_1"]["input"] = [[1]]
        bad["patterns"]["bad_key"] = {}
        with contextlib.redirect_stdout(io.StringIO()):
            results = main.analyze(bad)
        self.assertEqual(len(results), 7)
        self.assertIn("행 수", results[0]["reason"])
        self.assertTrue(results[1]["passed"])

    def test_console_retry_and_eof(self):
        """잘못 입력하면 다시 묻고, 입력이 끊기면 코드 1로 끝나는지 봅니다."""
        # 필터 A, 필터 B, 패턴 순서로 3줄씩 총 9줄을 입력합니다.
        rows = (
            "0 1 0\n1 1 1\n0 1 0\n1 0 1\n0 1 0\n1 0 1\n0 1 0\n1 1 1\n0 1 0\n"
        )
        run = subprocess.run(
            [
                sys.executable,
                "-X",
                "utf8",
                "main.py",
                "--mode",
                "1",
                "--repeats",
                "10",
            ],
            # 앞의 "bad", "1 2"는 일부러 틀린 입력입니다.
            input="bad\n1 2\n" + rows,
            capture_output=True,
            encoding="utf-8",
        )
        self.assertEqual(run.returncode, 0)
        self.assertEqual(run.stdout.count("입력 형식 오류"), 2)
        self.assertIn("판정: A", run.stdout)
        # 아무것도 입력하지 않으면(EOF) 종료 코드 1이어야 합니다.
        run = subprocess.run(
            [sys.executable, "main.py"], input="", capture_output=True
        )
        self.assertEqual(run.returncode, 1)


# python test_main.py 로 직접 실행할 때 모든 테스트를 돌립니다.
if __name__ == "__main__":
    unittest.main()
