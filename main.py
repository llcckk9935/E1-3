"""Mini NPU simulator. Python 3.8+, standard library only.

이 프로그램은 AI 칩(NPU)이 그림을 알아보는 방식을 아주 작게 흉내 냅니다.
- 패턴(입력 그림)과 필터(기준 그림)를 숫자 격자(행렬)로 표현합니다.
- 같은 위치의 숫자끼리 곱해서 모두 더하는 MAC 연산으로 점수를 냅니다.
- 점수가 더 높은 필터 쪽으로 'Cross(+)'인지 'X'인지 판정합니다.

실행 예: python main.py --mode 2   (data.json 분석)
"""

import argparse  # 명령줄 옵션(--mode 등) 읽기
import json  # data.json 읽기 / 결과를 JSON으로 출력
import math  # 숫자가 유한한지(무한대·NaN 아님) 검사
import re  # 정규식으로 패턴 이름(size_5_1 등) 검사
import time  # 연산 시간 측정
from pathlib import Path  # 파일 경로 다루기

# 두 점수의 차이가 이 값보다 작으면 '같은 점수'로 봅니다.
# (실수 계산은 0.1+0.2 처럼 아주 작은 오차가 생길 수 있기 때문입니다.)
EPSILON = 1e-9


def normalize_label(value):
    """'+', 'cross', 'x' 등의 표기를 'Cross'/'X'로 통일합니다."""
    # 왼쪽(입력 표기)을 오른쪽(표준 이름)으로 바꾸는 표입니다.
    labels = {"+": "Cross", "cross": "Cross", "x": "X"}
    # 소문자로 바꾸고 공백을 지운 뒤 표에 없으면 오류를 냅니다.
    if not isinstance(value, str) or value.strip().lower() not in labels:
        raise ValueError("알 수 없는 라벨: {!r}".format(value))
    return labels[value.strip().lower()]


def validate_matrix(matrix, n):
    """행렬이 n×n 크기이고 모든 값이 유한한 숫자인지 검사하는 기능입니다.

    문제가 없으면 받은 행렬을 그대로 돌려주고, 문제가 있으면
    ValueError로 무엇이 잘못됐는지 알려 줍니다.
    """
    # 행(가로줄)의 개수가 n인지 확인합니다.
    if not isinstance(matrix, list) or len(matrix) != n:
        raise ValueError("행 수 불일치: {}행 필요".format(n))
    for row in matrix:
        # 각 행의 열(칸) 개수가 n인지 확인합니다.
        if not isinstance(row, list) or len(row) != n:
            raise ValueError("열 수 불일치: {}열 필요".format(n))
        for value in row:
            # True/False는 숫자로 취급되지만 여기서는 허용하지 않습니다.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("유한한 숫자만 허용")
            # 무한대(inf)나 NaN(숫자 아님)을 걸러 냅니다.
            # 아주 큰 정수는 검사 중 OverflowError가 날 수 있어 대비합니다.
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError("유한한 숫자만 허용")
    return matrix


def mac(pattern, kernel):
    """Validated square matrices; one multiply and accumulate per cell.

    2차원(격자 모양) 행렬로 MAC 연산을 하는 기능입니다.
    같은 칸(i행 j열)의 숫자끼리 곱하고, 그 값을 모두 더해 점수를 만듭니다.
    """
    total = 0.0  # 곱한 값을 누적(더해 나갈) 변수
    try:
        # 이중 반복문: i는 행 번호, j는 열 번호
        for i in range(len(pattern)):
            for j in range(len(pattern)):
                total += pattern[i][j] * kernel[i][j]
    except OverflowError as exc:
        # 숫자가 너무 커서 계산할 수 없으면 이해하기 쉬운 오류로 바꿉니다.
        raise ValueError(
            "MAC 정수 연산 결과가 float의 유한 범위를 초과함"
        ) from exc
    # 결과가 무한대가 되면 올바른 점수가 아니므로 오류 처리합니다.
    if not math.isfinite(total):
        raise ValueError("MAC 결과가 유한 범위를 초과함")
    return total


def flatten(matrix):
    """2차원 행렬을 한 줄짜리 1차원 리스트로 펼치는 기능입니다.

    예: [[1, 2], [3, 4]] → [1, 2, 3, 4]
    """
    return [value for row in matrix for value in row]


def mac_flat(pattern, kernel):
    """1차원(한 줄로 펼친) 리스트로 MAC 연산을 하는 기능입니다.

    mac()과 결과는 같지만 반복문이 하나라서 더 빠른지 비교하는 데 씁니다.
    """
    total = 0.0
    try:
        for i in range(len(pattern)):
            total += pattern[i] * kernel[i]
    except OverflowError as exc:
        raise ValueError(
            "MAC 정수 연산 결과가 float의 유한 범위를 초과함"
        ) from exc
    if not math.isfinite(total):
        raise ValueError("MAC 결과가 유한 범위를 초과함")
    return total


def decide(a, b, labels=("Cross", "X")):
    """두 점수 a, b를 비교해 더 높은 쪽의 이름을 돌려주는 판정 기능입니다.

    차이가 EPSILON보다 작으면 판정할 수 없다는 뜻으로 'UNDECIDED'를 돌려줍니다.
    """
    if abs(a - b) < EPSILON:
        return "UNDECIDED"
    return labels[0] if a > b else labels[1]


def generate_patterns(n):
    """n×n 크기의 Cross(+) 모양과 X 모양 행렬을 자동으로 만드는 기능입니다.

    해당 모양이 지나가는 칸은 1.0, 나머지 칸은 0.0입니다.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("크기는 양의 정수여야 합니다")
    # Even sizes use the center row/column at index n // 2.
    # Cross: 가운데 행(i == n // 2) 또는 가운데 열(j == n // 2)이면 1
    cross = [
        [float(i == n // 2 or j == n // 2) for j in range(n)] for i in range(n)
    ]
    # X: 왼쪽 위→오른쪽 아래 대각선(i == j) 또는
    #    오른쪽 위→왼쪽 아래 대각선(i + j == n - 1)이면 1
    x = [[float(i == j or i + j == n - 1) for j in range(n)] for i in range(n)]
    return cross, x


def measure(function, pattern, kernel, repeats):
    """함수를 repeats번 실행해 1회 평균 시간(ms)을 재는 기능입니다."""
    # 첫 실행은 준비 과정 때문에 느릴 수 있어 측정에서 뺍니다(워밍업).
    function(pattern, kernel)  # warmup, excluded
    elapsed = 0  # 총 걸린 시간(나노초)
    for _ in range(repeats):
        start = time.perf_counter_ns()  # 시작 시각
        function(pattern, kernel)
        elapsed += time.perf_counter_ns() - start  # 걸린 시간 누적
    # 평균(나노초)을 밀리초(ms)로 바꿉니다. 1ms = 1,000,000ns
    return elapsed / repeats / 1_000_000


def benchmark(sizes, repeats=1000, samples=None):
    """여러 크기에서 2D MAC과 1D MAC의 속도를 비교해 표로 출력하는 기능입니다.

    samples가 주어지면 그 행렬을, 없으면 자동 생성한 패턴을 사용합니다.
    """
    print(
        "\n성능 분석: 평균/{}회, 단일 MAC 호출, 변환 및 I/O 제외".format(
            repeats
        )
    )
    print("크기 | 평균 시간(ms) 2D | 평균 시간(ms) 1D | 연산 횟수(N²) | 2D/1D")
    for n in sizes:
        # 측정에 쓸 패턴(p)과 필터(k)를 준비합니다.
        p, k = samples[n] if samples and n in samples else generate_patterns(n)
        # 1D 버전용으로 미리 펼쳐 둡니다(변환 시간은 측정에서 제외).
        fp, fk = flatten(p), flatten(k)
        # 두 방식의 계산 결과가 같은지 먼저 확인합니다.
        if mac(p, k) != mac_flat(fp, fk):
            raise ValueError("최적화 전후 점수 불일치")
        a = measure(mac, p, k, repeats)  # 2D 평균 시간
        b = measure(mac_flat, fp, fk, repeats)  # 1D 평균 시간
        # 크기, 각 시간, 곱셈 횟수(n²), 2D가 1D보다 몇 배 느린지 출력
        print(
            "{}×{} | {:.6f} | {:.6f} | {} | {:.3f}".format(
                n, n, a, b, n * n, a / b if b else 0
            )
        )


def read_matrix(name, n=3):
    """사용자에게 n줄의 숫자를 입력받아 n×n 행렬을 만드는 기능입니다.

    한 줄에 숫자가 n개가 아니거나 숫자가 아니면 그 줄만 다시 입력받습니다.
    """
    print("{} ({}줄, 각 줄 {}개 숫자)".format(name, n, n))
    matrix = []
    while len(matrix) < n:
        try:
            # "0 1 0" 처럼 공백으로 나눠 입력한 글자를 실수 리스트로 바꿉니다.
            row = [
                float(v)
                for v in input("{}행: ".format(len(matrix) + 1)).split()
            ]
            # 이 한 줄을 n번 복제한 행렬로 검사해 줄 길이·값을 확인합니다.
            validate_matrix([row] * n, n)
        except ValueError:
            print(
                (
                    "입력 형식 오류: 각 줄에 {}개의 유한한 숫자를 "
                    "공백으로 구분해 입력하세요."
                ).format(n)
            )
            continue  # 같은 줄을 다시 입력받습니다.
        matrix.append(row)
    return matrix


def interactive(repeats):
    """모드 1: 필터 A·B와 패턴을 직접 입력해 판정해 보는 기능입니다."""
    a = read_matrix("필터 A")
    b = read_matrix("필터 B")
    print("필터 저장 확인: A={}, B={}".format(a, b))
    p = read_matrix("패턴")
    # 패턴과 각 필터의 MAC 점수를 계산합니다.
    sa, sb = mac(p, a), mac(p, b)
    # {:.17g}: 실수를 반올림 없이 충분히 정밀하게 보여 줍니다.
    print("A 점수: {:.17g}\nB 점수: {:.17g}".format(sa, sb))
    verdict = decide(sa, sb, ("A", "B"))
    print(
        "판정: {}".format(
            "판정 불가 (UNDECIDED)" if verdict == "UNDECIDED" else verdict
        )
    )
    # 입력한 3×3 행렬로 성능도 측정해 보여 줍니다.
    benchmark([3], repeats, {3: (p, a)})


def analyze(document):
    """모드 2: data.json의 모든 패턴을 필터와 비교해 채점하는 기능입니다.

    한 패턴에 오류가 있어도 그 패턴만 FAIL로 기록하고
    나머지 패턴은 계속 검사합니다. 결과 목록을 돌려줍니다.
    """
    # 파일의 가장 바깥이 객체이고 그 안에 patterns 객체가 있어야 합니다.
    if not isinstance(document, dict) or not isinstance(
        document.get("patterns"), dict
    ):
        raise ValueError("최상위 객체와 patterns 객체가 필요합니다")
    filters = document.get("filters", {})
    results = []
    print("필터 로드 → 패턴 검증 및 분석")
    # key 예: "size_5_1" (5×5 크기의 1번 패턴), case: 입력과 기대 답
    for key, case in document["patterns"].items():
        result = {"id": key, "passed": False}
        try:
            # 이름이 size_숫자_숫자 형식인지 확인하고 크기(n)를 꺼냅니다.
            match = re.fullmatch(r"size_([1-9]\d*)_([1-9]\d*)", key)
            if not match:
                raise ValueError("패턴 키 형식 오류: size_N_idx 필요")
            n = int(match.group(1))
            # 같은 크기의 필터 묶음(size_n)을 찾습니다.
            if not isinstance(filters, dict):
                raise ValueError("filters 객체 필요")
            raw = filters.get("size_{}".format(n))
            if not isinstance(raw, dict):
                raise ValueError("해당 크기의 필터 객체 없음")
            # 필터 이름을 표준(Cross/X)으로 바꾸고 행렬을 검사합니다.
            kernels = {}
            for label, matrix in raw.items():
                normalized = normalize_label(label)
                if normalized in kernels:
                    raise ValueError("중복된 표준 필터 라벨")
                kernels[normalized] = validate_matrix(matrix, n)
            # Cross와 X 필터가 둘 다 있어야 비교할 수 있습니다.
            if set(kernels) != {"Cross", "X"}:
                raise ValueError("Cross/X 필터 모두 필요")
            if not isinstance(case, dict):
                raise ValueError("케이스 객체 필요")
            # 정답(expected)과 입력 패턴을 꺼내 검사합니다.
            expected = normalize_label(case.get("expected"))
            p = validate_matrix(case.get("input"), n)
            # 두 필터와의 점수를 계산하고 판정합니다.
            a, b = mac(p, kernels["Cross"]), mac(p, kernels["X"])
            verdict = decide(a, b)
            result.update(
                cross=a,
                x=b,
                verdict=verdict,
                expected=expected,
                passed=verdict == expected,
            )
            # 틀렸다면 그 이유(동점인지, 점수가 반대인지)를 남깁니다.
            if not result["passed"]:
                result["reason"] = (
                    "epsilon 동점 규칙"
                    if verdict == "UNDECIDED"
                    else "점수 우위와 expected 불일치"
                )
            print(
                (
                    "{}: Cross={:.17g}, X={:.17g}, 판정={} | expected={} | {}"
                ).format(
                    key,
                    a,
                    b,
                    verdict,
                    expected,
                    "PASS" if result["passed"] else "FAIL",
                )
            )
        except ValueError as exc:
            # 이 패턴만 실패로 기록하고 다음 패턴으로 넘어갑니다.
            result["reason"] = str(exc)
            print("{}: FAIL | {}".format(key, exc))
        results.append(result)
    return results


def summarize(results):
    """전체·통과·실패 개수와 실패한 패턴의 이유를 요약 출력하는 기능입니다."""
    # True는 1, False는 0으로 더해지므로 통과 개수가 됩니다.
    passed = sum(r["passed"] for r in results)
    print(
        "\n총 테스트: {}개 | 통과: {}개 | 실패: {}개".format(
            len(results), passed, len(results) - passed
        )
    )
    for r in results:
        if not r["passed"]:
            print("- {}: {}".format(r["id"], r["reason"]))


def main():
    """명령줄 옵션을 읽고 모드 1 또는 2를 실행하는 프로그램 시작 기능입니다.

    정상 종료면 0, 오류로 끝나면 1을 돌려줍니다.
    """
    # --- 명령줄 옵션 정의 ---
    parser = argparse.ArgumentParser(description=__doc__)
    # --mode 1: 직접 입력, --mode 2: data.json 분석 (생략하면 물어봄)
    parser.add_argument("--mode", choices=["1", "2"])
    # --data: 분석할 JSON 파일 경로(기본값: main.py 옆의 data.json)
    parser.add_argument(
        "--data", type=Path, default=Path(__file__).with_name("data.json")
    )
    # --repeats: 성능 측정 반복 횟수
    parser.add_argument("--repeats", type=int, default=1000)
    # --generate N: N×N Cross/X 패턴을 만들어 출력만 하고 끝냄
    parser.add_argument(
        "--generate",
        type=int,
        metavar="N",
        help="Cross/X 생성 JSON 출력 후 종료",
    )
    args = parser.parse_args()
    # 반복 횟수가 너무 적으면 측정이 부정확하므로 막습니다.
    if args.repeats < 10:
        parser.error("--repeats는 최소 10입니다")
    try:
        if args.generate is not None:
            c, x = generate_patterns(args.generate)
            print(
                json.dumps({"cross": c, "x": x}, ensure_ascii=False, indent=2)
            )
            return 0
        print("=== Mini NPU Simulator ===")
        mode = args.mode
        # 모드를 지정하지 않았으면 1 또는 2가 들어올 때까지 묻습니다.
        while mode not in ("1", "2"):
            mode = input(
                "1. 사용자 입력 (3×3)\n2. data.json 분석\n선택: "
            ).strip()
            if mode not in ("1", "2"):
                print("1 또는 2를 입력하세요.")
        if mode == "1":
            interactive(args.repeats)
        else:
            # utf-8-sig: 파일 맨 앞의 BOM 표시가 있어도 읽을 수 있게 합니다.
            document = json.loads(args.data.read_text(encoding="utf-8-sig"))
            results = analyze(document)
            benchmark([3, 5, 13, 25], args.repeats)
            summarize(results)
        return 0
    except (OSError, ValueError, EOFError, KeyboardInterrupt) as exc:
        # 파일 없음·잘못된 값·입력 중단 등은 오류 화면 대신 안내만 합니다.
        print("실행 종료: {}".format(exc))
        return 1


# 직접 실행했을 때만 main()을 부르고, 그 반환값(0/1)을 종료 코드로 씁니다.
if __name__ == "__main__":
    raise SystemExit(main())
