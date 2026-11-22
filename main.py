"""Mini NPU simulator. Python 3.8+, standard library only."""
import argparse
import json
import math
import re
import time
from pathlib import Path

EPSILON = 1e-9


def normalize_label(value):
    labels = {'+': 'Cross', 'cross': 'Cross', 'x': 'X'}
    if not isinstance(value, str) or value.strip().lower() not in labels:
        raise ValueError('알 수 없는 라벨: {!r}'.format(value))
    return labels[value.strip().lower()]


def validate_matrix(matrix, n):
    if not isinstance(matrix, list) or len(matrix) != n:
        raise ValueError('행 수 불일치: {}행 필요'.format(n))
    for row in matrix:
        if not isinstance(row, list) or len(row) != n:
            raise ValueError('열 수 불일치: {}열 필요'.format(n))
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError('유한한 숫자만 허용')
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError('유한한 숫자만 허용')
    return matrix


def mac(pattern, kernel):
    """Validated square matrices; one multiply and accumulate per cell."""
    total = 0.0
    try:
        for i in range(len(pattern)):
            for j in range(len(pattern)):
                total += pattern[i][j] * kernel[i][j]
    except OverflowError as exc:
        raise ValueError('MAC 정수 연산 결과가 float의 유한 범위를 초과함') from exc
    if not math.isfinite(total):
        raise ValueError('MAC 결과가 유한 범위를 초과함')
    return total


def flatten(matrix):
    return [value for row in matrix for value in row]


def mac_flat(pattern, kernel):
    total = 0.0
    try:
        for i in range(len(pattern)):
            total += pattern[i] * kernel[i]
    except OverflowError as exc:
        raise ValueError('MAC 정수 연산 결과가 float의 유한 범위를 초과함') from exc
    if not math.isfinite(total):
        raise ValueError('MAC 결과가 유한 범위를 초과함')
    return total


def decide(a, b, labels=('Cross', 'X')):
    if abs(a - b) < EPSILON:
        return 'UNDECIDED'
    return labels[0] if a > b else labels[1]


def generate_patterns(n):
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError('크기는 양의 정수여야 합니다')
    # Even sizes use the center row/column at index n // 2.
    cross = [[float(i == n // 2 or j == n // 2) for j in range(n)] for i in range(n)]
    x = [[float(i == j or i + j == n - 1) for j in range(n)] for i in range(n)]
    return cross, x


def measure(function, pattern, kernel, repeats):
    function(pattern, kernel)  # warmup, excluded
    elapsed = 0
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function(pattern, kernel)
        elapsed += time.perf_counter_ns() - start
    return elapsed / repeats / 1_000_000


def benchmark(sizes, repeats=1000, samples=None):
    print('\n성능 분석: 평균/{}회, 단일 MAC 호출, 변환 및 I/O 제외'.format(repeats))
    print('크기 | 평균 시간(ms) 2D | 평균 시간(ms) 1D | 연산 횟수(N²) | 2D/1D')
    for n in sizes:
        p, k = samples[n] if samples and n in samples else generate_patterns(n)
        fp, fk = flatten(p), flatten(k)
        if mac(p, k) != mac_flat(fp, fk):
            raise ValueError('최적화 전후 점수 불일치')
        a = measure(mac, p, k, repeats)
        b = measure(mac_flat, fp, fk, repeats)
        print('{}×{} | {:.6f} | {:.6f} | {} | {:.3f}'.format(n, n, a, b, n*n, a/b if b else 0))


def read_matrix(name, n=3):
    print('{} ({}줄, 각 줄 {}개 숫자)'.format(name, n, n))
    matrix = []
    while len(matrix) < n:
        try:
            row = [float(v) for v in input('{}행: '.format(len(matrix)+1)).split()]
            validate_matrix([row] * n, n)
        except ValueError:
            print('입력 형식 오류: 각 줄에 {}개의 유한한 숫자를 공백으로 구분해 입력하세요.'.format(n))
            continue
        matrix.append(row)
    return matrix


def interactive(repeats):
    a = read_matrix('필터 A')
    b = read_matrix('필터 B')
    print('필터 저장 확인: A={}, B={}'.format(a, b))
    p = read_matrix('패턴')
    sa, sb = mac(p, a), mac(p, b)
    print('A 점수: {:.17g}\nB 점수: {:.17g}'.format(sa, sb))
    verdict = decide(sa, sb, ('A', 'B'))
    print('판정: {}'.format('판정 불가 (UNDECIDED)' if verdict == 'UNDECIDED' else verdict))
    benchmark([3], repeats, {3: (p, a)})


def analyze(document):
    if not isinstance(document, dict) or not isinstance(document.get('patterns'), dict):
        raise ValueError('최상위 객체와 patterns 객체가 필요합니다')
    filters = document.get('filters', {})
    results = []
    print('필터 로드 → 패턴 검증 및 분석')
    for key, case in document['patterns'].items():
        result = {'id': key, 'passed': False}
        try:
            match = re.fullmatch(r'size_([1-9]\d*)_([1-9]\d*)', key)
            if not match:
                raise ValueError('패턴 키 형식 오류: size_N_idx 필요')
            n = int(match.group(1))
            if not isinstance(filters, dict):
                raise ValueError('filters 객체 필요')
            raw = filters.get('size_{}'.format(n))
            if not isinstance(raw, dict):
                raise ValueError('해당 크기의 필터 객체 없음')
            kernels = {}
            for label, matrix in raw.items():
                normalized = normalize_label(label)
                if normalized in kernels:
                    raise ValueError('중복된 표준 필터 라벨')
                kernels[normalized] = validate_matrix(matrix, n)
            if set(kernels) != {'Cross', 'X'}:
                raise ValueError('Cross/X 필터 모두 필요')
            if not isinstance(case, dict):
                raise ValueError('케이스 객체 필요')
            expected = normalize_label(case.get('expected'))
            p = validate_matrix(case.get('input'), n)
            a, b = mac(p, kernels['Cross']), mac(p, kernels['X'])
            verdict = decide(a, b)
            result.update(cross=a, x=b, verdict=verdict, expected=expected, passed=verdict == expected)
            if not result['passed']:
                result['reason'] = 'epsilon 동점 규칙' if verdict == 'UNDECIDED' else '점수 우위와 expected 불일치'
            print('{}: Cross={:.17g}, X={:.17g}, 판정={} | expected={} | {}'.format(key, a, b, verdict, expected, 'PASS' if result['passed'] else 'FAIL'))
        except ValueError as exc:
            result['reason'] = str(exc)
            print('{}: FAIL | {}'.format(key, exc))
        results.append(result)
    return results


def summarize(results):
    passed = sum(r['passed'] for r in results)
    print('\n총 테스트: {}개 | 통과: {}개 | 실패: {}개'.format(len(results), passed, len(results)-passed))
    for r in results:
        if not r['passed']:
            print('- {}: {}'.format(r['id'], r['reason']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['1', '2'])
    parser.add_argument('--data', type=Path, default=Path(__file__).with_name('data.json'))
    parser.add_argument('--repeats', type=int, default=1000)
    parser.add_argument('--generate', type=int, metavar='N', help='Cross/X 생성 JSON 출력 후 종료')
    args = parser.parse_args()
    if args.repeats < 10:
        parser.error('--repeats는 최소 10입니다')
    try:
        if args.generate is not None:
            c, x = generate_patterns(args.generate)
            print(json.dumps({'cross': c, 'x': x}, ensure_ascii=False, indent=2))
            return 0
        print('=== Mini NPU Simulator ===')
        mode = args.mode
        while mode not in ('1', '2'):
            mode = input('1. 사용자 입력 (3×3)\n2. data.json 분석\n선택: ').strip()
            if mode not in ('1', '2'):
                print('1 또는 2를 입력하세요.')
        if mode == '1':
            interactive(args.repeats)
        else:
            document = json.loads(args.data.read_text(encoding='utf-8-sig'))
            results = analyze(document)
            benchmark([3, 5, 13, 25], args.repeats)
            summarize(results)
        return 0
    except (OSError, ValueError, EOFError, KeyboardInterrupt) as exc:
        print('실행 종료: {}'.format(exc))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
