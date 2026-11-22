import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import main

class SimulatorTests(unittest.TestCase):
    def test_integer_overflow_isolated_in_batch(self):
        huge = [[10**200] * 5 for _ in range(5)]
        for function, pattern in [(main.mac, huge), (main.mac_flat, main.flatten(huge))]:
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(ValueError, 'MAC.*유한 범위'):
                    function(pattern, pattern)

        cross, x = main.generate_patterns(3)
        document = {
            'filters': {
                'size_5': {'cross': huge, 'x': huge},
                'size_3': {'cross': cross, 'x': x},
            },
            'patterns': {
                'size_5_1': {'input': huge, 'expected': '+'},
                'size_3_1': {'input': cross, 'expected': '+'},
            },
        }
        with contextlib.redirect_stdout(io.StringIO()):
            results = main.analyze(document)
        self.assertFalse(results[0]['passed'])
        self.assertIn('MAC', results[0]['reason'])
        self.assertTrue(results[1]['passed'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'overflow.json'
            path.write_text(json.dumps(document), encoding='utf-8')
            run = subprocess.run(
                [sys.executable, '-X', 'utf8', 'main.py', '--mode', '2',
                 '--data', str(path), '--repeats', '10'],
                capture_output=True, encoding='utf-8',
            )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('size_5_1: FAIL | MAC', run.stdout)
        self.assertRegex(run.stdout, r'size_3_1:.*PASS')
        self.assertIn('총 테스트: 2개 | 통과: 1개 | 실패: 1개', run.stdout)
        self.assertNotIn('Traceback', run.stderr)

    def test_scores_and_flatten(self):
        for n in (3, 5, 13, 25):
            c, x = main.generate_patterns(n)
            self.assertEqual(main.mac(c, c), 2*n-1)
            self.assertEqual(main.mac(c, x), 1)
            self.assertEqual(main.mac(c, x), main.mac_flat(main.flatten(c), main.flatten(x)))

    def test_epsilon_and_labels(self):
        self.assertEqual(main.decide(0.9, 0.8999999999999999), 'UNDECIDED')
        self.assertEqual(main.decide(2e-9, 0), 'Cross')
        for value, expected in [('+','Cross'), ('cross','Cross'), ('x','X'), ('X','X')]:
            self.assertEqual(main.normalize_label(value), expected)
        with self.assertRaises(ValueError):
            main.normalize_label('?')

    def test_schema_and_nonfinite(self):
        for matrix in ([[1]], [[1]*3]*2, [[float('nan')]*3]*3, [[True]*3]*3, [['x']*3]*3):
            with self.assertRaises(ValueError):
                main.validate_matrix(matrix, 3)

    def test_original_and_isolation(self):
        doc = json.loads(Path('data.json').read_text(encoding='utf-8-sig'))
        with contextlib.redirect_stdout(io.StringIO()):
            results = main.analyze(doc)
        self.assertEqual([r['id'] for r in results if not r['passed']], ['size_5_1','size_13_2','size_25_1'])
        bad = copy.deepcopy(doc)
        bad['patterns']['size_5_1']['input'] = [[1]]
        bad['patterns']['bad_key'] = {}
        with contextlib.redirect_stdout(io.StringIO()):
            results = main.analyze(bad)
        self.assertEqual(len(results), 7)
        self.assertIn('행 수', results[0]['reason'])
        self.assertTrue(results[1]['passed'])

    def test_console_retry_and_eof(self):
        rows = '0 1 0\n1 1 1\n0 1 0\n1 0 1\n0 1 0\n1 0 1\n0 1 0\n1 1 1\n0 1 0\n'
        run = subprocess.run([sys.executable,'-X','utf8','main.py','--mode','1','--repeats','10'], input='bad\n1 2\n'+rows, capture_output=True, encoding='utf-8')
        self.assertEqual(run.returncode, 0)
        self.assertEqual(run.stdout.count('입력 형식 오류'), 2)
        self.assertIn('판정: A', run.stdout)
        run = subprocess.run([sys.executable,'main.py'], input='', capture_output=True)
        self.assertEqual(run.returncode, 1)

if __name__ == '__main__':
    unittest.main()
