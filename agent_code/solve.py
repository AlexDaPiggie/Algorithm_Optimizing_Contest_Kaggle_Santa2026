"""Gap construction with a guaranteed positive Horner fallback."""
import csv
import json
import time
from pathlib import Path
from kaggle_agent import toolkit as kt


def calc(op, a, b, limit):
    if op == '+': v = a + b
    elif op == '-': v = a - b
    elif op == '*': v = a * b
    elif op == '/' and b and a % b == 0: v = a // b
    else: return None
    return v if 0 < v <= limit else None


def replay(p, program):
    if not isinstance(program, list) or len(program) > 4096:
        return False
    if len(json.dumps(program, separators=(',', ':')).encode('utf-8')) > 131072:
        return False
    regs = list(p['start']) + [None] * (48 - len(p['start']))
    remaining = set(p['targets']) - set(p['start'])
    for ins in program:
        if not isinstance(ins, list) or len(ins) != 4:
            return False
        op, i, j, d = ins
        if op not in ('+', '-', '*', '/'):
            return False
        if any(type(r) is not int or not 0 <= r < 48 for r in (i, j, d)):
            return False
        if regs[i] is None or regs[j] is None:
            return False
        value = calc(op, regs[i], regs[j], p['limit'])
        if value is None:
            return False
        regs[d] = value
        remaining.discard(value)
    return not remaining


def construct(p, radix):
    """Keep radix digits resident and construct gaps or targets in one register.

    Every Horner intermediate is a positive prefix of its goal, hence within
    value_limit. Starting at the leading digit (ultimately made from 1) makes
    the plan defined even when no existing value is a useful stepping stone.
    """
    limit = p['limit']
    radix = min(radix, limit)
    regs = list(p['start']) + [None] * (48 - len(p['start']))
    remaining = set(p['targets']) - set(p['start'])
    out = []

    def emit(op, i, j, d):
        value = calc(op, regs[i], regs[j], limit)
        if value is None:
            raise ValueError('invalid construction')
        out.append([op, i, j, d])
        regs[d] = value
        remaining.discard(value)

    if not remaining:
        return out
    # Digit n occupies register n-1. Overwriting a starting value is legal;
    # initial target coverage was recorded before these writes.
    emit('/', 0, 0, 0)
    if radix == 1:
        return out
    for digit in range(2, radix + 1):
        if digit % 2 == 0:
            emit('+', digit // 2 - 1, digit // 2 - 1, digit - 1)
        else:
            emit('+', digit - 2, 0, digit - 1)

    def horner(goal, dest):
        # Any resident exact radix prefix is a possible stepping stone. The
        # leading digit is always resident, providing an unconditional fallback.
        digits = []
        q = goal
        best = None
        lookup = {v: i for i, v in enumerate(regs) if v is not None}
        while q:
            if q in lookup:
                src = lookup[q]
                plan = []
                for digit in reversed(digits):
                    plan.append(['*', src, radix - 1, dest])
                    src = dest
                    if digit:
                        plan.append(['+', dest, digit - 1, dest])
                if not plan and src != dest:
                    plan = [['*', src, 0, dest]]
                if best is None or len(plan) < len(best):
                    best = plan
            q, digit = divmod(q, radix)
            digits.append(digit)
        if best is None:
            raise ValueError('missing leading digit')
        return best

    cursor = 0
    for target in sorted(p['targets']):
        if target not in remaining:
            continue
        dest = radix + cursor % (48 - radix)
        cursor += 1
        best = horner(target, dest)
        # Gaps may be cheaper than independent construction. An anchor must
        # survive the Horner writes, so exclude the destination register.
        for anchor, value in enumerate(regs):
            if value is None or anchor == dest or value >= target:
                continue
            gap = horner(target - value, dest)
            plan = gap + [['+', anchor, dest, dest]]
            if len(plan) < len(best):
                best = plan
        # Preserve exact one-instruction opportunities, including division.
        lookup = {v: i for i, v in enumerate(regs) if v is not None}
        for a, i in lookup.items():
            possibilities = [('+', target - a, False),
                             ('-', a - target, False),
                             ('-', a + target, True),
                             ('*', target // a if target % a == 0 else 0, False),
                             ('/', a // target if a % target == 0 else 0, False),
                             ('/', a * target, True)]
            for op, b, reverse in possibilities:
                if b in lookup and len(best) > 1:
                    j = lookup[b]
                    best = [[op, j, i, dest] if reverse else [op, i, j, dest]]
        for op, i, j, d in best:
            emit(op, i, j, d)
        if len(out) > 4096:
            return None
    return out if not remaining else None


def main():
    o = kt.optimization()
    data = Path(o['data_dir'])

    def locate(name):
        nested = data / 'data' / name
        return nested if nested.is_file() else data / name

    with locate('puzzles.csv').open(newline='', encoding='utf-8') as f:
        puzzles = {r['id']: dict(start=json.loads(r['starting_values']),
                                targets=json.loads(r['targets']),
                                limit=int(r['value_limit'])) for r in csv.DictReader(f)}
    with Path(o['best'] or locate('sample_submission.csv')).open(newline='', encoding='utf-8') as f:
        current = {r['id']: json.loads(r['program']) for r in csv.DictReader(f)}
    deadline = time.monotonic() + (4 if o['smoke'] else 3600)
    ids = sorted(puzzles)[:6] if o['smoke'] else sorted(puzzles)
    for pid in ids:
        old = current[pid]
        best = old
        attempts = []
        for radix in (16, 8, 32, 4, 2):
            if attempts and time.monotonic() > deadline:
                break
            candidate = construct(puzzles[pid], radix)
            valid = candidate is not None and replay(puzzles[pid], candidate)
            attempts.append((radix, len(candidate) if candidate is not None else None, valid))
            if valid and len(candidate) < len(best):
                best = candidate
        current[pid] = best
        print(pid, 'before', len(old), 'after', len(best), 'candidates', attempts, flush=True)
    out = Path(o['solution'])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'program'])
        for pid, puzzle in sorted(puzzles.items()):
            if not replay(puzzle, current[pid]):
                raise ValueError('invalid output ' + pid)
            writer.writerow([pid, json.dumps(current[pid], separators=(',', ':'))])


if __name__ == '__main__':
    main()
