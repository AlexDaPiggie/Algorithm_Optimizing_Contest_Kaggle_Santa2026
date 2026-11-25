"""Construct targets using cached multiplicative anchors and short offsets."""
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
    emit('/', 0, 0, 0)
    if radix == 1:
        return out
    for digit in range(2, radix + 1):
        if digit % 2 == 0:
            emit('+', digit // 2 - 1, digit // 2 - 1, digit - 1)
        else:
            emit('+', digit - 2, 0, digit - 1)

    def horner(goal, dest, excluded=()):
        digits = []
        q = goal
        best = None
        lookup = {v: i for i, v in enumerate(regs)
                  if v is not None and i not in excluded}
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
        return best

    cursor = 0
    for target in sorted(p['targets']):
        if target not in remaining:
            continue
        # The final register is a dedicated temporary; other registers cache
        # previous targets. The digit table remains resident throughout.
        dest = radix + cursor % (47 - radix)
        cursor += 1
        scratch = 47
        best = horner(target, dest)
        lookup = {v: i for i, v in enumerate(regs) if v is not None}
        for anchor, value in enumerate(regs):
            if value is None or anchor == dest:
                continue
            if value < target:
                plan = horner(target - value, dest)
                plan = plan + [['+', anchor, dest, dest]]
                if len(plan) < len(best):
                    best = plan

            # target = anchor * quotient +/- remainder. Construct the product
            # first and keep it alive while making the much smaller remainder.
            # Only a bounded quotient is needed for the shared/multiplicative
            # relationships; the direct Horner plan handles everything else.
            if anchor == scratch or value <= radix:
                continue
            quotient, remainder = divmod(target, value)
            for scale in (quotient, quotient + 1):
                if not 2 <= scale <= 65536:
                    continue
                product = value * scale
                if product > limit:
                    continue
                gap = abs(target - product)
                if scale in lookup and lookup[scale] != dest:
                    plan = [['*', anchor, lookup[scale], dest]]
                else:
                    plan = horner(scale, dest)
                    plan = plan + [['*', anchor, dest, dest]]
                if gap:
                    # dest now holds the product; its old value cannot be used
                    # to construct the offset. Scratch is otherwise disposable.
                    offset = horner(gap, scratch, excluded=(dest,))
                    plan += offset
                    plan += [['+' if product < target else '-', dest, scratch, dest]]
                if len(plan) < len(best):
                    best = plan

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
