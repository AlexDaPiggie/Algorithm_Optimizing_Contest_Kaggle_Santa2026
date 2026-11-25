"""Cached-anchor construction with a bounded E1 x E1 arithmetic frontier."""
import csv
import json
import math
import random
import time
from pathlib import Path
from kaggle_agent import toolkit as kt

DEEP_STATS = {'calls': 0, 'e1xe1': 0, 'plans': 0, 'shared': 0}


def deep_plan(regs, target, limit, bound, dest, protected, deadline):
    """Join E1/E2 with E1/E2/E3; E3 contains operations on two E1s.

    Recipes refer to values rather than expression copies. Compiling their DAG
    evaluates shared intermediates once and keeps them in actual registers.
    """
    if bound <= 1 or time.monotonic() >= deadline:
        return None
    DEEP_STATS['calls'] += 1
    base = {v: i for i, v in enumerate(regs) if v is not None}
    recipe = {}
    cost = {v: 0 for v in base}
    work = {v: frozenset() for v in base}

    def add(v, op, a, b, c):
        if not 0 < v <= limit:
            return
        # Count shared arithmetic nodes once, including x*x and nested reuse.
        if cost[a] and cost[b]:
            c -= len(work[a] & work[b])
        if c < cost.get(v, 99):
            cost[v] = c
            recipe[v] = (op, a, b)
            work[v] = work[a] | work[b] | {v}

    def expand(a, b, c):
        add(a + b, '+', a, b, c)
        add(a * b, '*', a, b, c)
        if a > b:
            add(a - b, '-', a, b, c)
        elif b > a:
            add(b - a, '-', b, a, c)
        if a % b == 0:
            add(a // b, '/', a, b, c)
        if b % a == 0:
            add(b // a, '/', b, a, c)

    values = sorted(base)
    for i, a in enumerate(values):
        for b in values[:i + 1]:
            expand(a, b, 1)
    e1 = tuple(cost)

    def joins(left, right, maximum):
        if target in cost:
            return True
        winner = None
        winner_cost = min(bound, maximum + 1)
        for a in left:
            ca = cost[a]
            if ca + 1 >= winner_cost:
                continue
            candidates = (('+', target - a, False),
                          ('-', a - target, False),
                          ('-', a + target, True),
                          ('*', target // a if target % a == 0 else 0, False),
                          ('/', a // target if a % target == 0 else 0, False),
                          ('/', a * target, True))
            for op, b, reverse in candidates:
                if b in right:
                    c = ca + cost[b] + 1 - len(work[a] & work[b])
                    if c < winner_cost:
                        winner_cost = c
                        winner = (op, b, a) if reverse else (op, a, b)
        if winner is not None:
            recipe[target] = winner
            cost[target] = winner_cost
            work[target] = work[winner[1]] | work[winner[2]] | {target}
        return winner is not None

    # Cheap E1 x E1 join first (at most three operations).
    if target not in cost:
        joins(e1, cost, 3)
    if target not in cost and bound > 3:
        # Include the whole E1 frontier against live registers (E2).
        for i, a in enumerate(e1):
            if i % 64 == 0 and time.monotonic() >= deadline:
                return None
            for b in values:
                expand(a, b, cost[a] + 1)
        e2 = tuple(v for v in cost if cost[v] <= 2)
        joins(e1, cost, 4)
        if target not in cost and bound > 4:
            joins(e2, cost, 5)
        if target not in cost and bound > 4 and time.monotonic() < deadline:
            DEEP_STATS['e1xe1'] += 1
            # Bounded E1 x E1 level, distinct from E1 x register.
            # Keep small factors, values near sqrt(target), and a broad spread.
            ordered = sorted(e1)
            pool = set(ordered[:64])
            root = math.isqrt(target)
            pool.update(sorted(e1, key=lambda v: abs(v - root))[:64])
            stride = max(1, len(ordered) // 128)
            pool.update(ordered[::stride][:128])
            pool = sorted(pool)
            for i, a in enumerate(pool):
                if i % 16 == 0 and time.monotonic() >= deadline:
                    return None
                for b in pool[:i + 1]:
                    expand(a, b, cost[a] + cost[b] + 1)
            joins(e1, cost, 5)
    if target not in cost or cost[target] >= bound:
        return None

    # Topological DAG and original leaves; no recipe ever reads an evicted leaf.
    nodes = []
    seen = set()
    leaves = set()

    def visit(v):
        if v in seen:
            return
        seen.add(v)
        if v in base:
            leaves.add(base[v])
            return
        op, a, b = recipe[v]
        visit(a)
        visit(b)
        nodes.append((v, op, a, b))

    visit(target)
    if not nodes or len(nodes) >= bound:
        return None
    DEEP_STATS['plans'] += 1
    if any(a == b and a not in base for _, _, a, b in nodes):
        DEEP_STATS['shared'] += 1
    available = dict(base)
    allocated = set(protected) | leaves
    plan = []
    # Five operations need at most four intermediate slots. Keep their values
    # after the target, making them anchors for subsequent target searches.
    slots = sorted(range(48), key=lambda i: (regs[i] is not None, i != 47, i != dest, -i))
    for index, (v, op, a, b) in enumerate(nodes):
        if index == len(nodes) - 1:
            d = dest
        else:
            d = next((i for i in slots if i not in allocated and i != dest), None)
            if d is None:
                return None
        plan.append([op, available[a], available[b], d])
        allocated.add(d)
        available[v] = d
    return plan


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
    if len(json.dumps(program, separators=(',', ':')).encode()) > 131072:
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


def construct(p, radix, order, deep_deadline=0):
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
    for target in order:
        if target not in remaining:
            continue
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
                    plan += [['*', anchor, dest, dest]]
                if gap:
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
        if len(best) > 1 and time.monotonic() < deep_deadline:
            plan = deep_plan(regs, target, limit, len(best), dest,
                             range(radix), min(deep_deadline, time.monotonic() + 0.06))
            if plan is not None and len(plan) < len(best):
                best = plan
        for op, i, j, d in best:
            emit(op, i, j, d)
        if len(out) > 4096:
            return None
    return out if not remaining else None


def mutate(order, rng):
    """Mix small swaps/relocations with large block moves."""
    trial = order[:]
    n = len(trial)
    if n < 2:
        return trial
    a, b = sorted(rng.sample(range(n), 2))
    move = rng.randrange(5)
    if move == 0:
        trial[a], trial[b] = trial[b], trial[a]
    elif move == 1:
        trial.insert(a, trial.pop(b))
    elif move == 2:
        trial[a:b+1] = reversed(trial[a:b+1])
    elif move == 3:
        block = trial[a:b+1]
        rng.shuffle(block)
        trial[a:b+1] = block
    else:
        block = trial[a:b+1]
        del trial[a:b+1]
        pos = rng.randrange(len(trial) + 1)
        trial[pos:pos] = block
    return trial


def optimize(p, old, deadline, rng):
    best = old
    order = sorted(set(p['targets']) - set(p['start']))
    # Seed the search with the existing constructor's best radix.
    state = None
    attempts = 0
    for radix in (16, 8, 32, 4, 2):
        if state is not None and time.monotonic() >= deadline:
            break
        candidate = construct(p, radix, order, deadline)
        attempts += 1
        if candidate is None or not replay(p, candidate):
            continue
        if len(candidate) < len(best):
            best = candidate
        if state is None or len(candidate) < state[0]:
            state = (len(candidate), radix, order[:])
    if state is None or len(order) < 2:
        return best, attempts
    cost, radix, order = state
    elite = (cost, radix, order[:])
    start = time.monotonic()
    stale = 0
    while time.monotonic() < deadline:
        trial = mutate(order, rng)
        trial_radix = radix
        if rng.random() < 0.08:
            trial_radix = rng.choice((8, 16, 32, 4))
        candidate = construct(p, trial_radix, trial, deadline)
        attempts += 1
        if candidate is None or not replay(p, candidate):
            continue
        score = len(candidate)
        progress = (time.monotonic() - start) / max(0.001, deadline - start)
        temperature = max(0.2, min(8.0, elite[0] * 0.012) * (1 - progress))
        if score <= cost or rng.random() < math.exp((cost - score) / temperature):
            cost, radix, order = score, trial_radix, trial
        if score < elite[0]:
            elite = (score, trial_radix, trial[:])
            stale = 0
        else:
            stale += 1
        if score < len(best):
            best = candidate
        if stale >= 20:
            cost, radix, saved = elite
            order = saved[:]
            stale = 0
    return best, attempts


def main():
    o = kt.optimization()
    if o['smoke']:
        # A five-operation E3/E1 expression and a four-operation shared DAG.
        initial = [101, 173, 257, 389, 521, 769]
        for goal, upper in (((101 + 173) * (257 + 389) + 521 * 769, 5),
                            ((101 + 173) ** 2 + 257 + 389, 4)):
            plan = deep_plan(initial + [None] * 42, goal, 10**9,
                             upper + 1, 46, range(6), time.monotonic() + 1)
            assert plan is not None and len(plan) <= upper, (goal, plan)
            assert replay(dict(start=initial, targets=[goal], limit=10**9), plan)
        print('DAG smoke checks passed', flush=True)
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
    ids = sorted(puzzles)[:6] if o['smoke'] else sorted(puzzles)
    rng = random.Random(o['seed'])
    deadline = time.monotonic() + (4 if o['smoke'] else 3000)
    for index, pid in enumerate(ids):
        puzzle = puzzles[pid]
        old = current[pid]
        if not replay(puzzle, old):
            raise ValueError('invalid baseline ' + pid)
        allocation = max(0, deadline - time.monotonic()) / (len(ids) - index)
        candidate, attempts = optimize(puzzle, old, time.monotonic() + allocation, rng)
        if len(candidate) < len(old) and replay(puzzle, candidate):
            current[pid] = candidate
        print(pid, 'before', len(old), 'after', len(current[pid]), 'attempts', attempts, flush=True)
    out = Path(o['solution'])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'program'])
        for pid, puzzle in sorted(puzzles.items()):
            if not replay(puzzle, current[pid]):
                raise ValueError('invalid output ' + pid)
            writer.writerow([pid, json.dumps(current[pid], separators=(',', ':'))])
    print('deep search', DEEP_STATS, flush=True)


if __name__ == '__main__':
    main()
