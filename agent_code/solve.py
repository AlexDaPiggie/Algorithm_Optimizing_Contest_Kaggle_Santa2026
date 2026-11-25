"""Cached-anchor construction with target-indexed, register-retained E1 x E1 joins."""
import csv
import json
import math
import random
import time
from pathlib import Path
from kaggle_agent import toolkit as kt

DEEP_STATS = {'calls': 0, 'e1xe1': 0, 'plans': 0, 'shared': 0}


def deep_plan(regs, target, limit, bound, dest, protected, deadline):
    """Stream E1 x E1 into a backward target index before expanding E2.

    Only matching E3 values need recipes. Nodes are immutable, so later cheaper
    recipes cannot change a previously costed DAG or introduce a dependency cycle.
    Each winning DAG is emitted once and its intermediates remain in registers.
    """
    if bound <= 1 or time.monotonic() >= deadline:
        return None
    DEEP_STATS['calls'] += 1
    base = {v: i for i, v in enumerate(regs) if v is not None}
    nodes = []  # (value, op, left node, right node, arithmetic-node set)
    lookup = {}
    for value, register in base.items():
        lookup[value] = len(nodes)
        nodes.append((value, None, register, None, frozenset()))

    def append(value, op, a, b):
        used = nodes[a][4] | nodes[b][4]
        previous = lookup.get(value)
        if previous is not None and len(nodes[previous][4]) <= len(used) + 1:
            return previous
        index = len(nodes)
        nodes.append((value, op, a, b, used | {index}))
        lookup[value] = index
        return index

    values = sorted(base)
    for i, a in enumerate(values):
        ai = lookup[a]
        for b in values[:i + 1]:
            bi = lookup[b]
            if a + b <= limit:
                append(a + b, '+', ai, bi)
            if a * b <= limit:
                append(a * b, '*', ai, bi)
            if a > b:
                append(a - b, '-', ai, bi)
            if a % b == 0:
                append(a // b, '/', ai, bi)
            if b % a == 0:
                append(b // a, '/', bi, ai)
    e1 = dict(lookup)
    best = lookup.get(target)
    best_cost = len(nodes[best][4]) if best is not None else bound

    def inverses(a):
        return (('+', target - a, False),
                ('-', a - target, False),
                ('-', a + target, True),
                ('*', target // a if target % a == 0 else 0, False),
                ('/', a // target if a % target == 0 else 0, False),
                ('/', a * target, True))

    def consider(op, ai, bi):
        nonlocal best, best_cost
        cost = len(nodes[ai][4] | nodes[bi][4]) + 1
        if cost < best_cost:
            best = append(target, op, ai, bi)
            best_cost = len(nodes[best][4])

    def join(frontier):
        for a, ai in frontier.items():
            if len(nodes[ai][4]) + 1 >= best_cost:
                continue
            for op, b, reverse in inverses(a):
                bi = frontier.get(b)
                if bi is not None:
                    consider(op, bi, ai) if reverse else consider(op, ai, bi)

    if best is None:
        join(e1)
    if best_cost > 3 and bound > 3 and time.monotonic() < deadline:
        # Index the E3 values that could finish a target with one E1 partner.
        # This avoids storing tens of thousands of unhelpful E3 expressions.
        wanted = {}
        for a, ai in e1.items():
            for op, b, reverse in inverses(a):
                if 0 < b <= limit:
                    wanted.setdefault(b, []).append((op, ai, reverse))

        def probe(value, op, ai, bi):
            if not 0 < value <= limit:
                return
            partners = wanted.get(value)
            if partners is None:
                return
            used = nodes[ai][4] | nodes[bi][4]
            existing = e1.get(value)
            # The final partner can share either input E1. Count it only once.
            for final_op, partner, reverse in partners:
                work = nodes[existing][4] if existing is not None else used
                cost = len(work | nodes[partner][4]) + (1 if existing is not None else 2)
                if cost < best_cost:
                    middle = existing if existing is not None else append(value, op, ai, bi)
                    consider(final_op, middle, partner) if reverse else consider(final_op, partner, middle)

        ordered = sorted(e1)
        root = math.isqrt(target)
        # Products of the largest live anchors are easy to miss in a numeric
        # spread. Put their E1 recipes first, including both sides of ab+cd.
        anchors = [v for v in values if v > 64][-8:]
        seed_values = set()
        for i, a in enumerate(anchors):
            for b in anchors[:i + 1]:
                seed_values.update((a + b, a * b, a - b))
                if a % b == 0:
                    seed_values.add(a // b)
        seed_values.intersection_update(e1)
        # A broad E1 pool, with extra small factors and values near sqrt(T).
        pool = set(ordered[:40])
        pool.update(sorted(ordered, key=lambda v: abs(v - root))[:40])
        stride = max(1, len(ordered) // 64)
        pool.update(ordered[::stride][:64])
        pool = sorted(seed_values) + sorted(pool - seed_values)
        DEEP_STATS['e1xe1'] += 1
        for i, a in enumerate(pool):
            if time.monotonic() >= deadline or (best is not None and best_cost <= 4):
                break
            ai = e1[a]
            for b in pool[:i + 1]:
                bi = e1[b]
                probe(a + b, '+', ai, bi)
                probe(a * b, '*', ai, bi)
                if a > b:
                    probe(a - b, '-', ai, bi)
                if a % b == 0:
                    probe(a // b, '/', ai, bi)
                if b % a == 0:
                    probe(b // a, '/', bi, ai)

    if best_cost > 4 and bound > 4 and time.monotonic() < deadline:
        # E2 x E2 covers the other five-operation tree shape. Even a partial
        # frontier is joined if the deadline arrives during its construction.
        for index, (a, ai) in enumerate(e1.items()):
            if index % 8 == 0 and time.monotonic() >= deadline:
                break
            for b in values:
                bi = e1[b]
                if a + b <= limit:
                    append(a + b, '+', ai, bi)
                if a * b <= limit:
                    append(a * b, '*', ai, bi)
                if a > b:
                    append(a - b, '-', ai, bi)
                elif b > a:
                    append(b - a, '-', bi, ai)
                if a % b == 0:
                    append(a // b, '/', ai, bi)
                if b % a == 0:
                    append(b // a, '/', bi, ai)
        join({v: node for v, node in lookup.items() if len(nodes[node][4]) <= 2})

    if best is None or best_cost >= bound:
        return None
    sequence = []
    seen = set()
    leaves = set()

    def visit(index):
        if index in seen:
            return
        seen.add(index)
        _, op, a, b, _ = nodes[index]
        if op is None:
            leaves.add(a)
            return
        visit(a)
        visit(b)
        sequence.append(index)

    visit(best)
    if len(sequence) >= bound:
        return None
    allocated = set(protected) | leaves
    available = {i: node[2] for i, node in enumerate(nodes) if node[1] is None}
    slots = sorted(range(48), key=lambda i: (regs[i] is not None, i != 47, -i))
    plan = []
    for index in sequence:
        _, op, a, b, _ = nodes[index]
        d = dest if index == best else next(
            (r for r in slots if r not in allocated and r != dest), None)
        if d is None:
            return None
        plan.append([op, available[a], available[b], d])
        allocated.add(d)
        available[index] = d
    DEEP_STATS['plans'] += 1
    uses = [operand for index in sequence for operand in nodes[index][2:4]
            if nodes[operand][1] is not None]
    DEEP_STATS['shared'] += int(len(uses) != len(set(uses)))
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


class HornerFrontier:
    """Compare signed radix B and unsigned radix B/2 using one small dictionary.

    The protected constants are 1..B/2 and B, requiring B/2+1 registers.
    Signed digits lie in [-B/2+1, B/2]. Multiplication before a subtraction
    may exceed the final target, so every such intermediate is checked here.
    Choices are immutable; instructions are allocated only for the winner.
    """

    def __init__(self, regs, radix, dest, limit):
        self.radix = radix
        self.dest = dest
        self.limit = limit
        self.lookup = {v: i for i, v in enumerate(regs) if v is not None}
        self.scratch_lookup = {v: i for i, v in enumerate(regs)
                               if v is not None and i != dest}
        self.cache = ({}, {})
        self.ones = (
            next((i for i, v in enumerate(regs) if v == 1 and i != dest), None),
            next((i for i, v in enumerate(regs)
                  if v == 1 and i not in (47, dest)), None),
        )

    def choose(self, goal, scratch=False):
        cache = self.cache[scratch]
        found = cache.get(goal)
        if found is not None:
            return found
        lookup = self.scratch_lookup if scratch else self.lookup
        dest = 47 if scratch else self.dest
        best = None
        for radix, signed in ((self.radix, True), (self.radix // 2, False)):
            if radix < 2:
                continue
            q = goal
            depth = nonzero = 0
            while q:
                src = lookup.get(q)
                if src is not None:
                    cost = depth + nonzero if depth else int(src != dest)
                    if best is None or cost < best[0]:
                        best = (cost, goal, src, depth, radix, signed)
                q, digit = divmod(q, radix)
                if signed and digit > radix // 2:
                    digit -= radix
                    q += 1
                if q and q * radix > self.limit:
                    break
                depth += 1
                nonzero += digit != 0
        if best is None:
            one = self.ones[scratch]
            if one is None:
                raise ValueError('missing protected one')
            cost = goal.bit_length() + goal.bit_count() - 2
            if goal == 1:
                cost = int(one != dest)
            best = (cost, goal, one, -1, 2, False)
        cache[goal] = best
        return best

    def emit(self, choice, scratch=False):
        _, goal, src, depth, radix, signed = choice
        dest = 47 if scratch else self.dest
        out = []
        if depth == -1:
            one = src
            for bit in bin(goal)[3:]:
                out.append(['+', src, src, dest])
                src = dest
                if bit == '1':
                    out.append(['+', dest, one, dest])
            if not out and src != dest:
                out.append(['*', one, one, dest])
        else:
            digits = []
            for _ in range(depth):
                goal, digit = divmod(goal, radix)
                if signed and digit > radix // 2:
                    digit -= radix
                    goal += 1
                digits.append(digit)
            radix_reg = self.radix // 2 if radix == self.radix else radix - 1
            for digit in reversed(digits):
                out.append(['*', src, radix_reg, dest])
                src = dest
                if digit:
                    out.append(['+' if digit > 0 else '-', dest, abs(digit) - 1, dest])
            if not out and src != dest:
                out.append(['*', src, 0, dest])
        return out


def target_plan(regs, target, radix, dest, limit):
    frontier = HornerFrontier(regs, radix, dest, limit)
    lookup = frontier.lookup
    direct = frontier.choose(target)
    best_cost = direct[0]
    winner = ('direct', direct)
    for anchor, value in enumerate(regs):
        if value is None or anchor == dest:
            continue
        if value != target:
            # A larger live anchor can be just as close as a smaller one.
            # Build the positive residual in dest before reading the anchor.
            choice = frontier.choose(abs(target - value))
            if choice[0] + 1 < best_cost:
                best_cost = choice[0] + 1
                winner = ('offset', anchor, choice, value < target)
        if anchor == 47 or value <= radix:
            continue
        quotient = target // value
        for scale in (quotient, quotient + 1):
            if not 2 <= scale <= 65536:
                continue
            product = value * scale
            if product > limit:
                continue
            gap = abs(target - product)
            scale_reg = lookup.get(scale)
            if scale_reg is not None and scale_reg != dest:
                scale_choice = None
                cost = 1
            else:
                scale_choice = frontier.choose(scale)
                cost = scale_choice[0] + 1
            # Offsets can only add cost; avoid even scoring hopeless moves.
            if cost + bool(gap) >= best_cost:
                continue
            offset = frontier.choose(gap, True) if gap else None
            if gap:
                cost += offset[0] + 1
            if cost < best_cost:
                best_cost = cost
                winner = ('multiply', anchor, scale_reg, scale_choice,
                          offset, product < target)
    if best_cost > 1:
        for a, i in lookup.items():
            possibilities = (('+', target - a, False),
                             ('-', a - target, False),
                             ('-', a + target, True),
                             ('*', target // a if target % a == 0 else 0, False),
                             ('/', a // target if a % target == 0 else 0, False),
                             ('/', a * target, True))
            for op, b, reverse in possibilities:
                if b in lookup:
                    j = lookup[b]
                    return [[op, j, i, dest] if reverse else [op, i, j, dest]]
    if winner[0] == 'direct':
        return frontier.emit(winner[1])
    if winner[0] == 'offset':
        return frontier.emit(winner[2]) + [
            ['+' if winner[3] else '-', winner[1], dest, dest]]
    _, anchor, scale_reg, scale_choice, offset, positive = winner
    if scale_choice is None:
        plan = [['*', anchor, scale_reg, dest]]
    else:
        plan = frontier.emit(scale_choice) + [['*', anchor, dest, dest]]
    if offset is not None:
        plan += frontier.emit(offset, True)
        plan.append(['+' if positive else '-', dest, 47, dest])
    return plan


def construct(p, radix, order, deep_deadline=0):
    limit = p['limit']
    # Use an even radix whose balanced dictionary fits in the register file.
    radix = min(radix, 2 ** (limit.bit_length() - 1))
    protected_count = radix // 2 + 1
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
    for digit in range(2, radix // 2 + 1):
        if digit % 2 == 0:
            emit('+', digit // 2 - 1, digit // 2 - 1, digit - 1)
        else:
            emit('+', digit - 2, 0, digit - 1)
    emit('+', radix // 2 - 1, radix // 2 - 1, radix // 2)

    cursor = 0
    for target in order:
        if target not in remaining:
            continue
        dest = protected_count + cursor % (47 - protected_count)
        cursor += 1
        best = target_plan(regs, target, radix, dest, limit)
        if len(best) > 1 and time.monotonic() < deep_deadline:
            plan = deep_plan(regs, target, limit, len(best), dest,
                             range(protected_count), min(deep_deadline, time.monotonic() + 0.06))
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
    # Compare balanced dictionaries cheaply before spending time on deep joins.
    state = None
    attempts = 0
    for radix in (32, 64, 16, 8, 4):
        if state is not None and time.monotonic() >= deadline:
            break
        candidate = construct(p, radix, order)
        attempts += 1
        if candidate is None or not replay(p, candidate):
            continue
        if len(candidate) < len(best):
            best = candidate
        if state is None or len(candidate) < state[0]:
            state = (len(candidate), radix, order[:])
    if state is None or len(order) < 2:
        return best, attempts
    # Descending traversal exposes larger targets as anchors immediately;
    # retain it only when its actual constructed program is shorter.
    if time.monotonic() < deadline:
        reverse_order = order[::-1]
        candidate = construct(p, state[1], reverse_order)
        attempts += 1
        if candidate is not None and replay(p, candidate):
            if len(candidate) < len(best):
                best = candidate
            if len(candidate) < state[0]:
                state = (len(candidate), state[1], reverse_order)
    cost, radix, order = state
    elite = (cost, radix, order[:])
    start = time.monotonic()
    stale = 0
    while time.monotonic() < deadline:
        trial = mutate(order, rng)
        trial_radix = radix
        if rng.random() < 0.08:
            trial_radix = rng.choice((16, 32, 64, 8))
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


def smoke_checks():
    """Exercise signed borrows, exact-limit overshoots, and scratch aliasing."""
    rng = random.Random(914103)
    checks = 0
    # E3 = (a*b + c*d), followed by an E1 partner; the first target reuses
    # a*b. Both require the new level and all four intermediates can stay live.
    initial = [101, 233, 487, 997]
    ab, cd = initial[0] * initial[1], initial[2] * initial[3]
    for target, maximum in (((ab + cd) * ab, 4),
                            ((ab + cd) * (initial[0] + initial[2]), 5)):
        for full in (False, True):
            start_values = initial + (list(range(2, 46)) if full else [])
            puzzle = dict(start=start_values, targets=[target], limit=10**18)
            regs = start_values + [None] * (48 - len(start_values))
            protected = range(4, 36) if full else ()
            plan = deep_plan(regs, target, puzzle['limit'], maximum + 1,
                             46, protected, time.monotonic() + 1)
            assert plan is not None and len(plan) <= maximum, (target, full, plan)
            assert replay(puzzle, plan), (puzzle, plan)
            state = regs[:]
            for op, i, j, d in plan:
                state[d] = calc(op, state[i], state[j], puzzle['limit'])
            assert state[46] == target
            assert all(state[i] == regs[i] for i in protected)
            assert len({ins[3] for ins in plan}) == len(plan)
    # Shared p in (p*p)+p is a three-node DAG even though the expression tree
    # has five nodes. It must also work when the incumbent has just four steps.
    value = ab * ab + ab
    puzzle = dict(start=initial, targets=[value], limit=value)
    plan = deep_plan(initial + [None] * 44, value, value, 4, 46, (),
                     time.monotonic() + 1)
    assert plan is not None and len(plan) <= 3 and replay(puzzle, plan), plan
    print('Retained E1 x E1 checks passed: 5', flush=True)
    # Residual 37 needs two Horner operations with this dictionary. The final
    # subtraction must read the preserved larger anchor, not overwritten dest.
    for target in (10**12, 10**18 - 40):
        for sign in (-1, 1):
            initial = list(range(1, 17)) + [32, 99, target + sign * 37]
            limit = max(initial[-1], target)
            puzzle = dict(start=initial, targets=[target], limit=limit)
            regs = initial + [None] * (48 - len(initial))
            plan = target_plan(regs, target, 32, 17, limit)
            assert len(plan) == 3, plan
            assert plan[-1] == ['-' if sign > 0 else '+', 18, 17, 17], plan
            assert replay(puzzle, plan), (puzzle, plan)
    print('Bidirectional residual checks passed: 4', flush=True)
    for radix in (2, 4, 8, 16, 32, 64):
        constants = list(range(1, radix // 2 + 1)) + [radix]
        for limit in (radix, radix ** 2 - 1, 10**6, 10**18):
            regs = constants + [None] * (48 - len(constants))
            for i in range(len(constants), 48):
                if rng.random() < 0.8:
                    regs[i] = rng.randint(1, limit)
            dest = len(constants)
            frontier = HornerFrontier(regs, radix, dest, limit)
            goals = {1, limit, max(1, limit - 1), radix - 1,
                     min(limit, radix ** 2 - 1)}
            goals.update(rng.randint(1, limit) for _ in range(8))
            for goal in goals:
                for scratch in (False, True):
                    choice = frontier.choose(goal, scratch)
                    plan = frontier.emit(choice, scratch)
                    assert len(plan) == choice[0], (choice, plan)
                    state = regs[:]
                    for op, i, j, d in plan:
                        if scratch:
                            assert i != dest and j != dest
                        assert state[i] is not None and state[j] is not None
                        value = calc(op, state[i], state[j], limit)
                        assert value is not None, (goal, limit, plan)
                        state[d] = value
                    assert state[47 if scratch else dest] == goal
                    assert state[:len(constants)] == constants
                    checks += 1
    # Signed base 64 represents 4095 as (64*64)-1. At limit 4095 that
    # multiplication is forbidden, requiring the unsigned or binary path.
    regs = list(range(1, 33)) + [64] + [None] * 15
    frontier = HornerFrontier(regs, 64, 33, 4096)
    signed = frontier.choose(4095)
    assert signed[0] == 2 and signed[-1] is True
    assert frontier.emit(signed) == [['*', 32, 32, 33], ['-', 33, 0, 33]]
    frontier = HornerFrontier(regs, 64, 33, 4095)
    assert frontier.choose(4095)[0] > 2
    for initial, targets, limit in (([3], [1, 2, 5, 8], 9),
                                     (list(range(2, 50)), [1, 63, 127, 255], 255),
                                     ([1], [1], 1)):
        puzzle = dict(start=initial, targets=targets, limit=limit)
        plan = construct(puzzle, 32, targets)
        assert plan is not None and replay(puzzle, plan)
    print('Balanced-radix checks passed:', checks, flush=True)


def main():
    o = kt.optimization()
    if o['smoke']:
        smoke_checks()
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
    before_total = sum(len(current[pid]) for pid in ids)
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
    print('Validated output:', len(puzzles), 'puzzles; selected total',
          before_total, '->', sum(len(current[pid]) for pid in ids), flush=True)


if __name__ == '__main__':
    main()
