"""Protected-one Horner fallback, followed by incumbent DAG rewrites.

An incumbent remains the fallback. Replace an exclusive dependency cone only
when its exact union of required instructions becomes smaller.
E1 contains every legal one-operation value over the current physical cache.
Inverse arithmetic joins two E1 entries, optionally under one more E1 operation,
finding expressions of up to five operations without a quadratic E1 product.
"""
import csv
import heapq
import json
import random
import time
from pathlib import Path

from kaggle_agent import toolkit as kt


def calc(op, a, b, limit):
    if op == '+':
        v = a + b
    elif op == '-':
        v = a - b
    elif op == '*':
        v = a * b
    elif op == '/' and b and a % b == 0:
        v = a // b
    else:
        return None
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
        v = calc(op, regs[i], regs[j], p['limit'])
        if v is None:
            return False
        regs[d] = v
        remaining.discard(v)
    return not remaining


def horner_fallback(p, radix=2):
    """Construct every missing target even when no useful anchor exists.

    Registers 0..radix-1 hold the protected constants 1..radix. Register
    47 is scratch; the remaining registers cache completed targets. The
    leading digit is always available, including 1 for binary Horner.
    Each subsequent positive prefix is at most the target, so neither
    doubling/multiplication nor adding a digit can exceed value_limit.
    """
    limit = p['limit']
    radix = min(radix, limit)
    if not 1 <= radix <= 32:
        raise ValueError('Horner radix must be in 1..32')
    regs = list(p['start']) + [None] * (48 - len(p['start']))
    covered = set(p['start'])
    program = []

    def emit(op, i, j, d):
        value = calc(op, regs[i], regs[j], limit)
        if value is None:
            raise ValueError('invalid Horner intermediate')
        program.append([op, i, j, d])
        regs[d] = value
        covered.add(value)

    if set(p['targets']) <= covered:
        return program
    if regs[0] != 1:
        emit('/', 0, 0, 0)
    if radix == 1:
        return program
    for value in range(2, radix + 1):
        if regs[value - 1] != value:
            emit('+', value - 2, 0, value - 1)

    cursor = 0
    for target in p['targets']:
        if target in covered:
            continue
        digits = []
        value = target
        while value:
            value, digit = divmod(value, radix)
            digits.append(digit)
        digits.reverse()
        cache = {v: r for r, v in enumerate(regs[:47]) if v is not None}
        # A missing stepping stone must not abort construction: the leading
        # digit is protected, and for radix 2 this unconditionally starts at 1.
        source = digits[0] - 1
        begin = 1
        prefix = 0
        for pos, digit in enumerate(digits):
            prefix = prefix * radix + digit
            if prefix in cache:
                source, begin = cache[prefix], pos + 1
        steps = []
        for digit in digits[begin:]:
            if radix == 2:
                steps.append(['+', source, source, 47])
            else:
                steps.append(['*', source, radix - 1, 47])
            source = 47
            if digit:
                steps.append(['+', source, digit - 1, 47])
        if not steps:
            continue
        # Preserve the final result without adding a copy instruction. Both
        # sources are read before the destination is overwritten.
        slots = list(range(radix, 47))
        empty = next((r for r in slots if regs[r] is None), None)
        dest = empty if empty is not None else slots[cursor % len(slots)]
        cursor += 1
        steps[-1][3] = dest
        for op, i, j, d in steps:
            emit(op, i, j, d)
    return program


def fallback_checks(puzzles):
    """Exercise absent anchors, tight limits, aliases and full registers."""
    cases = [
        dict(start=[97], targets=[1, 2, 3, 6, 127, 255, 256, 999], limit=999),
        dict(start=[1], targets=[1], limit=1),
        dict(start=[2], targets=[1, 2], limit=2),
        dict(start=[3], targets=[1, 2, 5, 8], limit=9),
        dict(start=list(range(101, 149)),
             targets=[1, 2, 3, 149, 65537, 10**18], limit=10**18),
        dict(start=[10**18], targets=[1, 2**59, 10**18 - 1], limit=10**18),
    ]
    if 'scored_0006' in puzzles:
        cases.append(puzzles['scored_0006'])
    # More completed targets than cache registers forces safe overwrites.
    cases.append(dict(start=[997], targets=list(range(1001, 1101)), limit=1100))
    count = 0
    for p in cases:
        for radix in (2, 4, 8, 16, 32):
            program = horner_fallback(p, radix)
            if not replay(p, program):
                raise ValueError('Horner fallback regression: radix ' + str(radix))
            count += 1
    print('Validated', count, 'Horner fallback regressions including absent anchors', flush=True)


def inverses(t, b, limit):
    """(left op right)=t with one operand b; return other value and order."""
    if t > b:
        yield t - b, '+', False
    if t + b <= limit:
        yield t + b, '-', False
    if b > t:
        yield b - t, '-', True
    if t % b == 0:
        yield t // b, '*', False
    if b <= limit // t:
        yield t * b, '/', False
    if b % t == 0:
        yield b // t, '/', True


def combine(op, a, b, reverse):
    return (op, b, a) if reverse else (op, a, b)


def expression_cost(expr):
    seen = set()

    def visit(e):
        if isinstance(e, int) or e in seen:
            return
        seen.add(e)
        visit(e[1])
        visit(e[2])
    visit(expr)
    return len(seen)


class Frontier:
    def __init__(self, regs, limit):
        self.limit = limit
        self.base = {}
        for r, v in enumerate(regs):
            if v is not None:
                self.base.setdefault(v, r)
        self.e1 = self.base.copy()
        vals = list(self.base.items())
        # Immutable expression tuples permit exact shared-intermediate reuse.
        for i, (a, ar) in enumerate(vals):
            for b, br in vals[i:]:
                for op, x, y, xr, yr in (
                    ('+', a, b, ar, br), ('*', a, b, ar, br),
                    ('-', a, b, ar, br), ('-', b, a, br, ar),
                    ('/', a, b, ar, br), ('/', b, a, br, ar)):
                    v = calc(op, x, y, limit)
                    if v is not None and v not in self.e1:
                        self.e1[v] = (op, xr, yr)
        self.items = list(self.e1.items())
        self.memo = {}

    def join(self, t, deadline):
        if not 0 < t <= self.limit:
            return None
        if t in self.e1:
            return self.e1[t]
        if t in self.memo:
            return self.memo[t]
        best = None
        best_cost = 4
        for k, (b, be) in enumerate(self.items):
            if k % 256 == 0 and time.monotonic() >= deadline:
                return best
            for a, op, reverse in inverses(t, b, self.limit):
                ae = self.e1.get(a)
                if ae is None:
                    continue
                expr = combine(op, ae, be, reverse)
                cost = expression_cost(expr)
                if cost < best_cost:
                    best, best_cost = expr, cost
                    if cost <= 2:
                        self.memo[t] = best
                        return best
        self.memo[t] = best
        return best

    def find(self, t, max_cost, deadline):
        best = self.join(t, deadline)
        cost = expression_cost(best) if best is not None else max_cost + 1
        if cost <= 3 or time.monotonic() >= deadline:
            return best if cost <= max_cost else None
        # First lift a three-operation E1 x E1 join under a cached value (4
        # steps). Then lift it under a one-operation value (5 steps). Probe
        # a bounded, diverse E1 subset rather than materializing E1 squared.
        outer = list(self.base.items())
        extras = [(v, e) for v, e in self.items if v not in self.base]
        extras.sort(key=lambda ve: (abs(ve[0].bit_length() - t.bit_length() // 2), ve[0]))
        if extras:
            stride = max(1, len(extras) // 24)
            probes = extras[:24] + extras[::stride][:24]
            outer += probes
        for b, be in outer:
            if time.monotonic() >= deadline:
                break
            outer_cost = 0 if isinstance(be, int) else 1
            if outer_cost + 2 > min(max_cost, cost - 1):
                continue
            for a, op, reverse in inverses(t, b, self.limit):
                ae = self.join(a, deadline)
                if ae is None:
                    continue
                expr = combine(op, ae, be, reverse)
                n = expression_cost(expr)
                if n < cost and n <= max_cost:
                    best, cost = expr, n
            if cost <= 3:
                break
        return best if cost <= max_cost else None


class ArithmeticDAG:
    """Value DAG plus an exact closure/register allocator for new MITM nodes."""
    def __init__(self, p, program):
        self.values = []
        self.recipes = []
        self.initial = {}
        self.rank = []
        index = {}
        regs = [None] * 48

        def add(v, recipe=None):
            if v in index:
                return index[v]
            n = len(self.values)
            index[v] = n
            self.values.append(v)
            self.recipes.append(recipe)
            self.rank.append(float(n))
            return n

        for r, v in enumerate(p['start']):
            n = add(v)
            self.initial.setdefault(n, r)
            regs[r] = n
        for op, i, j, d in program:
            a, b = regs[i], regs[j]
            v = calc(op, self.values[a], self.values[b], p['limit'])
            regs[d] = add(v, (op, a, b))
        self.roots = sorted({index[t] for t in p['targets']})

    def closure(self):
        selected = set()
        active = set()

        def visit(n):
            if n in self.initial or n in selected:
                return
            if n in active:
                raise ValueError('cyclic rewrite')
            active.add(n)
            _, a, b = self.recipes[n]
            visit(a)
            visit(b)
            active.remove(n)
            selected.add(n)
        for root in self.roots:
            visit(root)
        return selected

    def allocate(self, selected):
        uses = [0] * len(self.values)
        users = [[] for _ in self.values]
        deps = {}
        for n in selected:
            sources = set(self.recipes[n][1:])
            deps[n] = sum(a not in self.initial for a in sources)
            for a in sources:
                uses[a] += 1
                users[a].append(n)
        physical = {n: r for n, r in self.initial.items() if uses[n]}
        free = set(range(48)) - set(physical.values())
        ready = [(self.rank[n], n) for n in selected if not deps[n]]
        heapq.heapify(ready)
        out = []
        while ready:
            if len(free) <= 3:
                pos = min(range(len(ready)), key=lambda k: (
                    -sum(uses[a] == 1 for a in set(self.recipes[ready[k][1]][1:])),
                    ready[k][0]))
                _, n = ready.pop(pos)
                heapq.heapify(ready)
            else:
                _, n = heapq.heappop(ready)
            op, a, b = self.recipes[n]
            i, j = physical[a], physical[b]
            dying = [v for v in set((a, b)) if uses[v] == 1]
            if dying:
                d = min(physical[v] for v in dying)
            elif free:
                d = min(free)
            else:
                return None
            out.append([op, i, j, d])
            for v in set((a, b)):
                uses[v] -= 1
                if uses[v] == 0:
                    free.add(physical.pop(v))
            free.discard(d)
            if uses[n]:
                physical[n] = d
            else:
                free.add(d)
            for user in users[n]:
                deps[user] -= 1
                if deps[user] == 0:
                    heapq.heappush(ready, (self.rank[user], user))
        return out if len(out) == len(selected) else None

    def exclusive_cone(self, root, selected, uses):
        counts = uses.copy()
        removed = {root}
        pending = [root]
        protected = set(self.roots) - {root}
        while pending:
            n = pending.pop()
            for a in set(self.recipes[n][1:]):
                counts[a] -= 1
                if (counts[a] == 0 and a in selected and a not in protected
                        and a not in removed):
                    removed.add(a)
                    pending.append(a)
        return removed

    def install(self, root, expr, bases, limit):
        """New immutable nodes retain shared expressions without value aliasing."""
        memo = {}

        def node(e):
            if isinstance(e, int):
                return bases[e]
            if e in memo:
                return memo[e]
            op, ae, be = e
            a, b = node(ae), node(be)
            n = len(self.values)
            self.values.append(calc(op, self.values[a], self.values[b], limit))
            self.recipes.append((op, a, b))
            self.rank.append(self.rank[root] - .5 + len(memo) * .0001)
            memo[e] = n
            return n
        if isinstance(expr, int):
            return False
        op, a, b = expr
        self.recipes[root] = (op, node(a), node(b))
        return True


def optimize(p, old, deadline):
    """Apply deeper joins to exclusive dependency cones, even if interleaved."""
    dag = ArithmeticDAG(p, old)
    selected = dag.closure()
    candidate = dag.allocate(selected)
    best = candidate if candidate is not None and replay(p, candidate) and len(candidate) < len(old) else old
    hits = 0
    attempted = set()
    while time.monotonic() < deadline:
        selected = dag.closure()
        uses = [0] * len(dag.values)
        users = [[] for _ in dag.values]
        for n in selected:
            for a in set(dag.recipes[n][1:]):
                uses[a] += 1
                users[a].append(n)
        candidates = []
        for root in selected:
            if root in dag.initial or root in attempted:
                continue
            cone = dag.exclusive_cone(root, selected, uses)
            if len(cone) >= 3:
                candidates.append((len(cone), root, cone))
        if not candidates:
            break
        # Small internal cones often expose cheap arithmetic that target-only
        # search misses. Mix them with large target cones rather than spending
        # the entire budget on the largest, most difficult target values.
        candidates.sort(key=lambda item: (
            -min(item[0], 10), dag.values[item[1]].bit_length(), item[1]))
        size, root, cone = candidates[0]
        attempted.add(root)
        # All bases are already required outside the removed cone. Their cost
        # is free in the exact union of dependencies, not an approximate score.
        descendants = {root}
        pending = [root]
        while pending:
            for user in users[pending.pop()]:
                if user not in descendants:
                    descendants.add(user)
                    pending.append(user)
        available = [n for n in (selected - cone) | set(dag.initial)
                     if n not in descendants and dag.rank[n] < dag.rank[root]]
        available.sort(key=lambda n: (n not in dag.initial,
                                      -uses[n], -dag.rank[n]))
        bases = available[:48]
        if not bases:
            continue
        regs = [dag.values[n] for n in bases] + [None] * (48 - len(bases))
        share = (deadline - time.monotonic()) / max(1, min(8, len(candidates)))
        local_deadline = min(deadline, time.monotonic() + max(.006, share))
        frontier = Frontier(regs, p['limit'])
        expr = frontier.find(dag.values[root], min(5, size - 1), local_deadline)
        if expr is None:
            continue
        previous = dag.recipes[root]
        count = len(dag.values)
        if not dag.install(root, expr, bases, p['limit']):
            continue
        trial_selected = dag.closure()
        trial = dag.allocate(trial_selected) if len(trial_selected) < len(best) else None
        if trial is not None and len(trial) < len(best) and replay(p, trial):
            best = trial
            hits += 1
            # Other targets can now reuse the newly retained intermediates.
            attempted.clear()
        else:
            dag.recipes[root] = previous
            del dag.values[count:]
            del dag.recipes[count:]
            del dag.rank[count:]
    return best, hits


def smoke_checks():
    # Independent E1 branches below another E1 branch exercise genuine
    # four/five-operation plans and eliminate a longer incumbent cone.
    for target in (((101 * 109) + (127 * 137)) * 101,
                   ((101 * 109) + (127 * 137)) * (101 + 109)):
        puzzle = dict(start=[101, 109, 127, 137], targets=[target], limit=10**9)
        trace = [['/', 0, 0, 4], ['/', 0, 0, 5]]
        for bit in bin(target)[3:]:
            trace.append(['+', 5, 5, 5])
            if bit == '1':
                trace.append(['+', 5, 4, 5])
        assert replay(puzzle, trace)
        new, hits = optimize(puzzle, trace, time.monotonic() + .25)
        assert replay(puzzle, new) and len(new) <= 5 and hits, (target, len(new))
        print('Deeper E1 x E1 cone:', len(trace), '->', len(new), flush=True)
    # Equal E1 branches share one immutable intermediate in the final DAG.
    p = dict(start=[11, 7, 13, 5], targets=[324, 331], limit=10000)
    old = [['+', 0, 1, 4], ['+', 2, 3, 5], ['*', 4, 5, 6],
           ['+', 6, 1, 7]]
    assert replay(p, old)
    regs = p['start'] + [None] * 44
    f = Frontier(regs, p['limit'])
    e = f.find(324, 5, time.monotonic() + .3)
    dag = ArithmeticDAG(p, old)
    root = next(n for n in dag.roots if dag.values[n] == 324)
    dag.install(root, e, list(range(4)), p['limit'])
    ins = dag.allocate(dag.closure())
    assert ins is not None and replay(p, ins) and len(ins) == 3
    print('Shared E1 branches and retained intermediate check passed', flush=True)
    # Replay random valid incumbent traces and test live-out preservation.
    rng = random.Random(914)
    for full in (False, True):
        for _ in range(6):
            start = list(range(1, 49)) if full else [3, 5, 7]
            regs = start[:] + [None] * (48 - len(start))
            trace = []
            targets = set(start[:1])
            for k in range(45):
                defined = [r for r, v in enumerate(regs) if v is not None]
                i, j = rng.choice(defined), rng.choice(defined)
                op = rng.choice(('+', '-', '*', '/'))
                v = calc(op, regs[i], regs[j], 10**9)
                if v is None:
                    continue
                d = rng.randrange(48)
                regs[d] = v
                trace.append([op, i, j, d])
                if k % 8 == 0:
                    targets.add(v)
            puzzle = dict(start=start, targets=sorted(targets), limit=10**9)
            assert replay(puzzle, trace)
            new, _ = optimize(puzzle, trace, time.monotonic() + .015)
            assert replay(puzzle, new) and len(new) <= len(trace)
    print('Destructive aliases, exact division and liveness checks passed', flush=True)


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
    if o['smoke']:
        fallback_checks(puzzles)
    with Path(o['best'] or locate('sample_submission.csv')).open(newline='', encoding='utf-8') as f:
        current = {r['id']: json.loads(r['program']) for r in csv.DictReader(f)}
    if set(current) != set(puzzles):
        raise ValueError('baseline IDs do not match puzzles')
    for pid, p in puzzles.items():
        if not replay(p, current[pid]):
            raise ValueError('invalid baseline: ' + pid)
    if o['smoke']:
        for pid, p in puzzles.items():
            if not replay(p, horner_fallback(p, 16)):
                raise ValueError('invalid full-set Horner fallback: ' + pid)
        print('Fresh Horner construction validated for', len(puzzles), 'puzzles', flush=True)
    ids = sorted(puzzles)[:6] if o['smoke'] else sorted(puzzles)
    before = sum(len(current[pid]) for pid in ids)
    deadline = time.monotonic() + (4.0 if o['smoke'] else 3000.0)
    for k, pid in enumerate(ids):
        old = current[pid]
        for radix in (2, 4, 8, 16, 32):
            fallback = horner_fallback(puzzles[pid], radix)
            if len(fallback) < len(current[pid]) and replay(puzzles[pid], fallback):
                current[pid] = fallback
        share = max(0, deadline - time.monotonic()) / (len(ids) - k)
        new, hits = optimize(puzzles[pid], current[pid], time.monotonic() + share)
        if len(new) < len(old) and replay(puzzles[pid], new):
            current[pid] = new
        print(pid, len(old), '->', len(current[pid]), 'MITM cones', hits, flush=True)
    out = Path(o['solution'])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'program'])
        for pid, p in sorted(puzzles.items()):
            if not replay(p, current[pid]):
                raise ValueError('invalid output: ' + pid)
            writer.writerow([pid, json.dumps(current[pid], separators=(',', ':'))])
    print('Validated', len(puzzles), 'puzzles; selected total', before, '->',
          sum(len(current[pid]) for pid in ids), flush=True)


if __name__ == '__main__':
    main()
