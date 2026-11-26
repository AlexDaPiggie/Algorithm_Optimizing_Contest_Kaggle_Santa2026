"""Close residual-planner gaps with binary Horner construction from 1.

Build targets around products of cached one-operation dictionary entries,
then synthesize the signed residual. Preserve every shorter valid incumbent.

An incumbent remains the fallback. Replace an exclusive dependency cone only
when its exact union of required instructions becomes smaller.
E1 contains every legal one-operation value over the current physical cache.
Inverse arithmetic joins two E1 entries, optionally under one more E1 operation,
finding expressions of up to five operations without a quadratic E1 product.
"""
import csv
import bisect
import heapq
import json
import random
import time
import math
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
    # Bypass bootstrap to reproduce the missing-stepping-stone failure.
    # An expired search deadline must still yield a legal construction.
    planner_cases = [
        dict(start=[97], targets=[1, 2, 3, 6, 15, 17, 127, 255, 256, 999], limit=999),
        dict(start=[1], targets=[1, 3, 15, 17, 257, 65537], limit=65537),
        dict(start=[10**18], targets=[1, 2**59, 10**18 - 1], limit=10**18),
        dict(start=[3], targets=[1, 2, 5, 8], limit=9),
    ]
    if 'scored_0006' in puzzles:
        planner_cases.append(puzzles['scored_0006'])
    rng = random.Random(6006)
    for _ in range(12):
        limit = rng.randrange(1000, 10**18)
        planner_cases.append(dict(start=[rng.randrange(2, limit + 1)],
                                  targets=sorted({rng.randrange(1, limit + 1) for _ in range(4)}),
                                  limit=limit))
    checks = 0
    for p in planner_cases:
        for target in p['targets']:
            if target in p['start']:
                continue
            cache = ConstructCache(p)
            planner = RemainderPlanner(cache.regs, p['limit'], time.monotonic() - 1)
            expr = planner.find(target)
            if expr is None:
                raise ValueError('Horner planner returned no expression')
            cache.emit(expr)
            single = dict(p, targets=[target])
            if not replay(single, cache.program):
                raise ValueError('invalid Horner planner fallback')
            if planner.horner(target) != expr:
                raise ValueError('unstable Horner memo')
            checks += 1
    # Saturated registers exercise destruction of unprotected seeds while
    # the sole expression leaf and the shared unit remain live.
    p = dict(start=list(range(101, 149)), targets=[10**18 - 1], limit=10**18)
    cache = ConstructCache(p)
    cache.protected = {101}
    planner = RemainderPlanner(cache.regs, p['limit'], time.monotonic() - 1)
    cache.emit(planner.find(p['targets'][0]))
    if not replay(p, cache.program):
        raise ValueError('invalid full-register Horner fallback')
    print('Validated', checks + 1, 'sparse planner fallbacks including expired deadlines', flush=True)


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
        # A repeated E1 operand costs only two instructions. Test these before
        # the scan: base operands occur first, so the first remaining hit has
        # optimal cost (two with a base operand, otherwise three).
        e1 = self.e1
        if t % 2 == 0 and t // 2 in e1:
            e = e1[t // 2]
            result = ('+', e, e)
            self.memo[t] = result
            return result
        root = math.isqrt(t)
        if root * root == t and root in e1:
            e = e1[root]
            result = ('*', e, e)
            self.memo[t] = result
            return result
        get = e1.get
        limit = self.limit
        for k, (b, be) in enumerate(self.items):
            if k % 256 == 0 and time.monotonic() >= deadline:
                # A partial negative lookup must never enter the E2 memo.
                return None
            ae = get(t - b) if t > b else None
            if ae is not None:
                result = ('+', ae, be)
            else:
                ae = get(t + b) if t + b <= limit else None
                if ae is not None:
                    result = ('-', ae, be)
                else:
                    ae = get(b - t) if b > t else None
                    if ae is not None:
                        result = ('-', be, ae)
                    else:
                        ae = get(t // b) if t % b == 0 else None
                        if ae is not None:
                            result = ('*', ae, be)
                        else:
                            ae = get(t * b) if b <= limit // t else None
                            if ae is not None:
                                result = ('/', ae, be)
                            else:
                                ae = get(b // t) if b % t == 0 else None
                                if ae is None:
                                    continue
                                result = ('/', be, ae)
            self.memo[t] = result
            return result
        self.memo[t] = None
        return None

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
            if cost <= 4:
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
        # Later targets have the richest bank of retained operands. Try their
        # expensive cones first, then use any remaining time on internal cones.
        target_roots = set(dag.roots)
        candidates.sort(key=lambda item: (
            item[1] not in target_roots,
            -min(item[0], 10), -dag.rank[item[1]], item[1]))
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


class RemainderPlanner:
    """Meet in the middle near the target, then construct its residual.

    Instead of fixing one radix for every digit, choose a product of two
    one-step dictionary values. An exact, inexpensive residual completes it.
    All expressions refer to the current physical register snapshot.
    """
    def __init__(self, regs, limit, deadline):
        self.frontier = Frontier(regs, limit)
        self.e1 = self.frontier.e1
        self.base = self.frontier.base
        self.limit = limit
        self.deadline = deadline
        self.values = sorted(self.e1)
        self.two_memo = {}
        self.horner_memo = {}

    def two(self, t):
        if t in self.e1:
            return self.e1[t]
        if t in self.two_memo:
            return self.two_memo[t]
        if not 0 < t <= self.limit:
            return None
        for b, be in self.base.items():
            for a, op, rev in inverses(t, b, self.limit):
                ae = self.e1.get(a)
                if ae is not None:
                    expr = combine(op, ae, be, rev)
                    self.two_memo[t] = expr
                    return expr
        self.two_memo[t] = None
        return None

    def horner(self, t):
        if not 0 < t <= self.limit:
            return None
        if t in self.horner_memo:
            return self.horner_memo[t]
        result = self.two(t)
        if result is not None:
            self.horner_memo[t] = result
            return result
        best, cost = None, 10**6
        # These radices are present in the bootstrap; missing ones are skipped.
        for b in (16, 256, 65536, 4294967296):
            if b not in self.base or b >= t:
                continue
            q, r = divmod(t, b)
            qe = self.horner(q)
            re = self.horner(r) if r else None
            if qe is None or (r and re is None):
                continue
            prod = ('*', qe, self.base[b])
            expr = ('+', prod, re) if r else prod
            n = expression_cost(expr)
            if n < cost:
                best, cost = expr, n
        if best is None and 0 < t <= self.limit:
            # A sparse dictionary need not contain a radix or a stepping
            # stone. Self-division always supplies 1; binary Horner then
            # visits positive prefixes <= t, even with an expired deadline.
            # Immutable tuples retain the shared 1 and any cached prefixes.
            one = self.e1.get(1)
            if one is not None:
                two = self.e1.get(2)
                if two is None:
                    two = ('+', one, one)
                best = one
                prefix = 1
                for bit in bin(t)[3:]:
                    prefix = prefix * 2 + (bit == '1')
                    cached = self.e1.get(prefix)
                    if cached is not None:
                        best = cached
                        continue
                    # Multiplication by the shared 2 keeps expression tuples
                    # linear in size (nested repeated additions do not).
                    best = ('*', best, two)
                    if bit == '1':
                        best = ('+', best, one)
        self.horner_memo[t] = best
        return best

    def find(self, target):
        best = self.horner(target)
        cost = expression_cost(best) if best is not None else 10**6
        if cost <= 2:
            return best
        direct = self.frontier.join(target, self.deadline)
        if direct is not None and expression_cost(direct) < cost:
            best, cost = direct, expression_cost(direct)
        if cost <= 3 or time.monotonic() >= self.deadline:
            return best

        # Spend a bounded part of this target's allowance on exact depth 4/5
        # before resorting to an approximate product and residual. Emission
        # retains both E1 branches, so later targets can reuse their values.
        deep_end = time.monotonic() + max(0, self.deadline - time.monotonic()) * .3
        deep = self.frontier.find(target, min(5, cost - 1), deep_end)
        if deep is not None:
            n = expression_cost(deep)
            if n < cost:
                best, cost = deep, n
        if cost <= 4 or time.monotonic() >= self.deadline:
            return best

        # Nearest neighbours of target/a are the most promising E1 partners.
        # Retain a residual diversity set so a slightly larger but exactly
        # constructible residual can beat the nearest numerical product.
        proposals = {}
        vals, e1 = self.values, self.e1

        def offer(value, expr, prod_cost):
            nonlocal best, cost
            if not 0 < value <= self.limit:
                return
            residual = abs(target - value)
            if not residual:
                n = expression_cost(expr)
                if n < cost:
                    best, cost = expr, n
                return
            re = e1.get(residual)
            if re is not None:
                plan = ('+', expr, re) if value < target else ('-', expr, re)
                n = expression_cost(plan)
                if n < cost:
                    best, cost = plan, n
            if prod_cost + 1 >= cost:
                return
            key = (residual, value < target)
            old = proposals.get(key)
            if old is None or prod_cost < old[0]:
                proposals[key] = (prod_cost, expr)

        pos = bisect.bisect_left(vals, target)
        for a in vals[max(0, pos - 12):pos + 12]:
            ae = e1[a]
            offer(a, ae, int(not isinstance(ae, int)))
        center = bisect.bisect_left(vals, math.isqrt(target))
        step = max(1, len(vals) // 192)
        factors = list(dict.fromkeys(
            vals[max(0, center - 160):center + 160] +
            list(self.base) + vals[::step]))
        enum_deadline = time.monotonic() + max(0, self.deadline - time.monotonic()) * .45
        for k, a in enumerate(factors):
            if k % 32 == 0 and time.monotonic() >= enum_deadline:
                break
            if a == 1:
                continue
            q = target // a
            j = bisect.bisect_left(vals, q)
            ae = e1[a]
            for b in vals[max(0, j - 1):j + 2]:
                if b == 1:
                    continue
                value = a * b
                if value > self.limit:
                    continue
                be = e1[b]
                expr = ('*', ae, be)
                pcost = 1 + int(not isinstance(ae, int))
                if be != ae:
                    pcost += int(not isinstance(be, int))
                offer(value, expr, pcost)

        # Estimated residual complexity only ranks probes. Acceptance always
        # uses the exact union of immutable expression nodes.
        ranked = sorted(proposals.items(), key=lambda x: (
            x[1][0] * 6 + x[0][0].bit_length(), x[0][0]))
        for k, ((residual, below), (pcost, prod)) in enumerate(ranked[:192]):
            if k % 8 == 0 and time.monotonic() >= self.deadline:
                break
            re = self.two(residual)
            if re is None and k < 8:
                re = self.horner(residual)
            if re is None:
                continue
            expr = ('+' if below else '-', prod, re)
            n = expression_cost(expr)
            if n < cost:
                best, cost = expr, n
                if cost <= 3:
                    break
        for (residual, below), (pcost, prod) in ranked[:12]:
            if pcost + 3 >= cost or time.monotonic() >= self.deadline:
                continue
            re = self.frontier.join(residual, self.deadline)
            if re is not None:
                expr = ('+' if below else '-', prod, re)
                n = expression_cost(expr)
                if n < cost:
                    best, cost = expr, n
        return best


class ConstructCache:
    """Physical cache; pin all expression leaves until its last instruction."""
    def __init__(self, p):
        self.p = p
        self.regs = p['start'][:] + [None] * (48 - len(p['start']))
        self.program = []
        self.covered = set(p['start'])
        self.protected = set(p['start'])
        self.age = [0] * 48
        self.clock = 0

    def emit(self, expr):
        nodes = []
        seen = set()
        leaves = set()

        def visit(e):
            if isinstance(e, int):
                leaves.add(e)
                return
            if e in seen:
                return
            seen.add(e)
            visit(e[1])
            visit(e[2])
            nodes.append(e)
        visit(expr)
        loc = {r: r for r in leaves}
        uses = {e: 0 for e in leaves | seen}
        for e in nodes:
            for a in set(e[1:]):
                uses[a] += 1
        for e in nodes:
            op, a, b = e
            i, j = loc[a], loc[b]
            value = calc(op, self.regs[i], self.regs[j], self.p['limit'])
            if value is None:
                raise ValueError('invalid adaptive remainder expression')
            # Reuse a cached result only if the physical value is still live.
            existing = next((r for r, v in enumerate(self.regs) if v == value), None)
            if existing is not None:
                d = existing
            else:
                locked = {loc[x] for x, n in uses.items() if n and x in loc}
                # A last-use source can be overwritten by its result.
                for x in set((a, b)):
                    if uses[x] == 1:
                        r = loc[x]
                        if all(y == x or not n or y not in loc or loc[y] != r
                               for y, n in uses.items()):
                            locked.discard(r)
                slots = [r for r in range(48) if r not in locked
                         and self.regs[r] not in self.protected]
                if not slots:
                    raise ValueError('adaptive expression exceeds register budget')
                d = min(slots, key=lambda r: (
                    self.regs[r] is not None,
                    # Keep the most recent arithmetic intermediates.
                    self.age[r]))
                self.program.append([op, i, j, d])
                self.regs[d] = value
                self.clock += 1
                self.age[d] = self.clock
                self.covered.add(value)
            loc[e] = d
            for x in set((a, b)):
                uses[x] -= 1
        return loc[expr] if not isinstance(expr, int) else expr

    def bootstrap(self):
        # Preserve original seeds, rather than overwriting all of them with
        # consecutive small digits. They become free adaptive radices.
        if len(self.protected) > 12:
            return False
        for target in range(1, min(8, self.p['limit']) + 1):
            if target not in self.regs:
                f = Frontier(self.regs, self.p['limit'])
                expr = f.e1.get(target)
                if expr is None:
                    return False
                self.emit(expr)
            self.protected.add(target)
        if self.p['limit'] >= 16 and 16 not in self.regs:
            self.emit(('+', self.regs.index(8), self.regs.index(8)))
            self.protected.add(16)
        for v in (16, 256, 65536):
            if v in self.regs and v * v <= self.p['limit']:
                self.emit(('*', self.regs.index(v), self.regs.index(v)))
                self.protected.add(v * v)
        # Two native seed powers diversify product magnitudes without making
        # every target pay for a long fixed-radix bootstrap.
        root = math.isqrt(max(self.p['targets']))
        for seed in sorted(set(self.p['start']))[-4:]:
            v = seed
            if v > 1 and v in self.regs and v * v <= self.p['limit']:
                self.emit(('*', self.regs.index(v), self.regs.index(v)))
                v *= v
                # Place anchors near sqrt(target), the scale where a product
                # of two dictionary entries spans the target range.
                while v * 16 < root and v * 16 <= self.p['limit']:
                    self.emit(('*', self.regs.index(v), self.regs.index(16)))
                    v *= 16
                self.protected.add(v)
        return True


def construct_remainders(p, deadline, reverse=False):
    cache = ConstructCache(p)
    if not cache.bootstrap():
        return None
    targets = sorted(p['targets'], reverse=reverse)
    for k, target in enumerate(targets):
        if target in cache.covered:
            continue
        if time.monotonic() >= deadline:
            return None
        share = (deadline - time.monotonic()) / max(1, len(targets) - k)
        planner = RemainderPlanner(cache.regs, p['limit'], time.monotonic() + share)
        expr = planner.find(target)
        if expr is None:
            return None
        cache.emit(expr)
    if replay(p, cache.program):
        # Remove bootstrap values which the resulting program never used.
        dag = ArithmeticDAG(p, cache.program)
        compact = dag.allocate(dag.closure())
        if compact is not None and len(compact) < len(cache.program) and replay(p, compact):
            return compact
        return cache.program
    return None


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



class IncrementalClosure:
    """Exact active DAG maintained by reference counts, with reversible edits.

    A root has one permanent reference. Each active instruction contributes
    one reference per distinct operand. Crossing zero is the only event that
    requires visiting descendants; initial values never count as instructions.
    """
    def __init__(self, recipes, roots, initial):
        self.recipes = list(recipes)
        self.initial = set(initial)
        self.refs = [0] * len(recipes)
        self.live = set()
        for n in roots:
            self._add(n)

    def _add(self, n):
        pending = [n]
        refs, recipes, initial, live = self.refs, self.recipes, self.initial, self.live
        while pending:
            n = pending.pop()
            refs[n] += 1
            if refs[n] != 1 or n in initial:
                continue
            live.add(n)
            _, a, b = recipes[n]
            pending.append(a)
            if b != a:
                pending.append(b)

    def _remove(self, n):
        pending = [n]
        refs, recipes, initial, live = self.refs, self.recipes, self.initial, self.live
        while pending:
            n = pending.pop()
            refs[n] -= 1
            if refs[n] or n in initial:
                continue
            live.remove(n)
            _, a, b = recipes[n]
            pending.append(a)
            if b != a:
                pending.append(b)

    def replace(self, n, recipe):
        previous = self.recipes[n]
        if previous == recipe:
            return previous
        if self.refs[n] and n not in self.initial:
            _, a, b = previous
            _, x, y = recipe
            # Add first so overlapping dependency cones stay active. Sources
            # precede their destination, so propagation cannot visit n itself.
            if x != a and x != b:
                self._add(x)
            if y != x and y != a and y != b:
                self._add(y)
            if a != x and a != y:
                self._remove(a)
            if b != a and b != x and b != y:
                self._remove(b)
        self.recipes[n] = recipe
        return previous


def incremental_checks():
    """Differential test edits, rollback, aliases, roots and shared cones."""
    rng = random.Random(834192)
    size = 640
    initial = {0, 1, 2, 3}
    recipes = [None] * 4
    pools = [[] for _ in range(size)]
    for n in range(4, size):
        for _ in range(4):
            # A mixture of short edges, far edges and doubled sources.
            a = rng.randrange(n)
            b = a if rng.random() < .2 else rng.randrange(n)
            pools[n].append(('+', a, b))
        recipes.append(pools[n][0])
    roots = list(range(size - 30, size)) + [0, 100, 200]

    def reference(state):
        live = set()
        pending = list(roots)
        while pending:
            n = pending.pop()
            if n in initial or n in live:
                continue
            live.add(n)
            pending.extend(state[n][1:])
        counts = [0] * size
        for n in roots:
            counts[n] += 1
        for n in live:
            _, a, b = state[n]
            counts[a] += 1
            if a != b:
                counts[b] += 1
        return live, counts

    engine = IncrementalClosure(recipes, roots, initial)
    moves = []
    for k in range(1600):
        active = sorted(engine.live)
        edits = [(rng.choice(active) if active and rng.random() < .85
                  else rng.randrange(4, size), None)
                 for _ in range(1 if k % 5 else 6)]
        edits = [(n, rng.choice(pools[n])) for n, _ in edits]
        accept = rng.random() < .6
        moves.append((edits, accept))
        undo = []
        for n, recipe in edits:
            undo.append((n, engine.replace(n, recipe)))
        live, refs = reference(engine.recipes)
        assert engine.live == live and engine.refs == refs
        if not accept:
            for n, recipe in reversed(undo):
                engine.replace(n, recipe)
            live, refs = reference(engine.recipes)
            assert engine.live == live and engine.refs == refs

    def fast_run():
        state = IncrementalClosure(recipes, roots, initial)
        total = 0
        for edits, accept in moves:
            undo = [(n, state.replace(n, recipe)) for n, recipe in edits]
            total += len(state.live)
            if not accept:
                for n, recipe in reversed(undo):
                    state.replace(n, recipe)
        return total, state.recipes

    # Time the original closure-only traversal, excluding reference-count
    # verification so the comparison reflects the actual replaced hot path.
    def baseline_run():
        current = list(recipes)
        total = 0
        for edits, accept in moves:
            trial = list(current)
            for n, recipe in edits:
                trial[n] = recipe
            seen = set()
            pending = list(roots)
            while pending:
                n = pending.pop()
                if n in initial or n in seen:
                    continue
                seen.add(n)
                _, a, b = trial[n]
                pending.append(a)
                pending.append(b)
            total += len(seen)
            if accept:
                current = trial
        return total, current

    start = time.perf_counter()
    expected = baseline_run()
    full_time = time.perf_counter() - start
    start = time.perf_counter()
    actual = fast_run()
    fast_time = time.perf_counter() - start
    assert actual == expected
    print('Incremental closure: 1600 trials exactly matched; evaluation speedup',
          round(full_time / max(fast_time, 1e-9), 2), 'x', flush=True)


def anneal_recipes(p, old, deadline, rng):
    """Search whole dependency closures, including neutral and uphill moves.

    Recipes always read earlier donor nodes, making every mutation acyclic.
    Cost counts the union of instructions needed for all targets; it therefore
    rewards sharing rather than the isolated length of a target expression.
    """
    dag = ArithmeticDAG(p, old)
    initial = set(dag.initial)
    original = list(dag.recipes)
    pool = [[recipe] if recipe is not None else [] for recipe in original]
    index = {v: n for n, v in enumerate(dag.values)}
    selected = dag.closure()
    compact = dag.allocate(selected)
    best = old
    if compact is not None and len(compact) < len(best) and replay(p, compact):
        best = compact
    best_recipes = list(original)
    # Restrict dictionary size on very long baselines; retain target roots,
    # seeds and frequently used dependencies before sampling other values.
    uses = [0] * len(original)
    for n in selected:
        for a in set(original[n][1:]):
            uses[a] += 1
    sources = list(range(len(original)))
    if len(sources) > 700:
        roots = set(dag.roots)
        sources.sort(key=lambda n: (n not in initial, n not in roots,
                                    n not in selected, -uses[n], n))
        sources = sources[:700]
    harvest_end = min(deadline, time.monotonic() +
                      max(0, deadline - time.monotonic()) * .22)
    values = dag.values

    def offer(value, op, a, b):
        n = index.get(value)
        if n is None or n in initial or n <= max(a, b):
            return
        recipe = (op, a, b)
        if recipe not in pool[n]:
            pool[n].append(recipe)

    # Every matching result is a donor node, so all harvested operations are
    # positive and within value_limit without constructing speculative values.
    for k, a in enumerate(sources):
        if k % 8 == 0 and time.monotonic() >= harvest_end:
            break
        av = values[a]
        for b in sources[k:]:
            bv = values[b]
            offer(av + bv, '+', a, b)
            if av > bv:
                offer(av - bv, '-', a, b)
            elif bv > av:
                offer(bv - av, '-', b, a)
            if av <= p['limit'] // bv:
                offer(av * bv, '*', a, b)
            if av % bv == 0:
                offer(av // bv, '/', a, b)
            if bv % av == 0:
                offer(bv // av, '/', b, a)

    mutable = [n for n, recipes in enumerate(pool) if len(recipes) > 1]
    if not mutable or time.monotonic() >= deadline:
        return best, 0, 0, len(mutable)
    roots = dag.roots
    initial_mask = sum(1 << n for n in initial)

    def masks(recipes):
        # Integer bitsets make the backward rebuild's marginal-cost estimate
        # cheap; exact union size is recomputed before accepting the move.
        result = [0] * len(recipes)
        for n, recipe in enumerate(recipes):
            if recipe is not None:
                _, a, b = recipe
                result[n] = result[a] | result[b] | (1 << n)
        return result

    state = IncrementalClosure(original, roots, initial)
    current = state.recipes
    live = state.live
    cost = len(live)
    best_cost = cost
    best_state = list(current)
    trials = hits = 0
    start = time.monotonic()
    duration = max(.001, deadline - start)
    while time.monotonic() < deadline:
        trials += 1
        undo = []
        if trials % 17 == 0:
            trial = list(current)
            # Large neighborhood: rebuild backward from all roots. Operands
            # already required by other targets get a direct sharing credit.
            ancestor = masks(current)
            required = sum(1 << n for n in roots)
            noise = rng.uniform(.15, 1.8)
            for n in range(len(trial) - 1, -1, -1):
                if not (required >> n) & 1 or n in initial:
                    continue
                options = pool[n]
                if len(options) > 1:
                    def rank(recipe):
                        _, a, b = recipe
                        new = ((1 << a) | (1 << b)) & ~required
                        new &= ~initial_mask
                        cone = (ancestor[a] | ancestor[b]) & ~required
                        return new.bit_count() + .16 * cone.bit_count() + rng.random() * noise
                    trial[n] = min(options, key=rank)
                _, a, b = trial[n]
                required |= (1 << a) | (1 << b)
        else:
            trial = None
            active = [n for n in mutable if n in live]
            count = 1 if rng.random() < .78 else rng.randint(2, 7)
            for _ in range(count):
                choices = active if active and rng.random() < .85 else mutable
                n = rng.choice(choices)
                recipe = rng.choice(pool[n])
                undo.append((n, state.replace(n, recipe)))
        if trial is not None:
            for n, recipe in enumerate(trial):
                if recipe != current[n]:
                    undo.append((n, state.replace(n, recipe)))
        new_cost = len(live)
        delta = new_cost - cost
        phase = ((time.monotonic() - start) / duration * 5) % 1
        temperature = 1.7 * (1 - phase) ** 2 + .06
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_state = list(current)
            if cost < len(best):
                dag.recipes = current
                candidate = dag.allocate(live)
                if candidate is not None and replay(p, candidate):
                    best = candidate
                    best_recipes = list(current)
                    hits += 1
        else:
            for n, recipe in reversed(undo):
                state.replace(n, recipe)
        if trials % 211 == 0:
            # Reheat around the shortest closure, allowing neutral recipes to
            # change while keeping the valid output independently protected.
            state = IncrementalClosure(
                best_state if rng.random() < .8 else best_recipes, roots, initial)
            current, live = state.recipes, state.live
            cost = len(live)
    return best, hits, trials, len(mutable)


def annealing_checks():
    # 30 and 35 have separate two-step recipes. Sharing the initially unused
    # donor 5 yields both in three steps, but changing only 30 is cost neutral.
    p = dict(start=[1, 6, 7, 10], targets=[30, 35], limit=100)
    old = [['-', 1, 0, 9], ['+', 3, 3, 4], ['+', 4, 3, 5],
           ['*', 2, 2, 6], ['+', 2, 2, 7], ['-', 6, 7, 8]]
    assert replay(p, old)
    new, hits, trials, alternatives = anneal_recipes(
        p, old, time.monotonic() + .12, random.Random(27))
    assert replay(p, new) and len(new) == 3, (len(new), trials, alternatives)
    # Exact division and destination aliasing are independently replayed.
    p = dict(start=[9, 3], targets=[1, 2, 6, 18], limit=18)
    old = [['/', 0, 1, 2], ['/', 1, 1, 3], ['-', 1, 3, 4],
           ['*', 1, 4, 5], ['*', 0, 4, 0]]
    assert replay(p, old)
    new, _, _, _ = anneal_recipes(p, old, time.monotonic() + .03, random.Random(28))
    assert replay(p, new) and len(new) <= len(old)
    print('Recipe annealing: shared closures, exact division and aliases passed', flush=True)


def main():
    o = kt.optimization()
    if o['smoke']:
        smoke_checks()
        incremental_checks()
        annealing_checks()
    if o['smoke']:
        # Exercise shared branches, subtraction, fully occupied registers and
        # destination/source aliasing independently of the supplied baseline.
        p = dict(start=[11, 7], targets=[18, 324, 313], limit=324)
        c = ConstructCache(p)
        shared = ('+', 0, 1)
        c.emit(('-', ('*', shared, shared), 0))
        assert len(c.program) == 3 and replay(p, c.program)
        p = dict(start=list(range(101, 149)), targets=[sum(range(101, 149))], limit=10000)
        c = ConstructCache(p)
        c.protected = set()
        expr = 0
        for r in range(1, 48):
            expr = ('+', expr, r)
        c.emit(expr)
        assert replay(p, c.program)
        print('Adaptive cache: shared nodes, signed residuals and 48-register aliases passed', flush=True)
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
    if set(current) != set(puzzles):
        raise ValueError('baseline IDs do not match puzzles')
    for pid, p in puzzles.items():
        if not replay(p, current[pid]):
            raise ValueError('invalid baseline: ' + pid)

    if o['smoke']:
        fallback_checks(puzzles)

    ids = sorted(puzzles)[:6] if o['smoke'] else sorted(puzzles)
    before = sum(len(current[pid]) for pid in ids)
    deadline = time.monotonic() + (5.0 if o['smoke'] else 3000.0)
    for k, pid in enumerate(ids):
        p, old = puzzles[pid], current[pid]
        end = time.monotonic() + max(0, deadline - time.monotonic()) / (len(ids) - k)
        for radix in (2, 4, 8, 16, 32):
            fallback = horner_fallback(p, radix)
            if len(fallback) < len(current[pid]) and replay(p, fallback):
                current[pid] = fallback
        try:
            trial = construct_remainders(p, time.monotonic() + max(0, end - time.monotonic()) * .22)
        except (ValueError, KeyError, RecursionError) as exc:
            print(pid, 'construction discarded:', str(exc), flush=True)
            trial = None
        if trial is not None and len(trial) < len(current[pid]) and replay(p, trial):
            current[pid] = trial
        construction_length = len(trial) if trial is not None else None
        # The incumbent provides cheap retained operands. Lift an exact E1 x
        # E1 join by another base/E1 operation and replace only an exclusive
        # cone. The allocator keeps shared intermediates in real registers;
        # a lower closure size alone is never sufficient for acceptance.
        deep_end = time.monotonic() + max(0, end - time.monotonic()) * .65
        new, deep_hits = optimize(p, current[pid], deep_end)
        if len(new) < len(current[pid]) and replay(p, new):
            current[pid] = new
        new, hits, trials, alternatives = anneal_recipes(
            p, current[pid], end, random.Random(o['seed'] + k))
        if len(new) < len(current[pid]) and replay(p, new):
            current[pid] = new
        print(pid, len(old), '->', len(current[pid]),
              'adaptive construction', construction_length, 'annealing hits', hits,
              'deep joins', deep_hits, 'trials', trials,
              'mutable nodes', alternatives, flush=True)

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


