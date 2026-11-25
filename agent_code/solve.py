"""Incumbent-guided arithmetic DAG rewrites with target-order annealing.

Search a larger neighborhood than instruction deletion: every intermediate may
use any equivalent one-operation recipe over earlier incumbent values. Rebuild
only the union of dependencies needed by the targets, then allocate 48 physical
registers. An independent replay guards every accepted improvement.
"""
import csv
import heapq
import json
import math
import random
import time
from pathlib import Path

from kaggle_agent import toolkit as kt


def calc(op, a, b, limit):
    if op == '+':
        value = a + b
    elif op == '-':
        value = a - b
    elif op == '*':
        value = a * b
    elif op == '/' and b and a % b == 0:
        value = a // b
    else:
        return None
    return value if 0 < value <= limit else None


def replay(p, program):
    """Check all official per-program constraints, including serialized size."""
    if not isinstance(program, list) or len(program) > 4096:
        return False
    if len(json.dumps(program, separators=(',', ':')).encode('utf-8')) > 131072:
        return False
    regs = list(p['start']) + [None] * (48 - len(p['start']))
    remaining = set(p['targets']) - set(p['start'])
    for instruction in program:
        if not isinstance(instruction, list) or len(instruction) != 4:
            return False
        op, i, j, d = instruction
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


class RewriteSpace:
    def __init__(self, p, program, deadline):
        self.values = []
        self.index = {}
        self.initial = {}
        self.original = []

        def add(value, recipe=None):
            previous = self.index.get(value)
            if previous is not None:
                return previous
            node = len(self.values)
            self.index[value] = node
            self.values.append(value)
            self.original.append(recipe)
            return node

        regs = [None] * 48
        for r, value in enumerate(p['start']):
            node = add(value)
            self.initial.setdefault(node, r)
            regs[r] = node
        for op, i, j, d in program:
            a, b = regs[i], regs[j]
            value = calc(op, self.values[a], self.values[b], p['limit'])
            regs[d] = add(value, (op, a, b))

        self.roots = sorted({self.index[t] for t in p['targets']}
                            - self.initial.keys())
        self.recipes = [[] for _ in self.values]
        closures = [0] * len(self.values)
        # The original recipe is always retained. Alternatives strictly point
        # backwards in first-occurrence order, so every candidate stays acyclic.
        self.alternatives = 0
        for node, value in enumerate(self.values):
            if node in self.initial:
                continue
            original = self.original[node]
            candidates = {original}
            if time.monotonic() < deadline:
                for a, av in enumerate(self.values[:node]):
                    possibilities = [('+', value - av, a, False),
                                     ('-', av - value, a, False),
                                     ('-', av + value, a, True)]
                    if value % av == 0:
                        possibilities.append(('*', value // av, a, False))
                    if av % value == 0:
                        possibilities.append(('/', av // value, a, False))
                    if av <= p['limit'] // value:
                        possibilities.append(('/', av * value, a, True))
                    for op, bv, left, reverse in possibilities:
                        b = self.index.get(bv)
                        if b is None or b >= node:
                            continue
                        left, right = (b, left) if reverse else (left, b)
                        if op in ('+', '*') and left > right:
                            left, right = right, left
                        candidates.add((op, left, right))

            def rank(recipe):
                _, a, b = recipe
                return ((closures[a] | closures[b]).bit_count(),
                        recipe != original, max(a, b), recipe)

            ranked = sorted(candidates, key=rank)
            # Keep diverse direct recipes, rather than an unbounded quadratic
            # frontier of expensive ways to compute the same small constant.
            keep = ranked[:24]
            if original not in keep:
                keep.append(original)
            self.recipes[node] = keep
            _, a, b = ranked[0]
            closures[node] = closures[a] | closures[b] | (1 << node)
            self.alternatives += len(keep) - 1

    def choose(self, order, tie_seed=0):
        """Build target closures against the nodes selected for earlier targets.

        Bit masks count shared intermediate nodes exactly once. Recomputing the
        forward choices after each target makes already selected work free.
        """
        count = len(self.values)
        selected = 0
        choices = [None] * count
        # A small deterministic tie perturbation gives the order search access
        # to different equal-cost shared subgraphs without changing the metric.
        for root in order:
            if selected & (1 << root):
                continue
            masks = [0] * (root + 1)
            local = [None] * (root + 1)
            for node in range(root + 1):
                if node in self.initial or selected & (1 << node):
                    continue
                winner = None
                best_key = None
                best_mask = 0
                for ri, recipe in enumerate(self.recipes[node]):
                    _, a, b = recipe
                    mask = masks[a] | masks[b] | (1 << node)
                    tie = ((ri * 2654435761 + node * 2246822519 + tie_seed)
                           ^ (tie_seed * (ri + 1))) & 0xffff if tie_seed else ri
                    key = (mask.bit_count(), tie)
                    if best_key is None or key < best_key:
                        best_key, best_mask, winner = key, mask, recipe
                masks[node] = best_mask
                local[node] = winner
            new = masks[root]
            pending = new
            while pending:
                bit = pending & -pending
                node = bit.bit_length() - 1
                choices[node] = local[node]
                pending ^= bit
            selected |= new
        return self.allocate(choices, selected)

    def original_program(self):
        choices = self.original[:]
        selected = 0
        todo = self.roots[:]
        while todo:
            node = todo.pop()
            if node in self.initial or selected & (1 << node):
                continue
            selected |= 1 << node
            _, a, b = choices[node]
            todo.extend((a, b))
        return self.allocate(choices, selected)

    def allocate(self, choices, selected):
        """Schedule the chosen DAG, freeing registers on the last source read."""
        nodes = []
        pending = selected
        while pending:
            bit = pending & -pending
            nodes.append(bit.bit_length() - 1)
            pending ^= bit
        uses = [0] * len(self.values)
        users = [[] for _ in self.values]
        dependencies = {}
        for node in nodes:
            _, a, b = choices[node]
            operands = set((a, b))
            dependencies[node] = sum(v not in self.initial for v in operands)
            for operand in operands:
                uses[operand] += 1
                users[operand].append(node)
        physical = {node: r for node, r in self.initial.items() if uses[node]}
        occupied = set(physical.values())
        free = set(range(48)) - occupied
        ready = [node for node in nodes if dependencies[node] == 0]
        heapq.heapify(ready)
        out = []
        while ready:
            if len(free) <= 3:
                # Under pressure, consume a dying source before opening a new
                # live range. The ordinary schedule follows incumbent order.
                node = min(ready, key=lambda n: (
                    1 - sum(uses[a] == 1 for a in set(choices[n][1:])), n))
                ready.remove(node)
                heapq.heapify(ready)
            else:
                node = heapq.heappop(ready)
            op, a, b = choices[node]
            i, j = physical[a], physical[b]
            dying = [operand for operand in set((a, b)) if uses[operand] == 1]
            if dying:
                d = min(physical[operand] for operand in dying)
            elif free:
                d = min(free)
            else:
                return None
            out.append([op, i, j, d])
            for operand in set((a, b)):
                uses[operand] -= 1
                if uses[operand] == 0:
                    free.add(physical.pop(operand))
            free.discard(d)
            if uses[node]:
                physical[node] = d
            else:
                # A target's appearance covers it permanently, so it needs no
                # lifetime after its final arithmetic consumer has executed.
                free.add(d)
            for user in users[node]:
                dependencies[user] -= 1
                if dependencies[user] == 0:
                    heapq.heappush(ready, user)
        return out if len(out) == len(nodes) else None


def mutate(order, rng):
    trial = order[:]
    if len(trial) < 2:
        return trial
    a, b = sorted(rng.sample(range(len(trial)), 2))
    kind = rng.randrange(4)
    if kind == 0:
        trial[a], trial[b] = trial[b], trial[a]
    elif kind == 1:
        trial.insert(a, trial.pop(b))
    elif kind == 2:
        trial[a:b + 1] = reversed(trial[a:b + 1])
    else:
        block = trial[a:b + 1]
        del trial[a:b + 1]
        pos = rng.randrange(len(trial) + 1)
        trial[pos:pos] = block
    return trial


def optimize(p, old, deadline, rng):
    best = old
    space = RewriteSpace(p, old, deadline)
    attempts = 0

    def consider(candidate):
        nonlocal best, attempts
        attempts += 1
        if candidate is not None and len(candidate) < len(best) and replay(p, candidate):
            best = candidate

    consider(space.original_program())
    if not space.roots:
        return best, attempts, space.alternatives
    order = space.roots[::-1]
    current_cost = len(old)
    elite_order = order[:]
    elite_cost = current_cost
    begin = time.monotonic()
    stale = 0
    iteration = 0
    while time.monotonic() < deadline:
        if iteration == 0:
            trial = space.roots[::-1]
        elif iteration == 1:
            trial = space.roots[:]
        elif iteration == 2:
            trial = sorted(space.roots, key=lambda n: space.values[n] % 65536)
        else:
            trial = mutate(order, rng)
        candidate = space.choose(trial, 0 if iteration < 3 else rng.getrandbits(16))
        consider(candidate)
        score = len(candidate) if candidate is not None else len(old) + 100
        progress = (time.monotonic() - begin) / max(0.001, deadline - begin)
        temperature = max(0.15, 2.5 * (1 - progress))
        if score <= current_cost or rng.random() < math.exp(min(0, (current_cost - score) / temperature)):
            current_cost, order = score, trial
        if score < elite_cost:
            elite_cost, elite_order, stale = score, trial[:], 0
        else:
            stale += 1
        if stale >= 12:
            current_cost, order, stale = elite_cost, elite_order[:], 0
        iteration += 1
        # One operation per target absent from the starting values is the
        # official lower bound; no further neighborhood can improve it.
        if len(best) == len(space.roots):
            break
    return best, attempts, space.alternatives


def smoke_checks():
    # Producing 41 by a 3-step branch hides the alternative 50-9. Replacing
    # that entire dependency cone is impossible by single-instruction deletion.
    p = dict(start=[50, 9, 1], targets=[41, 82], limit=1000)
    old = [['+', 2, 2, 3], ['*', 3, 1, 4], ['+', 4, 4, 5],
           ['+', 5, 3, 6], ['+', 6, 3, 7], ['+', 7, 2, 8],
           ['+', 8, 8, 9]]
    assert replay(p, old)
    space = RewriteSpace(p, old, time.monotonic() + 1)
    new = space.choose(space.roots[::-1])
    assert replay(p, new) and len(new) == 2, new
    print('Arithmetic cone rewrite:', len(old), '->', len(new), flush=True)
    # Exact division, destructive aliases, duplicate values, initially covered
    # targets and full-register pressure are all legal calculator behaviors.
    cases = [
        (dict(start=[3], targets=[1, 2, 5, 8], limit=9),
         [['/', 0, 0, 1], ['-', 0, 1, 2], ['+', 0, 2, 3], ['+', 0, 3, 4]]),
        (dict(start=[2, 2, 7], targets=[2, 14, 1], limit=14),
         [['*', 0, 2, 2], ['/', 1, 1, 0]]),
        (dict(start=list(range(1, 49)), targets=[96, 47, 2], limit=96),
         [['+', 47, 47, 47], ['/', 47, 23, 0], ['-', 47, 0, 1]]),
        (dict(start=[1], targets=[1], limit=1), []),
    ]
    for puzzle, program in cases:
        assert replay(puzzle, program)
        graph = RewriteSpace(puzzle, program, time.monotonic() + 1)
        for order in (graph.roots, graph.roots[::-1]):
            candidate = graph.choose(order)
            assert candidate is not None and replay(puzzle, candidate)
        candidate = graph.original_program()
        assert candidate is not None and replay(puzzle, candidate)
    # Random traces exercise lifetime allocation with arbitrary source writes.
    rng = random.Random(63743)
    for full in (False, True):
        for _ in range(12):
            start = list(range(1, 49)) if full else [3, 5, 7]
            regs = start[:] + [None] * (48 - len(start))
            program = []
            targets = set(start[:2])
            for step in range(75):
                live = [r for r in range(48) if regs[r] is not None]
                i, j = rng.choice(live), rng.choice(live)
                op = rng.choice(('+', '-', '*', '/'))
                value = calc(op, regs[i], regs[j], 10**8)
                if value is None:
                    continue
                d = rng.randrange(48)
                program.append([op, i, j, d])
                regs[d] = value
                if step % 9 == 0:
                    targets.add(value)
            puzzle = dict(start=start, targets=sorted(targets), limit=10**8)
            assert replay(puzzle, program)
            graph = RewriteSpace(puzzle, program, time.monotonic() + 1)
            for order in (graph.roots, graph.roots[::-1]):
                candidate = graph.choose(order, rng.getrandbits(16))
                assert candidate is None or replay(puzzle, candidate)
    print('DAG replay and allocation checks passed', flush=True)


def main():
    options = kt.optimization()
    if options['smoke']:
        smoke_checks()
    data = Path(options['data_dir'])

    def locate(name):
        nested = data / 'data' / name
        return nested if nested.is_file() else data / name

    with locate('puzzles.csv').open(newline='', encoding='utf-8') as f:
        puzzles = {row['id']: dict(start=json.loads(row['starting_values']),
                                   targets=json.loads(row['targets']),
                                   limit=int(row['value_limit']))
                   for row in csv.DictReader(f)}
    with Path(options['best'] or locate('sample_submission.csv')).open(newline='', encoding='utf-8') as f:
        current = {row['id']: json.loads(row['program']) for row in csv.DictReader(f)}
    # Preserve every untouched puzzle, and refuse to start from invalid input.
    if set(current) != set(puzzles):
        raise ValueError('baseline puzzle IDs do not match data')
    for pid, puzzle in puzzles.items():
        if not replay(puzzle, current[pid]):
            raise ValueError('invalid baseline: ' + pid)
    ids = sorted(puzzles)[:6] if options['smoke'] else sorted(puzzles)
    rng = random.Random(options['seed'])
    before = sum(len(current[pid]) for pid in ids)
    deadline = time.monotonic() + (4.0 if options['smoke'] else 3000.0)
    for index, pid in enumerate(ids):
        old = current[pid]
        allocation = max(0.0, deadline - time.monotonic()) / (len(ids) - index)
        candidate, attempts, alternatives = optimize(
            puzzles[pid], old, time.monotonic() + allocation, rng)
        if len(candidate) < len(old) and replay(puzzles[pid], candidate):
            current[pid] = candidate
        print(pid, 'before', len(old), 'after', len(current[pid]),
              'trials', attempts, 'alternative recipes', alternatives, flush=True)
    out = Path(options['solution'])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'program'])
        for pid, puzzle in sorted(puzzles.items()):
            if not replay(puzzle, current[pid]):
                raise ValueError('invalid output: ' + pid)
            writer.writerow([pid, json.dumps(current[pid], separators=(',', ':'))])
    print('Validated', len(puzzles), 'puzzles; selected total', before, '->',
          sum(len(current[pid]) for pid in ids), flush=True)


if __name__ == '__main__':
    main()
