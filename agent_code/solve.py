"""Greedy construction: per step pick the cheapest remaining target via meet-in-the-middle
over expression sets E0 (registers), E1 (one op), E2 (E1 op E0); fallback = approximate
with nearest reachable value and recurse on the gap."""
import csv, json, sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from kaggle_agent import toolkit as kt

CAP = 22          # max persistent values used in search
MAXC = 5


def check(puz, prog):
    regs = [None] * 48
    for k, v in enumerate(puz["start"]):
        regs[k] = v
    rem = set(puz["targets"]) - set(puz["start"])
    L = puz["limit"]
    for op, i, j, d in prog:
        a, b = regs[i], regs[j]
        if a is None or b is None:
            return None
        if op == "+": v = a + b
        elif op == "*": v = a * b
        elif op == "-":
            if a <= b: return None
            v = a - b
        else:
            if a % b: return None
            v = a // b
        if v > L: return None
        regs[d] = v
        rem.discard(v)
    return len(prog) if not rem else None


def binop(a, b, code, L):
    """a,b int64 arrays (broadcast ok). codes: 0 +,1 *,2 a-b,3 b-a,4 a/b,5 b/a. returns val, mask"""
    a = np.broadcast_to(a, np.broadcast(a, b).shape)
    b = np.broadcast_to(b, a.shape)
    if code == 0:
        m = a <= L - b
        v = np.where(m, a + np.where(m, b, 0), 1)
    elif code == 1:
        m = a <= L // b
        v = np.where(m, a * np.where(m, b, 0), 1)
    elif code == 2:
        m = a > b; v = a - b
    elif code == 3:
        m = b > a; v = b - a
    elif code == 4:
        m = (a % b) == 0; v = a // b
    else:
        m = (b % a) == 0; v = b // a
    return v, m


class Level:
    def __init__(self, vals, ops, l, r):
        o = np.argsort(vals, kind="stable")
        self.vals, self.ops, self.l, self.r = vals[o], ops[o], l[o], r[o]


def build(Vv, L):
    n = len(Vv)
    Va = np.array(Vv, dtype=np.int64)
    # E1
    vs, os_, ls, rs = [], [], [], []
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    for code in range(5):
        v, m = binop(Va[ii], Va[jj], code, L)
        if code in (0, 1): m = m & (ii <= jj)
        if code == 4: m = m & (ii != jj)
        vs.append(v[m]); os_.append(np.full(m.sum(), code)); ls.append(ii[m]); rs.append(jj[m])
    v1 = np.concatenate(vs); o1 = np.concatenate(os_); l1 = np.concatenate(ls); r1 = np.concatenate(rs)
    # dedupe vs V and within
    keep = ~np.isin(v1, Va)
    v1, o1, l1, r1 = v1[keep], o1[keep], l1[keep], r1[keep]
    v1, idx = np.unique(v1, return_index=True)
    E1 = Level(v1, o1[idx], l1[idx], r1[idx])
    # E2 = E1 x V
    m1 = len(E1.vals)
    ii, jj = np.meshgrid(np.arange(m1), np.arange(n), indexing="ij")
    vs, os_, ls, rs = [], [], [], []
    for code in range(6):
        v, m = binop(E1.vals[ii], Va[jj], code, L)
        vs.append(v[m]); os_.append(np.full(m.sum(), code)); ls.append(ii[m]); rs.append(jj[m])
    v2 = np.concatenate(vs); o2 = np.concatenate(os_); l2 = np.concatenate(ls); r2 = np.concatenate(rs)
    keep = ~(np.isin(v2, Va) | np.isin(v2, E1.vals))
    v2, o2, l2, r2 = v2[keep], o2[keep], l2[keep], r2[keep]
    v2, idx = np.unique(v2, return_index=True)
    E2 = Level(v2, o2[idx], l2[idx], r2[idx])
    E0 = Level(Va.copy(), np.zeros(n, dtype=np.int64), np.arange(n), np.arange(n))
    return [E0, E1, E2]


def node(lv, k, E, Vreg):
    """expression tree for element k of level lv."""
    if lv == 0:
        return ("v", Vreg[int(E[0].l[k])])
    if lv == 1:
        code = int(E[1].ops[k])
        return ("op", code, ("v", Vreg[int(E[1].l[k])]), ("v", Vreg[int(E[1].r[k])]))
    code = int(E[2].ops[k])
    return ("op", code, node(1, int(E[2].l[k]), E, Vreg), ("v", Vreg[int(E[2].r[k])]))


def partner(t, s, code, L):
    """For t = s OP p (code semantic: a=s,b=p): return p array and mask."""
    # codes: 0 s+p, 1 s*p, 2 s-p, 3 p-s, 4 s/p, 5 p/s
    if code == 0:
        return t - s, s < t
    if code == 1:
        sm = np.where(s > 0, s, 1)
        return t // sm, (t % sm) == 0
    if code == 2:
        return s - t, s > t
    if code == 3:
        return t + s, s <= L - t
    if code == 4:
        return s // t, (s % t) == 0
    m = s <= L // t
    return t * np.where(m, s, 0), m


def lookup(t, small, big, L):
    """find small elem s, big elem p, code with t = s OP p. return (i,j,code) or None"""
    s = small.vals
    for code in range(6):
        p, m = partner(t, s, code, L)
        if not m.any():
            continue
        p = np.where(m, p, -1)
        pos = np.searchsorted(big.vals, p)
        pos = np.minimum(pos, len(big.vals) - 1)
        hit = m & (big.vals[pos] == p)
        if hit.any():
            i = int(np.argmax(hit))
            return i, int(pos[i]), code
    return None


PAIRS = {1: [(0, 0)], 2: [(0, 1)], 3: [(1, 1), (0, 2)], 4: [(1, 2)], 5: [(2, 2)]}


def search(t, E, Vreg, L, maxc=MAXC):
    for c in range(1, maxc + 1):
        for (a, b) in PAIRS[c]:
            A, B = E[a], E[b]
            if len(A.vals) == 0 or len(B.vals) == 0:
                continue
            small, big, sl, bl = (A, B, a, b) if len(A.vals) <= len(B.vals) else (B, A, b, a)
            r = lookup(t, small, big, L)
            if r:
                i, j, code = r
                # t = s OP p, code semantic with a=s, b=p
                return c, ("op", code, node(sl, i, E, Vreg), node(bl, j, E, Vreg))
    return None


def tree_cost(tr):
    return 0 if tr[0] == "v" else 1 + tree_cost(tr[2]) + tree_cost(tr[3])


def solve_value(x, E, Vreg, L, depth=0):
    """full tree for x (fallback allowed)."""
    if depth > 12:
        return None
    r = search(x, E, Vreg, L)
    if r:
        return r[1]
    # approximate: nearest value in E0..E2
    best = None
    for lv in range(3):
        vals = E[lv].vals
        if len(vals) == 0: continue
        p = int(np.searchsorted(vals, x))
        for q in (p - 1, p):
            if 0 <= q < len(vals) and vals[q] != x:
                g = abs(int(vals[q]) - x)
                score = g
                if best is None or score < best[0]:
                    best = (score, lv, q, int(vals[q]))
    if best is None or best[0] * 2 > x:
        return None
    g, lv, q, v = best
    sub = solve_value(g, E, Vreg, L, depth + 1)
    if sub is None:
        return None
    base = node(lv, q, E, Vreg)
    # x = v + g (v<x) or v - g (v>x)
    return ("op", 0 if v < x else 2, base, sub)


def solve_puzzle(puz):
    L = puz["limit"]
    start = puz["start"]
    rem = sorted(set(puz["targets"]) - set(start))
    prog = []
    persist = [(k, v) for k, v in enumerate(start)]  # (reg, value)
    nstart = len(start)
    regval = {k: v for k, v in persist}
    covered_all = set(start)

    def alloc():
        for r in range(48):
            if r not in regval:
                return r
        raise RuntimeError("no reg")

    def emit(tr):
        """returns (reg, is_temp)"""
        if tr[0] == "v":
            return tr[1], False
        _, code, a, b = tr
        ra, ta = emit(a)
        rb, tb = emit(b)
        if ta: d = ra
        elif tb: d = rb
        else: d = alloc()
        va, vb = regval[ra], regval[rb]
        if code == 0: op, i, j, val = "+", ra, rb, va + vb
        elif code == 1: op, i, j, val = "*", ra, rb, va * vb
        elif code == 2: op, i, j, val = "-", ra, rb, va - vb
        elif code == 3: op, i, j, val = "-", rb, ra, vb - va
        elif code == 4: op, i, j, val = "/", ra, rb, va // vb
        else: op, i, j, val = "/", rb, ra, vb // va
        if ta and tb:
            regval.pop(rb, None)
        regval[d] = val
        prog.append([op, i, j, d])
        if val in rem_set:
            rem_set.discard(val)
        return d, True

    rem_set = set(rem)
    while rem_set:
        Vreg = [r for r, _ in persist]
        Vv = [v for _, v in persist]
        E = build(Vv, L)
        cands = sorted(rem_set)
        best = None
        for c in range(1, MAXC + 1):
            for t in cands:
                r = None
                # exact cost c only: search up to c, accept if cost==c
                s = search(t, E, Vreg, L, c)
                if s:
                    best = (t, s[1]); break
            if best: break
        if best is None:
            t = cands[0]
            tr = solve_value(t, E, Vreg, L)
            if tr is None:
                # stepping stone: largest reachable value below t (prefer E2)
                sb = None
                for lv in (2, 1):
                    vals = E[lv].vals
                    q = int(np.searchsorted(vals, t)) - 1
                    if q >= 0 and (sb is None or vals[q] > sb[0] * 1.0):
                        sb = (int(vals[q]), lv, q)
                if sb is None or sb[0] <= max(Vv):
                    return None
                best = (None, node(sb[1], sb[2], E, Vreg))
            else:
                best = (t, tr)
        t, tr = best
        before = len(prog)
        # temp regs must not collide with persistent: regval has them
        r, tmp = emit(tr)
        # persist final value of t
        if tmp:
            # keep result register as persistent value; value may be t
            persist.append((r, regval[r]))
        else:
            pass
        if t is not None:
            rem_set.discard(t)
        # drop non-persistent leftover temps
        keepregs = {rr for rr, _ in persist}
        for rr in list(regval):
            if rr not in keepregs:
                regval.pop(rr)
        # cap
        while len(persist) > CAP:
            if len(persist) <= nstart:
                break
            k = min(range(nstart, len(persist)), key=lambda q: persist[q][1])
            persist.pop(k)
        keepregs = {rr for rr, _ in persist}
        for rr in list(regval):
            if rr not in keepregs:
                regval.pop(rr)
        if len(prog) > 4000:
            return None
    return prog


def work(args):
    pid, puz = args
    try:
        prog = solve_puzzle(puz)
    except Exception:
        return pid, None
    if prog is None or check(puz, prog) is None:
        return pid, None
    return pid, prog


def main():
    o = kt.optimization()
    data = Path(o["data_dir"])
    pf = data / "data" / "puzzles.csv"
    if not pf.is_file(): pf = data / "puzzles.csv"
    puzzles = {}
    with open(pf, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            puzzles[row["id"]] = dict(start=json.loads(row["starting_values"]), targets=json.loads(row["targets"]),
                                      limit=int(row["value_limit"]))
    base = o["best"] if o["best"] else data / "data" / "sample_submission.csv"
    cur = {}
    with open(base, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cur[row["id"]] = json.loads(row["program"])
    ids = sorted(puzzles)
    if o["smoke"]:
        ids = ids[:6]
        puzzles_iter = [(i, puzzles[i]) for i in ids]
        results = [work(a) for a in puzzles_iter]
    else:
        with Pool(max(1, int(o["threads"]))) as p:
            results = p.map(work, [(i, puzzles[i]) for i in ids], chunksize=1)
    imp = 0
    for pid, prog in results:
        if prog is not None and (pid not in cur or len(prog) < len(cur[pid])):
            cur[pid] = prog; imp += 1
    print("improved", imp, "of", len(results), file=sys.stderr)
    for pid, prog in results:
        print(pid, None if prog is None else len(prog), len(cur[pid]), file=sys.stderr)
    out = Path(o["solution"]); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "program"])
        for pid in sorted(puzzles):
            w.writerow([pid, json.dumps(cur[pid], separators=(",", ":"))])


if __name__ == "__main__":
    main()
