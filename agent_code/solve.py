"""Local search post-optimizer: start from best solution, remove dead instructions
(liveness + first-cover of targets), then try single-instruction deletions with full replay."""
import csv, json, sys, time
from pathlib import Path
from multiprocessing import Pool
from kaggle_agent import toolkit as kt


def replay(puz, prog):
    regs = [None] * 48
    for k, v in enumerate(puz["start"]):
        regs[k] = v
    rem = set(puz["targets"]) - set(puz["start"])
    L = puz["limit"]
    for op, i, j, d in prog:
        a, b = regs[i], regs[j]
        if a is None or b is None: return False
        if op == "+": v = a + b
        elif op == "*": v = a * b
        elif op == "-":
            if a <= b: return False
            v = a - b
        else:
            if a % b: return False
            v = a // b
        if v > L: return False
        regs[d] = v
        rem.discard(v)
    return not rem


def dce(puz, prog):
    regs = [None] * 48
    for k, v in enumerate(puz["start"]):
        regs[k] = v
    tg = set(puz["targets"]) - set(puz["start"])
    first = {}
    for n, (op, i, j, d) in enumerate(prog):
        a, b = regs[i], regs[j]
        v = a + b if op == "+" else a * b if op == "*" else a - b if op == "-" else a // b
        regs[d] = v
        if v in tg and v not in first: first[v] = n
    need_idx = set(first.values())
    live = set()
    keep = [False] * len(prog)
    for n in range(len(prog) - 1, -1, -1):
        op, i, j, d = prog[n]
        if n in need_idx or d in live:
            keep[n] = True
            live.discard(d)
            live.add(i); live.add(j)
    return [p for p, k in zip(prog, keep) if k]


def improve(puz, prog, tlimit):
    p = dce(puz, prog)
    if not replay(puz, p): return prog
    t0 = time.time()
    changed = True
    while changed and time.time() - t0 < tlimit:
        changed = False
        n = len(p) - 1
        while n >= 0 and time.time() - t0 < tlimit:
            q = p[:n] + p[n + 1:]
            if replay(puz, q):
                p2 = dce(puz, q)
                p = p2 if replay(puz, p2) else q
                changed = True
                n = min(n, len(p)) - 1
            else:
                n -= 1
    return p


def work(args):
    pid, puz, prog, tl = args
    try:
        p = improve(puz, prog, tl)
        if replay(puz, p) and len(p) < len(prog): return pid, p
    except Exception:
        pass
    return pid, None


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
    tl = 3 if o["smoke"] else 120
    if o["smoke"]: ids = ids[:6]
    jobs = [(i, puzzles[i], cur[i], tl) for i in ids]
    if o["smoke"]: res = [work(a) for a in jobs]
    else:
        with Pool(max(1, int(o["threads"]))) as pool: res = pool.map(work, jobs, chunksize=1)
    imp = saved = 0
    for pid, p in res:
        if p is not None:
            saved += len(cur[pid]) - len(p); cur[pid] = p; imp += 1
    print("improved", imp, "saved", saved, file=sys.stderr)
    out = Path(o["solution"]); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "program"])
        for pid in sorted(puzzles):
            w.writerow([pid, json.dumps(cur[pid], separators=(",", ":"))])


if __name__ == "__main__":
    main()
