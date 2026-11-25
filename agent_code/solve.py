"""Replace baseline blocks using a register-resident E1 x E1 MITM level."""
import csv
import json
import time
from pathlib import Path
from multiprocessing import Pool
from kaggle_agent import toolkit as kt


def calc(op, a, b, limit):
    if op == '+': v = a + b
    elif op == '*': v = a * b
    elif op == '-': v = a - b
    elif op == '/' and b and a % b == 0: v = a // b
    else: return None
    return v if 0 < v <= limit else None


def replay(p, program, trace=False):
    if len(program) > 4096: return None
    regs = list(p['start']) + [None] * (48-len(p['start']))
    remaining = set(p['targets']) - set(p['start'])
    states = [regs.copy()] if trace else None
    for ins in program:
        if not isinstance(ins, list) or len(ins) != 4: return None
        op, i, j, d = ins
        if op not in '+-*/' or any(type(r) is not int or not 0 <= r < 48 for r in (i,j,d)): return None
        if regs[i] is None or regs[j] is None: return None
        v = calc(op, regs[i], regs[j], p['limit'])
        if v is None: return None
        regs[d] = v
        remaining.discard(v)
        if trace: states.append(regs.copy())
    if remaining: return None
    if len(json.dumps(program, separators=(',', ':')).encode()) > 131072: return None
    return states if trace else True


def variants(a, b, limit):
    for op, x, y in (('+',a,b),('*',a,b),('-',a,b),('-',b,a),('/',a,b),('/',b,a)):
        v = calc(op,x,y,limit)
        if v is not None: yield v, op, x, y


def match(t, left, right, limit):
    # Solve t = a op b using hash lookups; both operand orders matter.
    for a in left:
        for op,b,reverse in (('+',t-a,False),('-',a-t,False),('-',a+t,True),
                             ('*',t//a if t%a == 0 else 0,False),
                             ('/',a//t if a%t == 0 else 0,False),('/',a*t,True)):
            if b in right:
                x,y = (b,a) if reverse else (a,b)
                if calc(op,x,y,limit) == t: return (op,x,y)
    return None


class Search:
    def __init__(self, regs, limit, deadline):
        self.limit, self.deadline = limit, deadline
        self.e0 = {v: None for v in regs if v is not None}
        self.e1 = dict(self.e0)
        vals = list(self.e0)
        for i,a in enumerate(vals):
            for b in vals[i:]:
                for v,op,x,y in variants(a,b,limit):
                    if v not in self.e1: self.e1[v] = (op,x,y)
        self.e2 = None

    def build(self):
        # Bound the quadratic level, retaining scales and arithmetic diversity.
        candidates = [v for v in self.e1 if v not in self.e0]
        candidates.sort()
        cap = 220
        if len(candidates) > cap:
            candidates = [candidates[i*(len(candidates)-1)//(cap-1)] for i in range(cap)]
        pool = list(dict.fromkeys(list(self.e0) + candidates))
        self.e2 = dict(self.e1)
        for i,a in enumerate(pool):
            if i % 8 == 0 and time.monotonic() > self.deadline: return
            for b in pool[i:]:
                for v,op,x,y in variants(a,b,self.limit):
                    if v not in self.e2: self.e2[v] = (op,x,y)

    def expression(self, t):
        if t in self.e1: return self.expand(t,self.e1)
        m = match(t,self.e0,self.e1,self.limit)
        if m:
            op,x,y=m
            return (op,self.expand(x,self.e1),self.expand(y,self.e1))
        m = match(t,self.e1,self.e1,self.limit)
        if m:
            op,x,y=m
            return (op,self.expand(x,self.e1),self.expand(y,self.e1))
        if self.e2 is None: self.build()
        if t in self.e2: return self.expand(t,self.e2)
        for left in (self.e0,self.e1):
            m=match(t,left,self.e2,self.limit)
            if m:
                op,x,y=m
                return (op,self.expand(x,self.e2),self.expand(y,self.e2))
        return None

    def expand(self,v,table):
        if v in self.e0: return v
        op,x,y=table[v]
        # E2 children are always E1; this also avoids cyclic representations.
        return (op,self.expand(x,self.e1),self.expand(y,self.e1))


def synthesize(regs, required, covered, live, limit, deadline, budget):
    search=Search(regs,limit,deadline)
    goals=list(dict.fromkeys(list(required.values())+sorted(covered)))
    trees=[]
    for v in goals:
        if v in regs: continue
        tree=search.expression(v)
        if tree is None or time.monotonic()>deadline: return None
        trees.append(tree)
    # Keep every initial leaf until all expressions have been emitted. Cache every
    # intermediate in a free register so shared subexpressions cost only once.
    cache={v:i for i,v in enumerate(regs) if v is not None}
    leaves=set()
    def collect(t):
        if isinstance(t,int): leaves.add(t)
        else: collect(t[1]); collect(t[2])
    for tree in trees: collect(tree)
    protected={cache[v] for v in leaves} | {r for r in live if regs[r] == required.get(r,regs[r])}
    free=[r for r in range(48) if r not in protected and r not in live]
    out=[]
    def emit(t):
        if isinstance(t,int): return cache[t]
        op,l,r=t
        i=emit(l); j=emit(r)
        v=calc(op,local[i],local[j],limit)
        if v in cache and local[cache[v]] == v: return cache[v]
        if not free: raise ValueError('register pressure')
        d=free.pop()
        old=local[d]
        if cache.get(old)==d: del cache[old]
        out.append([op,i,j,d]); local[d]=v; cache[v]=d
        return d
    local=regs.copy()
    try:
        for tree in trees: emit(tree)
        # Restore baseline destinations using a cached multiplicative identity.
        for d,v in required.items():
            if local[d] == v: continue
            src=cache.get(v)
            one=cache.get(1)
            if src is None or one is None: return None
            out.append(['*',src,one,d]); local[d]=v
        if len(out)>=budget: return None
    except (ValueError,KeyError): return None
    return out


def improve(args):
    pid,p,old,seconds=args
    deadline=time.monotonic()+seconds
    states=replay(p,old,True)
    if states is None: raise ValueError('invalid baseline '+pid)
    live=[set() for _ in range(len(old)+1)]
    for n in range(len(old)-1,-1,-1):
        op,i,j,d=old[n]
        live[n]=(live[n+1]-{d})|{i,j}
    result=[]; pos=0; replacements=0
    actual=states[0].copy()
    targets=set(p['targets'])
    while pos<len(old):
        if time.monotonic()>deadline:
            result.extend(old[pos:]); break
        end=min(pos+32,len(old))
        changed={ins[3] for ins in old[pos:end]}
        required={r:states[end][r] for r in changed & live[end]}
        covered={states[n+1][old[n][3]] for n in range(pos,end)} & targets
        new=synthesize(actual,required,covered,live[end],p['limit'],deadline,end-pos)
        if new is not None:
            result.extend(new); replacements+=1
        else: result.extend(old[pos:end])
        for op,i,j,d in (new if new is not None else old[pos:end]):
            actual[d]=calc(op,actual[i],actual[j],p['limit'])
        pos=end
    # Blocks preserve live outputs; replay additionally checks all target history.
    if len(result)<len(old) and replay(p,result):
        return pid,result,replacements
    return pid,old,0


def main():
    o=kt.optimization(); data=Path(o['data_dir'])
    def locate(name):
        a=data/'data'/name
        return a if a.is_file() else data/name
    with locate('puzzles.csv').open(newline='',encoding='utf-8') as f:
        puzzles={r['id']:dict(start=json.loads(r['starting_values']),targets=json.loads(r['targets']),limit=int(r['value_limit'])) for r in csv.DictReader(f)}
    with Path(o['best'] or locate('sample_submission.csv')).open(newline='',encoding='utf-8') as f:
        current={r['id']:json.loads(r['program']) for r in csv.DictReader(f)}
    ids=sorted(puzzles)[:6] if o['smoke'] else sorted(puzzles)
    jobs=[(pid,puzzles[pid],current[pid],0.65 if o['smoke'] else 25) for pid in ids]
    if o['smoke']: results=list(map(improve,jobs))
    else:
        with Pool(max(1,int(o['threads']))) as pool: results=pool.map(improve,jobs,chunksize=1)
    for pid,new,n in results:
        print(pid,len(current[pid]),len(new),'blocks',n,flush=True)
        if len(new)<len(current[pid]) and replay(puzzles[pid],new): current[pid]=new
    out=Path(o['solution']); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['id','program'])
        for pid in sorted(puzzles):
            if replay(puzzles[pid],current[pid]) is None: raise ValueError('invalid output '+pid)
            w.writerow([pid,json.dumps(current[pid],separators=(',',':'))])


if __name__ == '__main__': main()
