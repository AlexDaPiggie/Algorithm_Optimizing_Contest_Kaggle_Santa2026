"""Scorer for Santa 2026 - The Elf Calculator (follows the Overview/Evaluation and Data pages).

score(solution_path, data_dir) -> total instructions over the 1,000 scored puzzles (lower is better).
Raises ValueError for any invalid submission, as Kaggle rejects it (no partial credit).

Rules implemented:
- CSV with exactly two columns, id and program, in that order; every id scored_0001..scored_1000 exactly once.
- Program cell: JSON array of [op, i, j, d], op in + - * /, i, j, d integers 0..47; at most 4,096
  instructions and 131,072 UTF-8 bytes per cell.
- starting_values fill registers 0..len-1; all others undefined. Sources must be defined.
- Both sources are read before the write. Results must be positive integers <= value_limit;
  subtraction only when first > second; division only when exact.
- A target is covered when its value appears in any register at any time (including at the start).
  Every target must be covered. Every instruction counts, including ones after the last target.
"""
import csv
import json
from pathlib import Path

REGISTERS = 48
MAX_INSTRUCTIONS = 4096
MAX_CELL_BYTES = 131072
OPS = {"+", "-", "*", "/"}


def load_puzzles(data_dir):
    path = Path(data_dir) / "data" / "puzzles.csv"
    if not path.is_file():
        path = Path(data_dir) / "puzzles.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return {row["id"]: dict(start=json.loads(row["starting_values"]), targets=json.loads(row["targets"]),
                                limit=int(row["value_limit"]), registers=int(row["registers"]))
                for row in csv.DictReader(f)}


def run_program(puzzle, program, pid="puzzle"):
    """Replay one program. Returns its instruction count or raises ValueError."""
    if not isinstance(program, list):
        raise ValueError(f"{pid}: program must be a JSON array")
    if len(program) > MAX_INSTRUCTIONS:
        raise ValueError(f"{pid}: {len(program)} instructions > {MAX_INSTRUCTIONS}")
    registers = [None] * REGISTERS
    for k, value in enumerate(puzzle["start"]):
        registers[k] = value
    limit = puzzle["limit"]
    remaining = set(puzzle["targets"]) - set(puzzle["start"])
    for step, ins in enumerate(program):
        if not (isinstance(ins, list) and len(ins) == 4):
            raise ValueError(f"{pid} step {step}: instruction must be [op, i, j, d]")
        op, i, j, d = ins
        if op not in OPS:
            raise ValueError(f"{pid} step {step}: bad operator {op!r}")
        for r in (i, j, d):
            if type(r) is not int or not 0 <= r < REGISTERS:
                raise ValueError(f"{pid} step {step}: register {r!r} outside 0..47")
        a, b = registers[i], registers[j]
        if a is None or b is None:
            raise ValueError(f"{pid} step {step}: undefined source register")
        if op == "+":
            value = a + b
        elif op == "*":
            value = a * b
        elif op == "-":
            if a <= b:
                raise ValueError(f"{pid} step {step}: subtraction {a} - {b} is not positive")
            value = a - b
        else:
            if a % b:
                raise ValueError(f"{pid} step {step}: division {a} / {b} is not exact")
            value = a // b
        if value > limit:
            raise ValueError(f"{pid} step {step}: value {value} exceeds value_limit {limit}")
        registers[d] = value
        remaining.discard(value)
    if remaining:
        raise ValueError(f"{pid}: {len(remaining)} target(s) never produced")
    return len(program)


def score(solution_path, data_dir):
    puzzles = load_puzzles(data_dir)
    with open(solution_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != ["id", "program"]:
            raise ValueError("Columns must be exactly: id,program")
        seen, total = set(), 0
        for row in reader:
            if len(row) != 2:
                raise ValueError("Each row needs exactly two columns")
            pid, cell = row
            if pid not in puzzles:
                raise ValueError(f"Unknown id {pid}")
            if pid in seen:
                raise ValueError(f"Duplicate id {pid}")
            seen.add(pid)
            if len(cell.encode("utf-8")) > MAX_CELL_BYTES:
                raise ValueError(f"{pid}: program cell larger than {MAX_CELL_BYTES} bytes")
            try:
                program = json.loads(cell)
            except ValueError:
                raise ValueError(f"{pid}: program is not valid JSON")
            total += run_program(puzzles[pid], program, pid)
    missing = set(puzzles) - seen
    if missing:
        raise ValueError(f"{len(missing)} puzzle id(s) missing, e.g. {sorted(missing)[0]}")
    return float(total)


if __name__ == "__main__":
    import sys
    print(score(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "."))
