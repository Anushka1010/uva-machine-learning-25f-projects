"""
verifier.py

Lightweight verifier/simulator for PIM microprogram format.

Accepted program formats (either one):

1) String-based (legacy):
   {"step": 1, "instr": "ReadRowToSa(dram_row=ROW10)"}

2) Structured (LLM-friendly):
   {"step": 1, "op": "ReadRowToSa", "args": {"dram_row": "ROW10"}}

Supported ops:
  - ReadRowToSa(dram_row=ROWx)   : DRAM[row] -> RR0
  - WriteSaToRow(dram_row=ROWx)  : RR0 -> DRAM[row]
  - Swap(rr_index=i)             : swap RR0 <-> RR[i]
  - NOR() / Nor()                : RR0 := ~(RR0 | RR1)  (32-bit masked)
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

MASK32 = 0xFFFFFFFF


# -------------------------------------------------
# Instruction parsing helpers
# -------------------------------------------------

def parse_instr(instr: str) -> Tuple[str, Dict[str, Any]]:
    """Parse 'OP(k=v,...)' strings into (op, args)."""
    m = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*(.*?)\s*\)\s*", instr)
    if not m:
        raise ValueError(f"Bad instr format: {instr!r}")

    op = m.group(1)
    arg_str = m.group(2)

    args: Dict[str, Any] = {}
    if arg_str:
        parts = [p.strip() for p in arg_str.split(",") if p.strip()]
        for p in parts:
            if "=" not in p:
                raise ValueError(f"Bad arg token (expected k=v): {p!r} in {instr!r}")
            k, v = [x.strip() for x in p.split("=", 1)]
            args[k] = int(v) if re.fullmatch(r"-?\d+", v) else v

    return op, args


def _args_to_str(args: Dict[str, Any]) -> str:
    """Convert args dict to canonical k=v string (stable order)."""
    parts = []
    for k in sorted(args.keys()):
        parts.append(f"{k}={args[k]}")
    return ",".join(parts)


def step_to_instr(ins: Dict[str, Any]) -> str:
    """
    Convert a program step into canonical instruction string.

    Accepts either:
      {"step": n, "instr": "..."}
    or
      {"step": n, "op": "...", "args": {...}}
    """
    if "instr" in ins:
        return ins["instr"]

    if "op" not in ins:
        raise KeyError(f"Instruction missing 'instr' or 'op': {ins}")

    op = ins["op"]
    args = ins.get("args", {}) or {}

    if not isinstance(args, dict):
        raise TypeError(f"'args' must be a dict, got {type(args)} in {ins}")

    arg_str = _args_to_str(args)
    return f"{op}({arg_str})" if arg_str else f"{op}()"


# -------------------------------------------------
# Simulator
# -------------------------------------------------

@dataclass
class PIMState:
    RR: List[int]
    DRAM: Dict[str, int]


class PIMSimulator:
    def __init__(self, row_reg_count: int, dram_init: Dict[str, int]):
        if row_reg_count < 2:
            raise ValueError("row_reg_count must be >= 2 (needs RR0 and RR1)")
        self.state = PIMState(
            RR=[0] * row_reg_count,
            DRAM={k: (v & MASK32) for k, v in dram_init.items()},
        )

    def read_row_to_sa(self, dram_row: str) -> None:
        if dram_row not in self.state.DRAM:
            raise KeyError(f"DRAM row not found: {dram_row}")
        self.state.RR[0] = self.state.DRAM[dram_row] & MASK32

    def write_sa_to_row(self, dram_row: str) -> None:
        self.state.DRAM[dram_row] = self.state.RR[0] & MASK32

    def swap(self, rr_index: int) -> None:
        if not (0 <= rr_index < len(self.state.RR)):
            raise IndexError(f"rr_index out of range: {rr_index}")
        self.state.RR[0], self.state.RR[rr_index] = (
            self.state.RR[rr_index],
            self.state.RR[0],
        )

    def nor(self) -> None:
        self.state.RR[0] = (~(self.state.RR[0] | self.state.RR[1])) & MASK32

    def step(self, instr: str) -> None:
        op, args = parse_instr(instr)

        if op == "ReadRowToSa":
            self.read_row_to_sa(args["dram_row"])
        elif op == "WriteSaToRow":
            self.write_sa_to_row(args["dram_row"])
        elif op == "Swap":
            self.swap(int(args["rr_index"]))
        elif op in ("NOR", "Nor"):
            self.nor()
        else:
            raise ValueError(f"Unknown op: {op}")


def run_program(
    program: List[Dict[str, Any]],
    row_reg_count: int,
    dram_init: Dict[str, int],
    verbose: bool = False,
) -> PIMState:
    """Execute a program and return final state."""
    sim = PIMSimulator(row_reg_count=row_reg_count, dram_init=dram_init)

    for ins in sorted(program, key=lambda x: x["step"]):
        instr = step_to_instr(ins)
        if verbose:
            rr = sim.state.RR
            rr_snap = " ".join(f"RR{i}={rr[i]:08X}" for i in range(min(len(rr), 4)))
            print(f"[{ins['step']:02d}] {instr:<35} | {rr_snap}")
        sim.step(instr)

    return sim.state


# -------------------------------------------------
# Verification logic
# -------------------------------------------------

def expected_nor32(a: int, b: int) -> int:
    return (~(a | b)) & MASK32


def verify_nor_rows(
    program: List[Dict[str, Any]],
    row_reg_count: int,
    rowA: str,
    rowB: str,
    rowOUT: str,
    A: int,
    B: int,
    verbose: bool = False,
) -> bool:
    st = run_program(program, row_reg_count, {rowA: A, rowB: B, rowOUT: 0}, verbose=verbose)
    return (st.DRAM[rowOUT] & MASK32) == expected_nor32(A, B)


def verify_from_verifier_input(
    verifier_input: Dict[str, Any],
    row_reg_count: int,
    task_name: str,
    num_tests: int = 20,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Generic entrypoint used by code_gen.py.

    Returns:
      {
        "pass": bool,
        "task": str,
        "num_tests": int,
        "first_failure": {...} | None
      }
    """
    program = verifier_input["program"]
    io = verifier_input.get("io", {})
    input_rows = io.get("input_rows") or []
    output_row = io.get("output_row")

    if len(input_rows) < 2 or not output_row:
        raise ValueError("verifier_input.io must include input_rows (len>=2) and output_row")

    task = (task_name or "").lower()
    if "nor" not in task:
        raise NotImplementedError(
            f"Only NOR verification is implemented right now (task={task_name!r})"
        )

    rng = random.Random(seed)
    for t in range(1, num_tests + 1):
        A = rng.getrandbits(32)
        B = rng.getrandbits(32)

        ok = verify_nor_rows(
            program=program,
            row_reg_count=row_reg_count,
            rowA=input_rows[0],
            rowB=input_rows[1],
            rowOUT=output_row,
            A=A,
            B=B,
            verbose=False,
        )

        if not ok:
            got_state = run_program(
                program,
                row_reg_count,
                {input_rows[0]: A, input_rows[1]: B, output_row: 0},
                verbose=False,
            )
            return {
                "pass": False,
                "task": task_name,
                "num_tests": num_tests,
                "first_failure": {
                    "test": t,
                    "A": f"0x{A:08X}",
                    "B": f"0x{B:08X}",
                    "expected": f"0x{expected_nor32(A, B):08X}",
                    "got": f"0x{got_state.DRAM[output_row] & MASK32:08X}",
                },
            }

    return {
        "pass": True,
        "task": task_name,
        "num_tests": num_tests,
        "first_failure": None,
    }
