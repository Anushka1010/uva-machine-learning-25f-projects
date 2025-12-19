import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def make_architecture_block(
    architecture_explanation: str,
    logic_ops: List[str],
    row_reg_count: int,
    data_movement_rules: List[str],
    word_size_bits: int = 32,
    arch_type: str = "h_layout_subarray_level_pim",
) -> Dict[str, Any]:
    return {
        "type": arch_type,
        "architecture_explanation": architecture_explanation,
        "word_size_bits": word_size_bits,
        "shift_semantics": "logical_shifts_truncate_out_of_range_bits",
        "compute": {"logic_operations": logic_ops},
        "row_register_file": {
            "count": row_reg_count,
            "naming": [f"RR{i}" for i in range(row_reg_count)],
        },
        "data_movement": {
            "freedom": "restricted" if data_movement_rules else "free",
            "rules": data_movement_rules,
        },
    }


def make_isa_block(isa_operations: List[str]) -> Dict[str, Any]:
    return {"operations": isa_operations}


def build_pim_json(
    examples_db: List[Dict[str, Any]],
    query: Dict[str, Any],
) -> Dict[str, Any]:
    """
    JSON structure:
      - examples_db: arbitrary-length list of {db_id, architecture, isa, items:[...]}
      - query: exactly one {query_id, architecture, isa, item:{...}}
    """
    return {
        "schema_version": "1.1",
        "examples_db": examples_db,
        "query": query,
    }


def save_json(obj: Dict[str, Any], filepath: str) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"[OK] JSON saved to {filepath.resolve()}")


if __name__ == "__main__":
    # -----------------------------
    # EXAMPLES DB (can be 0..N entries, each with 0..M items)
    # -----------------------------
    archA_expl = (
        "H-layout subarray-level PIM. Operands are read from DRAM rows into row registers and "
        "results are written back to DRAM. Word size is 32-bit; shifts truncate out-of-range bits."
    )

    archA = make_architecture_block(
        architecture_explanation=archA_expl,
        logic_ops=["NOR"],
        row_reg_count=4,
        data_movement_rules=[
            "RR->RR moves allowed only within the same subarray",
            "No direct RR->DRAM row move",
            "DRAM rows can be read into and written from RR0 only",
            "RR0 may swap with any RRi",
            "Logic ops take RR0 and RR1 as operands; result is written to RR0",
        ],
    )

    isaA = make_isa_block([
        "ReadRowToSa(dram_row)     // DRAM row -> RR0",
        "WriteSaToRow(dram_row)    // RR0 -> DRAM row",
        "Swap(rr_index)            // swap RR0 <-> RR[rr_index]",
        "Nor()                     // RR0 := ~(RR0 | RR1)",
    ])

    examples_db = [
        {
            "db_id": "archA_examples",
            "architecture": archA,
            "isa": isaA,
            "items": [
                {
                    "id": "ex_nor_rows",
                    "task": {
                        "name": "nor_rows",
                        "bitwidth": 32,
                        "inputs": [
                            {"name": "A", "location": {"type": "dram_row", "id": "ROW1"}},
                            {"name": "B", "location": {"type": "dram_row", "id": "ROW2"}},
                        ],
                        "outputs": [
                            {"name": "OUT", "location": {"type": "dram_row", "id": "ROW3"}},
                        ],
                    },
                    "program": [
                        {"step": 1, "op": "ReadRowToSa",  "args": {"dram_row": "ROW1"}},
                        {"step": 2, "op": "Swap",         "args": {"rr_index": 1}},
                        {"step": 3, "op": "ReadRowToSa",  "args": {"dram_row": "ROW2"}},
                        {"step": 4, "op": "Nor",          "args": {}},
                        {"step": 5, "op": "WriteSaToRow", "args": {"dram_row": "ROW3"}},
                    ],
                    "correctness": {
                        "is_correct": True,
                        "evidence": {"method": "provided_by_user", "details": "Reference example."},
                    },
                }
            ],
        },

        # can append more example DB entries
    ]

    # -----------------------------
    # SINGLE QUERY (exactly one)
    # Can be a different architecture than the examples
    # -----------------------------
    archQ_expl = (
        "H-layout subarray-level PIM. Operands are read from DRAM rows into row registers and "
        "results are written back to DRAM. Word size is 32-bit; shifts truncate out-of-range bits."
    )

    archQ = make_architecture_block(
        architecture_explanation=archA_expl,
        logic_ops=["NOR"],
        row_reg_count=4,
        data_movement_rules=[
            "RR->RR moves allowed only within the same subarray",
            "No direct RR->DRAM row move",
            "DRAM rows can be read into and written from RR0 only",
            "RR0 may swap with any RRi",
            "Logic ops take RR0 and RR1 as operands; result is written to RR0",
        ],
    )

    isaQ = make_isa_block([
        "ReadRowToSa(dram_row)     // DRAM row -> RR0",
        "WriteSaToRow(dram_row)    // RR0 -> DRAM row",
        "Swap(rr_index)            // only rr_index in {1,2} allowed",
        "NOR()                     // RR0 := ~(RR0 | RR1)",
    ])

    query = {
        "query_id": "q_nor_rows_false_missing_preserve_A",
        "architecture": archQ,
        "isa": isaQ,
        "item": {
            "id": "q_item_001",
            "task": {
                "name": "nor_rows",
                "bitwidth": 32,
                "inputs": [
                    {"name": "A", "location": {"type": "dram_row", "id": "ROW10"}},
                    {"name": "B", "location": {"type": "dram_row", "id": "ROW11"}},
                ],
                "outputs": [
                    {"name": "OUT", "location": {"type": "dram_row", "id": "ROW12"}},
                ],
            },
            "program": [
                {"step": 1, "op": "ReadRowToSa",  "args": {"dram_row": "ROW10"}, "comment": "RR0 <- A"},
                {"step": 2, "op": "ReadRowToSa",  "args": {"dram_row": "ROW11"}, "comment": "RR0 <- B (overwrites A)"},
                {"step": 3, "op": "Nor",          "args": {},                   "comment": "RR1 never loaded with A"},
                {"step": 4, "op": "WriteSaToRow", "args": {"dram_row": "ROW12"}, "comment": "OUT <- RR0"},
            ],
            "correctness": {
                "is_correct": False,
                "evidence": {
                    "method": "static_check",
                    "details": "A is overwritten before being saved to RR1; RR1 is uninitialized.",
                },
            },
        },
    }

    pim_json = build_pim_json(examples_db=examples_db, query=query)
    save_json(pim_json, "pim_arch_examples.json")
