import json
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI

from verifier import verify_from_verifier_input


def load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(obj: Dict[str, Any], path: str) -> None:
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(f"[OK] Saved: {Path(path).resolve()}")


def build_prompt(db: Dict[str, Any]) -> str:
    # Keep examples small to avoid huge prompts; adjust as needed.
    examples_db = db.get("examples_db", [])

    # Option A: include the full examples_db (simple, but can get large)
    examples_block = json.dumps(examples_db, indent=2)

    # Option B (recommended later): include only the relevant bits / a few examples
    # For now, assume your examples_db is small enough.

    return f"""
Output ONLY valid JSON.

JSON FORMAT:
{{
  "verifier_input": {{
    "program": [
      {{ "step": 1, "op": "ReadRowToSa", "args": {{ "dram_row": "ROW10" }} }}
    ],
    "io": {{
      "input_rows": ["ROW10", "ROW11"],
      "output_row": "ROW12",
      "bitwidth": 32
    }}
  }},
  "reasoning_summary": [
    "Short bullet-point explanation of the correction."
  ]
}}

RULES:
- Output must be strictly valid JSON (no markdown, no extra text).
- verifier_input.program must contain ONLY fields needed for verification (step/op/args). Do NOT include comments.
- reasoning_summary must be concise, high-level, and must NOT include internal chain-of-thought.
- Use only operations in query.isa.operations and respect query.architecture rules.
- Follow the patterns shown in EXAMPLES (register usage, swapping, data movement rules).

EXAMPLES (reference microprograms; do not copy blindly—adapt to the query):
{examples_block}

QUERY (produce a corrected program for this item):
{json.dumps(db["query"], indent=2)}
""".strip()


def api_call(
    input_json_path: str,
    output_json_path: str = "api_output.json",
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """Call the model, parse JSON output, save to file, return parsed object."""
    client = OpenAI(api_key = "your_api_key_here")

    db = load_json(input_json_path)
    prompt = build_prompt(db)

    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You repair PIM microprograms under strict ISA/architecture constraints.",
            },
            {"role": "user", "content": prompt},
        ],
        store=False,
    )

    raw = resp.output_text
    print("=== RAW MODEL OUTPUT ===")
    print(raw)

    try:
        out_obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            "Model output was not valid JSON. "
            "Try reducing prompt length or add a retry loop that asks for JSON only."
        ) from e

    save_json(out_obj, output_json_path)
    return out_obj


def run_verification(input_json_path: str, api_out: Dict[str, Any]) -> Dict[str, Any]:
    """Run verifier on api_out and return verification report."""
    db = load_json(input_json_path)

    verifier_input = api_out.get("verifier_input")
    if not verifier_input:
        raise KeyError("API output missing 'verifier_input'")

    row_reg_count = db["query"]["architecture"]["row_register_file"]["count"]
    task_name = db["query"]["item"]["task"]["name"]

    report = verify_from_verifier_input(
        verifier_input=verifier_input,
        row_reg_count=row_reg_count,
        task_name=task_name,
        num_tests=50,
        seed=0,
    )
    return report


if __name__ == "__main__":
    INPUT_JSON = "pim_arch_examples.json"

    api_out = api_call(
        input_json_path=INPUT_JSON,
        output_json_path="api_output.json",
        model="gpt-5",
    )

    report = run_verification(INPUT_JSON, api_out)
    print("=== VERIFICATION REPORT ===")
    print(json.dumps(report, indent=2))

    # Save a combined artifact for debugging/logging
    combined = {
        "api_output": api_out,
        "verification_report": report,
    }
    save_json(combined, "api_output_with_verification.json")