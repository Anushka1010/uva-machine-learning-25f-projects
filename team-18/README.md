# Microprogram Code_Generation、
# Team 18: Yingming Ma

Writing microprograms is a challenging yet essential task in hardware research areas such as processing-in-memory (PIM). Fully exploring a design space often requires developing a large number of microprograms to cover different hardware configurations and architectural variations. This project serves as a preliminary exploration of the potential for leveraging modern large language models (LLMs) to offload part of this microprogram generation process.

## Usage

1. Set the API key directly in the code by finding the following line in code_gen.py
```bash
client = OpenAI(api_key="your_api_key_here")
```

2. Generate the example JSON:
```bash
python create_json.py
```

3. Run code generation:
```bash
python code_gen.py
```

## Output

- `pim_arch_examples.json` — example input  
- `api_output.json` — generated output  
- `api_output_with_verification.json` — generated output with verification results