#!/bin/bash
# Run after all domain JSONs are filled

python scripts/import_questions.py prompts/computer_vision.json
python scripts/import_questions.py prompts/machine_learning.json
python scripts/import_questions.py prompts/deep_learning.json
python scripts/import_questions.py prompts/genai.json