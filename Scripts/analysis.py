import os
import re
import csv
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# ------------------------------------------------------------
# Settings CHANGE ONLY HERE
# ------------------------------------------------------------
INPUT_FILE = "mrben_annotations.csv"
# input path → annotations/judgments_annotations_kasia.csv
# output path → outputs/judgments/judgments_annotations_kasia.csv


ANNOTATION_COLUMNS = {
    "llama": "annotation_Llama",
    "qwen": "annotation_Qwen",
}

# ------------------------------------------------------------
# Settings DON'T CHANGE
# ------------------------------------------------------------


INPUT_FOLDER = Path("Annotations")
OUTPUT_FOLDER = Path("Outputs/judgments")

csv_path = INPUT_FOLDER / INPUT_FILE
output_path = OUTPUT_FOLDER / f"judgments_{INPUT_FILE}"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

SLEEP_SECONDS = 0.1
TIMEOUT_SECONDS = 30
RETRIES = 3

CONSISTENCY_MODELS = {
    "llama": "meta-llama/llama-3.1-8b-instruct",
    "qwen": "qwen/qwen-2.5-7b-instruct",
}

# ------------------------------------------------------------
# API setup
# ------------------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ------------------------------------------------------------
# API call with timeout and retries
# ------------------------------------------------------------

def query_openrouter(model_id, prompt, max_new_tokens=10):
    for attempt in range(RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=0,
                timeout=TIMEOUT_SECONDS,
                extra_headers={
                    "HTTP-Referer": "https://github.com/annotation-project",
                    "X-Title": "ARC Statement Consistency Judgment",
                },
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"API error attempt {attempt + 1}/{RETRIES}: {e}", flush=True)

            if attempt < RETRIES - 1:
                time.sleep(10)
            else:
                return "ERROR"

# ------------------------------------------------------------
# Split statements by periods
# ------------------------------------------------------------

def split_into_statements(text):
    if pd.isna(text):
        return []

    text = str(text).strip()

    if not text:
        return []

    parts = re.split(r"\;\s+", text)

    statements = []

    for part in parts:
        part = part.strip()

        if not part:
            continue

        if not part.endswith("."):
            part += "."

        statements.append(part)

    return statements

# ------------------------------------------------------------
# Judge one statement in isolation
# ------------------------------------------------------------

def judge_statement_in_isolation(model_id, statement):
    prompt = f"""
Decide whether the following statement is factually TRUE or FALSE.

Judge the statement in isolation.
Do not use the original question, answer options, correct answer, or explanation.
Do not judge spelling or grammar.

Statement:
{statement}

Reply with ONLY one word:
TRUE or FALSE
"""

    raw = query_openrouter(
        model_id=model_id,
        prompt=prompt,
        max_new_tokens=5,
    )

    raw_upper = raw.strip().upper()

    if raw_upper.startswith("TRUE"):
        return "TRUE", raw
    elif raw_upper.startswith("FALSE"):
        return "FALSE", raw
    elif raw_upper == "ERROR":
        return "ERROR", raw
    else:
        return "?", raw

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    df = pd.read_csv(csv_path)

    output_rows = []

    # Continue from existing file if it exists
    if output_path.exists():
        old_df = pd.read_csv(output_path)
        output_rows = old_df.to_dict("records")
        done_keys = set(
            zip(
                old_df["question_id"],
                old_df["llm"],
                old_df["statement_index"],
            )
        )
        print(f"Resuming from existing file with {len(done_keys)} judgments.")
    else:
        done_keys = set()

    for row_index, row in df.iterrows():
        question_id = row.get("question_id", "")
        question = row.get("question", "")
        options = row.get("options", "")
        correct_answer = row.get("correct_answer", "")

        for llm_name, annotation_col in ANNOTATION_COLUMNS.items():
            if annotation_col not in df.columns:
                continue

            judge_model = CONSISTENCY_MODELS[llm_name]

            llm_answer = row.get(f"{llm_name}_answer", "")
            llm_explanation = row.get(f"{llm_name}_explanation", "")
            llm_correct = row.get(f"{llm_name}_correct", "")
            annotation_text = row.get(annotation_col, "")

            statements = split_into_statements(annotation_text)

            for statement_index, statement in enumerate(statements, start=1):
                key = (question_id, llm_name, statement_index)

                if key in done_keys:
                    continue

                print(
                    f"Question {row_index + 1}/{len(df)}, "
                    f"{llm_name}, statement {statement_index}/{len(statements)}",
                    flush=True,
                )

                judgment, raw_judge_response = judge_statement_in_isolation(
                    model_id=judge_model,
                    statement=statement,
                )

                output_rows.append(
                    {
                        "question_id": question_id,
                        "question": question,
                        "options": options,
                        "correct_answer": correct_answer,
                        "llm": llm_name,
                        "judge_model": judge_model,
                        "llm_answer": llm_answer,
                        "llm_correct": llm_correct,
                        "llm_explanation": llm_explanation,
                        "statement_index": statement_index,
                        "statement": statement,
                        "statement_true_in_isolation": judgment,
                        "raw_judge_response": raw_judge_response,
                    }
                )

                pd.DataFrame(output_rows).to_csv(
                    output_path,
                    index=False,
                    quoting=csv.QUOTE_MINIMAL,
                )

                time.sleep(SLEEP_SECONDS)

    print(f"Done. Saved {len(output_rows)} rows to {output_path}")

if __name__ == "__main__":
    main()