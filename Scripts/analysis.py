import os
import re
import csv
import time
import pandas as pd
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ------------------------------------------------------------
# Settings CHANGE ONLY HERE
# ------------------------------------------------------------
INPUT_FILE = Path("mrben_annotations.csv")
OUTPUT_FILE = Path("mrben_judgments.csv")

number_of_samples = 20 # the amount of questions to analyse the statements of

ANNOTATION_COLUMNS = {
    "llama": "annotation_Llama",
    "qwen": "annotation_Qwen",
}
HF_TOKEN = '' # put access token to llama here

# ------------------------------------------------------------
# Settings DON'T CHANGE
# ------------------------------------------------------------

INPUT_FOLDER = Path("Annotations")
OUTPUT_FOLDER = Path("Outputs/judgments")

# Properly join the folder and the file name
csv_path = INPUT_FOLDER / INPUT_FILE
output_path = OUTPUT_FOLDER / OUTPUT_FILE

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

SLEEP_SECONDS = 0.1

CONSISTENCY_MODELS = {
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Local Model Setup and Caching
# ------------------------------------------------------------

loaded_models = {}
loaded_tokenizers = {}


def get_model(model_name):
    """Loads the model and tokenizer only once to prevent VRAM overflow."""
    if model_name not in loaded_models:
        print(f"Loading model {model_name} into memory...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create the proper 4-bit configuration object
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        # Pass the config to the model loader instead of load_in_4bit=True
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=bnb_config,
            token=HF_TOKEN
        )

        if not torch.cuda.is_available():
            model.to(DEVICE)

        loaded_tokenizers[model_name] = tokenizer
        loaded_models[model_name] = model

    return loaded_models[model_name], loaded_tokenizers[model_name]


# ------------------------------------------------------------
# Local Inference Generation
# ------------------------------------------------------------

def query_local_model(model_id, prompt, max_new_tokens=10):
    """Generates the response using local transformers logic."""
    model, tokenizer = get_model(model_id)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    # do_sample=False is used for greedy decoding to ensure deterministic TRUE/FALSE outputs.
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return response


# ------------------------------------------------------------
# Split statements by semicolon
# ------------------------------------------------------------

def split_into_statements(text):
    if pd.isna(text):
        return []

    text = str(text).strip()

    if not text:
        return []

    # Remove the wrapping square brackets [ ] found in the CSV data
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()

    # Split by semicolon (with or without spaces)
    parts = re.split(r"\;\s*", text)

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
    # Prompt is as simple as possible with no distracting instructions
    prompt = f"""
Decide whether the following statement is factually true or false.

Statement:
{statement}

Reply with true or false only.
"""

    raw = query_local_model(
        model_id=model_id,
        prompt=prompt,
        max_new_tokens=5,
    )

    raw_upper = raw.strip().upper()

    if "TRUE" in raw_upper:
        return "TRUE", raw
    elif "FALSE" in raw_upper:
        return "FALSE", raw
    else:
        return "?", raw


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    df = pd.read_csv(csv_path)

    # --- NEW LOGIC: Filter for annotated Qwen rows and sample 20 ---
    # Dynamically drops rows based on the remaining keys in ANNOTATION_COLUMNS
    annotation_cols_to_check = list(ANNOTATION_COLUMNS.values())
    df_annotated = df.dropna(subset=annotation_cols_to_check, how="all")

    # Sample n random rows (or all if there are fewer than n)
    num_to_sample = min(number_of_samples, len(df_annotated))
    df = df_annotated.sample(n=num_to_sample, random_state=42).reset_index(drop=True)
    print(f"Sampled {num_to_sample} random annotated rows for processing.")
    # ----------------------------------------------------------

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

    for llm_name, annotation_col in ANNOTATION_COLUMNS.items():
        if annotation_col not in df.columns:
            continue

        judge_model = CONSISTENCY_MODELS[llm_name]
        print(f"\n--- Starting processing for model: {llm_name} ---")

        for row_index, row in df.iterrows():
            question_id = row.get("question_id", "")
            question = row.get("question", "")
            options = row.get("options", "")
            correct_answer = row.get("correct_answer", "")

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
                    f"Question {row_index + 1}/{len(df)} (ID: {question_id}), "
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

        # Free up memory when the model finishes
        if judge_model in loaded_models:
            print(f"Unloading {judge_model} to free VRAM...")
            del loaded_models[judge_model]
            del loaded_tokenizers[judge_model]
            torch.cuda.empty_cache()

    print(f"Done. Saved {len(output_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
