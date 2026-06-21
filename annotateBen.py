import os
import csv
import time
from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset

# ─────────────────────────────────────────────────────────────
# Load environment variables
# ─────────────────────────────────────────────────────────────

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY not found. Put your OpenRouter token in a .env file."
    )

# Initialize the OpenRouter Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ─────────────────────────────────────────────────────────────
# Models required by the assignment
# (Using OpenRouter's free tier endpoints)
# ─────────────────────────────────────────────────────────────

MODELS = {
    "llama": "meta-llama/llama-3.1-8b-instruct",
    "qwen": "qwen/qwen-2.5-7b-instruct",
}

# ─────────────────────────────────────────────────────────────
# Load Mr-Ben dataset
# ─────────────────────────────────────────────────────────────

print("Loading Mr-Ben dataset...")

mrben = load_dataset(
    "Randolphzeng/Mr-Ben",
    split="train"
)

# ─────────────────────────────────────────────────────────────
# Select 300 unique questions (by Question_UUID to avoid
# duplicate questions sampled from different models)
# ─────────────────────────────────────────────────────────────

seen_uuids = set()
QUESTIONS = []

for item in mrben:
    uuid = item["Question_UUID"]
    if uuid in seen_uuids:
        continue
    seen_uuids.add(uuid)

    # Parse options string into a dict  e.g. "A: foo  B: bar"
    options_raw = item["Options"]
    choices = {}
    if options_raw:
        import re
        for match in re.finditer(r'([A-E]):\s*(.*?)(?=\s+[A-E]:|$)', options_raw):
            choices[match.group(1)] = match.group(2).strip()

    QUESTIONS.append({
        "uuid": uuid,
        "question": item["Question"],
        "options": options_raw,
        "choices": choices,
        "correct_answer": item["Ground_Truth_Answer"],
        "subject": item["Subject"],
    })

    if len(QUESTIONS) == 300:
        break

# ─────────────────────────────────────────────────────────────
# Preview questions
# ─────────────────────────────────────────────────────────────

print(f"\nLoaded {len(QUESTIONS)} unique questions:\n")

for i, q in enumerate(QUESTIONS):
    print(f"\n── Q{i+1} [{q['subject']}] (correct: {q['correct_answer']}) ──")
    print(q["question"][:120])
    print(q["options"])

# ─────────────────────────────────────────────────────────────
# Prompt formatting
# ─────────────────────────────────────────────────────────────


def format_question(question, options_raw):
    return f"{question}\n\n{options_raw}"


# ─────────────────────────────────────────────────────────────
# Query OpenRouter
# ─────────────────────────────────────────────────────────────


def query_openrouter(model_id, prompt, max_new_tokens=100):
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0,
            extra_headers={
                "HTTP-Referer": "https://github.com/annotation-project",
                "X-Title": "Mr-Ben Annotation Script",
            }
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        raise Exception(f"OpenRouter API Error: {str(e)}")


# ─────────────────────────────────────────────────────────────
# Ask model: Answer only, then explain briefly
# ─────────────────────────────────────────────────────────────


def ask_model(model_id, question, options_raw, choices):

    formatted = format_question(question, options_raw)

    valid_labels = set(choices.keys()) if choices else {
        "A", "B", "C", "D", "E"}

    # Step 1: answer only
    step1_prompt = (
        "Answer the following multiple choice question.\n"
        "Reply with ONLY the letter of the correct answer "
        f"({', '.join(sorted(valid_labels))}).\n\n"
        f"{formatted}"
    )

    raw_answer = query_openrouter(
        model_id,
        step1_prompt,
        max_new_tokens=5
    )

    raw_answer = raw_answer.strip().upper()

    answer = next(
        (c for c in raw_answer if c in valid_labels),
        "?"
    )

    # Step 2: explanation
    step2_prompt = (
        step1_prompt
        + f"\n\nAnswer: {answer}\n\n"
        + "Explain your reasoning in clear sentences. "
        + "Do not repeat the answer letter, but justify your choice."
    )

    explanation = query_openrouter(
        model_id,
        step2_prompt,
        max_new_tokens=150
    )

    return answer, explanation


# ─────────────────────────────────────────────────────────────
# Main annotation loop
# ─────────────────────────────────────────────────────────────


def run_annotation():

    results = []

    for i, q in enumerate(QUESTIONS):

        print(
            f"\n── Question {i+1}/{len(QUESTIONS)} "
            "──────────────────────"
        )
        print(q["question"][:100] + "...")

        row = {
            "question_id": i + 1,
            "question_uuid": q["uuid"],
            "subject": q["subject"],
            "question": q["question"],
            "options": q["options"],
            "correct_answer": q["correct_answer"],
        }

        for model_name, model_id in MODELS.items():

            print(f"  Querying {model_name}...")

            try:
                answer, explanation = ask_model(
                    model_id,
                    q["question"],
                    q["options"],
                    q["choices"]
                )

                row[f"{model_name}_answer"] = answer
                row[f"{model_name}_explanation"] = explanation
                row[f"{model_name}_correct"] = (
                    "YES" if answer == q["correct_answer"] else "NO"
                )

                print(
                    f"  {model_name}: {answer} "
                    f"({'✓' if answer == q['correct_answer'] else '✗'})"
                )

            except Exception as e:

                print(f"  ERROR with {model_name}: {e}")

                row[f"{model_name}_answer"] = "ERROR"
                row[f"{model_name}_explanation"] = str(e)
                row[f"{model_name}_correct"] = "ERROR"

            # OpenRouter free tier has rate limits
            time.sleep(3)

        # Manual annotation notes
        row["notes"] = ""

        results.append(row)

    # ─────────────────────────────────────────────────────────
    # Save results
    # ─────────────────────────────────────────────────────────

    os.makedirs("outputs", exist_ok=True)
    output_file = "outputs/mrben_answers.csv"

    fieldnames = [
        "question_id",
        "question_uuid",
        "subject",
        "question",
        "options",
        "correct_answer",

        "llama_answer",
        "llama_explanation",
        "llama_correct",

        "qwen_answer",
        "qwen_explanation",
        "qwen_correct",

        "notes",
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ─────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────

    llama_correct = sum(
        1 for r in results if r["llama_correct"] == "YES"
    )
    qwen_correct = sum(
        1 for r in results if r["qwen_correct"] == "YES"
    )

    print("\n✓ Done!")
    print(f"Results saved to: {output_file}")
    print(f"\nLlama accuracy: {llama_correct}/{len(results)}")
    print(f"Qwen accuracy:  {qwen_correct}/{len(results)}")


# ─────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    run_annotation()
