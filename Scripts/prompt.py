
from datasets import load_dataset
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_questions(num_questions=2):

    train_dataset = load_dataset("Randolphzeng/Mr-Ben", split="train")
    #but inly of Subject = college_biology or college_medicine
    train_dataset = train_dataset.filter(lambda x: x["Subject"] in ["college_biology", "college_medicine"])
    random_indices = np.random.choice(len(train_dataset), size=num_questions, replace=False)
    questions = [train_dataset[i]["Question"] for i in random_indices]
    options = [train_dataset[i]["Options"] for i in random_indices]
    return questions, options

def build_explanation_prompt(question, options, answer):
    '''
    Question: {question text}
    Choose the most plausible answer,
    respond only with the answer and
    the description:
    A. {option A}
    B. {option B}
    C. {option C}
    D. {option D}
    Selected Answer: {model’s
    answer}
    Why did you make that choice?
    Explain briefly.
    Explanation: {model’s
    explanation}
    '''
    prompt = f"Question: {question}\nChoose the most plausible answer, respond only with the answer and the description:\n"
    for option in options.split("\n"):
        prompt += option + "\n"
    prompt += f"Selected Answer: {answer}\nWhy did you make that choice? Explain briefly.\nExplanation:"
    return prompt
def build_question_prompt(question, options):
    '''
    Question: {question text}
    Choose the most plausible answer,
    respond only with the answer and
    the description:
    A. {option A}
    B. {option B}
    C. {option C}
    D. {option D}
    Answer: {model's answer}
    '''

    #options look like this:
    '''
    A: Enhancer 
    B: Repressor 
    C: Operator 
    D: Promoter 

    '''
    prompt = f"Question: {question}\nChoose the most plausible answer, respond only with the answer and the description:\n"
    for option in options.split("\n"):
        prompt += option + "\n"
    prompt += "Answer:"
    return prompt

def prompt(prompt_text, model_name, max_tokens=400):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(DEVICE)

    if isinstance(prompt_text, str):
        prompt_text = [prompt_text]

    response = None
    for text in prompt_text:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(DEVICE)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return response


def main():
    models = ["Qwen/Qwen2.5-7B-Instruct"] #meta-llama/Llama-3.1-8B-Instruct
    for model in models:
        qs, options = get_questions(num_questions=20)
        prompts =[]
        answers =[]
        explanations = []
        for q, opt in zip(qs, options):
            prompt_text = build_question_prompt(q, opt)
            answer = prompt(prompt_text, model)
            prompts.append(prompt_text)
            answers.append(answer)

            explanation_prompt = build_explanation_prompt(q, opt, answer)
            explanation = prompt(explanation_prompt, model)
            explanations.append(explanation)
        
        for p, a, e in zip(prompts, answers, explanations):
            print(f"Model: {model}\n")
            print(f"----------------Prompt----------------\n\n{p}\n\n")
            print(f"----------------Answer----------------\n\n{a}\n\n")
            print(f"----------------Explanation----------------\n\n{e}\n\n")
            #save in a text file
            model_name = model.split("/")[-1]
            with open(f"{model_name}_results.txt", "a") as f:
                    f.write(f"Model: {model}\n")
                    f.write(f"----------------Prompt----------------\n\n{p}\n\n")
                    f.write(f"----------------Answer----------------\n\n{a}\n\n")
                    f.write(f"----------------Explanation----------------\n\n{e}\n\n")
            

if __name__ == "__main__":
    main()