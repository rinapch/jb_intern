from loguru import logger
import argparse
from fuzzywuzzy import fuzz
import re
from bleu import _bleu
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import PRETRAINED_MODEL, DATASET_HF
from datasets import load_dataset
import jsonlines
from tqdm import tqdm

torch.set_default_device("cuda")

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)



def get_predictions(model, prompts):
    predictions = []
    for i in tqdm(range(0, len(prompts), 4), desc="Generating predictions", total=len(prompts)//4):
        batch_prompts = prompts[i:i+4]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_new_tokens=512)
        inputs = {key: val.to(torch.device('cuda')) for key, val in inputs.items()}
        
        with torch.no_grad():  # Disable gradients for prediction
            outputs = model.generate(**inputs, max_new_tokens=512)
        
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        postprocessed_outputs = [post_process(output) for output in decoded_outputs]
        predictions.extend(postprocessed_outputs)
        
    return predictions
        

# prompt template taken from here: 
# https://huggingface.co/microsoft/phi-1_5#sample-code
def format_codexglue_prompt(signature, docstring):
    return f'''{signature}:
    """
    {docstring}
    """'''

# post-processing taken directly from CodeXGLUE
# https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/Method-Generation/evaluator/evaluator.py
def post_process(code):
    code = code.replace("<EOL>", "\n").replace("<INDENT>", " ").replace("<DEDENT>", " ")
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    lits = re.findall(pattern, code)
    for lit in lits:
        code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
    return " ".join(code.split())


def compute_scores(preds, answers):
    total = len(answers)
    edit_sim = 0.0
    for pred, answ in tqdm(zip(preds, answers), total=total):
        pred = post_process(pred.strip())
        answ = post_process(answ.strip())
        edit_sim += fuzz.ratio(pred, answ)

    bleu_score = round(_bleu(answers, preds), 2)
    logger.info(f"Edit sim: {round(edit_sim/total, 2)}, BLEU: {bleu_score}")


def prepare_codexglue_data(codexglue):
    prompts = []
    answers = []
    for sample in codexglue:
        prompt = format_codexglue_prompt(sample["signature"], sample["docstring"])
        prompts.append(prompt)
        answers.append(sample["docstring"])

    return prompts, answers

def prepare_kotlin_data(codexglue):
    prompts = []
    answers = []
    for sample in codexglue:
        prompts.append(sample["prompt"])
        answers.append(sample["completion"])

    return prompts, answers


def main():
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for code completion (line level).')
    parser.add_argument('--hf_repository', type=str, required=True, help='Hugging Face repository of the finetuned Phi-1.5 model.')
    args = parser.parse_args()

    logger.info("Loading CodeXGLUE method generation dataset")

    with jsonlines.open("data/test_codexglue.jsonl", mode="r") as reader:
        codexglue = [line for line in reader]
    codexglue_prompts, codexglue_answers = prepare_codexglue_data(codexglue[:100])

    logger.info("Loading Kotlin test dataset")

    kotlin = load_dataset(DATASET_HF, split="test")
    kotlin_prompts, kotlin_answers = prepare_kotlin_data(kotlin)

    logger.info(f"Running predictions for the {PRETRAINED_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(PRETRAINED_MODEL, torch_dtype="auto")

    predictions_pretrained_codexglue = get_predictions(model, codexglue_prompts)
    predictions_pretrained_kotlin = get_predictions(model, kotlin_prompts)

    compute_scores(predictions_pretrained_codexglue, codexglue_answers)
    compute_scores(predictions_pretrained_kotlin, kotlin_answers)

    logger.info(f"Running predictions for the {args.hf_repository}")
    # redefining model variable to save GPU space 
    model = AutoModelForCausalLM.from_pretrained(args.hf_repository, torch_dtype="auto")

    predictions_finetuned_codexglue = get_predictions(model, codexglue_prompts)
    predictions_finetuned_kotlin = get_predictions(model, kotlin_prompts)

    compute_scores(predictions_finetuned_codexglue, codexglue_answers)
    compute_scores(predictions_finetuned_kotlin, kotlin_answers)


if __name__ == "__main__":
    main()