from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch
from config import PRETRAINED_MODEL
import argparse

def merge_lora(hf_repository):
    model = AutoModelForCausalLM.from_pretrained(PRETRAINED_MODEL, trust_remote_code=True, torch_dtype=torch.float32)
    peft_model = PeftModel.from_pretrained(model, hf_repository, from_transformers=True)
    model = peft_model.merge_and_unload()
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge a PEFT model with a pre-trained model from Hugging Face.')
    parser.add_argument('--hf_repository', type=str, required=True, help='Hugging Face repository of the PEFT model to merge.')
    args = parser.parse_args()

    merged_model = merge_lora(args.hf_repository)
    merged_model.push_to_hub(args.hf_repository) # model is going to be uploaded to the same repository