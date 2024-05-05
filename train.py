from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from config import DATASET_HF, PRETRAINED_MODEL
import torch


# Load the dataset and set up the tokenizer
dataset = load_dataset(DATASET_HF)
tokenizer = AutoTokenizer.from_pretrained(
    PRETRAINED_MODEL, padding_side="left", use_fast=False
)  # there is a bug with a fast version
tokenizer.pad_token = tokenizer.eos_token
torch_ignore_index = -100

# Define quantization config and LoRA parameters
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["dense", "fc2", "q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLM.from_pretrained(
    PRETRAINED_MODEL, device_map={"": 0}, quantization_config=bnb_config
)

model = get_peft_model(model, lora_config)


# Helper function for attention_map
def calculate_length_in_tokens(text, tokenizer):
    return tokenizer(text, return_length=True)["length"][0]


def process_data(element):
    # Create text input for the model by concatenating prompt and completion
    text = element["prompt"] + element["completion"]

    enc = tokenizer(text, max_length=512, padding="max_length", return_tensors="pt")
    input_ids = enc["input_ids"].squeeze()
    attention_mask = enc["attention_mask"].squeeze()

    # Clone input_ids to use as labels
    labels = input_ids.clone()

    # Calculate prompt length in tokens
    prompt_len = calculate_length_in_tokens(element["prompt"], tokenizer)

    # Mask labels for the prompt part
    labels[:prompt_len] = torch_ignore_index

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# Apply tokenization
# Column names are the same
tokenized_train = dataset["train"].map(
    process_data, batched=True, remove_columns=dataset["train"].column_names
)
tokenized_val = dataset["validation"].map(
    process_data, batched=True, remove_columns=dataset["train"].column_names
)


# Define training arguments
training_args = TrainingArguments(
    output_dir="./phi1.5-kotlin",  # output directory (model will have this name in HF hub)
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    logging_steps=1000,
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    push_to_hub=True,
    hub_strategy="checkpoint",  # last_checkpint will be saved in the hub so we could continue training from it
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
