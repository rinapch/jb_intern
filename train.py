from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from config import DATASET_HF, PRETRAINED_MODEL


dataset = load_dataset(DATASET_HF)
tokenizer = AutoTokenizer.from_pretrained(
    PRETRAINED_MODEL, padding_side="left", use_fast=False
)  # there is a bug with a fast version
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(PRETRAINED_MODEL)


torch_ignore_index = -100


def calculate_length_in_tokens(text, tokenizer):
    return tokenizer(text, return_length=True)["length"][0]


def process_data(element):
    # Create text input for the model by concatenating prompt and completion
    text = element["prompt"] + element["completion"]

    # Use encode_plus to get tokenized output
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
tokenized_datasets = dataset.map(
    process_data, batched=True, remove_columns=dataset["train"].column_names
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./model",  # output directory
    num_train_epochs=3,  # number of training epochs
    per_device_train_batch_size=1,  # batch size for training
    per_device_eval_batch_size=1,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir="./logs",  # directory for storing logs
    logging_steps=10,
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Train the model
trainer.train()
