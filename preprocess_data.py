import re
import jsonlines
from config import (
    MIN_CONTEXT_LENGTH,
    OUTPUT_RAW_FILE,
    OUTPUT_READY_DATASET_PATH,
    EOS_TOKEN,
)
from loguru import logger
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from typing import List, Dict
from transformers import AutoTokenizer
from config import PRETRAINED_MODEL
from langchain_text_splitters import RecursiveCharacterTextSplitter

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

# Splitting based on token count in each chunk
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=MIN_CONTEXT_LENGTH,
    chunk_overlap=0,
    separators=["\n", "\t", "}", " "],
)


def get_clean_data(data: List[Dict[str, str]]) -> List[str]:
    """Parse JSON and clean the code text by removing specific copyright blocks."""
    cleaned_data = []
    for sample in data:
        code = sample["file_text"]

        # Regex to remove the copyright block as specified
        copyright_pattern = r"/\*\n \* Copyright.*?\*/"
        code = re.sub(copyright_pattern, "", code, flags=re.DOTALL)
        # I decided not to remove comments because they can be useful for context & desirable for code completion

        if _is_long_sample(code) and not _contains_long_values(code):
            cleaned_data.append(code)

        else:  # if sample is short just don't save it
            continue

    return cleaned_data


def _is_long_sample(text: str) -> bool:
    """Filter out entries that are too short based on line breaks."""
    if text.count("\n") <= 1:
        return False
    return True


def _contains_long_values(text: str) -> bool:
    # Pattern matches any sequence of digits longer than 40 or non-space characters longer than 200
    pattern = re.compile(r"(\d{40,})|([^\s]{200,})")

    if pattern.search(text):
        return True  # If there's a match, we might want to skip this file
    else:
        return False  # No concerning long sequences found


def chunk_text(data: List[str]) -> List[Dict[str, str]]:
    """Generate (context, target) pairs from the given dataset."""
    chunked_pairs = []
    for text in tqdm(data, total=len(data)):
        chunks = text_splitter.split_text(text)
        for i in range(
            len(chunks) - 1
        ):  # Stop at the second to last chunk to ensure there is a completion for each prompt
            prompt = chunks[i]
            completion = chunks[i + 1]
            chunked_pairs.append(
                {"prompt": prompt, "completion": completion + EOS_TOKEN}
            )

    return chunked_pairs


if __name__ == "__main__":
    with jsonlines.open(f"data/{OUTPUT_RAW_FILE}", mode="r") as reader:
        raw_data = [line for line in reader]

    logger.info("Data loaded. Starting preprocessing...")

    preprocessed_data = get_clean_data(raw_data)

    logger.info(f"Length of preprocessed data: {len(preprocessed_data)} files")

    logger.info("Data preprocessed. Starting chunking...")

    chunked_data = chunk_text(preprocessed_data)

    logger.info(f"Dataset prepared and saved. Number of samples: {len(chunked_data)}")

    logger.info("Splitting the dataset into train, test, and validation sets...")

    dataset = Dataset.from_list(chunked_data)

    # Splitting the dataset into training, validation, and test sets
    train_test_split = dataset.train_test_split(test_size=0.01)
    train_val_split = train_test_split["train"].train_test_split(test_size=0.01)

    # Combine splits into a single DatasetDict
    final_splits = DatasetDict(
        {
            "train": train_val_split["train"],
            "validation": train_val_split["test"],
            "test": train_test_split["test"],
        }
    )

    # Print some information about the datasets
    print("Train Dataset:", final_splits["train"])
    print("Validation Dataset:", final_splits["validation"])
    print("Test Dataset:", final_splits["test"])

    for split, split_dataset in final_splits.items():
        split_dataset.to_json(f"data/{OUTPUT_READY_DATASET_PATH}/{split}.jsonl")

    # Example data point from training set
    print("Example data point:", final_splits["train"][0])
