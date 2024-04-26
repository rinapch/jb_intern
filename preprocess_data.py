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


def get_clean_data(data: List[Dict[str, str]]) -> List[str]:
    """Parse JSON and clean the code text by removing specific copyright blocks."""
    cleaned_data = []
    for sample in data:
        code = sample["file_text"]

        # Regex to remove the copyright block as specified
        copyright_pattern = (
            r"/\*\n \* Copyright.*?\*/\n\n"
        )
        code = re.sub(copyright_pattern, "", code, flags=re.DOTALL)
        # I decided not to remove comments because they can be useful for context & desirable for code completion

        if _is_long_sample(code):
            cleaned_data.append(code)

        else:  # if sample is short just don't save it
            continue

    return cleaned_data


def _is_long_sample(text: str) -> bool:
    """Filter out entries that are too short based on line breaks."""
    if text.count("\n") <= 1:
        return False
    return True


def chunk_text(data: List[str]) -> List[Dict[str, str]]:
    """Generate (context, target) pairs from the given dataset."""
    # Not using word_tokenize here because spaces and tabs are important for code completion
    # and word_tokenize ignores them
    chunked_pairs = []
    for text in tqdm(data, total=len(data)):
        words = re.split(r"(\s+)", text)
        if len(words) < MIN_CONTEXT_LENGTH + 1:
            context, target = text.split(
                "\n", 1
            )  # here we are guaranteed to have at least one \n
            chunked_pairs.append({"prompt": context + "\n", "completion": target + EOS_TOKEN})
            continue

        i = 0   
        while i + MIN_CONTEXT_LENGTH + MIN_CONTEXT_LENGTH <= len(words):
            # Set context
            context = " ".join(words[i:i+MIN_CONTEXT_LENGTH])
            
            # Set target
            target = " ".join(words[i+MIN_CONTEXT_LENGTH:i+MIN_CONTEXT_LENGTH+MIN_CONTEXT_LENGTH])
            
            # Yield the context and target
            chunked_pairs.append({'prompt': context, 'completion': target + EOS_TOKEN})
            
            # Move the index forward by the context size for overlapping windows
            # or by context_size + target_size for non-overlapping windows
            i += MIN_CONTEXT_LENGTH

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

    final_splits.save_to_disk(f"data/{OUTPUT_READY_DATASET_PATH}")

    # Example data point from training set
    print("Example data point:", final_splits["train"][0])
