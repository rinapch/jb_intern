REPO_URL = "https://github.com/JetBrains/kotlin.git"
TMP_KOTLIN_REPO_PATH = "/tmp/kotlin_repo"  # Temporary path for cloning

OUTPUT_RAW_FILE = "kotlin_code_raw.jsonl"
OUTPUT_READY_DATASET_PATH = "kotlin_prepared_dataset"

MIN_CONTEXT_LENGTH = 100  # Number of tokens for chunking
EOS_TOKEN = "<|endoftext|>"

DATASET_HF = "rinapch/jb_intern_kotlin"  # preprocessed data uploaded to hf
PRETRAINED_MODEL = "microsoft/phi-1_5"
