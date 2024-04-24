REPO_URL = "https://github.com/JetBrains/kotlin.git"
TMP_KOTLIN_REPO_PATH = "/tmp/kotlin_repo"  # Temporary path for cloning

OUTPUT_RAW_FILE = "kotlin_code_raw.jsonl"
OUTPUT_TRAIN_FILE = "kotlin_code_train.json"
OUTPUT_TEST_FILE = "kotlin_code_test.json"
OUTPUT_VALID_FILE = "kotlin_code_valid.json"

MIN_CONTEXT_LENGTH = 50  # Number of words (or special characters) to consider as context for code completion
EOS_TOKEN = "<|endoftext|>"
