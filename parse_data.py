import jsonlines
from tqdm import tqdm
import subprocess
import glob
from loguru import logger
from config import REPO_URL, TMP_KOTLIN_REPO_PATH, OUTPUT_RAW_FILE


def clone_repository(url, clone_path):
    """Clones a git repository from GitHub to a local path
    Cloning the repositiry instead of fetching inividual files with Github API
    was chosen, because it is faster and eliminates the need to account for request limits.
    """
    subprocess.run(["git", "clone", url, clone_path], check=True)


def remove_directory(directory_path):
    """Removes a directory and all its contents"""
    subprocess.run(["rm", "-rf", directory_path], check=True)


def extract_kotlin_files(directory_path):
    """Extracts all Kotlin code from files in the given directory"""
    all_files = glob.glob(directory_path + "/**/*.kt", recursive=True)
    kotlin_code = []
    for file_path in tqdm(all_files, total=len(all_files)):
        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()
            kotlin_code.append({"file_text": file_content})
    return kotlin_code


def save_data(data, file_name):
    """
    Saves collected Kotlin code to a jsonlines file.
    """
    with jsonlines.open(f"data/{file_name}", mode="w") as writer:
        for item in data:
            writer.write(item)
    logger.info(f"Data saved to {file_name}")


def main():
    logger.info("Cloning repository (it's big so it might take a while)...")
    clone_repository(REPO_URL, TMP_KOTLIN_REPO_PATH)

    logger.info("Processing Kotlin files...")
    dataset = extract_kotlin_files(TMP_KOTLIN_REPO_PATH)

    logger.info(f"Found {len(dataset)} Kotlin files.")

    save_data(dataset, OUTPUT_RAW_FILE)

    logger.info("Cleaning up: removing cloned repository...")
    remove_directory(TMP_KOTLIN_REPO_PATH)

    logger.info("Process completed. Repository removed to save space.")


if __name__ == "__main__":
    main()
