# Finetuning Phi-1.5 for Kotlin code completion

This a code for solving a task for [JetBrains Internship ](https://internship.jetbrains.com/)"Improving LLMs on underrepresented programming languages" and "Project Adaptation for Code Modeling models"

## Setup

Just create a venv and install the requirements

```
python3 -m venv jb_intern
source jb_intern/bin/activate
pip3 install -r requirements.txt
```

## Parsing files

Done by running:

```
python3 parse_data.py
```

The script clones [JetBrains Kotlin repository](https://github.com/JetBrains/kotlin) to `'/tmp/kotlin_repo'` and collects all files with `.kt` extension. Alternatively one could recursively fetch those files with Github API without cloning. However, the repository is quite large and this approach ends up taking too long. Additionally, with such a large repository one will likely run into a rate limit even with authenticated requests.

After processing all the files, `'/tmp/kotlin_repo'` is automatically removed. The raw dataset with `.kt` files contents is saved to `'data/kotlin_code_raw.jsonl'`. It is a jsonlines file where each line is `{"file_text": str}`. It yields approximately 54k samples.

## Preprocessing the data & building a dataset

Done by running:

```
python3 preprocess_data.py
```

After collecting the flies, we need to clean the data and build an actual code complletion dataset.

Preprocessing includes the following steps:

* Removing copyright statements in the beginning of the files
* Removing files that are too short (sometimes the file consists of a single line, so we can't use it even for next line completion)
* Skipping examples with very large numbers or values. Some files store large values for testing, for example, `const   val   S127   =   "-\uD83C\uDF10--\uD83C\uDF10\ud83c\udf09\uD83C\uDF10-\uD83C\uDF10-\ud83c\udf09\uD83C\uDF10\ud83c\udf09\uD83C\uDF10\ud83c\udf09--\uD83C\uDF10\uD83C\uDF10\uD83C\uDF10-\uD83C\uDF10-\uD83C\uDF10---\uD83C\uDF10-\uD83C\uDF10--... `They will cause problem with tokenization and are not useful for training the model

Phi models do not use special tokens such as `<EOL>` or `<INDENT>`, so there is no need to paste them

As for the format of the final dataset, I've decided to construct it using a small sliding window (arbitrarily I chose 35 tokens). The reasons for choosing this format are the following:

- Phi-1.5 is a small model, and it's training data containes a lot of textbook Python tasks. This suggests that the finetuning data should be short-form: instead of predicting a large code chunk, we should stick to completing only a few lines at a time (e.g. a short method or function)
- Same goes for the suggested benchmark [CodeXGLUE Python](https://huggingface.co/datasets/microsoft/codexglue_method_generation) – the prompt there is always a function definition, and the reponse is a few-line solution, so I decided to organise my dataset in a similar way length-wise
- Ideally, I would extract function and class declarations with corresponding docstrings as a prompt and take the corresponding function or method code as a completion. But the given codebase has a different format – it's not a textbook or a competition task where we would get a detailed description of what the following code should do in order to provide the model with some context; mostly we get just plain code. Therefore, we can't really construct a dataset in instruct-like fashion and have to stick to just chunking code examples that we have
  So, to train this model further I would try to collect some textbook data, like the original paper [Textbooks Are All You Need](https://www.microsoft.com/en-us/research/publication/textbooks-are-all-you-need-ii-phi-1-5-technical-report/) suggests

As for sliding window, the text is chunked by roughly 100 tokens based on puctuation (`" `). 100 was chosen because the median length of the code in each file is 200 tokens, so id would suffice for for splittling even smaller files in at least one prompt-completion pair.

I decided to implement sliding window in this way because realistically a code completion model can be invoked in the middle ot the line or method. Therefore, ideally we would want a code completion model to not only write a full method / function solution, but also to complete the code mid-line.

Running this script saves train, test and validation splits to `data/kotlin_prepared_dataset`. Due to a large size, I uploaded these splits and the raw data file to huggingface instead of storing them in the Github repository. All the necesssary data can be found at [rinapch/jb_intern_kotlin](https://huggingface.co/datasets/rinapch/jb_intern_kotlin)

## Finetuning

Training script pushes the model to hub, so you need to authenticate:

```
huggingface-cli login
```

Then run the script:

```
python3 train.py
```

This script tokenizes and prapares a datatset and lauches training with Huggingface API.

Since I'm GPU poor and needed to speed up the training, I used quantization with bitsandbytes and LoRA. The model was trained on a single RTX 4090 for approximately 4.5 hours. For this, I sampled 100k observations from the train file. The validation loss kept falling so it would be a good idea to keep training!

Since we trained with LoRA, we need to merge the adapters into the corresponding layers (this does not require GPU)):

```
python3 merge_lora.py --hf_repository <your_repo>
```

`<your_repo>` should be a repository with your training checkpoint. This script will upload the fully merged model to the same repository.

The final model of my finetune run can be acessed at [rinapch/phi1.5-kotlin-finetune](https://huggingface.co/rinapch/phi1.5-kotlin-finetune/tree/main)

## Benchmarking

I will report the same metrics as in [CodeXGLUE for Python](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/Method-Generation/README.md#result): Edit similarity and BLEU score. For benchmarking run:

```
python3 benchmarking.py --hf_repository <your_model>
```

BLUE calculation is copied directly from [CodeXGLUE repository](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/Method-Generation/evaluator/bleu.py) (in the file `bleu.py`). benchmarking.py follows the preprocessing steps from the same repo. It runs pretrained phi-1.5 and finetuned <`your_model>` on CodeXGLUE method generation benchmark as well as on the test set of the Kotlin dataset obtained from running `preprocess_data.py`. It outputs a table with edit similarity and BLEU score between the two models.

NB: I am using a local fule with CodeXGLUE test set, because there is some issue with loading this data directly from Huggingface.

Here is the resulting table for my fintuned model:
