# Finetuning Phi-1.5 for Kotlin code completion

This a code for solving a task for [JetBrains Internship ](https://internship.jetbrains.com/)

## Setup

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

Phi models do not use special tokens such as `<EOL>` or `<INDENT>`, so there is no need to paste them

As for the format of the final dataset, I've decided to construct it using a small sliding window (arbitrarily I chose 50 tokens). The reasons for choosing this format are the following: 

- Phi-1.5 is a small model, and it's training data containes a lot of textbook Python tasks. This suggests that the finetuning data should be short-form: instead of predicting a large code chunk, we should stick to completing only a few lines at a time (e.g. a short method or function)
- Same goes for the suggested benchmark [CodeXGLUE Python](https://huggingface.co/datasets/microsoft/codexglue_method_generation) – the prompt there is always a function definition, and the reponse is a few-line solution, so I decided to organise my dataset in a similar way length-wise
- Ideally, I would extract function and class declarations with corresponding docstrings as a prompt and take the corresponding function or method code as a completion. But the given codebase has a different format – it's not a textbook or a competition task where we would get a detailed description of what the following code should do in order to provide the model with some context; mostly we get just plain code. Therefore, we can't really construct a dataset in instruct-like fashion and have to stick to just chunking code examples that we have
  So, to train this model further I would try to collect some textbook data, like the original paper [Textbooks Are All You Need](https://www.microsoft.com/en-us/research/publication/textbooks-are-all-you-need-ii-phi-1-5-technical-report/) suggests
