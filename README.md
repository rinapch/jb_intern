# Finetuning Phi-1.5 for Kotlin code completion

This a code for solving a task for [JetBrains Internship ](https://internship.jetbrains.com/)"Improving LLMs on underrepresented programming languages" and "Project Adaptation for Code Modeling models"

## Setup

Just create a venv and install the requirements

```
python3 -m venv jb_intern
pip install -r requirements.txt
```

## Parsing files

Done by running:

```
python3 parse_data.py
```

The script clones [JetBrains Kotlin repository](https://github.com/JetBrains/kotlin) to `'/tmp/kotlin_repo'` and collects all files with `.kt` extension. Alternatively one could recursively fetch those files with Github API without cloning. However, the repository is quite large and this approach ends up taking too long. Additionally, with such a large repository one will likely run into a rate limit even with authenticated requests.

After processing all the files, `'/tmp/kotlin_repo'` is automatically removed. The raw dataset with `.kt` files contents is saved to `'data/kotlin_code_raw.jsonl'`. It is a jsonlines file where each line is `{"file_text": str}`. It yields approximately 54k samples.

Since this file too large to be stored in Github, I uploaded it on Huggingface here

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

As for the format of the final dataset, I've decided to construct it using a small sliding window (arbitrarily I chose 35 tokens). The reasons for choosing this format are the following:

- Phi-1.5 is a small model, and it's training data containes a lot of textbook Python tasks. This suggests that the finetuning data should be short-form: instead of predicting a large code chunk, we should stick to completing only a few lines at a time (e.g. a short method or function)
- Same goes for the suggested benchmark [CodeXGLUE Python](https://huggingface.co/datasets/microsoft/codexglue_method_generation) – the prompt there is always a function definition, and the reponse is a few-line solution, so I decided to organise my dataset in a similar way length-wise
- Ideally, I would extract function and class declarations with corresponding docstrings as a prompt and take the corresponding function or method code as a completion. But the given codebase has a different format – it's not a textbook or a competition task where we would get a detailed description of what the following code should do in order to provide the model with some context; mostly we get just plain code. Therefore, we can't really construct a dataset in instruct-like fashion and have to stick to just chunking code examples that we have
  So, to train this model further I would try to collect some textbook data, like the original paper [Textbooks Are All You Need](https://www.microsoft.com/en-us/research/publication/textbooks-are-all-you-need-ii-phi-1-5-technical-report/) suggests

As for sliding window, the text is split using `re.split(r"(\s+)", text)` so that each word, special token, space or tab is taken separately. Then frist 35 "tokens" are selected as a prompt and next 35 tokens as a competion. If the text is shorter than 70 "tokens", I just split it in two parts based on a newline. If there are not enough tokens for a completion part, I just include the text until the end of the file.

I decided to implement sliding window in this way because realistically a code completion model can be invoked in the middle ot the line or method. Therefore, ideally we would want a code completion model to not only write a full method / function solution, but also to complete the code mid-line.

Running this script saves train, test and validation splits to `data/kotlin_prepared_dataset`. Due to a large size, I uploaded these splits and the raw data file to huggingface instead of storing them in the Github repository. All the necesssary data can be found at [rinapch/jb_intern_kotlin](https://huggingface.co/datasets/rinapch/jb_intern_kotlin)

## Benchmarking baseline model

## Finetuning

## Benchmarking funetuned model
