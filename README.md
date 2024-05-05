# Finetuning Phi-1.5 for Kotlin code completion

This a code for the task from [JetBrains Internship ](https://internship.jetbrains.com/)"Improving LLMs on underrepresented programming languages" and "Project Adaptation for Code Modeling models"

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

After collecting the files, we need to clean the data and build an actual code completion dataset.

Preprocessing includes the following steps:

* Removing copyright statements at the beginning of the files
* Removing files that are too short (sometimes the file consists of a single line, so we can't use it even for the next line completion)
* Skipping examples with very large numbers or values. Some files store large values for testing, for example, `const   val   S127   =   "-\uD83C\uDF10--\uD83C\uDF10\ud83c\udf09\uD83C\uDF10-\uD83C\uDF10-\ud83c\udf09\uD83C\uDF10\ud83c\udf09\uD83C\uDF10\ud83c\udf09--\uD83C\uDF10\uD83C\uDF10\uD83C\uDF10-\uD83C\uDF10-\uD83C\uDF10---\uD83C\uDF10-\uD83C\uDF10--... `They will cause problems with tokenization and are not useful for training the model

Phi models do not use special tokens such as `<EOL>` or `<INDENT>`, so there is no need to paste them

As for the format of the final dataset, I've decided to construct it using chunking with a small sliding window. The reasons for choosing this format are the following:

- Phi-1.5 is a small model, and its training data contains a lot of textbook Python tasks. This suggests that the finetuning data should be short-form: instead of predicting a large code chunk, we should stick to completing only a few lines at a time (e.g. a short method or function)
- The same goes for the suggested benchmark [CodeXGLUE Python](https://huggingface.co/datasets/microsoft/codexglue_method_generation) – the prompt there is always a function definition, and the reponse is a few-line solution, so I decided to organise my dataset in a similar way length-wise
- Ideally, I would extract function and class declarations with corresponding docstrings as a prompt and take the corresponding function or method code as a completion. But the given codebase has a different format – it's not a textbook or a competition task where we would get a detailed description of what the following code should do in order to provide the model with some context; mostly we get just plain code. Therefore, we can't really construct a dataset in instruct-like fashion and have to stick to just chunking code examples.
  So, to train this model further I would try to collect some textbook data, like the original paper [Textbooks Are All You Need](https://www.microsoft.com/en-us/research/publication/textbooks-are-all-you-need-ii-phi-1-5-technical-report/) suggests

As for the sliding window, the text is chunked by roughly 100 tokens based on puctuation (`"\n", "\t", "}" or " "`). 100 was chosen because the median length of the code content in each file is 200 tokens, so it would suffice for splittling even smaller files in at least one prompt-completion pair.

I decided to implement the sliding window in this way because realistically a code completion model can be invoked in the middle ot the line or method. Therefore, ideally, we would want the model to not only write a full method / function solutions, but also to be able to complete the code mid-line.

Running this script saves train, test, and validation splits to `data/kotlin_prepared_dataset`. Due to their large size, I uploaded these splits and the raw data file to Huggingface instead of storing them in the Github repository. All the necessary data can be found at [rinapch/jb_intern_kotlin](https://huggingface.co/datasets/rinapch/jb_intern_kotlin)

The numbers of samples are:

* train: 314k
* validation: 3.17k
* test: 3.2k

## Finetuning

Training script pushes the model to the hub, so you need to authenticate:

```
huggingface-cli login
```

Then run the script:

```
python3 train.py
```

This script tokenizes and prepares a dataset and launches training with Huggingface API

Since I'm GPU poor and needed to speed up the training, I used quantization with bitsandbytes and LoRA. The model was trained on a single RTX 4090 for approximately 4.5 hours. For this, I sampled 100k observations from the train file. The validation loss kept falling so it would be a good idea to keep training!

Since we trained with LoRA, we need to merge the adapters into the corresponding layers (this does not require GPU):

```
python3 merge_lora.py --hf_repository <your_model>
```

`<your_model>` should be a HF repository with your training checkpoint. This script will upload the fully merged model to the same repository.

The final model of my finetune run can be acessed at [rinapch/phi1.5-kotlin-finetune](https://huggingface.co/rinapch/phi1.5-kotlin-finetune/tree/main)

## Benchmarking

I will report the same metrics as in [CodeXGLUE for Python](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/Method-Generation/README.md#result): edit similarity and BLEU score. For benchmarking run:

```
python3 benchmarking.py --hf_repository <your_model> --sample <num>
```

BLUE calculation is copied directly from [CodeXGLUE repository](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/Method-Generation/evaluator/bleu.py) (it's in the file `bleu.py`). `benchmarking.py` follows the postprocessing steps from the same repo. It runs pretrained Phi-1.5 and finetuned `<your_model>` on CodeXGLUE method generation benchmark as well as on the test part of the Kotlin dataset obtained from running `preprocess_data.py`

NB: I am using a local file with CodeXGLUE test set (`data/test_codexglue.jsonl)`, because there is some problem with loading this data directly from Huggingface.

`--sample` argument lets you sample some number of test examples from the two datasets because CodeXGLUE contains 20k obervations which can be quite large computationally. By default all of the obervations are used. If you set more samples than there are in the dataset, it will also evaluate on the whole dataset. Since I'm still GPU poor, I used 500 observations from each data source

Here is the resulting table for my finetuned model:

| Model                          | CodeXGLUE edit sim | CodeXGLUE BLEU | Kotlin edit sim | Kotlin BLEU |
| ------------------------------ | ------------------ | -------------- | --------------- | ----------- |
| microsoft/phi-1_5              | 12.98              | 10.98          | 6.53            | 1.71        |
| rinapch/phi1.5-kotlin-finetune | 14.02              | 10.71          | 7.51            | 2.12        |

The increase isn't much, but it's honest work 🤠

To improve the model, I would

* Train on the whole train set (it would take approximately 13 hrs)
* Gather textbook and generally instruct-style data or produce it synthetically from a bigger model
