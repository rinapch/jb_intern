This subfolder stores the outputs for `parse_data.py` and `preprocess_data.py`

The resulting raw file with code scraped from Kotlin repository as well as training files were too large to be stored in Github directly, so they are uploaded to Huggingface at [rinapch/jb_intern_kotlin](https://huggingface.co/datasets/rinapch/jb_intern_kotlin).

`test_codexglue.jsonl `stores [CodeXGLUE for Python](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/Method-Generation/README.md#result) method generation test parts. It is required for `benchmark.py`, because loading it directly from Huggingface yields an error.
