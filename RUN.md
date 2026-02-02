# How to Run This Code

## 1. Fix the package name

The code imports the package as **`social_media_nlp`**, but the folder is named **`Code_Paper_news_popularity`**. You must make Python find `social_media_nlp`.

**Option A (recommended):** Rename the package folder and use `src` in `PYTHONPATH`:

- From the **project root** (`e:\nlp\social_media_nlp`), set:
  - **Windows (PowerShell):**  
    `$env:PYTHONPATH = "e:\nlp\social_media_nlp\src"`
  - **Windows (CMD):**  
    `set PYTHONPATH=e:\nlp\social_media_nlp\src`
  - **Linux/macOS:**  
    `export PYTHONPATH=/path/to/social_media_nlp/src`

**Option B:** Keep the folder name and add a symlink (Windows: need admin or Developer Mode for symlinks):

- Create `src/social_media_nlp` as a symlink to `src/Code_Paper_news_popularity`.
- Set `PYTHONPATH` to `e:\nlp\social_media_nlp\src` as above.

---

## 2. Environment

From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

- **GPU:** Install PyTorch with CUDA from [pytorch.org](https://pytorch.org) if you use GPU.
- **NLTK data** (needed for `data/preprocessing.py`):

  ```python
  import nltk
  nltk.download("punkt")
  nltk.download("stopwords")
  nltk.download("wordnet")
  ```

---

## 3. What you can run

All commands below are from the **project root** (`e:\nlp\social_media_nlp`) with `PYTHONPATH` set to `src` (and the package visible as `social_media_nlp`).

### ML (classical models + hyperparameter tuning)

Uses **tweet_eval/sentiment** from Hugging Face. Trains LogisticRegression and RandomForest with TF-IDF and embeddings, logs to MLflow, saves predictions under `./models/tweet_eval/predictions/`.

```bash
python -m social_media_nlp.experiments.ml.tune
```

### LLM fine-tuning (LoRA)

Fine-tunes a causal LM (e.g. Phi-3, Mistral) on tweet_eval/sentiment with LoRA. Needs GPU and enough RAM.

```bash
python -m social_media_nlp.experiments.llm.train MODEL_ID -r 8 -a 16 -e 3 -t 1000 -v 200
```

Example: `microsoft/Phi-3-mini-4k-instruct` (replace with your model id).

### LLM evaluation (fine-tuned or merged model)

Evaluates a trained/merged model on tweet_eval/sentiment test set. Saves metrics and predictions under `./models/tweet_eval/predictions/`.

```bash
python -m social_media_nlp.experiments.llm.evaluate /path/to/model
```

### LLM few-shot evaluation

Evaluates a base (non–fine-tuned) model with 0, 3, 6, … 21 few-shot examples.

```bash
python -m social_media_nlp.experiments.llm.evaluate_few_shot /path/to/model
```

### Merge LoRA adapters into base model

```bash
python -m social_media_nlp.experiments.llm.merge_models /path/to/base_model /path/to/fine_tuned_subfolder
```

### Seq LM (encoder sequence classification)

Fine-tunes a sequence-classification model (e.g. BERT) on tweet_eval/sentiment.

```bash
python -m social_media_nlp.experiments.seq_lm.train MODEL_ID 3
```

(3 = number of epochs.)

### Seq LM evaluation

```bash
python -m social_media_nlp.experiments.seq_lm.evaluation MODEL_PATH
```

---

## 4. Data

- **ML and LLM/seq_lm scripts** use **Hugging Face `tweet_eval` (sentiment)**; they download it automatically. No need to use your local `dataset/` for these.
- Your **`dataset/`** (Facebook, Instagram, TikTok, Twitter CSVs and train_*.txt) is **not** used by the current entry points. To use it you’d need to add a data loader that reads those files and passes them into the existing preprocessing/training code.

---

## 5. Quick checklist

1. Set `PYTHONPATH` to `e:\nlp\social_media_nlp\src`.
2. Create venv, install `requirements.txt`, install NLTK data.
3. Run from project root, e.g.  
   `python -m social_media_nlp.experiments.ml.tune`  
   or any of the commands above.
