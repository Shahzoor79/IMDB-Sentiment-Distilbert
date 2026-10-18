# Sentiment Analysis on IMDB Reviews: TF-IDF Baseline vs Fine-Tuned DistilBERT

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shahzoor79/imdb-sentiment-distilbert/blob/main/sentiment_analysis.ipynb)

## Problem
Classify movie reviews as positive or negative, and measure whether a fine-tuned transformer is worth its extra cost over a classic baseline. The goal is not just a headline accuracy, but understanding *where and why* the model fails.

## Approach
- **Data:** IMDB reviews from Hugging Face `datasets`: 5,000 train / 500 validation / 1,000 test, shuffled with a fixed seed. Validation is carved from the training pool, so the test set is used exactly once.
- **Baseline:** TF-IDF (1-2 grams) + Logistic Regression, with the regularisation strength chosen on the validation set.
- **Transformer:** `distilbert-base-uncased` fine-tuned with the Hugging Face `Trainer` (PyTorch) on a free Colab T4 GPU.
- **Evaluation:** identical metrics for both models on the same test set (accuracy, per-class precision / recall / F1, confusion matrices), plus training and inference time.
- **Error analysis:** the most confident mistakes are read and grouped into patterns (negation, mixed sentiment, sarcasm, truncation). A pattern is only treated as an explanation if reviews carrying it fail more often than average.
- **Demo:** a `predict` function and a small Gradio app.

## Design decisions worth knowing
- **No leakage:** separate train / validation / test; TF-IDF fitted on training text only; test never used for tuning.
- **Shuffle before subsetting:** raw IMDB is sorted by label, so an unshuffled test subset would be all one class.
- **Dynamic padding:** batches are padded to their own longest review instead of a fixed length, which is faster on a free GPU.
- **Truncation at 256 tokens:** a speed/accuracy trade-off, and its cost is measured in the error analysis.
- **Eval mode:** inference calls `model.eval()` and `torch.no_grad()` so dropout is off and predictions are stable.

## Repository structure
```
├── sentiment_analysis.ipynb   # the full project, stages 1-8
├── requirements.txt
├── outputs/                   # plots saved by the notebook
├── LICENSE
└── README.md
```

## How to run
1. Click the **Open in Colab** badge above (or upload `sentiment_analysis.ipynb` to Colab).
2. **Runtime > Change runtime type > T4 GPU**.
3. **Runtime > Run all** (about 10 minutes with the default subset).
4. To scale up, change `N_TRAIN` / `N_TEST` in the data cell (`N_TRAIN` up to 24,500).
5. To push the model to the Hub, add a Write token as a Colab secret named `HF_TOKEN`.

To run locally you need a CUDA GPU: `pip install -r requirements.txt`, then open the notebook in Jupyter and remove the `!pip install` line.

## Results
The notebook's final stage generates a `README.md` containing the results table, confusion matrices, error-analysis findings and the Hugging Face model link from your actual run. After running it, replace this file and the contents of `outputs/` with the generated `README.md` and plots (the notebook offers them as `portfolio.zip`).

## License
MIT. See `LICENSE`.
