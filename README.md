# AutoKeras Text Classification

Automated text classification with AutoKeras — spam detection and product review analysis, zero architecture decisions required.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wsamuelw/autokeras-text-classification-demo/blob/main/autokeras_demo.ipynb)

## Problem

Text classification typically requires choosing an architecture (CNN? LSTM? BERT?), tuning hyperparameters, and iterating for days. AutoKeras automates this — it searches for the best neural network architecture given your text data and labels.

## What's Inside

| Notebook | Task | Dataset | Classes |
|----------|------|---------|---------|
| `autokeras_demo.ipynb` | Spam detection | UCI SMS Spam Collection | Spam / Not Spam |
| `ecomm_review_demo.ipynb` | Purchase recommendation | Kaggle Women's Clothing Reviews | Recommend / Don't Recommend |

Both notebooks follow the same pipeline: load → clean → split → AutoKeras → evaluate → predict on unseen text.

## Pipeline

```
Raw text → Label encoding → Train/test split → AutoKeras architecture search → Evaluate → Predict
```

| Step | What Happens |
|------|-------------|
| Load | Pull data from UCI or upload CSV |
| Clean | Convert labels to numeric (spam=1, ham=0) |
| Split | 75/25 train/test |
| Search | AutoKeras tries `max_trials` architectures |
| Evaluate | Classification report (precision, recall, F1) |
| Predict | Classify any unseen text string |

## Results

### Spam Detection

AutoKeras searches architectures automatically. With `max_trials=3`, it tests 3 different neural network configurations and picks the best. The classification report shows precision, recall, and F1 for both spam and not-spam classes.

### Product Review Classification

Same approach on e-commerce reviews — predicting whether a customer would recommend a product based on their review text. With `max_trials=2`, the search is faster but still finds a competitive model.

## How AutoKeras Works

You don't design the architecture — AutoKeras does:

```python
import autokeras as ak

# that's it — it searches for the best text classifier
clf = ak.TextClassifier(max_trials=3)
clf.fit(x_train, y_train, validation_split=0.3)

# predict on new text
clf.predict(np.array(["Free money now!"]))
```

Behind the scenes, it tries different:
- **Embedding layers** — how to represent words as vectors
- **Architecture types** — CNN, LSTM, Transformer, or combinations
- **Hyperparameters** — layer sizes, dropout rates, learning rates

The `max_trials` parameter controls how many architectures to try (more = better results, but slower).

## Setup

### Google Colab

Click the badge above — runs entirely in the browser.

### Local

```bash
pip install autokeras tensorflow scikit-learn pandas
git clone https://github.com/wsamuelw/autokeras-text-classification-demo.git
cd autokeras-text-classification-demo
jupyter notebook autokeras_demo.ipynb
```

## Data Sources

| Dataset | Source | Rows | Description |
|---------|--------|------|-------------|
| SMS Spam Collection | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/SMSSpamCollection) | 5,574 | SMS messages labelled spam/ham |
| Women's Clothing Reviews | [Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) | 23,486 | Product reviews with recommendation labels |

## When to Use AutoKeras

- **Rapid prototyping** — get a baseline model in minutes, not days
- **No DL expertise needed** — the architecture search handles model design
- **Text, image, or tabular** — AutoKeras handles multiple data types
- **Limited time** — when you need results faster than manual tuning

## When NOT to Use It

- **Production** — AutoKeras gives you a model, not a production pipeline
- **Large datasets** — architecture search is slow on big data
- **Custom architectures** — if you know exactly what you need, build it manually
- **Interpretability** — neural networks are black boxes; use tree models if you need explainability

## Tech Stack

- **AutoKeras** — automated deep learning architecture search
- **TensorFlow / Keras** — underlying DL framework
- **scikit-learn** — train/test split and evaluation
- **pandas** — data manipulation

## References

- [AutoKeras documentation](https://autokeras.com/)
- [Classify text with BERT](https://www.tensorflow.org/text/tutorials/classify_text_with_bert)
- [Text classification with AutoKeras](https://towardsdatascience.com/text-classification-made-easy-with-autokeras-c1020ff60b17)

## License

MIT
