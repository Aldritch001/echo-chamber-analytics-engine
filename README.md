# Echo Chamber Analytics Engine 🌐

A real-time, end-to-end natural language processing (NLP) and machine learning pipeline built to monitor, cluster, and quantify ideological polarization across digital media streams.

## 🔬 System Architecture & Methodology

The platform implements a multi-stage data science pipeline designed to process unstructured textual streams into structural community insights on demand:

1. **Dynamic Data Harvester:** Uses a live XML processing layer to intercept real-time global news and media narrative streams based on user-selected keywords, bypassing local platform caching blocks completely.
2. **Supervised Sequence Classification:** Integrates a pre-trained `DistilBERT` transformer deep learning model (`distilbert-base-uncased-finetuned-sst-2-english`) via Hugging Face to evaluate categorical positive and negative sentiment distributions over variable string lengths.
3. **Unsupervised Vectorization & Clustering:** Transforms unstructured text records into high-dimensional geometric coordinate matrices utilizing a **TF-IDF Vectorizer** (filtering out English stop words). These matrices feed into a **K-Means Clustering** algorithm to isolate distinct communication pockets.
4. **Proprietary Echo Chamber Index (ECI):** Combines sentiment concentration ratios and cluster group distribution variance into a single, standardized mathematical metric:

$$ECI = (\text{Dominant Sentiment Ratio} \times 0.6) + (\text{Cluster Variance Skew} \times 0.4) \times 100$$

A higher ECI signifies that a topic's community has completely split into structurally isolated, un-crossing dialogue camps.

## 📊 Live Interactive Dashboard Features
* **Instant Auto-Submit:** Seamlessly switches sectors (Nvidia, Bitcoin, Tesla, Apple) and executes the entire machine learning pipeline on dropdown clicks.
* **Executive Topic Breakdown:** Features an automated data narrator panel that reads raw dataset values to isolate prominent vocabulary terms and trace dominant core cluster alignments dynamically.
* **Dual-Column UX:** Renders side-by-side comparative views matching data visualizations directly to text insights.

## 🛠️ Tech Stack & Dependencies
* **Core Language:** Python
* **UI Architecture:** Streamlit Framework
* **Machine Learning & NLP:** Scikit-Learn, Hugging Face Transformers, PyTorch
* **Data Visualization & Analytics:** Pandas, NumPy, Matplotlib, Seaborn

## 🏃‍♂️ Local Installation & Execution

To run this platform locally on a Windows architecture without modifying global environment path variables:

1. Clone this repository to your desktop.
2. Open your terminal engine inside the folder directory and run:
   ```bash
   pip install streamlit pandas numpy torch transformers scikit-learn matplotlib seaborn