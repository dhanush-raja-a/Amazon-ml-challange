# Smart Product Pricing Challenge - ML Solution

## Problem Statement

This project tackles the Smart Product Pricing Challenge of ML Challenge 2025, aimed at predicting optimal product prices for e-commerce platforms. Pricing products correctly is critical for marketplace success and customer satisfaction. The dataset includes product details comprising textual descriptions, images, and item pack quantities, where the price depends on a complex interplay of attributes such as brand, specifications, and quantity. The goal is to build a machine learning model that holistically analyzes these attributes and predicts product prices accurately.

## Dataset Description

- **Training dataset:** 75,000 products with complete details including prices.
- **Test dataset:** 75,000 products without prices for evaluation.
- **Columns:**
  - `sample_id`: Unique identifier for each product.
  - `catalog_content`: Concatenated text containing title, description, and item pack quantity.
  - `image_link`: URL to download the product image.
  - `price`: Target variable in training data (to predict for test set).

## Output Format

Submit a CSV file with two columns:
- `sample_id`: Matching ID from the test set.
- `price`: Predicted price as a positive float.

## Approach and Methodology

### Data Processing and Feature Engineering
- Extract and clean textual information from `catalog_content`.
- Extract item pack quantity using regex parsing.
- Download product images from `image_link` using provided utility functions.
- Extract visual features from product images using a pretrained ResNet50 model (feature extractor without classifier) for image embedding generation.
- Combine textual and image features into a final feature set.

### Model Architecture
- Trained LightGBM regression models separately for:
  - Text features (TF-IDF vectors + numerical features).
  - Image features from ResNet50 embeddings.
- Ensemble prediction by weighted averaging of text- and image-based model outputs.

### Implementation Details
- Code implemented in Python with PyTorch, torchvision, LightGBM, scikit-learn, and other standard libraries.
- Used GPU acceleration for image feature extraction.
- Extensive preprocessing to handle missing data and feature normalization.
- Used Symmetric Mean Absolute Percentage Error (SMAPE) as the primary evaluation metric.

### AWS SageMaker Utilization
- Employed AWS SageMaker for scalable model training and experimentation.
- Utilized SageMaker compute instances to efficiently train LightGBM models and process large datasets.
- Leveraged SageMaker’s managed environment to handle data storage, image downloading, and batch processing.
- Model training steps, feature extraction, and inference pipelines orchestrated on SageMaker.

## Evaluation Metric
- The SMAPE metric is used for model evaluation, comparing predicted prices with actual prices on a relative percentage scale.

\[
\text{SMAPE} = \frac{1}{n} \sum_{i=1}^n \frac{|P_i - A_i|}{(|A_i| + |P_i|)/2} \times 100\%
\]

where \(P_i\) is the predicted price and \(A_i\) is the actual price for sample \(i\).

## How to Run

1. Prepare environment with required Python packages (see notebook for pip install commands).
2. Use `src/utils.py` for image downloading with retry mechanisms.
3. Run notebook cells sequentially to:
   - Load and preprocess data.
   - Download and extract image features.
   - Train LightGBM models on text and image features.
   - Generate ensemble predictions.
4. Save final predictions matching the test set sample IDs for submission.

## Notes

- No external price lookup or data source outside the provided dataset was used.
- All predictions and features rely solely on the given training data and image URLs.
- Follow licensing terms (MIT/Apache 2.0) for final model use.
