# SageMaker Linear Learner Exercise

A hands-on exercise for learning Amazon SageMaker's Linear Learner algorithm for both classification and regression tasks, using **Batch Transform** for model evaluation.

---

## What is Linear Learner?

Linear Learner is Amazon SageMaker's built-in algorithm for supervised learning that supports:

- **Binary Classification**: Predict yes/no outcomes (e.g., customer churn)
- **Multiclass Classification**: Predict one of many categories
- **Regression**: Predict continuous values (e.g., house prices)

Key advantages:
- **Parallel Training**: Trains multiple models with different hyperparameters simultaneously
- **Built-in Normalization**: Handles feature scaling automatically
- **Regularization**: L1 and L2 regularization to prevent overfitting
- **Distributed Training**: Scales to large datasets

---

## Files Included

| File | Description |
|------|-------------|
| `linear_learner_sagemaker_exercise.ipynb` | Complete Jupyter notebook for the exercise |
| `generate_linear_learner_data.py` | Standalone script to generate synthetic data |

---

## Why Batch Transform (No Deployment)?

This exercise uses **Batch Transform** instead of deploying endpoints:

- **No endpoint costs**: Only pay for compute during the transform job
- **No cleanup required**: Resources automatically terminate after job completes
- **Ideal for evaluation**: Perfect for comparing predictions against test labels
- **Cost-effective**: Better for batch predictions and model evaluation

---

## Prerequisites

1. **AWS Account** with SageMaker access
2. **SageMaker Notebook Instance** or **SageMaker Studio**
3. **IAM Role** with permissions for:
   - S3 read/write
   - SageMaker training and batch transform
   - CloudWatch logs

---

## Step-by-Step Instructions

### Step 1: Set Up Your SageMaker Environment

**Option A: SageMaker Notebook Instance**
1. Go to AWS Console -> SageMaker -> Notebook instances
2. Click "Create notebook instance"
3. Choose `ml.t3.medium` (sufficient for this exercise)
4. Select or create an IAM role with SageMaker permissions
5. Wait for status to become "InService"
6. Click "Open JupyterLab"

**Option B: SageMaker Studio**
1. Go to AWS Console -> SageMaker -> Studio
2. Open your Studio domain
3. Create a new notebook with Python 3 kernel

### Step 2: Upload the Exercise Files

1. Download `linear_learner_sagemaker_exercise.ipynb` from this package
2. In JupyterLab/Studio, click the upload button
3. Upload the notebook file

### Step 3: Run the Notebook

Open `linear_learner_sagemaker_exercise.ipynb` and execute cells in order:

| Section | What It Does | Estimated Time |
|---------|--------------|----------------|
| Step 1-2 | Setup & generate data | 1-2 min |
| Step 3 | Visualize data | 30 sec |
| Step 4 | Upload to S3 | 1 min |
| **Part A: Classification** | | |
| Step 5A | Train classifier | **3-5 min** |
| Step 6A | Batch transform | **3-5 min** |
| Step 7A | Evaluate with metrics | 1-2 min |
| **Part B: Regression** | | |
| Step 5B | Train regressor | **3-5 min** |
| Step 6B | Batch transform | **3-5 min** |
| Step 7B | Evaluate with metrics | 1-2 min |

**Total time: ~25-35 minutes**

---

## Evaluation Metrics

### Classification Metrics

The notebook calculates these metrics for binary classification:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correctness (TP+TN) / Total |
| **Balanced Accuracy** | Average of recall for each class |
| **Precision** | True positives / Predicted positives |
| **Recall (Sensitivity)** | True positives / Actual positives |
| **Specificity** | True negatives / Actual negatives |
| **F1 Score** | Harmonic mean of precision and recall |
| **ROC AUC** | Area under the ROC curve |
| **Average Precision** | Area under the precision-recall curve |
| **Log Loss** | Logarithmic loss (probabilistic accuracy) |
| **Matthews Correlation** | Correlation between predicted and actual |
| **Negative Predictive Value** | True negatives / Predicted negatives |

**Visualizations included:**
- Confusion Matrix (counts and normalized)
- ROC Curve with AUC
- Precision-Recall Curve
- Score Distribution by Class
- Threshold Analysis (F1, Precision, Recall vs Threshold)

### Regression Metrics

The notebook calculates these metrics for regression:

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Square Error |
| **MAE** | Mean Absolute Error |
| **Median AE** | Median Absolute Error (robust to outliers) |
| **Max Error** | Worst case prediction error |
| **MSE** | Mean Squared Error |
| **MAPE** | Mean Absolute Percentage Error |
| **SMAPE** | Symmetric MAPE (handles zeros better) |
| **R-squared (R2)** | Coefficient of determination |
| **Adjusted R2** | R2 adjusted for number of predictors |
| **Explained Variance** | Proportion of variance explained |
| **Within X%** | Percentage of predictions within 5%, 10%, 20% |

**Visualizations included:**
- Actual vs Predicted Scatter Plot
- Residual Plot (vs Predicted Values)
- Residual Distribution with Normal Fit
- Percentage Error Distribution
- MAE/MAPE by Price Range
- Q-Q Plot for Normality Check
- Shapiro-Wilk Test for Residual Normality

---

## Understanding the Data Format

Linear Learner accepts **CSV format**:

**For Training** (label in first column):
```
1.0,0.5,2.3,1.0,...
0.0,1.2,0.8,0.5,...
```

**For Batch Transform Inference** (features only, no label):
```
0.5,2.3,1.0,...
1.2,0.8,0.5,...
```

---

## Exercise 1: Classification (Customer Churn)

### Synthetic Data

The classification data simulates customer churn prediction with these features:

| Feature | Description |
|---------|-------------|
| `tenure_months` | How long customer has been with company |
| `monthly_charges` | Monthly bill amount |
| `total_charges` | Total amount paid to date |
| `support_tickets` | Number of support tickets filed |
| `contract_type` | 0=Monthly, 1=1-year, 2=2-year |
| `payment_method` | Payment method (0-3) |
| `online_security` | Has online security service (0/1) |
| `tech_support` | Has tech support service (0/1) |
| `internet_service` | 0=None, 1=DSL, 2=Fiber |
| `num_products` | Number of products/services |

**Target**: Binary (0 = No Churn, 1 = Churn)

---

## Exercise 2: Regression (House Prices)

### Synthetic Data

The regression data simulates house price prediction with these features:

| Feature | Description |
|---------|-------------|
| `sqft` | Square footage of the house |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `lot_size_acres` | Size of the lot in acres |
| `year_built` | Year the house was built |
| `garage_capacity` | Number of cars garage holds |
| `has_pool` | Has swimming pool (0/1) |
| `distance_to_city` | Distance to city center (miles) |
| `school_rating` | Local school rating (1-10) |
| `crime_rate` | Local crime rate (per 1000) |

**Target**: Continuous (house price in dollars)

---

## Key Hyperparameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `predictor_type` | - | binary_classifier, multiclass_classifier, or regressor |
| `feature_dim` | - | Number of input features (required) |
| `epochs` | 15 | Training epochs |
| `mini_batch_size` | 1000 | Batch size |
| `learning_rate` | auto | Learning rate |
| `num_models` | auto | Number of parallel models to train |
| `normalize_data` | true | Normalize input features |
| `normalize_label` | auto | Normalize labels (for regression) |
| `l1` | auto | L1 regularization strength |
| `wd` | auto | L2 regularization (weight decay) |
| `optimizer` | auto | adam, sgd, or rmsprop |

---

## Using the Standalone Data Generator

If you want to generate data outside the notebook:

```bash
# Generate classification data (customer churn)
python generate_linear_learner_data.py --task classification

# Generate regression data (house prices)
python generate_linear_learner_data.py --task regression

# Custom parameters
python generate_linear_learner_data.py \
  --task classification \
  --num-samples 10000 \
  --test-ratio 0.3 \
  --output-dir ./data \
  --visualize
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | classification | Task type (classification or regression) |
| `--num-samples` | 5000 | Total number of samples |
| `--test-ratio` | 0.2 | Fraction for test set |
| `--seed` | 42 | Random seed |
| `--output-dir` | ./data | Output directory |
| `--visualize` | False | Generate visualization |

---

## Cost Estimates

| Resource | Instance | Cost (approx) |
|----------|----------|---------------|
| Training | ml.m5.large | ~$0.10 for 5 min |
| Batch Transform | ml.m5.large | ~$0.10 for 5 min |
| Notebook | ml.t3.medium | ~$0.05/hour |

**Note:** No endpoint cleanup required with batch transform!

---

## Troubleshooting

### "ResourceLimitExceeded" Error
- You've hit your account limits
- Request a limit increase in Service Quotas

### Training Job Fails
- Check CloudWatch logs for the training job
- Common issues: S3 permissions, data format errors

### Batch Transform Fails
- Ensure test data has same number of features as training data
- Check that input file is properly formatted CSV

### Poor Model Performance
- Try increasing epochs
- Adjust regularization (l1, wd)
- Check for data quality issues
- Consider feature engineering

---

## Resources

- [Linear Learner Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner.html)
- [Linear Learner Hyperparameters](https://docs.aws.amazon.com/sagemaker/latest/dg/ll_hyperparameters.html)
- [SageMaker Batch Transform](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
