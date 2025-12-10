# SageMaker DeepAR Exercise

A hands-on exercise for learning Amazon SageMaker's DeepAR time series forecasting algorithm.

---

## What is DeepAR?

DeepAR is Amazon's autoregressive recurrent neural network algorithm for time series forecasting. It excels when you have:

- **Many related time series** (e.g., demand for 1000s of products)
- **Cold-start problems** (new products with limited history)
- **Need for probabilistic forecasts** (uncertainty quantification)

---

## Files Included

| File | Description |
|------|-------------|
| `deepar_sagemaker_exercise.ipynb` | Complete Jupyter notebook for the exercise (includes synthetic data generation) |

---

## Prerequisites

1. **AWS Account** with SageMaker access
2. **SageMaker Notebook Instance** or **SageMaker Studio** (or run locally with AWS credentials)
3. **IAM Role** with permissions for:
   - S3 read/write
   - SageMaker training and deployment
   - CloudWatch logs

---

## Step-by-Step Instructions

### Step 1: Set Up Your SageMaker Environment

**Option A: SageMaker Notebook Instance**
1. Go to AWS Console → SageMaker → Notebook instances
2. Click "Create notebook instance"
3. Choose `ml.t3.medium` (sufficient for this exercise)
4. Select or create an IAM role with SageMaker permissions
5. Wait for status to become "InService"
6. Click "Open JupyterLab"

**Option B: SageMaker Studio**
1. Go to AWS Console → SageMaker → Studio
2. Open your Studio domain
3. Create a new notebook with Python 3 kernel

**Option C: Run Locally**
1. Configure AWS credentials with a profile that has SageMaker permissions
2. Update the `boto3.setup_default_session()` call with your profile name
3. Set the `role` variable to your SageMaker execution role ARN

### Step 2: Upload the Exercise Files

1. Download `deepar_sagemaker_exercise.ipynb` from this package
2. In JupyterLab/Studio, click the upload button
3. Upload the notebook file

### Step 3: Run the Notebook

Open `deepar_sagemaker_exercise.ipynb` and execute cells in order:

| Section | What It Does | Estimated Time |
|---------|--------------|----------------|
| Step 1 | Setup & imports | 1 min |
| Step 2 | Generate synthetic data | 30 sec |
| Step 3 | Visualize data | 30 sec |
| Step 4 | Save & upload to S3 | 1 min |
| Step 5 | Configure & train DeepAR model | **5-10 min** |
| Step 6 | Run batch transform | **3-5 min** |
| Step 7-9 | Download, evaluate & visualize predictions | 1 min |
| Step 10 | Deploy endpoint (optional) | **5-7 min** |
| Step 11-13 | Real-time predictions & evaluation | 1 min |
| Step 14 | **Clean up (IMPORTANT!)** | 1 min |

**Total time: ~20-30 minutes**

### Step 4: Clean Up (Critical!)

**SageMaker endpoints cost money while running!**

Always run the cleanup cell at the end:
```python
predictor.delete_endpoint()
```

---

## Understanding the Data Format

DeepAR requires **JSON Lines** format. Each line is one time series:

```json
{
  "start": "2022-01-01 00:00:00",
  "target": [100, 120, 95, 140, ...],
  "cat": [0],
  "dynamic_feat": [[0, 0, 1, 0, ...]]
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `start` | Yes | Start timestamp of the series |
| `target` | Yes | Array of values (the actual time series) |
| `cat` | No | Categorical features (integers) |
| `dynamic_feat` | No | Time-varying features (see important note below) |

### Important: dynamic_feat Length Requirements

The length requirement for `dynamic_feat` differs between training and inference:

| Context | target length | dynamic_feat length |
|---------|---------------|---------------------|
| **Training** | N days | N days (same as target) |
| **Inference** | N days (history) | N + prediction_length days |

For inference, `dynamic_feat` must extend `prediction_length` days beyond `target` because DeepAR needs to know the feature values for the forecast period. For example, if you have 700 days of history and want to forecast 30 days ahead, your inference data needs:
- `target`: 700 values
- `dynamic_feat`: 730 values (700 history + 30 forecast period)

This is why the notebook creates separate `train_data`, `test_data`, and `inference_data` datasets.

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prediction_length` | - | How many time steps to forecast (required) |
| `context_length` | - | History window for making predictions |
| `time_freq` | - | Data frequency: "D" (daily), "H" (hourly), etc. |
| `epochs` | 100 | Training iterations |
| `num_cells` | 40 | Number of neurons in each LSTM layer |
| `num_layers` | 2 | Number of stacked LSTM layers |
| `dropout_rate` | 0.0 | Regularization (fraction of neurons dropped during training) |
| `cardinality` | auto | Number of unique values per categorical feature |
| `likelihood` | student-t | Distribution: "gaussian", "negative-binomial", etc. |

**Rules of thumb:**
- Set `context_length` ≥ 2× `prediction_length`
- Use `negative-binomial` for count data (units sold, visitors)
- Use `gaussian` for continuous data that can be negative
- Start with `dropout_rate` of 0.05-0.1 if overfitting

---

## Cost Estimates

| Resource | Instance | Cost (approx) |
|----------|----------|---------------|
| Training | ml.m5.large | ~$0.15 for 10 min |
| Batch Transform | ml.m5.large | ~$0.10 for 5 min |
| Endpoint | ml.m5.large | ~$0.12/hour |
| Notebook | ml.t3.medium | ~$0.05/hour |

**Important:** Delete your endpoint immediately after the exercise!

---

## Troubleshooting

### "ResourceLimitExceeded" Error
- You've hit your account limits
- Request a limit increase in Service Quotas

### Training Job Fails
- Check CloudWatch logs for the training job
- Common issues: S3 permissions, data format errors

### Batch Transform Error: "dynamic_feat needs to be provided in the full prediction range"
- Your `dynamic_feat` doesn't extend beyond `target` for the forecast period
- For inference: `dynamic_feat` must have length = `target` length + `prediction_length`
- Use the separate `inference_data` dataset, not `test_data`

### Endpoint Deployment Fails
- Ensure your IAM role has SageMaker permissions
- Check if you have endpoint quota available

### Poor Forecast Quality
- Try increasing `context_length`
- Add more training data
- Experiment with `num_cells` and `num_layers`
- Adjust `dropout_rate` if overfitting

---

## Next Steps After This Exercise

1. **Hyperparameter Tuning**: Use SageMaker's automatic tuning
2. **Add More Features**: Weather, holidays, price data
3. **Real Data**: Apply to your own time series
4. **Compare Algorithms**: Try Prophet, ARIMA, or other SageMaker algorithms

---

## Resources

- [DeepAR Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html)
- [DeepAR Hyperparameters](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar_hyperparameters.html)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [Original DeepAR Paper](https://arxiv.org/abs/1704.04110)
