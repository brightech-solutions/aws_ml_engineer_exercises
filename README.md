# AWS SageMaker Algorithm Exercises

A collection of hands-on exercises covering AWS SageMaker's built-in machine learning algorithms. Each notebook walks through data preparation, model training, deployment, and inference using SageMaker.

I have run quite a few of the [algorithms](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html) available in Sagemaker, but definitely not all of them. I wanted to have a set of exercises for each of the major algorithms, especially for those that I have never used before. I will add to the collection of notebooks as I go through each algorightm. I hope that you will find this collection useful. If you see something that could be improved, please feel free to submit a PR.


## Available Exercises

| Algorithm | Description | Notebook | URL |
|-----------|-------------|----------|-----|
| **XGBoost** | Gradient boosting for classification and regression | `notebooks/xgboost/` | [AWS URL](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html)|
| **LightGBM** | Fast gradient boosting with leaf-wise tree growth | `notebooks/light_gbm/` | [AWS URL](https://docs.aws.amazon.com/sagemaker/latest/dg/lightgbm.html)|
| **Linear Learner** | Linear models for classification and regression | `notebooks/linear_learner/` | [AWS URL](https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner.html)|
| **DeepAR** | Probabilistic time series forecasting with RNNs | `notebooks/deep_ar/` | [AWS URL](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html)|
| **Seq2Seq** | Sequence-to-sequence for translation and summarization | `notebooks/04_seq_to_seq/` | [AWS URL](https://docs.aws.amazon.com/sagemaker/latest/dg/seq-2-seq.html)|

## Environment Setup

### Prerequisites

- Python 3.11
- AWS account with SageMaker access
- AWS CLI configured with a named profile

### 1. Install UV

[UV](https://docs.astral.sh/uv/) is a fast Python package manager.

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv
```

### 2. Create the Python Environment

```bash
uv sync --python 3.11
```

This creates a `.venv` directory with Python 3.11 and all dependencies.

### 3. Configure AWS Credentials

These notebooks use AWS named profiles for authentication. First, ensure you have the AWS CLI installed and configured:

```bash
# Configure a named profile (if you haven't already)
aws configure --profile your-profile-name
```

Then create your local environment file:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
AWS_PROFILE=your-profile-name
AWS_REGION=us-west-2
SAGEMAKER_ROLE_ARN=arn:aws:iam::123456789012:role/YourSageMakerRole
```

The notebooks will automatically load these environment variables.

### S3 Bucket

The notebooks use SageMaker's default bucket for storing training data and model artifacts. When you call `sagemaker_session.default_bucket()`, SageMaker automatically creates a bucket named:

```
sagemaker-{region}-{account_id}
```

For example: `sagemaker-us-west-2-123456789012`

This bucket is created in your AWS account the first time you run a notebook. No manual setup required.

### 4. Run Notebooks

**VS Code (recommended):** Open any `.ipynb` file directly. VS Code will prompt you to select the Python interpreter - choose the one from `.venv`.

**Jupyter Lab:** If you prefer the browser interface:
```bash
source .venv/bin/activate
jupyter lab
```

## Tracking SageMaker Costs

Running SageMaker training jobs and endpoints can get expensive. Use the included cost tracking script to monitor your spend:

```bash
# View last 30 days of SageMaker costs
./notebooks/sagemaker_costs.sh

# View last 7 days
./notebooks/sagemaker_costs.sh 7
```

The script shows daily spend with a visual breakdown by usage type (training, endpoints, etc.). Be sure to have your AWS_PROFILE variable properly set.

## Project Structure

```
notebooks/
├── xgboost/           # XGBoost exercises
├── light_gbm/         # LightGBM exercises
├── linear_learner/    # Linear Learner exercises
├── deep_ar/           # DeepAR time series exercises
└── 04_seq_to_seq/     # Seq2Seq translation exercises
common/                # Shared utilities
generators/            # Synthetic data generation
```

## Contributing

Suggestions and improvements are welcome!
