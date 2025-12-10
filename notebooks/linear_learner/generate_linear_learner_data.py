"""
Linear Learner Synthetic Data Generator
========================================
Generates synthetic data for training Amazon SageMaker's Linear Learner algorithm.

This script creates datasets for:
1. Binary Classification - Customer churn prediction
2. Regression - House price prediction

Linear Learner supports both classification and regression tasks, so we generate
data suitable for demonstrating both use cases.
"""

import numpy as np
import pandas as pd
import os
import argparse
from io import BytesIO


def generate_classification_data(
    num_samples: int = 5000,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Generate synthetic customer churn data for binary classification.

    Features simulate customer behavior patterns:
    - Account tenure (months)
    - Monthly charges
    - Total charges
    - Number of support tickets
    - Contract type (encoded)
    - Payment method (encoded)
    - Has online security
    - Has tech support
    - Internet service type (encoded)
    - Number of products/services

    Returns:
        tuple: (features, labels, feature_names)
    """
    np.random.seed(seed)

    feature_names = [
        'tenure_months',
        'monthly_charges',
        'total_charges',
        'support_tickets',
        'contract_type',
        'payment_method',
        'online_security',
        'tech_support',
        'internet_service',
        'num_products'
    ]

    # Generate features
    tenure = np.random.exponential(scale=24, size=num_samples).clip(1, 72)
    monthly_charges = np.random.normal(65, 30, num_samples).clip(20, 150)
    total_charges = tenure * monthly_charges * np.random.uniform(0.8, 1.0, num_samples)
    support_tickets = np.random.poisson(2, num_samples)
    contract_type = np.random.choice([0, 1, 2], num_samples, p=[0.5, 0.3, 0.2])  # Monthly, 1yr, 2yr
    payment_method = np.random.choice([0, 1, 2, 3], num_samples)  # Different payment types
    online_security = np.random.choice([0, 1], num_samples, p=[0.6, 0.4])
    tech_support = np.random.choice([0, 1], num_samples, p=[0.5, 0.5])
    internet_service = np.random.choice([0, 1, 2], num_samples, p=[0.2, 0.4, 0.4])  # None, DSL, Fiber
    num_products = np.random.choice([1, 2, 3, 4, 5], num_samples, p=[0.3, 0.3, 0.2, 0.15, 0.05])

    # Stack features
    X = np.column_stack([
        tenure,
        monthly_charges,
        total_charges,
        support_tickets,
        contract_type,
        payment_method,
        online_security,
        tech_support,
        internet_service,
        num_products
    ])

    # Generate labels based on realistic churn patterns
    # High churn probability factors: low tenure, high charges, many support tickets, monthly contract
    churn_score = (
        -0.05 * tenure +                    # Longer tenure = less churn
        0.02 * monthly_charges +            # Higher charges = more churn
        0.1 * support_tickets +             # More tickets = more churn
        -0.5 * contract_type +              # Longer contracts = less churn
        -0.3 * online_security +            # Security = less churn
        -0.3 * tech_support +               # Support = less churn
        0.2 * (internet_service == 2) +     # Fiber = slightly more churn
        -0.1 * num_products +               # More products = less churn
        np.random.normal(0, 0.5, num_samples)  # Random noise
    )

    # Convert to probability and then to binary labels
    churn_prob = 1 / (1 + np.exp(-churn_score))
    y = (np.random.random(num_samples) < churn_prob).astype(int)

    return X.astype(np.float32), y.astype(np.float32), feature_names


def generate_regression_data(
    num_samples: int = 5000,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Generate synthetic house price data for regression.

    Features simulate property characteristics:
    - Square footage
    - Number of bedrooms
    - Number of bathrooms
    - Lot size (acres)
    - Year built
    - Garage capacity
    - Has pool
    - Distance to city center
    - School rating (1-10)
    - Crime rate (per 1000)

    Returns:
        tuple: (features, labels, feature_names)
    """
    np.random.seed(seed)

    feature_names = [
        'sqft',
        'bedrooms',
        'bathrooms',
        'lot_size_acres',
        'year_built',
        'garage_capacity',
        'has_pool',
        'distance_to_city',
        'school_rating',
        'crime_rate'
    ]

    # Generate features
    sqft = np.random.normal(2000, 800, num_samples).clip(500, 6000)
    bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], num_samples, p=[0.05, 0.15, 0.35, 0.30, 0.12, 0.03])
    bathrooms = np.minimum(bedrooms, np.random.choice([1, 2, 3, 4], num_samples, p=[0.2, 0.4, 0.3, 0.1]))
    lot_size = np.random.exponential(0.3, num_samples).clip(0.1, 5.0)
    year_built = np.random.normal(1990, 20, num_samples).clip(1920, 2024).astype(int)
    garage_capacity = np.random.choice([0, 1, 2, 3], num_samples, p=[0.1, 0.25, 0.50, 0.15])
    has_pool = np.random.choice([0, 1], num_samples, p=[0.75, 0.25])
    distance_to_city = np.random.exponential(10, num_samples).clip(1, 50)
    school_rating = np.random.uniform(3, 10, num_samples)
    crime_rate = np.random.exponential(5, num_samples).clip(0.5, 30)

    # Stack features
    X = np.column_stack([
        sqft,
        bedrooms,
        bathrooms,
        lot_size,
        year_built,
        garage_capacity,
        has_pool,
        distance_to_city,
        school_rating,
        crime_rate
    ])

    # Generate prices based on realistic relationships
    # Base price around $200k with various adjustments
    price = (
        50000 +                             # Base
        150 * sqft +                        # $150 per sqft
        15000 * bedrooms +                  # $15k per bedroom
        20000 * bathrooms +                 # $20k per bathroom
        30000 * lot_size +                  # $30k per acre
        1000 * (year_built - 1950) +        # Newer = more expensive
        15000 * garage_capacity +           # $15k per garage spot
        40000 * has_pool +                  # $40k for pool
        -2000 * distance_to_city +          # Farther = cheaper
        10000 * school_rating +             # Good schools = expensive
        -3000 * crime_rate +                # High crime = cheaper
        np.random.normal(0, 30000, num_samples)  # Market noise
    )

    # Ensure positive prices
    y = np.maximum(price, 50000).astype(np.float32)

    return X.astype(np.float32), y, feature_names


def to_recordio_protobuf(X: np.ndarray, y: np.ndarray) -> bytes:
    """
    Convert data to RecordIO-Protobuf format for SageMaker.

    Note: This is a simplified version. For production, use sagemaker.amazon.common.
    For this exercise, we'll use CSV format which Linear Learner also supports.
    """
    # Linear Learner supports CSV format, which is simpler to work with
    # Format: label, feature1, feature2, ...
    pass


def save_csv_format(X: np.ndarray, y: np.ndarray, output_path: str, include_header: bool = False):
    """
    Save data in CSV format (label first, then features).
    SageMaker Linear Learner expects: label, feature1, feature2, ...
    """
    # Combine label and features
    data = np.column_stack([y.reshape(-1, 1), X])

    # Save without header for SageMaker
    np.savetxt(output_path, data, delimiter=',', fmt='%.6f')


def generate_dataset(
    task_type: str = "classification",
    num_samples: int = 5000,
    test_ratio: float = 0.2,
    seed: int = 42,
    output_dir: str = "./data"
) -> dict:
    """
    Generate complete dataset for Linear Learner.

    Args:
        task_type: "classification" or "regression"
        num_samples: Total number of samples to generate
        test_ratio: Fraction of data for testing
        seed: Random seed for reproducibility
        output_dir: Directory to save output files

    Returns:
        dict: Metadata about the generated dataset
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate appropriate data
    if task_type == "classification":
        X, y, feature_names = generate_classification_data(num_samples, seed)
        task_description = "Customer Churn Prediction (Binary Classification)"
    else:
        X, y, feature_names = generate_regression_data(num_samples, seed)
        task_description = "House Price Prediction (Regression)"

    # Split into train and test
    np.random.seed(seed)
    indices = np.random.permutation(num_samples)
    test_size = int(num_samples * test_ratio)
    train_indices = indices[test_size:]
    test_indices = indices[:test_size]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    # Save in CSV format (SageMaker Linear Learner format: label, features...)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    print(f"Generating {task_description} data...")
    print(f"Saving training data to {train_path}...")
    save_csv_format(X_train, y_train, train_path)

    print(f"Saving test data to {test_path}...")
    save_csv_format(X_test, y_test, test_path)

    # Save a human-readable version with headers
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df.insert(0, 'label', y_train)
    train_df.to_csv(os.path.join(output_dir, "train_with_headers.csv"), index=False)

    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df.insert(0, 'label', y_test)
    test_df.to_csv(os.path.join(output_dir, "test_with_headers.csv"), index=False)

    # Save metadata
    metadata = {
        "task_type": task_type,
        "task_description": task_description,
        "num_samples": num_samples,
        "num_train": len(y_train),
        "num_test": len(y_test),
        "num_features": X.shape[1],
        "feature_names": feature_names,
        "files": {
            "train": "train.csv",
            "test": "test.csv",
            "train_readable": "train_with_headers.csv",
            "test_readable": "test_with_headers.csv"
        }
    }

    if task_type == "classification":
        metadata["class_distribution"] = {
            "train": {
                "class_0": int((y_train == 0).sum()),
                "class_1": int((y_train == 1).sum())
            },
            "test": {
                "class_0": int((y_test == 0).sum()),
                "class_1": int((y_test == 1).sum())
            }
        }
    else:
        metadata["label_statistics"] = {
            "train": {
                "mean": float(y_train.mean()),
                "std": float(y_train.std()),
                "min": float(y_train.min()),
                "max": float(y_train.max())
            },
            "test": {
                "mean": float(y_test.mean()),
                "std": float(y_test.std()),
                "min": float(y_test.min()),
                "max": float(y_test.max())
            }
        }

    import json
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*50)
    print("Dataset Generation Complete!")
    print("="*50)
    print(f"  Task: {task_description}")
    print(f"  Total Samples: {num_samples}")
    print(f"  Training Samples: {len(y_train)}")
    print(f"  Test Samples: {len(y_test)}")
    print(f"  Number of Features: {X.shape[1]}")

    if task_type == "classification":
        train_churn_rate = y_train.mean() * 100
        print(f"  Training Churn Rate: {train_churn_rate:.1f}%")
    else:
        print(f"  Training Mean Price: ${y_train.mean():,.0f}")
        print(f"  Training Price Range: ${y_train.min():,.0f} - ${y_train.max():,.0f}")

    print(f"\nFiles created in '{output_dir}/':")
    print(f"  - train.csv (SageMaker format)")
    print(f"  - test.csv (SageMaker format)")
    print(f"  - train_with_headers.csv (human-readable)")
    print(f"  - test_with_headers.csv (human-readable)")
    print(f"  - metadata.json (dataset info)")

    return metadata


def visualize_data(data_dir: str = "./data", task_type: str = "classification"):
    """
    Create visualizations of the generated data.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping visualization.")
        return

    # Load data with headers
    train_df = pd.read_csv(os.path.join(data_dir, "train_with_headers.csv"))

    if task_type == "classification":
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        features_to_plot = ['tenure_months', 'monthly_charges', 'support_tickets',
                          'contract_type', 'num_products', 'total_charges']

        for idx, feature in enumerate(features_to_plot):
            ax = axes[idx]

            # Plot distribution by class
            for label, color, name in [(0, 'blue', 'No Churn'), (1, 'red', 'Churn')]:
                subset = train_df[train_df['label'] == label][feature]
                ax.hist(subset, bins=30, alpha=0.5, color=color, label=name)

            ax.set_xlabel(feature)
            ax.set_ylabel('Count')
            ax.legend()
            ax.set_title(f'{feature} Distribution by Churn Status')

        plt.suptitle('Customer Churn - Feature Distributions', fontsize=14)

    else:  # regression
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        features_to_plot = ['sqft', 'bedrooms', 'year_built',
                          'distance_to_city', 'school_rating', 'lot_size_acres']

        for idx, feature in enumerate(features_to_plot):
            ax = axes[idx]
            ax.scatter(train_df[feature], train_df['label'], alpha=0.3, s=5)
            ax.set_xlabel(feature)
            ax.set_ylabel('Price ($)')
            ax.set_title(f'{feature} vs Price')

        plt.suptitle('House Price - Feature Relationships', fontsize=14)

    plt.tight_layout()
    plot_path = os.path.join(data_dir, "data_visualization.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nVisualization saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data for Linear Learner")
    parser.add_argument("--task", type=str, default="classification",
                        choices=["classification", "regression"],
                        help="Task type (default: classification)")
    parser.add_argument("--num-samples", type=int, default=5000,
                        help="Number of samples to generate (default: 5000)")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Fraction of data for testing (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default="./data",
                        help="Output directory (default: ./data)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization of the data")

    args = parser.parse_args()

    metadata = generate_dataset(
        task_type=args.task,
        num_samples=args.num_samples,
        test_ratio=args.test_ratio,
        seed=args.seed,
        output_dir=args.output_dir
    )

    if args.visualize:
        visualize_data(args.output_dir, args.task)
