#!/bin/bash

# SageMaker Cost Report - Daily spend breakdown (before credits)
# Usage: ./sagemaker_costs.sh [days]
# Example: ./sagemaker_costs.sh 7    # Last 7 days
#          ./sagemaker_costs.sh      # Default: last 30 days

DAYS=${1:-30}
END_DATE=$(date +%Y-%m-%d)
START_DATE=$(date -v-${DAYS}d +%Y-%m-%d 2>/dev/null || date -d "$DAYS days ago" +%Y-%m-%d)

echo "=================================================="
echo "   SageMaker Cost Report - Last $DAYS Days"
echo "   (Gross spend before credits applied)"
echo "=================================================="
echo "Period: $START_DATE to $END_DATE"
echo ""

# Get daily costs using NetUnblendedCost which excludes discounts/credits impact
# We also exclude credit line items via the filter
aws ce get-cost-and-usage \
    --time-period Start=$START_DATE,End=$END_DATE \
    --granularity DAILY \
    --metrics NetUnblendedCost \
    --filter '{
        "And": [
            {
                "Dimensions": {
                    "Key": "SERVICE",
                    "Values": ["Amazon SageMaker"]
                }
            },
            {
                "Not": {
                    "Dimensions": {
                        "Key": "RECORD_TYPE",
                        "Values": ["Credit", "Refund"]
                    }
                }
            }
        ]
    }' \
    --profile brightech-secondary \
    --region us-east-1 \
    --output json 2>/dev/null | python3 -c "
import sys
import json

data = json.load(sys.stdin)
results = data.get('ResultsByTime', [])

if not results:
    print('No SageMaker costs found for this period.')
    sys.exit(0)

# Print header
print(f\"{'Date':<12} {'Cost':>12} {'Bar'}\")
print('-' * 50)

total = 0
costs = []

for r in results:
    date = r['TimePeriod']['Start']
    amount = float(r['Total']['NetUnblendedCost']['Amount'])
    # Take absolute value to show actual spend
    amount = abs(amount)
    costs.append((date, amount))
    total += amount

# Find max for scaling bar chart
max_cost = max(c[1] for c in costs) if costs else 1
max_cost = max(max_cost, 0.01)  # Avoid division by zero

for date, amount in costs:
    bar_len = int((amount / max_cost) * 25) if max_cost > 0 else 0
    bar = '█' * bar_len
    if amount > 0.001:
        print(f\"{date:<12} \${amount:>10.2f}  {bar}\")
    else:
        print(f\"{date:<12} \${amount:>10.2f}\")

print('-' * 50)
print(f\"{'TOTAL':<12} \${total:>10.2f}\")
print()

# Daily average
avg = total / len(costs) if costs else 0
print(f\"Daily Average: \${avg:.2f}\")
"

echo ""
echo "=================================================="

# Also show breakdown by usage type if there are costs
echo ""
echo "Breakdown by Usage Type:"
echo ""

aws ce get-cost-and-usage \
    --time-period Start=$START_DATE,End=$END_DATE \
    --granularity MONTHLY \
    --metrics NetUnblendedCost \
    --filter '{
        "And": [
            {
                "Dimensions": {
                    "Key": "SERVICE",
                    "Values": ["Amazon SageMaker"]
                }
            },
            {
                "Not": {
                    "Dimensions": {
                        "Key": "RECORD_TYPE",
                        "Values": ["Credit", "Refund"]
                    }
                }
            }
        ]
    }' \
    --group-by Type=DIMENSION,Key=USAGE_TYPE \
    --profile brightech-secondary \
    --region us-east-1 \
    --output json 2>/dev/null | python3 -c "
import sys
import json

data = json.load(sys.stdin)
results = data.get('ResultsByTime', [])

usage_totals = {}
for r in results:
    for group in r.get('Groups', []):
        usage_type = group['Keys'][0]
        amount = abs(float(group['Metrics']['NetUnblendedCost']['Amount']))
        usage_totals[usage_type] = usage_totals.get(usage_type, 0) + amount

if not usage_totals:
    print('No usage breakdown available.')
    sys.exit(0)

# Sort by cost descending
sorted_usage = sorted(usage_totals.items(), key=lambda x: x[1], reverse=True)

print(f\"{'Usage Type':<45} {'Cost':>12}\")
print('-' * 60)

for usage_type, amount in sorted_usage:
    if amount > 0.001:  # Only show non-trivial costs
        # Shorten the usage type for display
        short_type = usage_type.replace('USW2-', '').replace('USE1-', '').replace('APS1-', '')
        if len(short_type) > 43:
            short_type = short_type[:40] + '...'
        print(f\"{short_type:<45} \${amount:>10.2f}\")
"

echo ""
