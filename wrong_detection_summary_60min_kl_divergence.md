
# Wrong Detection Analysis Report

## Test Configuration
- **Time Interval**: 60 minutes
- **Similarity Method**: kl_divergence
- **Analysis Date**: 2025-07-18 12:04:05

## Performance Metrics
- **Precision**: 0.038
- **Recall**: 0.458
- **F1-Score**: 0.070
- **Accuracy**: 0.419
- **False Positive Rate**: 0.583
- **False Negative Rate**: 0.542

## Error Analysis

### False Positives (1664 cases)
- **Average Activity Score**: 2.362
- **Average Activity Level**: 0.397
- **Average File Accesses**: 2.579
- **Average Network Connections**: 1.483
- **Average Data Transfer**: 4.342

### False Negatives (78 cases)
- **Average Activity Score**: 136.854
- **Average Activity Level**: 0.765
- **Average File Accesses**: 19.154
- **Average Network Connections**: 21.423
- **Average Data Transfer**: 96.957

### True Positives (66 cases)
- **Average Activity Score**: 43.525
- **Average Activity Level**: 0.695
- **Average File Accesses**: 14.015
- **Average Network Connections**: 14.318
- **Average Data Transfer**: 31.423

## Key Insights

### False Positive Patterns
- High activity scores but normal behavior patterns
- Elevated file access or network activity during normal hours
- Sudden spikes in activity that are legitimate

### False Negative Patterns
- Low activity scores but malicious behavior
- Subtle attack patterns that don't trigger high activity scores
- Attacks during normal working hours that blend in

### True Positive Patterns
- High activity scores with malicious behavior
- Clear deviation from normal patterns
- Multiple indicators of suspicious activity

## Recommendations

### For Reducing False Positives
1. **Adjust threshold**: Increase threshold for more conservative detection
2. **Feature engineering**: Add more context-aware features
3. **Time-based filtering**: Consider work hours and patterns
4. **User-specific baselines**: Individual user behavior modeling

### For Reducing False Negatives
1. **Lower threshold**: Decrease threshold for more sensitive detection
2. **Additional features**: Include more subtle indicators
3. **Sequence analysis**: Longer sequence patterns
4. **Multi-modal detection**: Combine multiple detection methods

### For Overall Improvement
1. **Ensemble methods**: Combine multiple similarity approaches
2. **Deep learning**: LSTM/GRU for sequence modeling
3. **Real-time adaptation**: Dynamic threshold adjustment
4. **Domain expertise**: Incorporate security expert knowledge
