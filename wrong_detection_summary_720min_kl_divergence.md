
# Wrong Detection Analysis Report

## Test Configuration
- **Time Interval**: 720 minutes
- **Similarity Method**: kl_divergence
- **Analysis Date**: 2025-07-18 12:05:09

## Performance Metrics
- **Precision**: 0.398
- **Recall**: 0.285
- **F1-Score**: 0.332
- **Accuracy**: 0.340
- **False Positive Rate**: 0.585
- **False Negative Rate**: 0.715

## Error Analysis

### False Positives (62 cases)
- **Average Activity Score**: 14.650
- **Average Activity Level**: 0.384
- **Average File Accesses**: 30.532
- **Average Network Connections**: 16.581
- **Average Data Transfer**: 47.446

### False Negatives (103 cases)
- **Average Activity Score**: 35.633
- **Average Activity Level**: 0.416
- **Average File Accesses**: 43.592
- **Average Network Connections**: 34.553
- **Average Data Transfer**: 118.467

### True Positives (41 cases)
- **Average Activity Score**: 29.730
- **Average Activity Level**: 0.416
- **Average File Accesses**: 43.780
- **Average Network Connections**: 30.902
- **Average Data Transfer**: 95.096

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
