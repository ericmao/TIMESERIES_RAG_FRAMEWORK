
# Wrong Detection Analysis Report

## Test Configuration
- **Time Interval**: 120 minutes
- **Similarity Method**: kl_divergence
- **Analysis Date**: 2025-07-18 12:04:33

## Performance Metrics
- **Precision**: 0.072
- **Recall**: 0.368
- **F1-Score**: 0.121
- **Accuracy**: 0.485
- **False Positive Rate**: 0.502
- **False Negative Rate**: 0.632

## Error Analysis

### False Positives (681 cases)
- **Average Activity Score**: 3.256
- **Average Activity Level**: 0.378
- **Average File Accesses**: 4.815
- **Average Network Connections**: 2.831
- **Average Data Transfer**: 7.830

### False Negatives (91 cases)
- **Average Activity Score**: 68.984
- **Average Activity Level**: 0.573
- **Average File Accesses**: 21.187
- **Average Network Connections**: 22.341
- **Average Data Transfer**: 88.431

### True Positives (53 cases)
- **Average Activity Score**: 32.437
- **Average Activity Level**: 0.543
- **Average File Accesses**: 16.189
- **Average Network Connections**: 14.623
- **Average Data Transfer**: 40.372

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
