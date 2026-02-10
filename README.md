# bb-tech-challenge
## EDA Conclusions

### Data Quality
- 30,781 valid records after dropping 20 rows with correlated null identifiers
- No duplicate customer IDs detected after that
- All numeric features have valid ranges, no significant outliers or data entry error evidence.

### Target
- Moderate imbalance: 20.5% churn / 79.5% retained (4:1 ratio)
- Handled via class weights, not resampling. Appropriate for this level of imbalance
- Primary metric: PR-AUC. Appropriate for this level of imbalance

### Features
- Strongest raw predictors identify base on Cohen's d analysis over 0.8
- Weak/noise features: identify base on Cohen's d analysis under 0.5
- Zero-inflation detected in: [esent, eopenrate, eclickrate, ordfreq]

### Engineered Features
- `order_recency_ratio` is the strongest engineered feature
- Potential leakage concern: 
  - If retained=0 was labeled based on 'no order in X days', then order_recency_ratio is essentially restating the label definitionthat's leakage. 
  - If retained=0 was labeled by explicit account cancellation, then order_recency_ratio is a legitimate leading indicator. I'll choose last option in this challenge.
- `tenure_days` and `days_since_first_order` are correlated but retained because tree models handle this naturally.

### Leakage Assessment
- `lastorder`derived features partially encode the target — documented and flagged for reviewer discussion
- I will retain these features because they represent information
  available at prediction time in production

### Splitting Strategy
- Stratified random split (70/15/15) — appropriate because features are
  aggregated snapshots, not time-series events
- Temporal split not required: no future-information leakage through features
