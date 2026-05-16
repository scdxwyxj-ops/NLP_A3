# Top3 Train-KFold Fusion

Goal: select final evidence top3 ranking weights from train folds, then confirm once on dev.

## Selected Weights

| component | weight |
|---|---:|
| ce_score | 0.25 |
| embedding_score | 0.50 |
| source_rank | 0.00 |
| ce_rank | 0.25 |
| embedding_rank | 0.00 |

## Best Metrics

- CV evidence F@3: `0.175504`
- CV macro recall@3: `0.196029`
- Dev evidence F@3: `0.204700`
- Dev macro recall@3: `0.238745`
- Dev macro recall@10: `0.395887`
- Dev macro recall@64: `0.581602`

## Strict Notes

- Weight selection uses train-claim folds only.
- Dev labels are used only for confirmation metrics.
- The grid was designed after prior diagnostic work, so this should be reported as train-selected recovery, not as a blind preregistered leaderboard result.
- No forbidden path markers found.

## Top Configurations

| ce_score | embedding_score | source_rank | ce_rank | embedding_rank | cv F@3 | cv R@3 | dev F@3 | dev R@3 | dev R@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.50 | 0.00 | 0.25 | 0.00 | 0.1755 | 0.1960 | 0.2047 | 0.2387 | 0.3959 |
| 0.25 | 0.50 | 0.00 | 0.00 | 0.25 | 0.1749 | 0.1954 | 0.2025 | 0.2366 | 0.3943 |
| 0.25 | 0.50 | 0.25 | 0.00 | 0.00 | 0.1745 | 0.1948 | 0.2047 | 0.2387 | 0.3972 |
| 0.00 | 0.00 | 0.00 | 0.75 | 0.25 | 0.1744 | 0.1929 | 0.2034 | 0.2344 | 0.3983 |
| 0.25 | 0.25 | 0.50 | 0.00 | 0.00 | 0.1733 | 0.1921 | 0.2043 | 0.2377 | 0.3876 |
| 0.25 | 0.75 | 0.00 | 0.00 | 0.00 | 0.1723 | 0.1922 | 0.2001 | 0.2339 | 0.3978 |
| 0.25 | 0.25 | 0.25 | 0.00 | 0.25 | 0.1719 | 0.1913 | 0.2043 | 0.2377 | 0.3876 |
| 0.25 | 0.25 | 0.00 | 0.25 | 0.25 | 0.1718 | 0.1912 | 0.2069 | 0.2409 | 0.3892 |
| 0.25 | 0.25 | 0.00 | 0.00 | 0.50 | 0.1718 | 0.1912 | 0.2043 | 0.2377 | 0.3879 |
| 0.25 | 0.25 | 0.25 | 0.25 | 0.00 | 0.1718 | 0.1907 | 0.2043 | 0.2377 | 0.3876 |
