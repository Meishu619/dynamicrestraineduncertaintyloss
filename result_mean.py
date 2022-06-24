from scipy.stats import hmean

# ValEmoCCC: 0.601
# ValCountryUAR: 0.569
# # ValAgeMAE: 3.754
# 3.926
# 0.588
# 0.645



val_hmean_score = hmean([0.645, 0.588, 1/3.926])

print(val_hmean_score)