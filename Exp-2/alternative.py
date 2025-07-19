import math
# Helper function to compute entropy
def entropy(pos, neg):
    total = pos + neg
    if total == 0:
        return 0
    p_pos = pos / total if pos != 0 else 0
    p_neg = neg / total if neg != 0 else 0
    entropy_val = 0
    if p_pos > 0:
        entropy_val -= p_pos * math.log2(p_pos)
    if p_neg > 0:
        entropy_val -= p_neg * math.log2(p_neg)
    return round(entropy_val, 4)
# Parent set: 14 positive, 16 negative
parent_entropy = entropy(14, 16)

# First child: 13 positive, 4 negative
entropy_child1 = entropy(13, 4)
# Second child: 1 positive, 12 negative
entropy_child2 = entropy(1, 12)
# Weighted average entropy
total = 30
weighted_entropy = ((17 / total) * entropy_child1) + ((13 / total) * entropy_child2)
# Information Gain
info_gain = parent_entropy - weighted_entropy

# Output
print(f"Parent Entropy: {parent_entropy}")
print(f"Child Entropy 1 (13+, 4-): {entropy_child1}")
print(f"Child Entropy 2 (1+, 12-): {entropy_child2}")
print(f"Weighted Child Entropy: {round(weighted_entropy, 4)}")
print(f"Information Gain: {round(info_gain, 4)}")
