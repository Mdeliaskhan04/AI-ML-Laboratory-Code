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

# ----- Input from user -----
print("Enter Parent Set:")
p_pos = int(input("Positive examples: "))
p_neg = int(input("Negative examples: "))

print("\nEnter First Child Set:")
c1_pos = int(input("Child 1 - Positive examples: "))
c1_neg = int(input("Child 1 - Negative examples: "))

print("\nEnter Second Child Set:")
c2_pos = int(input("Child 2 - Positive examples: "))
c2_neg = int(input("Child 2 - Negative examples: "))

# ----- Calculations -----
parent_entropy = entropy(p_pos, p_neg)
entropy_child1 = entropy(c1_pos, c1_neg)
entropy_child2 = entropy(c2_pos, c2_neg)

# Weighted average entropy
total = c1_pos + c1_neg + c2_pos + c2_neg
child1_total = c1_pos + c1_neg
child2_total = c2_pos + c2_neg

weighted_entropy = ((child1_total / total) * entropy_child1) + ((child2_total / total) * entropy_child2)
weighted_entropy = round(weighted_entropy, 4)

# Information Gain
info_gain = round(parent_entropy - weighted_entropy, 4)

# ----- Output -----
print("\n===== Results =====")
print(f"Parent Entropy: {parent_entropy}")
print(f"Child Entropy 1 ({c1_pos}+, {c1_neg}-): {entropy_child1}")
print(f"Child Entropy 2 ({c2_pos}+, {c2_neg}-): {entropy_child2}")
print(f"Weighted Child Entropy: {weighted_entropy}")
print(f"Information Gain: {info_gain}")
