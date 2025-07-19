import numpy as np

# Input number of cities
n = int(input("Enter number of cities: "))

# Input distance matrix
print("\nEnter the distance matrix row by row (space-separated):")
dist_matrix = []
for i in range(n):
    row = list(map(int, input(f"Row {i+1}: ").split()))
    dist_matrix.append(row)

# Initialize
visited = [False] * n
path = []
total_cost = 0

# Starting from city 0
current_city = 0
visited[current_city] = True
path.append(current_city)

for _ in range(n - 1):
    nearest = None
    min_dist = float('inf')
    for city in range(n):
        if not visited[city] and dist_matrix[current_city][city] < min_dist:
            nearest = city
            min_dist = dist_matrix[current_city][city]
    visited[nearest] = True
    path.append(nearest)
    total_cost += min_dist
    current_city = nearest

# Return to starting city
total_cost += dist_matrix[current_city][path[0]]
path.append(path[0])

# Output
print("\n=== Greedy TSP Result ===")
print("Tour (city indices):", path)
print("Total Distance:", total_cost)
