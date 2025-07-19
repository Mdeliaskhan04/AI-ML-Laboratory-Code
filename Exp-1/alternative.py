import numpy as np
# Distance matrix
dist_matrix = [
    [0, 29, 20, 21],
    [29, 0, 15, 17],
    [20, 15, 0, 28],
    [21, 17, 28, 0]
]
n = len(dist_matrix)
visited = [False] * n
path = []
total_cost = 0
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

print("Tour:", path)
print("Total Distance:", total_cost)
