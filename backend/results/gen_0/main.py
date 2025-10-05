
# Initial TSP solution using nearest neighbor heuristic

def solve_tsp(cities, distance_matrix):
    """Solve TSP using nearest neighbor heuristic"""
    n = len(cities)
    tour = [0]  # Start from city 0
    unvisited = set(range(1, n))
    
    current = 0
    while unvisited:
        # Find nearest unvisited city
        nearest = min(unvisited, key=lambda city: distance_matrix[current][city])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    return tour


def calculate_tour_length(tour, distance_matrix):
    """Calculate total tour length"""
    total = 0
    for i in range(len(tour) - 1):
        total += distance_matrix[tour[i]][tour[i + 1]]
    # Return to start
    total += distance_matrix[tour[-1]][tour[0]]
    return total


def run_experiment(**kwargs):
    """Main experiment function"""
    cities = kwargs.get('cities')
    distance_matrix = kwargs.get('distance_matrix')
    
    tour = solve_tsp(cities, distance_matrix)
    distance = calculate_tour_length(tour, distance_matrix)
    
    return {
        'tour': tour,
        'distance': distance,
        'fitness': -distance  # Minimize distance
    }
