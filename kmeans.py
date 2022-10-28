import numpy as np

# Define Euclidean distance
def dist(array1, array2):
    dist_vector = array1 - array2
    dist_squared = np.dot(dist_vector, dist_vector)
    return np.sqrt(dist_squared)

class kmeans:

    def __init__(self, k, iterations, data, dist):
        self.k = k
        self.iterations = iterations
        self.data = data
        self.dist = dist

        # Set the number of clusters
        self.k = 2

        # Set number of iterations
        self.iterations = 100

        # Record efficiency of the code
        self.fit_values = np.zeros(self.iterations)

        # Select k random points on the dataset
        self.indices = np.random.choice(range(0, self.data.shape[0]), self.k)

        # Initialize the centroids
        self.centroids = np.array([self.data[i] for i in self.indices])

        # Create array for assigning kth cluster for each data point
        self.assignment = np.zeros(self.data.shape[0])

        # Initialize zeroth iteration
        self.iter = 0

    def run(self):
        
        # Start iterations
        while self.iter < self.iterations:
            # For calculating efficiency
            efficiency_holder = np.zeros(self.k)

            # Calculate distances
            for index1, (point) in enumerate(self.data):
                distances = np.zeros(k)
                for index2, (centroid) in enumerate(self.centroids):
                    distances[index2] = self.dist(point, centroid)
                    
                    # Determine which cluster is closest to the point
                    argmin = np.argmin(distances)
                    # Assign cluster
                    self.assignment[index1] = argmin
                    # Add distances of points belonging to same cluster
                    efficiency_holder[argmin] += distances[argmin]**2

            # After assigning values, group them into clusters to get new centroid
            for cluster in range(0, self.k):
                # Determine indices of points belonging to a cluster
                cluster_indices = np.argwhere(self.assignment == cluster).ravel()
                cluster_group = self.data[cluster_indices]

                # Determine centroid of this cluster group
                self.centroids[cluster] = np.sum(cluster_group, axis = 0) / len(cluster_group)

            # Record efficiency for the given iteration
            self.fit_values[iter] = np.sum(efficiency_holder)

            # Update iter values
            self.iter += 1