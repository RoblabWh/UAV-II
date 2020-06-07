import numpy as np
import math
from numba import jit

def find_random_points(samplesize, points):
    """Finds a set of unique random points from a point cloud
    :params:
        samplesize - size of the generated set
        points - set of points from which random points are selected

    :return:
        list of random points
    """
    random_points = []
    random_indices = np.random.choice(len(points), samplesize, replace=False)
    for index in random_indices:
        random_points.append(points[index])
    return random_points

# use HNF here generate plane from 3 points
def plane_from_3_points(points):
    """Determines the normal vector of a plane given by 3 points

    :params:
        points - list of 3 points
    :return:
        (n,d) - Hesse normal from of a plane
                n - normalized normal vector
                d - distance from origin
    """
    n = np.cross(points[0] - points[2], points[1] - points[2])
    n_0 = n / np.linalg.norm(n) if np.dot(points[2], n) >= 0 else -n / np.linalg.norm(n)
    d = np.dot(points[2], n_0)
    return n_0, d




# %%
def count_consensus(n, d, max_diff, points):
    """Counts the consensus
       counts the consensus and gives back the
       line numbers of the corresponding points

        :params:
            n - normal vector of the plane
            d - distance of the plane from origin
            max_diff - max distance of a point to plane
            points - array of points

        :return:
            new_points - list - Consensual points
            outliers - list - Outlier points
    """
    new_points = []
    outliers = []
    for point in points:
        if abs(np.dot(point, n) - d) <= max_diff:
            new_points.append(point)
        else:
            outliers.append(point)
    return new_points, outliers


# implement RANSAC
def ransac(points, z, w, max_dif):
    '''Finds the normal vectors of the planes in a scene

    :params:
        z: likelihood, that plane can be found
        w: outliers in percent
        t: threshold for accepted observations
    :return:
        n: - normalized normal vector
        d: - distance from origin
        new_points: - list - Consensual points
        outliers: - list - Outlier points
    '''
    n_max = 0
    d_max = 0
    max_points = []
    min_outliers = []
    iterations = int(math.log(1 - z) / math.log(1 - (1 - w) ** 3))
    print(iterations, 'Iterations needed')
    inliers = None
    for i in range(iterations):
        chosen_points = find_random_points(3, points)
        n, d = plane_from_3_points(chosen_points)
        inliers, outliers = count_consensus(n, d, max_dif, points)
        if len(inliers) > len(max_points):
            max_points = inliers
            min_outliers = outliers
            n_max = n
            d_max = d
    return n_max, d_max, np.array(max_points), np.array(min_outliers), inliers
