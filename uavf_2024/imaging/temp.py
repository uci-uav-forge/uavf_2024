import numpy as np

def ellipsoid_axes_lengths(covariance_matrix, confidence_level):
    # Compute eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Calculate chi square value for the given confidence level
    dof = len(eigenvalues)  # Degrees of freedom
    chi_square_value = np.sqrt(2 * dof) * np.sqrt(np.percentile(np.random.chisquare(dof, size=100000), 100 - confidence_level))

    # Calculate lengths of the axes of the ellipsoid
    axes_lengths = 2 * chi_square_value * np.sqrt(eigenvalues)

    return axes_lengths

# Example usage
# covariance_matrix = np.array([[5.6681, 4.6314],
#                                [4.6314, 5.5951]])  # Example covariance matrix
covariance_matrix = np.array([[1, 0, 0],
                           [0.5, 1, 0],
                           [0, 0, 3]])  # Example covariance matrix
confidence_level = 95  # Confidence level in percentage

axes_lengths = ellipsoid_axes_lengths(covariance_matrix, confidence_level)
print("Axis lengths of the ellipsoid (at 95% confidence level):", axes_lengths)


