from dataclasses import dataclass
import numpy as np
import cv2
import itertools

@dataclass
class ColorSegmentationResult:
    mask: np.ndarray
    shape_color: np.ndarray
    letter_color: np.ndarray

def color_segmentation(image: np.ndarray, rgb_mask_save_path: str = None):
    """
    Segments the image with k=3 k-means clustering, and returns the mask and the centroids.
    The mask is a numpy array of shape (w,h) where each pixel is an integer in [0,1,2]. 0=background, 1=shape, 2=image
    The centroids are a numpy array of shape (2,3) where the first row is the color of the shape and the second is the color of the letter.
    """

    #making a numpy with 5 dimensions
    center_augmented_data = []
    w, h = image.shape[:2]
    for x,y in itertools.product(range(w), range(h)):
        b,g,r = image[x][y]
        dist_to_center = np.sqrt((x-w/2)**2+(y-h/2)**2)
        center_augmented_data.append([dist_to_center, b,g,r])


    kmeans_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # Apply K-means clustering algorithm:
    k=3
    _ret, mask, centroids = cv2.kmeans(np.array(center_augmented_data).astype(np.float32), k, None, kmeans_criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # without center augmentation:
    # ret, labels, centroids = cv2.kmeans(image.reshape((-1, 3)).astype(np.float32), k, None, kmeans_criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    sort_indices = np.argsort(-centroids[:,0])
    just_rgb_centroids = centroids[:, 1:].astype(np.uint8)
    rgb_mask = np.zeros((mask.shape[0], 3), dtype=np.uint8)
    reordered_centroids = just_rgb_centroids[sort_indices]
    for i in range(len(mask)):
        if rgb_mask_save_path is not None:
            rgb_mask[i] = just_rgb_centroids[mask[i]]
        mask[i] = np.where(sort_indices==mask[i][0])[0][0]

    if rgb_mask_save_path is not None:
        cv2.imwrite(rgb_mask_save_path, rgb_mask.reshape((w,h,3)))
        
    return ColorSegmentationResult(
        mask=mask.reshape((w,h)),
        shape_color=reordered_centroids[1],
        letter_color=reordered_centroids[2]
    )

if __name__ == "__main__":
    image = cv2.imread("crop0.png")
    res = color_segmentation(image, rgb_mask_save_path="rgb_mask.png")
    print(res.shape_color, res.letter_color)
    cv2.imwrite('res.png', res.mask*127)


    


