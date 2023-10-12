import numpy as np

COLORS_TO_RGB = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'orange': (255, 165, 0),
    'purple': (128, 0, 128),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'brown': (165, 42, 42),
}

class ColorClassifier:
    def __init__(self):
        pass
    def predict(self, rgb_val: np.ndarray) -> np.ndarray:
        '''
        Takes shape (1,3) rgb array and returns (1,8) class probability distribution
        '''
        distances = np.zeros(8)
        for i, c in enumerate(COLORS_TO_RGB.values()):
            distances[i] = np.linalg.norm(rgb_val - np.array(c))
        
        # turn distances to probability distribution
        # TODO: use math for this that has a real theoretical grounding
        distances = max(distances) - distances
        distances /= sum(distances)

        return distances




