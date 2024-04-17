import numpy as np

class CameraModel:
    """
    A simple data structure describing camera parameters 
    
    The parameters describing the camera
    cam.f : float   --- camera focal length (in units of pixels)
    cam.c : (2,) tuple/array  --- offset of principle point (in units of pixels)
    cam.R : (3,3) matrix --- camera rotation (around the global origin)
    cam.t : (3,1) vector --- camera translation (location of camera center relative to the global origin)

    
    """ 
    def __init__(self,f: float, c: tuple[float,float], R: np.ndarray, t: np.ndarray):
        assert R.shape == (3,3)
        assert t.shape == (3,1), f"t.shape = {t.shape}"
        self.f = f
        self.c = c
        self.R = R
        self.t = t

        
    def project(self,pts3: np.ndarray):
        """
        Project the given 3D points in world coordinates into the specified camera    

        Parameters
        ----------
        pts3 : 2D numpy.array (dtype=float)
            Coordinates of N points stored in a array of shape (3,N)

        Returns
        -------
        pts2 : 2D numpy.array (dtype=float)
            Image coordinates of N points stored in an array of shape (2,N)

        """

        assert(pts3.shape[0]==3)

        K = np.array([
            [self.f, 0, self.c[0]], 
            [0, self.f, self.c[1]], 
            [0, 0, 1]])
        C = K @ np.hstack([self.R.T, -self.R.T@self.t])
        CP = C @ np.vstack([pts3, np.ones(pts3.shape[1])])
        pts2 = CP[:2, :] / CP[2, :]
        
        assert(pts2.shape[1]==pts3.shape[1])
        assert(pts2.shape[0]==2)
    
        return pts2