from uavf_2024.imaging.imaging_types import BoundingBox
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from scipy.optimize import linear_sum_assignment
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from torchvision.ops import box_iou
from dataclasses import dataclass

@dataclass
class Measurement:
    pose: tuple[np.ndarray, Rotation]
    box: BoundingBox

class Camera:
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

        #
        # your code goes here
        #

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

# https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse
def fit_ellipse(x, y):
    """

    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.
    """

    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()

def ellipse_to_box(ellipse: np.ndarray) -> BoundingBox:
    """
    Convert an ellipse to a bounding box

    Parameters
    ----------
    ellipse : np.ndarray
        Ellipse parameters in the form [a,b,c,d,e,f]

    Returns
    -------
    BoundingBox
        Bounding box with x, y, width, and height

    """
    a,b,c,d,e,f, = ellipse

    # Explanation for the math here:
    # first we solve the ellipse equation for one
    # of the variables, which yields a quadratic equation,
    # then setting the discriminant of that quadratic equation to 0.
    # This gives us a second quadratic equation, which we then solve
    # using the quadratic formula, which is the math below.
    # See visualization here: https://www.desmos.com/calculator/eyfn91opa4
    x_b = -(2*b*e-4*c*d)
    x_a = 2*(b**2-4*a*c)
    x_discriminant = np.sqrt(x_b**2 - 4*x_a*(e**2-4*c*f))

    y_b = -(2*b*d-4*a*e)
    y_a = 2*(b**2-4*a*c)
    y_discriminant = np.sqrt(y_b**2 - 4*y_a*(d**2-4*a*f))

    x_min = (-x_b - x_discriminant) / (2*x_a)
    x_max = (-x_b + x_discriminant) / (2*x_a)
    y_min = (-y_b - y_discriminant) / (2*y_a)
    y_max = (-y_b + y_discriminant) / (2*y_a)

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    # TODO: figure out how to account for when the bounding box is outside the image
    return BoundingBox(center_x, center_y, width, height)

class DroneTracker:
    resolution: tuple[int,int]
    focal_len_pixels: float
    def __init__(self, resolution: tuple[int,int], focal_len_pixels: float):
        self.tracks: list[self.Track] = []
        DroneTracker.resolution = resolution
        DroneTracker.focal_len_pixels = focal_len_pixels

    def predict(self, dt: float):
        for track in self.tracks:
            track.predict(dt)

    def update(self, cam_pose: tuple[np.ndarray, Rotation], measurements: list[BoundingBox]):
        # construct matrix of IOU similarity between measurements and tracks
        # iou_matrix[i,j] is the iou between track i and measurement j
        iou_matrix = np.zeros((len(self.tracks), len(measurements)))
        for i, track in enumerate(self.tracks):
            for j, measurement in enumerate(measurements):
                iou_matrix[i,j] = self._iou(track, measurement, cam_pose)

        # update tracks with measurements if iou > 0.5
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        for i, j in zip(row_ind, col_ind):
            if iou_matrix[i,j] > 0.5:
                self.tracks[i].update(measurements[j])
            else:
                self.tracks.append(self.Track(Measurement(cam_pose, measurements[j])))

        # create new tracks for unmatched measurements
        for j in range(len(measurements)):
            if j not in col_ind:
                self.tracks.append(self.Track(Measurement(cam_pose, measurements[j])))

        # prune tracks that don't have 50% seen to alive ratio
        self.tracks = [
            track for track in self.tracks 
            if  track.frames_alive == 0 
                or track.frames_seen/track.frames_alive > 0.5
        ]

    def _iou(self, track, box: BoundingBox, cam_pose: tuple[np.ndarray,Rotation]) -> float:
        '''
        `track` is a DroneTracker.Track object but the type annotation doesn't work
        '''
        track_box = track.simulate_measurement(cam_pose)
        track_box_arr = np.array([track_box.x, track_box.y, track_box.width, track_box.height])
        box_arr = np.array([box.x, box.y, box.width, box.height])
        return box_iou(track_box_arr, box_arr)
        

    '''
    Nested on purpose to be able to access self.resolution and self.focal_len_pixels
    '''
    class Track:
        def __init__(self, initial_measurement: Measurement):
            # x dimension is [x,y,z, vx,vy,vz, radius]
            self.kf = UnscentedKalmanFilter(
                dim_x=7,
                dim_z=4,
                dt=1/30, # TODO: figure out why this needs to be in the constructor since we can't guarantee the frame rate
                hx=self._measurement_fn,
                fx=self._state_transition,
                points = MerweScaledSigmaPoints(7, 1e-3, 2, 0)
            )
            self.cam_pose = initial_measurement.pose
            x, _covariance = self._generate_initial_state(initial_measurement.box)
            self.kf.x = x
            # TODO: figure out how to set the initial covariance
            
            self.frames_alive = 0
            self.frames_seen = 0

        @staticmethod
        def compute_measurement(cam_pose: tuple[np.ndarray, Rotation], state: np.ndarray) -> BoundingBox:
            '''
            `state` is the state of the track, which is a 7 element array
            '''
            cam = Camera(DroneTracker.focal_len_pixels, 
                         [DroneTracker.resolution[0]/2, DroneTracker.resolution[1]/2], 
                         cam_pose[1].as_matrix(), 
                         cam_pose[0].reshape(3,1))

            state_position = state[:3]
            state_radius = state[6]
            control_points = np.array(
                [
                    [state_radius, 0, 0],
                    [-state_radius, 0, 0],
                    [0, state_radius, 0],
                    [0, -state_radius, 0],
                    [0, 0, state_radius],
                    [0, 0, -state_radius]
                ]
            ) + state_position

            projected_points = cam.project(control_points.T)
            hull = ConvexHull(projected_points.T)
            hull_pts = projected_points.T[hull.vertices]

            projected_ellipse = fit_ellipse(hull_pts[0], hull_pts[1])

            return ellipse_to_box(projected_ellipse)

        def _generate_initial_state(self, box: BoundingBox) -> tuple[np.ndarray, np.ndarray]:
            '''
            Returns the initial state and covariance for the Kalman filter.

            This is done by assuming the radius of the drone's bounding sphere is 0.5 meters.

            The initial state is a 7 element array with the following elements:
            [x,y,z, vx,vy,vz, radius]

            TODO: The initial covariance is a 7x7 array but rn None is returned
            because I haven't figured out how to set the initial covariance yet
            '''

            box_center_ray = np.array([box.x - DroneTracker.resolution[0]//2, box.y - DroneTracker.resolution[1]//2, DroneTracker.focal_len_pixels])
            camera_look_vector = self.cam_pose[1].apply(box_center_ray)
            camera_look_vector = 0.1 * camera_look_vector / np.linalg.norm(camera_look_vector)
            box_area = box.width * box.height

            # This could probably be done analytically but binary search was easier
            low = 1
            high = 1000
            while low < high:
                distance = (low + high) / 2
                x_guess = self.cam_pose[0] + distance * camera_look_vector
                projected_box = self.compute_measurement(self.cam_pose, np.hstack([x_guess, np.array([0,0,0,0.5])]))
                projected_box_area = projected_box.width * projected_box.height
                if abs(projected_box_area - box_area) < 1:
                    break
                elif projected_box_area < box_area:
                    low = distance
                else:
                    high = distance

            return np.hstack([x_guess, np.array([0,0,0,0.5])]), None


        def _measurement_fn(self, x: np.ndarray) -> np.ndarray:
            '''
            Returns measurement of the state from self.cam_pose. 
            self.cam_pose needs to be set before calling this function

            The returned ndarray is of shape (7,) with the following elements:
            [x,y,z, vx,vy,vz, radius]
            ''' 
            return self.compute_measurement(self.cam_pose, x)
        
        def simulate_measurement(self, cam_pose: tuple[np.ndarray, Rotation]) -> BoundingBox:
            '''
            Simulates a measurement with the current state of the track
            '''
            
            return self.compute_measurement(cam_pose, self.kf.x)

        @staticmethod
        def _state_transition(x: np.ndarray, dt: float) -> np.ndarray:
            cur_pos = x[:3]
            cur_vel = x[3:6]
            return np.hstack([cur_pos + cur_vel*dt, cur_vel, x[6]])

        def predict(self, dt: float):
            self.frames_alive += 1
            self.kf.predict(dt)

        def update(self, measurement: Measurement):
            self.frames_seen += 1
            self.cam_pose = measurement.pose
            box_x = measurement.box.x
            box_y = measurement.box.y
            box_w = measurement.box.width
            box_h = measurement.box.height
            self.kf.update(np.array([box_x, box_y, box_w, box_h]))

