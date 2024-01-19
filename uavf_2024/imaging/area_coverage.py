from __future__ import annotations
import numpy as np
import cv2

class AreaCoverageTracker:
    def __init__(self, camera_hfov: float, camera_resolution: tuple[int,int]):
        '''
        `camera_fov` is the horizontal FOV, in degrees
        `camera_resolution` (w,h) in pixels. It doesn't really matter besides giving the aspect ratio
        '''
        self.camera_hfov = camera_hfov
        self.camera_resolution = camera_resolution

        # The frustrum projections and their labels are just aligned by index
        self.frustrum_projections: list[np.ndarray] = []
        self.labels: list[str] = []

    def update(self, camera_pose: np.ndarray, label: str = None):
        '''
            the pose is [x,y,z, altitude, azimuth, roll] in degrees, where the camera at (0,0,0) is
            pointed at the negative z axis, positive x axis is the right side of the camera and positive 
            y axis goes up from the camera. The rotations are applied relative to local frame in this order: 
            azimuth then altitude then roll.
        '''
        w,h = self.camera_resolution
        focal_len = w/(2*np.tan(np.deg2rad(self.camera_hfov/2)))

        rot_alt, rot_az, rot_roll = np.deg2rad(camera_pose[3:])
        camera_position = camera_pose[:3]

        rot_alt_mat = np.array([[1,0,0],
                                [0,np.cos(rot_alt),-np.sin(rot_alt)],
                                [0,np.sin(rot_alt),np.cos(rot_alt)]])
    
        rot_az_mat = np.array([[np.cos(rot_az),0,np.sin(rot_az)],
                                [0,1,0],
                                [-np.sin(rot_az),0,np.cos(rot_az)]])
        
        rot_roll_mat = np.array([[np.cos(rot_roll),-np.sin(rot_roll),0],
                                [np.sin(rot_roll),np.cos(rot_roll),0],
                                [0,0,1]])

        
        frustrum_projection = []

        # the vector pointing out the camera at the target, if the camera was facing positive Z
        for x,y in [(-w//2,-h//2),(-w//2,h//2),(w//2,h//2),(w//2,-h//2)]:
            initial_direction_vector = np.array([x,y,-focal_len])

            # rotate the vector to match the camera's rotation
            rotated_vector = rot_az_mat @ rot_alt_mat @ rot_roll_mat @ initial_direction_vector

            # solve camera_pose + t*rotated_vector = [x,0,z] = target_position
            t = -camera_position[1]/rotated_vector[1]
            ground_position = camera_position + t*rotated_vector
            position_2d = ground_position[[0,2]]
            frustrum_projection.append(position_2d)
        
        self.frustrum_projections.append(np.array(frustrum_projection))
        self.labels.append(label)

    def get_coverage_picture(self, max_res: int) -> tuple[np.ndarray, tuple[float,float,float,float]]:
        '''
        Returns [img, extent] where

        img is a (h,w,1) grayscale image where either h or w is max_res, 
        and the pixel value is the number of times that ground spot has been seen

        and extend is a (min_x, min_y, max_x, max_y) tuple of the extent of the graph
        '''
        min_x, min_y, max_x, max_y = [np.Inf, np.Inf, -np.Inf, -np.Inf]
        for quadrilateral in self.frustrum_projections:
            for x,y in quadrilateral:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
        
        width = max_x-min_x
        height = max_y-min_y

        scale = max_res/max(width, height)
        width = int(width*scale)
        height = int(height*scale)

        canvas = np.zeros((height,width,1), dtype=np.uint8)
        for quadrilateral in self.frustrum_projections:
            quadrilateral = ((quadrilateral-np.array([min_x,min_y]))*scale).astype(np.int32)
            new_drawing = np.zeros((height,width,1), dtype=np.uint8)
            cv2.fillConvexPoly(new_drawing, quadrilateral, 1)
            canvas += new_drawing

        return canvas, (min_x, min_y, max_x, max_y)

    def visualize(self, file_path: str, max_res: int) -> None:
        pic, (min_x, min_y, max_x, max_y) = self.get_coverage_picture(max_res)
        scale = max_res/max(max_x-min_x, max_y-min_y)
        # normalize so max value is 255
        max_cov: int = pic.max()
        pic *= 255//max_cov 
        pic = cv2.applyColorMap(pic, cv2.COLORMAP_VIRIDIS)

        for quadrilateral, label in zip(self.frustrum_projections, self.labels):
            if label is None: continue
            quad_center = (quadrilateral.mean(axis=0) - np.array([min_x,min_y]))*scale
            cv2.putText(pic, label, quad_center.astype(np.int32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.putText(pic, f"Graph Extent: ({min_x:.2f},{min_y:.2f}) to ({max_x:.2f},{max_y:.2f})", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(pic, f"Max coverage: {max_cov} times", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imwrite(file_path, pic)

    def times_seen(self, x: float, y: float, resolution: int  = 1000) -> int:
        '''
        returns the number of times the ground point (x,y) has been seen
        '''
        img, (min_x, min_y, max_x, max_y) = self.get_coverage_picture(resolution)

        pixel_x = int((x-min_x)/(max_x-min_x)*resolution)
        pixel_y = int((y-min_y)/(max_y-min_y)*resolution)

        return img[pixel_y, pixel_x, 0]



        
