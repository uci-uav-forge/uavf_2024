import numpy as np

def convert_NED_ENU_in_inertial(self, x) -> np.ndarray:
        ''' Converts a state between NED or ENU inertial frames.
            This operation is commutative. 
        '''
        assert len(x) == 3
        new_x = np.float32(
            [x[1], x[0], -x[2]])
        return new_x
    

def convert_NED_ENU_in_body(self, x) -> np.ndarray:
    ''' Converts a state between NED or ENU body frames.
        (More formally known as FRD or RLU body frames)
        This operation is commutative. 
    '''
    assert len(x) == 3
    new_x =  np.float32(
        [x[0], -x[1], -x[2]])
    return new_x


def convert_body_to_inertial_frame(self, ang_rate_body, ang) -> np.ndarray:
    ''' Converts body euler angle rates to their inertial frame counterparts.
    '''
    assert len(ang_rate_body) == 3
    assert len(ang) == 3

    phi, theta, psi = ang
    W_inv = np.float32([
        [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
        [0,               np.cos(phi),              -np.sin(phi)],
        [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)],
    ])
    ang_rate_inertial = W_inv @ ang_rate_body
    return ang_rate_inertial


def convert_inertial_to_body_frame(self, ang_rate_inertial, ang) -> np.ndarray:
    ''' Converts inertial euler angle rates to their body frame counterparts.
    '''
    assert len(ang_rate_inertial) == 3
    assert len(ang) == 3

    phi, theta, psi = ang
    W = np.float32([
        [1,           0,             -np.sin(theta)],
        [0,  np.cos(phi), np.cos(theta)*np.sin(phi)],
        [0, -np.sin(phi), np.cos(theta)*np.cos(phi)]
    ])
    ang_rate_body = W @ ang_rate_inertial
    return ang_rate_body