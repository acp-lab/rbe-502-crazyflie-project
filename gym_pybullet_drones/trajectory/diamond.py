import numpy as np

def basis_matrix_M(t):
    M = np.array([[1,   t,    t**2,   t**3,       t**4,      t**5],       # p(t)
                  [0,   1,    2*t,    3*t**2,     4*t**3,    5*t**4],     # v(t) 
                  [0,   0,    2,      6*t,        12*t**2,   20*t**3]])   # a(t)    
    
    return M


def p2p_motion(t, tf, start_state, end_state):
    '''
    Trajectory when in point to point motion

    Parameters:
        t: current time
        tf: time to complete motion
        start_state: 3xN
        end_state: 3xN
    
    Returns:
        cur_state: 3xN with axis as columns and pose,vel,acc as rows 
    '''
    
    A = np.vstack((basis_matrix_M(0), basis_matrix_M(tf))) #6x6
    b = np.vstack((start_state, end_state)) #6xN

        
    # Solve for coefficients
    coeffs = np.linalg.solve(A, b) # 6xN
    cur_state = basis_matrix_M(t) @ coeffs # 3xN

    return cur_state


def diamond(t, tfinal=8):
    """
    Generate the desired state of a drone following a diamond-shaped trajectory.

    The function computes the drone’s position, velocity, and acceleration at
    any given time 't' while following a diamond-shaped trajectory.

    Parameters:
        t (float): Current time (in seconds).
        tfinal (float): Total trajectory duration.

    Returns:
        desired_state (dict):
            - 'pos'   (np.ndarray, shape (3,)): Desired position [x, y, z].
            - 'vel'   (np.ndarray, shape (3,)): Desired velocity [vx, vy, vz].
            - 'acc'   (np.ndarray, shape (3,)): Desired acceleration [ax, ay, az].
            - 'jerk'  (np.ndarray, shape (3,)): Desired jerk (set to zero).
            - 'yaw'   (float): Desired yaw angle (set to zero).
            - 'yawdot' (float): Desired yaw rate (set to zero).
    """
    
    if t < 0:
        raise ValueError("Time is negative")
    
    # Diamond geometry
    u = 1.0 / np.sqrt(2)  # ~0.707
    ux = 1.0
    
    # Waypoints in y-z plane (position only)
    waypoints_pos = np.array([[0.0, 0.0],   
                              [u,   u],        
                              [0.0, 2*u],   
                              [-u,  u],     
                              [0.0, 0.0]])
    
    # Time parameters
    tseg = tfinal / 4  # 2 seconds per segment
    num_segments = 4
    
    # X-axis: Smooth motion from 0 to ux over entire trajectory
    x_start_state = np.array([[0.0], [0.0], [0.0]])
    x_end_state = np.array([[ux], [0.0], [0.0]])
    
    
    if t <= tfinal:
        # Compute x-axis motion 
        x_state = p2p_motion(t, tfinal, x_start_state, x_end_state).flatten()  # (3,)
        
        # Y-Z plane: Diamond trajectory
        # Determine segment
        segment = int(t / tseg)
        segment = min(segment, num_segments - 1)  # Clamp to last segment
        
        # Time within current segment
        cur_t = t - segment * tseg
        
        # Get start and end waypoints for current segment
        p_start = waypoints_pos[segment]
        p_end = waypoints_pos[segment + 1]
        
        yz_start_state = np.array([p_start,        
                                   [0.0, 0.0],
                                   [0.0, 0.0]])
        
        yz_end_state = np.array([p_end,     
                                 [0.0, 0.0],
                                 [0.0, 0.0]])
        
        # Use p2p_motion
        yz_state = p2p_motion(cur_t, tseg, yz_start_state, yz_end_state)  # 3x2
        yz_pos = yz_state[0]  # [y, z]
        yz_vel = yz_state[1]  # [vy, vz]
        yz_acc = yz_state[2]  # [ay, az]
        
    else:
        # Hold at final position
        x_state = x_end_state.flatten()
        yz_pos = waypoints_pos[-1]
        yz_vel = np.array([0.0, 0.0])
        yz_acc = np.array([0.0, 0.0])
        
    
    pos = np.array([x_state[0], yz_pos[0], yz_pos[1]])
    vel = np.array([x_state[1], yz_vel[0], yz_vel[1]])
    acc = np.array([x_state[2], yz_acc[0], yz_acc[1]])
    
    desired_state = {
        'pos': pos,
        'vel': vel,
        'acc': acc,
        'jerk': np.array([0, 0, 0]),
        'yaw': 0,
        'yawdot': 0
    }

    return desired_state