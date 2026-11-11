import numpy as np
import math
from typing import Optional

class PolynomialTrajectory:
    def __init__(self, to, tf, deg=5, derv=2, initial_state=None, final_state=None):
        '''
        Get the Boundary conditions for the Polynomial Trajectory
        @Parms:
            to: Initial time
            tf: Final time
            deg: Degree of the polynomial
            derv: Derivative order (max derivative to constrain)
            initial_state: shape (derv+1, n_dims) - [[p0x,p0y,p0z], [v0x,v0y,v0z], [a0x,a0y,a0z]]
            final_state: shape (derv+1, n_dims) - [[pfx,pfy,pfz], [vfx,vfy,vfz], [afx,afy,afz]]
        '''
        self.initial_time = to
        self.final_time = tf
        self.degree = deg
        self.derivative = derv
        self.A: Optional[np.ndarray] = None
        self.a: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self.n_dims: Optional[int] = None
        
        # Auto-compute coefficients if both states are provided
        if initial_state is not None and final_state is not None:
            self.initial_state = np.array(initial_state)
            self.final_state = np.array(final_state)
            
            # Verify shapes
            if self.initial_state.shape[0] > derv + 1:
                raise ValueError(f"initial_state({self.initial_state.shape[0]}) must have less than equal to {derv + 1} rows (derivatives)")
            if self.final_state.shape[0] > derv + 1:
                raise ValueError(f"final_state({self.final_state.shape[0]}) must have less than equal to {derv + 1} rows (derivatives)")
            if self.initial_state.shape[1] != self.final_state.shape[1]:
                raise ValueError("initial_state and final_state must have same number of dimensions")
            
            # Get number of dimensions from shape
            self.n_dims = self.initial_state.shape[1]
            
            # Auto-compute coefficients
            self.updateCoeffs()
        else:
            self.initial_state = None
            self.final_state = None
    
    def _getMij(self, t, i, j):
        '''
        Get the Mij element of the M matrix
        @Parms:
            t: time
            i: row index (derivative order)
            j: column index (coefficient index)
        '''
        if i <= j:
            return (math.factorial(j) / math.factorial(j - i)) * (t ** (j - i))
        else:
            return 0.0
    
    def _getM(self, t):
        '''
        Get the M matrix at time t
        @Parms:
            t: time
        '''
        M = np.zeros((self.derivative + 1, self.degree + 1))
        for i in range(self.derivative + 1):
            for j in range(self.degree + 1):
                M[i][j] = self._getMij(t, i, j)
        return M
    
    def _calcA(self):
        '''
        Get the A matrix by stacking M(t0) and M(tf)
        '''
        if self.initial_time is None or self.final_time is None:
            raise ValueError("Initial and Final times must be provided")
        
        M_0 = self._getM(self.initial_time)
        M_f = self._getM(self.final_time)

        return np.vstack((M_0, M_f))
    

    def updateCoeffs(self):
        '''
        Calculate the coefficients of the polynomial for all dimensions
        '''
        if self.initial_state is None or self.final_state is None:
            raise ValueError("Initial and Final states must be provided")
        
        # Calculate A matrix (same for all dimensions)
        self.A = self._calcA()
        
        # Construct b matrix: stack initial and final states vertically
        # b has shape (2*(derv+1), n_dims)
        self.b = np.vstack([self.initial_state, self.final_state])
        
        try:
            # Solve Aa = b for all dimensions at once
            # a will have shape (degree+1, n_dims)
            self.a = np.linalg.solve(self.A, self.b)
        except np.linalg.LinAlgError:
            raise ValueError("Matrix A is singular, cannot solve for coefficients")
    
    def getPoly(self, t) -> np.ndarray:
        '''
        Get the polynomial and all its derivatives at time t for all dimensions
        @Parms:
            t: time
        @Returns:
            Array of shape (derivative+1, n_dims) containing:
            [[px, py, pz, ...],
             [vx, vy, vz, ...],
             [ax, ay, az, ...],
             ...]
        '''
        if self.a is None:
            self.updateCoeffs()
        
        # Type assertion for Pylance
        assert self.a is not None, "Coefficients could not be computed"
        
        Mt = self._getM(t)
        # Mt @ self.a gives shape (derivative+1, n_dims)
        poly_t = Mt @ self.a
        
        return poly_t


def get_circle_state(theta_traj, z_d):
    theta = theta_traj[0][0]  # Extract angle
    theta_w = theta_traj[1][0]      # Extract angular velocity
    theta_acc = theta_traj[2][0]    # Extract angular acceleration
    pos = np.array([np.cos(theta), np.sin(theta), z_d])
    vel = np.array([-theta_w * np.sin(theta), theta_w * np.cos(theta), 0.0])
    acc = np.array([-theta_acc * np.sin(theta) - theta_w**2 * np.cos(theta),
                            theta_acc * np.cos(theta) - theta_w**2 * np.sin(theta),
                            0.0])
    return pos, vel, acc

#############CIRCLE PARAMETERS(EDIT ME)#############
climb_up_time_duration = 5.0  # Duration to climb up before starting the circle
circle_time_duration = 10.0  # Duration to complete one circle
climb_down_time_duration = 15.0  # Duration to climb down after completing the circle

z_d = 1.0  # Desired altitude during the circle
R = 1.0  # Radius of the circle

#Omega states
theta_initial = np.array([[0.0], [0.0], [0.0]])  # theta, omega, alpha
theta_final = np.array([[np.pi*2], [0.0], [0.0]])

#CLimbUp states
climb_up_initial = np.array([
    [0.0, 0.0, 0.5],   # [px0, py0, pz0]
    [0.0, 0.0, 0.0],   # [vx0, vy0, vz0]
    [0.0, 0.0, 0.0]    # [ax0, ay0, az0]
])

climb_up_final = np.array([
    [1.0, 0.0, 1.0],  # [pxf, pyf, pzf]
    [0.0, 0.0, 0.0],   # [vxf, vyf, vzf]
    [0.0, 0.0, 0.0]    # [axf, ayf, azf]
])
#CLimbDown states
climb_down_initial = climb_up_final
climb_down_final = climb_up_initial


#Trajectory Generators
climb_up_traj = PolynomialTrajectory(
    to=0.0, tf=5.0,
    initial_state=climb_up_initial,
    final_state=climb_up_final
)


theta_traj = PolynomialTrajectory(
    to=5.0, tf=10.0,
    initial_state=theta_initial,
    final_state=theta_final
)
climb_down_traj = PolynomialTrajectory(
    to=10.0, tf=15.0,
    initial_state=climb_down_initial,
    final_state=climb_down_final
)
###########################################
def circle(t, tf=8):
    """
    Generate the desired state of a drone following a circular trajectory.

    The function computes the drone’s position, velocity, and acceleration
    at a given time 't' while following a circular trajectory.

    Parameters:
        t (float): Current time (in seconds).
        tf (float): Total trajectory duration.

    Returns:
        desired_state (dict): 
            - 'pos'   (np.ndarray, shape (3,)): Desired position [x, y, z].
            - 'vel'   (np.ndarray, shape (3,)): Desired velocity [vx, vy, vz].
            - 'acc'   (np.ndarray, shape (3,)): Desired acceleration [ax, ay, az].
            - 'jerk'  (np.ndarray, shape (3,)): Desired jerk (set to zero).
            - 'yaw'   (float): Desired yaw angle (set to zero).
            - 'yawdot' (float): Desired yaw rate (set to zero).
    """

    """
    Write your code here.
    """
    pos = np.zeros(3)
    vel = np.zeros(3)
    acc = np.zeros(3)
    
    #Get trajectory
    if( t <= climb_up_time_duration ):
        # Climb Up Phase
        poly_t = climb_up_traj.getPoly(t)
        pos = poly_t[0]
        vel = poly_t[1]
        acc = poly_t[2]
    elif( t <= circle_time_duration ):
        # get w
        poly_theta = theta_traj.getPoly(t)
        pos, vel, acc = get_circle_state(poly_theta, z_d)
    elif( t <= climb_down_time_duration ):
        poly_t = climb_down_traj.getPoly(t)
        pos = poly_t[0]
        vel = poly_t[1]
        acc = poly_t[2]
    else:
        # Hover at final position
        pos = climb_up_initial[0]
        vel = climb_up_initial[1]
        acc = climb_up_initial[2]

    desired_state = {
        'pos': pos,
        'vel': vel,
        'acc': acc,
        'jerk': np.array([0, 0, 0]),
        'yaw': 0,
        'yawdot': 0
    }

    return desired_state
