import numpy as np
from .controller import Controller


class PDDecentralizedController(Controller):
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def calculate_control(self, q, q_dot, q_r, q_r_dot, q_r_ddot):
        #adrc flc 
        if isinstance(self.kp, np.ndarray) or isinstance(self.kd, np.ndarray):
            u = q_r_ddot +  self.kd@(q_r_dot- q_dot) + self.kp@(q_r-q)
        #mmac i zad1    
        else:
            u = q_r_ddot +  self.kd*(q_r_dot- q_dot) + self.kp*(q_r-q)
        return u
