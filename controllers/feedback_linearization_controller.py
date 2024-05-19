import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller
from .pd_controller import PDDecentralizedController


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp, kp, kd):
        self.m3 = 0.5
        self.r3 = 0.05
        self.model = ManiuplatorModel(Tp, self.m3, self.r3)
        self.pd_controller = PDDecentralizedController(kp, kd)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q1, q2, q1_dot, q2_dot = x
        q = np.array([q1, q2])
        q_dot = np.array([q1_dot, q2_dot])
        v = self.pd_controller.calculate_control(q, q_dot, q_r, q_r_dot, q_r_ddot)
        tau = self.model.M(x) @ v + self.model.C(x) @ q_r_dot
        return tau