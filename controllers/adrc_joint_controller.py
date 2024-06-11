import numpy as np
from observers.eso import ESO
from .controller import Controller
from .pd_controller import PDDecentralizedController

class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.initialize_parameters(b, kp, kd, p, q0, Tp)
        self.last_u = 0

    def initialize_parameters(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd
        self.pd_controller = PDDecentralizedController(self.kp, self.kd)
        A = self.create_matrix_A()
        B = self.create_matrix_B()
        L = self.create_matrix_L(p)
        W = self.create_matrix_W()
        self.eso = ESO(A, B, W, L, q0, Tp)

    def create_matrix_A(self):
        return np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])

    def create_matrix_B(self):
        return np.array([[0], [self.b], [0]])

    def create_matrix_L(self, p):
        return np.array([[3*p], [3*p**2], [p**3]])

    def create_matrix_W(self):
        return np.array([[1, 0, 0]])

    def set_b(self, b):
        self.eso.set_B(self.create_matrix_B())

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        q = x[0]
        q_dot = x[1]
        self.eso.update(q, self.last_u)
        q_hat, q_hat_dot, f = self.eso.get_state()
        v = self.pd_controller.calculate_control(q, q_dot, q_r, q_r_dot, q_r_ddot)
        u = self.calculate_u(v, f)
        self.last_u = u
        return u

    def calculate_u(self, v, f):
        return (v - f) / self.b
