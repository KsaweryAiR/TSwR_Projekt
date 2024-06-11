
import numpy as np
# from models.free_model import FreeModel
from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
from models.manipulator_model import ManiuplatorModel
from .pd_controller import PDDecentralizedController
# from models.ideal_model import IdealModel

class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.m3 = 0.5
        self.r3 = 0.05
        self.model = ManiuplatorModel(Tp, self.m3, self.r3)
        self.pd_controller = PDDecentralizedController(Kp, Kd)
        self.Kp = Kp
        self.Kd = Kd
        p1 = p[0]
        p2 = p[1]
        self.L = np.array([
            [3*p1, 0],
            [0, 3*p2],
            [3*p1**2, 0],
            [0, 3*p2**2],
            [p1**3, 0],
            [0, p2**3]
            ])
        W = np.array([[1., 0., 0., 0., 0., 0.],
                      [0., 1., 0., 0., 0., 0.]])
        A = np.array([[0., 0., 1., 0., 0., 0.],
                      [0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0., 1.],
                      [0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0.]])
        B = np.zeros((6,2))
        self.eso = ESO(A, B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        ### DONE Implement procedure to set eso.A and eso.B
        x = np.concatenate([q, q_dot], axis=0)
        M = self.model.M(x)
        M_inv = np.linalg.inv(M)
        C = self.model.C(x)

        A = np.zeros((6, 6))
        A[:2, 2:4] = np.eye(2)
        A[2:4, 4:6] = np.eye(2)
        A[2:4, 2:4] = -M_inv @ C

        B = np.zeros((6, 2))
        B[2:4, :] = M_inv

        self.eso.A = A
        self.eso.B = B

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        ### DONE implement centralized ADRFLC
        q1, q2, q1_dot, q2_dot = x
        q = np.array([q1, q2])
        q_dot = np.array([q1_dot, q2_dot])
        M = self.model.M(x)
        C = self.model.C(x)
        z_hat = self.eso.get_state()
        x_hat = z_hat[0:2]
        x_hat_dot = z_hat[2:4]
        f = z_hat[4:]
        v = self.pd_controller.calculate_control(q, q_dot, q_r, q_r_dot, q_r_ddot)
        # print("M ksztalt: ", M.shape)
        # print("f ksztalt: ", f.shape)
        # print("v ksztalt: ", v.shape)
        # print("xhat ksztalt: ", x_hat_dot.shape)
        u = M @ (v - f) + C @ x_hat_dot
        self.last_u = u
        self.update_params(x_hat, x_hat_dot)
        self.eso.update(q.reshape(len(q), 1), u.reshape(len(u), 1))
        return u