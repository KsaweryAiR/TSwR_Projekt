import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel
from .pd_controller import PDDecentralizedController


class MMAController(Controller):
    def __init__(self, Tp, kp, kd):
        self.pd_controller = PDDecentralizedController(kp, kd)

        self.Tp = Tp
        self.Kp = kp
        self.Kd = kd
        self.uP = None

        m3_1=0.1  
        r3_1=0.05
        self.model1 = ManiuplatorModel(Tp, m3_1, r3_1)

        m3_2=0.01  
        r3_2=0.01
        self.model2 = ManiuplatorModel(Tp, m3_2 , r3_2)

        m3_3=1.0  
        r3_3=0.3
        self.model3 = ManiuplatorModel(Tp, m3_3 , r3_3)

        self.models = [self.model1, self.model2, self.model3]
        self.i = 0
    
    def choose_model(self, x):
        high_error = float('inf')
        q1, q2, q1_dot, q2_dot = x
        qs = np.array([q1, q2])
        qs_dot = np.array([q1_dot, q2_dot])  

        for i, model in enumerate(self.models): 
            if self.uP is not None:  
                q_dot_dot = np.linalg.solve(model.M(x), self.uP - model.C(x) @ qs_dot)
                q_dot = qs_dot + q_dot_dot * self.Tp
                q = qs + qs_dot * self.Tp

                x_e = np.concatenate([q, q_dot])
                error = np.linalg.norm(x_e - x)

                if error < high_error:
                    high_error = error
                    self.i = i
        print("@@@@@@@@@@@@@@@ i :", self.i)
             

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        q1, q2, q1_dot, q2_dot = x
        q = np.array([q1, q2])
        q_dot = np.array([q1_dot, q2_dot])
        self.choose_model(x)
        v = self.pd_controller.calculate_control(q, q_dot, q_r, q_r_dot, q_r_ddot)
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v + C @ q_dot
        self.uP = u
        return u