import numpy as np


class SEQIJCR:
    __slots__ = ['N', 'b', 'e_E', 'e_Q', 'e_J', 'e_C',
                 'T_E', 'T_Q', 'T_I', 'T_J', 'T_C',
                 'w_E', 'w_Q', 'w_I', 'w_J', 'w_C',
                 'mu', 'Pi', 'pi',
                 'last_update_time', 'actions']
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.last_update_time = 0
        self.actions = {}

    def derivative(self, y, t):
        # S : 0
        # E : 1
        # Q : 2
        # I : 3
        # J : 4
        # C : 5
        # R : 6
        # D : 7

        # Derivatives
        Sp = self.get_Pi(t) - self.get_b(t) * y[0] / self.get_N(t) * (y[3] + self.get_e_E(t) * y[1] +
                                                                      self.get_e_Q(t) * y[2] + self.get_e_J(t) * y[4] +
                                                                      self.get_e_C(t) * y[5]) - self.get_mu(t) * y[0]
        Ep = self.get_pi(t) + self.get_b(t) * y[0] / self.get_N(t) * (y[3] + self.get_e_E(t) * y[1] +
                                                                      self.get_e_Q(t) * y[2] + self.get_e_J(t) * y[4] +
                                                                      self.get_e_C(t) * y[5]) - (1/self.get_T_E(t) + self.get_mu(t)) * y[1]
        Qp = (1 - self.get_w_E(t)) / self.get_T_E(t) * y[1] - (1/self.get_T_Q(t) + self.get_mu(t)) * y[2]
        Ip = self.get_w_E(t) / self.get_T_E(t) * y[1] - (1/self.get_T_I(t) + self.get_mu(t)) * y[3]
        Jp = self.get_w_Q(t) / self.get_T_Q(t) * y[2] + self.get_w_I(t) / self.get_T_I(t) * y[3] + (1 - self.get_w_C(t)) / self.get_T_C(t) * y[5] - (1/self.get_T_J(t) + self.get_mu(t)) * y[4]
        Cp = self.get_w_J(t) / self.get_T_J(t) * y[4] - (1/self.get_T_C(t) + self.get_mu(t)) * y[5]
        Rp = (1 - self.get_w_Q(t)) / self.get_T_Q(t) * y[2] + (1 - self.get_w_I(t)) / self.get_T_I(t) * y[3] + (1 - self.get_w_J(t)) / self.get_T_J(t) * y[4] - self.get_mu(t) * y[6]
        Dp = self.get_w_C(t) / self.get_T_C(t) * y[5]

        return np.array([Sp, Ep, Qp, Ip, Jp, Cp, Rp, Dp], dtype=float).transpose()

    def next_iteration(self, y, t, h):
        # Uses RK4
        k_1 = self.derivative(y, t)
        k_2 = self.derivative(y + k_1 * h / 2, t + h / 2)
        k_3 = self.derivative(y + k_2 * h / 2, t + h / 2)
        k_4 = self.derivative(y + k_3 * h, t + h)
        return y + (k_1 + 2 * k_2 + 2 * k_3 + k_4) * h / 6

    def prediction(self, y_0, t_start, t_end, h, is_initial_estimator=False):
        n = int((t_end - t_start) / h)
        x = t_start + np.arange(n + 1) / n * (t_end - t_start)
        y = np.zeros((8, n + 1))
        y[:, 0] = y_0
        for i in range(n):
            if not is_initial_estimator:
                self.update_parameters((i + 1) * h)
            y[:, i + 1] = self.next_iteration(y[:, i], (i + 1) * h, h)
        S = y[0, :]
        E = y[1, :]
        Q = y[2, :]
        I = y[3, :]
        J = y[4, :]
        C = y[5, :]
        R = y[6, :]
        D = y[7, :]
        return x, S, E, Q, I, J, C, R, D

    def get_y_0(self, infected):
        S_0 = self.N
        E_0 = 0
        Q_0 = 0
        I_0 = self.N * 10**(-7)
        J_0 = 0
        C_0 = 0
        R_0 = 0
        D_0 = 0

        y_0 = np.array([S_0,
                        E_0,
                        Q_0,
                        I_0,
                        J_0,
                        C_0,
                        R_0,
                        D_0], dtype=float).transpose()

        h = 1 / 5

        x, S, E, Q, I, J, C, R, D = self.prediction(y_0, 0, 365, h, True)

        m = np.argmax((Q + I + J + C + R + D)/2 > infected)

        y_0 = np.array([S[m],
                        E[m],
                        Q[m],
                        I[m],
                        J[m],
                        C[m],
                        R[m],
                        D[m]], dtype=float).transpose()
        return y_0

    def set_actions(self, actions):
        self.actions = actions

    def update_parameters(self, t):
        for time, action in self.actions.items():
            if (time < t) and (time > self.last_update_time):
                self.last_update_time = time
                for key, value in self.actions[time].items():
                    setattr(self, key, value)

    def get_N(self, t):
        return self.N

    def get_b(self, t):
        return self.b

    def get_e_E(self, t):
        return self.e_E

    def get_e_Q(self, t):
        return self.e_Q

    def get_e_J(self, t):
        return self.e_J

    def get_e_C(self, t):
        return self.e_C

    def get_T_E(self, t):
        return self.T_E

    def get_T_Q(self, t):
        return self.T_Q

    def get_T_I(self, t):
        return self.T_I

    def get_T_J(self, t):
        return self.T_J

    def get_T_C(self, t):
        return self.T_C

    def get_w_E(self, t):
        return self.w_E

    def get_w_Q(self, t):
        return self.w_Q

    def get_w_I(self, t):
        return self.w_I

    def get_w_J(self, t):
        return self.w_J

    def get_w_C(self, t):
        return self.w_C

    def get_mu(self, t):
        return self.mu

    def get_Pi(self, t):
        return self.Pi

    def get_pi(self, t):
        return self.pi

    def print_parameters(self):
        print(self.N)
        print(self.b)
        print(self.e_E)
        print(self.e_Q)
        print(self.e_J)
        print(self.e_C)
        print(self.T_E)
        print(self.T_Q)
        print(self.T_I)
        print(self.T_J)
        print(self.T_C)
        print(self.w_E)
        print(self.w_Q)
        print(self.w_I)
        print(self.w_J)
        print(self.w_C)
        print(self.mu)
        print(self.Pi)
        print(self.pi)

    def R_c(self):
        # The spectral radius of F*V^(-1)
        return np.abs(self.get_b(0) / (1 - self.get_w_J(0) + self.get_w_C(0) * self.get_w_J(0)) * (
                    - self.get_T_I(0) * self.get_w_E(0) +
                    self.get_T_I(0) * self.get_w_E(0) * self.get_w_J(0) -
                    self.get_T_I(0) * self.get_w_C(0) * self.get_w_E(0) * self.get_w_J(0) -
                    self.get_T_C(0) * self.get_w_I(0) * self.get_w_E(0) * self.get_w_J(0) * self.get_e_C(0) -
                    self.get_T_C(0) * self.get_w_J(0) * self.get_w_Q(0) * self.get_e_C(0) +
                    self.get_T_C(0) * self.get_w_E(0) * self.get_w_J(0) * self.get_w_Q(0) * self.get_e_C(0) -
                    self.get_T_E(0) * self.get_e_E(0) +
                    self.get_T_E(0) * self.get_w_J(0) * self.get_e_E(0) -
                    self.get_T_E(0) * self.get_w_C(0) * self.get_w_J(0) * self.get_e_E(0) -
                    self.get_T_J(0) * self.get_w_I(0) * self.get_w_E(0) * self.get_e_J(0) -
                    self.get_T_J(0) * self.get_w_Q(0) * self.get_e_J(0) +
                    self.get_T_J(0) * self.get_w_E(0) * self.get_w_Q(0) * self.get_e_J(0) -
                    self.get_T_Q(0) * self.get_e_Q(0) +
                    self.get_T_Q(0) * self.get_w_E(0) * self.get_e_Q(0) +
                    self.get_T_Q(0) * self.get_w_J(0) * self.get_e_Q(0) -
                    self.get_T_Q(0) * self.get_w_C(0) * self.get_w_J(0) * self.get_e_Q(0) -
                    self.get_T_Q(0) * self.get_w_E(0) * self.get_w_J(0) * self.get_e_Q(0) +
                    self.get_T_Q(0) * self.get_w_C(0) * self.get_w_E(0) * self.get_w_J(0) * self.get_e_Q(0)))


    def R_0(self):
        return np.abs((self.get_b(0) / (1 - self.get_w_J(0) + self.get_w_C(0) * self.get_w_J(0)) * (
                    -self.get_T_I(0) + self.get_T_I(0) * self.get_w_J(0) -
                    self.get_T_I(0) * self.get_w_C(0) * self.get_w_J(0) -
                    self.get_T_C(0) * self.get_w_I(0) * self.get_w_J(0) * self.get_e_C(0) -
                    self.get_T_E(0) * self.get_e_E(0) +
                    self.get_T_E(0) * self.get_w_J(0) * self.get_e_E(0) -
                    self.get_T_E(0) * self.get_w_C(0) * self.get_w_J(0) * self.get_e_E(0) -
                    self.get_T_J(0) * self.get_w_I(0) * self.get_e_J(0))))
