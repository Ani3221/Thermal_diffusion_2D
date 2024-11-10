import numpy as np
import matplotlib.pyplot as plt
import os

class F_cond_2D:
    # Physics parameters
    Lx = 20
    Ly = 20
    rhoCp = 1

    # Numerics
    nx = 200
    dx = Lx / (nx - 1)
    x = np.linspace(-Lx / 2, Lx / 2, nx)
    ny = 200
    dy = Ly / (ny - 1)
    y = np.linspace(-Ly / 2, Ly / 2, ny)
    nt = int(1e10)

    qT = np.zeros((2, np.size(x) - 1, np.size(y) - 1))
    X, Y = np.meshgrid(x, y, indexing='ij')

    def __init__(self, gradient = np.array([0, 0]), lam = 1):
        # Preprocessing / initial conditions
        self.gradient = gradient
        self.lam = lam

        self.T = np.zeros([np.size(self.x), np.size(self.y)])
        for i in range(np.size(self.x)):
            for j in range(np.size(self.y)):
                self.T[i][j] = self.x[i] * gradient[0] + self.y[j] * gradient[1]

        self.T_0 = self.T.copy()
        self.T_old = self.T.copy()

        self.lambdaa = np.zeros([np.size(self.x), np.size(self.y)])
        for i in range(np.size(self.x)):
            for j in range(np.size(self.y)):
                self.lambdaa[i][j] = lam

        self.dt_phys = self.dx ** 2 / self.rhoCp / lam / 4.5  # CFL

        plt.ion()
        self.graph = plt.pcolormesh(self.X, self.Y, self.T, vmax=10, vmin=-10)
        self.Color = plt.colorbar()
        plt.gca().set_aspect('equal')
        plt.pause(1)


    def effectiv_coef(self):
        coef = np.zeros([2, 2])

        sself = self

        sself.thermal_diffusoin_with_gragient(np.array([1, 0]))
        coef[0, 0] = - np.multiply(-self.lambdaa[:-1, :], (np.transpose(np.diff(np.transpose(self.T)) / self.dx))).mean()
        coef[0, 1] = - np.multiply(-self.lambdaa[:, :-1], (np.diff(self.T) / self.dy)).mean()
        sself.thermal_diffusoin_with_gragient(np.array([0, 1]))
        coef[1, 0] = - np.multiply(-self.lambdaa[:-1, :], (np.transpose(np.diff(np.transpose(self.T)) / self.dx))).mean()
        coef[1, 1] = - np.multiply(-self.lambdaa[:, :-1], (np.diff(self.T) / self.dy)).mean()

        print('effective coefficients:\n', coef)
        return coef

    def thermal_diffusoin_with_gragient(self, gradient):
        for i in range(np.size(self.x)):
            for j in range(np.size(self.y)):
                self.T[i][j] = self.x[i] * gradient[0] + self.y[j] * gradient[1]

        self.T_0 = self.T.copy()
        self.T_old = self.T.copy()

        self.thermal_diffusoin_result()

    def thermal_diffusoin_result(self):
        # Action / time loop
        for it in range(self.nt):
            self.T_old = self.T.copy()
            self.qT[0, :, :] = np.multiply(-self.lambdaa[:, 1:], (np.diff(self.T) / self.dy))[:-1, :]
            self.qT[1, :, :] = np.multiply(-self.lambdaa[1:, :], (np.transpose(np.diff(np.transpose(self.T)) / self.dx)))[:, :-1]

            self.T[1:-1, 1:-1] -= self.dt_phys * ((np.diff(self.qT[0, :, :]) / self.dy)[1:, :]
                                                  + (np.transpose(np.diff(np.transpose(self.qT[1, :, :])) / self.dx))[:, 1:]) / self.rhoCp

            if it % 1000 == 0:
                self.graph.remove()
                self.graph = plt.pcolormesh(self.X, self.Y, self.T, vmax=10, vmin=-10)
                plt.title(str(it + 1))
                plt.pause(0.01)

                print((abs(self.T - self.T_old).max()) / (self.T_0.max() - self.T_0.min()))
                if (abs(self.T - self.T_old).max())/(self.T_0.max() - self.T_0.min()) < 1.0e-8:
                    break

    def make_file(self):
        f = open('C:\\Users\\art20\\OneDrive\\Рабочий стол\\input.txt.txt', 'w')

        f.write(str(self.gradient[0]) + ' ' + str(self.gradient[1]) + ' ')
        f.write(str(self.lam))

        f.close()


class F_fiber_composite_2D(F_cond_2D):

    def __init__(self, gradient = np.array([0, 0]), R = 3.5, lambda_in = 2, lambda_out = 10):
        self.gradient = gradient
        self.R = R
        self.lambda_in = lambda_in
        self.lambda_out = lambda_out

        self.T = np.zeros([np.size(self.x), np.size(self.y)])
        for i in range(np.size(self.x)):
            for j in range(np.size(self.y)):
                self.T[i][j] = self.x[i] * gradient[0] + self.y[j] * gradient[1]

        self.T_0 = self.T.copy()
        self.T_old = self.T.copy()

        self.lambdaa = np.zeros([np.size(self.x), np.size(self.y)])
        for i in range(np.size(self.x)):
            for j in range(np.size(self.y)):
                if ((self.x[i] - self.x[(np.size(self.x) - 1)//2]) ** 2
                    + (self.y[j] - self.y[(np.size(self.y) - 1)//2]) ** 2) < R ** 2:
                    self.lambdaa[i][j] = lambda_out
                else:
                    self.lambdaa[i][j] = lambda_in

        self.dt_phys = self.dx ** 2 / self.rhoCp / abs(self.lambdaa).max() / 4.5  # CFL

        plt.ion()
        self.graph = plt.pcolormesh(self.X, self.Y, self.T, vmax=10, vmin=-10)
        self.Color = plt.colorbar()
        plt.gca().set_aspect('equal')
        plt.pause(1)

    def make_file(self):
        f = open('C:\\Users\\art20\\OneDrive\\Рабочий стол\\input.txt.txt', 'w')

        f.write(str(self.gradient[0]) + ' ' + str(self.gradient[1]) + ' ')
        f.write(str(self.R) + ' ')
        f.write(str(self.lambda_in) + ' ')
        f.write(str(self.lambda_out))

        f.close()


grad = np.array([0, 1])

# X = F_cond_2D(grad)
X = F_fiber_composite_2D(grad)

# F_fiber_composite_2D.make_file(X)
F_cond_2D.effectiv_coef(X)
# F_cond_2D.thermal_diffusoin_result(X)

# os.system(['nvcc boundary_problem.cu'])
# os.system(['a.exe'])
