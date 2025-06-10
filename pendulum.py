import numpy as np

def runge_kutta(theta0, L, m, g, tmax, dt):
    N = int(tmax / dt)
    t = np.linspace(0, tmax, N)
    theta = np.zeros(N)
    omega = np.zeros(N)
    theta[0] = theta0

    for i in range(N - 1):
        k1_theta = omega[i]
        k1_omega = - (g / L) * np.sin(theta[i])

        k2_theta = omega[i] + 0.5 * dt * k1_omega
        k2_omega = - (g / L) * np.sin(theta[i] + 0.5 * dt * k1_theta)

        k3_theta = omega[i] + 0.5 * dt * k2_omega
        k3_omega = - (g / L) * np.sin(theta[i] + 0.5 * dt * k2_theta)

        k4_theta = omega[i] + dt * k3_omega
        k4_omega = - (g / L) * np.sin(theta[i] + dt * k3_theta)

        theta[i+1] = theta[i] + (dt / 6) * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
        omega[i+1] = omega[i] + (dt / 6) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)

    return t, theta, omega

def calculate_energy_and_action(theta, omega, L, m, g, dt):
    T = 0.5 * m * (L * omega)**2
    V = m * g * L * (1 - np.cos(theta))
    Lagrangian = T - V
    Action = np.trapz(Lagrangian, dx=dt)
    return T, V, Lagrangian, Action
