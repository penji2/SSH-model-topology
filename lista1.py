import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import cmath


def define_hamiltonian(k, t, dt):
    v = t + dt
    w = t - dt
    hamiltonian = np.array([[0, v + w*np.exp(-1j*k)],
                            [v + w*np.exp(1j*k), 0]])
    return hamiltonian


def finite_hamiltonian(n, t, dt):
    hamiltonian = np.zeros((n, n))
    for i in range(n - 1):
        if i % 2 == 0:
            hamiltonian[i][i + 1] = t + dt
            hamiltonian[i+1][i] = t + dt
        else:
            hamiltonian[i][i + 1] = t - dt
            hamiltonian[i+1][i] = t - dt
    return hamiltonian


def define_d_vector(k, t, dt):
    d_vector = np. array([t + dt + (t - dt)*np.cos(k), (t - dt)*np.sin(k), 0])
    return d_vector


def get_d_vector_for_plot(t, dt):
    d_x = []
    d_y = []
    k_array = []
    k = -np.pi
    k_step = 0.01
    while k < np.pi:
        d_x.append(define_d_vector(k, t, dt)[0])
        d_y.append(define_d_vector(k, t, dt)[1])
        k_array.append(k)
        k = k + k_step
    return d_x, d_y


def get_eigenvalues_for_plot(t, dt):
    eigenvalues_array_0 = []
    eigenvalues_array_1 = []
    k_array = []
    k = -np.pi
    k_step = 0.001
    while k < np.pi:
        eigenvalues_array_0.append(np.linalg.eigh(define_hamiltonian(k, t, dt))[0][0])
        eigenvalues_array_1.append(np.linalg.eigh(define_hamiltonian(k, t, dt))[0][1])
        k_array.append(k)
        k = k + k_step
    return k_array, eigenvalues_array_0, eigenvalues_array_1


def get_eigenvectors(t, dt):
    eigenvectors_array_negative = []
    eigenvectors_array_positive = []
    k_array = []
    k = -np.pi
    k_step = 0.001
    while k < np.pi:
        eigenvectors_for_a_given_k = np.linalg.eigh(define_hamiltonian(k, t, dt))[1]
        eigenvectors_array_negative.append(eigenvectors_for_a_given_k[:, 0])
        eigenvectors_array_positive.append(eigenvectors_for_a_given_k[:, 1])
        k_array.append(k)
        k = k + k_step
    return k_array, eigenvectors_array_negative, eigenvectors_array_positive


def compute_berry_phase(t, dt):
    k_array, eigenvectors_array_negative, eigenvectors_array_positive = get_eigenvectors(t, dt)
    berry_phase_negative = 1
    berry_phase_positive = 1
    for i in range(len(k_array) - 1):
        tmp_negative = np.vdot(eigenvectors_array_negative[i], eigenvectors_array_negative[i + 1])
        tmp_positive = np.vdot(eigenvectors_array_positive[i], eigenvectors_array_positive[i + 1])
        if tmp_positive != 0:
            berry_phase_positive *= tmp_positive / abs(tmp_positive)
        if tmp_negative != 0:
            berry_phase_negative *= tmp_negative / abs(tmp_negative)
    if dt > 0:
            print('negative:', cmath.phase(berry_phase_negative), 'positive:', cmath.phase(berry_phase_positive), ' trivial case')
    else:
        print('negative:', cmath.phase(berry_phase_negative), 'positive:', cmath.phase(berry_phase_positive),' nontrivial case')



def plot_data(xaxis, data_set1, data_set2, dt):
    label_font_size = 20
    if dt > 0:
        plt.title("Energy bands, trivial", fontsize=label_font_size)
    else:
        plt.title("Energy bands, nontrivial", fontsize=label_font_size)
    plt.xlabel("k", fontsize=label_font_size)
    plt.ylabel("E", fontsize=label_font_size)
    plt.plot(xaxis, data_set1, 'o', markersize=3, color='red')
    plt.plot(xaxis, data_set2, 'o', markersize=3, color='red')
    plt.show()


def plot_d_vector(xaxis, data_set1, dt):
    label_font_size = 20
    if dt > 0:
        plt.title("$d_y$ in the function of $d_x$, trivial case", fontsize=label_font_size)
    else:
        plt.title("$d_y$ in the function of $d_x$, nontrivial case", fontsize=label_font_size)
    plt.axvline(x=0, color='black')
    plt.axhline(y=0, color='black')
    plt.axis('equal')
    plt.xlabel("$d_x$", fontsize=label_font_size)
    plt.ylabel("$d_y$", fontsize=label_font_size)
    plt.plot(xaxis, data_set1, 'o', markersize=3, color='red')
    plt.show()


def plot_finite_hamiltonian(hamiltonian, dt):
    label_font_size = 20
    xaxis = range(1, len(hamiltonian) + 1)
    data_set = np.linalg.eigh(hamiltonian)[0]
    if dt > 0:
        plt.title("Energy states, trivial", fontsize=label_font_size)
    else:
        plt.title("Energy states, nontrivial", fontsize=label_font_size)
    plt.xlabel("I'th node", fontsize=label_font_size)
    plt.ylabel("E", fontsize=label_font_size)
    plt.plot(xaxis, data_set, 'o', markersize=3, color='blue')
    plt.show()


def plot_wave_function_density(hamiltonian):
    label_font_size = 20
    data_set1 = []
    data_set2 = []
    xaxis = range(1, len(hamiltonian) + 1)
    data_set = np.linalg.eigh(hamiltonian)[1]

    psi_1 = data_set[:, int(len(hamiltonian) / 2 - 1)]
    psi_2 = data_set[:, int(len(hamiltonian) / 2)]
    for i in range(len(hamiltonian)):
        data_set1.append(np.vdot(psi_1[i], psi_1[i]))
        data_set2.append(np.vdot(psi_2[i], psi_2[i]))

    plt.title("Wave function density", fontsize=label_font_size)
    plt.xlabel("I'th node", fontsize=label_font_size)
    plt.ylabel("$|\psi|^2$", fontsize=label_font_size)
    plt.plot(xaxis, data_set1, 'o', markersize=3, color='red')
    plt.plot(xaxis, data_set2, 'o', markersize=3, color='blue')
    plt.show()


def main():
    t = 1
    dt = 0.3
    n = 50
    k = -np.pi
    # plot_data(get_eigenvalues_for_plot(t, dt)[0], get_eigenvalues_for_plot(t, dt)[1], get_eigenvalues_for_plot(t, dt)[2], dt)
    # plot_d_vector(get_d_vector_for_plot(t, dt)[0], get_d_vector_for_plot(t, dt)[1], dt)
    compute_berry_phase(t, dt)
    # finite_hamiltonian(n, t, dt)
    # plot_finite_hamiltonian(finite_hamiltonian(n, t, dt), dt)
    # plot_wave_function_density(finite_hamiltonian(n, t, dt))

main()

