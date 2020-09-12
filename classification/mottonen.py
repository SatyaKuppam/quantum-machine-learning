## Adapted PyQuil's arbitrary_state operator for cirq
## for amplitude encoding. The implementation is based
## on Mottonen et al.
## https://grove-docs.readthedocs.io/en/latest/arbitrary_state.html

import numpy as np
import cirq
from six.moves import input


def get_uniformly_controlled_rotation_matrix(k):
    M = np.full((2 ** k, 2 ** k), 2 ** -k)
    for i in range(2 ** k):
        g_i = i ^ (i >> 1) 
        for j in range(2 ** k):
            M[i, j] *= (-1) ** (bin(j & g_i).count("1"))
    return M


def get_cnot_control_positions(k):
    rotation_cnots = [1, 1]
    for i in range(2, k + 1):
        rotation_cnots[-1] = i
        rotation_cnots = rotation_cnots + rotation_cnots
    return rotation_cnots


def get_rotation_parameters(phases, magnitudes):
    z_thetas = []
    y_thetas = []
    new_phases = []
    new_magnitudes = []

    for i in range(0, len(phases), 2):
        phi = phases[i]
        psi = phases[i + 1]
        z_thetas.append(phi - psi)
        kappa = (phi + psi) / 2.
        new_phases.append(kappa)
        a = magnitudes[i]
        b = magnitudes[i + 1]
        if a == 0 and b == 0:
            y_thetas.append(0)
        else:
            y_thetas.append(
                2 * np.arcsin((a - b) / (np.sqrt(2 * (a ** 2 + b ** 2)))))
        c = np.sqrt((a ** 2 + b ** 2) / 2.)
        new_magnitudes.append(c)

    return z_thetas, y_thetas, new_phases, new_magnitudes


def get_reversed_unification_program(angles, control_indices,
                                     target, controls, mode):
    if mode == 'phase':
        gate = cirq.rz
    elif mode == 'magnitude':
        gate = cirq.ry
    else:
        raise ValueError("mode must be \'phase\' or \'magnitude\'")

    reversed_gates = []

    for j in range(len(angles)):
        if angles[j] != 0:
            reversed_gates.append(gate(-angles[j])(target))
        if len(controls) > 0:
            reversed_gates.append(cirq.CNOT(controls[control_indices[j] - 1],
                                       target))

    return cirq.Circuit(reversed_gates[::-1])


def encode_classical_datapoint(vector, qubits=None):
    vec_norm = vector / np.linalg.norm(vector)
    n = max(1, int(np.ceil(np.log2(len(vec_norm)))))

    if qubits is None:
        qubits = cirq.LineQubit.range(n)

    N = 2 ** n 
    while len(vec_norm) < N:
        vec_norm = np.append(vec_norm, 0)

    magnitudes = list(map(np.abs, vec_norm))
    phases = list(map(np.angle, vec_norm))
    M = get_uniformly_controlled_rotation_matrix(n - 1)
    rotation_cnots = get_cnot_control_positions(n - 1)
    reversed_prog = cirq.Circuit()

    for step in range(n):
        reversed_step_prog = cirq.Circuit()
        z_thetas, y_thetas, phases, magnitudes = \
            get_rotation_parameters(phases, magnitudes)

        converted_z_thetas = np.dot(M, z_thetas)
        converted_y_thetas = np.dot(M, y_thetas)
        phase_prog = get_reversed_unification_program(converted_z_thetas,
                                                      rotation_cnots,
                                                      qubits[0],
                                                      qubits[step + 1:],
                                                      'phase')
        prob_prog = get_reversed_unification_program(converted_y_thetas,
                                                     rotation_cnots,
                                                     qubits[0],
                                                     qubits[step + 1:],
                                                     'magnitude')
        if step < n - 1:
            reversed_step_prog.append(cirq.SWAP(qubits[0], qubits[step + 1]))
            M = M[0:int(len(M) / 2), 0:int(len(M) / 2)] * 2
            rotation_cnots = rotation_cnots[:int(len(rotation_cnots) / 2)]
            rotation_cnots[-1] -= 1

        reversed_step_prog.append(prob_prog)
        reversed_step_prog.append(phase_prog)
        reversed_prog.append(reversed_step_prog)
    reversed_prog.append(cirq.H(i) for i in qubits)
    reversed_prog.append([cirq.rz(-2 * phases[0])(qubits[0]), cirq.ZPowGate(exponent=2 * phases[0]/np.pi)(qubits[0])])
    return reversed_prog


if __name__ == "__main__":
    v = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    v = np.array([0.41917652, 0.90790474][::-1])
    qubits = cirq.LineQubit.range(4)
    print(v)
    p = encode_classical_datapoint(v, qubits[:1])
    wf = cirq.Simulator().simulate(p)
    print("Generated Wavefunction: ", wf)
    print(p)
