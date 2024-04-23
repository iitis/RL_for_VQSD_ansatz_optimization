import cudaq
import numpy as np
import copy

perr = [0.8, 0, 0] # Probability of occuring [X-error, Y-error, Z-error]

def controlled_if(init_kernel, meas):
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()
    # Define a function that performs certain operations on the
    # kernel and the qubit.
    def x_function():
        kernel.x(qubit)
    init_kernel.c_if(meas, x_function)


def syndrom_detection_circuit(init_kernel, toffoli_targ_qubits, toffoli_ctrl_qubits):
    """
    TASK: construct it with the RL-agent
    """
    init_kernel.cx(toffoli_targ_qubits[0], toffoli_ctrl_qubits[0])
    init_kernel.cx(toffoli_targ_qubits[1], toffoli_ctrl_qubits[0])
    init_kernel.cx(toffoli_targ_qubits[1], toffoli_ctrl_qubits[1])
    init_kernel.cx(toffoli_targ_qubits[2], toffoli_ctrl_qubits[1])
    return init_kernel

def error_induction_circuit(init_kernel, toffoli_targ_qubits):
    px,py,pz = np.random.rand(),np.random.rand(),np.random.rand()
    rand_qubit = np.random.randint(0,3)
    if px < perr[0]:
        init_kernel.x(toffoli_targ_qubits[rand_qubit])
    if py < perr[1]:
        init_kernel.y(toffoli_targ_qubits[rand_qubit])
    if pz < perr[2]:
        init_kernel.z(toffoli_targ_qubits[rand_qubit])
    return init_kernel

target = "nvidia"
cudaq.set_target(target)

def shor_5q_error_correction_code(shots):
    init_kernel = cudaq.make_kernel()
    toffoli_targ_qubits = init_kernel.qalloc(3)
    toffoli_ctrl_qubits = init_kernel.qalloc(2)

    # Random initialization
    randx, randy = np.random.random(),np.random.random()
    init_kernel.rx(randx, toffoli_targ_qubits[0])
    init_kernel.ry(randy, toffoli_targ_qubits[0])

    # Initial state measurement
    kernel_init_meas = cudaq.make_kernel()
    qbit_init_meas = kernel_init_meas.qalloc(1)
    kernel_init_meas.rx(randx, qbit_init_meas)
    kernel_init_meas.ry(randy, qbit_init_meas)
    kernel_init_meas.mz(qbit_init_meas)
    init_result = cudaq.sample(kernel_init_meas, shots_count=shots)

    # Encoding circuit (Maybe Agent!)
    init_kernel.cx(toffoli_targ_qubits[0], toffoli_targ_qubits[1])
    init_kernel.cx(toffoli_targ_qubits[0], toffoli_targ_qubits[2])

    # Error induction depending on probability (perr)
    init_kernel = error_induction_circuit(init_kernel, toffoli_targ_qubits)

    # Syndrom detection circuit (Agent!)
    init_kernel = syndrom_detection_circuit(init_kernel, toffoli_targ_qubits, toffoli_ctrl_qubits)


    # Syndrom detection measurement
    init_kernel.mz(toffoli_ctrl_qubits[0])
    init_kernel.mz(toffoli_ctrl_qubits[1])
    result = cudaq.sample(init_kernel, shots_count=shots)
    result_as_string = list(result)[0]
    
    # Syndrom correction circuit (Agent!)
    if result_as_string == '10':
        init_kernel.x(toffoli_targ_qubits[0])
    elif result_as_string == '01':
        init_kernel.x(toffoli_targ_qubits[2])
    elif result_as_string == '11':
        init_kernel.x(toffoli_targ_qubits[1])
    
    # Decoding circuit
    init_kernel.cx(toffoli_targ_qubits[0], toffoli_targ_qubits[2])
    init_kernel.cx(toffoli_targ_qubits[0], toffoli_targ_qubits[1])

    # Final state measurement
    init_kernel.mz(toffoli_targ_qubits[0])
    final_result = cudaq.sample(init_kernel, shots_count=shots)
    
    # Cost function
    cost0 = np.abs(init_result.count('0')-final_result.count('0'))
    cost1 = np.abs(init_result.count('1')-final_result.count('1'))
    cost = np.abs(cost0+cost1)/shots
    return cost

for _ in range(100):
    shor_5q_error_correction_code(10000)