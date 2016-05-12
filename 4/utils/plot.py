import matplotlib.pyplot as plt
import numpy as np

input_filename = "../data/total.txt"
input_file = open(input_filename, "r" )

size = []
cpu_chol_qr_time = []
gpu_chol_qr_time = []
cpu_qr_time = []

cpu_chol_qr_norm = []
gpu_chol_qr_norm = []
cpu_qr_norm = []


line = input_file.readline()
token = line.split()

while line != '':
    size.append(int(token[0]))

    line = input_file.readline()
    token = line.split()
    
    cpu_chol_qr_time.append(float(token[4])*1E-3)
    cpu_chol_qr_norm.append([float(token[5]), float(token[6])])
    flops = float(token[3])
    
    line = input_file.readline()
    token = line.split()

    cpu_qr_time.append(float(token[4])*1E-3)
    cpu_qr_norm.append([float(token[5]), float(token[6])])
        
    line = input_file.readline()
    token = line.split()

    gpu_chol_qr_time.append(flops / float(token[1]) * 1E-9)
    gpu_chol_qr_norm.append([float(token[5]), float(token[6])])

    line = input_file.readline()
    line = input_file.readline()
    token = line.split()

    
fig, ax = plt.subplots()
ax.plot(size, cpu_qr_time, 'ro-', label="cpu_qr")
ax.plot(size, cpu_chol_qr_time, 'go-', label="cpu_chol_qr")
ax.plot(size, gpu_chol_qr_time, 'bo-', label="gpu_chol_qr")
ax.set_yscale("log")
ax.legend(loc=2)
ax.set_xlabel("Matrix size [M, 128]")
ax.set_ylabel("GFLOPS")

fig.savefig("flops.png")

fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].plot(size, [n[0] for n in cpu_qr_norm], 'ro-', label="cpu_qr")
ax[0].plot(size, [n[0] for n in cpu_chol_qr_norm], 'go-', label="cpu_chol_qr")
ax[0].plot(size, [n[0] for n in gpu_chol_qr_norm], 'bo-', label="gpu_chol_qr")
ax[0].set_yscale("log")
ax[0].legend(loc=2)
ax[0].set_xticks(np.linspace(min(size), max(size), 5))
ax[0].set_xlabel("Matrix size [M, 128]")
ax[0].set_ylabel("Frobenius Norm A - QR")

ax[1].plot(size, [n[1] for n in cpu_qr_norm], 'ro-', label="cpu_qr")
ax[1].plot(size, [n[1] for n in cpu_chol_qr_norm], 'go-', label="cpu_chol_qr")
ax[1].plot(size, [n[1] for n in gpu_chol_qr_norm], 'bo-', label="gpu_chol_qr")
ax[1].set_yscale("log")
ax[1].legend(loc=2)
ax[1].set_xticks(np.linspace(min(size), max(size), 5))
ax[1].set_xlabel("Matrix size [M, 128]")
ax[1].set_ylabel("Frobenius Norm A - Q'Q")


fig.set_size_inches((9, 4))
fig.savefig("error.png")

