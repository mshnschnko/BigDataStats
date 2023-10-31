# # import numpy as np

# # h = 0.1
# # k = 500
# # k_list = [i for i in range(k+1)]

# # nd = np.random.standard_normal(k+1)
# # x_orig = np.array([0.5*np.sin(i*h) for i in range(k+1)])

# # x = x_orig + nd
# # print(x[:10])

# # import matplotlib.pyplot as plt

# # fig, ax = plt.subplots()
# # ax.plot(k_list, x)
# # ax.grid()
# # plt.xlabel("$k$")
# # plt.ylabel("$x_k$")
# # plt.title("Выборка с шумом")
# # ratio = 0.5
# # x_left, x_right = ax.get_xlim()
# # y_low, y_high = ax.get_ylim()
# # ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
# # # plt.axis('equal')
# # plt.show()

# # def ema(data, a):
# #     y = []
# #     y.append((data[0] + data[1])/2)
# #     for i in range(1, k+1):
# #         y.append(a*data[i] + (1-a)*y[i-1])
# #     return y


# # ema_list = []
# # for a in [0.01, 0.05, 0.1, 0.3]:
# #     ema_list.append(np.array(ema(x, a)))

# # plt.plot(k_list, x_orig, label='trend')
# # plt.scatter(k_list, x, label='$x_k$', s=[1]*len(k_list))
# # for i, a in enumerate([0.01, 0.05, 0.1, 0.3]):
# #     plt.plot(k_list, ema_list[i], label=f'ema $\\alpha$={a}', linewidth=1)
# # plt.grid()
# # plt.legend()
# # plt.xlabel("$k$")
# # plt.ylabel("$x_k$")
# # plt.title("Простое скользящее среднее")
# # plt.show()

# import numpy as np
# import scipy as sc
# import scipy.stats as st
# import matplotlib.pyplot as plt

# # globals
# N = 1000
# h = 0.1

# # Function to generate sample
# # Return sample from task 1
# def generateSample():
#     return np.array([0.5 * np.sin(k * h) + np.random.normal() for k in range(N)])

# # Function to generate model
# def generateModel():
#     return np.array([0.5 * np.sin(k * h) for k in range(N)])

# # Function to calculate slide exp mean points
# # In: sample, alpha
# # Out: array of mean points
# def slideExp(sample, alpha):
#     res = []
#     res.append(sample[0])
#     for i in range(1, sample.size):
#         res.append(alpha * sample[i] + (1 - alpha) * res[i - 1])
#     return res

# # Function to calculate rotation points
# # In: sample
# # Out: rotation point array
# def calculateRotationPoints(sample):
#     res = []
#     for i in range(1, len(sample) - 2):
#         if (sample[i] > sample[i - 1] and sample[i] > sample[i + 1]) or (sample[i] < sample[i - 1] and sample[i] < sample[i + 1]):
#             res.append(sample[i])
#     return res

# # Function to check randomness with Kandell
# # In: sample, trend
# def checkKandell(sample, trend):
#     #plt.figure()
#     #plt.title("Task 4")
#     #plt.plot(sample,label = "tail")
#     #plt.legend()

#     tail = sample - trend
#     rotationPoints = calculateRotationPoints(tail)

#     pMean = (2.0 / 3.0) * (len(sample) - 2)
#     pDisp = (16 * len(sample) - 29) / 90.0
#     pSize = len(rotationPoints)

#     print("Calculated rotation number's sum: ", pSize)
#     if pSize < pMean + pDisp and pSize > pMean - pDisp:
#         print("Randomness")
#     elif pSize > pMean + pDisp:
#         print("Rapidly oscillating")
#     elif pSize < pMean - pDisp:
#         print("Positively correlated")

#     # Check for mean and normal
#     print('mean ', tail.mean())
#     print('standart devotion ', st.tstd(tail))
#     # 0.05 or 0.005
#     print('probability of Normal = ', st.normaltest(tail)[1])


# if __name__ == "__main__":
#     # Task 1
#     print("Task 1: \n")
#     sample = generateSample()
#     print(sample)
#     plt.figure()
#     plt.title("Task 1")
#     plt.plot(sample, 'o', color = 'black', label = "sample")
#     plt.legend()

#     # Task 2
#     print("Task 2: \n")
#     model = generateModel()
#     slide001 = slideExp(sample, 0.01)
#     slide005 = slideExp(sample, 0.05)
#     slide01 = slideExp(sample, 0.1)
#     slide03 = slideExp(sample, 0.3)
#     plt.figure()
#     plt.title("Task 2")
#     plt.plot(sample, 'o', color = 'black', label = "sample")
#     plt.plot(model, label = "model")
#     plt.plot(slide001, label = "Slide exp mean, a = 0.01")
#     plt.plot(slide005, label = "Slide exp mean, a = 0.05")
#     plt.plot(slide01, label = "Slide exp mean, a = 0.1")
#     plt.plot(slide03, label = "Slide exp mean, a = 0.3")
#     plt.legend()

#     # Task 4
#     print("Task 4: \n")
#     plt.figure()
#     plt.title("Task 4")
#     f = np.fft.fft(sample)
#     ampSpec = 2 / N * np.abs(f[:len(sample) // 2])
#     freqs = np.linspace(0, 1 / (2.0), len(sample) // 2)
#     plt.plot(freqs, ampSpec)
#     freq = freqs[np.argmax(ampSpec)]
#     print(freq)

#     # Task 5
#     print("Task 5: \n")
#     print("Kandell for slide exp mean 0.01")
#     checkKandell(slide001, sample)
#     print("\nKandell for slide exp mean 0.05")
#     checkKandell(slide005, sample)
#     print("\nKandell for slide exp mean 0.1")
#     checkKandell(slide01, sample)
#     print("\nKandell for slide exp mean 0.3")
#     checkKandell(slide03, sample)

#     plt.show()


import numpy as np

h = 0.1
k = 500
k_list = [i for i in range(k+1)]

nd = np.random.standard_normal(k+1)
x_orig = np.array([0.5*np.sin(i*h) for i in range(k+1)])

x = x_orig + nd
print(x[:10])

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(k_list, x)
ax.grid()
plt.xlabel("$k$")
plt.ylabel("$x_k$")
plt.title("Выборка с шумом")
ratio = 0.5
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
# plt.axis('equal')
plt.show()

def ema(data, a):
    y = []
    y.append((data[0] + data[1])/2)
    for i in range(1, k+1):
        y.append(a*data[i] + (1-a)*y[i-1])
    return y


ema_list = []
for a in [0.01, 0.05, 0.1, 0.3]:
    ema_list.append(np.array(ema(x, a)))

plt.plot(k_list, x_orig, label='trend')
plt.scatter(k_list, x, label='$x_k$', s=[1]*len(k_list))
for i, a in enumerate([0.01, 0.05, 0.1, 0.3]):
    plt.plot(k_list, ema_list[i], label=f'ema $\\alpha$={a}', linewidth=1)
plt.grid()
plt.legend()
plt.xlabel("$k$")
plt.ylabel("$x_k$")
plt.title("Экспоненциальное скользящее среднее")
plt.show()

def FFT(x):
    fft = np.fft.fft(x)
    # a: np.complex128 = [1, 2]
    # print(a)
    abs_fft = []
    for i in range(len(fft)):
        # abs_fft.append(fft[i].real)
        abs_fft.append(abs(fft[i]))
    return np.array(abs_fft)

fft_x = FFT(x)
fft_list = []
for em in ema_list:
    fft_list.append(FFT(em))

# plt.yscale("log")
ordi = np.linspace(0, 0.5, len(k_list)//2)
plt.plot(ordi, fft_x[:len(fft_x)//2]/k, label='FFT(x)')
for i, a in enumerate([0.01, 0.05, 0.1, 0.3]):
    plt.plot(ordi, fft_list[i][:len(fft_list[i])//2]/k, label=f'FFT(ema) $\\alpha$={a}', linewidth=1)
# plt.plot(k_list[:len(k_list)//2], fft_list[0][:len(k_list)//2], label=f'FFT(ema) $\\alpha$={0.01}', linewidth=1)
plt.grid()
plt.legend()
plt.xlabel("$k$")
plt.ylabel("$x_k$")
plt.title("Преобразование Фурье")
plt.show()

noise_ema = np.empty((len(ema_list), len(ema_list[0])))
for i in range(len(ema_list)):
    noise_ema[i] = x - ema_list[i]

for i, a in enumerate([0.01, 0.05, 0.1, 0.3]):
    plt.plot(k_list, noise_ema[i], label='$\\Delta x_k$')
    plt.grid()
    plt.legend()
    plt.xlabel("$k$")
    plt.ylabel("$\\text{noise}\ \\alpha = $"+f"{a}")
    plt.title("")
    plt.show()

def turning_points(data):
    p = 0
    for i in range(len(data) - 2):
        if data[i] < data[i+1] and data[i+1] > data[i+2]:
            p += 1
        if data[i] > data[i+1] and data[i+1] < data[i+2]:
            p += 1
    return p

def kendall_coefficient(data, n):
    p = 0
    for i in range(len(data) - 1):
        for j in range(i+1, len(data)):
            if data[j] > data[i]:
                p += 1
    return 4*p/(n*(n-1)) - 1

E_t = 0
D_t = 2*(2*(k+1)+5)/(9*(k+1)*k)
s_t = np.sqrt(D_t)

print("E_t = ", E_t)
print("D_t = ", D_t)
print("s_t = ", s_t)

for i, a in enumerate([0.01, 0.05, 0.1, 0.3]):
    print("Среднее = ", noise_ema[i].mean())
    t = kendall_coefficient(noise_ema[i], k+1)
    print(f"t_{i} = {t}")
    if t > E_t + s_t:
        print("Возрастяющий тренд")
    elif t < E_t - s_t:
        print("Убывающий тренд")
    else:
        print("Ряд случаен")