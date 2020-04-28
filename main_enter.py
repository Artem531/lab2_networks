import random

import numpy as np
import matplotlib.pyplot as plt
from math import factorial as fac

def binomial(x, y):
    try:
        binom = fac(x) // fac(y) // fac(x - y)
    except ValueError:
        binom = 0
    return binom

def find(m):
    for i, val in enumerate(m):
        if val == 1:
            return len(m) - i - 1
    return 0

def div(a, mod):
    for i, val in enumerate(a):
        if val == 1:
            if find(a) >= find(mod):
                dif = (len(a) - i) - len(mod)
                a[i:i + len(mod) + dif] = np.bitwise_xor(a[i:i + len(mod) + dif], np.array(mod + [0] * dif))
            else:
                break
    return a

# a = np.array([1,0,1,1])
# b = np.array([1,1,1])
# print(div(a, b))


def rotate(arr, bits):
    lenA = len(arr)
    arr = arr + [0] * bits
    return arr

def gen(m, g, r):
    #powM = find(m)
    dif = r
    shifted = np.array(rotate(m, dif))
    c = div(shifted.copy(), g)
    code = np.bitwise_xor(shifted, c)
    return code

def waiting_inf(P):
    N = 10000
    n = 7
    k = 4  # max value of bits in message
    g = [1, 0, 1, 1]
    d = 3

    SendCount = 0
    stat = [0] * N
    for i in range(N):
        m = random.randint(0, 15)
        m = list('{0:b}'.format(m))
        m = np.array(['0'] * (k - len(m)) + m).astype("int32")
        m = list(m)
        code = gen(m, g, len(g) - 1)
        NoiseLessKey = False
        Count = 0
        while NoiseLessKey != True:

            Noise = [0 for _ in range(len(code))]
            Error = P > random.random()
            Noise[random.randint(0, len(code) - 1)] = 1 if Error else 0

            code_with_noise = np.bitwise_xor(code, Noise)
            # print(code_with_noise, code, Noise)
            if np.sum(div(code_with_noise.copy(), g)) == 0:
                NoiseLessKey = True
            print(Error, NoiseLessKey)
            Count += 1
            SendCount += 1
        print("____________________")
        stat[i] = Count
    print("result mean N", SendCount / N)

    print("calculated mean N", np.sum([count * (1 - P) * P ** (count - 1) for count in range(1000)]))
    print("calculated acc mean N", 1/ (1-P))


def waiting_N(number, P):
    N = 10000
    n = 7
    k = 4  # max value of bits in message
    g = [1, 0, 1, 1]
    d = 3

    SendCount = 0
    stat = [0] * N
    for i in range(N):
        #print(i)
        m = random.randint(0, 15)
        m = list('{0:b}'.format(m))
        m = np.array(['0'] * (k - len(m)) + m).astype("int32")
        m = list(m)
        code = gen(m, g, len(g) - 1)
        NoiseLessKey = False
        Count = 0
        for _ in range(number):

            Noise = [0 for _ in range(len(code))]
            Error = P > random.random()
            Noise[random.randint(0, len(code) - 1)] = 1 if Error else 0

            code_with_noise = np.bitwise_xor(code, Noise)
            # print(code_with_noise, code, Noise)
            if np.sum(div(code_with_noise.copy(), g)) == 0:
                NoiseLessKey = True
                Count += 1
                SendCount += 1
                print(Error, NoiseLessKey)
                break
            print(Error, NoiseLessKey)
            Count += 1
            SendCount += 1
        print("____________________")
        stat[i] = Count
    print("result mean N", SendCount / N)

    P = P
    tmp = np.sum( [count * (1 - P) * P ** (count - 1) for count in range(1, number)] )
    mean = (P * number if number == 2 else number * (P**(number))) + tmp
    print("calculated mean N", mean )
    print("calculated acc mean N", (1 - P**number)/ (1-P))


def waiting_N_with_error(number, P, PB):
    N = 10000
    n = 7
    k = 4  # max value of bits in message
    g = [1, 0, 1, 1]
    d = 3

    SendCount = 0
    stat = [0] * N
    for i in range(N):
        #print(i)
        m = random.randint(0, 15)
        m = list('{0:b}'.format(m))
        m = np.array(['0'] * (k - len(m)) + m).astype("int32")
        m = list(m)
        code = gen(m, g, len(g) - 1)
        NoiseLessKey = False
        Count = 0
        for _ in range(number):

            Noise = [0 for _ in range(len(code))]
            Error = P > random.random()
            Noise[random.randint(0, len(code) - 1)] = 1 if Error else 0

            code_with_noise = np.bitwise_xor(code, Noise)
            # print(code_with_noise, code, Noise)
            ErrorBack = PB > random.random()
            if np.sum(div(code_with_noise.copy(), g)) == 0 and not ErrorBack:
                NoiseLessKey = True
                Count += 1
                SendCount += 1
                print(Error, NoiseLessKey)
                break

            print(Error, NoiseLessKey)
            Count += 1
            SendCount += 1
        print("____________________")
        stat[i] = Count
    print("result mean N", SendCount / N)
    p_false = P*PB + P*(1-PB) + (1-P)*PB
    p_true = (1-P)*(1-PB)

    tmp = np.sum( [count * p_true * p_false ** (count - 1) for count in range(1, number)] )
    mean = (p_false * number if number == 2 else number * (p_false**(number))) + tmp
    print("calculated mean N", mean )
    print("calculated acc mean N", (1 - p_false**number)/ (1-p_false))

def waiting_inf_with_error(number, P, PB):
    N = 10000
    n = 7
    k = 4  # max value of bits in message
    g = [1, 0, 1, 1]
    d = 3

    SendCount = 0
    stat = [0] * N
    for i in range(N):
        #print(i)
        m = random.randint(0, 15)
        m = list('{0:b}'.format(m))
        m = np.array(['0'] * (k - len(m)) + m).astype("int32")
        m = list(m)
        code = gen(m, g, len(g) - 1)
        NoiseLessKey = False
        Count = 0
        while not NoiseLessKey:

            Noise = [0 for _ in range(len(code))]
            Error = P > random.random()
            Noise[random.randint(0, len(code) - 1)] = 1 if Error else 0

            code_with_noise = np.bitwise_xor(code, Noise)
            # print(code_with_noise, code, Noise)
            ErrorBack = PB > random.random()
            if np.sum(div(code_with_noise.copy(), g)) == 0 and not ErrorBack:
                NoiseLessKey = True
                Count += 1
                SendCount += 1
                print(Error, NoiseLessKey)
                break

            print(Error, NoiseLessKey)
            Count += 1
            SendCount += 1
        print("____________________")
        stat[i] = Count
    print("result mean N", SendCount / N)
    p_false = P*PB + P*(1-PB) + (1-P)*PB
    p_true = (1-P)*(1-PB)

    print("calculated mean N", np.sum([count * p_true * p_false ** (count - 1) for count in range(1000)]) )
    print("calculated acc mean N", 1 / (1 - p_false))

def back_waiting_inf_with_tao(P, tao):
    N = 1000
    n = 7
    k = 4  # max value of bits in message
    g = [1, 0, 1, 1]
    d = 3
    save_arr = [-1] * (tao + 1)
    save_code = [-1] * (tao + 1)

    SendCount = 0
    stat = [0] * N
    tao_idx = -1
    messIdx = -1

    while True:
        #print(i)
        tao_idx = (tao_idx + 1) % (tao + 1)
        if tao == 0:
            tao_idx = 0


        if messIdx > N and np.sum(np.array(save_arr).astype(int)) == (tao + 1):
            break

        if save_arr[tao_idx] == True:
            messIdx += 1

        if (save_arr[tao_idx] == True or save_arr[tao_idx] == -1) and messIdx < N:

            m = random.randint(0, 15)
            m = list('{0:b}'.format(m))
            m = np.array(['0'] * (k - len(m)) + m).astype("int32")
            m = list(m)
            code = gen(m, g, len(g) - 1)
        else:
            if save_arr[tao_idx] == True:
                continue
            save_arr = [save_arr[i] if i < tao_idx else False for i in range(len(save_arr))]
            code = save_code[tao_idx]

        save_code[tao_idx] = code
        Noise = [0 for _ in range(len(code))]
        Error = P > random.random()

        #save_arr[tao_idx, 1] =
        Noise[random.randint(0, len(code) - 1)] = 1 if Error else 0
        code_with_noise = np.bitwise_xor(code, Noise)
        save_arr[tao_idx] = np.sum(div(code_with_noise.copy(), g)) == 0 # save error state

        # print(code_with_noise, code, Noise)
        SendCount += 1
        #print("____________________")

    print("result n", N / SendCount)
    print("calculated n", (1 - P) / (1 + P * tao))


def waiting_inf_with_tao(P, tao):
    N = 100
    n = 7
    k = 4  # max value of bits in message
    g = [1, 0, 1, 1]
    d = 3
    save_arr = [-1] * (tao + 1)
    save_code = [-1] * (tao + 1)

    SendCount = 0
    stat = [0] * N
    tao_idx = -1
    messIdx = -1
    Error = False
    while True:

        if messIdx > N and not Error:
            break

        if not Error:
            messIdx += 1

        if not Error and messIdx < N:

            m = random.randint(0, 15)
            m = list('{0:b}'.format(m))
            m = np.array(['0'] * (k - len(m)) + m).astype("int32")
            m = list(m)
            code = gen(m, g, len(g) - 1)

        Noise = [0 for _ in range(len(code))]
        Error_key = P > random.random()

        #save_arr[tao_idx, 1] =
        Noise[random.randint(0, len(code) - 1)] = 1 if Error_key else 0
        code_with_noise = np.bitwise_xor(code, Noise)
        Error = np.sum(div(code_with_noise.copy(), g)) != 0
        SendCount += (tao + 1)
        #print("____________________")

    print("result n", N / SendCount)
    print("calculated n", (1 - P) / (1 + tao))




def main():
    number = 10
    #waiting_inf(0.7)
    #waiting_N(number, 0.7)
    #waiting_N_with_error(number, 0.9, 0.4)
    waiting_inf_with_error(number, 0.5, 0.5)
    #back_waiting_inf_with_tao(0.4, 10)
    #waiting_inf_with_tao(0.9, 5)

if __name__ == '__main__':
    main()
