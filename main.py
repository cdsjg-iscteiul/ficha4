import itertools
import random as rand
import time

import \
    matplotlib.pyplot as plt

import numpy as np

seed = time.time()
print(seed)
rand.seed(seed)

# 0 = f(w0 + x1w1 + x2w2)
# f(x) = 1 , se x > 0
# f(x) = -1 , se x <= 0

combinations = [[-1, -1], [-1, 1], [1, -1], [1, 1]]

# AND
d_response = [-1, -1, -1, 1]

n_fig = 1


# XOR
# d_response = [-1, 1, 1, -1]


def first(alpha):
    global n_fig
    w0 = rand.random() / 10
    w1 = rand.random() / 10
    w2 = rand.random() / 10
    delta_w0 = 0
    delta_w1 = 0
    delta_w2 = 0
    all_epoch = 0
    number_epoch = 0
    values_epoch = []
    for y in range(30):
        stop = None

        while stop is None:
            errors_0 = []
            for x in range(4):
                valores = combinations[x]
                # print("PRIMEIRO INPUT:  " + str(valores))
                result_e = d_response[x]
                x_1 = valores[0]
                x_2 = valores[1]
                result_r = first_guess(valores, w0, w1, w2)

                error = result_e - result_r

                if error == 0:
                    errors_0.append(error)
                # print("EXPECTED RESULT:  " + str(result_e))
                # print("EXPECTED REAL:  " + str(result_r))
                # print("VALUE OF ERROR:   " + str(error))

                delta_w0 += alpha * error
                delta_w1 += alpha * x_1 * error
                delta_w2 += alpha * x_2 * error
            w0 += delta_w0
            w1 += delta_w1
            w2 += delta_w2
            number_epoch += 1
            if len(errors_0) == 4:
                stop = "STOP!"

        delta_w0 = 0
        delta_w1 = 0
        delta_w2 = 0
        w0 = rand.random() / 10
        w1 = rand.random() / 10
        w2 = rand.random() / 10
        print("NUMBER OF EPOCH:  " + str(number_epoch))
        all_epoch += number_epoch
        values_epoch.append(number_epoch)

    average_epoch = all_epoch / 30
    print("MEDIA DE EPOCH:   " + str(average_epoch))
    deviation = np.std(values_epoch)
    print("DESVIO PADRÃO:  " + str(deviation))

    fig, ax1 = plt.subplots(1, 1)
    ax1.boxplot(values_epoch, vert=True)
    ax1.legend(['Figure ' + str(n_fig)], handlelength=0)
    n_fig += 1
    ax1.set_title('Difference with value of alfa  ' + str(alpha))
    plt.show()


def all_alphas():
    global n_fig
    a1 = 0.0000001
    a2 = 0.000001
    a3 = 0.00001
    a4 = 0.0001
    a5 = 0.001
    a6 = 0.01

    ally = [a1, a2, a3, a4, a5, a6]

    valores = [[]] * len(ally)

    for x in range(len(ally)):
        first(ally[x])


def first_guess(values, w0, w1, w2):
    x1 = values[0]
    x2 = values[1]

    x = w0 + (x1 * w1) + (x2 * w2)

    if x > 0:
        return 1

    else:
        return -1


def ten_guesses(values, weights_list, flip):
    value = 0
    w0 = weights_list[0]
    value += w0

    for x in range(len(values)):
        value += (values[x] * weights_list[x + 1])

    if flip is True:
        chance = rand.randint(0, 1)

        if chance >= 0.5:
            print("FLIPPED")
            if value > 0:
                return -1
            else:
                return 1

        else:
            if value > 0:
                return 1
            else:
                return -1
    else:
        if value > 0:
            return 1
        else:
            return -1


def make_list_combinations():
    combinations_10_all = list(map(list, itertools.product([0, 1], repeat=10)))
    combinations_10_mid = list()

    for x in range(10):
        add = rand.choice(combinations_10_all)
        combinations_10_mid.append(add)

    return combinations_10_mid


def make_list_resposnse():
    d_response_10 = list()
    for x in range(10):
        k = rand.randint(0, 1)

        if k == 0:
            d_response_10.append(-1)
        else:
            d_response_10.append(k)

    return d_response_10


def ten_inputs(flip):
    global n_fig
    weights_list = []
    list_inputs = make_list_combinations()
    list_response = make_list_resposnse()
    all_epoch = 0
    values_epoch = []
    alpha = 0.0000001

    for x in range(11):
        w = rand.random() / 10
        weights_list.append(w)

    values_delta = np.zeros(len(weights_list))

    for t in range(30):
        stop = None
        number_epoch = 0

        while stop is None:
            errors_0 = []
            errors = []
            for tt in range(len(list_inputs)):
                values = list_inputs[tt]
                expected_result = list_response[tt]
                real_result = ten_guesses(values, weights_list, flip)

                error = expected_result - int(real_result)
                errors.append(error)
                if error == 0:
                    errors_0.append(error)

                values_delta[0] += alpha * error
                for xx in range(len(values)):
                    delta_change = alpha * values[xx] * error
                    values_delta[xx + 1] += delta_change

            for tt in range(len(weights_list)):
                weights_list[tt] += values_delta[tt]

            number_epoch += 1
            if len(errors_0) == len(list_inputs):
                stop = "STOP!"

        values_delta = np.zeros(len(weights_list))

        for x in range(len(weights_list)):
            weights_list[x] = rand.random() / 10

        all_epoch += number_epoch
        values_epoch.append(number_epoch)

    average_epoch = all_epoch / 30
    print("MEDIA DE EPOCH:   " + str(average_epoch))
    deviation = np.std(values_epoch)
    print("DESVIO PADRÃO:  " + str(deviation))
    fig, ax1 = plt.subplots(1, 1)
    ax1.boxplot(values_epoch, vert=True)
    ax1.legend(['Figure ' + str(n_fig)], handlelength=0)
    n_fig += 1
    ax1.set_title('10 intputs with alfa:  ' + str(alpha) + ' and flip: ' + str(flip))
    plt.show()


all_alphas()
ten_inputs(False)
ten_inputs(True)
