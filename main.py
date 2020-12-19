import itertools
import random as rand
import numpy as np

# 0 = f(w0 + x1w1 + x2w2)
# f(x) = 1 , se x > 0
# f(x) = -1 , se x <= 0

combinations = [[-1, -1], [-1, 1], [1, -1], [1, 1]]

# AND
d_response = [-1, -1, -1, 1]


# XOR
# d_response = [-1, 1, 1, -1]


def first():
    w0 = rand.random() / 10
    w1 = rand.random() / 10
    w2 = rand.random() / 10
    all_epoch = 0
    values_epoch = []
    for y in range(30):
        alpha = 0.001
        delta_w0 = 0
        delta_w1 = 0
        delta_w2 = 0
        stop = None
        number_epoch = 0

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
            print("EPOCH DONE")
            w0 = delta_w0
            w1 = delta_w1
            w2 = delta_w2
            number_epoch += 1
            if len(errors_0) == 4:
                print("STOP!!!")
                stop = "STOP!"
        print("NUMBER OF EPOCH:  " + str(number_epoch))
        all_epoch += number_epoch
        values_epoch.append(number_epoch)
    average_epoch = all_epoch / 30
    print("MEDIA DE EPOCH:   " + str(average_epoch))
    deviation = np.std(values_epoch)
    print("DESVIO PADRÃO:  " + str(deviation))


def first_guess(values, w0, w1, w2):
    x1 = values[0]
    x2 = values[1]

    x = w0 + (x1 * w1) + (x2 * w2)

    if x > 0:
        return 1

    else:
        return -1


def ten_guesses(values, weights_list):
    value = 0
    w0 = weights_list[0]
    value += w0

    for x in range(len(values)):
        value += (values[x] * weights_list[x + 1])

    return value


def make_list_combinations():
    combinations_10_all = list(map(list, itertools.product([0, 1], repeat=2)))
    combinations_10_mid = list()

    for x in range(4):
        add = rand.choice(combinations_10_all)
        combinations_10_mid.append(add)

    return combinations_10_mid


def make_list_resposnse():
    d_response_10 = list()
    for x in range(4):
        k = rand.randint(0, 1)

        if k == 0:
            d_response_10.append(-1)
        else:
            d_response_10.append(k)

    return d_response_10


def ten_inputs():
    weights_list = []
    list_inputs = make_list_combinations()
    list_response = make_list_resposnse()
    all_epoch = 0
    values_epoch = []

    for x in range(3):
        w = rand.random() / 10
        weights_list.append(w)
    for t in range(30):
        alpha = 0.1
        values_delta = np.zeros(len(weights_list))
        stop = None
        number_epoch = 0

        while stop is None:
            errors_0 = []
            errors = []

            for tt in range(len(list_inputs) - 1):
                values = list_inputs[tt]
                expected_result = list_response[tt]

                real_result = ten_guesses(values, weights_list)

                error = expected_result - real_result
                errors.append(error)
                if error == 0:
                    print("FOUND ONE")
                    errors_0.append(error)

                values_delta[0] += alpha * error
                for xx in range(len(values)):
                    delta_change = alpha * values[xx] * error
                    values_delta[xx + 1] += delta_change
            print(errors)
            for tt in range(len(weights_list)):
                weights_list[tt] += values_delta[tt]
            number_epoch += 1
            if len(errors_0) == len(list_inputs):
                print("STOP!!!")
                stop = "STOP!"

        all_epoch += number_epoch
        values_epoch.append(number_epoch)
    average_epoch = all_epoch / 30
    print("MEDIA DE EPOCH:   " + str(average_epoch))
    deviation = np.std(values_epoch)
    print("DESVIO PADRÃO:  " + str(deviation))


ten_inputs()
