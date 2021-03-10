import matplotlib.pyplot as plt
import numpy as np

def plot_digit_grid(data, target, digit):
    number_index = (target == digit)
    digit_array = data[number_index]
    num_digits = digit_array.shape[0]

    columns = 5
    rows = 5

    indexs = np.random.randint(0, num_digits, rows*columns)
    
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, columns*rows +1):
        index = indexs[i-1]
        img = digit_array[index].reshape(28, 28)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='gray')
    plt.show()