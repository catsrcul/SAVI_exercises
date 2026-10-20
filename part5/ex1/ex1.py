#! /usr/bin/env python3

import sys
from matplotlib import pyplot as plt

def main():
    plt.title("Select points")
    plt.axis([-10, 10, -5, 5])
    plt.grid(True)

    positions = {'xs': [],
                 'ys' : []
                 }

    while True:
        click = plt.ginput(1) # gives coordinates in np.float64 which is not usefull for the data. we can convert is
        # to float and use it directly in a .json then.
        # print('click=' + str(click))

        x = round(float(click[0][0]),3)
        y = round(float(click[0][1]),3)

        positions['xs'].append(x)
        positions['ys'].append(y)

        plt.plot(positions['xs'], positions['ys'], 'xr')

        print('positions' + str(positions))

    plt.show()


if __name__ == "__main__":
    main()