import csv
import matplotlib.pyplot as plt
from itertools import cycle
with open('datosGraf.csv', 'r') as f:
    reader = csv.reader(f, delimiter = ',')

    color = cycle(('r','b','g','m','k','y','c'))
    plt.title('Comparativa modelos')
    for line in reader:

        long = len(line) - 1
        plt.plot(range(len(line[1:])),[float(x) for x in line[1:]],marker= '.', linestyle= 'solid', color=next(color), label=line[0])

    #plt.xticks(range(long),[str(x) for x in range(long,5)])
    plt.style.use('bmh')
    plt.xlabel('Instant', fontsize=14)
    plt.ylabel('Flow', fontsize=14)

    plt.legend()
    #plt.show()
    plt.savefig('Images/comparativa_modelos.png')