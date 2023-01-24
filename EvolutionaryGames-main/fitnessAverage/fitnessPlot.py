import matplotlib.pyplot as plt
import numpy as np
Averagefitnesses = []

f = open('helicopter.txt','r')
for row in f:
    row = row.split(' ')
    Averagefitnesses.append(float(row[2]))

#print(Averagefitnesses)
y = np.array(Averagefitnesses)
x = np.arange(1, int(len(Averagefitnesses) + 1))
plt.bar(x,y, color = 'g', label = 'File Data')

plt.xlabel('Generations', fontsize = 12)
plt.ylabel('Fitness Averag', fontsize = 12)

plt.title('Evaluation Chart ', fontsize = 20)
plt.legend()
plt.show()