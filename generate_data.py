import csv
import random

with open('data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['feature1', 'feature2', 'label'])

    for _ in range(1000):
        f1 = round(random.uniform(0, 100), 2)
        f2 = round(random.uniform(0, 100), 2)

        label = 1 if f1 + f2 > 100 else 0

        writer.writerow([f1, f2, label])
