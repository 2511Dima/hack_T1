import pandas as pd
import numpy as np

df = pd.read_csv('/Users/dmitrij/main/pythonProject/Hack_T1/clinic-data.csv', delimiter=',')

x = list()
y = list()
for x1 in df['x']:
    x.append(x1)
for y1 in df['y']:
    y.append(y1)

data = df[['x','y']].values

def count_ttt(centroids):
    diff = data[:, None, :] - centroids[None, :, :] #broadcasting
    dist = np.linalg.norm(diff, axis=2) #евклидово растояние
    nearest_building_idx = dist.argmin(axis=1) 
    a = 0.05
    b = 0.02

    cost = a * dist + b * dist**2
    min_cost_per_client = cost.min(axis=1)
    ttt = min_cost_per_client.sum()
    return ttt

df = pd.read_csv('/Users/dmitrij/main/pythonProject/Hack_T1/Suetolog_23.12_version_1.csv', delimiter=',')
centroids = np.array(df)
print(count_ttt(centroids))
initial_ttt = count_ttt(centroids)
best_ttt = count_ttt(centroids)
now_ttt = count_ttt(centroids)
step = 0.1

for p in range(10):
    print(f'Свинуто - {p} раз')
    for i in range(250):
        if i%10==0:
            print(i)
        for j in range(2):
            centroids[i][j] += step
            if count_ttt(centroids)<best_ttt:
                best_ttt = count_ttt(centroids)
            else:
                centroids[i][j] -= step
        for j in range(2):
            centroids[i][j] -= step
            if count_ttt(centroids)<best_ttt:
                best_ttt = count_ttt(centroids)
            else:
                centroids[i][j] += step
    print(f'{p} - {best_ttt}')
    if now_ttt==count_ttt(centroids):
        step = step/2
        continue
    now_ttt = count_ttt(centroids)

print(f'Изначальное ТТТ - {initial_ttt}\nСдвинутая ТТТ   - {best_ttt}', step)
np.savetxt("Suetolog_24.12_version_2.csv", centroids, delimiter=",", fmt='%f', header='x,y',comments='')