import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv('/Users/dmitrij/main/pythonProject/Hack_T1/clinic-data.csv', delimiter=',')
data = df[['x','y']].values

a = 0.05
b = 0.02
best_ttt = 0

kmeans = KMeans(n_clusters=250, random_state=1342, n_init=50)  # Увеличено с 20 до 50
kmeans.fit(data)
centroids = kmeans.cluster_centers_

# Вычисление TTT
diff = data[:, None, :] - centroids[None, :, :]
dist = np.linalg.norm(diff, axis=2)
cost = a * dist + b * dist**2
min_cost_per_client = cost.min(axis=1)
ttt = min_cost_per_client.sum()

prev_ttt = best_ttt
no_improvement_count = 0

for iteration in range(200):  # Увеличено до 200 итераций
    # Назначаю клиентов ближайшим клиникам
    diff = data[:, None, :] - centroids[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    cost = a * dist + b * dist**2
    assignments = cost.argmin(axis=1)
    
    # Подсчет клиентов на каждую клинику
    clinic_counts = np.bincount(assignments, minlength=250)
    empty_clinics = np.where(clinic_counts == 0)[0]

    # Обновление центроидов с градиентным спуском
    new_centroids = np.zeros_like(centroids)
    for clinic_id in range(250):
        mask = assignments == clinic_id
        if mask.sum() > 0:
            clinic_data = data[mask]
            centroid = centroids[clinic_id].copy()  # Начинаем с текущей позиции
            
            # Градиентный спуск для минимизации суммы (a*d + b*d²)
            learning_rate = 0.5
            best_centroid = centroid.copy()
            best_cost = float('inf')
            
            for iter_inner in range(100):  # Увеличено до 100
                # Вычисляем градиент
                diff_vec = clinic_data - centroid
                distances = np.linalg.norm(diff_vec, axis=1)
                distances = np.maximum(distances, 1e-10)
                
                # Градиент функции стоимости для каждого клиента
                grad = np.zeros(2)
                for i in range(len(clinic_data)):
                    if distances[i] > 1e-10:
                        direction = diff_vec[i] / distances[i]
                        # Производная: d/dx (a*d + b*d²) = (a + 2*b*d) * direction
                        grad += (a + 2 * b * distances[i]) * direction
                
                grad = grad / len(clinic_data)  # Нормализуем
                
                # Обновляем позицию
                new_centroid = centroid - learning_rate * grad
                
                # Проверяем улучшение
                new_distances = np.linalg.norm(clinic_data - new_centroid, axis=1)
                new_cost = np.sum(a * new_distances + b * new_distances**2)
                
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_centroid = new_centroid.copy()
                    centroid = new_centroid
                    learning_rate = min(learning_rate * 1.05, 2.0)  # Увеличиваем шаг при успехе
                else:
                    learning_rate *= 0.7  # Уменьшаем шаг при неудаче
                    if learning_rate < 1e-6:
                        break
            
            new_centroids[clinic_id] = best_centroid
        else:
            # Улучшенная обработка пустых кластеров
            if len(empty_clinics) > 0 and clinic_id in empty_clinics:
                # Размещаем рядом с самой перегруженной клиникой
                if clinic_counts.max() > 0:
                    overloaded_id = clinic_counts.argmax()
                    # Случайное смещение от перегруженной клиники
                    offset = np.random.normal(0, 1.5, 2)
                    new_centroids[clinic_id] = centroids[overloaded_id] + offset
                    new_centroids[clinic_id] = np.clip(new_centroids[clinic_id], -1000, 1000)
                else:
                    new_centroids[clinic_id] = centroids[clinic_id]
            else:
                new_centroids[clinic_id] = centroids[clinic_id]
    
    centroids = new_centroids
    
    diff = data[:, None, :] - centroids[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    cost = a * dist + b * dist**2
    min_cost_per_client = cost.min(axis=1)
    ttt = min_cost_per_client.sum()
    
    if iteration % 20 == 0:
        print(f"Итерация {iteration}: TTT={ttt:.2f}")
    
    # Механизм "встряхивания" для выхода из локального минимума
    if ttt < prev_ttt:
        no_improvement_count = 0
    else:
        no_improvement_count += 1
        
    # Если нет улучшения 10 итераций подряд, делаем "встряхивание"
    if no_improvement_count >= 10 and iteration > 30:
        # Случайно смещаем 10% клиник для выхода из локального минимума
        shake_count = max(5, 250 // 20)
        shake_indices = np.random.choice(250, shake_count, replace=False)
        for idx in shake_indices:
            shake_amount = 0.5 * (1.0 - iteration / 200)  # Уменьшаем со временем
            centroids[idx] += np.random.normal(0, shake_amount, 2)
            centroids[idx] = np.clip(centroids[idx], -1000, 1000)
        no_improvement_count = 0
        print(f"  Встряхивание на итерации {iteration}")
    
    # Более строгое условие остановки - только если совсем не меняется
    if abs(prev_ttt - ttt) < 0.001 and iteration > 50:
        print(f"Сходимость на итерации {iteration}")
        break
    prev_ttt = ttt

print('TTT -', prev_ttt)
np.savetxt("Suetolog_23.12_version_1.csv", centroids, delimiter=",", fmt='%f', header='x,y',comments='')