import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def gauss_algorithm(A, b):
    """
    Решает систему линейных уравнений Ax = b методом Гаусса.
    
    Аргументы:
        A (list или numpy.ndarray): Матрица коэффициентов системы (размер n x n).
        b (list или numpy.ndarray): Вектор правой части системы (размер n).
    
    Возвращает:
        x (numpy.ndarray): Решение системы уравнений (размер n).
    """
    # Размер системы (количество уравнений)
    n = len(b)
    
    # Преобразуем A и b в массивы типа float для корректных вычислений
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    # Прямой ход метода Гаусса
    for i in range(n):
        # Нормализация ведущего элемента (пивот)
        pivot = A[i, i]  # Ведущий элемент на диагонали
        if pivot == 0:  # Проверка на нулевой пивот
            raise ValueError("Нулевой пивот! Проверьте систему уравнений.")
        
        # Делим текущую строку на пивотный элемент
        A[i, :] = A[i, :] / pivot
        b[i] = b[i] / pivot
        
        # Обнуляем элементы ниже текущего ведущего элемента
        for j in range(i + 1, n):
            factor = A[j, i]  # Коэффициент для обнуления
            A[j, :] -= factor * A[i, :]  # Вычитаем текущую строку из нижней строки
            b[j] -= factor * b[i]  # Аналогично корректируем правую часть
    
    # Обратный ход метода Гаусса
    x = np.zeros(n)  # Создаём массив для решения
    for i in range(n - 1, -1, -1):  # Движемся от последней строки к первой
        # Вычисляем x[i] с учётом уже найденных значений x[i+1:], используя:
        # x[i] = b[i] - сумма(A[i, i+1:] * x[i+1:])
        x[i] = b[i] - np.sum(A[i, i+1:] * x[i+1:])
    
    # Возвращаем решение системы уравнений
    return x

def polynomial_regression(X, y, k):
    """
    Функция выполняет полиномиальную регрессию на данных.
    
    Аргументы:
        X (numpy.ndarray): Матрица входных признаков (n_samples, n_features).
        y (numpy.ndarray): Вектор целевых значений (n_samples, n_features).
        k (int): Степень полинома, которую нужно использовать.
    
    Возвращает:
        X_poly (numpy.ndarray): Преобразованная матрица входных признаков с полиномиальными признаками.
        y_pred (numpy.ndarray): Предсказанные значения целевой переменной.
        w_hat (numpy.ndarray): Коэффициенты полиномиальной регрессии.
    """
    # Создаём объект PolynomialFeatures для генерации полиномиальных признаков.
    # degree=k задаёт максимальную степень полинома.
    # include_bias=True добавляет столбец единиц (слагаемое нулевой степени).
    poly = PolynomialFeatures(degree=k, include_bias=True)
    
    # Преобразуем матрицу входных признаков X в матрицу полиномиальных признаков X_poly.
    X_poly = poly.fit_transform(X)
    
    # Рассчитываем веса (коэффициенты) w_hat с использованием нормального уравнения:
    # w_hat = (X_poly^T * X_poly)^(-1) * X_poly^T * y
    # Здесь np.linalg.inv используется для вычисления обратной матрицы.
    w_hat = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
    
    # Вычисляем предсказания y_pred, умножая матрицу полиномиальных признаков на веса.
    # y_pred = X_poly * w_hat
    y_pred = X_poly @ w_hat
    
    # Возвращаем матрицу полиномиальных признаков, предсказания и коэффициенты модели.
    return X_poly, y_pred, w_hat

def polynomial_regression_sk(X, y, k):
    """
    Выполняет полиномиальную регрессию с использованием sklearn.
    
    Аргументы:
        X (numpy.ndarray или pandas.DataFrame): Матрица входных признаков (n_samples, n_features).
        y (numpy.ndarray или pandas.DataFrame): Целевая переменная (n_samples, ).
        k (int): Степень полинома.
    
    Возвращает:
        X_poly (numpy.ndarray): Преобразованная матрица с полиномиальными признаками.
        y_pred (numpy.ndarray): Предсказанные значения целевой переменной.
        lr.coef_ (numpy.ndarray): Коэффициенты линейной регрессии.
    """
    # Генерация полиномиальных признаков степени k (без добавления bias-терма)
    poly = PolynomialFeatures(degree=k, include_bias=False)
    X_poly = poly.fit_transform(X)  # Преобразуем входные признаки в полиномиальные
    
    # Обучение линейной регрессии на полиномиальных признаках
    lr = LinearRegression().fit(X_poly, y)  # Линейная регрессия на новых признаках
    
    # Получение предсказаний на основе обученной модели
    y_pred = lr.predict(X_poly)  # Предсказание целевой переменной
    
    # Возвращаем полиномиальные признаки, предсказания и коэффициенты модели
    return X_poly, y_pred, lr.coef_