"""
🔢 Алгоритмы сортировки на Python
Визуализация времени работы каждого алгоритма.

Алгоритмы:
- Bubble Sort     (пузырьковая)
- Selection Sort  (выбором)
- Insertion Sort  (вставками)
- Merge Sort      (слиянием)
- Quick Sort      (быстрая)
- Heap Sort       (пирамидальная)
- Counting Sort   (подсчётом)
- Radix Sort      (поразрядная)
"""

import random
import time
import matplotlib.pyplot as plt



# 1. BUBBLE SORT — Пузырьковая сортировка
# Идея: сравниваем соседние элементы и меняем местами если не в порядке
# Сложность: O(n²) — медленная, но простая для понимания

def bubble_sort(arr):
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr



# 2. SELECTION SORT — Сортировка выбором
# Идея: находим минимальный элемент и ставим на нужное место
# Сложность: O(n²)
 
def selection_sort(arr):
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr



# 3. INSERTION SORT — Сортировка вставками
# Идея: берём элемент и вставляем его на правильное место среди уже отсортированных
# Сложность: O(n²), но быстрая на почти отсортированных данных

def insertion_sort(arr):
    arr = arr.copy()
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr



# 4. MERGE SORT — Сортировка слиянием
# Идея: делим массив пополам, сортируем каждую половину, сливаем обратно
# Сложность: O(n log n) — одна из лучших

def merge_sort(arr):
    arr = arr.copy()
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return _merge(left, right)


def _merge(left, right):
    """Вспомогательная функция — сливает два отсортированных массива."""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result



# 5. QUICK SORT — Быстрая сортировка
# Идея: выбираем опорный элемент (pivot), делим массив на меньшие и большие
# Сложность: O(n log n) в среднем, O(n²) в худшем случае

def quick_sort(arr):
    arr = arr.copy()
    _quick_sort_helper(arr, 0, len(arr) - 1)
    return arr


def _quick_sort_helper(arr, low, high):
    if low < high:
        pivot_idx = _partition(arr, low, high)
        _quick_sort_helper(arr, low, pivot_idx - 1)
        _quick_sort_helper(arr, pivot_idx + 1, high)


def _partition(arr, low, high):
    """Разделяем массив относительно опорного элемента."""
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1



# 6. HEAP SORT — Пирамидальная сортировка
# Идея: строим кучу (heap), затем извлекаем максимум по одному
# Сложность: O(n log n)

def heap_sort(arr):
    arr = arr.copy()
    n = len(arr)

    # Строим максимальную кучу
    for i in range(n // 2 - 1, -1, -1):
        _heapify(arr, n, i)

    # Извлекаем элементы из кучи по одному
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        _heapify(arr, i, 0)

    return arr


def _heapify(arr, n, i):
    """Поддерживаем свойство кучи для поддерева с корнем i."""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        _heapify(arr, n, largest)



# 7. COUNTING SORT — Сортировка подсчётом
# Идея: считаем сколько раз встречается каждый элемент
# Сложность: O(n + k), где k — диапазон значений
# Работает только с целыми числами!

def counting_sort(arr):
    if not arr:
        return arr
    arr = arr.copy()
    max_val = max(arr)
    min_val = min(arr)
    range_val = max_val - min_val + 1

    count = [0] * range_val
    for num in arr:
        count[num - min_val] += 1

    result = []
    for i, c in enumerate(count):
        result.extend([i + min_val] * c)

    return result



# 8. RADIX SORT — Поразрядная сортировка
# Идея: сортируем по цифрам — сначала единицы, потом десятки, и т.д.
# Сложность: O(n * k), где k — количество цифр

def radix_sort(arr):
    if not arr:
        return arr
    arr = arr.copy()
    max_val = max(arr)

    exp = 1
    while max_val // exp > 0:
        arr = _counting_sort_by_digit(arr, exp)
        exp *= 10

    return arr


def _counting_sort_by_digit(arr, exp):
    """Сортировка по одному разряду."""
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for num in arr:
        index = (num // exp) % 10
        count[index] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    for i in range(n - 1, -1, -1):
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1

    return output



# ЗАМЕР ВРЕМЕНИ И ВИЗУАЛИЗАЦИЯ

def measure_time(sort_func, arr):
    """Замеряет время выполнения функции в миллисекундах."""
    start = time.perf_counter()
    sort_func(arr)
    end = time.perf_counter()
    return (end - start) * 1000


def run_benchmark():
    """Сравниваем все алгоритмы на разных размерах массивов."""
    sizes = [100, 500, 1000, 2000, 3000]

    algorithms = {
        "Bubble Sort":     bubble_sort,
        "Selection Sort":  selection_sort,
        "Insertion Sort":  insertion_sort,
        "Merge Sort":      merge_sort,
        "Quick Sort":      quick_sort,
        "Heap Sort":       heap_sort,
        "Counting Sort":   counting_sort,
        "Radix Sort":      radix_sort,
    }

    results = {name: [] for name in algorithms}

    print("🔢 Сравнение алгоритмов сортировки")
    print("=" * 60)
    print(f"{'Алгоритм':<20}", end="")
    for size in sizes:
        print(f"  n={size}", end="")
    print()
    print("-" * 60)

    for name, func in algorithms.items():
        print(f"{name:<20}", end="")
        for size in sizes:
            arr = [random.randint(0, 10000) for _ in range(size)]
            t = measure_time(func, arr)
            results[name].append(t)
            print(f"  {t:5.1f}ms", end="")
        print()

    print("=" * 60)
    print("\n📊 Строю график...")

    # Строим график
    plt.figure(figsize=(12, 7))

    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71",
              "#1abc9c", "#3498db", "#9b59b6", "#34495e"]

    for (name, times), color in zip(results.items(), colors):
        plt.plot(sizes, times, marker="o", label=name, color=color, linewidth=2)

    plt.title("Сравнение алгоритмов сортировки", fontsize=16, fontweight="bold")
    plt.xlabel("Размер массива (n)", fontsize=13)
    plt.ylabel("Время (мс)", fontsize=13)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("benchmark.png", dpi=150)
    plt.show()
    print("✅ График сохранён в benchmark.png")


def verify_algorithms():
    """Проверяем что все алгоритмы сортируют правильно."""
    print("\n✅ Проверка корректности алгоритмов:")
    test = [random.randint(0, 100) for _ in range(20)]
    expected = sorted(test)

    algorithms = {
        "Bubble Sort":    bubble_sort,
        "Selection Sort": selection_sort,
        "Insertion Sort": insertion_sort,
        "Merge Sort":     merge_sort,
        "Quick Sort":     quick_sort,
        "Heap Sort":      heap_sort,
        "Counting Sort":  counting_sort,
        "Radix Sort":     radix_sort,
    }

    for name, func in algorithms.items():
        result = func(test)
        status = "✅" if result == expected else "❌"
        print(f"  {status} {name}")


if __name__ == "__main__":
    verify_algorithms()
    run_benchmark()"""
🔢 Алгоритмы сортировки на Python
Визуализация времени работы каждого алгоритма.

Алгоритмы:
- Bubble Sort     (пузырьковая)
- Selection Sort  (выбором)
- Insertion Sort  (вставками)
- Merge Sort      (слиянием)
- Quick Sort      (быстрая)
- Heap Sort       (пирамидальная)
- Counting Sort   (подсчётом)
- Radix Sort      (поразрядная)
"""

import random
import time
import matplotlib.pyplot as plt



# 1. BUBBLE SORT — Пузырьковая сортировка
# Идея: сравниваем соседние элементы и меняем местами если не в порядке
# Сложность: O(n²) — медленная, но простая для понимания

def bubble_sort(arr):
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr



# 2. SELECTION SORT — Сортировка выбором
# Идея: находим минимальный элемент и ставим на нужное место
# Сложность: O(n²)
 
def selection_sort(arr):
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr



# 3. INSERTION SORT — Сортировка вставками
# Идея: берём элемент и вставляем его на правильное место среди уже отсортированных
# Сложность: O(n²), но быстрая на почти отсортированных данных

def insertion_sort(arr):
    arr = arr.copy()
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr



# 4. MERGE SORT — Сортировка слиянием
# Идея: делим массив пополам, сортируем каждую половину, сливаем обратно
# Сложность: O(n log n) — одна из лучших

def merge_sort(arr):
    arr = arr.copy()
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return _merge(left, right)


def _merge(left, right):
    """Вспомогательная функция — сливает два отсортированных массива."""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result



# 5. QUICK SORT — Быстрая сортировка
# Идея: выбираем опорный элемент (pivot), делим массив на меньшие и большие
# Сложность: O(n log n) в среднем, O(n²) в худшем случае

def quick_sort(arr):
    arr = arr.copy()
    _quick_sort_helper(arr, 0, len(arr) - 1)
    return arr


def _quick_sort_helper(arr, low, high):
    if low < high:
        pivot_idx = _partition(arr, low, high)
        _quick_sort_helper(arr, low, pivot_idx - 1)
        _quick_sort_helper(arr, pivot_idx + 1, high)


def _partition(arr, low, high):
    """Разделяем массив относительно опорного элемента."""
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1



# 6. HEAP SORT — Пирамидальная сортировка
# Идея: строим кучу (heap), затем извлекаем максимум по одному
# Сложность: O(n log n)

def heap_sort(arr):
    arr = arr.copy()
    n = len(arr)

    # Строим максимальную кучу
    for i in range(n // 2 - 1, -1, -1):
        _heapify(arr, n, i)

    # Извлекаем элементы из кучи по одному
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        _heapify(arr, i, 0)

    return arr


def _heapify(arr, n, i):
    """Поддерживаем свойство кучи для поддерева с корнем i."""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        _heapify(arr, n, largest)



# 7. COUNTING SORT — Сортировка подсчётом
# Идея: считаем сколько раз встречается каждый элемент
# Сложность: O(n + k), где k — диапазон значений
# Работает только с целыми числами!

def counting_sort(arr):
    if not arr:
        return arr
    arr = arr.copy()
    max_val = max(arr)
    min_val = min(arr)
    range_val = max_val - min_val + 1

    count = [0] * range_val
    for num in arr:
        count[num - min_val] += 1

    result = []
    for i, c in enumerate(count):
        result.extend([i + min_val] * c)

    return result



# 8. RADIX SORT — Поразрядная сортировка
# Идея: сортируем по цифрам — сначала единицы, потом десятки, и т.д.
# Сложность: O(n * k), где k — количество цифр

def radix_sort(arr):
    if not arr:
        return arr
    arr = arr.copy()
    max_val = max(arr)

    exp = 1
    while max_val // exp > 0:
        arr = _counting_sort_by_digit(arr, exp)
        exp *= 10

    return arr


def _counting_sort_by_digit(arr, exp):
    """Сортировка по одному разряду."""
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for num in arr:
        index = (num // exp) % 10
        count[index] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    for i in range(n - 1, -1, -1):
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1

    return output



# ЗАМЕР ВРЕМЕНИ И ВИЗУАЛИЗАЦИЯ

def measure_time(sort_func, arr):
    """Замеряет время выполнения функции в миллисекундах."""
    start = time.perf_counter()
    sort_func(arr)
    end = time.perf_counter()
    return (end - start) * 1000


def run_benchmark():
    """Сравниваем все алгоритмы на разных размерах массивов."""
    sizes = [100, 500, 1000, 2000, 3000]

    algorithms = {
        "Bubble Sort":     bubble_sort,
        "Selection Sort":  selection_sort,
        "Insertion Sort":  insertion_sort,
        "Merge Sort":      merge_sort,
        "Quick Sort":      quick_sort,
        "Heap Sort":       heap_sort,
        "Counting Sort":   counting_sort,
        "Radix Sort":      radix_sort,
    }

    results = {name: [] for name in algorithms}

    print("🔢 Сравнение алгоритмов сортировки")
    print("=" * 60)
    print(f"{'Алгоритм':<20}", end="")
    for size in sizes:
        print(f"  n={size}", end="")
    print()
    print("-" * 60)

    for name, func in algorithms.items():
        print(f"{name:<20}", end="")
        for size in sizes:
            arr = [random.randint(0, 10000) for _ in range(size)]
            t = measure_time(func, arr)
            results[name].append(t)
            print(f"  {t:5.1f}ms", end="")
        print()

    print("=" * 60)
    print("\n📊 Строю график...")

    # Строим график
    plt.figure(figsize=(12, 7))

    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71",
              "#1abc9c", "#3498db", "#9b59b6", "#34495e"]

    for (name, times), color in zip(results.items(), colors):
        plt.plot(sizes, times, marker="o", label=name, color=color, linewidth=2)

    plt.title("Сравнение алгоритмов сортировки", fontsize=16, fontweight="bold")
    plt.xlabel("Размер массива (n)", fontsize=13)
    plt.ylabel("Время (мс)", fontsize=13)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("benchmark.png", dpi=150)
    plt.show()
    print("✅ График сохранён в benchmark.png")


def verify_algorithms():
    """Проверяем что все алгоритмы сортируют правильно."""
    print("\n✅ Проверка корректности алгоритмов:")
    test = [random.randint(0, 100) for _ in range(20)]
    expected = sorted(test)

    algorithms = {
        "Bubble Sort":    bubble_sort,
        "Selection Sort": selection_sort,
        "Insertion Sort": insertion_sort,
        "Merge Sort":     merge_sort,
        "Quick Sort":     quick_sort,
        "Heap Sort":      heap_sort,
        "Counting Sort":  counting_sort,
        "Radix Sort":     radix_sort,
    }

    for name, func in algorithms.items():
        result = func(test)
        status = "✅" if result == expected else "❌"
        print(f"  {status} {name}")


if __name__ == "__main__":
    verify_algorithms()
    run_benchmark()
