# main.py
"""
Полный цикл решения задачи линейного программирования.

Демонстрирует все 7 шагов:
1. Считывание задачи из файла
2. Приведение к каноническому виду
3. Формирование вспомогательной задачи
4. Решение вспомогательной задачи
5. Переход к основной задаче
6. Решение основной задачи
7. Запись результата в файл
"""

from simplex import SimplexSolver, read_problem_from_file, write_solution_to_file

try:
    # Шаг 1: Считывание текстового файла с постановкой ЗЛП
    print("\n[Шаг 1] Считывание файла input.txt...")
    num_vars, constraints, objective, free_variables = read_problem_from_file('input.txt')
    print("+ Файл успешно прочитан")

    # Вывод информации о задаче
    print(f"\nЦелевая функция: {objective[0]} z = {objective[1]}")
    print(f"Количество переменных: {num_vars}")
    print(f"Количество ограничений: {len(constraints)}")
    if free_variables:
        print(f"Свободные переменные: {', '.join([f'x_{i}' for i in free_variables])}")
    print("\nОграничения:")
    for i, constraint in enumerate(constraints, 1):
        print(f"  {i}. {constraint}")

    # Шаг 2-6: Решение задачи (выполняется внутри SimplexSolver)
    print("\n[Шаг 2] Приведение задачи к каноническому виду...")
    print("[Шаг 3-4] Формирование и решение вспомогательной задачи...")
    print("[Шаг 5] Переход к основной задаче...")
    print("[Шаг 6] Решение основной задачи...")

    solver = SimplexSolver(num_vars=num_vars,
                           constraints=constraints,
                           objective_function=objective,
                           free_variables=free_variables)

    print("+ Оптимальное решение найдено")

    # Вывод результата в консоль
    print("\n" + "=" * 60)
    print("Результат")
    print("=" * 60)
    print("\nОптимальная точка x*:")
    for var_name in sorted(solver.solution.keys(), key=lambda x: int(x.split('_')[1])):
        value = solver.solution[var_name]
        print(f"  {var_name} = {float(value):.6f}")

    print(f"\nЗначение целевой функции W(x*): {float(solver.optimal_value):.6f}")
    print(f"Количество итераций: {solver.iteration_count}")

    # ШАГ 7: Запись ответа в файл
    print("\n[Шаг 7] Запись результата в файл output.txt...")
    write_solution_to_file(solver, 'output.txt')
    print("+ Результат записан в файл")

    print("\n" + "=" * 60)
    print("Успешное завершение")
    print("=" * 60)

except ValueError as e:
    # Обработка ошибок (несовместная задача, неограниченная задача и т.д.)
    print("\n" + "=" * 60)
    print("Ошибка")
    print("=" * 60)
    print(f"\n{e}")
    print("\nРешений нет.")
    print("=" * 60)

except FileNotFoundError:
    print("\n- Ошибка: Файл input.txt не найден")
    print("Создайте файл input.txt с описанием задачи.")

except Exception as e:
    print(f"\n- Неожиданная ошибка: {e}")

