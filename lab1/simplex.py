from fractions import Fraction
from warnings import warn


def read_problem_from_file(filename):
    """
    ШАГ 1: Считывание текстового файла с постановкой ЗЛП.
    
    Формат файла:
        max/min: целевая_функция
        constraints:
            ограничение_1
            ограничение_2
            ...
    
    Аргументы:
        filename: путь к файлу с описанием задачи
    
    Возвращает:
        кортеж (num_vars, constraints, objective_function)
    """
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    
    lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
    
    objective_type = None
    objective_expr = None
    constraints = []
    num_vars = 0
    in_constraints = False
    
    for line in lines:
        if line.startswith('max') or line.startswith('min'):
            parts = line.split(':', 1)
            objective_type = parts[0].strip()
            objective_expr = parts[1].strip() if len(parts) > 1 else ""
            
            # Подсчёт количества переменных
            var_indices = []
            for token in objective_expr.split():
                if '_' in token:
                    try:
                        _, idx = token.split('_')
                        var_indices.append(int(idx))
                    except:
                        pass
            num_vars = max(var_indices) if var_indices else 0
            
        elif line.lower() == 'constraints:':
            in_constraints = True
            
        elif in_constraints:
            constraints.append(line)
            # Обновляем количество переменных
            var_indices = []
            for token in line.split():
                if '_' in token:
                    try:
                        _, idx = token.split('_')
                        var_indices.append(int(idx))
                    except:
                        pass
            if var_indices:
                num_vars = max(num_vars, max(var_indices))
    
    if not objective_type or not objective_expr:
        raise ValueError("Не найдена целевая функция в файле")
    
    if not constraints:
        raise ValueError("Не найдены ограничения в файле")
    
    return num_vars, constraints, (objective_type, objective_expr)


def write_solution_to_file(solver, filename):
    """
    ШАГ 7: Запись ответа в файл.
    
    Записывает:
    - Оптимальную точку x* ∈ X
    - Значение целевой функции W(x*)
    - Или информацию о том, что решений нет с указанием причины
    
    Аргументы:
        solver: экземпляр SimplexSolver с решением
        filename: путь к выходному файлу
    """
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(f"Тип задачи: {solver.objective_type.upper()}\n")
        file.write(f"Целевая функция: {solver.objective_expression}\n\n")
        
        # Оптимальная точка x*
        file.write("Оптимальная точка x*:\n")
        for var_name in sorted(solver.solution.keys(), key=lambda x: int(x.split('_')[1])):
            value = solver.solution[var_name]
            file.write(f"  {var_name} = {float(value):.6f}\n")
        
        # Значение целевой функции W(x*)
        file.write(f"\nЗначение целевой функции W(x*): {float(solver.optimal_value):.6f}\n")
        
        # Дополнительная информация
        file.write(f"\nКоличество итераций: {solver.iteration_count}\n")


# Константы
OBJECTIVE_ROW = 0  # Индекс строки целевой функции в таблице
FIRST_CONSTRAINT_ROW = 1  # Индекс первой строки ограничений
RHS_COLUMN = -1  # Индекс столбца правой части (последний столбец)
EPSILON = 1e-10  # Малая величина для проверки делителя
MAX_ITERATIONS = 1000  # Максимальное количество итераций для предотвращения зацикливания


class SimplexSolver:
    """
    Решатель задач линейного программирования двухфазным симплекс-методом.
    
    Поддерживает:
    - Задачи максимизации и минимизации
    - Ограничения типов <=, >=, =
    - Автоматическое добавление дополнительных и искусственных переменных
    - Обнаружение несовместных и неограниченных задач
    - Защиту от зацикливания (ограничение по итерациям)
    """
    
    def __init__(self, num_vars, constraints, objective_function, max_iterations=MAX_ITERATIONS):
        """
        Инициализация решателя.
        
        Аргументы:
            num_vars: количество переменных в задаче
            constraints: список ограничений в виде строк (формат: "2x_1 + 3x_2 <= 10")
            objective_function: кортеж (тип, выражение), где тип - 'max' или 'min'
            max_iterations: максимальное количество итераций (по умолчанию 1000)
        """
        self.num_vars = num_vars
        self.constraints = constraints
        self.objective_type = objective_function[0]
        self.objective_expression = objective_function[1]
        self.max_iterations = max_iterations
        self.iteration_count = 0  # Счётчик итераций
        
        # ШАГ 2: Приведение задачи к каноническому виду
        # Построение симплекс-таблицы из ограничений с добавлением slack и искусственных переменных
        self.coefficient_matrix, self.artificial_rows, self.num_slack_vars, self.num_artificial_vars = (
            self._construct_initial_matrix()
        )
        
        # Список базисных переменных (индексы)
        self.basic_variables = [0 for _ in range(len(self.coefficient_matrix))]
        
        # ШАГ 3 и ШАГ 4: Формирование и решение вспомогательной задачи
        # ФАЗА 1: поиск начального допустимого базисного решения
        self._phase_one()
        
        # ШАГ 5: Переход к основной задаче (если есть возможность)
        # Проверка допустимости: все искусственные переменные должны быть нулевыми
        self._check_feasibility()
        
        # Удаление искусственных переменных из таблицы
        self._remove_artificial_variables()
        
        # ШАГ 6: Решение основной задачи
        # ФАЗА 2: оптимизация целевой функции
        if 'min' in self.objective_type.lower():
            self.solution = self._minimize_objective()
        else:
            self.solution = self._maximize_objective()
        
        # Оптимальное значение целевой функции
        self.optimal_value = self.coefficient_matrix[OBJECTIVE_ROW][RHS_COLUMN]
    
    def _construct_initial_matrix(self):
        """
        Построение начальной симплекс-таблицы из ограничений.
        
        Добавляет дополнительные (slack/surplus) и искусственные переменные
        в зависимости от типа ограничения:
        - <= : добавляется slack переменная
        - >= : добавляется surplus переменная (-1) и искусственная (+1)
        - =  : добавляется искусственная переменная
        
        Возвращает:
            coefficient_matrix: симплекс-таблица
            artificial_rows: индексы строк с искусственными переменными
            num_slack_vars: количество дополнительных переменных
            num_artificial_vars: количество искусственных переменных
        """
        num_slack_vars = 0
        num_artificial_vars = 0
        
        # Подсчёт количества дополнительных переменных
        for constraint in self.constraints:
            if '>=' in constraint:
                num_slack_vars += 1
                num_artificial_vars += 1
            elif '<=' in constraint:
                num_slack_vars += 1
            elif '=' in constraint:
                num_artificial_vars += 1
        
        total_vars = self.num_vars + num_slack_vars + num_artificial_vars
        
        # Инициализация таблицы (строки: целевая функция + ограничения, столбцы: переменные + правая часть)
        coefficient_matrix = [
            [Fraction("0/1") for _ in range(total_vars + 1)]
            for _ in range(len(self.constraints) + 1)
        ]
        
        slack_index = self.num_vars
        artificial_index = self.num_vars + num_slack_vars
        artificial_rows = []
        
        # Заполнение таблицы из ограничений
        for row_idx in range(1, len(self.constraints) + 1):
            constraint_parts = self.constraints[row_idx - 1].split(' ')
            
            # Обработка коэффициентов переменных
            for part_idx in range(len(constraint_parts)):
                part = constraint_parts[part_idx]
                
                # Формат: "3x_1", разделяется на коэффициент и индекс
                if '_' in part:
                    coeff_str, var_index_str = part.split('_')
                    var_index = int(var_index_str) - 1
                    coeff = coeff_str[:-1]  # убираем 'x'
                    
                    # Проверка знака (если предыдущий элемент - минус)
                    if constraint_parts[part_idx - 1] == '-':
                        coefficient_matrix[row_idx][var_index] = Fraction(f"-{coeff}/1")
                    else:
                        coefficient_matrix[row_idx][var_index] = Fraction(f"{coeff}/1")
                
                # Обработка типа ограничения
                elif part == '<=':
                    # Добавляем slack переменную
                    coefficient_matrix[row_idx][slack_index] = Fraction("1/1")
                    slack_index += 1
                
                elif part == '>=':
                    # Добавляем surplus переменную (-1) и искусственную (+1)
                    coefficient_matrix[row_idx][slack_index] = Fraction("-1/1")
                    coefficient_matrix[row_idx][artificial_index] = Fraction("1/1")
                    slack_index += 1
                    artificial_index += 1
                    artificial_rows.append(row_idx)
                
                elif part == '=':
                    # Добавляем искусственную переменную
                    coefficient_matrix[row_idx][artificial_index] = Fraction("1/1")
                    artificial_index += 1
                    artificial_rows.append(row_idx)
            
            # Правая часть ограничения (последний элемент)
            coefficient_matrix[row_idx][-1] = Fraction(f"{constraint_parts[-1]}/1")
        
        return coefficient_matrix, artificial_rows, num_slack_vars, num_artificial_vars
    
    def _phase_one(self):
        """
        ФАЗА 1: Поиск начального допустимого базисного решения.
        
        Решает вспомогательную задачу минимизации суммы искусственных переменных.
        Если минимум = 0, то найдено допустимое решение исходной задачи.
        Если минимум > 0, то исходная задача несовместна.
        """
        # Формирование целевой функции фазы 1: min (сумма искусственных переменных)
        artificial_vars_start_index = self.num_vars + self.num_slack_vars
        
        for artificial_col_idx in range(artificial_vars_start_index, len(self.coefficient_matrix[0]) - 1):
            self.coefficient_matrix[0][artificial_col_idx] = Fraction("-1/1")
        
        # Обнуление коэффициентов при базисных переменных в целевой строке
        for row_idx in self.artificial_rows:
            self.coefficient_matrix[0] = self._add_rows(
                self.coefficient_matrix[0],
                self.coefficient_matrix[row_idx]
            )
            self.basic_variables[row_idx] = artificial_vars_start_index
            artificial_vars_start_index += 1
        
        # Установка начальных базисных переменных (slack переменные)
        slack_index = self.num_vars
        for row_idx in range(1, len(self.basic_variables)):
            if self.basic_variables[row_idx] == 0:
                self.basic_variables[row_idx] = slack_index
                slack_index += 1
        
        # Выполнение итераций симплекс-метода
        self._run_simplex_iterations(maximize=False)
    
    def _check_feasibility(self):
        """
        Проверка допустимости решения после фазы 1.
        
        Все искусственные переменные должны быть нулевыми (или небазисными).
        Если хотя бы одна искусственная переменная в базисе - задача несовместна.
        """
        artificial_vars_start_index = self.num_vars + self.num_slack_vars
        
        # Проверяем наличие искусственных переменных в базисе
        for basis_var_index in self.basic_variables:
            if basis_var_index >= artificial_vars_start_index:
                raise ValueError(
                    "Задача не имеет допустимых решений (несовместная система). "
                    f"Искусственная переменная x_{basis_var_index + 1} осталась в базисе."
                )
    
    def _run_simplex_iterations(self, maximize=True):
        """
        Основной цикл симплекс-метода.
        
        Выполняет итерации до достижения оптимального решения или превышения лимита итераций.
        
        Аргументы:
            maximize: True для максимизации, False для минимизации
        
        Исключения:
            ValueError: если превышено максимальное количество итераций
        """
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Выбор входящей переменной (ведущего столбца)
            if maximize:
                pivot_col = self._find_min_index(self.coefficient_matrix[OBJECTIVE_ROW])
                should_continue = self.coefficient_matrix[OBJECTIVE_ROW][pivot_col] < 0
            else:
                pivot_col = self._find_max_index(self.coefficient_matrix[OBJECTIVE_ROW])
                should_continue = self.coefficient_matrix[OBJECTIVE_ROW][pivot_col] > 0
            
            if not should_continue:
                # Оптимальное решение найдено
                self.iteration_count += iteration
                break
            
            # Выбор выходящей переменной (ведущей строки) по минимальному отношению
            pivot_row = self._find_pivot_row(pivot_col)
            
            # Обновление базиса
            self.basic_variables[pivot_row] = pivot_col
            
            # Нормализация ведущей строки
            pivot_element = self.coefficient_matrix[pivot_row][pivot_col]
            self._normalize_row(pivot_row, pivot_element)
            
            # Обнуление ведущего столбца (кроме ведущей строки)
            self._eliminate_column(pivot_col, pivot_row)
        else:
            # Цикл завершился по превышению лимита итераций
            raise ValueError(
                f"Превышено максимальное количество итераций ({self.max_iterations}). "
                "Возможно зацикливание. Попробуйте увеличить max_iterations."
            )
    
    def _find_pivot_row(self, pivot_col):
        """
        Поиск ведущей строки по минимальному отношению (min ratio test).
        
        Для каждой строки вычисляется отношение правой части к элементу ведущего столбца.
        Выбирается строка с минимальным положительным отношением.
        
        Аргументы:
            pivot_col: индекс ведущего столбца
        
        Возвращает:
            индекс ведущей строки
        
        Исключения:
            ValueError: если задача неограничена (нет положительных элементов в столбце)
        """
        min_ratio = float("inf")
        pivot_row_idx = 0
        
        # Перебираем все строки ограничений (пропускаем строку целевой функции)
        for row_idx in range(FIRST_CONSTRAINT_ROW, len(self.coefficient_matrix)):
            element = self.coefficient_matrix[row_idx][pivot_col]
            
            # Учитываем только положительные элементы
            if element > 0:
                ratio = self.coefficient_matrix[row_idx][RHS_COLUMN] / element
                
                if ratio < min_ratio:
                    min_ratio = ratio
                    pivot_row_idx = row_idx
        
        # Если не нашли ни одного положительного элемента - задача неограничена
        if min_ratio == float("inf"):
            raise ValueError(
                "Целевая функция неограничена (задача не имеет конечного решения). "
                f"В ведущем столбце {pivot_col} нет положительных элементов."
            )
        
        # Предупреждение о вырождении (базисная переменная равна нулю)
        if min_ratio == 0:
            warn(
                "Обнаружено вырождение базисного решения (базисная переменная = 0). "
                "Возможно замедление сходимости или зацикливание."
            )
        
        return pivot_row_idx
    
    def _normalize_row(self, row_idx, pivot_element):
        """
        Нормализация строки делением на ведущий элемент.
        
        Делит все элементы строки на ведущий элемент, чтобы сделать ведущий элемент равным 1.
        
        Аргументы:
            row_idx: индекс строки
            pivot_element: ведущий элемент (делитель)
        
        Исключения:
            ValueError: если ведущий элемент слишком мал (близок к нулю)
        """
        # Проверка на деление на очень малое число (может вызвать численную нестабильность)
        if abs(pivot_element) < EPSILON:
            raise ValueError(
                f"Ведущий элемент слишком мал ({pivot_element}). "
                "Возможна численная нестабильность. Попробуйте другую задачу или метод."
            )
        
        # Нормализация всех элементов строки
        for col_idx in range(len(self.coefficient_matrix[row_idx])):
            self.coefficient_matrix[row_idx][col_idx] /= pivot_element
    
    def _eliminate_column(self, pivot_col, pivot_row):
        """
        Обнуление элементов ведущего столбца (кроме ведущей строки).
        
        Применяет элементарные преобразования строк: из каждой строки (кроме ведущей)
        вычитается ведущая строка, умноженная на соответствующий коэффициент.
        Это делает все элементы ведущего столбца нулевыми, кроме ведущей строки.
        
        Аргументы:
            pivot_col: индекс ведущего столбца
            pivot_row: индекс ведущей строки
        """
        num_cols = len(self.coefficient_matrix[OBJECTIVE_ROW])
        
        # Обрабатываем все строки таблицы
        for row_idx in range(len(self.coefficient_matrix)):
            # Пропускаем ведущую строку
            if row_idx != pivot_row:
                # Коэффициент, который нужно обнулить
                multiplier = self.coefficient_matrix[row_idx][pivot_col]
                
                # Вычитаем из строки ведущую строку, умноженную на multiplier
                for col_idx in range(num_cols):
                    self.coefficient_matrix[row_idx][col_idx] -= (
                        self.coefficient_matrix[pivot_row][col_idx] * multiplier
                    )
    
    def _remove_artificial_variables(self):
        """
        Удаление искусственных переменных из симплекс-таблицы после фазы 1.
        """
        target_length = self.num_vars + self.num_slack_vars + 1
        
        for row in self.coefficient_matrix:
            while len(row) != target_length:
                del row[target_length - 1]
    
    def _update_objective_row(self):
        """
        Обновление строки целевой функции для фазы 2.
        
        Парсит выражение целевой функции и заполняет коэффициенты.
        """
        objective_parts = self.objective_expression.split()
        
        for part_idx in range(len(objective_parts)):
            part = objective_parts[part_idx]
            
            if '_' in part:
                coeff_str, var_index_str = part.split('_')
                var_index = int(var_index_str) - 1
                coeff = coeff_str[:-1]  # убираем 'x'
                
                # Для симплекс-метода коэффициенты целевой функции берутся с противоположным знаком
                if objective_parts[part_idx - 1] == '-':
                    self.coefficient_matrix[0][var_index] = Fraction(f"{coeff}/1")
                else:
                    self.coefficient_matrix[0][var_index] = Fraction(f"-{coeff}/1")
    
    def _check_alternate_solution(self):
        """
        Проверка наличия альтернативных оптимальных решений.
        
        Если в оптимальном решении коэффициент небазисной переменной в целевой строке
        равен нулю, это означает, что можно ввести эту переменную в базис без изменения
        значения целевой функции, получив тем самым альтернативное оптимальное решение.
        """
        # Проверяем все переменные (кроме столбца правой части)
        for col_idx in range(len(self.coefficient_matrix[OBJECTIVE_ROW]) - 1):
            # Если переменная небазисная и её коэффициент в целевой строке равен 0
            if (self.coefficient_matrix[OBJECTIVE_ROW][col_idx] == 0 and 
                col_idx not in self.basic_variables[FIRST_CONSTRAINT_ROW:]):
                print("✓ Обнаружено альтернативное оптимальное решение")
                print(f"  (Переменная x_{col_idx + 1} может быть введена в базис)")
                break

    def _minimize_objective(self):
        """
        ФАЗА 2: Минимизация целевой функции.
        
        Устанавливает коэффициенты целевой функции и выполняет итерации
        симплекс-метода для поиска минимума.
        
        Возвращает:
            словарь с оптимальным решением вида {'x_1': значение, 'x_2': значение, ...}
        """
        # Устанавливаем коэффициенты целевой функции из исходной задачи
        self._update_objective_row()
        
        # Обнуление коэффициентов при базисных переменных в целевой строке
        # (приведение целевой строки к каноническому виду)
        for row_idx, col_idx in enumerate(self.basic_variables[FIRST_CONSTRAINT_ROW:], start=FIRST_CONSTRAINT_ROW):
            if self.coefficient_matrix[OBJECTIVE_ROW][col_idx] != 0:
                # Вычитаем строку ограничения, умноженную на коэффициент
                correction_row = self._multiply_row(
                    -self.coefficient_matrix[OBJECTIVE_ROW][col_idx],
                    self.coefficient_matrix[row_idx]
                )
                self.coefficient_matrix[OBJECTIVE_ROW] = self._add_rows(
                    self.coefficient_matrix[OBJECTIVE_ROW],
                    correction_row
                )
        
        # Выполнение итераций симплекс-метода (для минимизации ищем максимальный положительный)
        self._run_simplex_iterations(maximize=False)
        
        # Формирование и возврат решения
        return self._extract_solution()
    
    def _maximize_objective(self):
        """
        ФАЗА 2: Максимизация целевой функции.
        
        Устанавливает коэффициенты целевой функции и выполняет итерации
        симплекс-метода для поиска максимума.
        
        Возвращает:
            словарь с оптимальным решением вида {'x_1': значение, 'x_2': значение, ...}
        """
        # Устанавливаем коэффициенты целевой функции из исходной задачи
        self._update_objective_row()
        
        # Обнуление коэффициентов при базисных переменных в целевой строке
        # (приведение целевой строки к каноническому виду)
        for row_idx, col_idx in enumerate(self.basic_variables[FIRST_CONSTRAINT_ROW:], start=FIRST_CONSTRAINT_ROW):
            if self.coefficient_matrix[OBJECTIVE_ROW][col_idx] != 0:
                # Вычитаем строку ограничения, умноженную на коэффициент
                correction_row = self._multiply_row(
                    -self.coefficient_matrix[OBJECTIVE_ROW][col_idx],
                    self.coefficient_matrix[row_idx]
                )
                self.coefficient_matrix[OBJECTIVE_ROW] = self._add_rows(
                    self.coefficient_matrix[OBJECTIVE_ROW],
                    correction_row
                )
        
        # Выполнение итераций симплекс-метода (для максимизации ищем минимальный отрицательный)
        self._run_simplex_iterations(maximize=True)
        
        # Формирование и возврат решения
        return self._extract_solution()
    
    def _extract_solution(self):
        """
        Извлечение оптимального решения из симплекс-таблицы.
        
        Базисные переменные получают значения из столбца правой части,
        небазисные переменные равны нулю.
        
        Возвращает:
            словарь вида {'x_1': значение, 'x_2': значение, ...}
        """
        solution = {}
        
        # Извлекаем значения базисных переменных из столбца правой части
        for row_idx, var_idx in enumerate(self.basic_variables[FIRST_CONSTRAINT_ROW:], start=FIRST_CONSTRAINT_ROW):
            # Учитываем только исходные переменные (не slack/artificial)
            if var_idx < self.num_vars:
                solution[f'x_{var_idx + 1}'] = self.coefficient_matrix[row_idx][RHS_COLUMN]
        
        # Небазисные переменные равны нулю
        for var_idx in range(self.num_vars):
            if var_idx not in self.basic_variables[FIRST_CONSTRAINT_ROW:]:
                solution[f'x_{var_idx + 1}'] = Fraction("0/1")
        
        # Проверка наличия альтернативных решений
        self._check_alternate_solution()

        return solution

    # Вспомогательные методы для работы со строками
    
    @staticmethod
    def _add_rows(row1, row2):
        """Покомпонентное сложение двух строк."""
        return [row1[i] + row2[i] for i in range(len(row1))]
    
    @staticmethod
    def _multiply_row(constant, row):
        """Умножение строки на константу."""
        return [constant * element for element in row]
    
    @staticmethod
    def _find_max_index(row):
        """Поиск индекса максимального элемента в строке (кроме последнего)."""
        max_idx = 0
        for i in range(len(row) - 1):
            if row[i] > row[max_idx]:
                max_idx = i
        return max_idx
    
    @staticmethod
    def _find_min_index(row):
        """Поиск индекса минимального элемента в строке (кроме последнего)."""
        min_idx = 0
        for i in range(len(row) - 1):
            if row[i] < row[min_idx]:
                min_idx = i
        return min_idx