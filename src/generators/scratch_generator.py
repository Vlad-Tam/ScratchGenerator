import numpy as np
import random
from typing import Generator
import cv2

from src.entities.entities import Defect, Coordinates
from src.generators.defect_generator import DefectGenerator


class ScratchGenerator(DefectGenerator):
    def __init__(self, image_path):
        self.original_image = cv2.imread(image_path)
        self.defected_image = self.original_image.copy()
        self.defects = []

    def generate_defects(self) -> np.ndarray:
        resolution = self.defected_image.shape[1], self.defected_image.shape[0]
        lines = []
        for _ in range(random.randint(10, 30)):
            use_catmull = random.choice([True, False])
            scratch, bezier_points = self._generate_scratch(self.defected_image.copy(), self.original_image, resolution, max_length=200,
                                                      use_catmull=use_catmull, noise_level=1.0, long_line=False)
            lines.append(bezier_points)
            self.defected_image = scratch

        if random.random() < 0.5:
            long_scratch, long_bezier_points = self._generate_scratch(self.defected_image.copy(), self.original_image, resolution,
                                                                max_length=200,
                                                                use_catmull=random.choice([True, False]),
                                                                noise_level=2.0,
                                                                long_line=True)
            lines.append(long_bezier_points)
            self.defected_image = long_scratch

        if random.random() < 0.5:
            self._draw_straight_line(self.defected_image, self.original_image, resolution, noise_level=2.0)

        for i in range(len(lines)):
            if not lines[i]:
                continue
            for j in range(i + 1, len(lines)):
                if not lines[j]:
                    continue
                if self._check_intersection((lines[i][0], lines[i][-1]),
                                      (lines[j][0], lines[j][-1])) and random.random() < 0.5:
                    lines[j] = lines[j][:len(lines[j]) // 2]
                    for point in lines[j]:
                        if 0 <= point[0] < self.defected_image.shape[1] and 0 <= point[1] < self.defected_image.shape[0]:
                            if random.random() < 0.2:
                                new_color = (255, 255, 255)
                            else:
                                new_color = self._random_color()

                            if np.array_equal(point, lines[j][0]) or np.array_equal(point, lines[j][-1]):
                                if random.random() < 0.1:
                                    new_color = np.clip(np.array(new_color) * 0.8, 0, 255).astype(int)

                            new_color = tuple(map(int, new_color))
                            cv2.circle(self.defected_image, (point[0], point[1]), 2, new_color, -1)
                    break
        return self.defected_image

    def _generate_scratch(self, img: np.ndarray, orig_img: np.ndarray, resolution: tuple, max_length: float,
                         use_catmull: bool = False, noise_level: float = 1.0, long_line: bool = False) -> np.ndarray:
        multiplicator = random.uniform(0.002, 0.005)
        line_thickness = int(min(resolution) * multiplicator)  # Пропорциональная толщина
        H, W = img.shape[:2]
        x, y = np.random.uniform(0, W), np.random.uniform(0, H)

        if long_line and random.random() < 0.3:
            rho1 = np.random.uniform(150, max_length * 2)
        else:
            rho1 = np.random.uniform(50, max_length)

        theta1 = np.random.uniform(0, np.pi * 2)
        p1 = np.array([x, y, 0])
        p3 = p1 + [rho1 * np.cos(theta1), rho1 * np.sin(theta1), 0]

        rho2 = np.random.uniform(0, rho1 / 2)
        theta2 = np.random.uniform(0, np.pi * 2)
        p2 = (p1 + p3) / 2 + [rho2 * np.cos(theta2), rho2 * np.sin(theta2), 0]

        p1[2] = np.random.uniform(1, 5)
        p2[2] = np.random.uniform(1, 5)
        p3[2] = np.random.uniform(1, 5)

        bezier_points = []

        if use_catmull:
            p0 = p1
            p1 = p2
            p2 = p3
            p3 = np.array([p1[0] + 20, p1[1] + 20])
            points = self._catmull_rom(p0[:2], p1[:2], p2[:2], p3[:2], noise_level=noise_level)
            self._draw_scratch(img, orig_img, points, line_thickness)
        else:
            bezier_points = list(self._bezier(p1, p2, p3, noise_level))
            self._draw_scratch(img, orig_img, bezier_points, line_thickness)

        return img, bezier_points

    def _draw_scratch(self, img: np.ndarray, orig_img: np.ndarray, bezier_points: list, line_thickness: int) -> None:
        line_color = self._random_color()  # Генерируем случайный цвет для линий
        for i, point in enumerate(bezier_points):
            if 0 <= point[0] < img.shape[1] and 0 <= point[1] < img.shape[0]:  # Проверка границ
                if random.random() < 0.9:
                    # Уменьшаем влияние исходного цвета и увеличиваем белизну
                    new_color = line_color  # Используем случайный цвет для царапин

                    # Логика для толщины линии
                    brightness_factor = 1 + (line_thickness / 10)  # Увеличение яркости в зависимости от толщины
                    new_color = np.clip(np.array(new_color) * brightness_factor, 0, 255).astype(int)

                    # Темные пиксели по краям
                    if i == 0 or i == len(bezier_points) - 1:
                        if random.random() < 0.1:  # 10% шанс на темный цвет
                            new_color = np.clip(np.array(new_color) * 0.8, 0, 255).astype(int)

                    new_color = tuple(map(int, new_color))
                    cv2.circle(img, (point[0], point[1]), line_thickness, new_color, -1)
                else:
                    orig_color = orig_img[point[1], point[0]]  # Получаем оригинальный цвет

                    # Приводим цвет к оттенку серого
                    gray_value = self._rgb_to_gray(orig_color)
                    new_color = (gray_value, gray_value, gray_value)

                    # Плавный переход к белому цвету
                    if i > 0:
                        alpha = i / len(bezier_points)
                        new_color = self._interpolate_colors(orig_color, new_color, alpha)

                    # Логика для толщины линии
                    brightness_factor = 1 + (line_thickness / 10)
                    new_color = np.clip(np.array(new_color) * brightness_factor, 0, 255).astype(int)

                    # Темные пиксели по краям
                    if i == 0 or i == len(bezier_points) - 1:
                        if random.random() < 0.1:  # 10% шанс на темный цвет
                            new_color = np.clip(np.array(new_color) * 0.8, 0, 255).astype(int)

                    new_color = tuple(map(int, new_color))  # Преобразование в кортеж из целых чисел
                    cv2.circle(img, (point[0], point[1]), line_thickness, new_color, -1)

    def _draw_straight_line(self, img: np.ndarray, orig_img: np.ndarray, resolution: tuple, noise_level: float = 3.0):
        multiplicator = random.uniform(0.007, 0.01)
        line_thickness = int(min(resolution) * multiplicator)  # Пропорциональная толщина
        H, W = img.shape[:2]

        start_side = random.choice(['top', 'bottom', 'left', 'right'])
        end_side = random.choice(['top', 'bottom', 'left', 'right'])

        if start_side == 'top':
            start_point = (random.randint(0, W - 1), 0)
        elif start_side == 'bottom':
            start_point = (random.randint(0, W - 1), H - 1)
        elif start_side == 'left':
            start_point = (0, random.randint(0, H - 1))
        else:  # right
            start_point = (W - 1, random.randint(0, H - 1))

        if end_side == 'top':
            end_point = (random.randint(0, W - 1), 0)
        elif end_side == 'bottom':
            end_point = (random.randint(0, W - 1), H - 1)
        elif end_side == 'left':
            end_point = (0, random.randint(0, H - 1))
        else:  # right
            end_point = (W - 1, random.randint(0, H - 1))

        # Получаем направление и длину линии
        line_length = int(np.linalg.norm(np.array(end_point) - np.array(start_point)))
        direction = (end_point[0] - start_point[0], end_point[1] - start_point[1])
        step_size = int(line_thickness / 2)
        if step_size == 0:
            step_size = 1

        points = []
        for d in range(0, line_length, step_size):
            x = int(start_point[0] + d * direction[0] / line_length)
            y = int(start_point[1] + d * direction[1] / line_length)

            # Добавляем небольшие случайные шумы
            noise_x = random.uniform(-noise_level, noise_level)
            noise_y = random.uniform(-noise_level, noise_level)
            points.append((int(x + noise_x), int(y + noise_y)))

        line_color = self._random_color()

        # Вычисление границ для дефекта
        if points:
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            top_left = (min(x_coords), min(y_coords))
            bottom_right = (max(x_coords), max(y_coords))

            defect_coordinates = Coordinates(
                start={"x": top_left[0], "y": top_left[1]},
                end={"x": bottom_right[0], "y": bottom_right[1]}
            )
            defect = Defect(type="scratch", coordinates=defect_coordinates)
            self.defects.append(defect)

        for i, point in enumerate(points):
            if 0 <= point[0] < W and 0 <= point[1] < H:
                new_color = line_color  # случайный цвет для царапин

                # Логика для толщины линии
                brightness_factor = 1 + (line_thickness / 10)  # Увеличение яркости в зависимости от толщины
                new_color = np.clip(np.array(new_color) * brightness_factor, 0, 255).astype(int)

                # Проверка на темные пиксели
                if i == 0 or i == len(points) - 1:
                    if random.random() < 0.1:  # 10% шанс на темный цвет
                        new_color = np.clip(np.array(new_color) * 0.8, 0, 255).astype(int)

                new_color = tuple(map(int, new_color))
                cv2.circle(img, point, line_thickness // 2, new_color, -1)

    def _check_intersection(self, line1, line2):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        A, B = line1
        C, D = line2
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    def _interpolate_colors(self, color1, color2, alpha):
        return (int(color1[0] * (1 - alpha) + color2[0] * alpha),
                int(color1[1] * (1 - alpha) + color2[1] * alpha),
                int(color1[2] * (1 - alpha) + color2[2] * alpha))

    def _rgb_to_gray(self, rgb):
        return int(0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2])

    def _random_color(self):
        """Генерирует случайный цвет в диапазоне от (180, 180, 180) до (255, 255, 255)."""
        gray_value = random.randint(160, 255)
        return (gray_value, gray_value, gray_value)

    def _bezier(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, noise_level: float) -> Generator[
        np.ndarray, None, None]:
        def calc(t):
            return t * t * p1 + 2 * t * (1 - t) * p2 + (1 - t) * (1 - t) * p3

        approx = cv2.arcLength(np.array([calc(t)[:2] for t in np.linspace(0, 1, 10)], dtype=np.float32), False)
        points = []

        for t in np.linspace(0, 1, round(approx * 1.2)):
            point = calc(t)
            noise_x = random.uniform(-noise_level, noise_level)
            noise_y = random.uniform(-noise_level, noise_level)
            noisy_point = np.round(point + [noise_x, noise_y, 0]).astype(np.int32)
            points.append(noisy_point)

        # Вычисление границ для дефекта
        if points:
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            top_left = (min(x_coords), min(y_coords))
            bottom_right = (max(x_coords), max(y_coords))

            defect_coordinates = Coordinates(
                start={"x": top_left[0], "y": top_left[1]},
                end={"x": bottom_right[0], "y": bottom_right[1]}
            )
            defect = Defect(type="scratch", coordinates=defect_coordinates)
            self.defects.append(defect)

        yield from points

    def _catmull_rom(self, p0, p1, p2, p3, num_points=100, noise_level=1.0):
        points = []

        for t in np.linspace(0, 1, num_points):
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t +
                       (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t ** 2 +
                       (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t ** 3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t +
                       (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t ** 2 +
                       (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t ** 3)
            noise_x = random.uniform(-noise_level, noise_level)
            noise_y = random.uniform(-noise_level, noise_level)
            points.append((int(x + noise_x), int(y + noise_y)))

        # Вычисление границ для дефекта
        if points:
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            top_left = (min(x_coords), min(y_coords))
            bottom_right = (max(x_coords), max(y_coords))

            defect_coordinates = Coordinates(
                start={"x": top_left[0], "y": top_left[1]},
                end={"x": bottom_right[0], "y": bottom_right[1]}
            )
            defect = Defect(type="scratch", coordinates=defect_coordinates)
            self.defects.append(defect)

        return points

    def highlight_defects(self) -> np.ndarray:
        """Обводит дефекты прямоугольниками на изображении."""
        image_with_defects = self.defected_image.copy()

        for defect in self.defects:
            start = defect.coordinates.start
            end = defect.coordinates.end
            cv2.rectangle(image_with_defects,
                          (start['x'], start['y']),
                          (end['x'], end['y']),
                          color=(0, 255, 0),
                          thickness=2)
        return image_with_defects
