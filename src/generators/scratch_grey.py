import os

import cv2
import numpy as np
import random
from collections.abc import Generator

def bezier(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, noise_level: float) -> Generator[np.ndarray, None, None]:
    def calc(t):
        return t * t * p1 + 2 * t * (1 - t) * p2 + (1 - t) * (1 - t) * p3

    approx = cv2.arcLength(np.array([calc(t)[:2] for t in np.linspace(0, 1, 10)], dtype=np.float32), False)
    for t in np.linspace(0, 1, round(approx * 1.2)):
        point = calc(t)
        noise_x = random.uniform(-noise_level, noise_level)
        noise_y = random.uniform(-noise_level, noise_level)
        yield np.round(point + [noise_x, noise_y, 0]).astype(np.int32)

def catmull_rom(p0, p1, p2, p3, num_points=100, noise_level=1.0):
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
    return points

def check_intersection(line1, line2):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    A, B = line1
    C, D = line2
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def interpolate_colors(color1, color2, alpha):
    return (int(color1[0] * (1 - alpha) + color2[0] * alpha),
            int(color1[1] * (1 - alpha) + color2[1] * alpha),
            int(color1[2] * (1 - alpha) + color2[2] * alpha))

def rgb_to_gray(rgb):
    return int(0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2])

def draw_straight_line(img: np.ndarray, orig_img: np.ndarray,  resolution: tuple, noise_level: float = 3.0):
    multiplicator = random.uniform(0.007, 0.01)  # Случайный множитель
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
    step_size = int(line_thickness / 2)  # Шаг равен радиусу круга
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

    for i, point in enumerate(points):
        if 0 <= point[0] < W and 0 <= point[1] < H:  # Проверка границ
            orig_color = orig_img[point[1], point[0]]  # Получаем оригинальный цвет

            # Приводим цвет к оттенку серого
            gray_value = rgb_to_gray(orig_color)
            new_color = (gray_value, gray_value, gray_value)

            # Плавный переход к белому цвету
            if i > 0:
                alpha = i / len(points)
                new_color = interpolate_colors(orig_color, new_color, alpha)

            # Логика для толщины линии
            brightness_factor = 1 + (line_thickness / 10)
            new_color = np.clip(np.array(new_color) * brightness_factor, 0, 255).astype(int)

            # Темные пиксели по краям
            if i == 0 or i == len(points) - 1:
                if random.random() < 0.1:  # 10% шанс на темный цвет
                    new_color = np.clip(np.array(new_color) * 0.8, 0, 255).astype(int)

            new_color = tuple(map(int, new_color))  # Преобразование в кортеж из целых чисел
            cv2.circle(img, point, line_thickness // 2, new_color, -1)  # Рисуем кружки

def draw_scratch(img: np.ndarray, orig_img: np.ndarray, bezier_points: list, line_thickness: int) -> None:
    for i, point in enumerate(bezier_points):
        if 0 <= point[0] < img.shape[1] and 0 <= point[1] < img.shape[0]:  # Проверка границ
            orig_color = orig_img[point[1], point[0]]  # Получаем оригинальный цвет

            # Приводим цвет к оттенку серого
            gray_value = rgb_to_gray(orig_color)
            new_color = (gray_value, gray_value, gray_value)

            # Плавный переход к белому цвету
            if i > 0:
                alpha = i / len(bezier_points)
                new_color = interpolate_colors(orig_color, new_color, alpha)

            # Логика для толщины линии
            brightness_factor = 1 + (line_thickness / 10)
            new_color = np.clip(np.array(new_color) * brightness_factor, 0, 255).astype(int)

            # Темные пиксели по краям
            if i == 0 or i == len(bezier_points) - 1:
                if random.random() < 0.1:  # 10% шанс на темный цвет
                    new_color = np.clip(np.array(new_color) * 0.8, 0, 255).astype(int)

            new_color = tuple(map(int, new_color))  # Преобразование в кортеж из целых чисел
            cv2.circle(img, (point[0], point[1]), line_thickness, new_color, -1)

def generate_scratch(img: np.ndarray, orig_img: np.ndarray, resolution: tuple, max_length: float, end_brush_range: tuple[float, float],
                     mid_brush_range: tuple[float, float], use_catmull: bool = False, noise_level: float = 1.0,
                     long_line: bool = False) -> np.ndarray:
    multiplicator = random.uniform(0.002, 0.005)  # Случайный множитель
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
        points = catmull_rom(p0[:2], p1[:2], p2[:2], p3[:2], noise_level=noise_level)
        draw_scratch(img, orig_img, points, line_thickness=line_thickness)
    else:
        bezier_points = list(bezier(p1, p2, p3, noise_level))
        draw_scratch(img, orig_img, bezier_points, line_thickness=line_thickness)

    return img, bezier_points

def add_scratches(original_image):
    resolution = original_image.shape[1], original_image.shape[0]
    image = original_image.copy()
    lines = []
    for _ in range(random.randint(20, 40)):
        use_catmull = random.choice([True, False])
        scratch, bezier_points = generate_scratch(image.copy(), original_image, resolution, max_length=200, end_brush_range=(0, 1),
                                                  mid_brush_range=(1, 5), use_catmull=use_catmull, noise_level=1.0,
                                                  long_line=False)
        lines.append(bezier_points)
        image = scratch

    if random.random() < 0.5:  # 50% шанс на генерацию длинной линии
        long_scratch, long_bezier_points = generate_scratch(image.copy(), original_image, resolution, max_length=200,
                                                            end_brush_range=(0, 1),
                                                            mid_brush_range=(1, 5),
                                                            use_catmull=random.choice([True, False]), noise_level=2.0,
                                                            long_line=True)
        lines.append(long_bezier_points)
        image = long_scratch

    if random.random() < 0.5:  # 50% шанс на добавление прямой линии
        draw_straight_line(image, original_image, resolution, noise_level=2.0)

    for i in range(len(lines)):
        if not lines[i]:  # Пропускаем пустые линии
            continue
        for j in range(i + 1, len(lines)):
            if not lines[j]:  # Пропускаем пустые линии
                continue
            if check_intersection((lines[i][0], lines[i][-1]),
                                  (lines[j][0], lines[j][-1])) and random.random() < 0.5:  # 50% шанс на прерывание
                lines[j] = lines[j][:len(lines[j]) // 2]  # Прерываем линию на половине
                for point in lines[j]:
                    if 0 <= point[0] < image.shape[1] and 0 <= point[1] < image.shape[0]:  # Проверка границ
                        orig_color = original_image[point[1], point[0]]
                        gray_value = rgb_to_gray(orig_color)  # Приводим к оттенку серого
                        new_color = (gray_value, gray_value, gray_value)  # Создаем серый цвет

                        # Проверка на темные пиксели
                        if np.array_equal(point, lines[j][0]) or np.array_equal(point, lines[j][-1]):
                            if random.random() < 0.1:  # 10% шанс на темный цвет
                                new_color = np.clip(np.array(new_color) * 0.8, 0, 255).astype(int)

                        new_color = tuple(map(int, new_color))  # Преобразование в кортеж из целых чисел
                        cv2.circle(image, (point[0], point[1]), 2, new_color, -1)  # Рисуем обрезанную линию
                break

    return image


def func_grey(path: str):
    # Load the image
    original_image = cv2.imread(path)
    scratched_image = add_scratches(original_image.copy())

    # Save the result
    directory, filename = os.path.split(path)
    name, extension = os.path.splitext(filename)
    new_filename = f"{name}_grey{extension}"
    new_file_path = os.path.join(directory, new_filename)
    cv2.imwrite(new_file_path, scratched_image)
    return new_file_path

if __name__ == '__main__':
    # Load the image
    original_image = cv2.imread(r'C:\Users\1\Desktop\Diploma\code\ScratchGenerator\photo3.jpg')
    scratched_image = add_scratches(original_image.copy())

    # Save the result
    cv2.imwrite(r'C:\Users\1\Desktop\Diploma\code\ScratchGenerator\photo2_changed_double.jpg', scratched_image)