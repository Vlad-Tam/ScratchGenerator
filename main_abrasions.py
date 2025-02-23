import cv2
import numpy as np
import random
import os


def create_noisy_polygon(center: tuple, size: tuple, noise_level: float, img_shape: tuple,
                         num_vertices: int = 6) -> np.ndarray:
    """Creates a noisy polygon mask based on the image shape."""
    angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)

    # Ensure radius values are valid
    radius = random.randint(5, min(size[0] // 2, size[1] // 2))  # Minimum radius of 5 to avoid zero or negative
    points = np.array([(center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)) for angle in angles])

    # Add noise to the points
    noise = np.random.normal(0, noise_level, points.shape)
    noisy_points = points + noise

    # Convert to integer and clip to image bounds
    noisy_points = np.clip(noisy_points, 0, [img_shape[1] - 1, img_shape[0] - 1]).astype(np.int32)

    # Create a mask using the polygon points
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, noisy_points, color=255)

    return mask


def draw_worn_area(img: np.ndarray, mask: np.ndarray, center: tuple, noise_level: float,
                   gradient_color: tuple = None) -> None:
    if gradient_color is not None:
        # Используем заданный цвет для потертости
        worn_area = np.full(img.shape, gradient_color, dtype=np.uint8)
    else:
        # Decide if the worn area will be a gradient or based on the original image
        if random.random() < 0.1:  # 10% probability
            # Create a white or light gray gradient
            gradient_value = random.randint(220, 255)  # Light gray to white
            worn_area = np.full(img.shape, gradient_value, dtype=np.uint8)
        else:
            # Get the original grayscale value at the center (for noise generation)
            original_value = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[center[1], center[0]]
            # Generate a new value based on the original value with some noise
            noise = np.random.normal(0, noise_level)  # Normal noise
            new_value = np.clip(original_value + noise, 0, 255)
            # Create the worn area with the new grayscale value
            worn_area = np.full(img.shape, new_value, dtype=np.uint8)

    # Определяем, будет ли цвет закрашенной области зависеть от исходного изображения
    if random.random() < 0.85:  # 85% вероятность
        # Используем цвет, зависящий от исходного изображения
        img[mask > 0] = cv2.addWeighted(img, 0.6, worn_area, 0.4, 0)[mask > 0]
    else:
        # Генерируем случайный цвет в оттенках серого
        gray_value = random.randint(200, 255)  # Случайный серый цвет
        uniform_gray_color = (gray_value, gray_value, gray_value)
        worn_area = np.full(img.shape, uniform_gray_color, dtype=np.uint8)  # Перезаписываем worn_area
        img[mask > 0] = worn_area[mask > 0]  # Используем только закрашенную область

    # Apply a stronger Gaussian blur to the worn area
    worn_area_blurred = cv2.GaussianBlur(worn_area, (35, 35), 0)  # Increased kernel size for stronger blur

    # Blend the worn area with the original image using the mask
    img[mask > 0] = cv2.addWeighted(img, 0.6, worn_area_blurred, 0.4, 0)[mask > 0]


def add_worn_areas(original_image: np.ndarray, num_areas: int = 5, noise_level: float = 10) -> np.ndarray:
    image = original_image.copy()
    H, W = image.shape[:2]

    # Определяем максимальный размер потертостей относительно разрешения изображения
    max_size = (int(W * 0.1), int(H * 0.1))  # 10% от ширины и высоты

    for _ in range(num_areas):
        center = (random.randint(0, W - 1), random.randint(0, H - 1))
        size = (random.randint(10, max_size[0]), random.randint(10, max_size[1]))

        # Create a noisy polygon mask
        mask = create_noisy_polygon(center, size, noise_level, image.shape)

        draw_worn_area(image, mask, center, noise_level)

    # Добавляем дополнительную потертость с шансом 10%
    if random.random() < 0.1:
        center = (random.randint(0, W - 1), random.randint(0, H - 1))
        size = (random.randint(10, max_size[0]), random.randint(10, max_size[1]))
        mask = create_noisy_polygon(center, size, noise_level, image.shape)

        # Генерируем цвет в диапазоне от (200, 200, 200) до (255, 255, 255)
        gradient_color = tuple(random.randint(200, 255) for _ in range(3))
        draw_worn_area(image, mask, center, noise_level, gradient_color)

    # Увеличиваем размер дефекта с вероятностью 10%
    if random.random() < 0.99:
        center = (random.randint(0, W - 1), random.randint(0, H - 1))
        size = (int(random.randint(10, max_size[0]) * random.uniform(1.2, 6)),
                int(random.randint(10, max_size[1]) * random.uniform(1.2, 6)))
        mask = create_noisy_polygon(center, size, noise_level, image.shape)

        # Генерируем случайный цвет в оттенках серого
        gray_value = random.randint(200, 255)
        uniform_gray_color = (gray_value, gray_value, gray_value)
        worn_area = np.full(image.shape, uniform_gray_color, dtype=np.uint8)
        image[mask > 0] = worn_area[mask > 0]  # Используем только закрашенную область

    return image


def func_abrasions(path: str):
    # Load the original color image
    original_image = cv2.imread(path)
    worn_image = add_worn_areas(original_image.copy(), num_areas=10, noise_level=15)

    # Save the result
    directory, filename = os.path.split(path)
    name, extension = os.path.splitext(filename)
    new_filename = f"{name}_worn{extension}"
    new_file_path = os.path.join(directory, new_filename)
    cv2.imwrite(new_file_path, worn_image)
    return new_file_path


if __name__ == '__main__':
    file_path = r'C:\Users\1\Desktop\Diploma\code\ScratchGenerator\photo6.jpg'
    print(func_abrasions(file_path))