import cv2
import numpy as np
import random

from src.entities.entities import Defect, Coordinates
from src.generators.defect_generator import DefectGenerator


class AbrasionGenerator(DefectGenerator):
    def __init__(self, image_path):
        self.original_image = cv2.imread(image_path)
        self.defected_image = self.original_image.copy()
        self.defects = []

    def generate_defects(self):
        """Добавляет изношенные области на изображение."""
        H, W = self.defected_image.shape[:2]
        max_size = (int(W * 0.1), int(H * 0.1))
        num_areas = random.randint(10, 15)
        noise_level = 15

        largest_defect_mask = None
        largest_defect_size = 0

        for _ in range(num_areas):
            center = (random.randint(0, W - 1), random.randint(0, H - 1))
            size = (random.randint(10, max_size[0]), random.randint(10, max_size[1]))
            mask = self._create_noisy_polygon(center, size, noise_level, self.defected_image.shape)
            self._draw_worn_area(self.defected_image, mask, center, noise_level)

            coords = np.column_stack(np.where(mask > 0))
            if coords.size > 0:
                top_left = coords.min(axis=0)
                bottom_right = coords.max(axis=0)
                defect_size = (bottom_right[0] - top_left[0]) * (bottom_right[1] - top_left[1])

                if defect_size > largest_defect_size:
                    largest_defect_size = defect_size
                    largest_defect_mask = mask

                defect_coordinates = Coordinates(
                    start={"x": top_left[1], "y": top_left[0]},
                    end={"x": bottom_right[1], "y": bottom_right[0]}
                )
                defect = Defect(type="abrasion", coordinates=defect_coordinates)
                self.defects.append(defect)

        if largest_defect_mask is not None:
            coords = np.column_stack(np.where(largest_defect_mask > 0))
            top_left = coords.min(axis=0)
            bottom_right = coords.max(axis=0)

            defect_coordinates = Coordinates(
                start={"x": top_left[1], "y": top_left[0]},
                end={"x": bottom_right[1], "y": bottom_right[0]}
            )
            defect = Defect(type="abrasion", coordinates=defect_coordinates)
            self.defects.append(defect)

        return self.defected_image

    def _create_noisy_polygon(self, center: tuple, size: tuple, noise_level: float, img_shape: tuple,
                              num_vertices: int = 6) -> np.ndarray:
        """Создает полигон с шумом вокруг заданного центра."""
        angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
        radius = random.randint(5, min(size[0] // 2, size[1] // 2))
        points = np.array(
            [(center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)) for angle in angles])
        noise = np.random.normal(0, noise_level, points.shape)
        noisy_points = points + noise
        noisy_points = np.clip(noisy_points, 0, [img_shape[1] - 1, img_shape[0] - 1]).astype(np.int32)
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, noisy_points, color=255)
        return mask

    def _draw_worn_area(self, img: np.ndarray, mask: np.ndarray, center: tuple, noise_level: float,
                        gradient_color: tuple = None) -> None:
        """Наносит изношенную область на изображение, используя маску."""
        if gradient_color is not None:
            worn_area = np.full(img.shape, gradient_color, dtype=np.uint8)
        else:
            if random.random() < 0.1:
                gradient_value = random.randint(220, 255)
                worn_area = np.full(img.shape, gradient_value, dtype=np.uint8)
            else:
                original_value = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[center[1], center[0]]
                noise = np.random.normal(0, noise_level)
                new_value = np.clip(original_value + noise, 0, 255)
                worn_area = np.full(img.shape, new_value, dtype=np.uint8)

        if random.random() < 0.85:
            img[mask > 0] = cv2.addWeighted(img, 0.6, worn_area, 0.4, 0)[mask > 0]
        else:
            gray_value = random.randint(200, 255)
            uniform_gray_color = (gray_value, gray_value, gray_value)
            worn_area = np.full(img.shape, uniform_gray_color, dtype=np.uint8)
            img[mask > 0] = worn_area[mask > 0]

        worn_area_blurred = cv2.GaussianBlur(worn_area, (35, 35), 0)
        img[mask > 0] = cv2.addWeighted(img, 0.6, worn_area_blurred, 0.4, 0)[mask > 0]

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
