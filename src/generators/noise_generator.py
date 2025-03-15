import cv2
import numpy as np
from scipy.stats import poisson
from src.entities.entities import Defect, Coordinates
from src.generators.defect_generator import DefectGenerator

class NoiseGenerator(DefectGenerator):
    def __init__(self, image_path):
        self.original_image = cv2.imread(image_path)
        self.defected_image = self.original_image.copy()
        self.defects = []

    def generate_defects(self):
        """Добавляет шумы на изображение различной формы и размера."""
        lambda_ = 40
        growth_steps = 200
        h, w = self.defected_image.shape[:2]
        centers = np.column_stack(
            (np.random.randint(0, w, lambda_), np.random.randint(0, h, lambda_))
        )

        for center in centers:
            x, y = center
            radius = poisson.rvs(10)
            shape_type = np.random.choice(['circle', 'ellipse'])
            noise_points = []

            for _ in range(growth_steps):
                if shape_type == 'circle':
                    dx, dy = np.random.normal(0, 20, size=2).astype(int)
                    x_new, y_new = x + dx, y + dy
                else:  # ellipse
                    angle = np.random.uniform(0, 2 * np.pi)
                    a = np.random.randint(5, 15)
                    b = np.random.randint(5, 15)
                    x_new = int(x + a * np.cos(angle))
                    y_new = int(y + b * np.sin(angle))

                if 0 <= x_new < w and 0 <= y_new < h:
                    self.defected_image[y_new, x_new] = np.clip(
                        self.defected_image[y_new, x_new] + np.random.exponential(50), 0, 255
                    )
                    noise_points.append((x_new, y_new))

            if noise_points:
                min_x = min(point[0] for point in noise_points)
                max_x = max(point[0] for point in noise_points)
                min_y = min(point[1] for point in noise_points)
                max_y = max(point[1] for point in noise_points)

                top_left = (max(0, min_x), max(0, min_y))
                bottom_right = (min(w - 1, max_x), min(h - 1, max_y))

                defect_coordinates = Coordinates(
                    start={"x": top_left[0], "y": top_left[1]},
                    end={"x": bottom_right[0], "y": bottom_right[1]}
                )
                defect = Defect(type="noise", coordinates=defect_coordinates)
                self.defects.append(defect)

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