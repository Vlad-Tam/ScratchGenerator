import os
import random
import json
from PIL import Image, ImageFilter, ImageDraw
import os
import random
from PIL import Image, ImageFilter, ImageDraw

from src.entities.entities import Defect, Coordinates
from src.generators.defect_generator import DefectGenerator


class BlurGenerator(DefectGenerator):
    def __init__(self, image_path):
        self.original_image = Image.open(image_path)
        self.defected_image = self.original_image.copy()
        self.defects = []

    def generate_defects(self):
        num_blurs = random.randint(7, 15)
        for _ in range(num_blurs):
            center_x = random.randint(0, self.defected_image.width)
            center_y = random.randint(0, self.defected_image.height)

            min_size = min(self.defected_image.width, self.defected_image.height) // 20
            max_size = min(self.defected_image.width, self.defected_image.height) // 5

            width = random.randint(min_size, max_size)
            height = random.randint(min_size, max_size)

            left = center_x - width // 2
            upper = center_y - height // 2
            right = center_x + width // 2
            lower = center_y + height // 2

            mask = Image.new('L', self.defected_image.size, 0)

            draw_mask = ImageDraw.Draw(mask)
            draw_mask.ellipse((left, upper, right, lower), fill=255)

            radius = random.randint(3, 8)
            blurred_image = self.defected_image.filter(ImageFilter.GaussianBlur(radius=radius))

            self.defected_image = Image.composite(blurred_image, self.defected_image, mask)

            defect_coordinates = Coordinates(
                start={"x": left, "y": upper},
                end={"x": right, "y": lower}
            )
            defect = Defect(type="blur", coordinates=defect_coordinates)
            self.defects.append(defect)

        return self.defected_image

    def highlight_defects(self) -> Image:
        """Обводит дефекты размытия прямоугольниками на изображении."""
        draw = ImageDraw.Draw(self.defected_image)
        for defect in self.defects:
            start = defect.coordinates.start
            end = defect.coordinates.end
            draw.rectangle(
                [start['x'], start['y'], end['x'], end['y']],
                outline=(0, 255, 0),
                width=2
            )

        return self.defected_image
