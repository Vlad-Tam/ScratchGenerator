import cv2
import matplotlib.pyplot as plt

from src.generators.abrasion_generator import AbrasionGenerator
from src.generators.blur_generator import BlurGenerator
from src.generators.noise_generator import NoiseGenerator
from src.generators.scratch_generator import ScratchGenerator

if __name__ == '__main__':
    gen = NoiseGenerator(r"C:\Users\1\Desktop\Diploma\code\ScratchGenerator\src\photo\original\photo1.jpg")
    gen.generate_defects()
    image_with_defects = gen.highlight_defects()

    save_path = r'C:\Users\1\Desktop\Diploma\code\ScratchGenerator\src\photo\defected\image.jpg'
    cv2.imwrite(save_path, image_with_defects)
    # image_with_defects.save(save_path)
