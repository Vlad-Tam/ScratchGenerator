from main_abrasions import func_abrasions
from main_grey import func_grey
from main_white import func_white

if __name__ == '__main__':
    file_path = r'C:\Users\1\Desktop\Diploma\code\ScratchGenerator\photo6.jpg'
    print(func_white(func_grey(func_abrasions(file_path))))
