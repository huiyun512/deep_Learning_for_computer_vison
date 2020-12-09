import os


class paths:
    def __init__(self):
        self.set = []

    def list_image(self, path):
        for root, dirs, files in os.walk(path):
            for dir in dirs:
                filenames = os.listdir(os.path.join(root, dir))
                for filename in filenames:
                    self.set.append(os.path.join(root + "\\" + dir, filename))
        return self.set

"""
if __name__ == '__main__':
    path = paths()
    dir = "C:\\Users\\eduardogao\\Desktop\\CSDN\\Deep-Learning-For-Computer-Vision-第一册start-代码-按数据集分类\\Deep-Learning-For-Computer-Vision-第一册start-代码-按数据集分\\Deep-Learning-For-Computer-Vision-master\\datasets\\animals"
    print(list(path.list_image(dir)))

"""
