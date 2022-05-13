import copy
import os
import imageio
import cv2
import numpy as np
from PIL import Image


class BatchRename():
    '''
    批量重命名文件夹中的图片文件
    '''

    def __init__(self):
        self.path = r'/DATA/ZX/Second paper/UNet-newmodle_udse_loss/data/DRHAGIS/test/image/'  # 表示需要命名处理的文件夹
        # self.label = '/DATA/ZX/uunet/CHASEDB1/test/'

    def rename(self):
        # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序
        filelist = os.listdir(self.path)
        total_num = len(filelist)  # 获取文件夹内所有文件个数
        i = 0  # 表示文件的命名是从1开始的
        for item in filelist:
            if item.endswith(('.jpeg', 'JPG','png', 'jpg','ppm')):
                # 初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的即可）

                src = os.path.join(os.path.abspath(self.path), item)
                print(src)
                dst = os.path.join(os.path.abspath(self.path),  str(i) + '.jpg')
                print(dst)
                # 处理后的格式也为jpg格式的，当然这里可以改成png格式
                # dst = os.path.join(os.path.abspath(self.path), '0000' + format(str(i), '0>3s') + '.jpg')
                # 这种情况下的命名格式为0000000.jpg形式，可以自主定义想要的格式
                try:
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        print('total %d to rename & converted %d jpgs' % (total_num, i))
    def retype(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)  # 获取文件夹内所有文件个数
        for image in filelist:
            img = Image.open(os.path.join(self.path,image))
            img.save(os.path.join(self.path,image).split('jpg')[0]+'png')
            # img.save(os.path.join(self.path,image).split('ppm')[0]+'png')
    def resize(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)  # 获取文件夹内所有文件个数
        for image in filelist:
            img = cv2.imread(os.path.join(self.path, image))

            r, g, b = cv2.split(img)
            # 以b，g，r分量重新生成新图像
            img = cv2.merge([b, g, r])

            print(img.shape)
            k = cv2.waitKey(0)
            if k == 511:
                cv2.destroyWindow('img')
            resize_img = cv2.resize(img, (1024, 1024))
            resize_img=Image.fromarray(resize_img)
            resize_img.save(os.path.join(self.path, image))
            x = cv2.waitKey(0)
            if x == 511:
                cv2.destroyWindow('img')

    def image_resize(self, width=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        filelist = os.listdir(self.path)
        for image in filelist:
            image = Image.open(os.path.join(self.path, image))
            dim = 3
            image=np.array(image)
            (h, w) = image.shape[:2]

            # if both the width and height are None, then return the
            # original image
            # resize the image
            resize_img = image.resize((512,512))
            # resize_img = Image.fromarray(resize_img)

            resize_img.save(os.path.join(self.path, resize_img))
            print("--------------")
            x = cv2.waitKey(0)
            if x == 511:
                cv2.destroyWindow('img')
            # return the resized image
def DataAugment(dir_path):
        if not os.path.exists(dir_path):
            print('路径不存在')
        else:
            dirs = os.listdir(dir_path)

            for subdir in dirs:
                sub_dir = dir_path + '/' + subdir
                img = cv2.imread(sub_dir)

                size = img.shape  # 获得图像的形状

                iLR = copy.deepcopy(img)  # 获得一个和原始图像相同的图像，注意这里要使用深度复制
                h = size[0]
                w = size[1]

                for i in range(h):  # 元素循环
                    for j in range(w):
                        iLR[i, w - 1 - j] = img[i, j]  # 注意这里的公式没，是不是恍然大悟了（修改这里）

                new_name = "%s" % (sub_dir)
                cv2.imwrite(new_name, iLR)

        print('done')


def rotate_bound1(image, angle):
    (h, w) = image.shape[:2]  # 返回(高,宽,色彩通道数),此处取前两个值返回
    # 抓取旋转矩阵(应用角度的负值顺时针旋转)。参数1为旋转中心点;参数2为旋转角度,正的值表示逆时针旋转;参数3为各向同性的比例因子
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    newW = int((h * np.abs(M[0, 1])) + (w * np.abs(M[0, 0])))
    newH = int((h * np.abs(M[0, 0])) + (w * np.abs(M[0, 1])))
    # 调整旋转矩阵以考虑平移
    M[0, 2] += (newW - w) / 2
    M[1, 2] += (newH - h) / 2
    # 执行实际的旋转并返回图像
    return cv2.warpAffine(image, M, (newW, newH))  # borderValue 缺省，默认是黑色



if __name__ == '__main__':
    demo = BatchRename()
    demo.retype()
    # demo.image_resize()
    # demo.resize()
    # demo.rename()
    # DataAugment(path)




