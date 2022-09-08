import glob
import os
import numpy as np

r = np.random
r.seed(42)

def get_kflod_data(data_path,Kflod=1):
    train_img_list=glob.glob(os.path.join(data_path, 'train/image/*.png'))
    test_img_list=glob.glob(os.path.join(data_path, 'test/image/*.png'))
    # mixed data
    train_img_list+=test_img_list
    train_img_list.sort()
    r.shuffle(train_img_list)
    size = int(len(train_img_list)/5) # Kflod 每分数据的大小

    # 测试下的代码对不对
    start = (Kflod-1)*size
    if Kflod == 5:# Kflod * 5 = 195 < 196 会lou一个图片
        end = len(train_img_list)
    else:
        end = Kflod*size
    test_img_list = train_img_list[start:end]  # 取得数据的范围：(Kflod-1)*size:Kflod*size
    # train 中排除测试集的数据

    train_img_list = list(set(train_img_list) - set(test_img_list))
    return train_img_list,test_img_list