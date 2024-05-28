from tensorflow.keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
import os
 
 
# 生成文件夹
def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError:
            pass
 
 
# 图片生成器ImageDataGenerator
pic_gen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest')
 
# 生成图片
def img_create(img_dir, save_dir, num=1):
    img = load_img(img_dir)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    img_flow = pic_gen.flow(
        x,
        batch_size=1,
        save_to_dir=save_dir,
        save_prefix="",
        save_format="jpg"
    )
    i = 0
    for batch in img_flow:
        i += 1
        if i > num:
            break
 
 
# 生成训练集
path = './t/'  # 车牌号数据集路径(车牌图片宽240，高80)
pic_name = sorted(os.listdir(path))
n = len(pic_name)

for i in range(n):
    print("正在读取第%d张图片" % i)
    
    img_dir = './tt/' + pic_name[i]
    save_dir = './data./' +  pic_name[i]
    ensure_dir(save_dir)
    img_create(img_dir, save_dir, num=200)
    print("train: ", i)
 

