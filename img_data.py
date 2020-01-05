import numpy as np
import glob
import os
from PIL import Image
from sklearn import datasets
from sklearn.model_selection import train_test_split

classes = ["udon", "soba"]
num_classes = len(classes)
image_size = 28
max_read = 280
X =[]
y =[]

for index,class_label in enumerate(classes):
  images_dir = "./data/" + class_label
  # print(images_dir)
  files = glob.glob(images_dir +"/*")
  # print(len(files))
  for i,file in enumerate(files):
    # if i >= 5:break
    image = Image.open(file)
    image = image.convert("RGB")
    image = image.resize((image_size,image_size))
    data = np.asarray(image)
    X.append(data)
    y.append(index)
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y)


f = open("progress.txt","w",encoding='utf-8')
f.write("学習用データ作成")
f.close()

# データの保存
Xy_data = (X_train, X_test, y_train, y_test)
if not os.path.exists("./data/us_data"):
  os.makedirs("./data/us_data")
np.save("./data/us_data/Xy_data.npy", Xy_data)


# データの利用
# X_train, X_test, y_train, y_test = np.load("./images/Xy_data.npy")


