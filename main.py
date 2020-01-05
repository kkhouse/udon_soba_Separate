import requests
import urllib.request as ur
from bs4 import BeautifulSoup
import lxml
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
query_word1 = sys.argv[1]
query_word2 = sys.argv[2]
max_num = int(sys.argv[3])

#一つ目
for num in range(max_num):
  headers = {"Users-Agent":"hoge"}
  URL = "https://search.yahoo.co.jp/image/search?fr=top_ga1_sa&p={0}&ei=UTF-8&b={1}".format(query_word1,1+20*num)
  resp = requests.get(URL,timeout=1,headers=headers)

  soup = BeautifulSoup(resp.text,"lxml")
  imgs_1 = soup.find_all(alt="「{}」の画像検索結果".format(query_word1))

  for i in range(len(imgs_1)):
    dir_name ="./data/{0}".format(query_word1)
    if not os.path.exists(dir_name):
      os.makedirs(dir_name)
    filepath =  dir_name + "/{0}-{1}.jpg".format(num,i)
    ur.urlretrieve(imgs_1[i]["src"],filepath)
#二つ目
for num in range(max_num):
  headers = {"Users-Agent":"hoge"}
  URL = "https://search.yahoo.co.jp/image/search?fr=top_ga1_sa&p={0}&ei=UTF-8&b={1}".format(query_word2,1+20*num)
  resp = requests.get(URL,timeout=1,headers=headers)

  soup = BeautifulSoup(resp.text,"lxml")
  imgs_2 = soup.find_all(alt="「{}」の画像検索結果".format(query_word2))

  for i in range(len(imgs_2)):
    dir_name ="./data/{0}".format(query_word2)
    if not os.path.exists(dir_name):
      os.makedirs(dir_name)
    filepath =  dir_name + "/{0}-{1}.jpg".format(num,i)
    ur.urlretrieve(imgs_2[i]["src"],filepath)



def scratch_image(img, flip=True, thr=True, filt=True, resize=True, erode=True):
    methods = [flip, thr, filt, resize, erode]

    img_size = img.shape
    filter1 = np.ones((3, 3))

    images = [img]
    
    scratch = np.array([
        lambda x: cv2.flip(x, 1),
        lambda x: cv2.threshold(x, 100, 255, cv2.THRESH_TOZERO)[1],
        lambda x: cv2.GaussianBlur(x, (5, 5), 0),
        lambda x: cv2.resize(cv2.resize(
                        x, (img_size[1] // 5, img_size[0] // 5)
                    ),(img_size[1], img_size[0])),
        lambda x: cv2.erode(x, filter1)
    ])
    
    doubling_images = lambda f, img: ([f(img)])

    for func in scratch[methods]:
        images += doubling_images(func, img)
    
    return images


soba_list = glob.glob("./data/{0}".format(query_word1)+ "/*")
udon_list = glob.glob("./data/{0}".format(query_word2) + "/*")

scratch_soba_images = []
scratch_udon_images  = []
for soba_read in soba_list:
  soba_img = cv2.imread(soba_read)
  scratch_soba_images +=scratch_image(soba_img)
for udon_read in udon_list:
  udon_img = cv2.imread(udon_read)
  scratch_udon_images += scratch_image(udon_img)

for num, im in enumerate(scratch_udon_images):
  dir_name = "./data/{0}_padding/".format(query_word1)
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
  cv2.imwrite(dir_name + str(num+1) + ".jpg" ,im) 
for num, im in enumerate(scratch_soba_images):
  dir_name = "./data/{0}_padding/".format(query_word2)
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
  cv2.imwrite(dir_name + str(num+1) + ".jpg" ,im) 

import img_data
import learning

#python main.py 〇〇 ◇◇ 1で実行
