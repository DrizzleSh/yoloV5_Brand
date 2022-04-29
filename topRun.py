import numpy as np
import time
from PIL import Image
import cv2
from PIL.JpegImagePlugin import JpegImageFile

from yolo import YOLO
import os
import json
import datetime

class MyEncoder(json.JSONEncoder):
   def default(self, obj):
      if isinstance(obj, datetime.datetime):    # 判断对象是否为已知类型
         print("MyEncoder-datetime.datetime")
         return obj.strftime("%Y-%m-%d %H:%M:%S")
      if isinstance(obj, bytes):
         return str(obj, encoding='utf-8')
      if isinstance(obj, int):
         return int(obj)
      elif isinstance(obj, float):
         return float(obj)
      elif isinstance(obj, np.ndarray):
         return obj.tolist()
      elif isinstance(obj, JpegImageFile):
         return str(obj)
      # elif isinstance(obj, Image):
      #    return str(obj)
      else:
         return super(MyEncoder, self).default(obj)

yolo = YOLO()
path = "F:/dataSets/brand_test/val/images/"
path_json = "F:/dataSets/brand_test/val/annotations/instances_val2017.json"
path_save = "./utils_coco/result.json"
content=[]
f = open(path_json, 'r', encoding='utf-8')
s = f.read()
rest = json.loads(s)
for i in rest['images']:
   file_name = i["file_name"]
   image_id = i["id"]      # 只时代表第几张图片
   image = Image.open(path+file_name)
   b=yolo.save_json(image,file_name,image_id,rest['categories'])
   if b != None:
      content.extend(b)
f.close()
with open(path_save, 'a') as f:
   json.dump(content,f,cls=MyEncoder,skipkeys=True,indent=4)