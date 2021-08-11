import streamlit  as st
from PIL import Image
import cv2
import pytesseract
import numpy as np
pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'

def detector( image):
  img_array=np.array(image)
  img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
  classifier=cv2.CascadeClassifier('/content/haarcascade_russian_plate_number.xml')
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  N_plate=classifier.detectMultiScale(gray, scaleFactor=1.38, minNeighbors=4)
  if N_plate is not None:

    for (x,y,w,h) in N_plate:
      a,b=(int(0.02*img.shape[0]),(int(0.025*img.shape[1])))
      opencv_plate=img[y+a:y+h-a,x+b:x+w-b, :]
      kernal=np.ones((1,1), np.uint8)
      opencv_plate=cv2.dilate(opencv_plate,kernal,iterations=1)
      opencv_plate=cv2.erode(opencv_plate,kernal,iterations=1)
      plate_gray=cv2.cvtColor(opencv_plate,cv2.COLOR_BGR2GRAY)
      th,opencv_plate=cv2.threshold(plate_gray,127,225,cv2.THRESH_BINARY)
      img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
      rgb_img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      cv2.imwrite('op.jpg',opencv_plate)
    
    return rgb_img,opencv_plate
  else:
    return None,None

st.title('Number plate Detection & Recognition')
img=st.sidebar.file_uploader('Choose the image')

if img is not None:
  read_img=Image.open(img)
  st.image(img)
  if st.button('FIND'):
    D_img,No_plate=detector(read_img)
    if No_plate is not None:
      st.write('Number Plate Detect :slightly_smiling_face:')
      st.image(D_img)
      st.image(No_plate)
      st.header('Convert Number plate Image into Text')
      op = pytesseract.image_to_string(No_plate)
      st.write(op)
