import cv2
import numpy as np
import matplotlib.pyplot as plt

#converting into hsv image
#green range
lower_green = np.array([36,0,0])
upper_green = np.array([86, 255, 255])
#blue range
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])
#Red range
lower_red = np.array([0,31,255])
upper_red = np.array([176,255,255])
#white range
lower_white = np.array([0,0,0])
upper_white = np.array([0,0,255])


stride_kernel = (25,25)
stride_aud = (30,30)
stride_close = (150,150)

def non_max(boxes , scores , iou_num):

    scores_sort = scores.argsort().tolist()
    keep = []
    while(len(scores_sort)):
        index = scores_sort.pop()
        keep.append(index)
        if(len(scores_sort) == 0):
            break
        iou_res = []
        for i in scores_sort:
            iou_res.append(iou(boxes[index] , boxes[i]))
        iou_res = np.array(iou_res)
        filtered_indexes = set((iou_res > iou_num).nonzero()[0])
        scores_sort = [v for (i,v) in enumerate(scores_sort) if i not in filtered_indexes]
    final = []    
    for i in keep:
        final.append(boxes[i])
        
    return np.array(final)

def iou(box1,box2):

    x1 = max(box1[0],box2[0])
    x2 = min(box1[2],box2[2])
    y1 = max(box1[1] ,box2[1])
    y2 = min(box1[3],box2[3])
    if x1 > x2 or y1>y2:
        return -2
    inter = (x2 - x1)*(y2 - y1)
    area1 = (box1[2] - box1[0])*(box1[3] - box1[1])
    area2 = (box2[2] - box2[0])*(box2[3] - box2[1])
    fin_area = area1 + area2 - inter   
    iou = inter/fin_area
        
    return iou

def get_contours(image):

    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(image, image, mask=mask)
    res_bgr = cv2.cvtColor(res,cv2.COLOR_HSV2BGR)
    res_gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    mask = 255 - mask
    kernel = np.ones(stride_kernel,np.uint8)
    kernel_aud = np.ones(stride_aud,np.uint8)
    kernel_close = np.ones(stride_close,np.uint8)
    thresh_aud = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_aud)
    thresh_players = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    thresh_aud = cv2.morphologyEx(thresh_aud, cv2.MORPH_OPEN, kernel_close)
    thresh_aud = 255 - thresh_aud
    thresh = cv2.bitwise_and( thresh_aud , thresh_players)
    _temp, im2, contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # im2, contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return im2, thresh_aud

def get_boxes( image , im2 ):    

    im1 = image.shape[0]
    im = image.shape[1]    
    maxx = 0
    needed = im2[0]
    boxes = []
    scores = []
    for c in im2:
        temp = 1
        clr = ( 255 , 0 , 0)
        x,y,w,h = cv2.boundingRect(c)
        arr_cont = w*h
        if arr_cont > maxx:
            maxx = arr_cont
            needed = c
        if h < image.shape[0]*0.01 or w < image.shape[1]*0.01:
            clr = (0,255,0)        
        if w > h:
            clr = (0,255,0)
        w = int(w + 0.5*w)
        h = int(h + 0.4*h)
        x = int(x - w*0.25)
        y = int(y - h*0.2)
        if x < 0:
            x = 0
        if y < 0:
            y = 0            
        if clr == (255,0,0):
            boxes.append([x,y , (x + w), (y + h)])
            scores.append(arr_cont)

    return boxes, scores