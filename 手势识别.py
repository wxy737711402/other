
import cv2
import numpy as np
import math
import win32con
import win32api
import time

cap = cv2.VideoCapture(0)
XBOOL=0
while(cap.isOpened()):
    ret,frame = cap.read() # 读取摄像头每帧图片
    
    frame = cv2.flip(frame,1)
    kernel = np.ones((2,2),np.uint8)
    roi = frame[100:300,100:300] # 选取图片中固定位置作为手势输入

    cv2.rectangle(frame,(100,100),(300,300),(0,0,255),0) # 用红线画出手势识别框
    # 基于hsv的肤色检测
    hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0,28,70],dtype=np.uint8)
    upper_skin = np.array([20, 255, 255],dtype=np.uint8)
    
    # 进行高斯滤波
    mask = cv2.inRange(hsv,lower_skin,upper_skin)
    mask = cv2.dilate(mask,kernel,iterations=4)
    mask = cv2.GaussianBlur(mask,(5,5),100)
    
    # 找出轮廓
    contours,h = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours,key=lambda x:cv2.contourArea(x))
    epsilon = 0.0005*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    hull = cv2.convexHull(cnt)
    areahull = cv2.contourArea(hull)
    areacnt = cv2.contourArea(cnt)
    arearatio = ((areahull-areacnt)/areacnt)*100
    # 求出凹凸点
    hull = cv2.convexHull(approx,returnPoints=False)
    defects = cv2.convexityDefects(approx,hull)
    l=0 #定义凹凸点个数初始值为0 
    for i in range(defects.shape[0]):
        s,e,f,d, = defects[i,0]
        start = tuple(approx[s][0])
        end = tuple(approx[e][0])
        far = tuple(approx[f][0])
        pt = (100,100)

        a = math.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0]-far[0])**2+(end[1]-far[1])**2)
        s = (a+b+c)/2
        ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
	# 手指间角度求取
        angle = math.acos((b**2 + c**2 -a**2)/(2*b*c))*57

        if angle<=90 and d>18:
            l+=1
            cv2.circle(roi,far,3,[255,0,0],-1)
        cv2.line(roi,start,end,[0,255,0],2) # 画出包络线

    font = cv2.FONT_HERSHEY_SIMPLEX
    # 下面的都是条件判断
    
    if l==1:
        if areacnt<2000:
            cv2.putText(frame,"请把手放入框中",(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
        else:
            if arearatio<12:
                cv2.putText(frame,'OUT',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
                XBOOL=XBOOL-1
            elif arearatio<17.5:
                cv2.putText(frame,"GO",(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
                XBOOL=XBOOL+1
            else:
                cv2.putText(frame,'GO',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
                XBOOL=XBOOL+1
    elif l==2:
        cv2.putText(frame,'NULL',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
        if XBOOL > 10:
            XBOOL=0
            win32api.keybd_event(39,0,0,0)  #按下

            time.sleep(0.1)                            #延迟一会儿

            win32api.keybd_event(39,0,win32con.KEYEVENTF_KEYUP,0)  #松开
        if XBOOL < -10:
            XBOOL=0
            win32api.keybd_event(37,0,0,0)  #按下

            time.sleep(0.1)                            #延迟一会儿

            win32api.keybd_event(37,0,win32con.KEYEVENTF_KEYUP,0)  #松开

        
       
    elif l==3:
        if arearatio<27:
            cv2.putText(frame,'GO',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
            XBOOL=XBOOL+1
        else:
            cv2.putText(frame,'GO',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
            XBOOL=XBOOL+1
    elif l==4:
        cv2.putText(frame,'GO',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
        XBOOL=XBOOL+1
    elif l==5:
        cv2.putText(frame,'GO',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
        XBOOL=XBOOL+1
    cv2.imshow('frame',frame)
    cv2.imshow('mask', mask)
    k = cv2.waitKey(25)& 0xff  
    if k == 27:     # 键盘Esc键退出
        break
cv2.destroyAllWindows()
cap.release()