import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def make_coordinate(image,line_parameters):
    slope,intercept=line_parameters
    y1=image.shape[0]
    y2=int(y1*(3/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])


def average_slope_intercept(image, lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2= line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept =parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average=np.average(left_fit,axis=0)
    right_fit_average=np.average(right_fit,axis=0)
    print(left_fit_average,'left')
    print(right_fit_average,'right')
    try:
        left_line=make_coordinate(image,left_fit_average)
        right_line=make_coordinate(image,right_fit_average)
        return np.array([left_line,right_line])
    except Exception as e:
        print(e, '\n')

        return None
        


def canny(image):
    gray=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    canny=cv2.Canny(blur,50,150)
    return canny

r1 = random.randint(0, 255)
r2 = random.randint(0, 255)
r3 = random.randint(0, 255)
def display_lines(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image,(x1,y1),(x2,y2),(130,0,75),10)
    return line_image


def roi(image):
    height=image.shape[0]
    triangle=np.array([
        [(150,height),(1250,height),(675,390)]
        ])
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,triangle,255)
    masked_image=cv2.bitwise_and(image,mask)
    return masked_image


#video code starts here
cap=cv2.VideoCapture("sample vid.mp4")

def rescale_frame(frame, percent=142):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


def make_1080p():
    cap.set(3,1820)
    cap.set(4,1080)

#make_1080p()

cascade_src = 'cars2.xml'
car_cascade = cv2.CascadeClassifier(cascade_src)
sign_cascade=cv2.CascadeClassifier('sign.xml')
font=cv2.FONT_HERSHEY_SIMPLEX


while(cap.isOpened()):
    _, frame=cap.read()
    frame = rescale_frame(frame, percent=142)
    canny_image=canny(frame)
    cropped_image=roi(canny_image)
    lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    averaged_lines=average_slope_intercept(frame,lines)
    line_image=display_lines(frame,averaged_lines )
    combo_image=cv2.addWeighted(frame,0.8,line_image,1,1)
    gray = cv2.cvtColor(combo_image, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.3,5)
    for (x,y,w,h) in cars:
        r1 = random.randint(0, 255)
        r2 = random.randint(0, 255)
        r3 = random.randint(0, 255)
        cv2.rectangle(combo_image,(x,y),(x+w,y+h),(200,170,255),2)
        cv2.putText(combo_image,'CAR AHEAD',(x+w,y+h),font,0.8,(0,0,255),2,cv2.LINE_AA)


    sign = sign_cascade.detectMultiScale(gray, 1.3,5)
    for(stx,sty,stw,sth) in sign :
        r1 = random.randint(0, 255)
        r2 = random.randint(0, 255)
        r3 = random.randint(0, 255)
        cv2.rectangle(combo_image,(stx,sty),(stx+stw,sty+sth),(r1,r2,r3),2)
        cv2.putText(combo_image,'sign board',(stx+stw,sty+sth),font,1.5,(0,255,255),2,cv2.LINE_AA)
         
        
    cv2.imshow('result',combo_image)
    cv2.imshow('original',frame)
    cv2.imshow('cropped_image',cropped_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()















    



    
