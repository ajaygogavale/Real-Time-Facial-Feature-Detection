import cv2

#img=cv2.imread(r"F:\photos\X-men wallpaper\img12.jpg") 
img=cv2.imread(r"C:\Users\hp\Downloads\people-with-crossed-arms-medium-shot.jpg")
#img=cv2.imread(r"C:\Users\hp\Downloads\medium-shot-happy-friends-city.jpg")
img=cv2.resize(img,(700,450))

#step1 coverted into grayscale img
gry=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# step2= haar cascade file
sm=cv2.CascadeClassifier(r"C:\Users\hp\Downloads\haarcascade_opencv\haarcascade_smile.xml") #(smile detect)
ey=cv2.CascadeClassifier(r"C:\Users\hp\Downloads\haarcascade_opencv\haarcascade_eye.xml") #if not found use below (eye detect)
#f=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
fc=cv2.CascadeClassifier(r"C:\Users\hp\Downloads\haarcascade_opencv\haarcascade_frontalface_default.xml") #(face detect)

# step 3 = detection
# face detect
f=fc.detectMultiScale(gry,scaleFactor=1.3, minNeighbors=5)
for (x,y,w,h) in f:
    # find region of interest(ROI)
    roi_gry=gry[y:y+h,x:x+w]
    roi_img=img[y:y+h,x:x+w]

    # detect smile
    s=sm.detectMultiScale(roi_img,scaleFactor=1.2, minNeighbors=15)
    for (xs,ys,ws,hs) in s:
        cv2.rectangle(roi_img,(xs,ys),(xs+ws,ys+hs),(0,255,0),3)
    
    # detect eye
    e=ey.detectMultiScale(roi_img,scaleFactor=1.1, minNeighbors=1)
    for (xe,ye,we,he) in e:
        cv2.rectangle(roi_img,(xe,ye),(xe+we,ye+he),(0,0,255),3)
   


#cv2.imshow('wscube',roi_img)
cv2.imshow('wscube',img)
cv2.waitKey(0)
cv2.destroyAllWindows()