import cv2
import numpy as np

img = cv2.imread("left0000.jpg")

points = []

def click(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x,y))
        print("Clicked:",x,y)

        cv2.circle(img,(x,y),5,(0,0,255),-1)
        cv2.imshow("image",img)

        if len(points)==2:
            x1,y1 = points[0]
            x2,y2 = points[1]

            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            print("Pixel distance:",dist)

cv2.imshow("image",img)
cv2.setMouseCallback("image",click)

cv2.waitKey(0)
cv2.destroyAllWindows()