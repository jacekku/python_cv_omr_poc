import cv2 as cv
import numpy as np
import sys
from matplotlib import pyplot as plt
import math 

def karta_detect_rects(name = 'karta_odp_wyp_rotr.jpg'):

    img_rgb = cv.imread(name)
    assert img_rgb is not None, "file could not be read, check with os.path.exists()"
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
        
    template = cv.imread('rect_template.png', cv.IMREAD_GRAYSCALE)
    assert template is not None, "file could not be read, check with os.path.exists()"
    w, h = template.shape[::-1]
    res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where( res >= threshold)
    # print(len(loc))
    # print(*zip(*loc[::-1]))
    for pt in zip(*loc[::-1]):
        # print(pt)
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    # cv.imshow("Display window", img_rgb)
    # k = cv.waitKey(0)
    return zip(*loc[::-1]),template.shape

def find_corners(positions, template):
    w, h = template[::-1]
    top_left = min(positions)
    bottom_right = max(positions)

    top_right = top_left
    bottom_left = bottom_right
    for pos in positions:
        if(pos[0] > top_right[0] and pos[1] <= top_right[1] ):
            top_right = pos
        if(pos[0] < bottom_left[0] and pos[1] >= bottom_left[1]):
            bottom_left = pos
        pass
    
    top_right = (top_right[0]+w,top_right[1])
    bottom_left = (bottom_left[0],bottom_left[1]+h)
    bottom_right = (bottom_right[0]+w,bottom_right[1]+h)

    return [top_left,top_right,bottom_left,bottom_right]

def karta_transformation(corners, name = 'karta_odp_wyp_rotr.jpg'):
    img = cv.imread(name)
    assert img is not None, "file could not be read, check with os.path.exists()"

    # pts1 = np.float32([corners[0:3]])
    # pts2 = np.float32([[0,0],[400,0],[0,565]])

    # rows,cols,ch = img.shape
    # M = cv.getAffineTransform(pts1,pts2)
    # dst = cv.warpAffine(img,M,(cols,rows))

    source_points = np.float32([corners])
    size = 600
    maxX,maxY = size,math.ceil(size*math.sqrt(2))
    output_points = np.float32([[0,0],[maxX,0],[0,maxY],[maxX,maxY]])
    M = cv.getPerspectiveTransform(source_points,output_points)
    dst = cv.warpPerspective(img,M,(maxX,maxY))
    # cv.imshow("Display window", dst)
    # k = cv.waitKey(0)
    return dst

img_name = "karta_odp_wyp.jpg"

positions,template = karta_detect_rects(img_name)
# print("positions",*positions)
corners = find_corners(list(positions), template)
print("corners")
print(corners)

transformed = karta_transformation(corners,img_name)
# print(transformed)
cv.imwrite("transformed.png", transformed)


def karta_detect_answers(img_gray):
    assert img_gray is not None, "file could not be read, check with os.path.exists()"

    template = cv.imread('answer_template_2.png', cv.IMREAD_GRAYSCALE)
    assert template is not None, "file could not be read, check with os.path.exists()"
    w, h = template.shape[::-1]
    res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
    threshold = 0.5
    loc = np.where( res >= threshold)
    print(len(list(zip(*loc[::-1]))))

    img_rgb = img_gray.copy()
    img_rgb = cv.cvtColor(img_rgb, cv.COLOR_RGB2RGBA)
    for pt in zip(*loc[::-1]):
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    cv.imshow("Display window", img_rgb)
    k = cv.waitKey(0)

    return zip(*loc[::-1]),template.shape



# karta_detect_answers(transformed)
# contours(transformed)


# img = cv.imread("transformed.png", cv.IMREAD_GRAYSCALE)
# img = cv.cvtColor(transformed, cv.COLOR_BGR2GRAY)
# # img = transformed
# assert img is not None, "file could not be read, check with os.path.exists()"
# _,img = cv.threshold(img,120,255,cv.THRESH_BINARY)
# kernel = np.ones((1,1),np.uint8)
# # img = cv.dilate(img, kernel,iterations =1)

# img = cv.dilate(img, np.ones((5,5),np.uint8),iterations =1)
# img = cv.erode(img, (5,5),iterations =5)
# # img  = cv.GaussianBlur(img, (11, 11), 0)
# img = cv.Canny(img, 120, 250, 1)


# (cnt, hierarchy) = cv.findContours(    img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
# rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# cv.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
# print("coins in the image : ", len(cnt))
# cv.imshow("Display window", rgb)
# k = cv.waitKey(0)

img = cv.cvtColor(transformed, cv.COLOR_BGR2GRAY)
img2 = cv.imread('mask3.png')
h,w,_ = img2.shape
img2 = karta_transformation([(0,0), (w,0), (0,h), (w,h)],'mask3.png')
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# cv.imshow("Display Image", img)
# k = cv.waitKey(0)

img = cv.bitwise_or(img2,img)
# cv.imshow("Display bitwise_or", img)
# k = cv.waitKey(0)

# img = cv.Canny(img, 30, 100, 20)
# cv.imshow("Display Canny", img)
# k = cv.waitKey(0)


ret,img = cv.threshold(img,150,255,cv.THRESH_BINARY)
# cv.imshow("Display Threshold", img)
# k = cv.waitKey(0)


img = cv.morphologyEx(img, cv.MORPH_CLOSE, np.ones((5,5),np.uint8))
# cv.imshow("Display MORPH_CLOSE", img)
# k = cv.waitKey(0)

img = cv.erode(img,np.ones((5,5),np.uint8),iterations =4)
# cv.imshow("Display erode", img)
# k = cv.waitKey(0)

# img = cv.dilate(img, np.ones((5,5),np.uint8),iterations = 1)
# cv.imshow("Display dilate", img)
# k = cv.waitKey(0)



img = cv.bitwise_or(img2,img)
# cv.imshow("Display bitwise_or_2", img)
# k = cv.waitKey(0)

img = cv.Canny(img, 30, 100, 20)
# cv.imshow("Display Canny", img)
# k = cv.waitKey(0)

cv.imwrite("transformed.png", img)

(countours, hierarchy) = cv.findContours(img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.drawContours(rgb, countours, -1, (0, 255, 0), 2)
print("coins in the image : ", len(countours))
points = []
for lst in countours:
    mini = lst[0]
    for c in lst:
        if(c[0][0] < mini[0][0] or c[0][1] < mini[0][1]):
            mini = c
    points.append(mini[0])


def ans(xValue):
    if (p[0] >=116 and p[0] <= 120):
        return "A" 
    if (p[0] >=145 and p[0] <= 150):
        return "B" 
    if (p[0] >=174 and p[0] <= 180):
        return "C" 
    if (p[0] >=203 and p[0] <= 210):
        return "D" 
    return ""


for i,p in enumerate(sorted(points, key=lambda p: p[1])):
    print(str(p) + " " +str(i)+ ": " +ans(p[0]))



cv.imshow("Display window", rgb)
k = cv.waitKey(0)



# karta_detect_answers(img)
