import cv2
from cv2 import threshold
import dlib
import numpy as np
from preprocessing import bgremoval

#####################################

scales = [0 , 7 , 10, 5 , 3 , 0]
test_route = "img.png"

#####################################

def getdistance(g):
    x = round((1/(g**2+1))**0.5,4)
    y = round(g * x,4)
    
    return [x,y]

def warpTriangle(img1, img2, pts1, pts2):
    x1,y1,w1,h1 = cv2.boundingRect(np.float32([pts1]))
    x2,y2,w2,h2 = cv2.boundingRect(np.float32([pts2]))
        
    roi1 = img1[y1:y1+h1, x1:x1+w1]
    roi2 = img2[y2:y2+h2, x2:x2+w2]
    
    offset1 = np.zeros((3,2), dtype=np.float32)
    offset2 = np.zeros((3,2), dtype=np.float32)
    for i in range(3):
        offset1[i][0], offset1[i][1] = pts1[i][0]-x1, pts1[i][1]-y1
        offset2[i][0], offset2[i][1] = pts2[i][0]-x2, pts2[i][1]-y2
    
    mtrx = cv2.getAffineTransform(offset1, offset2)
    warped = cv2.warpAffine( roi1, mtrx, (w2, h2), None, \
                        cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101 )
    
    mask = np.zeros((h2, w2), dtype = np.uint8)
    cv2.fillConvexPoly(mask, np.int32(offset2), (255))
    
    mask2 = cv2.bitwise_not(mask)
    
    warped_masked = cv2.bitwise_and(warped, warped, mask=mask)
    roi2_masked = cv2.bitwise_and(roi2, roi2, mask=mask2)
    roi2_masked = roi2_masked + warped_masked
    img2[y2:y2+h2, x2:x2+w2] = roi2_masked


triangles_i = []
mp = []
contours = []
contours2 = []
contours_l = []
moved_v = []
border = []
b_l = [0] * 6
b_h = [0] * 6


original_img = cv2.imread(test_route)
original_img2 = original_img.copy()
pose_border,landmarks,img = bgremoval(test_route)
_,img = threshold(img,253,255,cv2.THRESH_BINARY_INV)

height,width,_ = img.shape

necky = pose_border[0] * height


for x in pose_border:
    border.append(int(x*height))
        

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

temp = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

sum = 0
for x in range(len(temp)):
    for y in range(len(temp[x])):
        for z in range(len(temp[x][y])):   
            if sum == 5:
                contours.append(temp[x][y][z][0])
                sum = 0
            else:
                sum += 1
                    
contours = contours[:len(contours)-1]

for x in contours:
    contours_l.append([x[0],x[1]])
    
contours = []

for x in contours_l:
    contours.append([x[0],x[1]])
    
for x in landmarks:
    contours_l.append([x[0],x[1]])
        
subdiv = cv2.Subdiv2D((0,0,width,height))

for x in contours_l:
    subdiv.insert((x[0],x[1]))

triangleList = subdiv.getTriangleList()

    
for t in triangleList:
    triangles_i.append([contours_l.index([int(t[0]),int(t[1])]),contours_l.index([int(t[2]),int(t[3])]),contours_l.index([int(t[4]),int(t[5])])])
    pts = t.reshape(-1,2).astype(np.int32)
    xsum = 0
    ysum = 0
    for x in range(3):
        xsum += pts[x][0]
        ysum += pts[x][1]

    if img[int(ysum//3),int(xsum/3)][0] == 0:
        continue
    if (pts<0).sum() or (pts[:,0] > width).sum() or (pts[:,1] > height).sum():
        continue
        
contours_l2 = []
for x in contours_l:
    contours_l2.append(x)
        
mp = []
for w in range(len(contours)-1):
    p1_x = contours[w][0]
    p1_y = contours[w][1]
    
    p2_x = contours[w+1][0]
    p2_y = contours[w+1][1]
    
    p3_x = (p1_x+p2_x) // 2
    p3_y = (p1_y+p2_y) // 2
    
    if p1_x == p2_x:
        mp = [1,0]
        if 0 < p3_x+int(mp[0]*10) <width and 0 < p3_y+int(mp[1]*10) < height:
            if img[p3_y+int(mp[1]*10)][p3_x+int(mp[0]*10)][0] == 255:
                mp = [-1,0]
    elif p1_y == p2_y:
        mp = [0,1]
        if 0 < p3_x+int(mp[0]*10) <width and 0 < p3_y+int(mp[1]*10) < height:
            if img[p3_y+int(mp[1]*10)][p3_x+int(mp[0]*10)][0] == 255:
                mp = [0,-1]
    else:
        gradient = round(1 / ((contours[w+1][1] - contours[w][1]) / (contours[w+1][0] - contours[w][0])),4)
        mp = getdistance(gradient)
        if 0 < p3_x+int(mp[0]*10) <width and 0 < p3_y+int(mp[1]*10) < height:
            if img[p3_y+int(mp[1]*10)][p3_x+int(mp[0]*10)][0] == 255:
                
                mp[0] = -mp[0]
                mp[1] = -mp[1]
    moved_v.append(mp)
    
for m in range(len(contours)-2):
    scale = 0
    if m == 0:
        mvx = moved_v[m][0] + moved_v[-1][0] / 2
        mvy = moved_v[m][1] + moved_v[-1][1] / 2
    else:
        mvx = moved_v[m][0] + moved_v[m+1][0] / 2
        mvy = moved_v[m][1] + moved_v[m+1][1] / 2
        
        
    for x in range(5):
        l = border[x+1] - border[x]
        if border[x] <= contours[m][1] < border[x+1]:
            scale = (contours[m][1] - border[x]) / l* scales[x+1] + (border[x+1] - contours[m][1])/l*scales[x]
        
    if 0 <= contours[m][0] + mvx * scale < width and 0 <= contours[m][1] + mvy * scale < height:
        contours2.append([int(contours[m][0]+mvx*scale),int(contours[m][1] + mvy * scale)])
    
    
for x in range(len(contours2)):
    contours_l2[x] = contours2[x]
    
    

for t in triangles_i:
    skip = 0
    p1 = [contours_l[t[0]],contours_l[t[1]],contours_l[t[2]]]
    p2 = [contours_l2[t[0]],contours_l2[t[1]],contours_l2[t[2]]]
    
    for x in p1:
        if x[1] < necky:
            skip = 1
        if x[1] > min(landmarks[32][1],landmarks[31][1]):
            skip = 1
    
    for x in p2:
        if x[1] < necky:
            skip = 1
        if x[1] > min(landmarks[32][1],landmarks[31][1]):
            skip = 1
            
    if skip == 0:
        warpTriangle(original_img,original_img2,p1,p2)
           
    
cv2.imshow("org",original_img)    
cv2.imshow("img",original_img2)
cv2.imwrite("result.jpg",original_img2)
cv2.waitKey(000)
cv2.destroyAllWindows()