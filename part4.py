# Augmented Reality in opencv-python

# Camera calibration code modified from https://docs.opencv.org/4.5.1/dc/dbb/tutorial_py_calibration.html
import os 
import cv2
import numpy as np

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

object_points = []
image_points = []

chessboard_images = ['./diy_chessboard/{}.jpg'.format(i) for i in range(20)]

IMG_NUM = 20

for file_name in chessboard_images:
    if '.jpg' not in file_name:
        continue
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (8,6))
    if found:
        print("found corners in {}".format(file_name))
        object_points.append(objp)
        image_points.append(corners)
        corners_to_mark = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(img, (8,6), corners_to_mark, found)
        #cv2.imwrite('./diy_chessboard/marked/{}'.format(file_name), img)

# camera calibration
retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
rotation_matrixs = [cv2.Rodrigues(rvec) for rvec in rvecs]


# Augemented Reality

target_files = ['anaya.jpg','ar_pic.jpg','tower.jpg'] 
back_imgs = ['./diy_chessboard/0.jpg', './diy_chessboard/7.jpg', './diy_chessboard/5.jpg']
indexes = [0,7,5]
results_imgs = ['result_anaya.jpg','result_ar_pic.jpg','result_tower.jpg']


for target_file, back_img_file, index, result_img in zip(target_files, back_imgs, indexes, results_imgs):

    chessboard_image = cv2.imread('./chessboard.jpg')
    h_0, w_0 = chessboard_image.shape[:2]
    #print(h_0,w_0) #450,350

    target_image = cv2.imread(target_file)
    target_image = cv2.resize(target_image,(w_0,h_0))

    src_points = []

    for i in range(h_0):
        for j in  range(w_0):
            src_h = i/h_0*7
            src_w = j/w_0*5
            src_points.append([[src_h,src_w,0.0]])

    back_img = cv2.imread(back_img_file)

    src_points = np.concatenate(src_points,axis=0).astype(np.float32)
    #print(src_points.shape)
    projectPoints, _ = cv2.projectPoints(src_points, rvecs[index], tvecs[index], cameraMatrix, distCoeffs)

    #print(back_img.shape)
    #print(target_image.shape)

    for i in range(h_0):
        for j in range(w_0):
            project_h = round(projectPoints[i*w_0+j][0][0])
            project_w = round(projectPoints[i*w_0+j][0][1])
            back_img[project_w,project_h,:] = target_image[i,w_0-j-1,:]
            
            for p in range(-2,2):
                for q in range(-2,2):
                    new_h = project_h + p
                    new_w = project_w + q 
                    if new_h < 0 or new_w < 0 or new_h > 1440 or new_w > 1080:
                        continue
                    back_img[new_w,new_h,:] = target_image[i,w_0-j-1,:]
            
            
    cv2.imwrite(result_img, back_img)




