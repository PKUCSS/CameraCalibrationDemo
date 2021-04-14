# Camera calibration demo
# code modified from https://docs.opencv.org/4.5.1/dc/dbb/tutorial_py_calibration.html
import os 
import cv2
import numpy as np

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

object_points = []
image_points = []


IMG_NUM = 5

for file_name in sorted(os.listdir('./diy_chessboard'))[:IMG_NUM]:
    if '.jpg' not in file_name:
        continue
    img_path = r'./diy_chessboard/{}'.format(file_name)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (8,6))
    if found:
        print("found corners in {}".format(file_name))
        object_points.append(objp)
        image_points.append(corners)
        corners_to_mark = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(img, (8,6), corners_to_mark, found)
        cv2.imwrite('./diy_chessboard/marked/{}'.format(file_name), img)

# camera calibration
retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
rotation_matrixs = [cv2.Rodrigues(rvec) for rvec in rvecs]


print("retval: {}".format(retval))
print("cameraMatrix:")
print(cameraMatrix)
print("distCoeffs:")
print(distCoeffs)
print("rvecs:")
print(rvecs)
print("tvecs:")
print(tvecs)
print("rotation_matrixs:")
print(rotation_matrixs)


# Part3: Evaluation

import matplotlib.pyplot as plt 
index_list = [str(i) for i in range(IMG_NUM)]
errors_list = []
mean_error = 0
for i in range(len(object_points)):
    recovered_points, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
    error = cv2.norm(image_points[i], recovered_points, cv2.NORM_L2)/len(recovered_points)
    mean_error += error
    errors_list.append(error)

mean_error /= IMG_NUM
print( "mean error: {}".format(mean_error) )
plt.hlines(mean_error,0,20,colors='r', linestyles = "dashed",label='Mean Error = {:.4f}'.format(mean_error))
plt.bar(index_list, errors_list,fc='blue')
plt.xlabel("Image Index")
plt.ylabel("Mean Error in Pixels")
plt.legend()
plt.savefig("eval_{}.jpg".format(IMG_NUM))







