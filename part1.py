# Camera calibration demo
# code modified from https://docs.opencv.org/4.5.1/dc/dbb/tutorial_py_calibration.html
import os 
import cv2
import numpy as np

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((12*13,3), np.float32)
objp[:,:2] = np.mgrid[0:13,0:12].T.reshape(-1,2)

object_points = []
image_points = []

for file_name in os.listdir('./chessboard_example'):
    if '.tif' not in file_name:
        continue
    img_path = r'./chessboard_example/{}'.format(file_name)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (13,12))
    if found:
        print("found corners in {}".format(file_name))
        object_points.append(objp)
        image_points.append(corners)
        corners_to_mark = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(img, (13,12), corners_to_mark, found)
        cv2.imwrite('./chessboard_example/marked/{}'.format(file_name), img)

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

# read test sample for undistorting
img = cv2.imread('./chessboard_example/Image8.tif')
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w,h), 1, (w,h))

# undistort
dst = cv2.undistort(img, cameraMatrix, distCoeffs, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('undistored_Image8.tif', dst)