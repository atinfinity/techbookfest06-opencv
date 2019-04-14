import glob
import cv2
import numpy as np

PATTERN_SIZE = (9, 6)
objp = np.zeros((PATTERN_SIZE[0]*PATTERN_SIZE[1], 3), np.float32) 
objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)

obj_points = []
img_points = []

filelist = glob.glob('images/*.jpg')
for filename in filelist:
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, PATTERN_SIZE)

    if found == True:
        img_points.append(corners)
        obj_points.append(objp)

ret, camera_matrix, distortion, rvecs, tvecs, _ = cv2.calibrateCameraRO(obj_points, img_points, gray.shape[::-1], -1, None, None)
print('reprojection error:\n', ret)
print('camera matrix:\n', camera_matrix)
print('distortion:\n', distortion)
print('rvecs:\n', rvecs[0].shape)
print('tvecs:\n', tvecs[0].shape)

