import cv2

img = cv2.imread('left01.jpg', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

PATTERN_SIZE = (9, 6)
#found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE)
found, corners = cv2.findChessboardCornersSB(gray, PATTERN_SIZE)

if found:
    result = img.copy()
    cv2.drawChessboardCorners(result, PATTERN_SIZE, corners, found)
    cv2.imshow('Result of detection', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('Could not detect')
