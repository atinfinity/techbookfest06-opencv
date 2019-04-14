import cv2
import numpy as np

def display(im, bbox):
    n = len(bbox)
    for j in range(n):
        cv2.line(im, tuple(bbox[j][0]), tuple(bbox[ (j+1) % n][0]), (255, 0, 0), 3)
    cv2.imshow('Result of detection', im)

img = cv2.imread('link_github_ocv.jpg', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# インスタンス生成
qrDecoder = cv2.QRCodeDetector()

# QRコード検出、デコード
data, bbox, rect_img = qrDecoder.detectAndDecode(gray)

if len(data) > 0:
	# 検出結果表示
    display(img, bbox)

    # デコード結果表示
    print('Decoded Data : {}'.format(data))

    # rectify画像表示
    rect_img = np.uint8(rect_img);
    cv2.imshow('Rectified QRCode', rect_img);
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('Could not detect')
