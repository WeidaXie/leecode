import cv2
import numpy as np
import math
import os

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 5,
                       blockSize = 7 )

mask1_path = r'E:\Research\xiehe\Data\pre_mask\resunet_attention\wang wenli\wwli-50.png'
mask2_path = r'E:\Research\xiehe\Data\pre_mask\resunet_attention\wang wenli\wwli-54.png'
# image1 = cv2.imread('./image/wwli-50.png', 0)
# image2 = cv2.imread('./image/wwli-54.png', 0)
# mask1 = cv2.imread('./mask/wwli-50.png', 0)
# mask2 = cv2.imread('./mask/wwli-54.png', 0)
mask3 = cv2.imread(mask1_path)
mask4 = cv2.imread(mask2_path)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

image_path = r'E:\Research\xiehe\Data\xiehedata\val\wang wenli\newImage'
mask_path = r'E:\Research\xiehe\Data\xiehedata\val\wang wenli\newMask'
image_list = os.listdir(image_path)
len = len(image_list)
image_number = []
new_image_list = [0] * len
for index in range(0, len):
    number = image_list[index].split('.')[0].split('-')[1]
    image_number.append(int(number))
image_number.sort()
for index in range(0, len):
    new_image_list[index] = image_list[0].split('-')[0] + '-' + str(image_number[index]) \
                                      + '.png'
for index in range(0, len):
    image1 = cv2.imread(os.path.join(image_path, new_image_list[index]))
    if index + 1 < len:
        image2 = cv2.imread(os.path.join(image_path, new_image_list[index + 1]))
    else:
        image2 = cv2.imread(os.path.join(image_path, new_image_list[index]))
    mask1 = cv2.imread(os.path.join(mask_path, new_image_list[index]))
    if index + 1 < len:
        mask2 = cv2.imread(os.path.join(mask_path, new_image_list[index + 1]))
    else:
        mask2 = cv2.imread(os.path.join(mask_path, new_image_list[index]))
# Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    hsv = np.zeros_like(image1)

    # 遍历每一行的第1列
    hsv[..., 1] = 255
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
    mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
    # 返回一个两通道的光流向量，实际上是每个点的像素位移值
    flow = cv2.calcOpticalFlowFarneback(image1, image2, None, 0.5, 3, 30, 3, 5, 1.2, 0)
    flow2 = cv2.calcOpticalFlowFarneback(mask1, mask2, None, 0.5, 3, 30, 3, 5, 1.2, 0)
    # flow = flow.transpose(2, 0, 1)
    # print(flow.shape)
    # print(flow)
    step = 10
    # 绘制线
    h, w = image2.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)

    line = []
    for l in lines:
        if l[0][0]-l[1][0]>3 or l[0][1]-l[1][1]>3:
            line.append(l)
    # 笛卡尔坐标转换为极坐标，获得极轴和极角
    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # mag1, ang1 = cv2.cartToPolar(flow2[..., 0], flow2[..., 1])
    #
    # hsv[..., 0] = (ang-ang1) * 180 / np.pi / 2  # 角度
    # hsv[..., 2] = cv2.normalize(mag-mag1, None, 0, 255, cv2.NORM_MINMAX)
    # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # cv2.putText(image2, str(image_number[index]) + '.png', (5, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (125, 125, 255), 2)
    # cv2.putText(bgr, str(image_number[index]) + '.png', (5, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (125, 125, 255), 2)
    # cv2.imshow('frame2', bgr)
    # cv2.imshow("frame1", image2)

    cv2.putText(image2, str(image_number[index]) + '.png', (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (125, 125, 255), 2)
    cv2.polylines(image2, line, 0, (125,255,125))
    cv2.imshow('image_flow', image2)

    cv2.waitKey(0)
cv2.destroyAllWindows()

