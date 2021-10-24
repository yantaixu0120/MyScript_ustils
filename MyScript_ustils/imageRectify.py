
""""#------------------------------------------------#
#----------------------单相机标定---------------------#
#-------------------------------------------------#"""
# encoding:utf-8
import numpy as np
import cv2
import glob
import time
import os

def cameraCalib(imgPath, rectifyImg):
    # 设置终止条件，迭代30次或移动0.001
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # 准备对象点，类似（0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((8 * 11, 3), np.float32)
    objp[:, : 2] = np.mgrid[0: 11, 0: 8].T.reshape(-1, 2)  # np.mgrid()返回多维结构

    # 从所有图像中存储对象点和图像点的数组
    objpoints = []  # 真实世界的3D点
    imgpoints = []  # 图像的2D点

    """glob.globglob.glob函数的参数是字符串。这个字符串的书写和我们使用
    # linux的shell命令相似，或者说基本一样。也就是说，只要我们按照平常使
    # 用cd命令时的参数就能够找到我们所需要的文件的路径。字符串中可以包括“*”、
    # “?”和"["、"]"，其中“*”表示匹配任意字符串，“?”匹配任意单个字符，
    # [0-9]与[a-z]表示匹配0-9的单个数字与a-z的单个字符。"""

    images = glob.glob(os.path.join(imgPath, '*.jpg'))
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(gray.shape[:: -1])
        # 找到棋盘边界，角点检测
        print("start to find Chessboard Corners！--------")
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
        print(fname)
        print("Finded Chessboard Corners or Not: ", ret)
        # 如果找到，则添加对象点和图像点
        if ret == True:
            objpoints.append(objp)
            # 亚像素级角点检测，在角点检测中精确化角点位置
            print("start to find SubPix Corner！--------")
            corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
            imgpoints.append(corners)
            # 绘制并展示边界
            cv2.drawChessboardCorners(img, (11, 8), corners2, ret)
            cv2.namedWindow('img', flags=cv2.WINDOW_NORMAL)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()

    #开始标定
    print("start to calibrate！-----------")
    ret_left, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[:: -1], None, None)
    print (("ret: "),ret)
    print (("mtx: \n"),mtx)        # 内参数矩阵
    print (("dist: \n"),dist)      # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    print (("rvecs: \n"),rvecs)    # 旋转向量  # 外参数
    print (("tvecs: \n"),tvecs)    # 平移向量  # 外参数

    # 畸形校正
    img = cv2.imread(rectifyImg)
    cv2.namedWindow('source',flags= cv2.WINDOW_NORMAL)
    cv2.imshow('source', img)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # 使用cv2.undistort()，和上面得到的ROI对结果进行剪裁
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # 裁剪图像
    x, y, w, h = roi
    dst = dst[y: y + h, x: x + w]
    cv2.namedWindow('re1',flags= cv2.WINDOW_NORMAL)
    cv2.imshow('re1', dst)

    # 找到畸变图像到畸变图像的映射方程，再使用重映射方程
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # 裁剪图像
    x, y, w, h = roi
    dst = dst[y: y + h, x: x + w]
    cv2.namedWindow('re2',flags= cv2.WINDOW_NORMAL)
    cv2.imshow('re2', dst)
    '''
    反向投影误差，我们可以利用反向投影误差对我们找到的参数的准确性评估，
    得到的结果越接近0越好，有了内部参数、畸变参数和旋转变化矩阵，
    就可以使用cv2.projectPoints()将对象转换到图像点
    然后就可以计算变换得到的图像与角点检测算法的绝对差了
    最后计算所有标定图像的误差平均值
    '''
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: ", mean_error / len(objpoints))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


""""#-----------------------------------------------------------#
#----------------------双目校正及计算深度信息---------------------#
#------------------------------------------------------------#"""
def loadDataSet(fileName, indexStart, indexEnd, splitChar=','):
    """
    输入：文件名
    输出：数据集
    描述：从文件读入数据集
    """
    dataSet = []
    with open(fileName) as fr:
        for line in fr.readlines()[indexStart : indexEnd + 1]:
            curline = line.strip().split(splitChar)    #字符串方法strip():返回去除两侧（不包括）内部空格的字符串；字符串方法spilt:按照制定的字符将字符串分割成序列
            fltline = list(map(float, curline))        #list函数将其他类型的序列转换成字符串； map函数将序列curline中的每个元素都转为浮点型
            dataSet.append(fltline)
    return dataSet

def loadClibResult(calibResultPath):
    """
    输入：标定结果文件名
    输出：双目相机标定参数
    描述：从文件读入标定参数
    """
    dataset = loadDataSet(calibResultPath, 1, 3)  #左相机内参数矩阵
    print("第一个dataset：", dataset)
    print("第一个dataset数据类型：", type(dataset))
    C1 = np.array(dataset)
    print("读出的C1: ", C1)
    print("读出的C1数据类型: ", type(C1))

    dataset = loadDataSet(calibResultPath, 6, 6)  #左相机畸变系数
    dist1 = np.array(dataset)
    print("读出的dist1: ", dist1)

    dataset = loadDataSet(calibResultPath, 9, 11)  #右相机内参数矩阵
    C2 = np.array(dataset)
    print("读出的C2: ", C2)

    dataset = loadDataSet(calibResultPath, 14, 14)  #右相机畸变系数
    dist2 = np.array(dataset)
    print("读出的dist2: ", dist2)

    dataset = loadDataSet(calibResultPath, 17, 19)  #旋转矩阵
    R = np.array(dataset)
    print("读出的R: ", R)

    dataset = loadDataSet(calibResultPath, 22, 24)  #平移矩阵
    T = np.array(dataset)
    print("读出的T: ", T)

    return C1, dist1, C2, dist2, R, T

def stereoRectify0(imageLeft, imageRight, C1, dist1, C2, dist2, size, R, T):
#def computeRectifyMap(C1, dist1, C2, dist2, size, R, T):
    """
   输入：
        待校正的左右相机图像
        标定的双目参数
   输出：
        连接后的原图像
        校正后的左右图像
        用于计算深度信息的映射矩阵
   描述：双目校正
   """
    rectifyStartTime = time.time()
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(C1, dist1, C2, dist2, size, R, T, alpha = -1)
    print("validPixROI1: ", validPixROI1)

    """
    输出参数说明：
        R1-输出矩阵，第一个摄像机的校正变换矩阵（旋转变换），R2-输出矩阵，第二个摄像机的校正变换矩阵（旋转矩阵）
        P1-输出矩阵，第一个摄像机在新坐标系下的投影矩阵，P2-输出矩阵，第二个摄像机在想坐标系下的投影矩阵
        Q-4*4的深度差异映射矩阵
    """
    rempStartTime = time.time()
    print("rectify time: "+str(rempStartTime - rectifyStartTime))
    left_map1, left_map2 = cv2.initUndistortRectifyMap(C1, dist1, R1, P1, size, cv2.CV_16SC2)  #为什么有两个map，x、y方向的map不一样,cv2.CV_16SC2
    right_map1, right_map2 = cv2.initUndistortRectifyMap(C2, dist2, R2, P2, size, cv2.CV_16SC2)
    print("remap comput time: "+str(time.time() - rempStartTime))

    print('left_map1: ',left_map1)
    print('left_map1数据类型', type(left_map1))
    print('left_map1尺寸: ', left_map1.shape)
    print('left_map2尺寸: ', left_map2.shape)
    print('right_map1尺寸: ', right_map1.shape)
    print('right_map2尺寸: ', right_map2.shape)

    sourceImage = np.concatenate([imageLeft, imageRight], axis=1)
    print("合并原始图像后尺寸：", sourceImage.shape)
    for i in range(0, sourceImage.shape[0], 32):
        cv2.line(sourceImage, (0, i), (sourceImage.shape[1], i), (0, 255, 0), 1, 8)

    img1_rectified = cv2.remap(imageLeft, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imageRight, right_map1, right_map2, cv2.INTER_LINEAR)

    image = np.concatenate([img1_rectified, img2_rectified], axis=1)
    iamge_size = image.shape
    print("合并校正图像后尺寸：", image.shape)
    for i in range(0, iamge_size[0], 32):
        cv2.line(image, (0, i), (iamge_size[1], i), (0, 255, 0), 1, 8)

    return sourceImage, img1_rectified, img2_rectified, image, Q

    #return left_map1, left_map2, right_map1, right_map2, Q, validPixROI1, validPixROI2

def stereoRectify(imageLeft, imageRight, left_map1, left_map2, right_map1, right_map2, validPixROI1, validPixROI2):
    sourceImage = np.concatenate([imageLeft, imageRight], axis=1)
    print("合并原始图像后尺寸：", sourceImage.shape)
    for i in range(0, sourceImage.shape[0], 32):
        cv2.line(sourceImage, (0, i), (sourceImage.shape[1], i), (0, 255, 0), 1, 8)

    img1_rectified = cv2.remap(imageLeft, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imageRight, right_map1, right_map2, cv2.INTER_LINEAR)

    # x1, y1, w1, h1 = validPixROI1
    # x2, y2, w2, h2 = validPixROI2
    # img1_rectified = img1_rectified[y1 : y1 + h1, x1 : x1 +w1]
    # img2_rectified = img2_rectified[y2: y2 + h2, x2: x2 + w2]
    # print('img1_rectified尺寸： ',img1_rectified.shape)
    # print('img2_rectified尺寸： ',img2_rectified.shape)

    rectifiedImage = np.concatenate([img1_rectified, img2_rectified], axis=1)
    iamge_size = rectifiedImage.shape
    print("合并校正图像后尺寸：", rectifiedImage.shape)
    for i in range(0, iamge_size[0], 32):
        cv2.line(rectifiedImage, (0, i), (iamge_size[1], i), (0, 255, 0), 1, 8)

    return sourceImage, img1_rectified, img2_rectified, rectifiedImage

def callbackFunc(e, x, y, f, threeD):
    if e == cv2.EVENT_LBUTTONDOWN:
        print("y and x:", y, x, f)
        print(threeD[y][x])

def calcDepthInfo(imgLeft_rectified, imgRight_rectified, Q, sgbmFlag):

    cv2.namedWindow('depth', 30)
    cv2.setMouseCallback('depth', callbackFunc, None)

    num = cv2.getTrackbarPos('num', 'depth')
    blockSize = cv2.getTrackbarPos('blockSize', 'depth')
    if blockSize % 2 == 0:
        blockSize += 1
    if blockSize < 5:
        blockSize = 5

    #------SGBM立体匹配算法参数及创建---------#
    if sgbmFlag:
        imgL = imgLeft_rectified
        imgR = imgRight_rectified

        minDisparity = 0
        numDisparities = 64
        blockSize = 5

        SADWindowSize = 7
        cn = imgL.shape[2]
        P1 = 8 * cn * SADWindowSize * SADWindowSize
        P2 = 32 * cn * SADWindowSize * SADWindowSize
        disp12MaxDiff = -1
        preFilterCap = 64
        uniquenessRatio = 25
        speckleWindowSize = 32
        speckleRange = 32
        #mode = cv2.StereoSGBM.MODE_SGBM

        stereo_sgbm = cv2.StereoSGBM_create(
            minDisparity = minDisparity,
            numDisparities = numDisparities,
            blockSize = blockSize,
            P1 = P1,
            P2 = P2,
            disp12MaxDiff = disp12MaxDiff,
            preFilterCap = preFilterCap,
            uniquenessRatio = uniquenessRatio,
            speckleWindowSize = speckleWindowSize,
            speckleRange = speckleRange,
            #mode = mode   #此处mode用默认值
        )
        disparity = stereo_sgbm.compute(imgL, imgR)  # sgbm算法计算
    else:
        stereo = cv2.StereoBM_create(numDisparities = 0, blockSize = 5)
        imgL = cv2.cvtColor(imgLeft_rectified, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(imgRight_rectified, cv2.COLOR_BGR2GRAY)
        disparity = stereo.compute(imgL, imgR)

    disp = cv2.normalize(disparity, disparity, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)
    threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16., Q)  # 此三维坐标点的基准坐标系为左侧相机坐标系

    cv2.imshow('depth', threeD)
    return disp

if __name__ == '__main__':
    calibResultPath = '/home/wt/XinJiangDianShiLu/MyTestData/CalibrationData/20200401/calibResult/calibResult_stereo.txt'
    C1, dist1, C2, dist2, R, T = loadClibResult (calibResultPath)   #加载已获得的标定参数

    imageLeft = cv2.imread('/home/wt/XinJiangDianShiLu/MyTestData/CalibrationData/20200401/VI/0.jpg')
    imageRight = cv2.imread('/home/wt/XinJiangDianShiLu/MyTestData/CalibrationData/20200401/IR/0.jpg')
    size = (imageLeft.shape[1], imageLeft.shape[0])

    sourceImage, imgLeft_rectified, imgRight_rectified, rectifiedImage, Q = stereoRectify0(imageLeft, imageRight, C1, dist1, C2, dist2, size, R, T)  #图像校正

    # left_map1, left_map2, right_map1, right_map2, Q, validPixROI1, validPixROI2 = computeRectifyMap(C1, dist1, C2, dist2, size, R, T)
    # print('左图像校正重映射map：', left_map1.shape)
    # sourceImage, img1_rectified, img2_rectified, rectifiedImage = stereoRectify(imageLeft, imageRight, left_map1, left_map2, right_map1, right_map2,
    #                                                                             validPixROI1, validPixROI2)
    cv2.namedWindow('sourceImage', flags = cv2.WINDOW_NORMAL)
    cv2.imshow('sourceImage', sourceImage)
    cv2.namedWindow("rectifiedImage", cv2.WINDOW_GUI_NORMAL)
    cv2.imshow('rectifiedImage', rectifiedImage)
    #cv2.imwrite('/home/wt/XinJiangDianShiLu/MyTestData/CalibrationData/matchData/rectifiedImage.jpg', rectifiedImage)

    # cv2.namedWindow('img1_rectified', flags=cv2.WINDOW_NORMAL)
    # cv2.imshow('img1_rectified', img1_rectified)
    # cv2.namedWindow("img2_rectified", cv2.WINDOW_GUI_NORMAL)
    # cv2.imshow('img2_rectified', img2_rectified)

    #disp = calcDepthInfo(img1_rectified, img2_rectified, Q, sgbmFlag = True)  #计算深度信息的操作

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # imgPath = '/home/wt/XinJiangDianShiLu/MyTestData/CalibrationData/20200401/IR_noPadded/'
    # rectifyImg = '/home/wt/XinJiangDianShiLu/MyTestData/CalibrationData/20200401/IR_noPadded/0.jpg'
    # cameraCalib(imgPath, rectifyImg)