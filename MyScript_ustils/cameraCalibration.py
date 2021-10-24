""""#------------------------------------------------#
#---------------红外可见光图像叠加融合------------------#
#------------------------------------------------#"""
import cv2
import os

def imageProcess(IRImage, VIImage, diff_x, diff_y):
    h_in, w_in, _ = IRImage.shape
    h_vi, w_vi, _ = VIImage.shape
    proImage = VIImage.copy()
    start_x = diff_x
    start_y = diff_y
    for x in range(start_x, start_x + h_in):
        for y in range(start_y, start_y + w_in):
            proImage[x, y, :] = IRImage[x - start_x, y - start_y, :]
    return proImage

########-----------标定图像------------####
# fx = 37 / 31
# fy = 40 / 34
#
# diff_x = 481 - round(109*fy)
# diff_y = 851 - round(153*fx)
#
# diff_x1 = 481 - 109
# diff_y1 = 851 - 153
########-----------0318------------####
# fx = 230 / 211
# fy = 86 / 85
#
# diff_x = 487 - round(168*fy)
# diff_y = 748 - round(265*fx)
#
# diff_x1 = 487 - 168
# diff_y1 = 748 - 265

########-----------0323------------####
# fx = 307 / 284
# fy = 265 / 228
#
# diff_x0 = 707 - round(370*fy)
# diff_y0 = 882 - round(204*fx)
# print ("diff_x0 = %d,diff_y0 = %d"%(diff_x0,diff_y0))
# diff_x = 717 - 429
# diff_y = 943 - 281
# print ("diff_x = %d,diff_y = %d"%(diff_x,diff_y))
#
# diff_x1 = 707 - 370
# diff_y1 = 882 - 204


fx = 156 / 136
fy = 390 / 339

diff_x0 = 748 - round(396*fy)
diff_y0 = 803 - round(139*fx)
print ("diff_x0 = %d,diff_y0 = %d"%(diff_x0,diff_y0))
diff_x = 590 - 300
diff_y = 1191 - 549
print ("diff_x = %d,diff_y = %d"%(diff_x,diff_y))

diff_x1 = 645 - 308
diff_y1 = 1027 - 333

#IRImage = cv2.imread('/home/wt/XinJiangDianShiLu/MyTestData/CalibrationData/20200316/Processed/IR/2.jpg')  #0316标定图像
#IRImage = cv2.imread('/home/wt/XinJiangDianShiLu/MyTestData/frusedTestImage/0318/IR/2.jpg')  #0318

#VIImage = cv2.imread('/home/wt/XinJiangDianShiLu/MyTestData/CalibrationData/20200316/Processed/VI/2.jpg')
#VImage = cv2.imread('/home/wt/XinJiangDianShiLu/MyTestData/frusedTestImage/0318/VI/2.jpg')


IRImage = cv2.imread('D:/YanXu/XinJiangDianShiChangData/fuseTestImage/0323/IR/0.jpg')  #0318
print(IRImage.shape)

InImageResized = cv2.resize(IRImage, None, dst=None, fx=fx, fy=fy)
print(InImageResized.shape)


VIImage = cv2.imread('D:/YanXu/XinJiangDianShiChangData/fuseTestImage/0323/VI/0.jpg')

proImage = imageProcess(InImageResized, VIImage, diff_x, diff_y)
proImageNoScale = imageProcess(IRImage, VIImage, diff_x1, diff_y1)

cv2.namedWindow('InImageResized',flags = cv2.WINDOW_NORMAL)
cv2.imshow('InImageResized',InImageResized)

cv2.namedWindow('infraredImage',flags = cv2.WINDOW_NORMAL)
cv2.imshow('infraredImage',IRImage)
cv2.namedWindow('visiableImage',flags = cv2.WINDOW_NORMAL)
cv2.imshow('visiableImage',VIImage)

cv2.namedWindow('processedImage',flags = cv2.WINDOW_NORMAL)
cv2.imshow('processedImage', proImage)
cv2.namedWindow('proImageNoScale',flags = cv2.WINDOW_NORMAL)
cv2.imshow('proImageNoScale', proImageNoScale)

cv2.waitKey(0)


""""#-----------------------------------------------------------------#
#---------------Python-OpenCV单目相机标定并计算标定误差------------------#
#------------------------------------------------------------------#"""
# import cv2
# import numpy as np
# import glob
#
# # 找棋盘格角点，阈值
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# #棋盘格模板规格
# w = 12
# h = 9
# # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
# objp = np.zeros((w*h, 3), np.float32)
# objp[:,:2] = np.mgrid[0:w, 0:h].T.reshape(-1,2)  #自动转换为有2列的数据
# # 储存棋盘格角点的世界坐标和图像坐标对
# objpoints = [] # 在世界坐标系中的三维点
# imgpoints = [] # 在图像平面的二维点
#
# images = glob.glob("./calibData/ImageList_1/left/*.bmp")
# for fname in images:
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     # 找到棋盘格角点
#     print("find Chessboard Corners--------")
#     ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
#     # 如果找到足够点对，将其存储起来
#     if ret == True:
#         cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
#         objpoints.append(objp)
#         imgpoints.append(corners)
#         # 将角点在图像上显示
#         cv2.drawChessboardCorners(img, (w,h), corners, ret)
#         cv2.namedWindow('findCorners', flags=cv2.WINDOW_NORMAL)
#         cv2.imshow('findCorners',img)
#         cv2.waitKey(500)
# cv2.destroyAllWindows()
# print("start to calibration---------")
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# print (("ret:"),ret)
# print (("mtx:\n"),mtx)        # 内参数矩阵
# print (("dist:\n"),dist)      # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
# print (("rvecs:\n"),rvecs)    # 旋转向量  # 外参数
# print (("tvecs:\n"),tvecs)    # 平移向量  # 外参数
# # 去畸变
#
# img2 = cv2.imread("./matchData/cam01_01.bmp")
# h,w = img2.shape[:2]
# newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h)) # 自由比例参数
# dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
# # 根据前面ROI区域裁剪图片
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imwrite('calibresult.jpg',dst)
#
# # 反投影误差
# total_error = 0
# for i in range(len(objpoints)):
#     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#     error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
#     total_error += error
# print ("total error: " , total_error/len(objpoints))



""""#------------------------------------------------#
#---------------Python-OpenCV相机标定及获取深度信息------------------#
#------------------------------------------------#"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import glob
import os

def camerClibrat(imagePathVI, imagePathIR, resultVI, resultIR, resultStereo):
    """进行相机标定，将标定结果存储在txt中，无返回值"""
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    board_size = [11,8]
    scale = 30
    objp = np.zeros((board_size[1] * board_size[0], 3), np.float32)
    # print(objp)
    #objp[:, :2] = np.mgrid[0:(board_size[0]-1)*scale:complex(0,board_size[0]), 0:(board_size[1]-1)*scale:complex(0,board_size[1])].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    objp[:, :2] = np.mgrid[0:(board_size[0])*scale:complex(0,board_size[0]), 0:(board_size[1])*scale:complex(0,board_size[1])].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    # print(objp[:, :2])
    obj_points = []  #存储3D点
    img_points = []  #存储左侧相机2D点
    img_points_r = []  #存储右侧相机2D点

    """左相机内参标定"""
    imagesVI = glob.glob(os.path.join(imagePathVI, '*.jpg'))
    imagesVI.sort(key = lambda x: (int(x[len(imagePathVI) : -4])))
    print("-------------------start to calibrat left camera!-----------------------")
    for fname in imagesVI:
        img = cv2.imread(fname)
        print("原始图像尺寸: ", img.shape)
        #img = cv2.resize(img, (720, 1280))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        print(size)
        ret, corners = cv2.findChessboardCorners(gray, (board_size[0], board_size[1]), None)
        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria) # 在原角点的基础上寻找亚像素角点
        if corners2.any:
            img_points.append(corners2/1.0)
        else:
            img_points.append(corners/1.0)
        cv2.drawChessboardCorners(img, (board_size[0], board_size[1]), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
        cv2.namedWindow('img_left', 30)
        cv2.imshow('img_left', img)
        cv2.waitKey(500)
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None,flags = 0 )
    print("-------------------对右相机1标定结果进行保存-----------------------")
    with open(resultVI, 'a') as txt:
        txt.write("mtx: ")
        txt.write('\n')
        np.savetxt(txt, mtx, fmt='%f',delimiter=',')  #有两种写入数据的方式
        txt.write('\n')

        txt.write("dist: ")
        txt.write('\n')
        np.savetxt(txt, dist, fmt='%f', delimiter=',')
        #txt.write('\n')

        # txt.write("rvecs: ")
        # txt.write('\n')
        # np.savetxt(txt, rvecs, fmt='%f', delimiter=',')
        # txt.write('\n')
        #
        # txt.write("tvecs: ")
        # txt.write('\n')
        # np.savetxt(txt, tvecs, fmt='%f', delimiter=',')

    print (("ret:"),ret)
    print (("mtx:\n"),mtx)        # 内参数矩阵
    print (("dist:\n"),dist)      # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    #print (("rvecs:\n"),rvecs)    # 旋转向量  # 外参数
    #print (("tvecs:\n"),tvecs)    # 平移向量  # 外参数

    print("-------------------start to calibrat right camera!-----------------------")
    """右相机内参标定"""
    imagesIR = glob.glob(os.path.join(imagePathIR, '*.jpg'))
    imagesIR.sort(key = lambda x: (int(x[len(imagePathIR) : -4])))
    obj_points = []    #存储3D点
    for fname in imagesIR:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (board_size[0], board_size[1]), None)
        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria) # 在原角点的基础上寻找亚像素角点
        if corners2.any:
            img_points_r.append(corners2/1.0)
        else:
            img_points_r.append(corners/1.0)
        cv2.drawChessboardCorners(img, (board_size[0], board_size[1]), corners, ret) # 记住，OpenCV的绘制函数一般无返回值
        cv2.namedWindow('img_right', 30)
        cv2.imshow('img_right', img)
        cv2.waitKey(500)
    cv2.destroyAllWindows()
    ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(obj_points, img_points_r, size, None, None,flags = 0 )
    print (("ret:"),ret1)
    print (("mtx:\n"),mtx1)        # 内参数矩阵
    print (("dist:\n"),dist1)      # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    #print (("rvecs:\n"),rvecs1)    # 旋转向量  # 外参数
    #print (("tvecs:\n"),tvecs1)    # 平移向量  # 外参数
    print("-------------------对右相机2标定结果进行保存----------------------")
    with open(resultIR, 'a') as txt:
        txt.write("mtx: ")
        txt.write('\n')
        np.savetxt(txt, mtx1, fmt='%f',delimiter=',')
        txt.write('\n')

        txt.write("dist: ")
        txt.write('\n')
        np.savetxt(txt, dist1, fmt='%f', delimiter=',')


    """双目立体矫正及左右相机内参进一步修正"""
    print("-------------------start to calibrat stereo camera!-----------------------")
    rms, C1, dist1, C2, dist2, R, T, E,F = cv2.stereoCalibrate(obj_points, img_points, img_points_r,
                                                               mtx, dist, mtx1, dist1, size, flags=cv2.CALIB_USE_INTRINSIC_GUESS )
    print("读出的C1: ",C1)
    print("读出的dist1: ", dist1)
    print("读出的C2: ", C2)
    print("读出的dist2: ", dist2)
    print("读出的R: ", R)
    print("读出的T: ", T)

    print("-------------------对双目标定数据进行保存-----------------------")
    with open(resultStereo, 'a') as txt:
        txt.write("C1: ")
        txt.write('\n')
        np.savetxt(txt, C1, fmt='%f',delimiter=',')
        txt.write('\n')

        txt.write("dist1: ")
        txt.write('\n')
        np.savetxt(txt, dist1, fmt='%f', delimiter=',')
        txt.write('\n')

        txt.write("C2: ")
        txt.write('\n')
        np.savetxt(txt, C2, fmt='%f', delimiter=',')
        txt.write('\n')

        txt.write("dist2: ")
        txt.write('\n')
        np.savetxt(txt, dist2, fmt='%f', delimiter=',')
        txt.write('\n')

        txt.write("R: ")
        txt.write('\n')
        np.savetxt(txt, R, fmt='%f', delimiter=',')
        txt.write('\n')

        txt.write("T: ")
        txt.write('\n')
        np.savetxt(txt, T, fmt='%f', delimiter=',')
    print("平移矩阵：", T)


""""#-------------------------------------------------#
#------------------对图像进行padding--------------------#
#--------------------------------------------------#"""
def image_preporcess(image):  #将图片进行等比例缩放
    ih, iw    = 720, 1280 # resize 尺寸
    IR_H, IR_W, _ = image.shape
    print (IR_H, IR_W)
    image_paded = np.full(shape=[ih, iw, 3], fill_value=200)# 制作一张画布，画布的尺寸就是我们想要的尺寸
    # start_x = int((ih - IR_H) / 2)
    # start_y = int((iw - IR_W) / 2)
    start_x = 200
    start_y = 480
    for x in range(start_x, start_x + IR_H):
        for y in range(start_y, start_y + IR_W):
            image_paded[x, y, :] = image[x - start_x, y - start_y, :]
    return image_paded

#用opencv及numpy库实现padding
def iamgePadding_0(imagePath, imageSavePath):
    data = glob.glob(os.path.join(imagePath, "*.jpg"))
    for i in range((len(data))):
        image = cv2.imread(data[i])
        paddedImage = image_preporcess(image)
        iamgeName = data[i].split('/')[-1]
        saveImgName = os.path.join(imageSavePath,iamgeName)
        cv2.imwrite(saveImgName, paddedImage)

#调用TensorFlow中函数实现
def iamgePadding(imagePath, imageSavePath):
    """对图像进行padding操作，无返回值"""
    data = glob.glob(os.path.join(imagePath, "*.jpg"))

    for i in range((len(data))):
        imageRawData = tf.gfile.FastGFile(data[i], 'rb').read()
        with tf.Session() as sess:
            imageData = tf.image.decode_jpeg(imageRawData)
            padded = tf.image.resize_image_with_crop_or_pad(imageData, 720, 1280)
            encodImage = tf.image.encode_jpeg(padded)
            imageName = data[i].split('/')[-1]
            saveImgName = os.path.join(imageSavePath, imageName)
            with tf.gfile.GFile(saveImgName, 'wb') as f:
                f.write(encodImage.eval())
            # plt.imshow(padded.eval())
            # plt.show()
            # plt.axis('off')

# if __name__ == '__main__':
#
#     imagePathVI = '/home/wt/XinJiangDianShiLu/MyTestData/CalibrationData/20200401/VI/'
#     imagePathIR = '/home/wt/XinJiangDianShiLu/MyTestData/CalibrationData/20200401/IR_padded0/'
#     resultVI = "/home/wt/XinJiangDianShiLu/MyTestData/CalibrationData/20200401/calibResult/calibResult_interVI.txt"
#     resultIR = "/home/wt/XinJiangDianShiLu/MyTestData/CalibrationData/20200401/calibResult/calibResult_interIR.txt"
#     resultStereo = "/home/wt/XinJiangDianShiLu/MyTestData/CalibrationData/20200401/calibResult/calibResult_stereo.txt"
#
#     camerClibrat(imagePathVI, imagePathIR, resultVI, resultIR, resultStereo)
#
    # data_dir = '/home/wt/XinJiangDianShiLu/MyTestData/CalibrationData/20200401/IR_noPadded/'
    # imageSavePath = '/home/wt/XinJiangDianShiLu/MyTestData/CalibrationData/20200401/IR_padded0/'
    # iamgePadding_0(data_dir, imageSavePath)



#
#
# """#立体校正及深度图获取"""
# cv2.namedWindow('depth', 30)
# def callbackFunc(e, x, y, f, p):
#     if e == cv2.EVENT_LBUTTONDOWN:
#         print("y and x:", y, x, f, p)
#
# cv2.setMouseCallback('depth', callbackFunc, None)
#
# R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(C1, dist1, C2, dist2, size, R, T, alpha = -1)
# left_map1, left_map2 = cv2.initUndistortRectifyMap(C1, dist1, R1, P1, size, cv2.CV_16SC2)
# right_map1, right_map2 = cv2.initUndistortRectifyMap(C2, dist2, R2, P2, size, cv2.CV_16SC2)
# frame1 = cv2.imread("./matchData/cam01_01.bmp")
# frame2 = cv2.imread("./matchData/cam02_01.bmp")
# sourceImage = np.concatenate([frame1, frame2], axis=1)
# print("合并原始图像后尺寸：", sourceImage.shape)
# for i in range(0, sourceImage.shape[0], 32):
#     cv2.line(sourceImage, (0, i), (sourceImage.shape[1], i), (0, 255, 0), 1, 8)
#
# cv2.namedWindow('sourceImage', flags=cv2.WINDOW_NORMAL)
# cv2.imshow('sourceImage', sourceImage)
#
# img1_rectified = cv2.remap(frame1, left_map1, left_map2, cv2.INTER_LINEAR)
# img2_rectified = cv2.remap(frame2, right_map1, right_map2, cv2.INTER_LINEAR)
# cv2.namedWindow('left', flags=cv2.WINDOW_NORMAL)
# cv2.imshow('left', img1_rectified)
# cv2.namedWindow('right', flags=cv2.WINDOW_NORMAL)
# cv2.imshow('right', img2_rectified)
#
# image = np.concatenate([img1_rectified, img2_rectified], axis=1)
# iamge_size = image.shape
# print("合并校正图像后尺寸：", image.shape)
# for i in range(0, iamge_size[0], 32):
#     cv2.line(image, (0, i), (iamge_size[1], i), (0, 255, 0), 1, 8)
# cv2.namedWindow("rectified", cv2.WINDOW_GUI_NORMAL)
# cv2.imshow('rectified', image)
#
# imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
# imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
# num = cv2.getTrackbarPos('num', 'depth')
# blockSize = cv2.getTrackbarPos('blockSize', 'depth')
# if blockSize % 2 == 0:
#     blockSize += 1
# if blockSize < 5:
#     blockSize = 5
#
# stereo = cv2.StereoBM_create(numDisparities=0, blockSize=5)
# disparity = stereo.compute(imgL, imgR)
# disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32)/16., Q)     #此三维坐标点的基准坐标系为左侧相机坐标系
# cv2.imshow('depth', disp)
# cv2.waitKey(0)
# cv2.destroyAllWindows()