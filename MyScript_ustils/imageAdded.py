import cv2
import math
import os
import numpy as np

def imageProcess(IRImage, VIImage, diff_x, diff_y, alpha):
    h_in, w_in, _ = IRImage.shape
    h_vi, w_vi, _ = VIImage.shape
    proImage = VIImage.copy()
    start_x = diff_x
    start_y = diff_y
    for x in range(start_x, start_x + h_in):
        if x< 0 | x > h_vi:
            continue
        else:
            for y in range(start_y, start_y + w_in):
                if y < 0 | y > w_in:
                    continue
                else:
                    proImage[x, y, :] = (1 - alpha / 255) * proImage[x, y, :] + (alpha / 255) * IRImage[x - start_x, y - start_y, :]
    return proImage

def imageProcessPadded(IRImage, VIImage, diff_x, diff_y, alpha):
    h_in, w_in, _ = IRImage.shape
    h_vi, w_vi, _ = VIImage.shape
    proImage = np.zeros([VIImage.shape])
    start_x = diff_x
    start_y = diff_y
    for x in range(start_x, start_x + h_in):
        for y in range(start_y, start_y + w_in):
                proImage[x, y, :] = IRImage[x - start_x, y - start_y, :]
    return proImage

def nothing(x):
    pass

def ReadFile(filepath):
    binfile = open(filepath, 'rb') #打开二进制文件
    size = os.path.getsize(filepath) #获得文件大小
    dataList = []
    for i in range(size):
        data = binfile.read(258) #每次输出一个字节
        #print('字节型： ',data)
        data1 = data.decode('utf-8')
        # print('函数内的字符串型： \n', data1)
        #print('函数内的字符串长度： \n', len(data1))
        if len(data1) == 0:
            continue
        else:
            dataList.append(data1)
    return dataList
    binfile.close()

def getCoordinata(filepath):
    data = ReadFile(filepath)
    # print('字符串型：', data)
    # print('字符串长度：', len(data))
    list = data[0]
    print(list)
    # print(len(list))
    index1 = list.find('[')
    index1_dot1 = list.find('.', index1 + 1)
    index1_dot2 = list.find('.', index1_dot1 + 1)
    # print(index1)
    # if list[index1 + 1].isspace():
    #     print('True')
    VI_x1 = int(list[index1 + 2: index1_dot1])
    VI_y1 = int(list[index1_dot1 + 3: index1_dot2])
    print('可见光第一个点坐标（%d, %d）： ' % (VI_x1, VI_y1))

    index2 = list.find('[', index1_dot2 + 1)
    index2_dot1 = list.find('.', index2 + 1)
    index2_dot2 = list.find('.', index2_dot1 + 1)
    VI_x2 = int(list[index2 + 2: index2_dot1])
    VI_y2 = int(list[index2_dot1 + 3: index2_dot2])
    print('可见光第二个点坐标（%d, %d）： ' % (VI_x2, VI_y2))

    index3 = list.find('[', index2_dot2 + 1)
    index3_dot1 = list.find('.', index3 + 1)
    index3_dot2 = list.find('.', index3_dot1 + 1)
    IR_x1 = int(list[index3 + 2: index3_dot1])
    IR_y1 = int(list[index3_dot1 + 3: index3_dot2])
    print('红外光第一个点坐标（%d, %d）： ' % (IR_x1, IR_y1))

    index4 = list.find('[', index3_dot2 + 1)
    index4_dot1 = list.find('.', index4 + 1)
    index4_dot2 = list.find('.', index4_dot1 + 1)
    IR_x2 = int(list[index4 + 2: index4_dot1])
    IR_y2 = int(list[index4_dot1 + 3: index4_dot2])
    print('红外光第二个点坐标（%d, %d）： ' % (IR_x2, IR_y2))

    return VI_x1, VI_y1, VI_x2, VI_y2, IR_x1, IR_y1, IR_x2, IR_y2

if __name__ == '__main__':
    # IRImg = '/home/wt/XinJiangDianShiLu/MyTestData/CalibrationData/20200401/IR_noPadded/0.jpg'
    # VIImg = '/home/wt/XinJiangDianShiLu/MyTestData/CalibrationData/20200401/VI/0.jpg'
    # binFilePath = '/home/wt/Desktop/Preset/20200401/testPoint/VI1/meterKeys.bin'

    IRImg = '/home/wt/Desktop/Preset/20200417_133537/templates/testPointMap0/zoom0.jpg'
    VIImg = '/home/wt/Desktop/Preset/20200417_133537/testPoint/VI0/zoom0.jpg'
    binFilePath = '/home/wt/Desktop/Preset/20200417_133537/testPoint/VI1/meterKeys.bin'

    saveImagePath = '/home/wt/Desktop/Preset/20200417_133537/saveImage/'

    VI_x1, VI_y1, VI_x2, VI_y2, IR_x1, IR_y1, IR_x2, IR_y2 = getCoordinata(binFilePath)
    #if (VI_y2 - VI_y1) < (IR_y2 - IR_y1):
    fxScale1 = (VI_y2 - VI_y1) / (IR_y2 - IR_y1)
    fyScale1 = (VI_x2 - VI_x1) / (IR_x2 - IR_x1)
    print('x方向缩放情况：(%d/%d)'% ((VI_y2 - VI_y1), (IR_y2 - IR_y1)))
    print('y方向缩放情况：(%d/%d)'% ((VI_x2 - VI_x1), (IR_x2 - IR_x1)))
    diffX1 = VI_y1 - round(IR_y1 * fyScale1)
    diffY1 = VI_x1 - round(IR_x1 * fxScale1)
    print("软件手点缩放系数：fxScale1 = %f, fyScale1 = %f" % (fxScale1, fyScale1))
    print("软件手点图像偏移：diffX1 = %d, diffY1 = %d" % (diffX1, diffY1))

    # fxScale = 188 / 272  #第二台相机缩小，第一台增大
    # fyScale = 268 / 374
    # diffX = 477 - round(161*fxScale)
    # diffY = 582 - round(172*fyScale)
    # print ("自己手点缩放系数：fxScale = %f, fyScale = %f" % (fxScale, fyScale))
    # print ("自己手点图像偏移：diffX = %d, diffY = %d" %(diffX, diffY))

    IRImage = cv2.imread(IRImg)
    VIImage = cv2.imread(VIImg)

    rows, cols = VIImage.shape[:2]
    print('可见光图像尺寸：',rows, cols)
    # 第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
    #Map = cv2.getRotationMatrix2D((rows / 2, cols / 2), -2, 1)
    Map = cv2.getRotationMatrix2D((521 , 481), -2, 1)
    # 第三个参数：变换后的图像大小
    RotationImage = cv2.warpAffine(VIImage, Map, (cols, rows))
    print('旋转后图像尺寸：', RotationImage.shape)
    cv2.namedWindow('RotationImage', flags = cv2.WINDOW_NORMAL)
    cv2.imshow('RotationImage', RotationImage)

    IRImageResized = cv2.resize(IRImage, None, dst = None, fx = fxScale1, fy = fyScale1)
    print('红外图像原始尺寸大小：', IRImage.shape)
    print('红外图像resize后尺寸大小：', IRImageResized.shape)

    cv2.namedWindow('processedImage', flags = cv2.WINDOW_NORMAL)
    alpha = 0
    cv2.createTrackbar('alpha', 'processedImage', 0, 255, nothing)

    cv2.namedWindow('processedImageRo', flags = cv2.WINDOW_NORMAL)
    cv2.createTrackbar('alpha', 'processedImageRo', 0, 255, nothing)
    # count  = 0
    while (1):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        alpha = cv2.getTrackbarPos('alpha', 'processedImage')
        alpha1 = cv2.getTrackbarPos('alpha', 'processedImageRo')
        proImage = imageProcess(IRImageResized, VIImage, diffX1, diffY1, alpha)
        proImageRo = imageProcess(IRImageResized, RotationImage, diffX1, diffY1, alpha1)

        # saveImageName = saveImagePath + str(count) + '.jpg'
        # cv2.imwrite(saveImageName, proImage)

        #count = count +1
        cv2.imshow('processedImage', proImage)
        cv2.imshow('processedImageRo', proImageRo)
        # cv2.namedWindow('IRImageResized', flags = cv2.WINDOW_NORMAL)
        # cv2.imshow('IRImageResized', IRImageResized)
        # cv2.namedWindow('infraredImage', flags = cv2.WINDOW_NORMAL)
        # cv2.imshow('infraredImage', IRImage)
        # cv2.namedWindow('visiableImage', flags = cv2.WINDOW_NORMAL)
        # cv2.imshow('visiableImage', VIImage)
    cv2.destroyAllWindows()


""""#-----------------------------------------------------------------#
#----------------------------测试鼠标滑动条参数--------------------------#
#------------------------------------------------------------------#"""
# import cv2
# d = 0
# color = 0
# space = 0
#
# def change_d(x):
#     d = x
#     blur = cv2.bilateralFilter(img, d, color, space)
#     cv2.imshow("myImg", blur)
#
# def change_color(x):
#     color = x
#     blur = cv2.bilateralFilter(img, d, color, space)
#     cv2.imshow("myImg", blur)
#
#
# def change_space(x):
#     space = x
#     blur = cv2.bilateralFilter(img, d, color, space)
#     cv2.imshow("myImg", blur)
#
#
# img = cv2.imread('/home/wt/tensorflow-yolov3/docs/images/car2.jpg')
# cv2.namedWindow('myImg', cv2.WINDOW_NORMAL)
# cv2.createTrackbar('d', 'myImg', 1, 500, change_d)
# cv2.createTrackbar('color', 'myImg', 1, 500, change_color)
# cv2.createTrackbar('space', 'myImg', 1, 500, change_space)
#
# while (1):
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break
#     d = cv2.getTrackbarPos('d', 'myImg')
#     color = cv2.getTrackbarPos('color', 'myImg')
#     space = cv2.getTrackbarPos('space', 'myImg')
# cv2.destroyAllWindows()


""""#-----------------------------------------------------------------#
#---------------样例中调试得较好的参数------------------#
#------------------------------------------------------------------#"""
#IRImage = /home/wt/XinJiangDianShiLu/MyTestData/CalibrationData/20200316/Processed/IR/2.jpg'  #0316标定图像
#IRImage = '/home/wt/XinJiangDianShiLu/MyTestData/frusedTestImage/0318/IR/2.jpg'  #0318
#VIImage = '/home/wt/XinJiangDianShiLu/MyTestData/CalibrationData/20200316/Processed/VI/2.jpg'
#VImage = '/home/wt/XinJiangDianShiLu/MyTestData/frusedTestImage/0318/VI/2.jpg'
########-----------0316标定图像------------#######
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
# fxScale = 156 / 136
# fyScale = 390 / 339
#
# diffX0 = 748 - round(396*fxScale)
# diffY0 = 803 - round(139*fyScale)
# print ("diffX0 = %d, diffY0 = %d" %(diffX0, diffY0))
#
# diffX = 590 - 300
# diffY = 1191 - 549
# print ("diffX = %d, diffY = %d" %(diffX, diffY))
#
# diffX1 = 645 - 308
# diffY1 = 1027 - 333

########-----------0401标定图像参数1------------####
# fxScale = 129 / 191  #第二台相机是缩小，第一台是增大
# fyScale = 243 / 343
#
# diffX = 497 - round(189*fxScale)
# diffY = 572 - round(160*fyScale)
# print ("diffX0 = %d, diffY0 = %d" %(diffX, diffY))


########------------写的一个融合函数，暂不需要------------####
# def imageAdded(IRImg, VIImg, fxScale, fyScale, diffX, diffY):
#     IRImage = cv2.imread(IRImg)
#     VIImage = cv2.imread(VIImg)
#     IRImageResized = cv2.resize(IRImage, None, dst=None, fx=fxScale, fy=fyScale)
#     print('红外图像原始尺寸大小：', IRImage.shape)
#     print('红外图像resize后尺寸大小：',IRImageResized.shape)
#
#     proImage = imageProcess(IRImageResized, VIImage, diffX, diffY)
#
#     cv2.namedWindow('IRImageResized',flags = cv2.WINDOW_NORMAL)
#     cv2.imshow('IRImageResized',IRImageResized)
#
#     cv2.namedWindow('infraredImage',flags = cv2.WINDOW_NORMAL)
#     cv2.imshow('infraredImage',IRImage)
#     cv2.namedWindow('visiableImage',flags = cv2.WINDOW_NORMAL)
#     cv2.imshow('visiableImage',VIImage)
#     return proImage