import cv2
import math
import os
import numpy as np
import glob

def imageProcessPadded(IRImage, size, diff_x, diff_y):
    h_in, w_in, _ = IRImage.shape
    proImage = np.zeros(size)
    #print(proImage.shape)
    start_x = diff_x
    start_y = diff_y
    for x in range(start_x, start_x + h_in):
        if x >= size[0]:
            break
        else:
            for y in range(start_y, start_y + w_in):
                if y >= size[1]:
                    break
                else:
                    proImage[x, y, :] = IRImage[x - start_x, y - start_y, :]
    return proImage

def imageProcessCut(VIImage, IRImageSize, diff_x, diff_y):
    h_in, w_in = IRImageSize[0], IRImageSize[1]
    h_vi, w_vi, _ = VIImage.shape
    cutImage = np.zeros(IRImageSize)
    start_x = diff_x + 5
    start_y = diff_y
    print('h_vi, w_vi:(%d, %d)'%(h_vi, w_vi))
    print('start_x + h_in, start_y + w_in:(%d, %d)' % ( start_x + h_in, start_y + w_in))
    for x in range(start_x, start_x + h_in):
        if x >= h_vi:
            break
        else:
            for y in range(start_y, start_y + w_in):
                if y >= w_vi:
                    break
                else:
                    cutImage[x - start_x, y - start_y, :] = VIImage[x, y, :]
    return cutImage

def prepare_data_IR(datasetPath):
    data = glob.glob(os.path.join(datasetPath, "*.jpg"))
    data.sort(key=lambda x: int(x[len(datasetPath) + 16 : -4]))
    return data

def prepare_data_VI(datasetPath):
    data = glob.glob(os.path.join(datasetPath, "*.jpg"))
    data.sort(key=lambda x: int(x[len(datasetPath) + 16 : -4]))
    return data

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
    list = data[0]
    print(list)
    index1 = list.find('[')
    index1_dot1 = list.find('.', index1 + 1)
    index1_dot2 = list.find('.', index1_dot1 + 1)
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

""""#------------------------------------------------#/home/wt/Desktop/Preset/20200414_175303_2/testPoint/VI0
#------------------图片名中插入秒------------------#
#------------------------------------------------#"""
def insertSecond(path, second):
    count = 0
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            nameSpilt = name.split('_')
            newImgName = nameSpilt[0] + '_' + nameSpilt[1] + str(second) +'_' + nameSpilt[2].split('.')[0] + '_IR.jpg'
            newName = os.path.join(root, newImgName)
            oldName = os.path.join(root,name)
            try:
                os.rename(oldName, newName)
            except OSError:
                print('加入秒钟出错！！！')
            count =count + 1
    print("替换完成,图片总数：%d"%(count))


""""#------------------------------------------------#/home/wt/Desktop/Preset/20200414_175303_2/testPoint/VI0
#------------------图片名中插入秒------------------#
#------------------------------------------------#"""
def insertIR(path):
    count = 0
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            nameSpilt = name.split('_')
            newImgName = nameSpilt[0] + '_' + nameSpilt[1] + '_' + nameSpilt[2].split('.')[0] + 'IR.jpg'
            newName = os.path.join(root, newImgName)
            oldName = os.path.join(root,name)
            try:
                os.rename(oldName, newName)
            except OSError:
                print('加入IR标志出错！！！')
            count =count + 1
    print("替换完成,图片总数：%d"%(count))

""""#------------------------------------------------#/home/wt/Desktop/Preset/20200414_175303_2/testPoint/VI0
#------------------以数字序号命名图片名------------------#
#------------------------------------------------#"""
def insertIR1(path):
    count = 0
    data = prepare_data_IR(path)
    for ind in range(len(data)):
        # nameSpilt = data[ind].split('/')
        newImgName = str(count) + '.jpg'
        newName = os.path.join(path, newImgName)
        try:
            os.rename(data[ind], newName)
        except OSError:
            print('重命名出错！！！')
        count =count + 1
    print("替换完成,图片总数：%d"%(count))

##-------------------将可见光图像裁剪至红外图像尺寸并对齐------------------##
# if __name__ == '__main__':
#     size = (720, 1280, 3)
#     IR_ImagePath = '/home/wootion/yx/VI-IR_DataSet-2/IR_Image/20200417_103100_0-7500/'
#     VI_ImagePath = '/home/wootion/yx/VI-IR_DataSet-2/VI_Image/20200417_103100_0-7500/'
#     imagesSavePathIR = '/home/wootion/yx/VI-IR_DataSet-2/IR_Image(1)/20200417_103100_0-7500(1)/'
#     imagesSavePathVI = '/home/wootion/yx/VI-IR_DataSet-2/VI_Image(1)/20200417_103100_0-7500(1)/'
#     if not os.path.exists(imagesSavePathIR):
#         os.makedirs(imagesSavePathIR)
#     if not os.path.exists(imagesSavePathVI):
#         os.makedirs(imagesSavePathVI)
#     dataIR = prepare_data_IR(IR_ImagePath)
#     dataVI = prepare_data_VI(VI_ImagePath)
#
#     binFilePath = '/home/wootion/yx/FuseImage/Preset/20200417_103100/testPoint/VI1/meterKeys.bin'
#     VI_x1, VI_y1, VI_x2, VI_y2, IR_x1, IR_y1, IR_x2, IR_y2 = getCoordinata(binFilePath)
#
#     fxScale1 = (VI_y2 - VI_y1) / (IR_y2 - IR_y1)
#     fyScale1 = (VI_x2 - VI_x1) / (IR_x2 - IR_x1)
#     print('x方向缩放情况：(%d/%d)' % ((VI_y2 - VI_y1), (IR_y2 - IR_y1)))
#     print('y方向缩放情况：(%d/%d)' % ((VI_x2 - VI_x1), (IR_x2 - IR_x1)))
#     diffX1 = VI_y1 - round(IR_y1 * fyScale1)
#     diffY1 = VI_x1 - round(IR_x1 * fxScale1)
#     print("软件手点缩放系数：fxScale1 = %f, fyScale1 = %f" % (fxScale1, fyScale1))
#     print("软件手点图像偏移：diffX1 = %d, diffY1 = %d" % (diffX1, diffY1))
#     print(len(dataIR))
#     for ind in range(len(dataIR)):
#         IRImage = cv2.imread(dataIR[ind])
#         VIImage = cv2.imread(dataVI[ind])
#         IRImageResized = cv2.resize(IRImage, None, dst=None, fx=fxScale1, fy=fyScale1)
#
#         IRImageSize = IRImageResized.shape
#         print('红外图像缩放后尺寸：', IRImageSize)
#         imageCuted = imageProcessCut(VIImage, IRImageSize, diffX1, diffY1)
#
#         #准备红外图像保存的名字
#         nameSpilt = dataIR[ind].split('/')[-1].split('_')
#         newImgName = nameSpilt[0] + '_' + nameSpilt[1] + '_' + nameSpilt[2].split('.')[0] +'IRPad.jpg'
#         newImgName = os.path.join(imagesSavePathIR,newImgName)
#         #准备可见光图像保存的名字
#         nameSpiltVI = dataVI[ind].split('/')[-1]
#         newImgNameVI = os.path.join(imagesSavePathVI, nameSpiltVI)
#
#         #cv2.imwrite(newImgName, IRImageResized)
#         cv2.imwrite(newImgNameVI, imageCuted)
#
#         print(newImgName)
#         print(newImgNameVI)
#     print('Done!!')

##-------------------将红外图像扩展到可将光图像尺寸并对齐------------------##
# if __name__ == '__main__':
#     size = (720, 1280, 3)
#     IR_ImagePath = '/home/wootion/yx/VI-IR_DataSet-2/IR_Image/20200417_103100_0-7500/'
#     imagesSavePathIR = '/home/wootion/yx/VI-IR_DataSet-2/IR_ImagePadded/20200417_103100_0-7500/'
#     if not os.path.exists(imagesSavePathIR):
#         os.makedirs(imagesSavePathIR)
#     dataIR = prepare_data_IR(IR_ImagePath)
#     #dataIR = prepare_data_IR(IR_ImagePath)
#
#     binFilePath = '/home/wootion/yx/FuseImage/Preset/20200417_103100/testPoint/VI1/meterKeys.bin'
#     VI_x1, VI_y1, VI_x2, VI_y2, IR_x1, IR_y1, IR_x2, IR_y2 = getCoordinata(binFilePath)
#
#     fxScale1 = (VI_y2 - VI_y1) / (IR_y2 - IR_y1)
#     fyScale1 = (VI_x2 - VI_x1) / (IR_x2 - IR_x1)
#     print('x方向缩放情况：(%d/%d)' % ((VI_y2 - VI_y1), (IR_y2 - IR_y1)))
#     print('y方向缩放情况：(%d/%d)' % ((VI_x2 - VI_x1), (IR_x2 - IR_x1)))
#     diffX1 = VI_y1 - round(IR_y1 * fyScale1)
#     diffY1 = VI_x1 - round(IR_x1 * fxScale1)
#     print("软件手点缩放系数：fxScale1 = %f, fyScale1 = %f" % (fxScale1, fyScale1))
#     print("软件手点图像偏移：diffX1 = %d, diffY1 = %d" % (diffX1, diffY1))
#
#     for ind in range(len(dataIR)):
#         IRImage = cv2.imread(dataIR[ind])
#         IRImageResized = cv2.resize(IRImage, None, dst=None, fx=fxScale1, fy=fyScale1)
#
#         IRImageSize = IRImageResized.shape
#         print('红外图像缩放后尺寸：', IRImageSize)
#         imageProcess = imageProcessPadded(IRImageResized, size, diffX1, diffY1)
#
#         #准备红外图像保存的名字
#         nameSpilt = dataIR[ind].split('/')[-1].split('_')
#         #newImgName = nameSpilt[0] + '_' + nameSpilt[1] + '_' + nameSpilt[2] + nameSpilt[3].split('.')[0] +'Pad.jpg'
#         newImgName = nameSpilt[0] + '_' + nameSpilt[1] + '_' + nameSpilt[2].split('.')[0] + 'IRPad.jpg'
#         newImgName = os.path.join(imagesSavePathIR,newImgName)
#
#         cv2.imwrite(newImgName, imageProcess)
#         print(newImgName)
#     print('Done!!')

if __name__ == '__main__':
    imagePath = '/media/wootion/1E1AB215A6819C01/yx/VI-IR_DataSet-2/VI_Image(1)/20200417_103100_0-7500(1)/'
    insertIR1(imagePath)