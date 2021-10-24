# -*- encoding: utf-8 -*-
""""#------------------------------------------------#
#---------------保存视频中图片帧------------------#
#------------------------------------------------#"""
import cv2
import os
import glob
import random
import shutil

def video2Image(videoPath, imageSavePath):
    if not os.path.exists(imageSavePath):
        os.mkdir(imageSavePath)
    cap = cv2.VideoCapture(videoPath)
    middleName = videoPath.split('/')[-1].split('.')[0]
    middleName = middleName.split('_')[0] + '_' + middleName.split('_')[1]
    imgName = imageSavePath + middleName + '_'
    count = 62538
    while (1):
        success, frame = cap.read()
        if success:
            imageNewName = imgName + str(count) + '.jpg'
            cv2.imwrite(imageNewName, frame)
            print(imageNewName)
            count = count + 1
        else:
            break
    print(count)
    cap.release()


"""#------------------------------------------------#
#----------------读入数据名字并排序--------------------#
#-------------------------------------------------#"""
def prepare_data(datasetPath):
    data = glob.glob(os.path.join(datasetPath, "*.png"))
    data.sort(key=lambda x: int(x[len(datasetPath) + 1: -4]))  # 只按照一种条件排序
    # data.sort(key=lambda x: (int(x[len(datasetPath) : len(datasetPath) + 8]), int(x[len(datasetPath) + 16 : -4])))  # 按照多种条件排序
    return data


"""#------------------------------------------------#
#-------------图像重命名，减去较小数字-----------------#
#-------------------------------------------------#"""
def imageRename(imagePath):
    data = glob.glob(os.path.join(imagePath, "*.jpg"))
    data.sort(key=lambda x: int(x[len(imagePath) + 16:-4]))
    count = 0
    for ind in range(len(data)):
        oldName = data[ind]
        print(oldName)
        index_prefix = oldName.split('/')[-1].split('_')
        newName = imagePath + index_prefix[0] + '_' + index_prefix[1] + '_' + str(count) + '.jpg'
        print(newName)
        try:
            os.rename(oldName, newName)
        except OSError:
            print('重命名失败！！！')
        count = count + 1
    print('重命名完成，图像总数： %d' % count)


"""#------------------------------------------------#
#-------------图像重命名，减去较小数字并加上时间-----------------#
#-------------------------------------------------#"""
def imageRename2(imagePath):
    data = glob.glob(os.path.join(imagePath, "*.png"))
    # data.sort(key=lambda x: int(x[len(imagePath) : -4]))
    count = 0
    for ind in range(len(data)):
        oldName = data[ind]
        print(oldName)
        imgName = oldName.split('/')
        # newName = imagePath + '20200316_141608_' + str(count) + '.jpg'
        newName = imagePath + imgName[-1].split('.')[0] + '.jpg'
        #newName = imagePath + '{:04d}.png'.format(count)
        print(newName)
        try:
            os.rename(oldName, newName)
        except OSError:
            print('重命名失败！！！')
        count = count + 1
    print('重命名完成，图像总数： %d' % count)


"""#------------------------------------------------#
#------------------间隔固定行删除照片--------------------#
#-------------------------------------------------#"""
def deletImgInter(ImgPath, Inter):
    data = prepare_data(ImgPath)
    print('文件夹中图片总数：%d' % (len(data)))
    count = 0
    for ind in range(len(data)):
        if ind % int(Inter) == 0:
            count += 1
            print(data[ind])
        else:
            os.remove(data[ind])
    print('number Image after delected: %d' % count)


"""#------------------------------------------------#
#------------------彩色图像转伪彩色--------------------#
#-------------------------------------------------#"""
def color2Pesudo(imgSrcPath, imgGraySavePath, imgPesudoSavePath):
    data = prepare_data(imgSrcPath)
    count = 0
    for ind in range(len(data)):
        print(data[ind])
        im_gray = cv2.imread(data[ind], cv2.IMREAD_GRAYSCALE)
        # imgScaled = im_gray / 255 * 360  #就不用循环单独赋值了，速度快且效果好,网上所找高清图像不用scale
        im_Presudo = cv2.applyColorMap(im_gray, cv2.COLORMAP_AUTUMN)
        imageName1 = data[ind].split('/')[-1]
        imageNameGray = imgGraySavePath + imageName1
        # imageNamePesudo = imgPesudoSavePath + imageName1.split('_')[0] + '_' + imageName1.split('_')[1] + '_' + str(count) +'AUTUMN.jpg'
        imageNamePesudo = imgPesudoSavePath + imageName1.split('.')[0] + '_AUTUMN.jpg'
        # im_graySaveName = os.path.join(imgGraySavePath, imageNameGray)
        # im_pesudoSaveName = os.path.join(imgPesudoSavePath, imageNamePesudo)
        cv2.imwrite(imageNameGray, im_gray)
        cv2.imwrite(imageNamePesudo, im_Presudo)
        count += 1
    print('Total Nmber of Images:%d' % count)


"""#------------------------------------------------#
#---------------裁剪图片指定区域------------------#
#-------------------------------------------------#"""
def cutImage(imgSrcPath, imgSavePath, x, y, h, w):
    data = prepare_data(imgSrcPath)
    for ind in range(len(data)):
        img = cv2.imread(data[ind])
        if len(img.shape) == 2:
            cuttedImg = img[x: x + h, y: y + w]
        if len(img.shape) == 3:
            cuttedImg = img[x: x + h, y: y + w, :]
        imgName = data[ind].split('\\')[-1]
        imgSaveName = os.path.join(imgSavePath, imgName)
        print(imgSaveName)
        cv2.imwrite(imgSaveName, cuttedImg)
    print('Total Image Number: %d' % (len(data)))
    print('Cutted Done!!!!')


"""#------------------------------------------------#
#------------------灰度图像转伪彩色--------------------#
#-------------------------------------------------#"""
def color2Pesudo2(imgGraySavePath, imgPesudoSavePath):
    data = prepare_data(imgGraySavePath)
    count = 0
    # COLORMAP = [cv2.COLORMAP_BONE, cv2.COLORMAP_JET, cv2.COLORMAP_WINTER, cv2.COLORMAP_RAINBOW, cv2.COLORMAP_OCEAN,
    #             cv2.COLORMAP_SUMMER, cv2.COLORMAP_SPRING, cv2.COLORMAP_COOL, cv2.COLORMAP_HSV, cv2.COLORMAP_PINK, cv2.COLORMAP_HOT]
    COLORMAP = [cv2.COLORMAP_AUTUMN]
    for ind in range(len(data)):
        print(data[ind])
        im_gray = cv2.imread(data[ind], cv2.IMREAD_GRAYSCALE)
        for i in COLORMAP:
            print (i)
            #im_Presudo = cv2.applyColorMap(im_gray, cv2.COLORMAP_PINK)
            im_Presudo = cv2.applyColorMap(im_gray, i)
            imageName = data[ind].split('/')[-1]
            # imageNamePesudo = imgPesudoSavePath + imageName.split('_')[0] + '_' + imageName.split('_')[1] + '_' + str(count) +'HOT.jpg'
            #imageNamePesudo = imgPesudoSavePath + imageName.split('.')[0] + '_'+ str(i).split('_')[-1] + '.jpg'
            imageNamePesudo = imgPesudoSavePath + imageName.split('.')[0] + '_'+ str(0) + '.jpg'
            print(imageNamePesudo)
            cv2.imwrite(imageNamePesudo, im_Presudo)
        count += 1
    print('Total Nmber of Images:%d'%count)


"""#------------------------------------------------#
#------从一个文件夹中随机抽取图片保存到另一个文件夹中--------#
#-------------------------------------------------#"""
def moveFile(fileDirHR, tarDirHR, fileDirLR, tarDirLR, ):
    pathDir = os.listdir(fileDirHR)  # 取图片的原始路径
    filenumber = len(pathDir)
    rate = 0.000668428  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print(sample)
    count = 0
    for name in sample:
        shutil.move(fileDirHR + name, tarDirHR + name)
        shutil.move(fileDirLR + name, tarDirLR + name)
        print('ind:%d--->>>%s' % (count, name))
        count += 1
    print('Total number of removed: %d' % count)
    return


"""#------------------------------------------------#
#------取LR文件夹中与HR文件夹中同名图片到验证文件夹下--------#
#-------------------------------------------------#"""
def moveFile2(HRFliePath, LRFilePath, LRTarFilePath):
    data = glob.glob(os.path.join(HRFliePath, '*.jpg'))
    count = 0
    for ind in range(len(data)):
        dataName = data[ind].split('/')[-1]
        dataNameLR = LRFilePath + dataName
        shutil.move(dataNameLR, LRTarFilePath)
        count += 1
    print('Total number of removed: %d' % count)

"""#------------------------------------------------#
#---------移动一个文件夹中所有图片到另一个文件夹中----------#
#-------------------------------------------------#"""
def moveFile3(LRFilePath, LRTarFilePath):
    data = glob.glob(os.path.join(LRFilePath, '*.jpg'))
    count = 0
    for ind in range(len(data)):
        shutil.move(data[ind], LRTarFilePath)
        count += 1
    print('Total number of removed: %d' % count)


"""#------------------------------------------------#
#------－－－－－删除两个文件夹中不同名图片－－－－－--------#
#-------------------------------------------------#"""
def deletImg(pathHR, pathLR):
    listHR = []
    listLR = []
    for root, dir, files in os.walk(pathHR):
        for file in files:
            listHR.append(file)

    for root, dir, files in os.walk(pathLR):
        for file in files:
            listLR.append(file)

    diffInHRNotInLR = set(listLR).difference(set(listHR))  # 在HR中但不在LR中
    diffInLRNotInHR = set(listHR).difference(set(listLR))  # 在LR中但不在HR中
    countLR = 0
    for imgLR in diffInHRNotInLR:
        imgNameLR = pathLR + imgLR
        print(imgNameLR)
        os.remove(imgNameLR)
        countLR += 1
    print('在HR中不在LR中图片总数：%d' % countLR)

    countHR = 0
    for imgHR in diffInLRNotInHR:
        imgNameHR = pathHR + imgHR
        print(imgNameHR)
        os.remove(imgNameHR)
        countHR += 1
    print('在LR中不在HR中图片总数：%d' % countLR)


"""#------------------------------------------------#
#----------显示两个文件夹中图片总数--------#
#-------------------------------------------------#"""
def totalNumber(imgPathHR, imgPathLR):
    dataHR = glob.glob(os.path.join(imgPathHR, '*.jpg'))
    dataLR = glob.glob(os.path.join(imgPathLR, '*.jpg'))
    dataHR.extend(glob.glob(os.path.join(imgPathHR, '*.png')))
    dataLR.extend(glob.glob(os.path.join(imgPathLR, '*.png')))
    print('HR文件夹中图片总数：', len(dataHR))
    print('LR文件夹中图片总数：', len(dataLR))

    ends = ['.jpg', '.png']
    filepaths = [f for f in os.listdir(dataHR) if os.path.splitext(f)[-1].lower() in ends]
    num_files = len(filepaths)
    print('HR文件夹中图片总数：', num_files)


if __name__ == '__main__':
    # 彩色图片转换为伪彩色图片
    # imageSrcPath = '/media/wootion/1E1AB215A6819C01/yx/ESRGAN-TrainDataSet/DIV2K/Mi10PRO468k/'
    # imgGraySavePath = '/media/wootion/1E1AB215A6819C01/yx/ESRGAN-TrainDataSet/DIV2K/DIV2K_train_HR_Mod/HR/x4'
    # imgPesudoSavePath = '/media/wootion/1E1AB215A6819C01/yx/ESRGAN-TrainDataSet/DIV2K/DIV2K_train_HR_Mod/LR/x4'
    # color2Pesudo2(imgGraySavePath, imgPesudoSavePath)
    # totalNumber(imgGraySavePath, imgPesudoSavePath)
    # moveFile3(imgGraySavePath, imgPesudoSavePath)

    # #视频转换为图片
    # imageSavePath = '/media/wt/DATA1/YanXu/XinJiangDianShiChangData/PictureGrabed_0417/VI_Image/20200417_133537/'
    # videoPath = '/media/wt/DATA1/YanXu/XinJiangDianShiChangData/PictureGrabed_0417/VI_Video/20200417_133537_2.mp4'
    # video2Image(videoPath, imageSavePath)

    # 图片重命名
    # imagePath = '/media/wootion/1E1AB215A6819C01/yx/ESRGAN-TrainDataSet/DIV2K/Mi10PRO468k/'
    # imageRename2(imagePath)

    # 间隔固定行删除照片
    # imgPath = '/home/idriver/idriver_data/ImageBirdEye/CS55yuanqu_211021_deleted/'
    # inter = 5
    # deletImgInter(imgPath, inter)

    imgPath = '/home/idriver/idriver_data/ImageBirdEye/CS55yuanqu_211021_deleted/'
    inter = 5
    deletImgInter(imgPath, inter)

    # 裁剪图片指定区域
    imgSrcPath = 'D:/YanXu/ESRGAN/Pseudo_IR_TrainData/Pseudo_Image_JET/'
    imgSavePath = 'D:/YanXu/ESRGAN/Pseudo_IR_TrainData/Pseudo_Image_JET_Cuted/'
    x, y, h, w = 80, 320, 480, 640 #指定需要裁减的区域
    cutImage(imgSrcPath, imgSavePath, x, y, h, w)

    # # 图像重命名，减去较小数字并加上时间
    # imageSrcPath = 'D:/YanXu/ESRGAN/Pseudo_IR_TrainData/VI_Image/'
    # #imageRename2(imageSrcPath)
    # Inter = 16
    # deletImgInter(imageSrcPath, Inter)

    # 从一个文件夹中随机取图片保存到另一个文件夹中
    # fileDirHR = '/media/wootion/1E1AB215A6819C01/yx/ESRGAN-TrainDataSet/DIV2K/DIV2K_train_HR_Mod/HR/x4/'
    # tarDirHR = '/media/wootion/1E1AB215A6819C01/yx/ESRGAN-TrainDataSet/DIV2K/DIV2K_train_HR_Mod/VAL/HR/'
    # fileDirLR = '/media/wootion/1E1AB215A6819C01/yx/ESRGAN-TrainDataSet/DIV2K/DIV2K_train_HR_Mod/LR/x4/'
    # tarDirLR = '/media/wootion/1E1AB215A6819C01/yx/ESRGAN-TrainDataSet/DIV2K/DIV2K_train_HR_Mod/VAL/LR/'
    # moveFile(fileDirHR, tarDirHR, fileDirLR, tarDirLR)
    # moveFile(tarDirHR, fileDirHR, tarDirLR, fileDirLR)
    # totalNumber(fileDirHR, fileDirLR)
    # totalNumber(tarDirHR, tarDirLR)
    # deletImg(tarDirHR, tarDirLR)

    # 取低LR文件夹中与HR文件夹中同名图片到验证文件夹下
    # HRFliePath = '/media/wootion/1E1AB215A6819C01/yx/Pseudo_IR_TrainData/VI-Pesudo_Image_Cutted_mod/VAL/HR/'
    # LRFilePath = '/media/wootion/1E1AB215A6819C01/yx/Pseudo_IR_TrainData/VI-Pesudo_Image_Cutted_mod/LR/x4/'
    # LRTarFilePath = '/media/wootion/1E1AB215A6819C01/yx/Pseudo_IR_TrainData/VI-Pesudo_Image_Cutted_mod/VAL/LR/'
    # moveFile2(HRFliePath, LRFilePath, LRTarFilePath)

    # 不调用函数直接将图像resized
    # img = cv2.imread('/home/wootion/yx/ESRGAN/ESRGAN-master/LR_XianC/5593.jpg')
    # imgResized = cv2.resize(img, (640, 480))
    # imgSaveNname = '/home/wootion/yx/ESRGAN/ESRGAN-master/LR_XianC/5593_resized.jpg'
    # cv2.imwrite(imgSaveNname, imgResized)
