import cv2
import os
import numpy as np


def main():
    #  imgName = "out3/0.bmp"
    #  print("current File:" + imgName)
    #  imSrc = cv2.imread(imgName)
    #
    #  blur_usm = cv2.GaussianBlur(imSrc,(0,0),5)
    #  dst = cv2.addWeighted(imSrc,1.5,blur_usm,-0.5,0)
    #
    #  cv2.imshow("usm",dst)
    #
    #  blur_laplace = cv2.Laplacian(imSrc,-1)
    #  dst = cv2.addWeighted(imSrc, 1, blur_laplace, -0.5, 0)
    #  cv2.imshow("laplace",dst)
    #  cv2.waitKey(0)
    each_dir = "out"
    for file in os.listdir(each_dir):
        imgName = each_dir + "/" + file
        print("current File:" + imgName)
        if os.path.isdir(imgName):
            continue
        saveImgNameUSM = each_dir + "/" + "USM25/" + file
        saveImgNameLaplacian = each_dir + "/" + "Laplace/" + file
        imSrc = cv2.imread(imgName)

        blur_usm = cv2.GaussianBlur(imSrc,(0,0),25)
        dst = cv2.addWeighted(imSrc,1.5,blur_usm,-0.5,0)
        cv2.imwrite(saveImgNameUSM, dst)
        print("USM:" + saveImgNameUSM)

        blur_laplace = cv2.Laplacian(imSrc,-1)
        dst = cv2.addWeighted(imSrc, 1, blur_laplace, -0.5, 0)
        cv2.imwrite(saveImgNameLaplacian,dst)
        print("Laplace:" + saveImgNameLaplacian)




    #  imSrc = cv2.imread("../bmp/0.bmp")
    #  blur_usm = cv2.GaussianBlur(imSrc, (0,0), 5)
    #  dst = cv2.addWeighted(imSrc, 1.5, blur_usm, -0.5, 0)
    #  cv2.imshow("USM",dst)
    #  cv2.waitKey(0)

if __name__ == "__main__":
    main()
