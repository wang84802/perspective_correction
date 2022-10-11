import numpy as np
import cv2
import math

class perspective_correction:
    def __init__(self, p1, p2, p3, p4):
        self.__get_image()
        self.__inverse_mapping(p1, p2, p3, p4)
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

    def __get_image(self):
        img = cv2.imread('1_800x600.jpg')
        cv2.imshow("Input", img)
        cv2.waitKey(0)
        self.img = img

    def __inverse_mapping(self, p1, p2, p3, p4):
        print(p1, p2, p3, p4)
        A = np.array([
            [p1[0], p3[1], p1[0]*p3[1], 1, 0, 0, 0, 0],
            [p2[0], p4[1], p2[0]*p4[1], 1, 0, 0, 0, 0],
            [p3[0], p3[1], p3[0]*p3[1], 1, 0, 0, 0, 0],
            [p4[0], p4[1], p4[0]*p4[1], 1, 0, 0, 0, 0],
            [0, 0, 0, 0, p1[0], p3[1], p1[0]*p3[1], 1],
            [0, 0, 0, 0, p2[0], p4[1], p2[0]*p4[1], 1],
            [0, 0, 0, 0, p3[0], p3[1], p3[0]*p3[1], 1],
            [0, 0, 0, 0, p4[0], p4[1], p4[0]*p4[1], 1]
        ])

        B = np.array([p1[0], p2[0], p3[0], p4[0], p1[1], p2[1], p3[1], p4[1]]).reshape(8, 1)
        A_inv = np.linalg.inv(A)
        ans = A_inv.dot(B)
        self.ans = ans

    def cal_pos(self, x, y):
        x += 46
        y += 186
        x1 = self.ans[0]*x + self.ans[1]*y + self.ans[2]*x*y + self.ans[3]
        y1 = self.ans[4]*x + self.ans[5]*y + self.ans[6]*x*y + self.ans[7]
        return x1, y1

    def nearest_neighbor(self, x, y):
        p1 = self.img[math.ceil(x), math.ceil(y)]
        p2 = self.img[math.floor(x), math.ceil(y)]
        p3 = self.img[math.ceil(x), math.floor(y)]
        p4 = self.img[math.floor(x), math.floor(y)]
        pixel_list = [p1, p2, p3, p4]
        return [sum(x)/len(pixel_list) for x in zip(*pixel_list)]

    def bilinear(self, x, y):
        p1 = self.img[math.floor(x), math.floor(y)]
        p2 = self.img[math.floor(x), math.ceil(y)]
        p3 = self.img[math.ceil(x), math.floor(y)]
        p4 = self.img[math.ceil(x), math.ceil(y)]
        a = float(x - math.floor(x))
        b = float(y - math.floor(y))
        p_result = [0, 0, 0]
        p_result[0] = (1-a)*(1-b)*p1[0] + a*(1-b)*p2[0] + (1-a)*b*p3[0] + a*b*p4[0]
        p_result[1] = (1-a)*(1-b)*p1[1] + a*(1-b)*p2[1] + (1-a)*b*p3[1] + a*b*p4[1]
        p_result[2] = (1-a)*(1-b)*p1[2] + a*(1-b)*p2[2] + (1-a)*b*p3[2] + a*b*p4[2]
        return p_result

    def transform_image(self):
        width = self.p3[0] - self.p1[0]
        height = self.p4[1] - self.p3[1]
        img1 = np.ones((width, height, 3), dtype=np.uint8)
        img1 *= 0

        for i in range(width):
            for j in range(height):
                x, y = self.cal_pos(i, j)
                img1[i, j] = self.bilinear(x, y)

        cv2.imshow('result', img1)
        cv2.waitKey(0)

mapping = perspective_correction([46, 254], [46, 559], [539, 186], [539, 656])
mapping.transform_image()
