import cv2
import numpy as np
from math import tan

kernel_gx = np.array([
    [1, 1],
    [-1, -1],
]) * 2

kernel_gy = np.array([
    [1, -1],
    [1, -1]
]) * 2


def calc_grad(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    gx = cv2.filter2D(blur, cv2.CV_64F, kernel_gx)
    gy = cv2.filter2D(blur, cv2.CV_64F, kernel_gy)
    grad = np.sqrt(np.square(gx) + np.square(gy))
    tgs = gy / gx
    tgs[np.isnan(tgs)] = 0
    return (grad, tgs)


def nms(grad, tgs):
    after_nms = np.zeros(grad.shape)
    height, width = grad.shape
    points_nearby = []
    for i in range(height):
        for j in range(width):
            t = tgs[i][j]
            if t >= tan(-np.pi / 8) and t <= tan(np.pi / 8):
                points_nearby = [
                    [i + 1, j],
                    [i - 1, j]
                ]
            elif t > tan(np.pi / 8) and t <= tan(np.pi * 3 / 8):
                points_nearby = [
                    [i + 1, j + 1],
                    [i - 1, j - 1]
                ]
            elif t > tan(-np.pi * 3 / 8) and t < tan(-np.pi / 8):
                points_nearby = [
                    [i + 1, j - 1],
                    [i - 1, j + 1]
                ]
            elif t > tan(np.pi * 3 / 8) or t < tan(-np.pi * 3 / 8):
                points_nearby = [
                    [i, j + 1],
                    [i, j - 1]
                ]
            else:
                print(t)
                raise Exception('bad tangent value')

            p1, p2 = points_nearby
            grad_nearby_1 = 0
            grad_nearby_2 = 0
            if validate_index(p1[0], p1[1], height, width):
                grad_nearby_1 = grad[p1[0]][p1[1]]
            if validate_index(p2[0], p2[1], height, width):
                grad_nearby_2 = grad[p2[0]][p2[1]]
            if grad[i][j] > grad_nearby_1 and grad[i][j] > grad_nearby_2:
                after_nms[i][j] = grad[i][j]
    return after_nms


def validate_index(i, j, height, width):
    return i >= 0 and i < height and j > 0 and j < width


# def double_thresholding(image, t1, t2):
#     img_low = (image > t1) * image
#     img_high = (image > t2) * image
#     return img_low, img_high


# def connect_img_high(img_high, img_low):
#     result = np.zeros(img_high.shape)
#     height, width = img_high.shape
#     for i in range(height):
#         for j in range(width):
#             if img_high[i][j] > 0:
#                 result[i][j] = img_high[i][j]
#             elif img_high[i][j] == 0 and img_low[i][j] > 0 and look_around(img_low, i, j):
#                 result[i][j] = img_low[i][j]
#     return result

def double_thresholding(image, t_high, t_low):
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            if image[i][j] >= t_high:
                image[i][j] = 255
                find_component(image, i, j, t_low)

    image[image < 255] = 0
    return image


def find_component(image, x, y, t_low):
    height, width = image.shape
    for i in range(x - 1, x + 2):
        for j in range(y - 1, y + 2):
            if validate_index(i, j, height, width) and i != x and j != y:
                if image[i][j] != 255:
                    if image[i][j] >= t_low:
                        image[i][j] = 255
                        find_component(image, i, j, t_low)
                    else:
                        image[i][j] = 0


def main():
    img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    grad, tgs = calc_grad(img)
    cv2.imshow('grad', grad)
    # cv2.imwrite('grad.jpg', grad)
    after_nms = nms(grad, tgs)
    cv2.imshow('nms', after_nms)
    # cv2.imwrite('nms.jpg', after_nms)
    result = double_thresholding(after_nms, 30, 60)
    cv2.imshow('res', result)
    # cv2.imwrite('res.jpg', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
