import sys
import cv2
import numpy as np
from math import tan

kernel_gx = np.array([
    [1, 1],
    [-1, -1],
])

kernel_gy = np.array([
    [1, -1],
    [1, -1]
])


def validate_index(x, y, height, width):
    return x >= 0 and x < height and y >= 0 and y < width


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
            grad_p1 = 0
            grad_p2 = 0
            if validate_index(p1[0], p1[1], height, width):
                grad_p1 = grad[p1[0]][p1[1]]
            if validate_index(p2[0], p2[1], height, width):
                grad_p2 = grad[p2[0]][p2[1]]
            if grad[i][j] > grad_p1 and grad[i][j] > grad_p2:
                after_nms[i][j] = grad[i][j]
    return after_nms


def hysteresis(image, t_low, t_high):
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


def show_result(grad, img_nms, result):
    cv2.imshow('grad', grad.astype(np.uint8))
    cv2.imshow('nms', img_nms.astype(np.uint8))
    cv2.imshow('result', result.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_result(grad, img_nms, result):
    cv2.imwrite('grad.jpg', grad.astype(np.uint8))
    cv2.imwrite('nms.jpg', img_nms.astype(np.uint8))
    cv2.imwrite('result.jpg', result.astype(np.uint8))


def main():
    if len(sys.argv) != 2:
        print('usage:python canny.py <image_path>')
        exit(0)
    img_path = sys.argv[1]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    grad, tgs = calc_grad(img)
    mean = np.mean(grad)
    after_nms = nms(grad, tgs)
    img_nms = after_nms.copy()
    result = hysteresis(after_nms, mean * 0.7, mean * 2)
    show_result(grad, after_nms, result)


if __name__ == '__main__':
    main()
