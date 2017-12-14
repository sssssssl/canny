import cv2
import numpy as np

kernel_gx = np.array([
    [0, 1, 1],
    [0, -1, -1],
    [0, 0, 0]
])

kernel_gy = np.array([
    [0, 1, -1],
    [0, 1, -1],
    [0, 0, 0]
])

nearby_indexes = [
    [1, 0],
    [1, 1],
    [0, 1],
    [-1, 1],
    [-1, 0],
    [-1, -1],
    [0, -1],
    [1, -1]
]


def calc_grad(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    gx = cv2.filter2D(blur, cv2.CV_64F, kernel_gx)
    gy = cv2.filter2D(blur, cv2.CV_64F, kernel_gy)
    grad = np.sqrt(np.square(gx) + np.square(gy))
    angles = np.arctan2(gy, gx)
    return (grad, angles)


def validate_index(i, j, height, width):
    return i >= 0 and i < height and j >= 0 and j < width


def normal_nms(grad, angles):
    after_nms = np.zeros(grad.shape)
    height, width = grad.shape
    for i in range(height):
        for j in range(width):
            grad_p1 = 0
            grad_p2 = 0
            p1, p2 = get_nearby_points(i, j, angles[i][j])
            if validate_index(p1[0], p1[1], height, width):
                grad_p1 = grad[p1[0]][p1[1]]
            if validate_index(p2[0], p2[1], height, width):
                grad_p2 = grad[p2[0]][p2[1]]
            if grad[i][j] > grad_p1 and grad[i][j] > grad_p2:
                after_nms[i][j] = grad[i][j]
    return after_nms


def get_nearby_points(i, j, theta):
    points = []
    pi = np.pi

    if theta >= -(pi + 0.01) and theta <= -0.75 * pi:
        points = [
            [i - 1, j],
            [i - 1, j - 1]
        ]
    elif theta >= -0.75 * pi and theta < -0.5 * pi:
        points = [
            [i - 1, j - 1],
            [i, j - 1]
        ]
    elif theta >= -0.5 * pi and theta < -0.25 * pi:
        points = [
            [i, j - 1],
            [i + 1, j - 1]
        ]
    elif theta >= -0.25 * pi and theta < 0:
        points = [
            [i + 1, j - 1],
            [i + 1, j]
        ]
    elif theta >= 0 and theta < 0.25 * pi:
        points = [
            [i + 1, j],
            [i + 1, j + 1]
        ]
    elif theta >= 0.25 * pi and theta < 0.5 * pi:
        points = [
            [i + 1, j + 1],
            [i, j + 1]
        ]
    elif theta >= 0.5 * pi and theta < 0.75 * pi:
        points = [
            [i, j + 1],
            [i - 1, j + 1]
        ]
    elif theta >= 0.75 * pi and theta < (pi + 0.01):
        points = [
            [i - 1, j + 1],
            [i - 1, j]
        ]
    else:
        print(theta)
        raise Exception("wrong angle")
    return np.array(points)


def main():
    img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    grad, angles = calc_grad(img)
    cv2.imwrite('grad.jpg', grad)
    after_nms = normal_nms(grad, angles)
    cv2.imwrite('normal_nms.jpg', after_nms)

if __name__ == '__main__':
    main()
