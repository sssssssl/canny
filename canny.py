import cv2
import numpy as np

# kernel_gx = np.array([
#     [-1, 0, 1],
#     [-2, 0, 2],
#     [-1, 0, 1]
# ])

# kernel_gy = np.array([
#     [1, 2, 1],
#     [0, 0, 0],
#     [-1, -2, -1]
# ])

# kernel_gx = np.array([
#     [0, 1, 1],
#     [0, -1, -1],
#     [0, 0, 0]
# ]) *2

# kernel_gy = np.array([
#     [0, 1, -1],
#     [0, 1, -1],
#     [0, 0, 0]
# ]) *2

kernel_gx = np.array([
    [1, 1],
    [-1, -1],
]) *2

kernel_gy = np.array([
    [1, -1],
    [1, -1]
]) *2

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


def map_angles(theta):
    """
    arctan2 返回 (-pi, pi]（-pi是取不到的）,将其小于0的部分加上2pi，映射为 [0, 2) 区间
    乘上4映射为 [0, 8) 区间，将所有元素截断取整（等价于将区间划成八块）
    最大值抑制时要访问的相邻点的坐标差[m,n]按逆时针顺序保存在列表 nearby_indexes 中
    """
    fracs = theta / np.pi  # [-1, 1]
    neg_fracs = np.minimum(fracs, 0)
    pos_fracs = np.clip(fracs, 0, 1)
    neg_fracs[neg_fracs < 0] += 2  # from [-1, 0] to [1, 2]
    indexes = ((pos_fracs + neg_fracs) * 4).astype(int)  # `*4` means `*180/45`
    return indexes


def nms(grad, angles):
    indexes = map_angles(angles)
    after_nms = np.zeros(grad.shape)
    height, width = grad.shape
    for i in range(height):
        for j in range(width):
            index_1 = indexes[i][j]
            index_2 = (index_1 + 1) % 8
            grad_nearby_1 = get_nearby_grad(grad, i, j, index_1, height, width)
            grad_nearby_2 = get_nearby_grad(grad, i, j, index_2, height, width)
            if grad[i][j] > grad_nearby_1 and grad[i][j] > grad_nearby_2:
                after_nms[i][j] = grad[i][j]
    return after_nms


def get_nearby_grad(grad, i, j, index, height, width):
    val = 0
    dx, dy = nearby_indexes[index]
    pi, pj = i + dx, j + dy
    if validate_index(pi, pj, height, width):
        val = grad[pi, pj]
    return val


def validate_index(i, j, height, width):
    return i >= 0 and i < height and j > 0 and j < width


def double_thresholding(image, t1, t2):
    img_low = (image > t1) * image
    img_high = (image > t2) * image
    return img_low, img_high


def connect_img_high(img_high, img_low):
    result = np.zeros(img_high.shape)
    height, width = img_high.shape
    for i in range(height):
        for j in range(width):
            if img_high[i][j] > 0:
                result[i][j] = img_high[i][j]
            elif img_high[i][j] == 0 and img_low[i][j] > 0 and look_around(img_low, i, j):
                result[i][j] = img_low[i][j]
    return result


def look_around(img, i, j):
    height, width = img.shape
    for dx, dy in nearby_indexes:
        x, y = i + dx, j + dy
        if validate_index(x, y, height, width) and img[x][y] > 0:
            return True
    return False


def main():
    img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    grad, angles = calc_grad(img)
    after_nms = nms(grad, angles)
    # t1, t2 = double_thresholding(after_nms, 20, 50)
    # res = connect_img_high(t2, t1)
    # cv2.imwrite('grad.jpg', grad)
    # cv2.imwrite('nms.jpg', after_nms)
    # cv2.imwrite('res.jpg', res)
    cv2.imshow('grad', grad)
    cv2.imshow('nms', after_nms)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
