import math
import cv2 as cv
import numpy as np


def detect_game_board_from_image():
    filename = 'images/tic-tac-toe.jpg'
    # Загрузка изображения
    image = cv.imread(cv.samples.findFile(filename))

    if image is None:
        print('Ошибка открытия изображения')
        return -1

    cv.imshow("Source", image)
    cv.waitKey()

    # Применение алгоритма Canny для обнаружения границ
    dst = cv.Canny(image, 50, 200, None, 3)

    # Преобразование изображения в оттенки серого
    gray_image = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    # Выполнение преобразования Хафа для поиска прямых линий
    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

    delta = 0.05  # Погрешность на угол theta к осям координат для линий поля
    horizontal_theta = 1.57  # Горизонтальные линии
    vertical_theta = 0.0  # Вертикальные линии

    same_line_delta = 20  # если линии находятся на таком расстоянии, значит это одна линия

    horizontal_lines = []
    vertical_lines = []

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            if abs(vertical_theta - theta) < delta:
                same_exists = False
                for line in vertical_lines:
                    if abs(rho - line[0][0]) < same_line_delta:
                        same_exists = True
                if not same_exists:
                    vertical_lines.append(lines[i])
                    cv.line(gray_image, pt1, pt2, (255, 255, 255), 3, cv.LINE_AA)

            elif abs(horizontal_theta - theta) < delta:
                same_exists = False
                for line in horizontal_lines:
                    if abs(rho - line[0][0]) < same_line_delta:
                        same_exists = True
                if not same_exists:
                    horizontal_lines.append(lines[i])
                    cv.line(gray_image, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", gray_image)
    cv.waitKey()

    # Мы нашли 2 вертикальные и 2 горизонтальные линии
    # Теперь найдем центры клеток поля

    horizontal_point_distance_half_coord = abs(horizontal_lines[0][0][0] - horizontal_lines[1][0][0]) / 2  # Половина расстояния между горизонтальными линиями
    vertical_point_distance_half_coord = abs(vertical_lines[0][0][0] - vertical_lines[1][0][0]) / 2  # Половина расстояния между вертикальными линиями

    min_horizontal_line_coord = min(float(horizontal_lines[0][0][0]), float(horizontal_lines[1][0][0]))  # Расстояние до нижней горизонтальной линии
    min_vertical_line_coord = min(float(vertical_lines[0][0][0]), float(vertical_lines[1][0][0]))  # Расстояние до левой вертикальной линии

    max_horizontal_line_coord = max(float(horizontal_lines[0][0][0]), float(horizontal_lines[1][0][0]))  # Расстояние до верхней горизонтальной линии
    max_vertical_line_coord = max(float(vertical_lines[0][0][0]), float(vertical_lines[1][0][0]))  # Расстояние до правой вертикальной линии

    top_left_cell = [
        min_vertical_line_coord - vertical_point_distance_half_coord,
        max_horizontal_line_coord + horizontal_point_distance_half_coord
    ]

    top_center_cell = [
        min_vertical_line_coord + vertical_point_distance_half_coord,
        max_horizontal_line_coord + horizontal_point_distance_half_coord
    ]

    top_right_cell = [
        max_vertical_line_coord + vertical_point_distance_half_coord,
        max_horizontal_line_coord + horizontal_point_distance_half_coord
    ]

    middle_left_cell = [
        min_vertical_line_coord - vertical_point_distance_half_coord,
        min_horizontal_line_coord + horizontal_point_distance_half_coord
    ]

    middle_center_cell = [
        min_vertical_line_coord + vertical_point_distance_half_coord,
        min_horizontal_line_coord + horizontal_point_distance_half_coord
    ]

    middle_right_cell = [
        max_vertical_line_coord + vertical_point_distance_half_coord,
        min_horizontal_line_coord + horizontal_point_distance_half_coord
    ]

    bottom_left_cell = [
        min_vertical_line_coord - vertical_point_distance_half_coord,
        min_horizontal_line_coord - horizontal_point_distance_half_coord
    ]

    bottom_center_cell = [
        min_vertical_line_coord + vertical_point_distance_half_coord,
        min_horizontal_line_coord - horizontal_point_distance_half_coord
    ]

    bottom_right_cell = [
        max_vertical_line_coord + vertical_point_distance_half_coord,
        min_horizontal_line_coord - horizontal_point_distance_half_coord
    ]

    cells_center_coords = [[top_left_cell, top_center_cell, top_right_cell],
                           [middle_left_cell, middle_center_cell, middle_right_cell],
                           [bottom_left_cell, bottom_center_cell, bottom_right_cell]]

    print(cells_center_coords)

    cells_elements = [['-', '-', '-'],
                      ['-', '-', '-'],
                      ['-', '-', '-']]

    # Поиск кругов - ноликов
    circle_image = cv.imread(filename)

    circle_image = cv.cvtColor(circle_image, cv.COLOR_BGR2GRAY)

    circle_image = cv.medianBlur(circle_image, 5)

    rows = circle_image.shape[0]
    circles = cv.HoughCircles(circle_image, cv.HOUGH_GRADIENT, 1, rows / 8,
                              param1=100, param2=30,
                              minRadius=0, maxRadius=100)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])

            cv.circle(gray_image, center, 1, (200, 100, 100), 3)
            radius = i[2]
            cv.circle(gray_image, center, radius, (100, 255, 255), 3)

            center_x = center[0]
            center_y = center[1]

            if check_less(center_x, min_vertical_line_coord) and check_less(center_y, min_horizontal_line_coord):
                cells_elements[0][0] = 'o'
            elif check_middle(center_x, min_vertical_line_coord, max_vertical_line_coord) and check_less(center_y, min_horizontal_line_coord):
                cells_elements[0][1] = 'o'
            elif check_large(center_x, max_vertical_line_coord) and check_less(center_y, min_horizontal_line_coord):
                cells_elements[0][2] = 'o'
            elif check_less(center_x, min_vertical_line_coord) and check_middle(center_y, min_horizontal_line_coord, max_horizontal_line_coord):
                cells_elements[1][0] = 'o'
            elif check_middle(center_x, min_vertical_line_coord, max_vertical_line_coord) and check_middle(center_y, min_horizontal_line_coord, max_horizontal_line_coord):
                cells_elements[1][1] = 'o'
            elif check_large(center_x, max_vertical_line_coord) and check_middle(center_y, min_horizontal_line_coord, max_horizontal_line_coord):
                cells_elements[1][2] = 'o'
            elif check_less(center_x, min_vertical_line_coord) and check_large(center_y, max_horizontal_line_coord):
                cells_elements[2][0] = 'o'
            elif check_middle(center_x, min_vertical_line_coord, max_vertical_line_coord) and check_large(center_y, max_horizontal_line_coord):
                cells_elements[2][1] = 'o'
            elif check_large(center_x, max_vertical_line_coord) and check_large(center_y, max_horizontal_line_coord):
                cells_elements[2][2] = 'o'

    # Поиск крестиков
    cross_image = cv.imread(filename)
    cross_image = cv.cvtColor(cross_image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(cross_image, 100, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv.contourArea(contour)
        if 1000 < area < 10_000:
            M = cv.moments(contour)
            if M['m00'] != 0:
                center_x = int(M['m10'] / M['m00'])
                center_y = int(M['m01'] / M['m00'])

                if check_less(center_x, min_vertical_line_coord) and check_less(center_y, min_horizontal_line_coord) and cells_elements[0][0] != 'o':
                    cells_elements[0][0] = 'x'
                elif check_middle(center_x, min_vertical_line_coord, max_vertical_line_coord) and check_less(center_y,min_horizontal_line_coord) and cells_elements[0][1] != 'o':
                    cells_elements[0][1] = 'x'
                elif check_large(center_x, max_vertical_line_coord) and check_less(center_y, min_horizontal_line_coord) and cells_elements[0][2] != 'o':
                    cells_elements[0][2] = 'x'
                elif check_less(center_x, min_vertical_line_coord) and check_middle(center_y, min_horizontal_line_coord,max_horizontal_line_coord) and cells_elements[1][0] != 'o':
                    cells_elements[1][0] = 'x'
                elif check_middle(center_x, min_vertical_line_coord, max_vertical_line_coord) and check_middle(center_y, min_horizontal_line_coord,max_horizontal_line_coord) and cells_elements[1][1] != 'o':
                    cells_elements[1][1] = 'x'
                elif check_large(center_x, max_vertical_line_coord) and check_middle(center_y,min_horizontal_line_coord,max_horizontal_line_coord) and cells_elements[1][2] != 'o':
                    cells_elements[1][2] = 'x'
                elif check_less(center_x, min_vertical_line_coord) and check_large(center_y, max_horizontal_line_coord) and cells_elements[2][0] != 'o':
                    cells_elements[2][0] = 'x'
                elif check_middle(center_x, min_vertical_line_coord, max_vertical_line_coord) and check_large(center_y,max_horizontal_line_coord) and cells_elements[2][1] != 'o':
                    cells_elements[2][1] = 'x'
                elif check_large(center_x, max_vertical_line_coord) and check_large(center_y,max_horizontal_line_coord) and cells_elements[2][2] != 'o':
                    cells_elements[2][2] = 'x'
                cv.circle(gray_image, (center_x, center_y), 10, (255, 0, 0), -1)

    print(cells_elements)

    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", gray_image)
    cv.waitKey()

    return cells_center_coords, cells_elements


def check_less(src_coord, coord):
    return src_coord < coord

def check_large(src_coord, coord):
    return src_coord > coord

def check_middle(src, min, max):
    return min < src < max


detect_game_board_from_image()
