from time import time

import numpy as np
from sklearn.cluster import KMeans

from Plot import plot_dot


def division(gb_list, gb_list_not):
    gb_list_new = []
    for gb in gb_list:
        if len(gb) > 16:
            ball_1, ball_2 = spilt_ball(gb)
            # if len(ball_2) == 0 or len(ball_1) == 0:
            #     gb_list_not.append(gb)
            #     continue
            dm_parent = get_dm(gb)
            dm_child_1 = get_dm(ball_1)
            dm_child_2 = get_dm(ball_2)
            w = len(ball_1) + len(ball_2)
            w1 = len(ball_1) / w
            w2 = len(ball_2) / w
            w_child = w1 * dm_child_1 + w2 * dm_child_2

            if w_child < dm_parent:
                gb_list_new.extend([ball_1, ball_2])
            else:
                gb_list_not.append(gb)
        else:
            gb_list_not.append(gb)
    return gb_list_new, gb_list_not


def normalized_ball(gb_list, gb_list_not, radius_detect):
    gb_list_temp = []
    for gb in gb_list:
        if len(gb) < 2:
            gb_list_not.append(gb)
        else:
            if get_radius(gb) <= 2 * radius_detect:
                gb_list_not.append(gb)
            else:
                ball_1, ball_2 = spilt_ball(gb)
                gb_list_temp.extend([ball_1, ball_2])
    return gb_list_temp, gb_list_not


def spilt_ball(gb, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(gb)
    labels = kmeans.labels_
    ball1 = gb[labels == 0]
    ball2 = gb[labels == 1]
    return ball1, ball2


def get_dm(gb):
    center = gb.mean(axis=0)
    sq_distances = np.sum((gb - center) ** 2, axis=1)
    mean_radius = np.sqrt(np.mean(sq_distances))
    return mean_radius


def get_radius(gb):
    center = gb.mean(0)
    sq_distances = np.sum((gb - center) ** 2, axis=1)
    radius = np.max(np.sqrt(sq_distances))
    return radius


def gbc(data_list, draw_ball=False):
    gb_list_temp = [data_list]  # 1*n*d
    gb_list_not_temp = []
    # 按照质量分化
    while 1:
        ball_number_old = len(gb_list_temp) + len(gb_list_not_temp)
        gb_list_temp, gb_list_not_temp = division(gb_list_temp, gb_list_not_temp)
        ball_number_new = len(gb_list_temp) + len(gb_list_not_temp)
        if ball_number_new == ball_number_old:
            gb_list_temp = gb_list_not_temp
            break
    # 全局归一化
    radius = []
    for gb in gb_list_temp:
        if len(gb) >= 2:
            radius.append(get_radius(gb))
    radius_median = np.median(radius)
    radius_mean = np.mean(radius)
    radius_detect = max(radius_median, radius_mean)
    gb_list_not_temp = []
    while 1:
        ball_number_old = len(gb_list_temp) + len(gb_list_not_temp)
        gb_list_temp, gb_list_not_temp = normalized_ball(gb_list_temp, gb_list_not_temp, radius_detect)
        ball_number_new = len(gb_list_temp) + len(gb_list_not_temp)
        if ball_number_new == ball_number_old:
            gb_list_temp = gb_list_not_temp
            break
    gb_list = gb_list_temp
    if draw_ball:
        plot_dot(data_list, show=False)
        draw_ball(gb_list_temp)

    return gb_list


def label_mapping(data, gb_list_temp):
    ball_index = {}
    # ballCenters = np.array([np.mean(ball, axis=0) for ball in gb_list_temp])
    data_np = np.array(data)

    for j, ball in enumerate(gb_list_temp):
        ball_np = np.array(ball)
        matches = np.all(data_np[:, None] == ball_np, axis=-1)
        matched_indices = np.where(matches.any(axis=1))[0]
        if matched_indices.size > 0:
            ball_index.setdefault(j, set()).update(matched_indices)

    # return ballCenters, ball_index
    return ball_index

def assign_label(x, y, gb_list):
    """
    超高效版本（使用空间换时间，预先建立完整索引）

    参数:
    x: 原始数据集 (n_samples, n_features) 的numpy数组
    y: 原始标签 (n_samples,) 的numpy数组
    gb_list: 粒球集合，每个粒球是原始数据的索引列表

    返回:
    gb_labels: 每个粒球的标签 (n_granules,)
    """
    # 转换为numpy数组
    x_np = np.asarray(x)
    y_np = np.asarray(y)

    # 建立快速查找结构（使用空间换时间）
    # 将浮点数据转换为字符串作为键（解决浮点精度问题）
    str_points = [','.join(map(str, point)) for point in x_np]
    point_to_index = {s: i for i, s in enumerate(str_points)}

    gb_labels = []

    for granule in gb_list:
        # 获取粒球中所有点的索引
        indices = []
        for point in granule:
            s = ','.join(map(str, point))
            if s in point_to_index:
                indices.append(point_to_index[s])

        # 统计标签
        if indices:
            label_counts = np.bincount(y_np[indices])
            gb_labels.append(np.argmax(label_counts))
        else:
            gb_labels.append(-1)

    return np.array(gb_labels)
