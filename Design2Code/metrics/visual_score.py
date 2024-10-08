import cv2
import numpy as np

# This is a patch for color map, which is not updated for newer version of numpy
def patch_asscalar(a):
    return a.item()
setattr(np, "asscalar", patch_asscalar)

import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import random
import os
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from tqdm import tqdm 
from pathlib import Path
from PIL import Image, ImageDraw
import torch
import clip
from copy import deepcopy
from collections import Counter
from Design2Code.metrics.ocr_free_utils import get_blocks_ocr_free
from Design2Code.data_utils.dedup_post_gen import check_repetitive_content
from bs4 import BeautifulSoup, NavigableString, Comment
import re
import math
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 计算两个文本块的文本相似性
def calculate_similarity(block1, block2, max_distance=1.42):
    text_similarity = SequenceMatcher(None, block1['text'], block2['text']).ratio()
    return text_similarity

# 调整成本矩阵以考虑上下文相似性
def adjust_cost_for_context(cost_matrix, consecutive_bonus=1.0, window_size=20):
    if window_size <= 0:
        return cost_matrix

    n, m = cost_matrix.shape
    adjusted_cost_matrix = np.copy(cost_matrix)

    for i in range(n):
        for j in range(m):
            bonus = 0
            if adjusted_cost_matrix[i][j] >= -0.5:
                continue
            nearby_matrix = cost_matrix[max(0, i - window_size):min(n, i + window_size + 1), max(0, j - window_size):min(m, j + window_size + 1)]
            flattened_array = nearby_matrix.flatten()
            sorted_array = np.sort(flattened_array)[::-1]
            sorted_array = np.delete(sorted_array, np.where(sorted_array == cost_matrix[i, j])[0][0])
            top_k_elements = sorted_array[- window_size * 2:]
            sum_top_k = np.sum(top_k_elements)
            bonus = consecutive_bonus * sum_top_k
            adjusted_cost_matrix[i][j] += bonus
    return adjusted_cost_matrix
# 创建成本矩阵，用于块匹配
def create_cost_matrix(A, B):
    '''
    这个函数 create_cost_matrix 的目的是生成一个成本矩阵，用于表示块列表 A 和 B 之间的匹配成本。成本矩阵中的每个元素表示块 A[i] 和块 B[j] 之间的匹配成本。具体来说，成本矩阵中的元素是块之间相似度的负值。
'''
    n = len(A)
    m = len(B)
    cost_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cost_matrix[i, j] = -calculate_similarity(A[i], B[j])  # 以两个文本块的文本相似性的负值作为成本矩阵中的元素
    return cost_matrix

# 在图像上绘制匹配的边界框
def draw_matched_bboxes(img1, img2, matched_bboxes):
    # Create copies of images to draw on
    img1_drawn = img1.copy()
    img2_drawn = img2.copy()

    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    

    # Iterate over matched bounding boxes
    for bbox_pair in matched_bboxes:
        # Random color for each pair
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Ensure that the bounding box coordinates are integers
        bbox1 = [int(bbox_pair[0][0] * w1), int(bbox_pair[0][1] * h1), int(bbox_pair[0][2] * w1), int(bbox_pair[0][3] * h1)]
        bbox2 = [int(bbox_pair[1][0] * w2), int(bbox_pair[1][1] * h2), int(bbox_pair[1][2] * w2), int(bbox_pair[1][3] * h2)]

        # Draw bbox on the first image
        top_left_1 = (bbox1[0], bbox1[1])
        bottom_right_1 = (bbox1[0] + bbox1[2], bbox1[1] + bbox1[3])
        img1_drawn = cv2.rectangle(img1_drawn, top_left_1, bottom_right_1, color, 2)

        # Draw bbox on the second image
        top_left_2 = (bbox2[0], bbox2[1])
        bottom_right_2 = (bbox2[0] + bbox2[2], bbox2[1] + bbox2[3])
        img2_drawn = cv2.rectangle(img2_drawn, top_left_2, bottom_right_2, color, 2)

    return img1_drawn, img2_drawn

# 计算两个点之间的最大距离
def calculate_distance_max_1d(x1, y1, x2, y2):
    distance = max(abs(x2 - x1), abs(y2 - y1))
    return distance

# 计算两个高度的比率
def calculate_ratio(h1, h2):
    return max(h1, h2) / min(h1, h2)

# 用于颜色相似性计算
def rgb_to_lab(rgb):
    """
    Convert an RGB color to Lab color space.
    RGB values should be in the range [0, 255].
    """
    # Create an sRGBColor object from RGB values
    rgb_color = sRGBColor(rgb[0], rgb[1], rgb[2], is_upscaled=True)
    
    # Convert to Lab color space
    lab_color = convert_color(rgb_color, LabColor)
    
    return lab_color
# 用于颜色相似性计算
def color_similarity_ciede2000(rgb1, rgb2):
    """
    Calculate the color similarity between two RGB colors using the CIEDE2000 formula.
    Returns a similarity score between 0 and 1, where 1 means identical.
    """
    # Convert RGB colors to Lab
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)
    
    # Calculate the Delta E (CIEDE2000)
    delta_e = delta_e_cie2000(lab1, lab2)
    
    # Normalize the Delta E value to get a similarity score
    # Note: The normalization method here is arbitrary and can be adjusted based on your needs.
    # A delta_e of 0 means identical colors. Higher values indicate more difference.
    # For visualization purposes, we consider a delta_e of 100 to be completely different.
    similarity = max(0, 1 - (delta_e / 100))
    
    return similarity

# 计算当前匹配的成本
def calculate_current_cost(cost_matrix, row_ind, col_ind):
    return cost_matrix[row_ind, col_ind].sum()

# 合并两个块而不进行检查
def merge_blocks_wo_check(block1, block2):
    # Concatenate text
    merged_text = block1['text'] + " " + block2['text']

    # Calculate bounding box
    x_min = min(block1['bbox'][0], block2['bbox'][0])
    y_min = min(block1['bbox'][1], block2['bbox'][1])
    x_max = max(block1['bbox'][0] + block1['bbox'][2], block2['bbox'][0] + block2['bbox'][2])
    y_max = max(block1['bbox'][1] + block1['bbox'][3], block2['bbox'][1] + block2['bbox'][3])
    merged_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

    # Average color
    merged_color = tuple(
        (color1 + color2) // 2 for color1, color2 in zip(block1['color'], block2['color'])
    )

    return {'text': merged_text, 'bbox': merged_bbox, 'color': merged_color}

# 计算当前匹配的成本
def calculate_current_cost(cost_matrix, row_ind, col_ind):
    return cost_matrix[row_ind, col_ind].tolist()

# 找到最大匹配
def find_maximum_matching(A, B, consecutive_bonus, window_size):
    '''
    find_maximum_matching 函数的目的是在两个块列表 A 和 B 之间找到最优匹配，同时计算匹配的成本。它使用了匈牙利算法（线性和分配算法）来解决这个匹配问题。
    '''
    cost_matrix = create_cost_matrix(A, B)  # 使用 create_cost_matrix 函数生成一个成本矩阵 cost_matrix，表示块列表 A 和 B 之间的匹配成本
    cost_matrix = adjust_cost_for_context(cost_matrix, consecutive_bonus, window_size) # ？？ 使用 adjust_cost_for_context 函数根据 consecutive_bonus 和 window_size 调整成本矩阵。这可能用于奖励连续匹配的块或在某些范围内调整成本。
    row_ind, col_ind = linear_sum_assignment(cost_matrix)   # ？？ 使用 linear_sum_assignment 函数（匈牙利算法）在调整后的成本矩阵上找到最优匹配。该函数返回行索引 row_ind 和列索引 col_ind，表示最优匹配。
    current_cost = calculate_current_cost(cost_matrix, row_ind, col_ind)  # 使用 calculate_current_cost 函数计算当前匹配的总成本。
    return list(zip(row_ind, col_ind)), current_cost, cost_matrix  # 返回最优匹配 list(zip(row_ind, col_ind))，当前成本 current_cost，以及成本矩阵 cost_matrix。

# 用于块合并
def remove_indices(lst, indices):
    for index in sorted(indices, reverse=True):
        if index < len(lst):
            lst.pop(index)
    return lst

# 用于块合并
def merge_blocks_by_list(blocks, merge_list):
    pop_list = []
    while True:
        if len(merge_list) == 0:
            remove_indices(blocks, pop_list)
            return blocks

        i = merge_list[0][0]
        j = merge_list[0][1]
    
        blocks[i] = merge_blocks_wo_check(blocks[i], blocks[j])
        pop_list.append(j)
    
        merge_list.pop(0)
        if len(merge_list) > 0:
            new_merge_list = []
            for k in range(len(merge_list)):
                if merge_list[k][0] != i and merge_list[k][1] != i and merge_list[k][0] != j and merge_list[k][1] != j:
                    new_merge_list.append(merge_list[k])
            merge_list = new_merge_list

# 打印匹配结果
def print_matching(matching, blocks1, blocks2, cost_matrix):
    for i, j in matching:
        print(f"{blocks1[i]} matched with {blocks2[j]}, cost {cost_matrix[i][j]}")

# 计算两个列表的均值差异
def difference_of_means(list1, list2):
    counter1 = Counter(list1)
    counter2 = Counter(list2)

    for element in set(list1) & set(list2):
        common_count = min(counter1[element], counter2[element])
        counter1[element] -= common_count
        counter2[element] -= common_count

    unique_list1 = [item for item in counter1.elements()]
    unique_list2 = [item for item in counter2.elements()]

    mean_list1 = sum(unique_list1) / len(unique_list1) if unique_list1 else 0
    mean_list2 = sum(unique_list2) / len(unique_list2) if unique_list2 else 0

    if mean_list1 - mean_list2 > 0:
        if min(unique_list1) > min(unique_list2):
            return mean_list1 - mean_list2
        else:
            return 0.0
    else:
        return mean_list1 - mean_list2

# 找到可能的合并，通过这种方式，函数可以找到两个块列表之间的最佳匹配，同时尽可能减少块的数量
def find_possible_merge(A, B, consecutive_bonus, window_size, debug=False):
    '''
    这个函数通过迭代优化的方法，不断尝试合并相邻的块，并评估匹配成本的变化，直到无法进一步优化为止。它使用了多个辅助函数来计算匹配、合并块和评估成本变化。通过这种方式，函数可以找到两个块列表之间的最佳匹配，同时尽可能减少块的数量。
    '''
    merge_bonus = 0.0
    merge_windows = 1
    # 用于根据 diff（匹配成本的变化）对合并列表进行排序
    def sortFn(value):
        return value[2]

    while True:
        A_changed = False
        B_changed = False

        matching, current_cost, cost_matrix = find_maximum_matching(A, B, merge_bonus, merge_windows)  # 计算当前匹配和成本
        if debug:
            print("Current cost of the solution:", current_cost)
            print_matching(matching, A, B, cost_matrix)
    
        if len(A) >= 2:  # 如果 A 的长度大于等于 2，尝试合并相邻的块
            merge_list = []
            for i in range(len(A) - 1):
                new_A = deepcopy(A)  # deepcopy 是 Python 标准库 copy 模块中的一个函数，用于创建一个对象的深拷贝
                new_A[i] = merge_blocks_wo_check(new_A[i], new_A[i + 1])  # 返回{'text': merged_text, 'bbox': merged_bbox, 'color': merged_color}
                new_A.pop(i + 1)
    
                updated_matching, updated_cost, cost_matrix = find_maximum_matching(new_A, B, merge_bonus, merge_windows)
                diff = difference_of_means(current_cost, updated_cost)  # 计算成本变化 diff，如果成本变化大于 0.05（成本下降了），则将这对块添加到 merge_list 中
                if  diff > 0.05:
                    merge_list.append([i, i + 1, diff])
                    if debug:
                        print(new_A[i]['text'], diff)

            merge_list.sort(key=sortFn, reverse=True)   # 对 merge_list 按 diff 进行排序，并合并 A 中的块
            if len(merge_list) > 0:
                A_changed = True
                A = merge_blocks_by_list(A, merge_list)
                matching, current_cost, cost_matrix = find_maximum_matching(A, B, merge_bonus, merge_windows)  # 重新计算优化之后得到的新的A和之前的B的匹配
                if debug:
                    print("Cost after optimization A:", current_cost)
        #  与A的优化合并同理
        if len(B) >= 2:
            merge_list = []
            for i in range(len(B) - 1):
                new_B = deepcopy(B)
                new_B[i] = merge_blocks_wo_check(new_B[i], new_B[i + 1])
                new_B.pop(i + 1)
    
                updated_matching, updated_cost, cost_matrix = find_maximum_matching(A, new_B, merge_bonus, merge_windows)
                diff = difference_of_means(current_cost, updated_cost)
                if diff > 0.05:
                    merge_list.append([i, i + 1, diff])
                    if debug:
                        print(new_B[i]['text'], diff)

            merge_list.sort(key=sortFn, reverse=True)
            if len(merge_list) > 0:
                B_changed = True
                B = merge_blocks_by_list(B, merge_list)
                matching, current_cost, cost_matrix = find_maximum_matching(A, B, merge_bonus, merge_windows)
                if debug:
                    print("Cost after optimization B:", current_cost)
        # 如果 A 和 B 都没有发生变化，退出循环，无需在进行优化了
        if not A_changed and not B_changed:
            break
    matching, _, _ = find_maximum_matching(A, B, consecutive_bonus, window_size)  # 返回优化后的块列表 A 和 B 以及最终匹配 matching
    return A, B, matching

# 按边界框合并块
def merge_blocks_by_bbox(blocks):
    '''
    函数的目的是合并具有相同边界框（bounding box）的块。它遍历输入的块列表，将具有相同边界框的块合并为一个块，并将合并后的块返回。
    '''
    merged_blocks = {}
    
    # Traverse and merge blocks
    for block in blocks:
        bbox = tuple(block['bbox'])  # Convert bbox to tuple for hashability
        if bbox in merged_blocks:
            # Merge with existing block
            existing_block = merged_blocks[bbox]
            existing_block['text'] += ' ' + block['text']
            existing_block['color'] = [(ec + c) / 2 for ec, c in zip(existing_block['color'], block['color'])]
        else:
            # Add new block
            merged_blocks[bbox] = block

    return list(merged_blocks.values())

# 使用图像修复技术掩盖边界框
def mask_bounding_boxes_with_inpainting(image, bounding_boxes):
    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Create a black mask
    mask = np.zeros(image_cv.shape[:2], dtype=np.uint8)

    height, width = image_cv.shape[:2]

    # Draw white rectangles on the mask
    for bbox in bounding_boxes:
        x_ratio, y_ratio, w_ratio, h_ratio = bbox
        x = int(x_ratio * width)
        y = int(y_ratio * height)
        w = int(w_ratio * width)
        h = int(h_ratio * height)
        mask[y:y+h, x:x+w] = 255

    # Use inpainting
    inpainted_image = cv2.inpaint(image_cv, mask, 3, cv2.INPAINT_TELEA)

    # Convert back to PIL format
    inpainted_image_pil = Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))

    return inpainted_image_pil

# 重新缩放图像并掩盖块
def rescale_and_mask(image_path, blocks):
    # Load the image
    with Image.open(image_path) as img:
        if len(blocks) > 0:
            # use inpainting instead of simple mask
            img = mask_bounding_boxes_with_inpainting(img, blocks)

        width, height = img.size

        # Determine which side is shorter
        if width < height:
            # Width is shorter, scale height to match the width
            new_size = (width, width)
        else:
            # Height is shorter, scale width to match the height
            new_size = (height, height)

        # Resize the image while maintaining aspect ratio
        img_resized = img.resize(new_size, Image.LANCZOS)

        return img_resized

# 计算两个图像块的CLIP相似性
def calculate_clip_similarity_with_blocks(image_path1, image_path2, blocks1, blocks2):
    # Load and preprocess images
    image1 = preprocess(rescale_and_mask(image_path1, [block['bbox'] for block in blocks1])).unsqueeze(0).to(device)
    image2 = preprocess(rescale_and_mask(image_path2, [block['bbox'] for block in blocks2])).unsqueeze(0).to(device)

    # Calculate features
    with torch.no_grad():
        image_features1 = model.encode_image(image1)
        image_features2 = model.encode_image(image2)

    # Normalize features
    image_features1 /= image_features1.norm(dim=-1, keepdim=True)
    image_features2 /= image_features2.norm(dim=-1, keepdim=True)

    # Calculate cosine similarity
    similarity = (image_features1 @ image_features2.T).item()

    return similarity

# 截断重复的HTML元素
def truncate_repeated_html_elements(soup, max_count=50):
    content_counts = {}

    for element in soup.find_all(True):
        if isinstance(element, (NavigableString, Comment)):
            continue
        
        try:
            element_html = str(element)
        except:
            element.decompose()
            continue
        content_counts[element_html] = content_counts.get(element_html, 0) + 1

        if content_counts[element_html] > max_count:
            element.decompose()

    return str(soup)

# 生成简单的HTML文件
def make_html(filename):
    with open(filename, 'r') as file:
        content = file.read()

    if not re.search(r'<html[^>]*>', content, re.IGNORECASE):
        new_content = f'<html><body><p>{content}</p></body></html>'
        with open(filename, 'w') as file:
            file.write(new_content)

# 预处理HTML文件
def pre_process(html_file):
    check_repetitive_content(html_file)
    make_html(html_file)
    with open(html_file, 'r') as file:
        soup = BeautifulSoup(file, 'html.parser')
    soup_str = truncate_repeated_html_elements(soup)
    with open(html_file, 'w') as file:
        file.write(soup_str)

# 是主要的评估函数，用于比较多个预测HTML文件和一个原始HTML文件之间的相似性
def visual_eval_v3_multi(input_list, debug=False):
    # input_list:包含预测HTML文件列表和一个原始HTML文件
    # 初始化和预处理
    # 从 input_list 中提取预测HTML文件列表和原始HTML文件。
    # 生成对应的PNG图像文件名列表。 
    predict_html_list, original_html = input_list[0], input_list[1]
    predict_img_list = [html.replace(".html", ".png") for html in predict_html_list]  # 生成对应的PNG图像文件名列表
    # try:
    # 处理预测HTML文件
    # 对每个预测HTML文件进行预处理，修复HTML语法错误。
    # 使用系统命令生成PNG图像。
    # 使用 get_blocks_ocr_free 函数从图像中获取块信息，并存储在 predict_blocks_list 中。
    predict_blocks_list = []
    for predict_html in predict_html_list:
        predict_img = predict_html.replace(".html", ".png")
        # This will help fix some html syntax error
        pre_process(predict_html)
        # os.system(f"python3 metrics/screenshot_single.py --html {predict_html} --png {predict_img}")
        predict_blocks = get_blocks_ocr_free(predict_img)
        predict_blocks_list.append(predict_blocks)
    # 处理原始HTML文件
    # 生成原始HTML文件对应的PNG图像。
    # 使用 get_blocks_ocr_free 函数从图像中获取块信息，并使用 merge_blocks_by_bbox 函数合并块。
    original_img = original_html.replace(".html", ".png")
    os.system(f"python3 metrics/screenshot_single.py --html {original_html} --png {original_img}")
    original_blocks = get_blocks_ocr_free(original_img)
    original_blocks = merge_blocks_by_bbox(original_blocks)

    # Consider context similarity for block matching
    # 块匹配和相似性计算
    # 设置连续奖励和窗口大小参数。
    # 初始化返回分数列表。
    consecutive_bonus, window_size = 0.1, 1

    return_score_list = []
    # 对每个预测块进行处理。如果没有检测到块，计算CLIP相似性分数并继续下一个块。
    for k, predict_blocks in enumerate(predict_blocks_list):
        if len(predict_blocks) == 0:
                print("[Warning] No detected blocks in: ", predict_img_list[k])
                final_clip_score = calculate_clip_similarity_with_blocks(predict_img_list[k], original_img, predict_blocks, original_blocks)
                return_score_list.append([0.0, 0.2 * final_clip_score, (0.0, 0.0, 0.0, 0.0, final_clip_score)])
                continue
        elif len(original_blocks) == 0:
                print("[Warning] No detected blocks in: ", original_img)
                final_clip_score = calculate_clip_similarity_with_blocks(predict_img_list[k], original_img, predict_blocks, original_blocks)
                return_score_list.append([0.0, 0.2 * final_clip_score, (0.0, 0.0, 0.0, 0.0, final_clip_score)])
                continue
        # 块合并和匹配
        # 合并预测块。
        # 使用 find_possible_merge 函数找到可能的块合并和匹配。
        # 过滤掉文本相似性低于0.5的匹配。
        if debug:
            print(predict_blocks)
            print(original_blocks)
    
        predict_blocks = merge_blocks_by_bbox(predict_blocks)  # 合并具有相同边界框（bounding box）的块
        predict_blocks_m, original_blocks_m, matching = find_possible_merge(predict_blocks, deepcopy(original_blocks), consecutive_bonus, window_size, debug=debug)  # ！??
        
        filtered_matching = []
        for i, j in matching:
            text_similarity = SequenceMatcher(None, predict_blocks_m[i]['text'], original_blocks_m[j]['text']).ratio()
            # Filter out matching with low similarity
            if text_similarity < 0.5:
                continue
            filtered_matching.append([i, j, text_similarity])
        matching = filtered_matching
        # 计算未匹配的面积
        indices1 = [item[0] for item in matching]
        indices2 = [item[1] for item in matching]

        matched_list = []
        sum_areas = []
        matched_areas = []
        matched_text_scores = []
        position_scores = []
        text_color_scores = []
    
        unmatched_area_1 = 0.0
        for i in range(len(predict_blocks_m)):
            if i not in indices1:
                unmatched_area_1 += predict_blocks_m[i]['bbox'][2] * predict_blocks_m[i]['bbox'][3]
        unmatched_area_2 = 0.0
        for j in range(len(original_blocks_m)):
            if j not in indices2:
                unmatched_area_2 += original_blocks_m[j]['bbox'][2] * original_blocks_m[j]['bbox'][3]
        sum_areas.append(unmatched_area_1 + unmatched_area_2)
        # 计算匹配块的相似性
        # 计算匹配块的面积、位置相似性、颜色相似性等。
        # 将匹配块的信息存储在各自的列表中
        for i, j, text_similarity in matching:
            sum_block_area = predict_blocks_m[i]['bbox'][2] * predict_blocks_m[i]['bbox'][3] + original_blocks_m[j]['bbox'][2] * original_blocks_m[j]['bbox'][3]

            # Consider the max postion shift, either horizontally or vertically
            position_similarity = 1 - calculate_distance_max_1d(predict_blocks_m[i]['bbox'][0] + predict_blocks_m[i]['bbox'][2] / 2, \
                                                    predict_blocks_m[i]['bbox'][1] + predict_blocks_m[i]['bbox'][3] / 2, \
                                                    original_blocks_m[j]['bbox'][0] + original_blocks_m[j]['bbox'][2] / 2, \
                                                    original_blocks_m[j]['bbox'][1] + original_blocks_m[j]['bbox'][3] / 2)
            # Normalized ciede2000 formula
            text_color_similarity = color_similarity_ciede2000(predict_blocks_m[i]['color'], original_blocks_m[j]['color'])
            matched_list.append([predict_blocks_m[i]['bbox'], original_blocks_m[j]['bbox']])
    
            # validation check
            if min(predict_blocks_m[i]['bbox'][2], original_blocks_m[j]['bbox'][2], predict_blocks_m[i]['bbox'][3], original_blocks_m[j]['bbox'][3]) == 0:
                print(f"{predict_blocks_m[i]} matched with {original_blocks_m[j]}")
            assert calculate_ratio(predict_blocks_m[i]['bbox'][2], original_blocks_m[j]['bbox'][2]) > 0 and calculate_ratio(predict_blocks_m[i]['bbox'][3], original_blocks_m[j]['bbox'][3]) > 0, f"{predict_blocks_m[i]} matched with {original_blocks_m[j]}"
    
            sum_areas.append(sum_block_area)
            matched_areas.append(sum_block_area)
            matched_text_scores.append(text_similarity)
            position_scores.append(position_similarity)
            text_color_scores.append(text_color_similarity)
    
            if debug:
                print(f"{predict_blocks_m[i]} matched with {original_blocks_m[j]}")
                print(SequenceMatcher(None, predict_blocks_m[i]['text'], original_blocks_m[j]['text']).ratio())
                print("text similarity score", text_similarity)
                print("position score", position_similarity)
                print("color score", text_color_similarity)
                print("----------------------------------")
                pass
        """
        if debug:
            img1 = cv2.imread(predict_img_list[k])
            img2 = cv2.imread(original_img)
            img1_with_boxes, img2_with_boxes = draw_matched_bboxes(img1, img2, matched_list)
        
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(img1_with_boxes, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(img2_with_boxes, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        # """
        # 计算最终得分
        # 计算最终得分，包括尺寸得分、文本匹配得分、位置得分、颜色得分和CLIP相似性得分。
        # 将最终得分存储在返回列表中
        if len(matched_areas) > 0:
            sum_sum_areas = np.sum(sum_areas)
    
            final_size_score = np.sum(matched_areas) / np.sum(sum_areas)
            final_matched_text_score = np.mean(matched_text_scores)
            final_position_score = np.mean(position_scores)
            final_text_color_score = np.mean(text_color_scores)
            final_clip_score = calculate_clip_similarity_with_blocks(predict_img_list[k], original_img, predict_blocks, original_blocks)
            final_score = 0.2 * (final_size_score + final_matched_text_score + final_position_score + final_text_color_score + final_clip_score)
            return_score_list.append([sum_sum_areas, final_score, (final_size_score, final_matched_text_score, final_position_score, final_text_color_score, final_clip_score)])
        else:
            print("[Warning] No matched blocks in: ", predict_img_list[k])
            final_clip_score = calculate_clip_similarity_with_blocks(predict_img_list[k], original_img, predict_blocks, original_blocks)
            return_score_list.append([0.0, 0.2 * final_clip_score, (0.0, 0.0, 0.0, 0.0, final_clip_score)])
    return return_score_list
    
    # except:
    #     print("[Warning] Error not handled in: ", input_list)
    #     return [[0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0)] for _ in range(len(predict_html_list))]



if __name__ == "__main__":
    import os
    from pathlib import Path

    # 修改原始页面路径和生成页面路径
    original_html_path = "/root/Design2Code/Design2Code/demohtml/original_html/19.html"
    generated_html_path = "/root/Design2Code/Design2Code/demohtml/gen_html/19_gen.html"

    # 将路径添加到列表中，作为输入参数
    input_list = [[generated_html_path], original_html_path]

    # 调用visual_eval_v3_multi函数进行相似度计算
    scores = visual_eval_v3_multi(input_list, debug=True)

    # 输出相似度得分
    print("Similarity Scores:", scores)
    visual_eval_v3_multi([], debug=True)



