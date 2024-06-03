import cv2
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from skimage import color, util, filters, feature

from postprocessing.image_operations import load_image

im_size = (850, 1100)
threshold = im_size[0] * .2

def process_image(path):
    image = load_image(path, as_gray=True)
    image = util.invert(image)
    np.resize(image, im_size)
    return image

def detect_staff_lines(image):
    horiz_sum = np.sum(image, axis=1)
    horiz_sum[horiz_sum < threshold] = 0
    staff_lines = feature.peak_local_max(horiz_sum).flatten()
    staff_lines = np.reshape(staff_lines, (-1, 5))
    staff_lines = np.sort(staff_lines, axis=0)
    return staff_lines

def load_features(path):
    return np.genfromtxt(
        path, 
        dtype="i8,i8,f8,S5",
        names=['x','y','length','type'],
        delimiter=',')

def find_feature_staffs(features, staffs):
    matched_staffs = np.zeros(features.shape[0])
    for f in range(features.shape[0]):
        _, y, _, _ = features[f]
        y = float(y)
        staff_dists = np.sum(np.absolute(staffs - y), axis=1)
        matched_staffs[f] = np.argmin(staff_dists)
    return matched_staffs

def find_staff_distance(staffs):
    avg_dist = 0
    for s in range(staffs.shape[0]):
        staff = staffs[s]
        dist = 0
        for l in range(staff.shape[0] - 1):
            dist += np.absolute(staff[l] - staff[l + 1])
        avg_dist += dist / float(staff.shape[0] - 1)
    avg_dist = (avg_dist / float(staffs.shape[0])) / 2
    return avg_dist

def find_pitches(features, staffs, matched_staffs):
    staff_dist = find_staff_distance(staffs)
    matched_pitches = np.zeros(features.shape[0])

    for f in range(features.shape[0]):
        staff = staffs[matched_staffs[f].astype(int)]
        highest_line = np.max(staff)
        _, y, _, _ = features[f]
        y = float(y)

        staff_line = -np.round((y - highest_line) / staff_dist)
        matched_pitches[f] = staff_line

    return matched_pitches

def construct_note(feature, pitch):
    _, _, length, type = feature
    return (type, float(length), float(pitch))

def construct_notes(features, staffs, matched_staffs, pitches):
    notes = []
    num_staffs = staffs.shape[0]

    num_notes = 0
    for i in range(num_staffs):
        notes_indices = np.where(matched_staffs == i)

        feature_x = features[notes_indices, 0][0]
        feature_x = [float(x) for x in feature_x]
        sorted_notes_indices = np.argsort(feature_x)

        these_pitches = (pitches[notes_indices])[sorted_notes_indices]
        these_features = (features[notes_indices])[sorted_notes_indices]

        these_notes = [construct_note(these_features[i], these_pitches[i]) for i in range(len(these_pitches))]

        notes[num_notes:num_notes + sorted_notes_indices.shape[0]] = these_notes
        num_notes += sorted_notes_indices.shape[0]
    return notes

def get_staff_lines(width, height, in_img, threshold=0.8):
    initial_lines = []
    row_histogram = [0] * height
    staff_lines = []
    staff_lines_thicknesses = []

    for r in range(height):
        for c in range(width):
            if in_img[r][c] == 0:
                row_histogram[r] += 1

    for row in range(len(row_histogram)):
        if row_histogram[row] >= (width * threshold):
            initial_lines.append(row)

    it = 0
    cur_thickness = 1

    while it < len(initial_lines):
        if cur_thickness == 1:
            staff_lines.append(initial_lines[it])

        if it == int(len(initial_lines) - 1):
            staff_lines_thicknesses.append(cur_thickness)
        elif initial_lines[it] + 1 == initial_lines[it + 1]:
            cur_thickness += 1
        else:
            staff_lines_thicknesses.append(cur_thickness)
            cur_thickness = 1

        it += 1

    return staff_lines_thicknesses, staff_lines

def remove_single_line(line_thickness, line_start, in_img, width):
    line_end = line_start + line_thickness - 1

    for col in range(width):
        if in_img.item(line_start, col) == 0 or in_img.item(line_end, col) == 0:
            if in_img.item(line_start - 1, col) == 255 and in_img.item(line_end + 1, col) == 255:
                for j in range(line_thickness):
                    in_img.itemset((line_start + j, col), 255)
            elif in_img.item(line_start - 1, col) == 255 and in_img.item(line_end + 1, col) == 0:
                if (col > 0 and in_img.item(line_end + 1, col - 1) != 0) and (col < width - 1 and in_img.item(line_end + 1, col + 1) != 0):
                    thick = line_thickness + 1
                    if thick < 1:
                        thick = 1
                    for j in range(int(thick)):
                        in_img.itemset((line_start + j, col), 255)
            elif in_img.item(line_start - 1, col) == 0 and in_img.item(line_end + 1, col) == 255:
                if (col > 0 and in_img.item(line_start - 1, col - 1) != 0) and (col < width - 1 and in_img.item(line_start - 1, col + 1) != 0):
                    thick = line_thickness + 1
                    if thick < 1:
                        thick = 1
                    for j in range(int(thick)):
                        in_img.itemset((line_end - j, col), 255)
    return in_img

def remove_staff_lines(in_img, width, staff_lines, staff_lines_thicknesses):
    it = 0
    while it < len(staff_lines):
        line_start = staff_lines[it]
        line_thickness = staff_lines_thicknesses[it]
        in_img = remove_single_line(line_thickness, line_start, in_img, width)
        it += 1
    return in_img

def preprocess(img):
    height = img.shape[0]
    width = img.shape[1]
    staff_lines_thicknesses, staff_lines = get_staff_lines(width, height, img)
    img = remove_staff_lines(img, width, staff_lines, staff_lines_thicknesses)
    return img
