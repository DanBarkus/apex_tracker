import csv
import re
import os
import cv2 as cv
import numpy as np
import pytesseract as pt
from matplotlib import pyplot as plt

pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
dirname = os.path.dirname(__file__)
frame = 0

# Number of clip to pick up
clip_nbr = 2
# Creates folder for resulting files generated from clip
folder = os.path.join(dirname, 'test_clip_' + str(clip_nbr))
if not os.path.exists(folder):
    os.mkdir(folder)

with open('test_clip_' + str(clip_nbr) + '\\loc_output_' + str(clip_nbr) + '.csv', mode='w') as output_csv:
    # CSV setup
    csv_writer = csv.writer(output_csv, delimiter='|', quoting=csv.QUOTE_ALL, lineterminator='\n')
    csv_writer.writerow(['Frame', 'Center', 'Confidence', 'Curr Ammo', 'Spare Ammo'])

    # Load gun sheet, convert to B+W, apply threshold
    bw_template = cv.imread('template_guns.png')
    bw_template = cv.cvtColor(bw_template, cv.COLOR_BGR2GRAY)
    res, bw_template = cv.threshold(bw_template, 200, 255,cv.THRESH_BINARY)

    size = bw_template.shape[::-1]
    crop_size = (410,148*2)

    # Set up output video streams for generated files
    vid_out = cv.VideoWriter('test_clip_' + str(clip_nbr) + '\\weapons_' + str(clip_nbr) + '.avi', cv.VideoWriter_fourcc(*'MJPG'), 30, size)
    thresh_out = cv.VideoWriter('test_clip_' + str(clip_nbr) + '\\crop_thresh_' + str(clip_nbr) + '.avi', cv.VideoWriter_fourcc(*'MJPG'), 30, crop_size)

    # Load in clip to analyze
    cap = cv.VideoCapture('test_clips\\test_clip_' + str(clip_nbr) + '.mp4')
    vid_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    while(cap.isOpened()):
        ret, in_img = cap.read()

        if ret == True:
            # Crops for different regions of the screen
            weapon_crop = in_img[1910:2058, 3040:3450]
            curr_ammo_crop = in_img[1920:2000, 3450:3565]
            spare_ammo_crop = in_img[2000:2050, 3450:3565]

            # Image Transformations

            # weapon
            img_gray = cv.cvtColor(weapon_crop, cv.COLOR_BGR2GRAY)
            w, h = img_gray.shape[::-1]
            # current ammo
            curr_ammo_crop = cv.cvtColor(curr_ammo_crop, cv.COLOR_BGR2GRAY)
            ret, curr_ammo_crop = cv.threshold(curr_ammo_crop, 200, 255,cv.THRESH_BINARY)
            curr_ammo_crop = 255 - curr_ammo_crop
            # spare ammo
            spare_ammo_crop = cv.cvtColor(spare_ammo_crop, cv.COLOR_BGR2GRAY)
            ret, spare_ammo_crop = cv.threshold(spare_ammo_crop, 100, 255,cv.THRESH_BINARY)
            spare_ammo_crop = 255 - spare_ammo_crop

            # cv.imwrite("test_clip_" + str(clip_nbr) + "\\ammo_images\\current_ammo_" + str(frame).zfill(4) + ".png", curr_ammo_crop)
            # cv.imwrite("test_clip_" + str(clip_nbr) + "\\ammo_images\\spare_ammo_" + str(frame).zfill(4) + ".png", spare_ammo_crop)

            # get ammo count in number
            ammo = (pt.image_to_string(curr_ammo_crop, lang='Apex_Ammo', config='--psm 7'))
            ammo = re.sub("[^0-9]", "", ammo)
            # get spare ammo count
            spare_ammo = (pt.image_to_string(spare_ammo_crop, lang='Apex_Ammo', config='--psm 7'))
            spare_ammo = re.sub("[^0-9]", "", spare_ammo)

            # print(spare_ammo)

            # weapon sheet for matching against
            template = bw_template
            ret, thresh = cv.threshold(img_gray, 200, 255,cv.THRESH_BINARY)
            # thresh = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

            # apply template matching to find selected weapon
            res = cv.matchTemplate(bw_template,thresh,cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

            # print(max_val)

            # calculate positions for drawing box
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            center = (top_left[0] + w/2, top_left[1] + h/2)

            # convert back to 'color' so we can add colored rectangle
            template = cv.cvtColor(template, cv.COLOR_GRAY2BGR)

            # decide if we're confident enough to draw a box
            if max_val > 0.6 and max_val < 1.0:
                tmplte_out = cv.rectangle(template, top_left, bottom_right, (0,0,255), 2)

            thresh_img = np.zeros((h*2, w), dtype="uint8")
            thresh_img[0:h, 0:w] = img_gray
            thresh_img[h:crop_size[1], 0:w] = thresh
            thresh_img= cv.cvtColor(thresh_img, cv.COLOR_GRAY2BGR)
            thresh_out.write(thresh_img)

            vid_out.write(template)
            csv_writer.writerow([frame, center, max_val, ammo, spare_ammo])
            frame+=1
            print("{:.2%}".format(frame / vid_frames))

        else:
            break

    vid_out.release()
    thresh_out.release()
output_csv.close()