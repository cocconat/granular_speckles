# This file belongs to DWGranularSpeckles project.
# The software is realeased with MIT license.
import os
import sys
import subprocess


def get_frame_rate(filename):
    if not os.path.exists(filename):
        sys.stderr.write("ERROR: filename %r was not found!" % (filename,))
        return -1
    out = subprocess.check_output(["ffprobe", filename, "-v", "0", "-select_streams",
                                   "v", "-print_format", "flat", "-show_entries", "stream=r_frame_rate"])
    rate = out.split('=')[1].strip()[1:-1].split('/')
    if len(rate) == 1:
        return float(rate[0])
    if len(rate) == 2:
        return float(rate[0])/float(rate[1])
    return -1


def videoToFrame(args):
    import cv2
    vidcap = cv2.VideoCapture(args.videofile)
    success, image = vidcap.read()
    count = 0
    success = True
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    success, image = vidcap.read()
    while success:
        # print 'Read a new frame: ', success
        cv2.imwrite(args.image_folder+"/frame_%04d.png" %
                    count, image)     # save frame as JPEG file
        count += 1
        success, image = vidcap.read()
    return count
