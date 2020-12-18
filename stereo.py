import cv2
import numpy as np
from scipy.ndimage import maximum_filter
import argparse
import time
import json
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

class Stereo:
    '''
    Class for computing depth maps using stereo vision

    @attribute float baseline_cm: the distance between the two camera centers in cm
    @attribute float focal_length_px: the focal length of the cameras in pixels
    @attribute bool verbose: flag for status messages
    @attribute bool debug: flag for debug outputs
    '''

    def __init__(self, baseline_cm, focal_length_px, verbose=False, debug=False):
        self.baseline_cm = baseline_cm
        self.focal_length_px = focal_length_px
        self.verbose = True if debug else verbose
        self.debug = debug

    def get_depth_map(self, img1, img2, canny_thresholds=(100,200), post_processing_size=3, knn_threshold=25.0):
        '''
        Computes a depth map given stereo images

        @param ndarray (H,W) img1: rectified left stereo image
        @param ndarray (H,W) img2: rectified right stereo image
        @optional int post_processing_size: square matrix size for post-processing step
        @return ndarray (H,W): depth map
        '''
        start = time.time()

        # Get features
        orb = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)
        edges1 = cv2.Canny(img1, canny_thresholds[0], canny_thresholds[1])
        edges2 = cv2.Canny(img2, canny_thresholds[0], canny_thresholds[1])
        edges1 = maximum_filter(edges1, size=(post_processing_size,post_processing_size))
        edges2 = maximum_filter(edges2, size=(post_processing_size,post_processing_size))
        edges1_pts = np.array(np.where(edges1[:,:])).T
        edges2_pts = np.array(np.where(edges2[:,:])).T
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)

        # Match features
        match_start = time.time()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)
        if self.verbose:
            print("{} seconds for feature matching".format(time.time() - match_start))
            print("{} matches made".format(len(matches)))
        if self.debug:
            match_img = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50], None, flags=2)
            cv2.imshow("Matches",match_img)
            #cv2.waitKey(0)

        # Post-processing
        post_processing_start = time.time()
        matched_kp1 = np.zeros((len(matches),2), dtype=np.int16)
        matched_kp2 = np.zeros((len(matches),2), dtype=np.int16)
        for i, match in enumerate(matches):
            matched_kp1[i,:] = np.array(kp1[match.queryIdx].pt)
            matched_kp2[i,:] = np.array(kp2[match.trainIdx].pt)
        matched_kp1 = matched_kp1[1:]
        matched_kp2 = matched_kp2[1:]
        if self.verbose:
            print("{} seconds for post-processing".format(time.time() - post_processing_start))

        # Compute disparity map
        disparity_start = time.time()
        disparities = np.linalg.norm(matched_kp1-matched_kp2, axis=1)
        disparity_map = np.zeros(img1.shape[0:2], dtype=np.uint16)
        for i, kp in enumerate(matched_kp1):
            disparity_map[kp[1],kp[0]] = disparities[i]
        knn_start = time.time()
        tree = cKDTree(matched_kp1[:,[1,0]])

        canny_nn = tree.query(edges1_pts, distance_upper_bound=knn_threshold)[1]
        valid_nn = canny_nn < tree.n
        canny_disparities = disparities[canny_nn[valid_nn]]
        for i, canny_pt in enumerate(edges1_pts[valid_nn]):
            disparity_map[canny_pt[0],canny_pt[1]] = canny_disparities[i]

        if self.verbose:
            print("{} seconds for KNN".format(time.time() - knn_start))
            print("{} seconds for disparity calculation".format(time.time() - disparity_start))
        if self.debug:
            cv2.imshow("Disparity", cv2.convertScaleAbs(disparity_map))
            #cv2.waitKey(0)

        # Filters
        # gaussian_start = time.time()
        # disparity_map = maximum_filter(disparity_map, size=(7,7))
        # disparity_map = cv2.GaussianBlur(disparity_map, (11,11), 6)
        # print(f"{time.time() - gaussian_start} seconds for filters")

        if self.verbose:
            print("{} seconds for disparity calculation".format(time.time() - disparity_start))
        if self.debug:
            cv2.imshow("Disparity", cv2.convertScaleAbs(disparity_map))
            #cv2.waitKey(0)

        # Compute depth map
        depth_start = time.time()
        depth_map = np.zeros(img1.shape[0:2])
        depth_map[disparity_map > 0] = self.baseline_cm * self.focal_length_px / disparity_map[disparity_map > 0]

        if self.verbose:
            print("{} seconds for depth calculation".format(time.time() - depth_start))
        if self.debug:
            vis_map = np.zeros(img1.shape, dtype=np.uint8)
            print(depth_map.max())
            vis_map[:,:,1] = depth_map != 0
            vis_map[vis_map != 0] *= (255 * depth_map[np.nonzero(depth_map)].min() / depth_map[depth_map != 0]).astype(np.uint8)
            cv2.imshow("Depth", vis_map)
            #cv2.waitKey(0)
        print("{} seconds total".format(time.time() - start))
        return depth_map


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Stereo Vision (rzfeng, dleroys)")
    # parser.add_argument("-v", "--verbose", action="store_true", help="print status messages")
    # parser.add_argument("-d", "--debug", action="store_true", help="print debug messages and images; includes verbose")
    # parser.add_argument("image", metavar="image", type=str, nargs=2, help="stereo images")
    # parser.add_argument("calibration", metavar="calibration", type=str, help="camera calibration json")
    # opts = parser.parse_args()

    calib = json.load(open('calibration.json'))

    stereo = Stereo(calib["baseline_cm"], calib["focal_length_px"], True, True)

    cap1 = cv2.VideoCapture('http://rpi0.local:8080/stream/video.mjpeg')
    cap2 = cv2.VideoCapture('http://rpi1.local:8080/stream/video.mjpeg')

    while True:
        _, frame1 = cap1.read()
        _, frame2 = cap2.read()

    # img1 = cv2.imread(opts.image[0])
    # img2 = cv2.imread(opts.image[1])
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
        stereo.get_depth_map(frame1, frame2)