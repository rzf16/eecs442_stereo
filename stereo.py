import cv2
import numpy as np
from scipy.ndimage import maximum_filter
import argparse
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

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

    def get_depth_map(self, img1, img2, canny_thresholds=(50,150), post_processing_size=(11,11)):
        '''
        Computes a depth map given stereo images

        @param ndarray (H,W) img1: rectified left stereo image
        @param ndarray (H,W) img2: rectified right stereo image
        @optional tuple (2) canny_thresholds: thresholds for Canny edge detection
        @optional tuple (2) post_processing_size: matrix size for post-processing step; should be square
        @return ndarray (H,W): depth map
        '''
        assert canny_thresholds[0] < canny_thresholds[1]

        # Canny edge detection
        start = time.time()
        edges1 = cv2.Canny(img1, canny_thresholds[0], canny_thresholds[1])
        edges2 = cv2.Canny(img2, canny_thresholds[0], canny_thresholds[1])
        print("{} seconds for Canny".format(time.time() - start))

        # Post-processing around detected edges
        if post_processing_size is not None:
            assert post_processing_size[0] == post_processing_size[1]
            start = time.time()
            edges1 = maximum_filter(edges1, size=post_processing_size)
            edges2 = maximum_filter(edges2, size=post_processing_size)
            print("{} seconds for post-processing".format(time.time() - start))

        if self.verbose:
            print("{} feature points detected in img1".format(np.count_nonzero(edges1)))
            print("{} feature points detected in img2".format(np.count_nonzero(edges2)))

        start = time.time()
        matches = self.get_correspondences(img1, img2, edges1, edges2)
        print("{} seconds for feature matching".format(time.time() - start))

        if self.verbose:
            print("{} feature matches made".format(matches.shape[0]))

        if self.debug:
            img_spacer = (np.ones((img1.shape[0], 20, 3)) * 255).astype(np.uint8)
            edge_spacer = (np.ones((img1.shape[0], 20)) * 255).astype(np.uint8)
            side_by_side_img = np.hstack((img1, img_spacer, img2))
            side_by_side_edges = np.hstack((edges1, edge_spacer, edges2))
            side_by_side_edges = np.stack((side_by_side_edges,)*3, axis=-1)
            display_offset = np.array([0, img1.shape[1] + img_spacer.shape[1]])
            it = 0
            # TODO: replace with random sampling
            # for match in matches:
            #     if it % 250 == 0:
            #         pt1 = tuple(match[0])[::-1]
            #         pt2 = tuple((match[1] + display_offset))[::-1]
            #         cv2.line(side_by_side_img, pt1, pt2, (0, 255, 0), thickness=2)
            #         cv2.line(side_by_side_edges, pt1, pt2, (0, 255, 0), thickness=2)
            #     it += 1
            # cv2.imshow("Matches", side_by_side_edges)
            # cv2.waitKey(0)
        
        disparity_map = np.zeros(img1.shape[0:2], dtype=np.uint16)
        for match in matches:
            disparity_map[match[0,0],match[0,1]] = np.abs(match[0,1] - match[1,1])
        cv2.imshow("Disparity", cv2.convertScaleAbs(disparity_map))
        cv2.waitKey(0)

        depth_map = np.zeros(img1.shape[0:2])
        depth_map[disparity_map > 0] = self.baseline_cm * self.focal_length_px / disparity_map[disparity_map > 0]

        vis_map = np.zeros(img1.shape, dtype=np.uint8)
        vis_map[:,:,2] = (255 * depth_map / self.baseline_cm * self.focal_length_px).astype(np.uint8)
        vis_map[vis_map != 0] = 255 - vis_map[vis_map != 0]
        cv2.imshow("Depth", vis_map)
        cv2.waitKey(0)

    def get_correspondences(self, img1, img2, features1, features2, search_rows=5, matrix_size=(3,3), disp_limit=150):
        '''
        Estimates correspondences for two feature images

        @param ndarray (H,W) img1: rectified left stereo image
        @param ndarray (H,W) img2: rectified right stereo image
        @param ndarray (H,W) features1: left feature image with 0 for no feature and 255 for feature
        @param ndarray (H,W) features2: right feature image with 0 for no feature and 255 for feature
        @optional int search_rows: number of rows to search for each feature point
        @optional tuple (2) matrix_size: size of matrix used to compute cost for each feature point; should be square
        @return ndarray (N,2,2): N feature matches; dim0 represents the match, dim1 represents the image,
                                 dim2 represents the coordinates
        '''
        assert matrix_size[0] == matrix_size[1]

        pts1 = np.array(np.where(features1[:,:])).T
        # pts2 = np.array(np.where(np.logical_or(features2[:,:]==0, features2[:,:]))).T
        pts2 = np.array(np.where( features2[:,:])).T

        # Do not use feature points near image edges
        matrix_offset = int((matrix_size[0]-1) / 2)
        pts1 = self.remove_edge_pts(pts1, img1.shape[1], img1.shape[0], matrix_offset)
        pts2 = self.remove_edge_pts(pts2, img2.shape[1], img2.shape[0], matrix_offset)

        matches = np.zeros((pts1.shape[0], 2, 2), dtype=np.int32)
        search_offset = int((search_rows-1) / 2)
        for i, pt1 in enumerate(pts1):
            matches[i,0] = pt1
            search_mask = np.logical_and(pts2[:,0] == pt1[0], pts2[:,1] >= pt1[1] - 150)
            search_mask = np.logical_and(search_mask, pts2[:,1] <= pt1[1])
            if not search_mask.any():
                matches[i,1] = np.array([-1,-1])
            else:
                matched_pt = self.find_best_match(pt1, pts2[search_mask], img1, img2, matrix_size)
                matches[i,1] = matched_pt

        return matches[matches[:,1,0] != -1]

    def find_best_match(self, pt1, search_pts, img1, img2, matrix_size):
        '''
        Finds the best match for pt1 in the search space search_pts

        @param ndarray (2,) pt1: point in features1 for which to find a match
        @param ndarray (N,2) search_pts: N points in features2 which can be matched with pt1
        @param ndarray (H,W) img1: rectified left stereo image
        @param ndarray (H,W) img2: rectified right stereo image
        @optional tuple (2) matrix_size: size of matrix used to compute cost for each feature point; should be square
        '''
        assert matrix_size[0] == matrix_size[1]

        offset = int((matrix_size[0]-1) / 2)
        y1, x1 = pt1

        best_pt = None
        best_mad = matrix_size[0] * matrix_size[1] * 255 * 3

        for pt2 in search_pts:
            m1 = img1[y1-offset:y1+offset+1, x1-offset:x1+offset+1]
            m2 = img2[pt2[0]-offset:pt2[0]+offset+1, pt2[1]-offset:pt2[1]+offset+1]
            mad = cv2.absdiff(m1, m2).max()
            if mad < best_mad:
                best_pt = pt2
                best_mad = mad
            if mad == 0:
                break

        return best_pt

    def remove_edge_pts(self, pts, x_lim, y_lim, offset):
        '''
        Removes points with offset of the limits [0,x_lim] and [0,y_lim]

        @param ndarray (N,2) pts: points to filter
        @param int x_lim: limit in x
        @param int y_lim: limit in y
        @param int offset: distance from limits to filter out
        @return ndarray (M,2): filtered points
        '''
        left_mask = pts[:,1] >= offset
        right_mask = pts[:,1] < x_lim - offset
        top_mask = pts[:,0] >= offset
        bot_mask = pts[:,0] < y_lim - offset
        hori_mask = np.logical_and(left_mask, right_mask)
        vert_mask = np.logical_and(top_mask, bot_mask)
        mask = np.logical_and(hori_mask, vert_mask)
        return pts[mask]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo Vision (rzfeng, dleroys)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print status messages")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug messages and images; includes verbose")
    parser.add_argument("image", metavar="image", type=str, nargs=2, help="stereo images")
    parser.add_argument("calibration", metavar="calibration", type=str, help="camera calibration json")
    opts = parser.parse_args()

    calib = json.load(open(opts.calibration))

    stereo = Stereo(calib["baseline_cm"], calib["focal_length_px"], opts.verbose, opts.debug)

    img1 = cv2.imread(opts.image[0])
    img2 = cv2.imread(opts.image[1])

    stereo.get_depth_map(img1, img2, canny_thresholds=(100,200), post_processing_size=(3,3))
