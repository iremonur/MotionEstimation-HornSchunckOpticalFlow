
from asyncio.streams import FlowControlMixin
from distutils.command.build_ext import build_ext
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
import os
import glob
import natsort
from argparse import ArgumentParser
from flow_interpolation import warp_flow


class OpticalClass():
    def __init__(self, H, W, smooth_parameter, iter_num):
        self.alpha     = smooth_parameter
        self.max_iter  = iter_num
        self.u         = np.zeros((H,W))
        self.v         = np.zeros((H,W))
        self.scale     = 3
        self.GaussKernel = (15,15)
        self.kernelX   = 0.25 * np.array([[-1,1],[-1,1]])
        self.kernelY   = 0.25 * np.array([[-1,-1],[1,1]])
        self.kernelt   = 0.25 * np.array([[1,1],[1,1]])
        self.kernelLap = np.array([[1 / 12, 1 / 6, 1 / 12],
                                [1 / 6, 0, 1 / 6],
                                [1 / 12, 1 / 6, 1 / 12]], float)

    def compute_grads(self, prev_frame, next_frame):
        Ix = convolve2d(prev_frame,self.kernelX,mode='same') + \
            convolve2d(next_frame,self.kernelX,mode='same')
        Iy = convolve2d(prev_frame,self.kernelY,mode='same') + \
            convolve2d(next_frame,self.kernelY,mode='same')
        It = - convolve2d(prev_frame,self.kernelt,mode='same') + \
            convolve2d(next_frame,self.kernelt,mode='same')
        return Ix, Iy, It

    def compute_flow(self, prev_frame, next_frame):
        prev_img = cv2.imread(prev_frame, cv2.IMREAD_GRAYSCALE).astype(float)
        next_img = cv2.imread(next_frame, cv2.IMREAD_GRAYSCALE).astype(float)
        if prev_img  is None:
            raise NameError("Can't find image: \"" + prev_frame + '\"')
        elif next_img is None:
            raise NameError("Can't find image: \"" + next_frame + '\"')
        prev_img = cv2.GaussianBlur(prev_img ,self.GaussKernel,0)
        next_img = cv2.GaussianBlur(prev_img ,self.GaussKernel,0)
        Ix, Iy, It  = self.compute_grads(prev_img, next_img)
        I = [Ix, Iy, It]
        iter_num    = 0
        errors      = []
        while True:
            iter_num += 1
            u_avg = convolve2d(self.u,self.kernelLap,mode='same')
            v_avg = convolve2d(self.v,self.kernelLap,mode='same')
            p = Ix * u_avg + Iy * v_avg + It
            d = 4 * self.alpha**2 + Ix**2 + Iy**2
            U = u_avg - Ix * (p / d)
            V = v_avg - Iy * (p / d)
            err = self.compute_err(U)
            errors.append(err)
            self.u = U
            if  iter_num == self.max_iter:
                break
        return U,V, errors, I

    def compute_err(self, new_u):
        return np.linalg.norm(new_u - self.u, 2)

    def plot_flow(self, U, V, prev_frame, idx):

        prev_img = cv2.imread(prev_frame, cv2.IMREAD_GRAYSCALE).astype(float)
        ax = plt.figure().gca()
        ax.imshow(prev_img, cmap = 'gray')
        magnitudeAvg = self.get_magnitude(U,V)

        for i in range(0, U.shape[0], 8):
            for j in range(0, U.shape[1],8):
                dy = V[i,j] * self.scale
                dx = U[i,j] * self.scale
                magnitude = (dx**2 + dy**2)**0.5
                #draw only significant changes
                if magnitude > magnitudeAvg:
                    ax.arrow(j,i, dx, dy, color = 'red')
        plt.savefig(os.path.join(RESULT_DIR, 'frame_{}.png'. format(idx)))

    def get_magnitude(self, U, V):
        sum = 0.0
        counter = 0.0

        for i in range(0, U.shape[0], 8):
            for j in range(0, U.shape[1],8):
                counter += 1
                dy = V[i,j] * self.scale
                dx = U[i,j] * self.scale
                magnitude = (dx**2 + dy**2)**0.5
                sum += magnitude

        mag_avg = sum / counter
        return mag_avg

    def estimate_frame(self):
        return 0



if __name__ == '__main__':

    SRC_DIR   = os.getcwd()
    ROOT_DIR = os.path.join(SRC_DIR, '..')
    FRAMES_DIR = os.path.join(ROOT_DIR, 'frames')
    RESULT_DIR = os.path.join(ROOT_DIR, 'results') # results dir (optical flow vectors)
    EST_DIR    = os.path.join(ROOT_DIR, 'estimated') # estimated dir
    parser = ArgumentParser(description = 'Horn Schunck Optical Flow Estimation')
    parser.add_argument('video_path', type = str, help = 'Vide path (include format)')
    args = parser.parse_args()

    vidcap = cv2.VideoCapture(args.video_path)
    success,image = vidcap.read()
    H,W,C = image.shape
    count = 0
    while success:
        cv2.imwrite(os.path.join(FRAMES_DIR, 'frame_{}.png'. format(count, image)), image)    # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1
    
    flows = []
    derivs = []
    frame_list = glob.glob(os.path.join(FRAMES_DIR, '*.png'))
    frame_list = natsort.natsorted(frame_list)
    for idx, frame in enumerate(frame_list):
        #if idx < len(frame_list)-1: # Comment out to reach results for all the frames
        if idx < 5: # Comment out reach results for the first 4 frames
            print(idx)
            flow_obj = OpticalClass(H,W,smooth_parameter=1, iter_num=25)
            U, V, errors, I = flow_obj.compute_flow(frame, frame_list[idx+1])
            flow = [U,V]
            flows.append(flow)
            derivs.append(I)
            flow_obj.plot_flow(U,V, frame, idx)
            if len(flows) == 2:
                warp_flow(frame, frame_list[idx+1], flows[0], derivs[0], flows[1], derivs[1], idx, EST_DIR)
                flows.pop(0)
                derivs.pop(0)

    
    #To plot the error
    """
    x = np.arange(1,26) 
    plt.title("Distribution of error") 
    plt.xlabel("Iteration") 
    plt.ylabel("Error") 
    plt.plot(x,errors) 
    plt.savefig("erros_0.png")
    plt.show()
    """
    
    