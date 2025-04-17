import cv2
import numpy as np
import copy

class image_dehazer:
    def __init__(self, airlightEstimation_windowSze=15, boundaryConstraint_windowSze=3, C0=20, C1=300,
                 regularize_lambda=0.1, sigma=0.5, delta=0.85, showHazeTransmissionMap=True):
        self.airlightEstimation_windowSze = airlightEstimation_windowSze
        self.boundaryConstraint_windowSze = boundaryConstraint_windowSze
        self.C0 = C0
        self.C1 = C1
        self.regularize_lambda = regularize_lambda
        self.sigma = sigma
        self.delta = delta
        self.showHazeTransmissionMap = showHazeTransmissionMap
        self._A = []
        self._Transmission = None

    def __AirlightEstimation(self, HazeImg):
        self._A = []
        if len(HazeImg.shape) == 3:
            for ch in range(HazeImg.shape[2]):
                kernel = np.ones((self.airlightEstimation_windowSze, self.airlightEstimation_windowSze), np.uint8)
                minImg = cv2.erode(HazeImg[:, :, ch], kernel)
                self._A.append(int(minImg.max()))
        else:
            kernel = np.ones((self.airlightEstimation_windowSze, self.airlightEstimation_windowSze), np.uint8)
            minImg = cv2.erode(HazeImg, kernel)
            self._A.append(int(minImg.max()))

    def __BoundCon(self, HazeImg):
        if len(HazeImg.shape) == 3:
            t_b = np.maximum((self._A[0] - HazeImg[:, :, 0].astype(float)) / (self._A[0] - self.C0),
                             (HazeImg[:, :, 0].astype(float) - self._A[0]) / (self.C1 - self._A[0]))
            t_g = np.maximum((self._A[1] - HazeImg[:, :, 1].astype(float)) / (self._A[1] - self.C0),
                             (HazeImg[:, :, 1].astype(float) - self._A[1]) / (self.C1 - self._A[1]))
            t_r = np.maximum((self._A[2] - HazeImg[:, :, 2].astype(float)) / (self._A[2] - self.C0),
                             (HazeImg[:, :, 2].astype(float) - self._A[2]) / (self.C1 - self._A[2]))

            MaxVal = np.maximum(t_b, t_g, t_r)
            self._Transmission = np.minimum(MaxVal, 1)
        else:
            self._Transmission = np.maximum((self._A[0] - HazeImg.astype(float)) / (self._A[0] - self.C0),
                                            (HazeImg.astype(float) - self._A[0]) / (self.C1 - self._A[0]))
            self._Transmission = np.minimum(self._Transmission, 1)

    def __removeHaze(self, HazeImg):
        epsilon = 0.0001
        Transmission = pow(np.maximum(self._Transmission, epsilon), self.delta)
        HazeCorrectedImage = copy.deepcopy(HazeImg)
        if len(HazeImg.shape) == 3:
            for ch in range(HazeImg.shape[2]):
                temp = ((HazeImg[:, :, ch].astype(float) - self._A[ch]) / Transmission) + self._A[ch]
                HazeCorrectedImage[:, :, ch] = np.clip(temp, 0, 255)
        else:
            temp = ((HazeImg.astype(float) - self._A[0]) / Transmission) + self._A[0]
            HazeCorrectedImage = np.clip(temp, 0, 255)
        return HazeCorrectedImage.astype(np.uint8)

    def remove_haze(self, HazeImg):
        if HazeImg is None:
            raise ValueError("Input image is empty. Please check the file path.")
        self.__AirlightEstimation(HazeImg)
        self.__BoundCon(HazeImg)
        haze_corrected_img = self.__removeHaze(HazeImg)
        return haze_corrected_img, self._Transmission
