import cv2
import time
import argparse
import numpy as np
from scipy.signal import convolve2d as convolution

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="input image path", required=True)
    parser.add_argument("-o", "--output", type=str, help="output image path", required=True)
    parser.add_argument("-s", "--shape", type=int, nargs=2, help="output image shape", default=None)
    parser.add_argument("-m", "--max_iter", type=int, help="maximum iteration for initial prototype binary pattern generation", default=100000)
    parser.add_argument("-p", "--prototype", type=str, help="path to initial prototype binary pattern", default=None)
    parser.add_argument("-d", "--dither", type=str, help="path to dither matirx", default=None)
    args_ = parser.parse_args()
    return args_

class clock():
    def tick(self):
        self.tick = time.time()

    def tock(self):
        self.tock = time.time()

    def get_time(self):
        return self.tock - self.tick

def read_image(path, resize=None):
    if not path: return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if resize: img = cv2.resize(img, (resize[1], resize[0]))
    img = img / 255.
    return img

def save_image(path, img):
    if np.max(img) <= 1.0: img = img * 255.
    cv2.imwrite(path, img)

def show_image(img, title=None):
    plt.figure()
    if title: plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()

class void_and_cluster():
    def __init__(self, init=None, shape=None, kernel=None, max_iter=100000, prototype=None, dither_matrix=None, verbose=1):
        timer = clock()
        timer.tick()
        self.init = self._get_random_pattern(init, shape)
        assert(np.mean(self.init) < 0.5), "ERROR: Invalid initial binary pattern"
        self.shape = self._get_shape(shape)
        self.ones = np.sum(self.init).astype(int)
        self.size = self.shape[0] * self.shape[1]
        self.kernel = self._get_kernel(kernel)
        self.kernel_size = self.kernel.shape
        self.cluster_score = self._get_score(self.init, 'cluster')
        self.void_score = self._get_score(self.init, 'void')
        self.max_iter = max_iter
        self.prototype = prototype
        self.dither_matrix = dither_matrix
        self.verbose = verbose
        timer.tock()
        if self.verbose: print("INFO: VAC initialization done, time: %.5fs" % timer.get_time())

    def _flip01(self, arr):
        return np.logical_not(arr)

    def _gaussian_kernel(self, kernel_size=9, sigma=1.5, mean=0.0):
        ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        return kernel / np.sum(kernel)

    def _get_random_pattern(self, init, shape):
        if init is not None: return init
        if shape is not None: RP = np.random.randint(0, 2, size=shape, dtype=int)
        else: 
            RP = np.random.random(size=test_shape)
            threshold = 0.8
            RP[RP >= threshold] = 1.0
            RP[RP < threshold] = 0.0
        if np.mean(RP) > 0.5: RP = self._flip01(RP)
        return RP

    def _get_kernel(self, kernel):
        if kernel is not None: return kernel
        return self._gaussian_kernel(9, sigma=1.5, mean=0.0)

    def _get_shape(self, shape):
        if shape is not None: return shape
        return self.init.shape

    def _get_score(self, pattern, target):
        if target == 'void': pattern = self._flip01(pattern)
        score = convolution(pattern, self.kernel, mode='same', boundary='wrap')
        score[pattern == 0.0] = 0.0
        return score

    def _find_void_cluster(self, target):
        if target == 'cluster': max_index = np.argmax(self.cluster_score)
        if target == 'void': max_index = np.argmax(self.void_score)
        return np.unravel_index(max_index, self.shape)

    def _update_void_cluster_score(self, prototype, pos, target=None):
        x, y = pos
        kx_, ky_ = self.kernel_size
        kx = kx_ // 2
        ky = ky_ // 2
        pad_prototype = np.pad(prototype, (2*kx, 2*ky), mode='wrap')
        px_min, px_max = x, x + 4*kx + 1
        py_min, py_max = y, y + 4*ky + 1
        patch = pad_prototype[px_min:px_max, py_min:py_max]
        offset_x, offset_y = (x - kx) % self.shape[0], (y - ky) % self.shape[1]
        
        # Cluster score update
        if target is None or target == 'cluster':
            cluster_patch = patch
            cluster_patch_score = convolution(cluster_patch, self.kernel, mode='valid')
            cluster_patch_score[cluster_patch[kx:-kx, ky:-ky] == 0.0] = 0.0
            # First roll the score matrix to move the update patch to (0, 0)
            # purpose for this is to avoid overflow wrap around problems
            cluster_shift_score = np.roll(np.roll(self.cluster_score, -offset_x, axis=0), -offset_y, axis=1)
            cluster_shift_score[:kx_, :ky_] = cluster_patch_score
            self.cluster_score = np.roll(np.roll(cluster_shift_score, offset_x, axis=0), offset_y, axis=1)
        
        # Void score update
        if target is None or target == 'void':
            void_patch = self._flip01(patch)
            void_patch_score = convolution(void_patch, self.kernel, mode='valid')
            void_patch_score[void_patch[kx:-kx, ky:-ky] == 0.0] = 0.0
            # First roll the score matrix to move the update patch to (0, 0)
            # purpose for this is to avoid overflow wrap around problems
            void_shift_score = np.roll(np.roll(self.void_score, -offset_x, axis=0), -offset_y, axis=1)
            void_shift_score[:kx_, :ky_] = void_patch_score
            self.void_score = np.roll(np.roll(void_shift_score, offset_x, axis=0), offset_y, axis=1)

    def run_operation_1(self):
        # initial prototype binary pattern generation
        timer = clock()
        timer.tick()
        prototype = self.init.copy()
        for i in range(self.max_iter):
            # Find tightest cluster
            cluster_pos = self._find_void_cluster('cluster')
            prototype[cluster_pos] = 0.0
            self._update_void_cluster_score(prototype, cluster_pos)
            
            # Find largest void
            void_pos = self._find_void_cluster('void')
            prototype[void_pos] = 1.0
            self._update_void_cluster_score(prototype, void_pos)
            if cluster_pos == void_pos: break
        self.prototype = prototype
        timer.tock()
        if self.verbose: print("INFO: VAC operation 1 done, time: %.5fs" % timer.get_time())

    def run_operation_2(self):
        # dither matrix generation
        timer = clock()
        timer.tick()
        dither_matrix = np.zeros(self.shape, dtype=float)
        
        # Phase I
        pattern = self.prototype.copy()
        self.cluster_score = self._get_score(pattern, 'cluster')
        for rank in reversed(range(self.ones)):
            cluster_pos = self._find_void_cluster('cluster')
            pattern[cluster_pos] = 0
            self._update_void_cluster_score(pattern, cluster_pos, 'cluster')
            dither_matrix[cluster_pos] = rank
            
        # Phase II
        pattern = self.prototype.copy()
        self.void_score = self._get_score(pattern, 'void')
        for rank in range(self.ones, self.size):
            void_pos = self._find_void_cluster('void')
            pattern[void_pos] = 1
            self._update_void_cluster_score(pattern, void_pos, 'void')
            dither_matrix[void_pos] = rank
            
        # Normalize
        dither_matrix_norm = dither_matrix / self.size
        
        self.dither_matrix = dither_matrix_norm
        timer.tock()
        if self.verbose: print("INFO: VAC operation 2 done, time: %.5fs" % timer.get_time())

    def run(self):
        self.run_operation_1()
        self.run_operation_2()
        
    def halftone(self, image):
        image = cv2.resize(image, (self.shape[1], self.shape[0]))
        halftone_img = np.zeros(image.shape)
        halftone_img[image > self.dither_matrix] = 1.0
        return halftone_img

if __name__ == "__main__":
    args = get_args()

    # Load image
    img = read_image(args.input)
    prototype = read_image(args.prototype)
    dither_matrix = read_image(args.dither)

    # Run void-and-cluster algorithm
    vac = void_and_cluster(shape=args.shape, max_iter=args.max_iter, prototype=prototype, dither_matrix=dither_matrix)
    if dither_matrix is None:
        # if dither matrix is provided, we could skip generating it and go for halftoning
        if prototype is None:
            # No initial prototype binary pattern and dither matrix provided
            # run from scratch
            vac.run()
            save_image(args.output+".prototype.png", vac.prototype)
            save_image(args.output+".dither.png", vac.dither_matrix)
        else:
            # Initial prototype binary pattern in provided
            # directly run operation 2
            assert(list(args.shape) == list(prototype.shape)), "Given shape doesn't match given prototype shape"

            vac.run_operation_2()
            save_image(args.output+".dither.png", vac.dither_matrix)
    else:
        assert(list(args.shape) == list(dither_matrix.shape)), "Given shape doesn't match given dither matrix shape"

    # Generate halftone image
    halftone_img = vac.halftone(img)

    # Save image
    save_image(args.output, halftone_img)
