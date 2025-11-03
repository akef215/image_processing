import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from scipy.signal import convolve2d

class Image:
    """
        Core Algorithms of Image Processing
    """
    # Kernels:   
    # 1D kernels
    kernel_dx_1d = np.array([[-1, 0, 1]]) 
    kernel_dy_1d = kernel_dx_1d.T
    kernel_id_1d = np.ones(shape=(1, 3), dtype=np.uint8)

    # Gaussian kernels ()
    kernel_gauss_1d = np.array([[1, 2, 1]])
    kernel_gauss_2d = kernel_gauss_1d.T @ kernel_gauss_1d/16

    # 2D-gardient kernels
    kernel_dx_2d = np.matmul(kernel_id_1d.T, kernel_dx_1d)
    kernel_dy_2d = np.matmul(kernel_dy_1d, kernel_id_1d)

    # 2D-sobel kernels
    kernel_sobel_x = np.matmul(kernel_gauss_1d.T, kernel_dx_1d)
    kernel_sobel_y = np.matmul(kernel_dy_1d, kernel_gauss_1d)

    # Constructor
    def __init__(self, img_path, img_mat=None):
        if img_mat is None:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError("Check the img_path variable")
        else:
            img = img_mat    
        self.img_ = img

    def show_img(self, title="Original Image", cmap='gray', axis=False):
        plt.title(title)
        if not axis:
            plt.axis('off')
        plt.imshow(self.img_, cmap=cmap)

    def profile(self, offset, axis='H'):
        assert axis == "H" or axis == "V", "axis must be either 'H' or 'V'"
        assert offset >= 0, "The offset must be positive"
        if axis == "H":
            assert offset < self.img_.shape[0], "the offset of a horizental \
                profile can't be greater than the number of rows of the image"
            
            return self.img_[offset, :]
        else:
            assert offset < self.img_.shape[1], "the offset of a vertical \
                profile can't be greater than the number of columns of the image"
            
            return self.img_[:, offset]

    def plot_profile(self, offset, axis='H'):
        profile_ = self.profile(self.img_, offset, axis)
        plt.figure()
        axis_name = "Horizental" if axis=="H" else "Vertical"
        plt.title(axis_name + f" Profile at {offset}")
        plt.ylabel("NdG")
        plt.plot(range(len(profile_)), profile_)

    def projection(self, offset, axis='H'):
        return self.profile(offset, axis).sum()

    def plot_projection(self, axis='H'):
        axe = 0 if axis=='H' else 1
        proj = [self.projection(offset, axis) \
                for offset in range(self.img_.shape[axe])]
        
        plt.figure()
        axis_name = "Horizental" if axis=="H" else "Vertical"
        plt.title(axis_name + f" Projection")
        plt.ylabel("NdG")
        plt.plot(range(len(proj)), proj)   

    def histogram(self, normal=False, cumulative=False):
        img_int = self.img_.astype(int)
        intensities, counts_ = np.unique(img_int, return_counts=True)
        counts = np.zeros(256)
        counts[intensities] = counts_
        if normal:
            counts = counts/img_int.size

        if cumulative:
            counts = np.cumsum(counts)   

        return intensities, counts     

    def plot_hist(self, normal=False, cumulative=False, title="Histogram"):
        _, counts = self.histogram(normal, cumulative)
        plt.title(title), plt.xlabel("NdG"), plt.ylabel("#NdG")
        plt.bar(range(256), counts)          

    def hist_stretch(self, bornes='minmax'):
        assert bornes == "minmax" or bornes == "percentilles", \
            "axis must be either 'minmax' or 'percentilles'"
        
        if bornes == 'minmax':
            inf = self.img_.min()
            sup = self.img_.max()
        else:
            inf = np.percentile(self.img_, 10)
            sup = np.percentile(self.img_, 90)

        # Transform the Look up Table    
        LuT = np.arange(256)
        LuT = np.clip((LuT - inf)/(sup - inf)*255, 0, 255)  
        
        return LuT[self.img_.astype(int)]
    
    def hist_equal(self):
        _, counts = self.histogram(normal=True, cumulative=True)
        return np.round(counts[self.img_]*255)
    
    def gamma_corr(self, gamma):
        return self.img_**(1/gamma)
    
    def otsu(self, verbose=False):
        n = 256
        intra_std = np.ones(n)*np.inf
        for i in range(n):
            class_0 = self.img_ < i
            class_1 = self.img_ >= i

            std_0 = self.img_[class_0].std() if np.any(class_0) else 0
            std_1 = self.img_[class_1].std() if np.any(class_1) else 0

            w1 = class_0.sum()/self.img_.size
            w2 = class_1.sum()/self.img_.size  
            if verbose:
                print(f"Threshold : {i}")
                print(f"std_0 : {std_0:.2f} | std_1 : {std_1:.2f}") 
                print(f"w_1 : {w1:.2f} | w_2 : {w2:.2f}") 
                print(f"intra_std : {w1*std_0 + w2*std_1:.2f}")
                print("_________")
            intra_std[i] = w1*std_0 + w2*std_1

        return intra_std.argmin()
    
    def thresh(self, threshold=None):
        if threshold is None:
            threshold = self.otsu(self.img_)
        else:
            assert 0 <= threshold <= 255\
            , "threshold must be in range(0,256)"    

        binary_img = np.zeros(shape=self.img_.shape, dtype=np.uint8)    
        binary_img[self.img_ >= threshold] = 255
        return binary_img, threshold
    
    def subimages(self, n):
        r = int(np.floor(self.img_.shape[0]/n))
        c = int(np.floor(self.img_.shape[1]/n))
        subimages_ = [] 
        for i in range(n):
            for j in range(n):
                subimages_.append(self.img_[r*i:r*(i+1), c*j:c*(j+1)])
        return subimages_  
    
    @staticmethod
    def assemble_imgs(imgs, n):
        unit_r, unit_c = imgs[0].shape
        image = np.zeros(shape=(unit_r*n, unit_c*n), dtype=np.uint8)
        for i in range(n):
            for j in range(n):
                image[i*unit_r : (i+1)*unit_r, j*unit_c : (j+1)*unit_c] = imgs[i*n+j]
        return image 
    
    def local_thresh(self, n):
        thresholds = []
        thresh_parts = []
        for img in self.subimages(n):
            img_tmp = Image("", img_mat=img)
            img, thresh_ = self.thresh(img_tmp)
            thresh_parts.append(img)
            thresholds.append(int(thresh_))
            #plt.figure(), show_img(img_), plt.show()

        return self.assemble_imgs(thresh_parts, n), np.array(thresholds)
    
    # Linear kernels:
    @staticmethod
    def avg_kernel(n):
        assert n > 0, "n must be greather than 0"
        return np.ones(shape=(n, n))/(n**2)

    @staticmethod
    def gaussian_kernel(n, sigma):
        assert n > 0, "n must be greather than 0"
        assert n % 2 == 1, "n must be odd"
        kernel = np.zeros((n, n), dtype=float)
        center = n // 2

        for i in range(n):
            for j in range(n):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)

        kernel /= kernel.sum()
        return kernel

    def gaussian_filter(self, n=3, sigma=1):
        kernel = self.gaussian_kernel(n, sigma)
        return convolve2d(self.img_, kernel, mode='same', boundary='fill', \
                          fillvalue=0)

    def avg_filter(self, n=3):
        kernel = self.avg_kernel(n)
        return convolve2d(self.img_, kernel, mode='same', boundary='fill', \
                          fillvalue=0)  
    
    def median_filter(self, n=3):
        assert n > 0, "n must be greather than 0"
        assert n % 2 == 1, "n must be odd"

        r = n//2
        img_pad = np.pad(self.img_, r, mode='constant', constant_values=0)
        median = np.zeros(shape=self.img_.shape, dtype=self.img_.dtype)

        for i in range(self.img_.shape[0]):
            for j in range(self.img_.shape[1]):
                patch = img_pad[i:i+n, j:j+n]
                median[i, j] = np.median(patch)
        return median
   
    def sobel_x(self):
        kernel = kernel_sobel_x
        return convolve2d(self.img_, kernel, mode='same', boundary='fill')

    def sobel_y(self):
        kernel = kernel_sobel_y
        return convolve2d(self.img_, kernel, mode='same', boundary='fill')

    def gradient(self):
        g_x, g_y = self.sobel_x(), self.sobel_y()
        return np.sqrt(g_x**2 + g_y**2), np.atan(g_y/g_x)

    def sobel(self, norm="L1"):
        assert norm=="L1" or norm=="L2", "norm must be L1 or L2"
        if norm=="L1":
            return np.abs(self.sobel_x()) + np.abs(self.sobel_y())
        else:
            return np.sqrt(self.sobel_x()**2 + self.sobel_y()**2)

    @staticmethod
    def difference(p, q):
        p, q = int(p), int(q)
        return 1/(abs(p-q) + 1) if p != q else np.inf

    @staticmethod
    def four_NN(x, y):
        return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

    @staticmethod
    def eight_NN(x, y):
        return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),\
                (x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1)]

    def KNN(self, x, y, visited, graph_dict, neighberhood_size=four_NN):
        h, w = self.img_.shape
        nearest_neighbours = []
        for i, j in neighberhood_size(x, y):
            # vérifier que le voisin est dans l'image
            if 0 <= i < h and 0 <= j < w:
                graph_dict[x*self.img_.shape[1]+y].append(i*self.img_.shape[1]+j)
                if not visited[i, j]:
                    nearest_neighbours.append((i, j))
                    visited[i, j] = True

        return nearest_neighbours

    def graph_img(self, neighberhood_size=four_NN, difference=difference):
        graph = []
        graph_dict = {i : [] for i in range(self.img_.size)}
        visited = np.zeros(shape=self.img_.shape, dtype=bool)
        for i in range(self.img_.shape[0]):
            for j in range(self.img_.shape[1]):
                visited[i, j] = True
                coordinate = i, j
                intensity = self.img_[i, j]
                for pixel in self.KNN(i, j, visited, graph_dict, neighberhood_size):
                    weight = difference(intensity, self.img_[pixel])
                    graph.append((coordinate, pixel, weight))
                    
        with open("graph_dict.json", "w") as f:
            json.dump(graph_dict, f, indent=4)
        return graph
     