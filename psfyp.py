#%%
import cv2 as cv
import numpy as np
import os

import scipy.fft as fft

from matplotlib import pyplot as plt
import matplotlib as mpl

from sklearn.preprocessing import normalize

# MAX_IMAGES = -1

# # ps constants
IMRANGE = 1.0

# average intensity needed to mask out
MASK_THRESH = (15 / 255) * IMRANGE

# sbl
MAX_ITERS = 100

# default weights
NOISE_WEIGHT = 0.3
SPECULAR_WEIGHT = 0.1

SPECULAR_SHINE = 10.0

# constants when applying specular highlights to images
SELF_SHADOW_THRESH = (40 / 255) * IMRANGE
SELF_SHADOW_FALLOFF = (40 / 255) * IMRANGE


#%%

# useful functions

def scale_image(img):
    """Scale values in an image to be in the range [0,1]

    Args:
        img (np.ndarray): Image as a numpy array

    Returns:
        np.ndarray: Scaled image
    """
    return (img / img.max()) * IMRANGE
    

def angular_error(m1, m2, mask=None, unit='d'):
    """Calculate angular error between vectors.

    Args:
        m1 (np.ndarray): First vector or matrix of vectors. If a 3-dimensional matrix, shape of last dimension should be 3.
        m2 (np.ndarray): Second vector or matrix of vectors. If a 3-dimensional matrix, shape of last dimension should be 3.
        mask (np.array[bool], optional): Mask of vectors to ignore - these values will have an error of 0. Defaults to None.
        unit (str, optional): Unit to return error in ('d':degrees, 'r':radians). Defaults to 'd'.

    Returns:
        np.ndarray: Element-wise angular error
    """

    if m1.shape == (3, 1):

        out = np.arccos(
            np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))
        )

    else:
        
        assert(m1.shape == m2.shape)
        assert(m1.shape[2] == 3)

        if mask.shape != m1.shape:
            mask = np.reshape(mask.copy(), m1.shape[:2])

        out = np.zeros(m1.shape[:2])

        for i in range(m1.shape[0]):
            for j in range(m1.shape[1]):
                v1 = m1[i][j]
                v2 = m2[i][j]

                if mask is not None and mask[i][j]:
                    continue

                if not (np.linalg.norm(v1) == 0 and np.linalg.norm(v2) == 0):
                    out[i][j] = np.arccos(
                        np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2) + np.finfo(float).eps)
                    )

                    if unit == 'd':
                        out[i][j] = out[i][j] * (180 / np.pi)
                    elif unit != 'r':
                        raise Exception("Invalid unit for angular error - should be one of 'r' or 'd'")

    return out


def multiplot(imgs, titles, save=None, show=True, textsize=None, cmaps=None, leftlabel=None, leftlabel_x=0.09, leftlabel_y=0.495, cbar=None):
    """Utility function for generating nice side-by-side plots. Essentially just a quick way to iterate through plt subplots.

    Args:
        imgs (list[np.ndarray]): Images to display
        titles (list[str]): Titles for each image
        save (str, optional): Path to write plot to. Defaults to None.
        show (bool, optional): If True, display the resulting image. Defaults to True.
        textsize (int, optional): Font size for image titles. Defaults to None.
        cmaps (list[str], optional): cmap for each image. Defaults to None.
        leftlabel (str, optional): Label to display at the left of the plot. Defaults to None.
        leftlabel_x (float, optional): x position for the left label. Defaults to 0.09.
        leftlabel_y (float, optional): y position for the left label. Defaults to 0.495.
        cbar (dict, optional): Parameters for a colourbar to be shown beside the final image. Defaults to None.
    """

    if textsize is None:
        textsize = 100/len(imgs)

    if cmaps is None:
        cmaps = ['gray'] * len(imgs)

    fig, axarr = plt.subplots(1,len(imgs)+(1 if cbar is not None else 0), figsize=(100/len(imgs), 20), width_ratios=([1]*len(imgs) + ([0.1] if cbar is not None else [])))

    for i, (img, title) in enumerate(zip(imgs, titles)):
        axarr[i].imshow(img, cmap=cmaps[i]) if len(img.shape) == 2 else axarr[i].imshow(img)
        axarr[i].title.set_text(title)
        axarr[i].title.set_size(textsize)
        axarr[i].axis('off')

    if leftlabel is not None:
        fig.text(leftlabel_x, leftlabel_y, leftlabel, fontdict={'size':textsize})

    if cbar is not None:

        cax = axarr[-1]
        cax.axis('off')

        # https://matplotlib.org/stable/users/explain/colors/colorbar_only.html
        norm = mpl.colors.Normalize(vmax=imgs[-1].max(), vmin=imgs[-1].min())

        cb = fig.colorbar(
            # plt.pcolormesh(imgs[-1], vmax=imgs[-1].max(), vmin=imgs[-1].min()), 
            mpl.cm.ScalarMappable(norm=norm, cmap=cbar['cmap']),
            ax=cax, fraction=cbar['fraction'], pad=0.04)
        cb.ax.tick_params(labelsize=cbar['label_size'])
        cb.set_label(cbar['title'], rotation=270, size=cbar['title_size'], labelpad=cbar['title_pad'])
        
        plt.set_cmap(cbar['cmap'])

    if save is not None:
        
        impath = '\\'.join(save.split('\\')[:-1]) + '\\'

        if not os.path.exists(impath):
            os.makedirs(impath)

        plt.savefig(save, dpi=fig.dpi, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def disp_heights(heights, mask):
    """Display a height map as an image, while using a mask to ignore pixels not containing a point.

    Args:
        heights (np.ndarray): Matrix of heights
        mask (np.ndarray[bool]): True if pixel does not contain a point, False otherwise
    """

    z_disp = heights.copy()

    # show masked points "below" the surface
    zero_val = heights.min() - 10

    n, m = heights.shape[:2]

    for i in range(n):
        for j in range(m):
            if mask[i][j]:
                z_disp[i][j] = zero_val

    plt.imshow(z_disp, cmap='gray')


def corrupt_images(
        imgs, 
        normals, 
        ldirs, 
        noise_weight=NOISE_WEIGHT, 
        specular_weight=SPECULAR_WEIGHT, 
        specular_shine=SPECULAR_SHINE, 
        self_shadow_thresh=SELF_SHADOW_THRESH, 
        self_shadow_falloff=SELF_SHADOW_FALLOFF, 
        write=None
    ):
    """Take in a list of images, a normal field, and a list of lighting directions, then add corruptions to these images.

    Args:
        imgs (list[np.ndarray]): List of images. Intensities should be in the range [0,1]
        normals (np.ndarray): Matrix of normals. Should be the same size as an image
        ldirs (np.ndarray): Lighting direction matrix in correspondance with images
        noise_weight (float, optional): Weight of addition when adding Gaussian noise overlay to images. Defaults to NOISE_WEIGHT.
        specular_weight (float, optional): Weight of addition when adding specular highlight overlay to images. Defaults to SPECULAR_WEIGHT.
        specular_shine (float, optional): "shininess" (k) parameter for specular highlights. Defaults to SPECULAR_SHINE.
        self_shadow_thresh (float, optional): Intensity threshold in source image for specular highlights to be masked out. Defaults to SELF_SHADOW_THRESH.
        self_shadow_falloff (float, optional): Additional threshold above self_shadow_thresh where a weighted portion of the specular highlight is added, instead of the full value. Defaults to SELF_SHADOW_FALLOFF.
        write (str, optional): Path to write corrupted images to. Defaults to None.

    Returns:
        (list[np.ndarray], list[dict]): Tuple containing: a list of corrupted images; a list of dicts containing supplementary information for each corrupted image (e.g. the specular mask)
    """

    out_images = []

    out_aux = []

    for k, img in enumerate(imgs):

        mask_specular = np.zeros(img.shape)

        mask_specular_raw = np.zeros(img.shape)

        # create a vector describing the light direction
        ldir = np.asarray(ldirs[k].T).ravel()

        # calculate the specular mask
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):

                normal = normals[i][j].copy() # get the normal for the pixel

                if np.linalg.norm(normal) == 0:
                    mask_specular[i][j] = 0
                    continue

                view_dir = np.array([0, 0, 1])

                # flip the y-axis to align with image space
                normal[1] = -normal[1] 

                reflection = ldir - ((2 * np.dot(ldir, normal)) * normal)

                specular = max((np.dot(reflection, view_dir)**specular_shine), 0)

                mask_specular[i][j] = specular


        # scale the raw specular mask to be in the range [0, 1]
        mask_specular = scale_image(mask_specular)
        mask_specular_raw = mask_specular.copy()

        # apply damping
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                # dampen the contribution if near self-shadow threshold
                if img[i][j] <= self_shadow_thresh:
                    mask_specular[i][j] = 0
                elif img[i][j] <= self_shadow_thresh + self_shadow_falloff:
                    mask_specular[i][j] *= (img[i][j] - self_shadow_thresh) / self_shadow_falloff
        
        # generate the gaussian noise mask
        mask_gaussian = np.zeros_like(img)
        mask_gaussian = np.random.normal(0, 1, img.shape)
        
        # clamp at 0
        mask_gaussian = np.maximum(mask_gaussian, 0)

        # normalize intensities
        mask_gaussian = scale_image(mask_gaussian)

        # add specular highlights
        img_specular = cv.addWeighted(img, 1.0, mask_specular, specular_weight, 0.0, dtype=-1)
        img_specular_raw = cv.addWeighted(img, 1.0, mask_specular_raw, specular_weight, 0.0, dtype=-1)

        # add noise
        img_masked = cv.addWeighted(img_specular, 1.0, mask_gaussian, noise_weight, 0.0, dtype=-1)

        out_images.append(img_masked)

        out_aux.append({
            'mask_specular' : mask_specular, 
            'mask_gaussian' : mask_gaussian, 
            'mask_specular_raw' : mask_specular_raw,
            'img_specular' : img_specular,
            'img_specular_raw' : img_specular_raw})
        
    # normalize intensities across images
    max_intensity = np.max([img.max() for img in out_images])
    out_images = [img / max_intensity for img in out_images]

    # write images to folder
    if write is not None:

        if not write.endswith('\\'):
            write += '\\'

        for i, img in enumerate(out_images):
            if not os.path.exists(write):
                os.makedirs(write)

            cv.imwrite(fr"{write}image-s{specular_weight:.2f}-n{noise_weight:.2f}-{i}.png", 255 * img)

    return out_images, out_aux



#%%

# class definitions

class PSSolver:

    mask = None
    
    SBL_MAX_ITERS = 100
    LAMBDA = 1.0e-5 # scaling factor
    SIGMA = 1.0e8 # source variance

    def __init__(self, observations, ldirs):
        """Photometric Solver class. Given a list of images and a list of lighting directions, initialize matrices to solve via photometric stereo.

        Args:
            images (np.ndarray): Observation matrix of shape m x n, for n images with m pixels each. Values should be in the range [0,1].
            ldirs (np.ndarray): Lighting matrix. Should be 3 x n matrix, where n is the number of images.
        """

        # turn images into observation matrix

        self.O = observations

        # generate lighting matrix
        
        self.L = ldirs

        # generate the mask

        self.set_mask()


    def set_mask(self, mask_thresh=MASK_THRESH, use_mask=None):
        """Set the mask of the solver. If use_mask is None, the mask is calculated based on average intensity of pixels.

        Args:
            shadow_thresh (float, optional): Value in the range [0,1]. If the average intensity of a pixel across images is below this value, it is masked out. Defaults to SHADOW_THRESH.
            use_mask (np.ndarray[bool], optional): Mask to use instead of calculating a new one. Defaults to None.
        """

        if use_mask is not None: # manually provide a mask
            self.mask = use_mask
            return

        # number of pixels
        m = self.O.shape[0]

        out = np.zeros((self.O.shape[0], ))

        for i in range(m):

            # if the average intensity of the pixel lies below the threshold, mask it out
            if np.mean(self.O[i]) < mask_thresh:
                out[i] = True
            else:
                out[i] = False

        self.mask = out

    
    def _reshape_matrix(self, mat, shape, final_dim=None):
        """Reshape a matrix, optionally fixing thhe size of the final dimension

        Args:
            mat (np.ndarray): Matrix to reshape
            shape (tuple[int]): Shape to resize matrix to
            final_dim (int, optional): Size to fix last dimension to. Defaults to None.

        Returns:
            np.ndarray: Reshaped matrix.
        """

        # preserve the final dimension of the matrix, i.e. the normals
        if final_dim == None:
            final_dim = shape[-1]

        if len(shape) == 2:
            mat = np.reshape(mat, shape + (final_dim, ))
        else:
            shape[-1] = final_dim

            mat = np.reshape(mat, shape)

        return mat


    def ps_basic(self, out_shape=None, use_mask=True):
        """Performs the basic photometric stereo algorithm with least-squares solving, using the stored images and lighting directions. Algorithm is given according to:
            
            Woodham, R.J., 1980. Photometric method for determining surface orientation from multiple images. 
            Optical engineering, 19(1), pp.139-144.

            and solved via the equation in https://www.youtube.com/watch?v=dNkMLqBUNKI&list=PL2zRqk16wsdpyQNZ6WFlGQtDICpzzQ925&index=11

        Args:
            out_shape (tuple[int], optional): Shape of matrix used to return normals. The last dimension should be omitted and is set to 3. Defaults to None.
            use_mask (bool, optional): If True, use the stored mask to skip normals for those pixels, and return [0,0,0]^T. Defaults to True.

        Returns:
            np.ndarray: Normals returned by basic photometric stereo.
        """

        N = np.zeros((self.O.shape[0], 3))

        # equation from https://www.youtube.com/watch?v=dNkMLqBUNKI&list=PL2zRqk16wsdpyQNZ6WFlGQtDICpzzQ925&index=11
        rhs = self.L * self.O.T

        square = np.linalg.inv((self.L * self.L.T))

        N = (square * rhs).T

        N = normalize(np.asarray(N), axis=1)
        
        # apply mask
        if self.mask is not None and use_mask:
                mask_flat = self.mask.flatten()
                for i in range(N.shape[0]):
                    if mask_flat[i]:
                        N[i] = [0, 0, 0]

        # flip y-axis on normals, to be aligned with coordinate system
        N[:,1] = -N[:,1]

        if out_shape is not None:

            N = self._reshape_matrix(N, out_shape, final_dim=3)

        return N
    

    def ps_sbl(self, max_iters=SBL_MAX_ITERS, lambda_=LAMBDA, sigma=SIGMA, num_processes=-1, out_shape=None, use_paper_algorithm=False):
        """Perform SBL photometric stereo as described by:

            Ikehata, S., Wipf, D., Matsushita, Y. and Aizawa, K., 2012. Robust photometric stereo
            using sparse regression. 2012 ieee conference on computer vision and pattern recognition.
            pp.318-325. https://ieeexplore.ieee.org/abstract/document/6247691

        Multiprocessing implementation was adapted from https://github.com/yasumat/RobustPhotometricStereo/blob/master/rps.py

        Args:
            max_iters (int, optional): Maximum number of iterations for pixel to converge. Defaults to SBL_MAX_ITERS.
            lambda_ (float, optional): Lambda parameter for SBL. Higher values assume sparser errors. Defaults to LAMBDA.
            sigma (float, optional): Prior variance toward the orientation of normals. Defaults to SIGMA.
            num_processes (int, optional): Number of processes to use for normal reconstruction. If -1, use half of the available processes. Defaults to -1.
            out_shape (tuple[int], optional): Shape of matrix used to return normals. The last dimension should be omitted and is set to 3. Defaults to None.
            use_paper_algorithm (bool, optional): If True, use the update rules provided in the original paper. This does not work!. Defaults to False.

        Returns:
            np.ndarray: Normals returned by SBL photometric stereo.
        """

        self.SBL_MAX_ITERS = max_iters
        self.LAMBDA = lambda_
        self.SIGMA = sigma
        
        # multiprocessing implementation adapted from https://github.com/yasumat/RobustPhotometricStereo/blob/master/rps.py
        indices = range(self.O.shape[0])

        if num_processes != 1:

            from multiprocessing import Pool
            import multiprocessing
            from tqdm import tqdm

            # https://stackoverflow.com/questions/41920124/multiprocessing-use-tqdm-to-display-a-progress-bar

            if num_processes == -1:
                num_processes = multiprocessing.cpu_count() // 2

            chunksize = int(len(indices)/(num_processes*25))

            # create processes
            with Pool(processes=num_processes) as p:
                if use_paper_algorithm:
                    out = list(tqdm(
                        p.imap(self._ps_sbl_solve_pixel_paper, indices, chunksize=chunksize), total=len(indices)
                    ))

                    # return values early - don't process the normals
                    return out[0], np.array([np.asmatrix(y).flatten().T for _, y in out]).squeeze()
    
                else:
                    out = list(tqdm(
                        p.imap(self._ps_sbl_solve_pixel, indices, chunksize=chunksize), total=len(indices)
                    ))
                p.close()

            # repair matrix of normals
            N = np.array([x.ravel() for x, _ in out])
            
            errors = np.array([np.asmatrix(y).flatten().T for _, y in out]).squeeze()

        
        else:
            
            N = np.zeros((self.O.shape[0], 3))
            errors = np.zeros((self.O.shape[0], self.O.shape[1]))

            # iterate through all pixels and solve one by one

            for index in indices:
                if use_paper_algorithm:
                    x, error = self._ps_sbl_solve_pixel_paper(index)
                else:
                    x, error = self._ps_sbl_solve_pixel(index)
                    errors[index, :] = error.ravel()
                    N[index, :] = x.ravel()

        
        # same issue as before - need to invert y-axis on normals
        N[:, 1] = -N[:, 1]

        if out_shape is not None:

            # reshape normals to the desired shape
            N = self._reshape_matrix(N, out_shape, final_dim=3)

            errors = self._reshape_matrix(errors, out_shape, final_dim=self.O.shape[1])

        return N, errors
    

    def _ps_sbl_solve_pixel(self, index):
        """Solve the normal and error component for a given pixel, as adapted from the implementation by:
        https://satoshi-ikehata.github.io/ -> https://drive.google.com/file/d/19IyBj3oynHXD7VQqX5BhG5l4lXlXhe4V/view
        in `min_SBL_error_regularizer.m`

        Args:
            index (int): Index of pixel to solve

        Returns:
            tuple: Tuple of recovered normal and error contribution
        """

        ERR_THR     = 1e-8 # For stopping criteria
        GAMMA_THR   = 1e-8 # For numerical stability
        sigma       = self.SIGMA # Source variance
        lambda_     = self.LAMBDA # Scaling factor

        # if masked, return an empty normal and no error variances
        if self.mask.flatten()[index]:
            return (np.array([0, 0, 0]), np.array([0] * self.O.shape[1]).T)

        A = self.L.T

        b = np.array(self.O[index, :]).T

        n,m = A.shape
        e_old = 1000*np.ones((n,1))
        gamma = np.ones((n,1))

        # update rules as according to https://satoshi-ikehata.github.io/ -> https://drive.google.com/file/d/19IyBj3oynHXD7VQqX5BhG5l4lXlXhe4V/view
        for _ in range(self.SBL_MAX_ITERS):
            
            Gamma_e = np.diagflat(gamma)
            Sigma_d = gamma + lambda_
            invD = np.diagflat(1./Sigma_d)

            sig_ge = np.linalg.solve((sigma**-1)*np.identity(3) + A.T*invD*A, A.T*invD)
            g_e = (invD - invD*A*sig_ge)*b
            
            e = Gamma_e*g_e
                
            if (np.linalg.norm(e-e_old) < ERR_THR):
                break
            
            e_old = e
            
            Ei = (invD - invD*A*sig_ge)*Gamma_e
            Sigma_e = Gamma_e - Gamma_e*Ei
            gamma = np.diag(e*e.T) + np.diag(Sigma_e)
            gamma = np.maximum(gamma, GAMMA_THR)
            

        Xi = np.linalg.solve(lambda_*np.identity(n) + Gamma_e + sigma*A*A.T,b)
        x = sigma*A.T*Xi
            
        # these equations for mean and standard deviation are taken from the original paper - they also work
        # Sigma = np.linalg.inv((sigma**-1)*np.identity(3) + A.T * invD * A)
        # x = Sigma * A.T * invD * b

        x = normalize(np.asarray(x), axis=0)
        error = e

        return x, error
    
    
    def _ps_sbl_solve_pixel_paper(self, index):
        """The below is implemented as described in the paper of Ikehata et al. (2012):
        https://ieeexplore.ieee.org/abstract/document/6247691

        Args:
            index (int): Index of pixel to solve.

        Returns:
            tuple: Tuple of recovered normal and error contribution
        """

        ERR_THR     = 1e-8 # For stopping criteria
        GAMMA_THR   = 1e-8 # For numerical stability
        sigma       = self.SIGMA # Source variance
        lambda_     = self.LAMBDA # Scaling factor

        if self.mask.flatten()[index]:
            return (np.array([0, 0, 0]), np.array([0] * self.O.shape[1]).T)

        A = self.L.T

        b = np.array(self.O[index, :]).T

        n, m = A.shape

        z = np.zeros((n, 1))
        u = np.zeros((n, 1))
        gamma = 1000 * np.ones((n, 1))


        # implement update rules, as given in Ikehata et al. (2012)
        for _ in range(self.SBL_MAX_ITERS):

            gamma_new = np.power(z, 2) + u

            gamma_new = np.maximum(gamma_new, GAMMA_THR)

            if (np.linalg.norm(gamma-gamma_new) < ERR_THR):
                gamma = gamma_new
                break

            Gamma = np.diagflat(gamma_new)

            D = np.linalg.inv(Gamma + lambda_*np.identity(n))

            S = D - D*A*np.linalg.inv((sigma**-1)*np.identity(3) + A.T*D*A) * A.T * D

            z = (Gamma * np.linalg.inv(S) * b).T

            u = np.diag(Gamma - Gamma * Gamma * np.linalg.inv(S))

            gamma = gamma_new


        Sigma = np.linalg.inv((sigma**-1)*np.identity(3) + A.T*D*A)
        mu = Sigma * A.T * np.linalg.inv(Gamma + lambda_ * np.eye(n)) * b

        x = mu
        error = gamma

        return x, error




class PSIntegrator:

    def __init__(self, normals, shape, mask=None):
        """Given a matrix of normals and a shape, integrate the normals into a height map with the given shape.

        Args:
            normals (np.ndarray): Matrix of normals. Will be reshaped to shape + (3, )
            shape (tuple[int]): Output shape of height map (i.e. the same shape as an image); should be 2-dimensional
            mask (np.ndarray[bool], optional): Mask out pixels in the normal field; when integrated, masked normals are set to [0, 0, -1]^T. Defaults to None.
        """

        self.shape = shape

        # normal matrix should be in "image shape"
        self.normals = np.reshape(normals, shape + (3, ))

        # also fit the mask to the appropriate shape
        if mask is not None:
            self.mask = np.reshape(mask, self.shape)
        else:
            self.mask = np.zeros(shape)

        # integrate the passed normals
        self.heights = self._frankott_chellappa()


    
    def set_normals(self, normals, shape=None):
        """Set the normals of the integrator. Useful when heights have already been calculated, and an .obj file is required with different normals to those origininally supplied.

        Args:
            normals (np.ndarray): Matrix of normals
            shape (tuple[int], optional): Shape to resize normal matrix to. Final dimension of 3 should not be provided and is added automatically. Defaults to None.
        """

        if shape is not None:
            self.normals = np.reshape(normals, shape + (3, ))
        else:
            self.normals = normals


    def _frankott_chellappa(self):
        """Perform Frankot-Chellappa integration on the stored normal field. Masked normals are taken as facing directly towards the camera.
        meshgrid and np.finfo(float).eps taken from https://github.com/labrieth/spytlab/blob/master/frankoChellappa.py

            Frankot, R. and Chellappa, R., 1988. A method for enforcing integrability in shape from
            shading algorithms. Ieee transactions on pattern analysis and machine intelligence, 10(4),
            pp.439-451.

        Returns:
            np.ndarray: Matrix of heights.
        """
        
        # first modify normals so that any masked out ones are flat, towards the camera
        normals_mod = self.normals.copy()

        n, m = self.shape

        for i in range(n):
            for j in range(m):
                if self.mask[i][j]:
                    normals_mod[i][j][0] = 0
                    normals_mod[i][j][1] = 0
                    normals_mod[i][j][2] = -1

        # generate matrices of ps and qs

        ps = -(normals_mod[:,:,0]) / (normals_mod[:,:,2] + np.finfo(float).eps)

        qs = -(normals_mod[:,:,1]) / (normals_mod[:,:,2] + np.finfo(float).eps)

        # calculate fourier transforms

        Ps = fft.fft2(ps)

        Qs = fft.fft2(qs)

        # https://github.com/labrieth/spytlab/blob/master/frankoChellappa.py
        us, vs = np.meshgrid(
            fft.fftfreq(m) * 2 * np.pi,
            fft.fftfreq(n) * 2 * np.pi, 
            indexing='xy'
        )

        # fourier transform of optimal heights

        Z_tilde = ((1j * us * Ps) + (1j * vs * Qs)) / (us**2 + vs**2 + np.finfo(float).eps)

        z_tilde = fft.ifft2(Z_tilde)

        z_tilde = -np.real(z_tilde) # depth axis is inverted

        return z_tilde


    def get_heights(self):
        return self.heights
    

    def to_obj(self, filename='out.obj'):
        """Export the stored depths and normals to an .obj file. Masked normals are ignored and not written.

        Args:
            filename (str, optional): Filename to write to. Defaults to 'out.obj'.
        """

        if not filename.endswith('.obj'):
            filename += '.obj'

        # assign index to each point
        indices = np.zeros(self.shape, dtype=int)

        counter = 1

        n, m = self.shape

        for i in range(n):
            for j in range(m):
                if not self.mask[i][j]:
                    indices[i][j] = counter
                    counter += 1
                else:
                    indices[i][j] = 0

        # write the .obj file
        with open(filename, 'w') as f:
            for i in range(n): # start with vertices
                for j in range(m):
                    if indices[i][j] != 0:
                        f.write(f'v {i} {j} {self.heights[i][j]}\n')

            f.write('\n\n')

            for i in range(n): # then write normals
                for j in range(m):
                    if indices[i][j] != 0:
                        f.write(f'vn {self.normals[i][j][0]} {self.normals[i][j][1]} {self.normals[i][j][2]}\n')

            f.write('\n\n')

            for i in range(0, n-1): # finally, create faces
                for j in range(0, m-1):

                    v1 = indices[i][j]
                    v2 = indices[i][j+1]
                    v3 = indices[i+1][j]
                    v4 = indices[i+1][j+1]

                    v1f = indices[i][j] != 0
                    v2f = indices[i][j+1] != 0
                    v3f = indices[i+1][j] != 0
                    v4f = indices[i+1][j+1] != 0

                    # if there are any triangles formed by adjacent pixels, write a face

                    if v1f and v2f and v3f and v4f:
                        f.write(f'f {v1}//{v1} {v3}//{v3} {v2}//{v2}\n')
                        f.write(f'f {v2}//{v2} {v3}//{v3} {v4}//{v4}\n')
                    elif v1f and v2f and v3f:
                        f.write(f'f {v1}//{v1} {v3}//{v3} {v2}//{v2}\n')
                    elif v1f and v2f and v4f:
                        f.write(f'f {v1}//{v1} {v4}//{v4} {v2}//{v2}\n')
                    elif v1f and v3f and v4f:
                        f.write(f'f {v1}//{v1} {v3}//{v3} {v4}//{v4}\n')
                    elif v2f and v3f and v4f:
                        f.write(f'f {v2}//{v2} {v3}//{v3} {v4}//{v4}\n')


            f.close()


    def display(self):
        """Show the height map as an image. Masked normals at set to appear "behind" other points.
        """

        z_disp = self.heights.copy()

        # display masked values "behind" the heights
        zero_val = self.heights.min() - 10

        n, m = self.shape

        for i in range(n):
            for j in range(m):
                if self.mask[i][j]:
                    z_disp[i][j] = zero_val

        plt.imshow(z_disp, cmap='gray')


    
    def smooth_normals(self, num_processes=-1, window_size=7, sigma_f=2.0, sigma_g=0.1, chunk_step_size=25):
        """Smooth the normals stored in the integrator using the stored height information. Smoothing algorithm is implemented according to:

            Jones, T., Durand, F. and Zwicker, M., 2004. Normal improvement for point rendering. Ieee
            computer graphics and applications, 24(4), pp.53-56.

            Derivation of partial derivatives and implementation was done as part of the project.

        Args:
            num_processes (int, optional): Number of processes to use for normal smoothing. If -1, use half of the available processes. Defaults to -1.
            window_size (int, optional): Window of adjacent points to consider when calculating smoothed position for normal. Defaults to WINDOW_SIZE.
            sigma_f (float, optional): Standard deviation for Gaussian function weighing distance between source and contributing point. Defaults to SIGMA_F.
            sigma_g (float, optional): Standard deviation for Gaussian function weighing distance between source and predicted point. Defaults to SIGMA_G.
            chunk_step_size (int, optional): Number of normals to process per chunk when using multiple processes. Defaults to 25.

        Returns:
            np.ndarray: Matrix of smoothed normals, in "image shape" (m x n x 3 matrix)
        """
        # normal smoothing

        self.WINDOW_SIZE = window_size

        self.SIGMA_F = sigma_f
        self.SIGMA_G = sigma_g

        # generate vectors for point positions

        def generate_points(heights):
            points = np.zeros(heights.shape + (3,))

            n, m = self.shape

            for i in range(n):
                for j in range(m):
                    points[i][j] = [i, j, heights[i][j]]

            return points

        self.points = generate_points(self.heights)

        # then apply bilateral filter

        MAX_HEIGHT, MAX_WIDTH = self.shape

        new_normals = np.zeros_like(self.normals).reshape((MAX_HEIGHT * MAX_WIDTH, 3))
        
        if num_processes != 1:

            from multiprocessing import Pool
            import multiprocessing
            from tqdm import tqdm


            indices = range(MAX_HEIGHT * MAX_WIDTH)

            if num_processes == -1:
                num_processes = multiprocessing.cpu_count() // 2

            chunksize = int(len(indices)/(num_processes*chunk_step_size))

            # https://stackoverflow.com/questions/41920124/multiprocessing-use-tqdm-to-display-a-progress-bar
            with Pool(processes=num_processes) as p:
                    out = list(tqdm(p.imap(self._jones_smoothing, indices, chunksize=chunksize), total=len(indices)))
                    p.close()

            new_normals = np.array([norm.ravel() for norm in out]).reshape(self.shape + (3, ))
        
        else:
            for i in range(MAX_HEIGHT):
                for j in range(MAX_WIDTH):
                    if self.mask[i][j]:
                        continue
                    new_normals[i][j] = self._compute_new_normal(self.points[i][j], i, j).ravel()

        return new_normals
    
    # contituent functions for F(s)

    def _f(self, s, p):

        std = self.SIGMA_F

        x = np.linalg.norm(s - p)
        
        return np.exp(-(x**2)/(2*(std**2)))


    def _g(self, s, p, p_norm):
            
        std = self.SIGMA_G

        x = np.linalg.norm(self._Pi(s, p, p_norm) - s)

        return np.exp(-(x**2)/(2*(std**2)))


    def _Pi(self, s, p, p_norm):
        return s + (np.dot(p - s, p_norm) * p_norm)


    def _k(self, s, i_x, i_y):
        """Calculate the sum of the weights for surrounding points

        Args:
            s (np.array): Source point
            i_x (int): Image x for source point
            i_y (int): Image y for source point
            window_size (int, optional): Size of window of adjacent points to sample. Defaults to WINDOW_SIZE.

        Returns:
            float: k(s) value
        """
        total = 0

        # scores = np.zeros((2*window_size + 1, 2*window_size + 1))

        # we really should take a combination of scores across all points - however, weighting falls off quickly, so much more efficient to use a small window

        MAX_HEIGHT, MAX_WIDTH = self.shape

        for i in range(i_x-self.WINDOW_SIZE, i_x+self.WINDOW_SIZE+1):

            if i < 0 or i >= MAX_HEIGHT:
                # scores[i - (i_x-window_size),:] = 0
                continue


            for j in range(i_y-self.WINDOW_SIZE, i_y+self.WINDOW_SIZE+1):

                if j < 0 or j >= MAX_WIDTH or self.mask[i][j]:
                    # scores[i - (i_x-window_size), j - (i_y-window_size)] = 0
                    continue

                p = self.points[i][j]
                p_norm = self.normals[i][j]

                total += self._f(s, p) * self._g(s, p, p_norm)
                # scores[i - (i_x-window_size)][j - (i_y-window_size)] = self.f(s, p) * self.g(s, p, p_norm)

        return total
    
    
    def _values(self, s, i_x, i_y):
        """This is the "numerator" part of the function (Pi_p(s) * f(||s-p||) * g(||Pi_p(s)-s||))

        Args:
            s (np.array): Source point
            i_x (int): Image x for source point
            i_y (int): Image y for source point
            window_size (int, optional): Size of window of adjacent points to sample. Defaults to WINDOW_SIZE.

        Returns:
            float: Pi_p(s) * f(||s-p||) * g(||Pi_p(s)-s||) value
        """

        total = np.zeros((3, ))

        MAX_HEIGHT, MAX_WIDTH = self.shape

        for i in range(i_x-self.WINDOW_SIZE, i_x+self.WINDOW_SIZE+1):

            if i < 0 or i >= MAX_HEIGHT:
                continue

            for j in range(i_y-self.WINDOW_SIZE, i_y+self.WINDOW_SIZE+1):

                if j < 0 or j >= MAX_WIDTH or self.mask[i][j]:
                    continue
                
                p = self.points[i][j]
                p_norm = self.normals[i][j]

                total += self._Pi(s, p, p_norm) * self._f(s, p) * self._g(s, p, p_norm)

        return total

    # define partial derivatives of each funtion

    def _f_arg_sqr_dx(self, s, p, wrt):
        """Partial derivative of ||s-p||^2

        Args:
            s (np.array): Source point
            p (np.array): Contributing point
            wrt (int): Index of axis to differentiate with respect to

        Returns:
            float: Evaluated value
        """
        return 2 * (s[wrt] - p[wrt])


    def _g_arg_sqr_dx(self, s, p, p_norm, wrt):
        """Partial derivative of ||Pi_p(s)-s||^2

        Args:
            s (np.array): Source point
            p (np.array): Contributing point
            p_norm (np.array): Normal at contributing point
            wrt (int): Index of axis to differentiate with respect to

        Returns:
            float: Evaluated value
        """

        coefficient = -2 * p_norm[wrt] * ((p_norm[0]**2)+(p_norm[1]**2)+(p_norm[2]**2))

        return coefficient * np.dot(p - s, p_norm)


    def _fg_dx(self, s, p, p_norm, wrt):
        """Partial derivative of f(||s-p||)*g(||Pi_p(s)-s||)

        Args:
            s (np.array): Source point
            p (np.array): Contributing point
            p_norm (np.array): Normal at contributing point
            wrt (int): Index of axis to differentiate with respect to

        Returns:
            float: Evaluated value
        """
        coefficient = -(self._f_arg_sqr_dx(s, p, wrt)/(2*(self.SIGMA_F**2)) + self._g_arg_sqr_dx(s, p, p_norm, wrt)/(2*(self.SIGMA_G**2)))

        return coefficient * self._f(s, p) * self._g(s, p, p_norm)


    def _Pi_dx(self, p_norm, wrt):
        """Partial derivative of Pi_p(s)

        Args:
            p_norm (np.array): Normal at contributing point
            wrt (int): Index of axis to differentiate with respect to

        Returns:
            np.array: evaluated value
        """

        out = np.array([
            -p_norm[wrt] * p_norm[0],
            -p_norm[wrt] * p_norm[1],
            -p_norm[wrt] * p_norm[2]
        ])

        out[wrt] += 1

        return out
    

    def _k_dx(self, s, i_x, i_y, wrt):
        """Differential sum of the weights for surrounding points

        Args:
            s (np.array): Source point
            i_x (int): Image x for source point
            i_y (int): Image y for source point
            wrt (int): Index of axis to differentiate with respect to
            window_size (int, optional): Size of window of adjacent points to sample. Defaults to WINDOW_SIZE.

        Returns:
            float: evaluated value
        """

        total = 0

        # scores = np.zeros((2*window_size + 1, 2*window_size + 1))

        # we really should take a combination of scores - however, weighting falls off quickly, so much more efficient to use a small window
        MAX_HEIGHT, MAX_WIDTH = self.shape

        for i in range(i_x-self.WINDOW_SIZE, i_x+self.WINDOW_SIZE+1):

            if i < 0 or i >= MAX_HEIGHT:
                # scores[i - (i_x-window_size),:] = 0
                continue


            for j in range(i_y-self.WINDOW_SIZE, i_y+self.WINDOW_SIZE+1):

                if j < 0 or j >= MAX_WIDTH or self.mask[i][j]:
                    # scores[i - (i_x-window_size), j - (i_y-window_size)] = 0
                    continue
                
                p = self.points[i][j]
                p_norm = self.normals[i][j]

                total += self._fg_dx(s, p, p_norm, wrt=wrt)

                # scores[i - (i_x-window_size)][j - (i_y-window_size)] = - self.fg_dx(s, p, p_norm, wrt=wrt)/((self.f(s, p) * self.g(s, p, p_norm) +  + np.finfo(float).eps)**2)

        return total
    

    def _values_dx(self, s, i_x, i_y, wrt):
        """Partial derivative of the "numerator" part of the function (Pi_p(s) * f(||s-p||) * g(||Pi_p(s)-s||))

        Args:
            s (np.array): Source point
            i_x (int): Image x for source point
            i_y (int): Image y for source point
            wrt (int): Index of axis to differentiate with respect to
            window_size (int, optional): Size of window of adjacent points to sample. Defaults to WINDOW_SIZE.

        Returns:
            float: evaluated value
        """

        total = np.zeros((3, ))

        MAX_HEIGHT, MAX_WIDTH = self.shape

        for i in range(i_x-self.WINDOW_SIZE, i_x+self.WINDOW_SIZE+1):

            if i < 0 or i >= MAX_HEIGHT:
                continue

            for j in range(i_y-self.WINDOW_SIZE, i_y+self.WINDOW_SIZE+1):

                if j < 0 or j >= MAX_WIDTH or self.mask[i][j]:
                    continue
                
                p = self.points[i][j]
                p_norm = self.normals[i][j]

                total += (self._fg_dx(s, p, p_norm, wrt=wrt) * self._Pi(s, p, p_norm)) + (self._f(s, p) * self._g(s, p, p_norm) * self._Pi_dx(p_norm, wrt=wrt))

        return total
    
    
    def _compute_new_normal(self, p, pos_x, pos_y):
        """Given a point and its image coordinated, calculate the smoothed normal as according to https://ieeexplore.ieee.org/abstract/document/1310211

        Args:
            p (np.array): Cartesian coordinates of point
            pos_x (int): Image x for point
            pos_y (int): Image y for point

        Returns:
            np.array: Smoothed normal
        """

        try:
            jacobian = np.zeros((3, 3))

            values = self._values(p, pos_x, pos_y)
            k = self._k(p, pos_x, pos_y)

            for i in range(3):

                column = self._values_dx(p, pos_x, pos_y, wrt=i) - ((self._k_dx(p, pos_x, pos_y, wrt=i) * values) / k)

                jacobian[:, i] = column

            jacobian = np.linalg.inv(np.matrix(jacobian))
            new_normal = np.linalg.inv(jacobian).T * np.matrix(self.normals[pos_x][pos_y]).T

            # another way of calculating the normal - gives the same results
            # jacobian = np.matrix(jacobian)
            # new_normal = jacobian.getH().T * np.matrix(self.normals[pos_x][pos_y]).T

            new_normal = normalize(np.asarray(new_normal), axis=0)

            # sometimes predicted normals face away from the camera - this is impossible for a photometric stereo task
            # in these cases, just set the point to be flat toward the camera
            if np.dot(new_normal.ravel(), np.array([0, 0, 1])) < 0:
                new_normal = np.array([0, 0, 1])
        except:
            # default to a normal towards the camera in the case of an error
            new_normal = np.array([0, 0, 1])

        return new_normal
    
    
    def _jones_smoothing(self, index):
        """Multiprocessing function to smooth a normal. Retrieves image coordinates and point, and calculates the smoothed normal

        Args:
            index (int): Index of flattened list of points to smooth

        Returns:
            np.array: Smoothed normal
        """

        _, MAX_WIDTH = self.shape

        # calculate the 2D position for the given index
        i = int(np.floor(index / MAX_WIDTH))
        j = int(index - (i * MAX_WIDTH))

        if self.mask[i][j]:
            return np.array([0, 0, 0])

        return self._compute_new_normal(self.points[i][j], i, j).ravel()

        







#%%

if __name__ == '__main__':
    
    pass

