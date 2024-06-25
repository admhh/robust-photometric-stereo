import argparse
import os

import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt

from psfyp import PSSolver, PSIntegrator, scale_image, corrupt_images, angular_error

# function to write normals to an image
def write_normals(normals, path):
    normals_write = np.zeros_like(normals)

    # need to shuffle axes around
    normals_write[:,:,0] = normals[:,:,2].copy()
    normals_write[:,:,1] = normals[:,:,1].copy()
    normals_write[:,:,2] = normals[:,:,0].copy()

    # scale and write image
    cv.imwrite(path, 255 * normals_write)

if __name__ == "__main__":
    
    # arguments
    parser = argparse.ArgumentParser("Argument Parser")
    
    parser.add_argument("--images", help="Path to directory containing images", type=str, required=True)
    parser.add_argument("--ldirs", help="Path to file containing lighting directions", type=str, required=True)

    parser.add_argument("--results", help="Path to write results to", type=str, required=False, default='results\\')
    parser.add_argument("--algorithm", help="'basic' - Basic (Least-Squares) Photometric Stereo. 'sbl' - SBL Photometric Stereo", type=str, required=False, default='sbl')
    parser.add_argument("--num-processes", dest='num_processes', help="Number of processes to use for SBL Photometric Stereo and normal smoothing. If -1, use all but 1 available processes. Defaults to -1", type=int, required=False, default=-1)
    parser.add_argument("--region", help="If using the 'frog' dataset, crop images to a specific region of the model. Can be one of 'whole', 'head' or 'tummy'.", type=str, required=False, default='whole')

    parser.add_argument("--specular", help="Specular weight for corrupted data", type=float, required=False, default=0.3)
    parser.add_argument("--noise", help="Noise weight for corrupted data", type=float, required=False, default=0.1)
    
    parser.add_argument("--sbl-lambda", dest='sbl_lambda', help="Lambda parameter for SBL Photometric Stereo", type=float, required=False, default=1.0e-3)
    parser.add_argument("--sbl-sigma", dest='sbl_sigma', help="Sigma parameter for SBL Photometric Stereo", type=float, required=False, default=1.0e6)
    parser.add_argument("--sbl-max-iters", dest='sbl_max_iters', help="Set maximum number of per-pixel iterations for SBL Photometric Stereo", type=int, required=False, default=100)
    parser.add_argument("--sbl-use-paper-algorithm", dest='sbl_use_paper_algorithm', help="Use the original update rules as provided in the paper by Ikehata et al. (2012). (This doesn't work)", action='store_true', required=False, default=False)

    parser.add_argument("--no-smooth", dest='smooth', help="Don't smooth corrupted normals after reconstruction", required=False, action='store_false', default=True)
    
    parser.add_argument("--smoothing-f", dest='smoothing_f', help="Sigma_f parameter for normal smoothing", type=float, required=False, default=2.0)
    parser.add_argument("--smoothing-g", dest='smoothing_g', help="Sigma_g parameter for normal smoothing", type=float, required=False, default=0.1)
    parser.add_argument("--window-size", dest='window_size', help="Size of window from which points are sampled when smoothing the reconstructed normals. If -1, use twice the value of sigma_f", type=int, required=False, default=-1)

    args = parser.parse_args()
    
    # ===================================================================================================================== #

    # read arguments
    IMG_PATH = args.images
    if not IMG_PATH.endswith('\\'):
        IMG_PATH += '\\'
        
    LDIRS_FILE = args.ldirs
    if not LDIRS_FILE.endswith('.txt'):
        raise Exception("ldirs argument should be a .txt file")
    
    RESULTS_PATH = args.results
    if not RESULTS_PATH.endswith('\\'):
        RESULTS_PATH += '\\'

    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    ALGORITHM = args.algorithm
    if ALGORITHM not in ['basic', 'sbl']:
        raise Exception("Invalid algorithm - should be one of 'basic' or 'sbl'")
    
    NUM_PROCESSES = args.num_processes

    REGION = args.region

    SPECULAR_WEIGHT = args.specular
    NOISE_WEIGHT = args.noise

    SBL_LAMBDA = args.sbl_lambda
    SBL_SIGMA = args.sbl_sigma
    SBL_MAX_ITERS = args.sbl_max_iters
    SBL_USE_PAPER_ALGORITHM = args.sbl_use_paper_algorithm

    SMOOTH = args.smooth

    SIGMA_F = args.smoothing_f
    SIGMA_G = args.smoothing_g

    WINDOW_SIZE = args.window_size
    if WINDOW_SIZE == -1:
        WINDOW_SIZE = int(2 * SIGMA_F)

    # ===================================================================================================================== #

    # read images
    images = []
    i = 0
    for img in os.listdir(IMG_PATH):

        new_img = cv.imread(f"{IMG_PATH}{img}")

        # convert images to greyscale
        new_img = cv.cvtColor(new_img, cv.COLOR_RGB2GRAY)

        # scale image to range [0, 1]
        new_img = scale_image(new_img)

        images.append(new_img)

    if REGION == 'head':
        images = [image[10:310,170:470] for image in images] # frog head
    elif REGION == 'tummy':
        images = [image[280:480, 250:450] for image in images] # frog tummy

    IMAGE_SHAPE = images[0].shape[:2]

    # flatten images into m x n observation matrix
    O = np.matrix([image.flatten() for image in images]).T

    NUM_IMAGES = len(images)

    # ===================================================================================================================== #

    # read ldirs
    with open(LDIRS_FILE) as f:
        txt = f.readlines()

        ldirs = [
            [np.float64(comp) for comp in line.split(' ') if len(comp) > 0]
            for line in txt
        ]

        ldirs = np.matrix([
            [x, y, z] for x, y, z in zip(ldirs[0], ldirs[1], ldirs[2])
        ])

        f.close()

    L = ldirs.T
    
    try:
        assert(len(images) == ldirs.shape[0])
    except:
        raise Exception('Number of images and lighting directions does not match!')
    
    # ===================================================================================================================== #

    # create solver
    solver = PSSolver(O, L)

    # get the mask
    mask = solver.mask

    print('Reconstructing normals...')

    if ALGORITHM == 'basic':
        normals = solver.ps_basic(out_shape=IMAGE_SHAPE)
    else: # sbl
        normals, errors = solver.ps_sbl(
            max_iters=SBL_MAX_ITERS,
            lambda_=SBL_LAMBDA,
            sigma=SBL_SIGMA,
            num_processes=NUM_PROCESSES,
            out_shape=IMAGE_SHAPE,
            use_paper_algorithm=SBL_USE_PAPER_ALGORITHM
        )

        # write errors

        if not os.path.exists(fr'{RESULTS_PATH}\sbl-error-variances\\'):
            os.makedirs(fr'{RESULTS_PATH}\sbl-error-variances\\')

        for idx in range(NUM_IMAGES):

            errors_show = errors[:,:,idx].copy()
            errors_show = np.absolute(errors_show)

            plt.imshow(errors_show)
            plt.axis('off')
            plt.set_cmap('plasma')
            plt.title('Abs. SBL Error Variances', size=20)
            cb = plt.colorbar(
                plt.pcolormesh(errors_show, vmax=errors_show.max(), vmin=errors_show.min())
                , fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=12)
            cb.set_label('Abs. Variance', rotation=270, labelpad=18, size=12)

            plt.savefig(fr'{RESULTS_PATH}\sbl-error-variances\image-{idx:2}.png', dpi=plt.gcf().dpi, bbox_inches='tight')
            plt.close()

    # save normals
    write_normals(normals, fr"{RESULTS_PATH}normals.png")

    print('Integrating surface...')

    integrator = PSIntegrator(normals, IMAGE_SHAPE, mask)

    integrator.to_obj(f'{RESULTS_PATH}surface.obj')

    # ===================================================================================================================== #

    # corrupt images
    print('Corrupting images...')

    images_corrupted, _ = corrupt_images(images, normals, ldirs, specular_weight=SPECULAR_WEIGHT, noise_weight=NOISE_WEIGHT)
    
    # flatten corrupted images into m x n observation matrix
    O_corrupted = np.matrix([image.flatten() for image in images_corrupted]).T

    if not os.path.exists(fr'{RESULTS_PATH}\corrupted-images\\'):
            os.makedirs(fr'{RESULTS_PATH}\corrupted-images\\')

    # save the corrupted images
    for i in range(NUM_IMAGES):
        cv.imwrite(fr'{RESULTS_PATH}\corrupted-images\image-corrupted-{i:2}.png', 255 * images_corrupted[i])
        
    # ===================================================================================================================== #

    # create the corrupted solver
    solver_corrupted = PSSolver(O_corrupted, L)
    
    solver_corrupted.set_mask(use_mask=mask)

    print('Reconstructing normals (corrupted)...')

    if ALGORITHM == 'basic':
        normals_corrupted = solver_corrupted.ps_basic(out_shape=IMAGE_SHAPE)
    else: # sbl
        normals_corrupted, errors_corrupted = solver_corrupted.ps_sbl(
            max_iters=SBL_MAX_ITERS,
            lambda_=SBL_LAMBDA,
            sigma=SBL_SIGMA,
            num_processes=NUM_PROCESSES,
            out_shape=IMAGE_SHAPE,
            use_paper_algorithm=SBL_USE_PAPER_ALGORITHM
        )
        
        # write errors
        for idx in range(NUM_IMAGES):

            errors_show = errors_corrupted[:,:,idx].copy()
            errors_show = np.absolute(errors_show)

            plt.imshow(errors_show)
            plt.set_cmap('plasma')
            plt.axis('off')
            plt.title('Abs. SBL Error Variances', size=20)
            cb = plt.colorbar(
                plt.pcolormesh(errors_show, vmax=errors_show.max(), vmin=errors_show.min())
                , fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=12)
            cb.set_label('Abs. Variance', rotation=270, labelpad=18, size=12)

            plt.savefig(fr'{RESULTS_PATH}\sbl-error-variances\image-corrupted-{idx:2}.png', dpi=plt.gcf().dpi, bbox_inches='tight')
            plt.close()

    # save normals
    write_normals(normals_corrupted, fr"{RESULTS_PATH}normals-corrupted.png")

    # calculate mean angular error

    if not os.path.exists(fr'{RESULTS_PATH}\angular-differences\\'):
        os.makedirs(fr'{RESULTS_PATH}\angular-differences\\')


    ae = angular_error(normals, normals_corrupted, solver.mask)

    plt.set_cmap('viridis')
    plt.imshow(ae)
    plt.title(f'Mean Angular Difference: {ae.mean():.4f}$^\circ$')
    plt.axis('off')
    cb = plt.colorbar(
        plt.pcolormesh(ae, vmax=ae.max(), vmin=ae.min())
        , fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=12)
    cb.set_label('Difference ($^\circ$)', rotation=270, labelpad=18, size=12)

    plt.savefig(fr'{RESULTS_PATH}\angular-differences\baseline-corrupted.png', dpi=plt.gcf().dpi, bbox_inches='tight')
    plt.close()

    print('Integrating surface (corrupted)...')

    integrator_corrupted = PSIntegrator(normals_corrupted, IMAGE_SHAPE, mask)

    integrator_corrupted.to_obj(f'{RESULTS_PATH}surface-corrupted.obj')
    
    # ===================================================================================================================== #

    if SMOOTH:

        print('Smoothing corrupted normals...')

        WINDOW_SIZE = 16

        normals_smoothed = integrator_corrupted.smooth_normals(
            sigma_f=SIGMA_F,
            sigma_g=SIGMA_G,
            window_size=WINDOW_SIZE,
            num_processes=NUM_PROCESSES
        )
        
        # save normals
        write_normals(normals_smoothed, fr"{RESULTS_PATH}normals-smoothed.png")


        # angular difference
        ae = angular_error(normals, normals_smoothed, solver.mask)


        plt.set_cmap('viridis')
        plt.imshow(ae)
        plt.title(f'Mean Angular Difference: {ae.mean():.4f}$^\circ$')
        plt.axis('off')
        cb = plt.colorbar(
            plt.pcolormesh(ae, vmax=ae.max(), vmin=ae.min())
            , fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=12)
        cb.set_label('Difference ($^\circ$)', rotation=270, labelpad=18, size=12)

        plt.savefig(fr'{RESULTS_PATH}\angular-differences\baseline-smoothed.png', dpi=plt.gcf().dpi, bbox_inches='tight')
        plt.close()

        print('Integrating surface (smoothed)...')

        integrator_smoothed = PSIntegrator(normals_smoothed, IMAGE_SHAPE, mask)

        integrator_smoothed.to_obj(f'{RESULTS_PATH}surface-smoothed.obj')

