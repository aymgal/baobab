import os
import numpy as np
import galsim
from lenstronomy.Util import util

def get_galsim_image(image_size, pixel_size, catalog_dir=None, catalog_name=None, catalog_index=0, 
                     galsim_scale=1, galsim_angle=0, galsim_center_x=0, galsim_center_y=0,
                     psf_size=49, psf_pixel_size=0.074, galaxy_type='real',
                     psf_type='real', psf_gaussian_fwhm=0.2, no_convolution=False,
                     draw_image_method='auto', verbose=False):
    """
    Generates a realistic galaxy using galsim (HST F814W extracted).
    """

    # Load catalog of galaxies
    cat = galsim.COSMOSCatalog(dir=catalog_dir, file_name=catalog_name)
    if verbose:
        print("Number of galaxies in catalog '{}' : {}".format(catalog_name, cat.nobjects))
    
    # effective pixel size
    pixel_size_eff = pixel_size / galsim_scale
    
    # Get galaxy object
    gal = cat.makeGalaxy(catalog_index, gal_type=galaxy_type, noise_pad_size=0)
    
    # apply rotation -> we do it after accessing the PSF otherwise raises an error
    #if angle != 0:
    #    gal_rot = gal.rotate(angle * galsim.radians)
    
    if psf_type == 'real':
        if galaxy_type == 'real':
            # Get original (untouched) PSF
            psf_kernel_untouched = gal.psf_image.array
            # Dilate the PSF to match required resolution
            psf_dilate_factor = psf_pixel_size / 0.074  # taken for HST F814W band
            psf = gal.original_psf.dilate(psf_dilate_factor).withFlux(1.)
            # Get the actual image of the psf
            # note that we set 'use_true_center' to False to make sure that the PSF is centered on a pixel (event if even-size image)
            psf_kernel = psf.drawImage(nx=psf_size, ny=psf_size, use_true_center=False, 
                                       scale=pixel_size_eff).array
        else:
            psf_kernel = np.zeros((image_size, image_size))
            psf_kernel_untouched = np.zeros((image_size, image_size))
    elif psf_type == 'gaussian':
        # Dilate the PSF to match required resolution
        psf = galsim.Gaussian(fwhm=psf_gaussian_fwhm, flux=1.0)
        # Get the actual image of the psf
        # note that we set 'use_true_center' to False to make sure that the PSF is centered on a pixel (event if even-size image)
        psf_kernel = psf.drawImage(nx=psf_size, ny=psf_size, use_true_center=False, 
                                   scale=pixel_size_eff).array
        psf_kernel_untouched = np.zeros((image_size, image_size))
    
    # apply rotation
    if galsim_angle != 0:
        gal = gal.rotate(galsim_angle * galsim.radians)
    
    # Performs convolution with PSF
    if no_convolution is False:
        if psf_type == 'real' and galaxy_type == 'parametric':
            print("WARNING : no 'real' PSF convolution possible with gal_type 'parametric' !")
        else:
            gal = galsim.Convolve(gal, psf)
    
    # Project galaxy on an image grid
    image_galaxy = gal.drawImage(nx=image_size, ny=image_size, use_true_center=True,
                                 offset=[galsim_center_x, galsim_center_y],
                                 scale=pixel_size_eff, method=draw_image_method).array
    
    return image_galaxy, psf_kernel, psf_kernel_untouched

def kwargs_galsim2interpol(image_size, pixel_size, supersampling_factor, 
                           kwargs_galsim_setup, kwargs_galsim_param):
    """
    Takes as input galsim parameters, generates a galsim galaxy from those
    and setup the 'INTERPOL' light profile of lenstronomy with the 
    """
    kwargs_interpol = {}
    if 'magnitude' in kwargs_galsim_param:
        # magnitude normalization performed in baobab/lenstronomy afterwards, not in galsim
        kwargs_interpol['magnitude'] = kwargs_galsim_param.pop('magnitude')
    # prepare for galsim
    pixel_size_ss, args, kwargs = _prepare_galsim(image_size, pixel_size, supersampling_factor,
                                                  kwargs_galsim_setup, kwargs_galsim_param)
    # generate galaxy
    image, psf_kernel, _ = get_galsim_image(*args, **kwargs)

    # import matplotlib.pyplot as plt
    # plt.imshow(image, origin='lower', cmap='gist_stern')
    # plt.show()

    # setup the 'INTERPOL' profile
    kwargs_interpol.update({
        'image': image,
        'scale': pixel_size_ss,
        'center_x': 0,  # performed by galsim
        'center_y': 0,  # performed by galsim
        'phi_G': 0,     # performed by galsim
        # Note that 'amp' should not be set here, it will be computed according to the magnitude
    })
    return kwargs_interpol

def _prepare_galsim(image_size, pixel_size, supersampling_factor, 
                    kwargs_galsim_setup, kwargs_galsim_param):
    # make sure the index is integer
    kwargs_galsim_param['catalog_index'] = int(kwargs_galsim_param['catalog_index'])
    kwargs_galsim_param_ = kwargs_galsim_param.copy()
    kwargs_galsim_setup_ = kwargs_galsim_setup.copy()
    # update pixel sizes and image sizes so the galsim resolution matches the lenstronomy one *after* supersampling
    image_size_ss = image_size * supersampling_factor
    pixel_size_ss = pixel_size / supersampling_factor
    # if 'psf_gaussian_fwhm' in kwargs_galsim_setup_:
    #     kwargs_galsim_setup_['psf_gaussian_fwhm'] /= supersampling_factor
    if 'psf_pixel_size' in kwargs_galsim_setup_:
        kwargs_galsim_setup_['psf_pixel_size'] /= supersampling_factor
    if 'psf_size' in kwargs_galsim_setup_:
        kwargs_galsim_setup_['psf_size'] /= supersampling_factor
    # galsim takes offset in pixel units instead of physical units
    if 'galsim_center_x' in kwargs_galsim_param_:
        kwargs_galsim_param_['galsim_center_x'] /= pixel_size
        kwargs_galsim_param_['galsim_center_y'] /= pixel_size
    # pack galsim settings
    args = (image_size_ss, pixel_size_ss)
    kwargs = util.merge_dicts(kwargs_galsim_setup_, kwargs_galsim_param_)
    return pixel_size_ss, args, kwargs
