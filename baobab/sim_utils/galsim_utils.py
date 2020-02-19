import galsim
import numpy as np
from lenstronomy.Util import util

def get_galsim_image(image_size, pixel_size, catalog_index=0, 
                     scale=1, angle=0, center_x=0, center_y=0,
                     psf_size=49, psf_pixel_size=0.074, galaxy_type='real',
                     psf_type='real', psf_gaussian_fwhm=0.2, no_convolution=False,
                     draw_image_method='auto', catalog_dir=None, catalog_name=None):
    
    if catalog_dir is None:
        catalog_dir = '/Users/aymericg/Documents/EPFL/PhD_LASTRO/Code/divers/GalSim-releases-2.2/examples/data'
    
    if catalog_name is None:
        catalog_name = 'real_galaxy_catalog_23.5_example.fits'
        
    # effective pixel size
    pixel_size_eff = pixel_size / scale
        
    # dilate factor for the PSF wrt to original HST
    psf_dilate_factor = psf_pixel_size / 0.074  # taken for HST F814W band

    # Load catalog of galaxies
    cat = galsim.COSMOSCatalog(dir=catalog_dir, file_name=catalog_name)
    print("Number of galaxies in catalog '{}' : {}".format(catalog_name, cat.nobjects))
    
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
            psf = gal.original_psf.dilate(psf_dilate_factor).withFlux(1.)
            # Get the actual image of the psf
            psf_kernel = psf.drawImage(nx=psf_size, ny=psf_size, use_true_center=False, 
                                       scale=pixel_size_eff).array
        else:
            psf_kernel = np.zeros((image_size, image_size))
            psf_kernel_untouched = np.zeros((image_size, image_size))
    elif psf_type == 'gaussian':
        # Dilate the PSF to match required resolution
        psf = galsim.Gaussian(fwhm=psf_gaussian_fwhm, flux=1.0)
        # Get the actual image of the psf
        psf_kernel = psf.drawImage(nx=psf_size, ny=psf_size, use_true_center=False, 
                                   scale=pixel_size_eff).array
        psf_kernel_untouched = np.zeros((image_size, image_size))
    
    # apply rotation
    if angle != 0:
        gal = gal.rotate(angle * galsim.radians)
    
    # Performs convolution with PSF
    if no_convolution is False:
        if psf_type == 'real' and galaxy_type == 'parametric':
            print("Warning : no 'real' PSF convolution possible with gal_type 'parametric' !")
        else:
            gal = galsim.Convolve(gal, psf)
    
    # Project galaxy on an image grid
    image_galaxy = gal.drawImage(nx=image_size, ny=image_size, use_true_center=True,
                                 offset=[center_x, center_y],
                                 scale=pixel_size_eff, method=draw_image_method).array
    
    return image_galaxy, psf_kernel, psf_kernel_untouched


def kwargs_galsim2interpol(image_size, pixel_size, kwargs_galsim_setup, kwargs_galsim_param):
    kwargs_galsim_all = util.merge_dicts(kwargs_galsim_setup, kwargs_galsim_param)
    image, psf_kernel, _ = get_galsim_image(image_size, pixel_size, **kwargs_galsim_all)
    kwargs_interpol = {
        'image': image,
        'scale': pixel_size,
        'center_x': 0,  # performed by galsim
        'center_y': 0,  # performed by galsim
        'phi_G': 0,     # performed by galsim
    }
    return kwargs_interpol
