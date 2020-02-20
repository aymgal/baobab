import os
import unittest
import numpy as np
import numpy.testing as npt
import baobab
from baobab.sim_utils import galsim_utils

class TestGalsimUtils(unittest.TestCase):
    """Tests for the metadata utils module used to convert between parameter definitions

    """
    def test_get_galsim_image(self):
        catalog_dir = os.path.join(os.path.dirname(baobab.__file__), 'in_data/galsim_data')
        catalog_name = 'real_galaxy_catalog_23.5_example.fits'
        image_size = 25
        pixel_size = 0.32
        psf_size = 11

        image, psf_kernel, psf_kernel_ori \
            = galsim_utils.get_galsim_image(image_size, pixel_size, catalog_index=0, 
                     galsim_scale=1, galsim_angle=0, galsim_center_x=0, galsim_center_y=0,
                     psf_size=psf_size, psf_pixel_size=0.074, galaxy_type='real',
                     psf_type='real', psf_gaussian_fwhm=0.2, no_convolution=False,
                     draw_image_method='auto', catalog_dir=catalog_dir, catalog_name=catalog_name,
                     verbose=False)
        assert image.shape == (image_size, image_size)
        assert psf_kernel.shape == (psf_size, psf_size)
        npt.assert_almost_equal(psf_kernel.sum(), 1.0, decimal=5)

        # when galaxy_type is 'parametric', no original PSF
        _, psf_kernel, psf_kernel_ori \
            = galsim_utils.get_galsim_image(image_size, pixel_size, catalog_index=0, 
                     galsim_scale=1, galsim_angle=0, galsim_center_x=0, galsim_center_y=0,
                     psf_size=psf_size, psf_pixel_size=0.074, galaxy_type='parametric',
                     psf_type='real', psf_gaussian_fwhm=0.2, no_convolution=False,
                     draw_image_method='auto', catalog_dir=catalog_dir, catalog_name=catalog_name,
                     verbose=False)
        assert not np.any(psf_kernel)  # only 0s
        assert not np.any(psf_kernel_ori)  # only 0s

        # but if psf_type is 'gaussian', that's ok
        _, psf_kernel, psf_kernel_ori \
            = galsim_utils.get_galsim_image(image_size, pixel_size, catalog_index=0, 
                     galsim_scale=1, galsim_angle=0, galsim_center_x=0, galsim_center_y=0,
                     psf_size=psf_size, psf_pixel_size=0.074, galaxy_type='parametric',
                     psf_type='gaussian', psf_gaussian_fwhm=0.2, no_convolution=False,
                     draw_image_method='auto', catalog_dir=catalog_dir, catalog_name=catalog_name,
                     verbose=False)
        npt.assert_almost_equal(psf_kernel.sum(), 1.0, decimal=5)
        assert not np.any(psf_kernel_ori)  # only 0s

    def test_kwargs_galsim2interpol(self):
        image_size = 25
        pixel_size = 0.32
        psf_size = 11
        kwargs_galsim_setup = {
            'galaxy_type': 'real',
            'psf_type': 'real',
            'psf_size': 11,
            'catalog_dir': os.path.join(os.path.dirname(baobab.__file__), 'in_data/galsim_data'),
            'catalog_name': 'real_galaxy_catalog_23.5_example.fits',
            'draw_image_method': 'auto',
            # ...
        }
        kwargs_galsim_param = {
            'magnitude': 21,
            'catalog_index': 0, 
            'galsim_scale': 1, 
            'galsim_angle': np.pi/2., 
            'galsim_center_x': 10, 
            'galsim_center_y': -10,
        }
        supersampling_factor = 1
        kwargs_interpol = galsim_utils.kwargs_galsim2interpol(image_size, pixel_size, supersampling_factor, 
                                                              kwargs_galsim_setup, kwargs_galsim_param)
        assert 'magnitude' in kwargs_interpol and not 'amp' in kwargs_interpol


if __name__ == '__main__':
    unittest.main()
    