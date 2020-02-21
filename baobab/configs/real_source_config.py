import os
import numpy as np
from addict import Dict

base_path = '/Users/aymericg/Documents/EPFL/PhD_LASTRO/Code/Forked/baobab_forked/demo_test'

cfg = Dict()
cfg.name = 'real_source'
cfg.bnn_prior_class = 'RealSourcePrior'
cfg.out_dir = os.path.join(base_path, 'galsim_test')
cfg.seed = 213 # random seed
cfg.n_data = 50 # number of images to generate
cfg.train_vs_val = 'train'
cfg.components = ['lens_mass', 'external_shear', 'src_light']  #, 'lens_light']
cfg.checkpoint_interval = 1

cfg.selection = dict(
                 magnification=dict(
                                    min=2.0
                                    ),
                 initial=[] #["lambda x: x['lens_mass']['theta_E'] > 0.5",]
                 )

cfg.instrument = dict(
              pixel_scale=0.08, # scale (in arcseonds) of pixels
              ccd_gain=4.5, # electrons/ADU (analog-to-digital unit). A gain of 8 means that the camera digitizes the CCD signal so that each ADU corresponds to 8 photoelectrons.
              )

cfg.bandpass = dict(
                magnitude_zero_point=25.9463, # (effectively, the throuput) magnitude in which 1 count per second per arcsecond square is registered (in ADUs)
                )

cfg.observation = dict(
                  exposure_time=100.0, # exposure time per image (in seconds)
                  )

cfg.psf = dict(
           type='PIXEL', # string, type of PSF ('GAUSSIAN' and 'PIXEL' supported)
           kernel_size=91, # dimension of provided PSF kernel, only valid when profile='PIXEL'
           which_psf_maps=101, # None if rotate among all available PSF maps, else seed number of the map to generate all images with that map
           )

cfg.numerics = dict(
                supersampling_factor=3  # prevent 5th image at the center due to numerical issues with cuspy mass profiles
                )

cfg.image = dict(
             num_pix=99, # cutout pixel size
             inverse=False, # if True, coord sys is ra to the left, if False, to the right 
             )

# put here some general (fixed) settings that are passed to the mass/light profiles (when supported)
cfg.external = dict(
                    src_light = dict(
                            # the following parameters are meant to be used with 'GALSIM' profile
                            galaxy_type='real',
                            psf_type='gaussian',
                            psf_gaussian_fwhm=0.12,
                            # psf_pixel_size=0.074,  # only with psf_type='real'
                            # psf_size=49,  # only with psf_type='real'
                            no_convolution=False,
                            catalog_dir='/Users/aymericg/Documents/EPFL/PhD_LASTRO/Code/divers/GalSim-releases-2.2/examples/data',
                            catalog_name='real_galaxy_catalog_23.5_example.fits',
                            draw_image_method='auto',
                        )
                    )

cfg.bnn_omega = dict(
                 lens_mass = dict(
                                 profile='SPEMD',
                                 center_x = dict(
                                          dist='normal',
                                          mu=0.0,
                                          sigma=1.e-6,
                                          ),
                                 center_y = dict(
                                          dist='normal',
                                          mu=0.0,
                                          sigma=1.e-6,
                                          ),
                                 gamma = dict(
                                              dist='normal',
                                              mu=2,
                                              sigma=0.05,
                                              lower=1.6,
                                              upper=2.4
                                              ),
                                 theta_E = dict(
                                                dist='normal',
                                                mu=1.2,
                                                sigma=0.05,
                                                lower=0.8,
                                                upper=1.6
                                                ),
                                 e1 = dict(
                                          dist='beta',
                                          a=4.0,
                                          b=4.0,
                                          lower=-0.5,
                                          upper=0.5),
                                e2 = dict(
                                          dist='beta',
                                          a=4.0,
                                          b=4.0,
                                          lower=-0.5,
                                          upper=0.5),
                                 ),

                 external_shear = dict(
                                       profile='SHEAR_GAMMA_PSI',
                                       gamma_ext = dict(
                                                         dist='normal',
                                                         mu=0, # See overleaf doc
                                                         sigma=0.05,
                                                         ),
                                       psi_ext = dict(
                                                     dist='normal',
                                                     mu=0,
                                                     sigma=0.05,
                                                     lower=-np.pi,
                                                     upper=np.pi,
                                                     )
                                       ),

                 src_light = dict(
                                profile='GALSIM',

                                catalog_index = dict(
                                             dist='uniform',
                                             lower=0,
                                             upper=95,
                                             ),
                                magnitude = dict(
                                             dist='normal',
                                             mu=20.407,
                                             sigma=1,
                                             ),
                                galsim_scale = dict(
                                            dist='normal',
                                            mu=0.8,
                                            sigma=0.05,
                                            lower=0.6,
                                            upper=1,
                                            ),
                                galsim_angle = dict(
                                             dist='uniform',
                                             lower=-np.pi,
                                             upper=np.pi,
                                             ),
                                galsim_center_x = dict(
                                                dist='uniform',
                                                lower=-2,
                                                upper=2,
                                                ),
                                galsim_center_y = dict(
                                                dist='uniform',
                                                lower=-2,
                                                upper=2,
                                                ),
                                ),

                 )