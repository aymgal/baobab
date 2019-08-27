import os, sys

name = 'gamma'
seed = 1113 # random seed
is_interim = True
n_data = 20 # number of images to generate
train_vs_val = 'train'
out_dir = os.path.join('out_data', '{:s}_{:s}_{:s}_seed{:d}'.format(name,
                                                                    train_vs_val,
                                                                    'interim' if is_interim else 'empirical',
                                                                    seed))
components = ['lens_mass', 'src_light',] #'lens_light', 'agn_light']

image = dict(
             sigma_bkg=0.05, 
             exposure_time=100.0, # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit),
             numPix=100, # cutout pixel size
             deltaPix=0.08, # arcsec/pix
             inverse=False, # if True, coord sys is ra to the left, if False, to the right 
             )

psf = dict(
           type='PIXEL', # one of ['gaussian', 'PIXEL', 'NONE']
           kernel_dir=os.path.join('in_data', 'psf_maps'), # only valid when type='PIXEL'
           kernel_size=91, # dimension of provided PSF kernel, only valid when type='PIXEL'
           fwhm=0.1, # only valid when type='gaussian'
           )

bnn_omega = dict(
                lens_mass = dict(
                                 type=['SPEMD', 'SHEAR_GAMMA_PSI'], # only available type now
                                 params=['center_x', 'center_y', 'gamma', 'theta_E', 'e1', 'e2'] + ['gamma_ext', 'psi_ext'],
                                 # Normal(mu, sigma^2)
                                 x = dict(
                                          mu=0.0,
                                          sigma=1.e-7),
                                 y = dict(
                                          mu=0.0,
                                          sigma=1.e-7),
                                 log_gamma = dict(
                                              mu=0.7,
                                              sigma=0.02),
                                 log_theta_E = dict(
                                                mu=0.0,
                                                sigma=0.1),
                                 # Beta(a, b)
                                 e1 = dict(
                                           a=4.0,
                                           b=4.0,
                                           min=-0.9,
                                           max=0.9),
                                 e2 = dict(
                                           a=4.0,
                                           b=4.0,
                                           min=-0.9,
                                           max=0.9,),
                                 gamma_ext1 = dict(
                                                   a=4.0,
                                                   b=4.0,
                                                   min=-0.2,
                                                   max=0.2),
                                 gamma_ext2 = dict(
                                                   a=4.0,
                                                   b=4.0,
                                                   min=-0.2,
                                                   max=0.2)
                                 ),

                src_light = dict(
                                type=['SERSIC_ELLIPSE'], # only available type now
                                params=['amp', 'center_x', 'center_y', 'n_sersic', 'R_sersic', 'e1', 'e2'],
                                # Normal(mu, sigma^2)
                                log_amp = dict(
                                        mu=5.0,
                                        sigma=0.3),
                                x = dict(
                                        mu=0.0,
                                        sigma=0.01),
                                y = dict(
                                        mu=0.0,
                                        sigma=0.01),
                                log_n_sersic = dict(
                                                  mu=1.1,
                                                  sigma=0.2),
                                log_r_eff = dict(
                                            mu=-0.7,
                                            sigma=0.4,),
                                # Beta(a, b)
                                e1 = dict(
                                          a=4.0,
                                          b=4.0,
                                          min=-0.9,
                                          max=0.9),
                                e2 = dict(
                                          a=4.0,
                                          b=4.0,
                                          min=-0.9,
                                          max=0.9),
                                ),
                )