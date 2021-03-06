import numpy as np
from addict import Dict

cfg = Dict()

cfg.name = 'gamma'
cfg.seed = 1113 # random seed
cfg.bnn_prior_class = 'EmpiricalBNNPrior'
cfg.n_data = 200 # number of images to generate
cfg.train_vs_val = 'train'
cfg.components = ['lens_mass', 'external_shear', 'src_light']

cfg.selection = dict(
                 magnification=dict(
                                    min=2.0
                                    ),
                 initial=["lambda x: x['lens_mass']['theta_E'] > 0.5",]
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
           which_psf_maps=None, # None if rotate among all available PSF maps, else seed number of the map to generate all images with that map
           )

cfg.numerics = dict(
                supersampling_factor=1)

cfg.image = dict(
             num_pix=100, # cutout pixel size
             inverse=False, # if True, coord sys is ra to the left, if False, to the right 
             )

cfg.bnn_omega = dict(
                 # Inference hyperparameters defining the cosmology
                 cosmology = dict(
                                  H0=70.0, # Hubble constant at z = 0, in [km/sec/Mpc]
                                  Om0=0.3, # Omega matter: density of non-relativistic matter in units of the critical density at z=0.
                                  ),
                 redshift = dict(
                                model='differential_comoving_volume',
                                # Grid of redshift to sample from
                                grid = dict(
                                            start=0.01, # min redshift
                                            stop=5.0, # max redshift
                                            step=0.1, # resolution of the z_grid
                                            ),
                                ),

                 kinematics = dict(
                                   # Grid of velocity dispersion to sample from
                                   vel_disp = dict(
                                                  model = 'vel_disp_function_CPV2007', # one of ['vel_disp_function_CPV2007',] -- see docstring for details 
                                                  grid = dict(
                                                             start=100.0, # km/s
                                                             stop=400.0, # km/s
                                                             step=10.0, # km/s
                                                             ),
                                                  )
                                   ),
                 lens_mass = dict(
                                 profile='SPEMD', # only available type now
                                 # Normal(mu, sigma^2)
                                 center_x = dict(
                                          dist='normal', # one of ['normal', 'beta']
                                          mu=0.0,
                                          sigma=1.e-7,
                                          ),
                                 center_y = dict(
                                          dist='normal',
                                          mu=0.0,
                                          sigma=1.e-7,
                                          ),
                                 gamma = dict(
                                              model='FundamentalMassHyperplane',
                                              model_kwargs = dict(
                                                                  fit_data='SLACS',
                                                                  ),
                                              ),
                                 theta_E = dict(
                                                model='approximate_theta_E_for_SIS',
                                                ),
                                 # Beta(a, b)
                                 e1 = dict(
                                           dist='beta',
                                           a=4.0,
                                           b=4.0,
                                           lower=-0.9,
                                           upper=0.9),
                                 e2 = dict(
                                           dist='beta',
                                           a=4.0,
                                           b=4.0,
                                           lower=-0.9,
                                           upper=0.9,),
                                 ),

                 external_shear = dict(
                                       profile='SHEAR_GAMMA_PSI',
                                       gamma_ext = dict(
                                                         dist='lognormal',
                                                         mu=-2.73, # See overleaf doc
                                                         sigma=1.05,
                                                         ),
                                       psi_ext = dict(
                                                     dist='generalized_normal',
                                                     mu=0.0,
                                                     alpha=0.5*np.pi,
                                                     p=10.0,
                                                     lower=-0.5*np.pi,
                                                     upper=0.5*np.pi
                                                     ),
                                       ),

                 lens_light = dict(
                                  profile='SERSIC_ELLIPSE', # only available type now
                                  # Centered at lens mass
                                  magnitude = dict(
                                                   model='FaberJackson',
                                                   model_kwargs = dict(
                                                                     fit_data='ETGs',
                                                                     ),
                                                   ),
                                  R_sersic = dict(
                                                  model='FundamentalPlane',
                                                  model_kwargs = dict(
                                                                    fit_data='SDSS',),
                                                  ),
                                  n_sersic = dict(
                                                  dist='normal',
                                                  mu=1.25,
                                                  sigma=0.13,
                                                  ),
                                  # axis ratio
                                  q = dict(
                                           model='AxisRatioRayleigh',
                                           model_kwargs = dict(
                                                             fit_data='SDSS'
                                                             ),
                                           ),
                                  # ellipticity angle
                                  phi = dict(
                                             dist='generalized_normal',
                                             mu=np.pi,
                                             alpha=np.pi,
                                             p=10.0,
                                             lower=0.0,
                                             upper=2.0*np.pi,
                                             ),
                                  ),

                 src_light = dict(
                                profile='SERSIC_ELLIPSE', # only available type now 
                                magnitude = dict(
                                                 model='redshift_binned_luminosity_function',
                                                 ),
                                n_sersic = dict(
                                                dist='normal',
                                                mu=0.7,
                                                sigma=0.4,
                                                ),
                                # Normal(mu, sigma^2)
                                center_x = dict(
                                         dist='generalized_normal',
                                         mu=0.0,
                                         alpha=0.03,
                                         p=10.0,
                                         ),
                                center_y = dict(
                                               dist='generalized_normal',
                                               mu=0.0,
                                               alpha=0.03,
                                               p=10.0,      
                                                ),
                                R_sersic = dict(
                                                model='size_from_luminosity_and_redshift_relation',
                                                ),
                                q = dict(
                                         dist='one_minus_rayleigh',
                                         scale=0.3,
                                         lower=0.2
                                         ),
                                phi = dict(
                                           dist='generalized_normal',
                                           mu=np.pi,
                                           alpha=np.pi,
                                           p=10.0,
                                           lower=0.0,
                                           upper=2.0*np.pi
                                           ),
                                ),

                 agn_light = dict(
                                 profile='LENSED_POSITION', # contains one of 'LENSED_POSITION' or 'SOURCE_POSITION'
                                 # Centered at host
                                 # Pre-magnification, image-plane amplitudes if 'LENSED_POSITION'
                                 magnitude = dict(
                                                 model='AGNLuminosityFunction',
                                                 model_kwargs = dict(
                                                                     M_grid=np.arange(-30.0, -19.0, 0.2).tolist(),
                                                                     fit_data='combined',
                                                                     ),
                                                 ),
                                 ),
                 )