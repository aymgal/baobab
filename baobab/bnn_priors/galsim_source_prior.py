from addict import Dict
import lenstronomy.Util.param_util as param_util
from .base_bnn_prior import BaseBNNPrior
from baobab.sim_utils import galsim_utils

class GalsimSourcePrior(BaseBNNPrior):
    """BNN prior with independent parameters

    Note
    ----
    first test for Learnlet training set

    """
    def __init__(self, bnn_omega, components, external):
        """
        Note
        ----
        The dictionary attributes are copies of the config corresponding to each component.
        The number of attributes depends on the number of components.

        Attributes
        ----------
        components : list
            list of components, e.g. `lens_mass`
        lens_mass : dict
            profile type and parameters of the lens mass
        src_light : dict
            profile type and parameters of the source light
            
        """
        BaseBNNPrior.__init__(self, bnn_omega, components, external=external)
        self.params_to_exclude = []
        self.set_params_list(self.params_to_exclude)
        self.set_comps_qphi_to_e1e2()
        if 'src_light' in self.components and bnn_omega.src_light.profile in ['GALSIM']:  # can add other here
            self._src_light_is_pixel = True
        else:
            self._src_light_is_pixel = False

    def sample(self):
        """Gets kwargs of sampled parameters to be passed to lenstronomy

        Returns
        -------
        dict
            dictionary of config-specified components (e.g. lens mass), itself
            a dictionary of sampled parameters corresponding to the config-specified
            profile of that component

        """
        # Initialize nested dictionary of kwargs
        kwargs = Dict()

        # Realize samples
        for comp, param_name in self.params_to_realize:
            hyperparams = getattr(self, comp)[param_name].copy()
            kwargs[comp][param_name] = self.sample_param(hyperparams)

        # In case of pixelated light such as galsim galaxies, translate to 'interpol' profile of lenstronomy
        self.original_sample = kwargs.copy()  # save for public access
        if self._src_light_is_pixel:
            kwargs_interpol = self._kwargs_pixel2interpol(kwargs['src_light'])
            kwargs['src_light'] = kwargs_interpol

        # Convert any q, phi into e1, e2 as required by lenstronomy
        for comp in self.comps_qphi_to_e1e2: # e.g. 'lens_mass'
            q = kwargs[comp].pop('q')
            phi = kwargs[comp].pop('phi')
            e1, e2 = param_util.phi_q2_ellipticity(phi, q)
            kwargs[comp]['e1'] = e1
            kwargs[comp]['e2'] = e2

        # Source pos is defined wrt the lens pos
        kwargs['src_light']['center_x'] += kwargs['lens_mass']['center_x']
        kwargs['src_light']['center_y'] += kwargs['lens_mass']['center_y']

        # Ext shear is defined wrt the lens center
        kwargs['external_shear']['ra_0'] = kwargs['lens_mass']['center_x']
        kwargs['external_shear']['dec_0'] = kwargs['lens_mass']['center_y']
        
        if 'lens_light' in self.components:
            # Lens light shares center with lens mass
            kwargs['lens_light']['center_x'] = kwargs['lens_mass']['center_x']
            kwargs['lens_light']['center_y'] = kwargs['lens_mass']['center_y']
        return kwargs

    def setup_pixel_profiles(self, image, instruments, numerics):
        self._pixel_image_size  = image.num_pix
        self._pixel_pixel_scale = instruments.pixel_scale
        # we take into account the supersampling factor, ensuring the best resolution on the pixel grid
        self._pixel_pixel_scale /= numerics.get('supersampling_factor', 1)

    def _kwargs_pixel2interpol(self, kwargs_pixel):
        """Converts sampled parameters destined to pixelated realistic light profiles
        into parameters supported by the interpolation scheme of lenstronomy ('INTERPOL' profile)
        """
        if not hasattr(self, '_pixel_image_size'):
            raise ValueError("Image size, pixel scale, etc. not provided, use self._pixel_image_size() to do so")
        kwargs_setup = self.external['src_light']
        if self.src_light.profile == 'GALSIM':
            kwargs_interpol = galsim_utils.kwargs_galsim2interpol(self._pixel_image_size, self._pixel_pixel_scale, 
                                                                  kwargs_setup, kwargs_pixel)
        else:
            raise NotImplementedError("Pixelated light profiles other than 'GALSIM' not implemented")
        return kwargs_interpol
