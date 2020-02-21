from addict import Dict
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from .base_bnn_prior import BaseBNNPrior

class RealSourcePrior(BaseBNNPrior):
    """BNN prior with independent parameters

    Note
    ----
    first test for pixelated realistic source training set

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
        lens_model_list = [bnn_omega.lens_mass.profile]
        self._lensModel = LensModel(lens_model_list)
        self._lensModelExt = LensModelExtensions(self._lensModel)

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
            param_value = self.sample_param(hyperparams)
            kwargs[comp][param_name] = param_value

        # We want the source to be close to the caustics
        margin = 1  # units of pixel size
        kwargs = self._ensure_center_in_caustics('src_light', kwargs, margin=margin, verbose=False)

        # Ext shear is defined wrt the lens center
        kwargs['external_shear']['ra_0'] = kwargs['lens_mass']['center_x']
        kwargs['external_shear']['dec_0'] = kwargs['lens_mass']['center_y']

        # Source pos is defined wrt the lens pos
        if 'galsim_center_x' in kwargs['src_light']:
            kwargs['src_light']['galsim_center_x'] += kwargs['lens_mass']['center_x']
            kwargs['src_light']['galsim_center_y'] += kwargs['lens_mass']['center_y']
        else:
            kwargs['src_light']['center_x'] += kwargs['lens_mass']['center_x']
            kwargs['src_light']['center_y'] += kwargs['lens_mass']['center_y']

        if 'lens_light' in self.components:
            # Lens light shares center with lens mass
            kwargs['lens_light']['center_x'] = kwargs['lens_mass']['center_x']
            kwargs['lens_light']['center_y'] = kwargs['lens_mass']['center_y']

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

        return kwargs

    def setup_pixel_profiles(self, image, instruments, numerics):
        self._pixel_image_size  = image.num_pix
        self._pixel_pixel_scale = instruments.pixel_scale
        # we want to take into account the supersampling factor, ensuring the best resolution on the pixel grid
        self._pixel_supersampling_factor = numerics.get('supersampling_factor', 1)

    def _kwargs_pixel2interpol(self, kwargs_pixel):
        """Converts sampled parameters destined to pixelated realistic light profiles
        into parameters supported by the interpolation scheme of lenstronomy ('INTERPOL' profile)
        """
        if not hasattr(self, '_pixel_image_size'):
            raise ValueError("Image size, pixel scale, etc. not provided, use self._pixel_image_size() to do so")
        kwargs_setup = self.external['src_light']
        if self.src_light.profile == 'GALSIM':
            from baobab.sim_utils import galsim_utils
            kwargs_interpol = galsim_utils.kwargs_galsim2interpol(self._pixel_image_size, self._pixel_pixel_scale, 
                                                                  self._pixel_supersampling_factor, kwargs_setup, kwargs_pixel)
        else:
            raise NotImplementedError("Pixelated light profiles other than 'GALSIM' not implemented")
        return kwargs_interpol

    def _ensure_center_in_caustics(self, comp, kwargs, margin=0, verbose=False):
        """
        Re-realize (galsim_center_x, galsim_center_y) of source light if it is too far from the caustics.
        For now the simple criterion is the rectangle defined by min/max coordinates of caustics.
        """
        if comp != 'src_light':
            return kwargs
        hyperparams_x = getattr(self, comp)['galsim_center_x'].copy()
        hyperparams_y = getattr(self, comp)['galsim_center_y'].copy()
        current_value_x = kwargs[comp]['galsim_center_x']
        current_value_y = kwargs[comp]['galsim_center_y']

        # compute caustics by ray-shooting critical lines
        kwargs_lens = [kwargs['lens_mass']]
        x_crit_list, y_crit_list = self._lensModelExt.critical_curve_tiling(kwargs_lens, compute_window=5, 
                                                                            start_scale=0.5, max_order=10)
        x_caustic_list, y_caustic_list = self._lensModel.ray_shooting(x_crit_list, y_crit_list, kwargs_lens) 
        
        # sample again if centroids is not inside the rectangle defined by caustics
        param_value_x, param_value_y = current_value_x, current_value_y
        while not (min(x_caustic_list)-margin <= param_value_x and param_value_x <= max(x_caustic_list)+margin and
                   min(y_caustic_list)-margin <= param_value_y and param_value_y <= max(y_caustic_list)+margin):
            param_value_x = self.sample_param(hyperparams_x)
            param_value_y = self.sample_param(hyperparams_y)
            kwargs[comp]['galsim_center_x'] = param_value_x
            kwargs[comp]['galsim_center_y'] = param_value_y
            if verbose:
                print("Center offset ({}, {}) replaced by ({}, {})".format(current_value_x, current_value_y, 
                                                                           param_value_x, param_value_y))
        return kwargs
