# -*- coding: utf-8 -*-
"""Generating the training data.

This script generates the training data according to the config specifications.

Example
-------
To run this script, pass in the desired config file as argument::

    $ generate baobab/configs/tdlmc_diagonal_config.py --n_data 1000

"""

import os, sys
import random
import argparse
import gc
from types import SimpleNamespace
from tqdm import tqdm
import numpy as np
import pandas as pd
# Lenstronomy modules
import lenstronomy
print("Lenstronomy path being used: {:s}".format(lenstronomy.__path__[0]))
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.SimulationAPI.data_api import DataAPI
import lenstronomy.Util.util as util
# Baobab modules
from baobab.configs import BaobabConfig
import baobab.bnn_priors as bnn_priors
from baobab.sim_utils import instantiate_PSF_models, get_PSF_model, generate_image, Selection

def parse_args():
    """Parse command-line arguments

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--n_data', default=None, dest='n_data', type=int,
                        help='size of dataset to generate (overrides config file)')
    args = parser.parse_args()
    # sys.argv rerouting for setuptools entry point
    if args is None:
        args = SimpleNamespace()
        args.config = sys.argv[0]
        args.n_data = sys.argv[1]
    return args

def main():
    args = parse_args()
    cfg = BaobabConfig.from_file(args.config)
    if args.n_data is not None:
        cfg.n_data = args.n_data
    # Seed for reproducibility
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    # Create data directory
    save_dir = cfg.out_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("Destination folder path: {:s}".format(save_dir))
        print("Log path: {:s}".format(cfg.log_path))
        cfg.export_log()
    else:
        raise OSError("Destination folder already exists.")
    # Instantiate PSF models
    psf_models = instantiate_PSF_models(cfg.psf, cfg.instrument.pixel_scale)
    n_psf = len(psf_models)
    # Instantiate density models
    # in case of pixelated light profiles, e.g. from galsim galaxies
    pixel_source_bool = cfg.bnn_omega.src_light.profile in ['GALSIM']  # add other if necessary
    if pixel_source_bool is True:
        src_light_profile_eff = 'INTERPOL'
    else:
        src_light_profile_eff = cfg.bnn_omega.src_light.profile
    kwargs_model = dict(
                    lens_model_list=[cfg.bnn_omega.lens_mass.profile, cfg.bnn_omega.external_shear.profile],
                    source_light_model_list=[src_light_profile_eff],
                    )       
    lens_mass_model = LensModel(lens_model_list=kwargs_model['lens_model_list'])
    src_light_model = LightModel(light_model_list=kwargs_model['source_light_model_list'])
    lens_eq_solver = LensEquationSolver(lens_mass_model)
    lens_light_model = None
    ps_model = None                                     
    if 'lens_light' in cfg.components:
        kwargs_model['lens_light_model_list'] = [cfg.bnn_omega.lens_light.profile]
        lens_light_model = LightModel(light_model_list=kwargs_model['lens_light_model_list'])
    if 'agn_light' in cfg.components:
        kwargs_model['point_source_model_list'] = [cfg.bnn_omega.agn_light.profile]
        ps_model = PointSource(point_source_type_list=kwargs_model['point_source_model_list'], fixed_magnification_list=[False])
    # Instantiate Selection object
    selection = Selection(cfg.selection, cfg.components)
    if not pixel_source_bool:  # those have ellipticiy components
        selection.add_ellipticity_selections()
    # Initialize BNN prior
    bnn_prior = getattr(bnn_priors, cfg.bnn_prior_class)(cfg.bnn_omega, cfg.components, 
                                                         external=getattr(cfg, 'external', None))
    if pixel_source_bool:
        bnn_prior.setup_pixel_profiles(cfg.image, cfg.instrument, cfg.numerics)
    # Initialize empty metadata dataframe
    metadata = pd.DataFrame()
    metadata_path = os.path.join(save_dir, 'metadata.csv')
    current_idx = 0 # running idx of dataset
    pbar = tqdm(total=cfg.n_data)
    while current_idx < cfg.n_data:
        sample = bnn_prior.sample() # FIXME: sampling in batches
        # Selections on sampled parameters
        if selection.reject_initial(sample):
            continue
        psf_model = get_PSF_model(psf_models, n_psf, current_idx)
        # Instantiate the image maker data_api with detector and observation conditions 
        kwargs_detector = util.merge_dicts(cfg.instrument, cfg.bandpass, cfg.observation)
        kwargs_detector.update(seeing=cfg.psf.fwhm,
                               psf_type=cfg.psf.type,
                               kernel_point_source=psf_model,
                               background_noise=0.0)
        data_api = DataAPI(cfg.image.num_pix, **kwargs_detector)

        # Generate the image
        img, img_features = generate_image(sample, psf_model, data_api, lens_mass_model, src_light_model, lens_eq_solver, 
                                           cfg.instrument.pixel_scale, cfg.image.num_pix, cfg.components, cfg.numerics, 
                                           min_magnification=cfg.selection.magnification.min, 
                                           lens_light_model=lens_light_model, ps_model=ps_model,
                                           add_noise=False)
        
        if img is None: # couldn't make the magnification cut
            continue
        # Save image file
        img_filename = 'X_{0:07d}.npy'.format(current_idx)
        img_path = os.path.join(save_dir, img_filename)
        np.save(img_path, img)
        # Save labels
        meta = {}
        for comp in cfg.components:
            for param_name, param_value in sample[comp].items():
                meta['{:s}_{:s}'.format(comp, param_name)] = param_value
            
            # typically for pixelated profiles, originally sampled param can be added to metadata 
            if hasattr(bnn_prior, 'original_sample'):
                for param_name, param_value in bnn_prior.original_sample[comp].items():
                    if '{:s}_{:s}'.format(comp, param_name) not in meta:
                        meta['{:s}_{:s}'.format(comp, param_name)] = param_value  

        meta['src_light_amp'] = img_features['src_light_amp']
        meta['total_magnification'] = img_features['total_magnification']
        meta['img_filename'] = img_filename
        
        # In case of 'INTERPOL' models being used, delete their cache
        if pixel_source_bool is True:
            src_light_model.delete_interpol_caches()
            del meta['src_light_image']  # useless in a csv file

        metadata = metadata.append(meta, ignore_index=True)
        # Export metadata.csv for the first time
        if current_idx == 0:
            # Sort columns lexicographically
            metadata = metadata.reindex(sorted(metadata.columns), axis=1)
            # Export to csv
            metadata.to_csv(metadata_path, index=None)
            # Initialize empty dataframe for next checkpoint chunk
            metadata = pd.DataFrame()
            gc.collect()

        # Export metadata every checkpoint interval
        if (current_idx + 1)%cfg.checkpoint_interval == 0:
            # Export to csv
            metadata.to_csv(metadata_path, index=None, mode='a', header=None)
            # Initialize empty dataframe for next checkpoint chunk
            metadata = pd.DataFrame()
            gc.collect()

        # Update progress
        current_idx += 1
        pbar.update(1)
    
    # Export to csv
    metadata.to_csv(metadata_path, index=None, mode='a', header=None)
    pbar.close()
    
if __name__ == '__main__':
    main()