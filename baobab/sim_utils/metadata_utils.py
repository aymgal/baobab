import fnmatch
import lenstronomy.Util.param_util as param_util
__all__ = ['add_qphi_columns', 'add_g1g2_columns', 'add_relative_src_offset']

def add_qphi_columns(metadata):
    """Add alternate ellipticity definitions (axis ratio and angle) for each component for which ellipticity was defined in terms of e1, e2

    Parameters
    ----------
    metadata : pd.DataFrame
        the metadatadata generated by Baobab

    Returns
    -------
    pd.DataFrame
        metadata augmented with e1, e2 for each relevant component

    """
    e1_col_names = sorted(fnmatch.filter(metadata.columns.values, '*_e1'))
    e2_col_names = sorted(fnmatch.filter(metadata.columns.values, '*_e2'))
    for i, e1_col_name in enumerate(e1_col_names):
        e2_col_name = e2_col_names[i]
        comp_name = e1_col_name.split('_e1')[0] # component name, e.g. 'lens_light'
        e1 = metadata[e1_col_name].values
        e2 = metadata[e2_col_name].values
        phi, q = param_util.ellipticity2phi_q(e1, e2)
        metadata['{:s}_q'.format(comp_name)] = q
        metadata['{:s}_phi'.format(comp_name)] = phi

def add_g1g2_columns(metadata):
    """Add alternate shear definitions (gamma1, gamma2) for external shear defined in terms of shear modulus and angle (gamma_ext, psi_ext)

    Parameters
    ----------
    metadata : pd.DataFrame
        the metadatadata generated by Baobab

    Returns
    -------
    pd.DataFrame
        metadata augmented with gamma1, gamma2 for the external shear component

    """
    gamma_ext = metadata['external_shear_gamma_ext'].values
    psi_ext = metadata['external_shear_psi_ext'].values
    gamma1, gamma2 = param_util.shear_polar2cartesian(phi=psi_ext, gamma=gamma_ext)
    metadata['external_shear_gamma1'] = gamma1
    metadata['external_shear_gamma2'] = gamma2
    return metadata

def add_relative_src_offset(metadata):
    """Get the source offset relative to the lens center

    Parameters
    ----------
    metadata : pd.DataFrame
        the metadata generated by Baobab

    Returns
    -------
    pd.DataFrame
        metadata augmented with relative source offset columns added

    """
    metadata['src_light_pos_offset_x'] = metadata['src_light_center_x'] - metadata['lens_mass_center_x']
    metadata['src_light_pos_offset_y'] = metadata['src_light_center_y'] - metadata['lens_mass_center_y']
    return metadata