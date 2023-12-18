"""Routines to read catalogues."""

import numpy as np
from astropy.io import fits
import re
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astropy.table import Table

import moonstools.local as local

def read_fits(loc):
    """Read an entire fits file into a python dict."""
    data = {}
    with fits.open(loc) as f:
        n_columns = len(f[1].columns)
        for icol in range(n_columns):
            col_name = f[1].columns[icol].name
            data[col_name] = np.array(f[1].data[col_name])

    return data


def load_photo_table(field, gal_only=False):
    if field.lower() in ['xmm', 'xmm-lss']:
        loc = local.xmm_photometry_loc
    elif field.lower() == 'cdfs':
        loc = local.cdfs_photometry_loc
    else:
        raise ValueError(f"Unknown field '{field}'")
    photo = Table.read(loc, format='fits')
    photo.rename_column('zphot', 'Redshift')

    if gal_only:
        ind_gal = np.nonzero(photo['Stellar_locus_flag'] == 0)[0]
        photo = photo[ind_gal]

    return photo


def read_photometry(loc=None, field=None, full=False, print_columns=False):
    """Read the photometric catalogue for XMM-LSS or CDFS.

    Parameters
    ----------
    loc : string, optional
        File to read the catalogue from. If None (default), it is constructed
        for the specified `field`.
    field : string, optional
        Survey field for which to read the catalogue ("xmm" or "cdfs"). If
        None (default), `loc` must be specified.
    full : bool, optional
        Read the entire catalogue, rather than just the core data. Not yet
        implemented, and not clear if it is needed...
    print_columns : bool, optional
        Print the full list of columns (default: False)

    Returns
    -------
    photo : dict
        Dictionary with the loaded catalogue fields.
    """
    data = {}

    # Define the elements to read. The key is the name of the column in the
    # FITS file to read, its value the name of the key in the output dict.
    elements = {
        'RA': 'RA',
        'Dec': 'Dec',
        'zphot': 'zPhot',
        'VIDEO_Ks': 'Ks',
        'VIDEO_Ks_err': 'Ks_err',
        'Stellar_locus_flag': 'Star_Flag',
        'UVJ_U': 'U',
        'UVJ_V': 'V',
        'UVJ_J': 'J',
    }
    conversion = {
        'Star_Flag': int,    
    }

    if loc is None:
        if field.lower() in ['xmm', 'xmm-lss']:
            loc = local.xmm_photometry_loc
        elif field.lower() == 'cdfs':
            loc = local.cdfs_photometry_loc
        else:
            raise ValueError(f"Unknown field '{field}'")

    # Transcribe entries from the FITS file into the output dict
    with fits.open(loc) as f:
        if print_columns:
            print(f[1].columns)

        # Iterate over all columns and load the ones that are needed.
        # Slightly wasteful if we're not loading the full catalogue, but won't
        # make any performance difference in practice.      
        n_columns = len(f[1].columns)
        for icol in range(n_columns):
            col_name = f[1].columns[icol].name
            if full or col_name in elements:
                out_name = (
                    elements[col_name] if col_name in elements else col_name)

                data[out_name] = np.array(f[1].data[col_name])
                if out_name in conversion:
                    data[out_name] = (
                        data[out_name].astype(conversion[out_name]))

    return data


def match(ra1, dec1, ra2, dec2, z1=None, z2=None,
          max_sep_arcsec=1.0, max_dz=None):
    """Find the object in one catalogue that is closest to objects in another.

    For every object in the input catalogue, exactly one match from the 
    reference catalogue is found. This is either the one that is closest
    in position to the input object, or `-1` if no reference object exists
    within the maximum search radius.

    Parameters
    ----------
    ra1 : ndarray(float)
        RA values of objects in the input catalogue, in degrees.
    dec1 : ndarray(float)
        Dec values of objects in the input catalogue, in degrees.
    ra2 : ndarray(float)
        RA values of objects in the reference catalogue, in degrees.
    dec2 : ndarray(float)
        Dec values of objects in the reference catalogue, in degrees.
    z1 : ndarray(float)
        Redshift values of objects in input catalogue (optional)
    z2 : ndarray(float)
        Redshift values of objects in reference catalogue (optional)
    max_sep_arcsec : float, optional
        Maximum search radius around input objects in arcsec (default: 1).
    max_dz : float, optional
        Maximum redshift offset from target object. If None (default), no
        redshift cut is applied.

    Returns
    -------
    idx : ndarray(int)
        The indices in the reference catalogue for the best match to each
        object in the input catalogue. `-1` for objects that could not be
        matched.
    d2d : ndarray(float)
        The angular separation between each input object and its matched
        reference object. ?Units, what happens if no match?
    dz : ndarray(float), optional
        The redshift offset from the input to reference object. Only returned
        if max_dz is not None.
    """
    try:
        ra1 = ra1.value
    except AttributeError:
        pass
    try:
        ra2 = ra2.value
    except AttributeError:
        pass
    try:
        dec1 = dec1.value
    except AttributeError:
        pass
    try:
        dec2 = dec2.value
    except AttributeError:
        pass

    # Actual matching is done differently depending on whether or not we want
    # to limit matches to some interval in redshift...
    if max_dz is None:
        c1 = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)
        c2 = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)
        idx, d2d, d3d = c1.match_to_catalog_sky(c2)
        dz_all = None
    else:
        idx = np.zeros(len(ra1), dtype=int) - 1
        d2d = Angle(np.zeros(len(ra1)), unit='degree') - 1*u.degree
        dz_all = np.zeros(len(ra1)) - 1
        for ii1 in range(len(ra1)):
            c1 = SkyCoord(ra=ra1[ii1]*u.degree, dec=dec1[ii1]*u.degree)
            dz = z2 - z1[ii1]
            ind_z = np.nonzero(np.abs(dz) <= max_dz)[0]
            if len(ind_z) > 0:
                c2 = SkyCoord(ra=ra2[ind_z]*u.degree, dec=dec2[ind_z]*u.degree)            
                iidx, id2d, id3d = c1.match_to_catalog_sky(c2)
                idx[ii1] = ind_z[iidx]
                d2d[ii1] = id2d
                dz_all[ii1] = dz[ind_z[iidx]]

    max_sep = max_sep_arcsec * u.arcsec
    ind_far = np.nonzero(d2d > max_sep)[0]
    idx[ind_far] = -1

    if dz_all is None:
        return idx, d2d
    else:
        return idx, d2d, dz_all


def check_bijective(match_out, match_back):
    """Check which matches out of a provided set are bijective.

    A bijective match means that an object A1 in catalogue A is matched to
    an object B1 in catalogue B, B1 is also matched to A1 when the matching
    is done in reverse. This is not guaranteed, as B1 could have an even
    better match in A, e.g. A2. In this case, A1 is not bijectively matched,
    but B1 might be, as long as A2 is itself matched to B1.

    Parameters
    ----------
    match_out : ndarray(int)
        A list of matches from catalogue A to catalogue B. In other words,
        element i gives the object in catalogue B that is matched to object i
        in catalogue A.
    match_back : ndarray(int)
        A list of matches from catalogue B to catalogue A. In other words,
        element i gives the object in catalogue A that is matched to object i
        in catalogue B.

    Returns
    -------
    flag : ndarray(bool)
        Bijective match status for each object in catalogue A: `True` if it
        could be matched bijectively, `False` otherwise.
    """
    n_orig = len(match_out)
    flag_bijective = np.zeros(n_orig, dtype=bool)
    ind_bijective = np.nonzero(
        (match_back[match_out] == np.arange(n_orig)) &
        (match_out >= 0)
    )[0]
    flag_bijective[ind_bijective] = True
    print(f'{len(ind_bijective)} bijective matches out of {n_orig}.')
    return flag_bijective


def bijective_match(ra1, dec1, ra2, dec2, z1=None, z2=None,
                    max_sep_arcsec=1.0, max_dz=None):
    """Bijective matching between two catalogues."""
    # `match` returns 2 or 3 elements depending on whether we have a max_dz
    # constraint. Therefore we capture the result in a list first.
    match_12 = match(
        ra1, dec1, ra2, dec2, z1=z1, z2=z2,
        max_sep_arcsec=max_sep_arcsec, max_dz=max_dz
    )
    match_21 = match(
        ra2, dec2, ra1, dec1, z1=z2, z2=z1,
        max_sep_arcsec=max_sep_arcsec, max_dz=max_dz
    )
    idx_12, d2d_12 = match_12[0], match_12[1]
    idx_21, d2d_21 = match_21[0], match_21[1]

    flag_bijective_12 = check_bijective(idx_12, idx_21)
    flag_bijective_21 = check_bijective(idx_21, idx_12)
    
    out = {
        'Match': idx_12,
        'Distance': d2d_12.arcsec,
        'Bijective': flag_bijective_12
    }
    back = {
        'Match': idx_21,
        'Distance': d2d_21.arcsec,
        'Bijective': flag_bijective_21
    }
    if max_dz is not None:
        out['DeltaZ'] = match_12[2]
        back['DeltaZ'] = match_21[2]

    return out, back


def bijective_catalogue_match(cat1, cat2, max_sep_arcsec=1.0, max_dz=None):
    """Shorthand for bijective matching of two catalogues."""
    if 'Redshift' in cat1.columns and 'Redshift' in cat2.columns:
        return bijective_match(
            cat1['RA'], cat1['Dec'], cat2['RA'], cat2['Dec'],
            z1=cat1['Redshift'], z2=cat2['Redshift'],
            max_sep_arcsec=max_sep_arcsec, max_dz=max_dz
        )
    else:
        if max_dz is not None:
            print("*** WARNING ***\nmax_dz specified, but one of the input "
                  "catalogues does not contain redshifts. Ignoring max_dz!")
        return bijective_match(
            cat1['RA'], cat1['Dec'], cat2['RA'], cat2['Dec'],
            max_sep_arcsec=max_sep_arcsec, max_dz=None
        )

def get_frame(cat, indices=None, stretch=1.0):
    """Compute the frame holding a catalogue.

    Parameters
    ----------
    cat : Table
        The input catalogue. It must contain columns named 'RA' and 'Dec',
        which correspond to the sky coordinates in degrees.
    indices : ndarray(int), optional
        The sub-indices to be used, if not None.
    stretch: float
        Size of the returned frame relative to the maximum extent of the 
        catalogue. Default: 1.0, i.e. just fit the catalogue.

    Returns
    -------
    x_lim : array(2)
        Maximum and minimum RA coordinate. Note that the maximum is listed
        first, corresponding to standard astronomical orientation.
    y_lim : array(2)
        Maximum and minimum Dec coordinate.
    """
    # Truncate catalogue to desired indices, if appropriate. Note that this
    # generates an internal copy, it does *not* modify the input catalogue.
    if indices is not None:
        cat = cat[indices]

    # Compute desired extent
    xmin = cat['RA'].min()
    xmax = cat['RA'].max()
    ymin = cat['Dec'].min()
    ymax = cat['Dec'].max()
    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2
    dx = (xmax - xmin)
    dy = (ymax - ymin)
    x_lim = np.array([xmid + dx/2*stretch, xmid - dx/2*stretch])
    y_lim = np.array([ymid - dy/2*stretch, ymid + dy/2*stretch])

    return x_lim, y_lim


def clean_string_array(array):
    """Remove non-ASCII characters and trailing spaces from a string array."""
    for i, name in enumerate(array):
        if not name.isascii():
            print(f"Warning: non-ASCII name '{name}'")
        nn = remove_non_ascii(name)

        if nn[-1] == ' ':
            print(f"Warning: empty spaces at end of '{nn}'")
            while (nn[-1] == ' '):
                nn = nn[:-1]
                
        array[i] = nn


def remove_non_ascii(text):
    """Remove non-ASCII characters from a string."""
    return re.sub(r'[^\x00-\x7F]', '-', text)