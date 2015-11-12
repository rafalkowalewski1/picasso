"""
    picasso.io
    ~~~~~~~~~~

    General purpose library for handling input and output of files

    :author: Joerg Schnitzbauer, 2015
"""


import os.path as _ospath
import numpy as _np
import yaml as _yaml
import tifffile as _tifffile
import glob as _glob
import h5py as _h5py


class FileFormatNotSupported(Exception):
    pass


def load_raw(path, memory_map=True):
    info = load_raw_info(path)
    dtype = _np.dtype(info['Data Type'])
    if memory_map:
        movie = _np.memmap(path, dtype, 'r', shape=tuple(info['Shape']))
    else:
        movie = _np.fromfile(path, dtype)
        movie = _np.reshape(movie, tuple(info['Shape']))
    return movie, info


def load_raw_info(path):
    path_base, path_extension = _ospath.splitext(path)
    with open(path_base + '.yaml', 'r') as info_file:
        info = _yaml.load(info_file)
    return info


def load_tif(path):
    info = {}
    with _tifffile.TiffFile(path) as tif:
        movie = tif.asarray(memmap=True)
        info = {}
        info['Byte Order'] = tif.byteorder
        info['Data Type'] = _np.dtype(tif.pages[0].dtype).name
        info['File'] = tif.filename
        info['Shape'] = list(movie.shape)
        info['Frames'], info['Height'], info['Width'] = movie.shape
        try:
            info['Comments'] = tif.micromanager_metadata['comments']['Summary']
            info['Computer'] = tif.micromanager_metadata['summary']['ComputerName']
            info['Directory'] = tif.micromanager_metadata['summary']['Directory']
            micromanager_metadata = tif.pages[0].tags['micromanager_metadata'].value
            info['Camera'] = {'Manufacturer': micromanager_metadata['Camera']}
            if info['Camera']['Manufacturer'] == 'Andor':
                _, type, model, serial_number, _ = (_.strip() for _ in micromanager_metadata['Andor-Camera'].split('|'))
                info['Camera']['Type'] = type
                info['Camera']['Model'] = model
                info['Camera']['Serial Number'] = int(serial_number)
                info['EM RealGain'] = int(micromanager_metadata['Andor-Gain'])
                info['Pre-Amp Gain'] = int(micromanager_metadata['Andor-Pre-Amp-Gain'].split()[1])
                info['Readout Mode'] = micromanager_metadata['Andor-ReadoutMode']
            info['Excitation Wavelength'] = int(micromanager_metadata['TIFilterBlock1-Label'][-3:])
        except Exception as error:
            print('Exception in io.load_tif:')
            print(error)
    return movie, info


def _to_raw_single(path):
    path_base, path_extension = _ospath.splitext(path)
    path_extension = path_extension.lower()
    if path_extension in ['.tif', '.tiff']:
        movie, info = load_tif(path)
    else:
        raise FileFormatNotSupported("File format must be '.tif' or '.tiff'.")
    raw_file_name = path_base + '.raw'
    if info['Byte Order'] == '>':
        movie = movie.byteswap()    # Numpy default is little endian, Numba does not work with big endian
        info['Byte Order'] = '<'
    movie.tofile(raw_file_name)
    info['Generated by'] = 'Picasso ToRaw'
    info['Original File'] = info.pop('File')
    info['File'] = _ospath.basename(raw_file_name)
    with open(path_base + '.yaml', 'w') as info_file:
        _yaml.safe_dump(info, info_file)


def to_raw(path, verbose=True):
    paths = _glob.glob(path)
    n_files = len(paths)
    if n_files:
        for i, path in enumerate(paths):
            if verbose:
                print('Converting file {}/{}...'.format(i + 1, n_files), end='\r')
            _to_raw_single(path)
    else:
        if verbose:
            print('No files matching {}'.format(path))
    return n_files


def save_locs(path, locs, movie_info, identification_parameters):
    with _h5py.File(path, 'w') as locs_file:
        locs_dataset = locs_file.create_dataset('locs', locs.shape, dtype=locs.dtype)
        locs_dataset[...] = locs
    loc_info = identification_parameters
    loc_info['Generated by'] = 'Picasso Localize'
    base, ext = _ospath.splitext(path)
    if base.endswith('_locs'):
        base = base[:-5]
    with open(base + '_locs.yaml', 'w') as info_file:
        _yaml.dump(movie_info, info_file, indent=4)
        info_file.write('---\n')
        _yaml.dump(loc_info, info_file, default_flow_style=False)
