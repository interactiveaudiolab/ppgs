from struct import pack

def sph_get_header_size(sphere_file_object):
    sphere_file_object.seek(0)
    assert sphere_file_object.readline() == b'NIST_1A\n'
    header_size = int(sphere_file_object.readline().decode('utf-8')[:-1])
    sphere_file_object.seek(0)
    return header_size

header_type_flag_mapping = {
    'i': int,
    'r': float,
    's': str
}

def sph_get_header(sphere_file_object, header_size):
    sphere_file_object.seek(16)
    header = sphere_file_object.read(header_size-16).decode('utf-8').split('\n')
    header = header[:header.index('end_head')]
    header = [header_item.split(' ') for header_item in header if header_item[0] != ';']
    return {h[0]: header_type_flag_mapping[h[1][1]](h[2]) for h in header}

def wav_make_header(sph_header):
    try:
        return pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            sph_header['sample_count'] * sph_header['sample_n_bytes'] + 36, #file size -8 = datasize + 36
            b'WAVE',
            b'fmt ',
            16, #fmt size
            1, #TODO fill in from header??? #pcm header
            sph_header['channel_count'],  #MONO vs STEREO
            sph_header['sample_rate'],  #sample rate
            sph_header['sample_rate'] * sph_header['sample_n_bytes'], #bytes per second
            sph_header['sample_n_bytes'], #block align (bytes per sample)
            sph_header['sample_n_bytes']*8, #bits per sample
            b'data',
            sph_header['sample_count'] * sph_header['sample_n_bytes'] #data size
        )
    except KeyError as e:
        raise KeyError('key not found in sph header:', e)

def sph_get_samples(sphere_file_object, sphere_header_size):
    sphere_file_object.seek(sphere_header_size)
    return sphere_file_object.read()

def pcm_sph_to_wav(sphere_file):
    with open(sphere_file, 'rb') as f:
        header_size = sph_get_header_size(f)
        header = sph_get_header(f, header_size)
        new_header = wav_make_header(header)
        samples = sph_get_samples(f, header_size)
        return new_header + samples