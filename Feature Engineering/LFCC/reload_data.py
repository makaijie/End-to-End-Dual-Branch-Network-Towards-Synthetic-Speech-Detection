import pickle
from librosa.util import find_files
import scipy.io as sio

path_to_mat = 'LFCCFeatures_2019LA_mat/'
path_to_feature = 'LFCCFeatures/'

def reload_data(path_to_feature, part):
    matfiles = find_files(path_to_mat + part + '/' , ext='mat')
    for i in range(len(matfiles)):
        if matfiles[i][-21:].startswith('LFCC'):
            audioid = matfiles[i][-16:-4]
            print(audioid)
            lfcc = sio.loadmat(matfiles[i], verify_compressed_data_integrity=False)['x']
            with open(path_to_feature + part + '/' + audioid + 'LFCC.pkl', 'wb') as handle2:
                pickle.dump(lfcc, handle2, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    reload_data(path_to_feature, 'eval')