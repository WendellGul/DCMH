import h5py
import scipy.io as scio


def load_data(path):
    file = h5py.File(path)
    images = file['IAll'][:]
    labels = file['LAll'][:]
    tags = file['YAll'][:]

    file.close()
    return images, tags, labels


if __name__ == '__main__':
    data_path = 'data/vgg_net.mat'
    data = scio.loadmat(data_path)
    print(data['net']['normalization'])
