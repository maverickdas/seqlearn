from gensim.models import KeyedVectors
# from gensim.test.utils import datapath
import os.path as osp
# import os


def assert_file_path(fpath):

    if fpath is not None:
        assert osp.isfile(fpath), ("ERROR: Invalid path '{}'.".format(fpath))


class Embedder(object):
    """
    Embedder class within 'utils'
        :param object:
    """
    def __init__(self, modelpath=None, modeltype='gensim'):

        self.models = ['gensim']
        if modeltype is not None:
            assert modeltype in self.models, (
                "ERROR: Invalid 'modeltype' '{}'.".format(modeltype))
            self.model = modeltype
        if modelpath is not None:
            assert osp.isfile(modelpath), (
                "ERROR: Invalid 'modelpath' '{}'.".format(modelpath))
            self.modelpath = modelpath
        self.vectors = None
        self.vocab = None
        self.loaded = False

    def load_from_path(self, path=None, load_to_mem=True):
        """
        loads Embedder model from path, either passed as parameter or
        specified already in 'modelpath' data member
            :param path: file path to saved binary model
            :param load_to_mem=True: if True, vector and vocab data members are defined
        """
        assert_file_path(path)
        if path is None:
            path = self.modelpath
        if self.model == 'gensim':
            wv_from_bin = KeyedVectors.load_word2vec_format(path, binary=True)
            if load_to_mem:
                self.vectors = wv_from_bin.wv
                self.vocab = wv_from_bin.wv.vocab.keys()
                self.loaded = True
            else:
                return True, wv_from_bin
        return True
