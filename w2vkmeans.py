import argparse
from logging import getLogger

import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib
from sklearn.decomposition import RandomizedPCA
from matplotlib import pyplot as pl
from itertools import cycle
import matplotlib.colors as colors




logger = getLogger(__name__)


def make_dataset(model):
    """Make dataset from pre-trained Word2Vec model.

    Paramters
    ---------
    model: gensim.models.word2vec.Word2Vec
        pre-traind Word2Vec model as gensim object.

    Returns
    -------
    numpy.ndarray((vocabrary size, vector size))
        Sikitlearn's X format.
    """
    V = model.index2word
    X = np.zeros((len(V), model.vector_size))

    for index, word in enumerate(V):
        X[index, :] += model[word]
    return X


def train(X, K):
    """Learn K-Means Clustering with MiniBatchKMeans.

    Paramters
    ---------
    X: numpy.ndarray((sample size, feature size))
        training dataset.
    K: int
        number of clusters to use MiniBatchKMeans.

    Returens
    --------
    sklearn.cluster.MiniBatchKMeans
        trained model.
    """
    logger.info('start to fiting KMeans with {} classs.'.format(K))
    classifier = MiniBatchKMeans(n_clusters=K, 
                        random_state=0,
                        init='k-means++', 
                        n_init=1,
                        init_size=1000,
                        batch_size=1000)
    classifier.fit(X)
    return classifier

def reduce_dems(X_train):
    rpca=RandomizedPCA(n_components=2)
    return rpca.fit_transform(X_train)

def plot(kmeans,reduced_data):
    kmeans.fit(reduced_data)
    colors_ = cycle(colors.cnames.keys())
    h = 0.1
    x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
    y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    pl.figure(1)
    pl.clf()
    pl.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    for this_centroid, k, col in zip(kmeans.cluster_centers_, range(len(kmeans.labels_)), colors_):
        mask = kmeans.labels_ == k
        pl.plot(reduced_data[mask, 0], reduced_data[mask, 1], 'w', markerfacecolor=col, marker='.')
        pl.plot(this_centroid[0], this_centroid[1], '+', markeredgecolor='k', markersize=5)
    pl.xlim(x_min, x_max)
    pl.ylim(y_min, y_max)
    pl.xticks(())
    pl.yticks(())
    pl.show()


    

def main():
    parser = argparse.ArgumentParser(
        description='Python Word2Vec Cluster')

    parser.add_argument('model',
                        action='store',
                        help='Name of word2vec binary modelfile.')

    parser.add_argument('-o', '--out',
                        action='store',
                        default='model.pkl',
                        help='Set output filename.')

    parser.add_argument('-k', '--K',
                        action='store',
                        type=int,
                        default=500,
                        help='Num of classes on KMeans.')

    parser.add_argument('-p', '--pre-trained-model',
                        action='store',
                        default=None,
                        help='Use pre-trained KMeans Model.')

    parser.add_argument('-w', '--words-to-pred',
                        action='store',
                        nargs='+',
                        type=str,
                        default=None,
                        help='List of word to predict.')

    args = parser.parse_args()

    model = Word2Vec.load(args.model)

    if not args.pre_trained_model:
        X = make_dataset(model)
        classifier = train(X, args.K)
        joblib.dump(classifier, args.out)
        reduced =  reduce_dems(X)
        plot(classifier, reduced)

    else:
        classifier = joblib.load(args.pre_trained_model)

    if args.words_to_pred:

        X = [model[word] for word in args.words_to_pred if word in model]
        classes = classifier.predict(X)

        result = []
        i = 0
        for word in args.words_to_pred:
            if word in model:
                result.append(str(classes[i]))
                i += 1
            else:
                result.append(str(-1))
        print(' '.join(result))


if __name__ == '__main__':
    main()