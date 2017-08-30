# -*- coding: utf8 -*-
import csv
from gensim.models.word2vec import Word2Vec
from sklearn.mixture.gmm import GMM, log_multivariate_normal_density
import numpy as np
import preprocess
from sklearn.externals import joblib

print_topics = True

# GMM settings
num_topics_set = [10, 50, 100, 200]
num_gmm_iterations = 100

# W2V settings
wlens = [11, 13, 15, 17]
num_dims_set = [50, 100, 200]
num_w2v_iterations = 5

print_topics = True

train_filename = '../Data/train-PROCESSED-FINAL.csv'
test_filename = '../Data/test-PROCESSED-FINAL.csv'
output_file_template = "/Volumes/GOFLEX/models/word2vec/{run_id}"

class TweetCSVIterator(object):
	"""Allows for iteration over tweets in a corpus directory."""

	def __init__(self, train_filename):
		# Load in train data
		self.train_filename = train_filename

	def __iter__(self):
		# Load in train data
		with open(self.train_filename, 'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter=',', quotechar='"')
			for row in reader:
				yield row[-1].split()

if __name__ == '__main__':
	# Load in test data
	test_tokens = []
	with open(test_filename, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='"')
		for row in reader:
			test_tokens += row[-1].split()

	# Iterate
	for num_topics in num_topics_set:
		print "#########"
		print "Topics: "+str(num_topics)
		print "#########"
		for window in wlens:
			print "++++++++"
			print "Window: "+str(window)
			print "++++++++"
			for num_dims in num_dims_set:
				print "----------"
				print "Dims: "+str(num_dims)
				print "----------"

				w2v_run_id = "w2v_K{topics}_W{window}_D{dims}.gensim".format(
					topics=num_topics, window=window, dims=num_dims)
				w2v_output_file = output_file_template.format(run_id=w2v_run_id)
				print w2v_run_id

				gmm_run_id = "gmm_K{topics}_W{window}_D{dims}.pkl".format(
					topics=num_topics, window=window, dims=num_dims)
				gmm_output_file = output_file_template.format(run_id=gmm_run_id)
				print gmm_run_id

				# Create the w2v model
				print 'Starting W2V training'
				sentences = TweetCSVIterator(train_filename)
				w2v_model = Word2Vec(sentences, min_count=5, size=num_dims, workers=4, iter=num_w2v_iterations, window=window) #null_word=True
				# w2v_model.finalize_vocab()
				w2v_model.init_sims(replace=True)
				# w2v_model.save(w2v_output_file)
				print 'Done W2V training'

				# Set up test vectors
				test_vectors = []
				unknown_words = {}
				for token in test_tokens:
					if token in w2v_model:
						test_vectors.append(w2v_model[token])
					elif token in unknown_words:
						test_vectors.append(unknown_words[token])
					else:
						unknown_vec = w2v_model.seeded_vector(token)
						unknown_words[token] = unknown_vec
						test_vectors.append(unknown_vec)

				# Train GMM
				print 'Starting GMM training'
				words = w2v_model.vocab.keys()
				word_vectors = w2v_model.syn0
				gmm_model = GMM(n_components=num_topics, n_iter=num_gmm_iterations, covariance_type='diag')
				gmm_model.fit(word_vectors)
				# joblib.dump(gmm_model, gmm_output_file)
				print 'Done GMM training'

				# Get the likelihood of each word vector under each Gaussian component
				scores = gmm_model.score(test_vectors)
				print scores
				ll = sum(scores)
				print "LL:   "+str(ll)

				# Print topics if desired
				if print_topics:
					log_probs = log_multivariate_normal_density(word_vectors, gmm_model.means_, gmm_model.covars_, gmm_model.covariance_type)
					print np.min(log_probs)
					_, num_col = log_probs.shape
					for col in xrange(num_col):
						top_n = 10
						log_component_probs = (log_probs[:,col]).T
						sorted_indexes = np.argsort(log_component_probs)[::-1][:top_n]
						ordered_word_probs = [(w2v_model.index2word[idx], log_component_probs[idx]) for idx in sorted_indexes]

						print '---'
						print "Topic {0}".format(col+1)
						print "Total prob:" + str(sum(log_component_probs))
						print ", ".join(["{w}: {p}".format(w=w, p=p) for w, p in ordered_word_probs])