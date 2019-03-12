import gensim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import distance

def get_list_of_sentences(data_file_path):
	documents, labels = get_docs_from_file(data_file_path)
	for doc in documents:
		yield gensim.utils.simple_preprocess(doc)
def train_model_and_save_kv(documents, word_vectors_file='word2vec.kv'):
	model = gensim.models.Word2Vec(list(documents), size=50, min_count=1, window=3, workers=5)
	model.train(documents, total_examples=len(documents),epochs=10)
	word_vectors = model.wv
	word_vectors.save(word_vectors_file)

def load_word_vecs(word_vectors_file='word2vec.kv'):
	return gensim.models.KeyedVectors.load(word_vectors_file, mmap='r')

def instances_from_wordvecs(word_vecs, list_of_terms):
	return np.array([wv.word_vec(t) for t in list_of_terms])

def display_words_embedding_2d(data, labels):
	x_coords = data[:, 0]
	y_coords = data[:, 1]
	plt.scatter(x_coords, y_coords)
	for label, x, y in zip(labels, x_coords, y_coords):
	    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
	plt.show()

def get_centroids(data, labels):
	kmeans = KMeans(n_clusters=2)  
	kmeans.fit(data)
	x_coords = data[:, 0]
	y_coords = data[:, 1]
	for centroid in kmeans.cluster_centers_:
		print(centroid)
		distances_list = [(distance.euclidean(centroid, data[i]), i) for i in range(len(data))]
		sorted_list_idx = [tup[1] for tup in sorted(distances_list)]
		for i in range(10):
			print(labels[sorted_list_idx[i]])
	for label, x, y in zip(labels, x_coords, y_coords):
	    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
	plt.scatter(x_coords, y_coords, c=kmeans.labels_, cmap='rainbow')  
	plt.show()

def print_most_similars_for_list_of_terms(wv, terms):
	for t in terms:
		print("+" + "-" * 30 + "+")
		print(t)
		for similar in wv.most_similar(positive=t):
			print(similar)
		print("+" + "-" * 30 + "+")
# documents = list(get_list_of_sentences('./agresividad_es_mx/agresividad_spanish.csv'))
# print("{} documents".format(len(documents)))
# train_model_and_save_kv(documents)

wv = load_word_vecs()
# terms = ['loca', 'hdp', 'putos', 'tu', 'pendejo', 'hijos', 'mamar', 'mierda', 'puto', 'puta']
terms = ['loca', 'hdp', 'putos', 'tu', 'pendejo', 'mamar', 'hijos', 'mierda', 'puto', 'puta', 'estoy', 'pendejos', 'son', 'pinche', 'hijo', 'mi', 'los', 'verga', 'chinguen', 'pinches', 'chinga', 'chingas', 'mundial', 'su', 'hondureños', 'lameculos', 'chingar', 'esos', 'gringos', 'soy', 'esta', 'vida', 'mil', 'rateros', 'bola', 'perra', 'sus', 'vale', 'luchona', 'vas', 'amo', 'está', 'holanda', 'chilenos', 'valer', 'argentinos', 'día', 'una', 'periodistas', 'vergazos', 'en', 'culero', 'asco', 'tengo', 'valgo', 'yo', 'mediocre', 'la', 'huevos', 'maricon', 'valiendo', 'putita', 'pri', 'mañana', 'quiero', 'saben', 'corruptos', 'tus', 'necesito', 'teresa', 'ratas', 'vete', 'políticos', 'andan', 'noche', 'vayan', 'ando', 'van', 'porque', 'noerapenal', 'jugadores', 'rata', 'imbéciles', 'valió', 'calcuta', 'putito', 'vuelve', 'alguien', 'sé', 'robar', 'siento', 'volviendo', 'maldito', 'mexicanos', 'reputa', 'pario', 'hoy', 'culeros', 'puedo', 'aparte', 'deja', 'ojalá', 'unos', 'se', 'simios', 'estos', 'eres', 'osorio', 'amor', 'les', 'árbitro', 'hermosa', 'chica', 'cobran', 'éstos', 'antoja', 'gerardo', 'defender', 'ratero', 'juegan', 'robaron', 'argentina', 'madre', 'todos', 'puro', 'escuela', 'tienen', 'messi', 'peña', 'mandó', 'las', 'asquerosa', 'quedaron', 'hocico', 'nacos']
X = instances_from_wordvecs(wv, terms)
print(X.shape)
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)
print(X_pca.shape)
get_centroids(X_pca, terms)
display_words_embedding_2d(X_pca, terms)


