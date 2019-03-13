import gensim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import distance



def get_docs_from_file(file_path, content_pos=0, label_pos=1):
	docs = []
	labels = []
	with open(file_path, 'r') as file:
		for line in file.readlines():
			_ = line.strip().split(',')
			docs.append(_[content_pos])
			labels.append(_[label_pos])
	return docs, labels

def get_list_of_sentences(data_file_path):
	documents, labels = get_docs_from_file(data_file_path)
	for doc in documents:
		yield gensim.utils.simple_preprocess(doc, min_len=1)
def train_model_and_save_kv(documents, word_vectors_file='word2vec.kv'):
	model = gensim.models.Word2Vec(list(documents), size=50, min_count=1, window=3, workers=5)
	model.train(documents, total_examples=len(documents), epochs=10)
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
	kmeans = KMeans(n_clusters=4)  
	kmeans.fit(data)
	x_coords = data[:, 0]
	y_coords = data[:, 1]
	for centroid in kmeans.cluster_centers_:
		print(centroid)
		distances_list = [(distance.euclidean(centroid, data[i]), i) for i in range(len(data))]
		sorted_list_idx = [tup[1] for tup in sorted(distances_list)]
		plt.scatter(centroid[0], centroid[1], c='red')
	# plt.scatter  
		for i in range(10):
			print(labels[sorted_list_idx[i]])
	for label, x, y in zip(labels, x_coords, y_coords):
	    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
	plt.scatter(x_coords, y_coords, c=kmeans.labels_, cmap='Accent')
	plt.show()

def print_most_similars_for_list_of_terms(wv, terms):
	for t in terms:
		s =  ", ".join([similar[0] for similar in wv.most_similar(positive=t)])
		print(t + "; " + s)	
# documents = list(get_list_of_sentences('./davidson_agresividad_en/pre_processing_agres_en.csv'))
# print("{} documents".format(len(documents)))
# train_model_and_save_kv(documents)

wv = load_word_vecs()
# terms = ['loca', 'hdp', 'putos', 'tu', 'pendejo', 'hijos', 'mamar', 'mierda', 'puto', 'puta']
# terms = ['loca',  'hdp',  'putos',  'pendejo',  'hijos',  'mamar',  'mierda',  'puto',  'puta',  'pendejos',  'pinche',  'hijo',  'chinguen',  'verga',  'pinches',  'chinga',  'chingas',  'mundial',  'hondureños',  'lameculos',  'chingar',  'gringos',  'vida',  'mil',  'bola',  'rateros',  'perra',  'vas',  'amo',  'luchona',  'chilenos',  'argentinos',  'holanda',  'valer',  'vergazos',  'pri',  'vale',  'culero',  'valgo',  'mediocre',  'periodistas',  'valiendo',  'día',  'maricon',  'corruptos',  'huevos',  'teresa',  'asco',  'mañana',  'putita',  'saben',  'vete',  'políticos',  'andan',  'ratas',  'ando',  'noche',  'noerapenal',  'jugadores',  'rata',  'vayan',  'imbéciles',  'vuelve',  'putito',  'van',  'quiero',  'robar',  'volviendo',  'valió',  'siento',  'sé',  'maldito',  'mexicanos',  'calcuta',  'reputa',  'pario',  'necesito',  'hoy',  'aparte',  'deja',  'culeros',  'alguien',  'puedo',  'simios',  'osorio',  'árbitro',  'hermosa',  'chica',  'ojalá',  'robaron',  'ratero',  'juegan',  'defender',  'antoja',  'gerardo',  'cobran',  'éstos',  'machorra',  'argentina',  'cagan',  'mandó',  'asquerosa',  'closet',  'nacos',  'maricones',  'hocico',  'quedaron',  'amor',  'televisa',  'peña',  'escuela',  'lame',  'humano',  'ches',  'marrano',  'morder',  'cerebro',  'priistas', 'australia',  'puro',  'mandarte',  'cagada',  'semana',  'nieto',  'presidente',  'basura',  'culo',  'toca',  'semanas',  'cabrón',  'penal',  'vergazo',  'tarea',  'ustedes',  'dejen',  'selección',  'dos',  'nervios',  'aqui',  'hija',  'cárcel',  'calvo',  'traidor',  'arbitro',  'ladrones',  'anciano',  'traidores',  'ticos',  'largate',  'reputisima',  'dije',  'word',  'pueden',  'pedazo']
terms = ['faggot', 'nigger', 'bitch', 'white', 'niggers', 'faggots', 'fag', 'bitches', 'fags', 'spic', 'racist', 'fucking', 'queer', 'hoes', 'dyke', 'pussy', 'hoe', 'coon', 'jew', 'kill', 'nigga', 'whitey', 'hashtag', 'gay', 'bird', 'mention', 'black', 'beaner', 'people', 'charlie', 'cracker', 'wetback', 'coons', 'queers', 'chink', 'jews', 'youre', 'race', 'hate', 'trash', 'aint', 'homo', 'chinks', 'nig', 'stupid', 'retards', 'america', 'niggas', 'ass', 'birds', 'beaners', 'gook', 'wetbacks', 'savages', 'fuck', 'retard', 'trailer', 'panthers', 'radical', 'url', 'blacks', 'retarded', 'ugly', 'feminist', 'uncle', 'teabagger', 'folk', 'sand', 'kike', 'shut', 'got', 'hipster', 'ignorant', 'human', 'bad', 'ashy', 'towel', 'yoself', 'spics', 'lovers', 'filth', 'campus', 'babies', 'store', 'pundits', 'genos', 'roid', 'activity', 'witcho', 'faux', 'monkeys', 'liberty', 'donts', 'aryan', 'hebrew', 'maryland', 'breeds', 'lotto', 'lifestyle', 'darling', 'drakes', 'molester', 'phelps', 'homophobic', 'flattering', 'israel', 'fuckin', 'ho', 'yellow', 'border', 'americans', 'breed', 'toms', 'drafted', 'deported', 'disgust', 'brownies', 'whore', 'ultimate', 'state', 'marriage', 'color', 'basic', 'females']
print_most_similars_for_list_of_terms(wv, terms[:10])
# # terms = ['loca', 'hdp', 'putos', 'pendejo', 'hijos', 'mamar', 'mierda', 'puto', 'puta', 'pendejos', 'pinche', 'hijo', 'chinguen', 'verga', 'pinches', 'chinga', 'chingas', 'mundial', 'hondureños', 'lameculos', 'chingar', 'gringos', 'vida', 'mil', 'bola', 'rateros', 'perra', 'vas', 'amo', 'luchona', 'chilenos', 'argentinos', 'holanda', 'valer', 'vergazos', 'pri', 'vale', 'culero', 'valgo', 'mediocre', 'periodistas', 'valiendo', 'día', 'maricon', 'corruptos', 'huevos', 'teresa', 'asco', 'mañana', 'putita', 'saben', 'vete', 'políticos', 'andan', 'ratas', 'ando', 'noche', 'noerapenal', 'jugadores', 'rata', 'vayan', 'imbéciles', 'vuelve', 'putito', 'van', 'quiero', 't', 'robar', 'volviendo', 'valió', 'siento', 'sé', 'maldito', 'mexicanos', 'calcuta', 'reputa', 'pario', 'necesito', 'hoy', 'aparte', 'deja', 'culeros', 'alguien', 'puedo', 'simios', 'osorio', 'árbitro', 'hermosa', 'chica', 'ojalá', 'robaron', 'ratero', 'juegan', 'defender', 'antoja', 'gerardo', 'cobran', 'éstos', 'machorra', 'argentina', 'cagan', 'mandó', 'asquerosa', 'closet', 'nacos', 'maricones', 'hocico', 'quedaron']
X = instances_from_wordvecs(wv, terms[:100])
print(X.shape)
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)
print(X_pca.shape)
# get_centroids(X_pca, terms[:100])
display_words_embedding_2d(X_pca, terms)


