import nltk
nltk.download('brown')
from nltk.corpus import brown
from nltk.corpus import stopwords
nltk.download('stopwords')
import gensim, logging


#tuto: https://rare-technologies.com/word2vec-tutorial/
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#tokenisation
corpus = brown.sents()
list_stopwords= set(stopwords.words('english'))
list_sentences_tok = []
for e in range(len(corpus)):
    list_to_tok = corpus[e]
    list_tok_words_in_sentence = []
    for i in range(len(list_to_tok)):
        word = list_to_tok[i]
        word = word.lower()
        if word.isdigit():
            print(word)
        else:
            if (word != ".") and (word != ",") and (word != "!") and (word != "?") and (word != ":") and (
                    word != "``") and (word != "\'\'") and (word != ";") and (word != "-") and (word != "--") and (
                    word != "(") and (word != ")") and (word != "\n") and (word != "\b"):
                list_tok_words_in_sentence.append(word)

    list_sentences_tok.append(list_tok_words_in_sentence)



#train word2vec
model = gensim.models.Word2Vec(list_sentences_tok, min_count=5, size=100, iter=10 )
model.save("word2vec_model_100n_spw")

#load and use Word2Vec for cosine similarities
new_model = gensim.models.Word2Vec.load('word2vec_model_100n_spw')
print(new_model.most_similar(positive=['do', 'seen'], negative=['see'], topn=10))