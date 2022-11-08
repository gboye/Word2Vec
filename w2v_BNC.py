import numpy as np
import gensim, logging
import operator
import scipy
import os
import xml.etree.ElementTree as et

#_______________________________________________ functions: compute model accuracy ___________________________________________________

def min_value (tupple_verb_occ, form):
    occ_inf = tupple_verb_occ[3]
    occ_p = tupple_verb_occ[4]
    occ_pp = tupple_verb_occ[5]
    min_occ_index = 0
    if form == "inf_p_pp":
        if occ_inf <= occ_p and occ_inf <= occ_pp:
            min_occ_index = 3
        if occ_p <= occ_inf and occ_p <= occ_pp:
            min_occ_index = 4
        if occ_pp <= occ_p and occ_pp <= occ_inf:
            min_occ_index = 5
    if form == "p_pp":
        if occ_p <= occ_pp:
            min_occ_index = 4
        if occ_pp <= occ_p:
            min_occ_index = 5
        return min_occ_index
    if form == "inf":
        min_occ_index = 3
    if form == "p":
        min_occ_index = 4
    if form == "pp":
        min_occ_index = 5

    return min_occ_index

def compute_cosine_similarity(list_predictor_positive, list_predictor_neg, homog_pos, nb_cosine_simi_sample):
    list_cosine_simi = []
    for e in range(len(list_predictor_positive)):
        predictor_pos = list_predictor_positive[e]
        predictor_neg = list_predictor_neg[e]
        cos_similiraty = new_model.most_similar(positive=[homog_pos, predictor_pos], negative=[predictor_neg], topn=nb_cosine_simi_sample)
        list_cosine_simi.append(cos_similiraty)

    return list_cosine_simi


# la fonction determine_nb_sample_cos_simi est inutile pour vous, elle m'a permis creuser des hypothèses d'optimisation de min côté 
def determine_nb_sample_cos_simi(upadate_simi_sample_double_if_true, value_sample_cos_similarity):
    nb_sample_similarity_updated = 0
    if upadate_simi_sample_double_if_true == "True":
        nb_sample_similarity_updated = value_sample_cos_similarity * 2
    else:
        nb_sample_similarity_updated = value_sample_cos_similarity

    return nb_sample_similarity_updated


def predict_form(cosine_similarities, len_list_predictors, nb_sample_cos_similarity, t_f_simi_sample_double):
    dict_cosine_similarities = {}
    result_prediction = 0
    nb_sample_similarity = determine_nb_sample_cos_simi(t_f_simi_sample_double, nb_sample_cos_similarity)
    for index in range(len_list_predictors):
        cosine_similarity_line = cosine_similarities[index]
        for i in range(nb_sample_similarity):
            one_cosine_similarity = cosine_similarity_line[i]
            str_verb_similarity = one_cosine_similarity[0]
            if str_verb_similarity in dict_cosine_similarities:
                dict_cosine_similarities[str_verb_similarity] = dict_cosine_similarities[str_verb_similarity] + 1
            else:
                dict_cosine_similarities[str_verb_similarity] = 1

    sorted_dic_simi = dict(sorted(dict_cosine_similarities.items(), key=operator.itemgetter(1), reverse=True))
    result_prediction = list(sorted_dic_simi.keys())[0]
    return result_prediction


def results(verb_to_predict_inf, dict_tupple_pred, pred_true_p, pred_true_pp):
    all_true_p = dict_tupple_pred[verb_to_predict_inf][0] + pred_true_p
    all_true_pp = dict_tupple_pred[verb_to_predict_inf][0] + pred_true_pp
    overall_success = all_true_p + all_true_pp
    pair_success = 0
    if pred_true_pp + pred_true_p == 2:
        pair_success = 1
    all_pair_success = dict_tupple_pred[verb_to_predict_inf][3] + pair_success
    updated_tupple = (all_true_p, all_true_pp, overall_success, all_pair_success)
    dict_tupple_pred[verb_to_predict_inf] = updated_tupple


def compute_accuracy_frequency(start_value, stop_value, step_value, dict_occ_tupple, dict_predict, form, print_true):
    for e in range(start_value, stop_value, step_value):
        print("______________________",e , " < Min occurence < ", e + step_value, "____________________________")
        sum_true_pred = 0
        nb_verbs_with_occ = 0
        for verb in dict_occ_tupple:
            index_min_occ = min_value(dict_occ_tupple[verb], form)
            if dict_occ_tupple[verb][index_min_occ] <= e + step_value:
                if dict_occ_tupple[verb][index_min_occ] > e :
                    sum_true_pred = sum_true_pred + dict_predict[verb][3]
                    nb_verbs_with_occ = nb_verbs_with_occ + 1
                    if print_true == "True":
                        print("Numbers of true pairs predicted for ", verb, ": ", dict_predict[verb][3])

        if nb_verbs_with_occ > 0:
            moy_true_pred = sum_true_pred / (nb_iter * nb_verbs_with_occ)
            print("Average true predicted pairs: ", moy_true_pred)



#___________________________________________________ hyper parameters __________________________________________________
# nb_iter = nombre d'itération de l'algorithme de prédiction
nb_iter = 40
# nombre de similarités cosinus retenues pour chaque verbe prédicteur qui serviront à la prédiction
nb_cosine_similarity_sample = 3

#hyper-parameters word2vec (see doc gensim). Ces variables modifient directement les paramètres du modéle word2vec (pas besoin d'aller fouiller dans le code)
min_frequency = 50
neurons = 200
window = 7
negative_sampling = 15
epochs = 10

#____________________________________________________ List verb samples __________________________________________________
#verbes dont on cherche à calculer les vecteurs de formes opposées (ils servent également de verbes prédicteurs)


list_all_verb_homog_inf = ["like", "apply", "emerge", "gain", "understand", "realize", "seek", "achieve", "approach", "love", "extend", "establish", "recognise", "miss", "acquire", "indicate", "point", "observe", "imagine", "attack", "attempt", "destroy", "stress", "accept", "drop", "marry", "argue", "earn", "demonstrate", "exist", "arrange", "collect", "prove", "note", "declare", "mention", "succeed", "discover", "deny", "form", "play", "return", "decide", "plan", "discuss", "launch", "inform", "disappear", "learn", "shoot", "claim", "arrive", "occur", "allow", "attend", "reject", "receive", "sign", "reach", "publish", "report", "present", "manage", "explain", "visit", "die", "advise", "repeat", "let", "seize", "lift", "check", "cross", "warn", "urge", "accuse", "save", "hope", "remind", "propose", "confirm", "promise", "expect", "teach", "replace", "regard", "ignore", "fight", "recommend", "issue", "sweep", "hang", "switch", "remove", "gather", "serve", "greet", "press", "lay", "prefer", "knock", "act", "force", "score", "back", "reduce", "prompt",
                    "develop", "feel", "bring", "show", "grow", "know", "see", "have_inf", "do_inf", "put", 'move', 'turn', 'set', 'tell', 'find', 'get', 'look', 'use',
                   'say', 'cut', 'hold', "make", "ask", "want", "call", "run", "need", "believe",
                   "happen", "remember", "produce", "help", "spend", "kill", "follow", "try", "carry",
                   "send", "notice", "push", "open", "win", "read", "fail", "involve", "describe", "add", "study",
                   "pick", "think", "close", "work", "live", "loose", "catch", "hear", "keep", "buy",
                   "assume", "provide", "consider", "express", "require", "fill", "mean", "enter", "face", "meet",
                   "hit", "finish", "pass", "raise", "lay", "join", "offer", "pay", "continue", "head", "settle", "order",
                   "recognize", "enjoy", "include", "agree", "cause"]
list_all_verb_homog_p = ["likedp", "appliedp", "emergedp", "gainedp", "understoodp", "realizedp", "soughtp", "achievedp", "approachedp", "lovedp", "extendedp", "establishedp", "recognisedp", "missedp", "acquiredp", "indicatedp", "pointedp", "observedp", "imaginedp", "attackedp", "attemptedp", "destroyedp", "stressedp", "acceptedp", "droppedp", "marriedp", "arguedp", "earnedp", "demonstratedp", "existedp", "arrangedp", "collectedp", "provedp", "notedp", "declaredp", "mentionedp", "succeededp", "discoveredp", "deniedp", "formedp", "playedp", "returnedp", "decidedp", "plannedp", "discussedp", "launchedp", "informedp", "disappearedp", "learnedp", "shotp", "claimedp", "arrivedp", "occurredp", "allowedp", "attendedp", "rejectedp", "receivedp", "signedp", "reachedp", "publishedp", "reportedp", "presentedp", "managedp", "explainedp", "visitedp", "diedp", "advisedp", "repeatedp", "letp", "seizedp", "liftedp", "checkedp", "crossedp", "warnedp", "urgedp", "accusedp", "savedp", "hopedp", "remindedp", "proposedp", "confirmedp", "promisedp", "expectedp", "taughtp", "replacedp", "regardedp", "ignoredp", "foughtp", "recommendedp", "issuedp", "sweptp", "hungp", "switchedp", "removedp", "gatheredp", "servedp", "greetedp", "pressedp", "laidp", "preferredp", "knockedp", "actedp", "forcedp", "scoredp", "backedp", "reducedp", "promptedp",
                    "developedp", "feltp", "broughtp", "showedp", "grewp", "knewp", "sawp", "hadp", "didp", "putp", 'movedp', 'turnedp', 'setp', 'toldp', 'foundp', 'gotp',
                     'lookedp', 'usedp', 'saidp', 'cutp', 'heldp', "madep", "askedp", "wantedp", "calledp",
                     "ranp", "neededp", "believedp", "happenedp", "rememberedp", "producedp",
                     "helpedp", "spentp", "killedp", "followedp", "triedp", "carriedp", "sentp",
                     "noticedp", "pushedp", "openedp", "wonp", "readp", "failedp", "involvedp", "describedp",
                     "addedp", "studiedp", "pickedp", "thoughtp", "closedp", "workedp", "livedp", "lostp",
                     "caughtp", "heardp", "keptp", "boughtp", "assumedp", "providedp",
                     "consideredp", "expressedp", "requiredp", "filledp", "meantp", "enteredp", "facedp", "metp",
                     "hitp", "finishedp", "passedp", "raisedp", "laidp", "joinedp", "offeredp", "paidp", "continuedp", "headedp",
                     "settledp", "orderedp", "recognizedp", "enjoyedp", "includedp", "agreedp", "causedp"]
list_all_verb_homog_pp = ["likedpp", "appliedpp", "emergedpp", "gainedpp", "understoodpp", "realizedpp", "soughtpp", "achievedpp", "approachedpp", "lovedpp", "extendedpp", "establishedpp", "recognisedpp", "missedpp", "acquiredpp", "indicatedpp", "pointedpp", "observedpp", "imaginedpp", "attackedpp", "attemptedpp", "destroyedpp", "stressedpp", "acceptedpp", "droppedpp", "marriedpp", "arguedpp", "earnedpp", "demonstratedpp", "existedpp", "arrangedpp", "collectedpp", "provedpp", "notedpp", "declaredpp", "mentionedpp", "succeededpp", "discoveredpp", "deniedpp", "formedpp", "playedpp", "returnedpp", "decidedpp", "plannedpp", "discussedpp", "launchedpp", "informedpp", "disappearedpp", "learnedpp", "shotpp", "claimedpp", "arrivedpp", "occurredpp", "allowedpp", "attendedpp", "rejectedpp", "receivedpp", "signedpp", "reachedpp", "publishedpp", "reportedpp", "presentedpp", "managedpp", "explainedpp", "visitedpp", "diedpp", "advisedpp", "repeatedpp", "letpp", "seizedpp", "liftedpp", "checkedpp", "crossedpp", "warnedpp", "urgedpp", "accusedpp", "savedpp", "hopedpp", "remindedpp", "proposedpp", "confirmedpp", "promisedpp", "expectedpp", "taughtpp", "replacedpp", "regardedpp", "ignoredpp", "foughtpp", "recommendedpp", "issuedpp", "sweptpp", "hungpp", "switchedpp", "removedpp", "gatheredpp", "servedpp", "greetedpp", "pressedpp", "laidpp", "preferredpp", "knockedpp", "actedpp", "forcedpp", "scoredpp", "backedpp", "reducedpp", "promptedpp",
                      "developedpp", "feltpp", "broughtpp", "shownpp", "grownpp", "knownpp", "seenpp", "hadpp", "donepp", "putpp", 'movedpp', 'turnedpp', 'setpp', 'toldpp', 'foundpp',
                      'gotpp', 'lookedpp', 'usedpp', 'saidpp', 'cutpp', 'heldpp', "madepp", "askedpp", "wantedpp",
                      "calledpp", "runpp", "neededpp", "believedpp", "happenedpp",
                      "rememberedpp", "producedpp", "helpedpp", "spentpp", "killedpp", "followedpp",
                      "triedpp", "carriedpp", "sentpp", "noticedpp", "pushedpp", "openedpp", "wonpp", "readpp",
                      "failedpp", "involvedpp", "describedpp", "addedpp", "studiedpp", "pickedpp", "thoughtpp",
                      "closedpp", "workedpp", "livedpp", "lostpp", "caughtpp", "heardpp", "keptpp",
                      "boughtpp", "assumedpp", "providedpp", "consideredpp", "expressedpp", "requiredpp",
                      "filledpp", "meantpp", "enteredpp", "facedpp", "metpp", "hitpp", "finishedpp", "passedpp",
                      "raisedpp", "laidpp", "joinedpp", "offeredpp", "paidpp", "continuedpp", "headedpp", "settledpp",
                      "orderedpp", "recognizedpp", "enjoyedpp", "includedpp", "agreedpp", "causedpp"]


#_____________________________ Parse xml files from BNC (don't forget to copy the root to the folder in variable "folder_directory") _________________________


list_words_in_sentence = []
list_sentences = []
unc_removed = 0
folder_directory = 'C:/Users/Lavigne/PycharmProjects/pythonchanson/Texts/B'

for subdir, dirs, files in os.walk(folder_directory):
    for file in files:
        print(os.path.join(subdir, file))
        arbre = et.parse(os.path.join(subdir, file))
        root = arbre.getroot()

        for elem in arbre.iter():
            if elem.tag == 's':
                for w in elem.iter():
                    if w.tag == "w":
                        if w.attrib["pos"] == "UNC":
                            unc_removed += 1
                        else:
                            new_tupple = (w.text, w.attrib["c5"])
                            list_words_in_sentence.append(new_tupple)

                    elif w.tag == "c":
                        if w.text == "." or w.text == "?" or w.text == "!":
                            list_sentences.append(list_words_in_sentence)
                            list_words_in_sentence = []

print("UNC removed: " + str(unc_removed))



#_________________________________ tokenization ________________________________________
#remarque: je ne discrimine pas les formes infinitives (car j'aurai besoin d'ajouter le suffixe _inf à tous les verbes 
#de list_all_verb_homog_inf et j'ai la flemme, j'ai déjà discriminé les noms avec le suffixe _n et les formes p et pp ce qui limite les homographies). 
#Si vous n'avez pas la flemme, go ahead, c'est juste 3 lignes de code à ajouter et une liste à mofifier

list_sents_tok = []
count_words = 0
print_add_word = 0

for sentence in range(len(list_sentences)):
    word_sentence = list_sentences[sentence]
    list_raw_word_tag = []

    for word_in_sent in range(len(word_sentence)):
        word_tag = word_sentence[word_in_sent]
        list_raw_word_tag.append(word_tag)
        list_word_tok = []

    for z in range(len(list_raw_word_tag)):

        word_tag_check = list_raw_word_tag[z]
        word_check = word_tag_check[0]
        word_check = word_check.split()
        word_check = word_check[0]
        tag_check = word_tag_check[1].lower()
        word_check = word_check.lower()

        if (word_check != ".") and (word_check != ",") and (word_check != "!") and (word_check != "?") and (
                word_check != ":") and (word_check != "``") and (word_check != "\'\'") and (
                word_check != ";") and (
                word_check != "-") and (word_check != "--") and (word_check != "(") and (word_check != ")"):

            if (len(word_tag_check) <= 1):
                test = test + 1

            else:
                count_words += 1
                if (tag_check == "vvd"):
                    word_p = word_check + "p"
                    list_word_tok.append(word_p)

                elif (tag_check == "vdb"): #tag_check == "vdi" or
                    do = word_tag_check[0]
                    do_inf = "do" + "_inf"
                    list_word_tok.append(do_inf)

                elif ((tag_check == "vdd")):
                    didp = "didp"
                    list_word_tok.append(didp)

                elif ((tag_check == "vdn")):
                    donepp = "donepp"
                    list_word_tok.append(donepp)

                elif (tag_check == "vhb"): #tag_check == "vhi" or
                    have = word_tag_check[0]
                    have_inf = "have" + "_inf"
                    list_word_tok.append(have_inf)

                elif ((tag_check == "vhd")):
                    hadp = "hadp"
                    list_word_tok.append(hadp)

                elif ((tag_check == "vhn")):
                    hadpp = "hadpp"
                    list_word_tok.append(hadpp)

                elif ((tag_check == "nn0") or (tag_check == "nn1") or (tag_check == "nn2")):
                    noun_n = word_tag_check[0] + "_n"
                    list_word_tok.append(noun_n)

                elif (tag_check == "vvn"):
                    word_pp = word_check + "pp"
                    list_word_tok.append(word_pp)

                elif (tag_check == "vvb"):
                    word_inf = word_check
                    list_word_tok.append(word_inf)

                else:
                    list_word_tok.append(word_check)

    print_add_word += 1
    if print_add_word == 100:
        print("\rTokenized Words added:" + str(count_words), end="")
        print_add_word = 0
    list_sents_tok.append(list_word_tok)

print("\nTokenization process completed. Tokenized words from corpus: " + str(count_words))
print("___________________________________")


#________________________________________ compute verb samples frequency ___________________________________________________
#le tupple que Gilles avait demandé qu'on aurait pu associer à une classe lexique pour appliquer des fonctions (ce que j'en ai compris en tout cas)
dict_homog_occ_tupple = {}
dict_homog_predict_methode2 = {}
dict_homog_predict_methode1 = {}
dict_homog_predict_methode3 = {}
dict_homog_predict_methode3bis = {}
verb_added_to_dict = 0
for i in range(len(list_all_verb_homog_inf)):
    verb_form_inf = list_all_verb_homog_inf[i]
    verb_form_p = list_all_verb_homog_p[i]
    verb_form_pp = list_all_verb_homog_pp[i]
    count_occ_inf = 0
    count_occ_p = 0
    count_occ_pp = 0
    for sent in list_sents_tok:
        for word in sent:
            if word == verb_form_inf:
                count_occ_inf = count_occ_inf + 1
            if word == verb_form_p:
                count_occ_p = count_occ_p + 1
            if word == verb_form_pp:
                count_occ_pp = count_occ_pp + 1
    tupple_verb = (verb_form_inf, verb_form_p, verb_form_pp, count_occ_inf, count_occ_p, count_occ_pp)
    tupple_pred = (0, 0, 0, 0)
    dict_homog_occ_tupple[verb_form_inf] = tupple_verb
    dict_homog_predict_methode1[verb_form_inf] = tupple_pred
    dict_homog_predict_methode2[verb_form_inf] = tupple_pred
    dict_homog_predict_methode3[verb_form_inf] = tupple_pred
    dict_homog_predict_methode3bis[verb_form_inf] = tupple_pred
    verb_added_to_dict += 1
    print("\rVerb added in dict: " + str(verb_added_to_dict), end="")


#------------------ keep samples of verbs with frequency > min_ferquency for training and prediciton  -----------------
#Pour un corpus donné je ne garde dans la liste des verbes dont on va prédire les formes que ceux que ont une fréquence > n, n = min_frequency

list_verb_homog_inf = []
list_verb_homog_p = []
list_verb_homog_pp = []
list_predictors_inf = []
list_predictors_p = []
list_predictors_pp = []


for verb in list_all_verb_homog_inf:
    tupple_verb = dict_homog_occ_tupple[verb]
    if tupple_verb[3] > min_frequency and tupple_verb[4] > min_frequency and tupple_verb[5] > min_frequency:
        list_verb_homog_inf.append(tupple_verb[0])
        list_verb_homog_p.append(tupple_verb[1])
        list_verb_homog_pp.append(tupple_verb[2])
        list_predictors_inf.append(tupple_verb[0])
        list_predictors_p.append(tupple_verb[1])
        list_predictors_pp.append(tupple_verb[2])
        print("Verb added to list predicor / prediction: " + tupple_verb[0])





#_________________________________________________ training and predict form ________________________________________________________________________

list_acc_p_methode1 = []
list_acc_pp_methode1 = []
list_acc_p_methode2 = []
list_acc_pp_methode2 = []
list_results = []

for iter_w2v in range(nb_iter):

#_________________________________ Word2vec initialization and training __________________________________
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
     #train word2vec
    model = gensim.models.Word2Vec(list_sents_tok, min_count=min_frequency, size=neurons, iter=epochs, window=window, negative=negative_sampling)
    model.save("word2vec_model_100n_spw_homog2")
    # load and use Word2Vec for cosine similarities
    new_model = gensim.models.Word2Vec.load('word2vec_model_100n_spw_homog2')


#________________________________________ predictive algorithm _________________________________________
# ----------------------------- methode 1 ------------------------------------
#------------------------------ predict p ----------------------------------
    list_similarity_p = []
    for homographic_pos in list_verb_homog_inf:
        simi = compute_cosine_similarity(list_predictors_p, list_predictors_inf, homographic_pos, nb_cosine_similarity_sample)
        list_similarity_p.append(simi)


    count_true_pred_p_methode1 = 0
    list_true_prediction_p_m1 = []
    for index in range(len(list_similarity_p)):
        prediction_true = 0
        one_cosine_similarity = list_similarity_p[index]
        predicted_form = predict_form(one_cosine_similarity, 72, nb_cosine_similarity_sample, "")

        if predicted_form == list_verb_homog_p[index]:
            prediction_true = 1
            count_true_pred_p_methode1 = count_true_pred_p_methode1 + 1

        list_true_prediction_p_m1.append(prediction_true)

    #_______________________________________ predict pp _________________________________________
    list_similarity_pp = []
    for homographic_pos in list_verb_homog_inf:
        simi = compute_cosine_similarity(list_predictors_pp, list_predictors_inf, homographic_pos, nb_cosine_similarity_sample)
        list_similarity_pp.append(simi)


    count_true_pred_pp_methode1 = 0
    list_true_prediction_pp = []
    for index in range(len(list_similarity_pp)):
        prediction_true = 0
        one_cosine_similarity = list_similarity_pp[index]
        predicted_form = predict_form(one_cosine_similarity, 72, nb_cosine_similarity_sample, "")

        if predicted_form == list_verb_homog_pp[index]:
            prediction_true = 1
            count_true_pred_pp_methode1 = count_true_pred_pp_methode1 + 1

        list_true_prediction_pp.append(prediction_true)

#-------------------------------- Results methode 1 for this iteration -----------------------------------

    pairs_true_methode1 = 0
    for index in range(len(list_true_prediction_pp)):
        true_pred_p = list_true_prediction_p_m1[index]
        true_pred_pp = list_true_prediction_pp[index]
        if true_pred_p == 1 and true_pred_pp == 1:
            pairs_true_methode1 = pairs_true_methode1 + 1


    for index in range(len(list_verb_homog_inf)):
        verb_inf_to_predict = list_verb_homog_inf[index]
        pred_true_p = list_true_prediction_p_m1[index]
        pred_true_pp = list_true_prediction_pp[index]
        results(verb_inf_to_predict, dict_homog_predict_methode1, pred_true_p, pred_true_pp)


    number_pred_computed = len(list_true_prediction_pp)
    prediction_success_p_methode1 = count_true_pred_p_methode1
    prediction_success_pp_methode1 = count_true_pred_pp_methode1

    pred_acc_p_methode1 = prediction_success_p_methode1 / number_pred_computed * 100
    pred_acc_pp_methode1 = prediction_success_pp_methode1 / number_pred_computed * 100
    overall_acc_pred_methode1 = (prediction_success_p_methode1 + prediction_success_pp_methode1) / (2 * number_pred_computed) * 100
    Pairs_accuracy_methode1 = pairs_true_methode1 / number_pred_computed * 100

# ----------------------------- methode 2 ------------------------------------
#------------------------------ predict p ----------------------------------

    simi_matrix = np.zeros((len(list_verb_homog_inf), len(list_predictors_inf)))

    list_similarity_p = []
    for index_homographic_pos in range(len(list_verb_homog_pp)):
        homographic_pos = list_verb_homog_pp[index_homographic_pos]
        homographic_to_predict = list_verb_homog_p[index_homographic_pos]
        simi = compute_cosine_similarity(list_predictors_p, list_predictors_pp, homographic_pos, nb_cosine_similarity_sample)
        list_similarity_p.append(simi)

    count_true_pred_p_methode2 = 0
    list_true_prediction_p = []
    for index in range(len(list_similarity_p)):
        prediction_true = 0
        one_cosine_similarity = list_similarity_p[index]
        predicted_form = predict_form(one_cosine_similarity, 72, nb_cosine_similarity_sample, "")

        if predicted_form == list_verb_homog_p[index]:
            prediction_true = 1
            count_true_pred_p_methode2 = count_true_pred_p_methode2 + 1

        list_true_prediction_p.append(prediction_true)

    #_______________________________________ predict pp _________________________________________
    list_similarity_pp = []
    for homographic_pos in list_verb_homog_p:
        simi = compute_cosine_similarity(list_predictors_pp, list_predictors_p, homographic_pos, nb_cosine_similarity_sample)
        list_similarity_pp.append(simi)


    count_true_pred_pp_methode2 = 0
    list_true_prediction_pp = []
    for index in range(len(list_similarity_pp)):
        prediction_true = 0
        one_cosine_similarity = list_similarity_pp[index]
        predicted_form = predict_form(one_cosine_similarity, 72, nb_cosine_similarity_sample, "")


        if predicted_form == list_verb_homog_pp[index]:
            prediction_true = 1
            count_true_pred_pp_methode2 = count_true_pred_pp_methode2 + 1

        list_true_prediction_pp.append(prediction_true)

    #-------------------------------- Results methode 2 for this iteration -----------------------------------

    pairs_true_methode2 = 0
    for index in range(len(list_true_prediction_pp)):
        true_pred_p = list_true_prediction_p[index]
        true_pred_pp = list_true_prediction_pp[index]
        if true_pred_p == 1 and true_pred_pp == 1:
            pairs_true_methode2 = pairs_true_methode2 + 1


    number_pred_computed = len(list_true_prediction_pp)
    prediction_success_p_methode2 = count_true_pred_p_methode2
    prediction_success_pp_methode2 = count_true_pred_pp_methode2

    pred_acc_p_methode2 = prediction_success_p_methode2 / number_pred_computed * 100
    pred_acc_pp_methode2 = prediction_success_pp_methode2 / number_pred_computed * 100
    overall_acc_pred_methode2 = (prediction_success_p_methode2 + prediction_success_pp_methode2) / (2 * number_pred_computed) * 100
    Pairs_accuracy_methode2 = pairs_true_methode2 / number_pred_computed * 100

    for index in range(len(list_verb_homog_inf)):
        verb_inf_to_predict = list_verb_homog_inf[index]
        pred_true_p = list_true_prediction_p[index]
        pred_true_pp = list_true_prediction_pp[index]
        results(verb_inf_to_predict, dict_homog_predict_methode2, pred_true_p, pred_true_pp)



#_______________________________________________ Print results for both methods _________________________________________________
    print("------------------------ Acuracy methode 1 -------------------------")
    str_result_methode1 = "Accuracy predicting preterit: ", pred_acc_p_methode1, "% ", "Accuracy predicting past participle: ", pred_acc_pp_methode1, "% ", "Total accurancy: ", str(
        overall_acc_pred_methode1), "%" + "Pairs corectly predicted :", Pairs_accuracy_methode1, "%"
    list_results.append(str_result_methode1)
    list_acc_p_methode1.append(pred_acc_p_methode1)
    list_acc_pp_methode1.append(pred_acc_pp_methode1)
    print(str_result_methode1)

    print("------------------------ Acuracy methode 2 -------------------------")
    str_result_methode2 = "Accuracy predicting preterit: ", pred_acc_p_methode2, "% ", "Accuracy predicting past participle: ", pred_acc_pp_methode2, "% ", "Total accurancy: ", str(overall_acc_pred_methode2), "%" + "Pairs corectly predicted :", Pairs_accuracy_methode2, "%"
    list_results.append(str_result_methode2)
    list_acc_p_methode2.append(pred_acc_p_methode2)
    list_acc_pp_methode2.append(pred_acc_pp_methode2)
    print(str_result_methode2)


    print("Iteration: " + str(iter_w2v))





# ____________________________ average true prediction per frequency range for all iterations _________________________________


#Ici possibilité de modifier les plages de fréquences retenues pour calculer la précision des prédictions de l'algorithme par tranche de fréquence
#instruction: print les verbes par tranche de fréquence, le nombre de prédiction exacte pour chaque verbe, la précision moyenne des prédictions par tranche de fréquence.

#les paramêtres de la fonction que vous pouvez modifier: (start_value, stop_value, step_value, dict_occ_tupple, dict_predict, form, print_true)

#start_value: valeur minimale de l'itération par tranche de fréquence
#stop_value: valeur maximale de l'itération par tranche de fréquence
#step_value: taille de la tranche de fréquence 
#ex: start_value = 10, stop_value = 25, step_value = 5: l'algorithme vous retourne les tranches de fréquence: 11-15, 16-20, 21-25 
#(la borne minimale de l'intervalle de fréquence est exclue sinon un verbe pourrait se trouver dans 2 tranches de fréquences simultanément)

#dict_occ_tupple, dict_predict, ces deux paramêtres dépendent de la méthode que vous voulez tester, 
#ils ne sont pas à modifier dans la mesure où les 2 méthodes sont déjà présentes

#form: choisir sur quelle forme est calculée la fréquence minimale (par exemple les fréquences minimales pour les formes prétérit form="p"
#les fréquences minimales parmi prétérit et participe passé confondues form="p_pp" ect...) Paramètres possibles: "p", "pp", "inf", "p_pp", "inf_p_pp"

#print_true: ce paramêtre est une bidouille dégueulasse pour print la moyenne sur l'ensemble tranches sans re-print tous les verbes. 
#Dégueulasse mais fontionnel. Je sais comment améliorer ça mais pas le temps + flemme. 

print("----------------- accuracy per frequency methode 1 ----------------------------")
compute_accuracy_frequency(50, 200, 50, dict_homog_occ_tupple, dict_homog_predict_methode1, "inf_p_pp", "True")
compute_accuracy_frequency(200, 500, 100, dict_homog_occ_tupple, dict_homog_predict_methode1,"inf_p_pp", "True")
compute_accuracy_frequency(500, 900, 200, dict_homog_occ_tupple, dict_homog_predict_methode1, "inf_p_pp", "True")
compute_accuracy_frequency(900, 1900, 222222, dict_homog_occ_tupple, dict_homog_predict_methode1, "inf_p_pp", "True")
compute_accuracy_frequency(0, 11000, 11111, dict_homog_occ_tupple, dict_homog_predict_methode1, "inf_p_pp", "False")



print("----------------- accuracy per frequency methode 2 ----------------------------")
compute_accuracy_frequency(50, 200, 50, dict_homog_occ_tupple, dict_homog_predict_methode2, "p_pp", "True")
compute_accuracy_frequency(200, 500, 100, dict_homog_occ_tupple, dict_homog_predict_methode2,"p_pp", "True")
compute_accuracy_frequency(500, 700, 200, dict_homog_occ_tupple, dict_homog_predict_methode2, "p_pp", "True")
compute_accuracy_frequency(700, 1200, 500, dict_homog_occ_tupple, dict_homog_predict_methode2, "p_pp", "True")
compute_accuracy_frequency(1200, 10000, 222222, dict_homog_occ_tupple, dict_homog_predict_methode2, "p_pp", "True")
compute_accuracy_frequency(0, 11000, 16000, dict_homog_occ_tupple, dict_homog_predict_methode2, "p_pp", "False")

