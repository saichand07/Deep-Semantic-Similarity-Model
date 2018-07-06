import numpy as np
import tensorflow as tf
from keras import backend
from keras.layers import Activation, Input
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.convolutional import Convolution1D
from keras.layers.merge import concatenate, dot
from keras.models import Model
#from keras import optimizers

Tri_gram = 3 
Window_size = 3 
Total_Tri_grams = int(3 * 1e4)
Word_depth = Window_size * Total_Tri_grams
K = 150 # Dimensionality of the max-pooling layer
L = 64 # Dimensionality of latent semantic space
J = 3 # Number of random unclicked documents serving as negative examples for a query
FILTER_LENGTH = 1 # We only consider one time step for convolutions

# Input tensors holding the query, positive (clicked) document, and negative (unclicked) documents.
# The first dimension is None because the queries and documents can vary in length.
query = Input(shape = (None, Word_depth))
pos_doc = Input(shape = (None, Word_depth))
neg_docs = [Input(shape = (None, Word_depth)) for j in range(J)]


query_conv = Convolution1D(K, FILTER_LENGTH, padding = "same", input_shape = (None, Word_depth), activation = "relu")(query) 
query_max = Lambda(lambda x: backend.max(x, axis = 1), output_shape = (K, ))(query_conv)
query_sem = Dense(L, activation = "relu", input_dim = K)(query_max)

doc_conv = Convolution1D(K, FILTER_LENGTH, padding = "same", input_shape = (None, Word_depth), activation = "relu")
doc_max = Lambda(lambda x: backend.max(x, axis = 1), output_shape = (K, ))
doc_sem = Dense(L, activation = "relu", input_dim = K)

pos_doc_conv = doc_conv(pos_doc)
neg_doc_convs = [doc_conv(neg_doc) for neg_doc in neg_docs]

pos_doc_max = doc_max(pos_doc_conv)
neg_doc_maxes = [doc_max(neg_doc_conv) for neg_doc_conv in neg_doc_convs]

pos_doc_sem = doc_sem(pos_doc_max)
neg_doc_sems = [doc_sem(neg_doc_max) for neg_doc_max in neg_doc_maxes]

# This layer calculates the cosine similarity between the semantic representations of
# a query and a document.
R_Q_D_p = dot([query_sem, pos_doc_sem], axes = 1, normalize = True) 
R_Q_D_ns = [dot([query_sem, neg_doc_sem], axes = 1, normalize = True) for neg_doc_sem in neg_doc_sems]

concat_Rs = concatenate([R_Q_D_p] + R_Q_D_ns)
concat_Rs = Reshape((J + 1, 1))(concat_Rs)


weight = np.array([1]).reshape(1, 1, 1)
with_gamma = Convolution1D(1, 1, padding = "same", input_shape = (J + 1, 1), activation = "linear", use_bias = False, weights = [weight])(concat_Rs)
with_gamma = Reshape((J + 1, ))(with_gamma)

# Finally, we use the softmax function to calculate P(D+|Q).
prob = Activation("softmax")(with_gamma) 

# We now have everything we need to define our model.
model = Model(inputs = [query, pos_doc] + neg_docs, outputs = prob)
model.compile(optimizer = "Adam", loss = "categorical_crossentropy")

# Build a random data set.
sample_size = 5
Query_macs = []
pos_Docs = []

# Variable length input must be handled differently from padded input.
BATCH = True

(query_len, doc_len) = (5, 50)

for i in range(sample_size):
    
    if BATCH:
        Query_mac = np.random.rand(query_len, Word_depth)
        Query_macs.append(Query_mac)
        
        Doc = np.random.rand(doc_len, Word_depth)
        pos_Docs.append(Doc)
    else:
        query_len = np.random.randint(1, 5)
        Query_mac = np.random.rand(1, query_len, Word_depth)
        Query_macs.append(Query_mac)
        
        doc_len = np.random.randint(5, 50)
        Doc = np.random.rand(1, doc_len, Word_depth)
        pos_Docs.append(Doc)

neg_Docs = [[] for j in range(J)]

for i in range(sample_size):
    possibilities = list(range(sample_size))
    possibilities.remove(i)
    negatives = np.random.choice(possibilities, J, replace = False)
    for j in range(J):
        negative = negatives[j]
        neg_Docs[j].append(pos_Docs[negative])

if BATCH:
    y = np.zeros((sample_size, J + 1))
    y[:, 0] = 1
    
    Query_macs = np.array(Query_macs)
    pos_Docs = np.array(pos_Docs)
    for j in range(J):
        neg_Docs[j] = np.array(neg_Docs[j])
    
    history = model.fit([Query_macs, pos_Docs] + [neg_Docs[j] for j in range(J)], y, epochs = 1, verbose = 0) 
    ''' verbose = logging output'''
else:
    y = np.zeros(J + 1).reshape(1, J + 1)
    y[0, 0] = 1
    
    for i in range(sample_size):
        history = model.fit([ Query_macs[i], pos_Docs[i]] + [neg_Docs[j][i] for j in range(J)], y, epochs = 1, verbose = 0)


get_R_Q_D_p = backend.function([query, pos_doc], [R_Q_D_p])
if BATCH:
    get_R_Q_D_p([Query_macs, pos_Docs])
else:
    get_R_Q_D_p([Query_macs[0], pos_Docs[0]])

# A slightly more complex function. Both neg_docs and the output are lists.
 
get_R_Q_D_ns = backend.function([query] + neg_docs, R_Q_D_ns)
if BATCH:
    get_R_Q_D_ns([Query_macs] + [neg_Docs[j] for j in range(J)])
else:
    get_R_Q_D_ns([Query_macs[0]] + neg_Docs[0])
    
    
    
    
    
    
    