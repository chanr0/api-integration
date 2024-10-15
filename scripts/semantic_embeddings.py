__author__ = 'Katya Mirylenka'
'''Script produces the .pkl files with the semantic embedding fur further semantic search'''

import openai
import chromadb
import dill
import pickle as pkl
from sentence_transformers import SentenceTransformer

model_emb = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
sentence_embeddings = True

openai.api_key = "EMPTY" # Not support yet
openai.api_base = "http://missy.tryit.zurich.co.com:8000/v1"
model = "openllama-enr-join_api"

with open('/Users/kmi_local/Downloads/ikg.pkl', 'rb') as file:
    ikg = dill.load(file)

data = []
metadata = []
n_paths = 0
for app, obj_dict in ikg.items():
    for obj, operation_dict in obj_dict.items():
        for operation in operation_dict:
            data.append(f"{app}.{obj}.{operation} Flow description: {operation_dict[operation]['command_description']}")
            #data.append(f"{app} {obj} {operation} {operation_dict[operation]['command_description']}")
            #data.append(f"{operation_dict[operation]['command_description']}. {app}.{obj}.{operation}")
            metadata.append({"Application": operation_dict[operation]['app_display_name']})
            n_paths += 1

print(f'Number of paths = {n_paths}')


def get_embedding(text):
    return openai.Embedding.create(input=text, model=model)['data'][0]['embedding']


chroma_client = chromadb.Client() # in-memory vector db
collection = chroma_client.create_collection(name="my_collection")

n_init = len(data)
data_ = data[:n_init]
metadata_ = metadata[:n_init]
filehandler = open('ikg_data_sent.pkl', 'wb')
pkl.dump(data, filehandler)

#filehandler = open('ikg_metadata.pkl', 'wb')
#pkl.dump(metadata, filehandler)

print(f'datasize = {len(data_)}')
if sentence_embeddings:
    embeddings = model_emb.encode(data_, convert_to_numpy=True).tolist()
else:
    embeddings = [get_embedding(x) for x in data_] #[get_embedding(x) for x in data200]

#filehandler = open('ikg_emb_sent.pkl', 'wb') # .pkl files for sentence embeddings
#pkl.dump(embeddings, filehandler)

ids = []
i = 0
for el in data_:
    ids.append('id'+ str(i))
    i += 1

collection.add(
    embeddings=embeddings,
    documents=data_,
    ids=ids,
    metadatas=metadata_
)

print(collection.count())

# testing a sequence
query = "Create an Amazon Simple Notification Service Tags."
if sentence_embeddings:
    emb = model_emb.encode([query], convert_to_numpy=True)[0].tolist()
docs = collection.query(query_embeddings=emb,
    n_results=5,
)

print(docs)
