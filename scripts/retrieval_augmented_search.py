__author__ = 'Katya Mirylenka'

import openai
import pickle as pkl
import chromadb
from tqdm import tqdm
from python.eval_util import normalized_token_edit_distance, unify_varnames
from transformers import AutoTokenizer

from sentence_transformers import SentenceTransformer
model_emb = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-3b")

openai.api_key = "EMPTY" # Not support yet
openai.api_base = "http://missy.tryit.zurich.co.com:8000/v1"
model = "openllama-enr-join_api-medium-230803"

sentence_embeddings = True

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection")

def get_embedding(text):
    return openai.Embedding.create(input=text, model=model)['data'][0]['embedding']

def get_completion(prompt, model=model, _temp = 0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=_temp # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def split_search_result(str1, path_only=True):
    ar = str1.split(' Flow description: ')
    path, description = ar[0], ar[1]
    if path_only:
        return f"{path}"
    return f"{description}: {path}"

def remove_free_text(response):
    response = unify_varnames(response).replace("\t", " ")
    res = response.replace("\n", " ")
    ar = res.split(':')
    if len(ar) == 1:
        return response
    elif 'if ' in ar[0].lower() or 'for ' in ar[0].lower():
        return response
    return ar[1].strip()

def if_semantic_fits(docs, label):
    for doc in docs:
        ar = doc.split(' Flow description: ')
        candidate, description = ar[0], ar[1]
        if candidate in label:
            return 1
    return 0

if sentence_embeddings:
    data_file = 'ikg_data_sent.pkl'
    embeddings_file = 'ikg_emb_sent.pkl'
else:
    data_file = 'ikg_data.pkl'
    embeddings_file = 'ikg_emb.pkl'

with open(data_file, 'rb') as f:
    data = pkl.load(f)
with open(embeddings_file, 'rb') as f:
    embeddings = pkl.load(f)

print(f'datasize = {len(data)}')

ids = []
i = 0
for el in data:
    ids.append('id' + str(i))
    i += 1
#print(embeddings[0])

collection.add(
    embeddings=embeddings,
    documents=data,
    ids=ids
)

#print(collection.count())

#query = "Create an Amazon Simple Notification Service Tags."
#docs = collection.query(query_embeddings=get_embedding(query),
#                        n_results=5)

with open('/Users/kmi_local/Downloads/0728_paraphrased_test_combined.pkl', 'rb') as f:
    test = pkl.load(f)
n_test = len(test)
print(f'number of test samples = {n_test}')
counter = 0
result = []
evaluation = test.copy()
counter_semantic = 0
#for i in tqdm(range(n_test)):
for i in tqdm(range(n_test)):
    utterance = test['utterances'][i]
    label = unify_varnames(test['commands'][i]).replace("\t", " ")
    if sentence_embeddings:
        emb = model_emb.encode([utterance])[0].tolist()
    else:
        emb = get_embedding(utterance)
    docs = collection.query(query_embeddings=emb, n_results=5)
    #print(docs)
    docs = docs['documents'][0]

    #print(type(docs))
    #print(docs)
    # evaluate how many find paths are in the final utterance:
    counter_semantic += if_semantic_fits(docs, label)
    prompt = f""" Given the user utterance in <> below, produce an API call or calls 
    where a single api call can be similar or one of the following (description, call) pairs:
    {split_search_result(docs[0], path_only=False)}
    {split_search_result(docs[1], path_only=False)}
    {split_search_result(docs[2], path_only=False)}
    {split_search_result(docs[3], path_only=False)}
    {split_search_result(docs[4], path_only=False)}
    
    The first  part of the API calls, namely the applications should take values from the following set:
    applications = {{amazons3, amazonsns, asana, azureblobstorage, bigcommerce, box, ciscospark, cloudantdb, cmis, concur, confluence, coupa, domino, dropbox, email, eventbrite, flexengage, freshdesk, gmail, googleanalytics, googledrive, googlesheet, hubspotcrm, hubspotmarketing, cocoss3, codb2, costerlingiv, costerlingoms, costerlingsciv1, cotwc, ift, insightly, intacct, jira, kronos, magento, mailchimp, marketo, maximo, msad, msdynamicscrmrest, msoffice365, msonedrive, mssharepoint, netsuitecrm, netsuiteerp, netsuitefinance, oraclesalescloud, quickbooksonline, rediscache, salesforce, salesforcepardot, saphybris, sapodata, servicenow, sfcommerceclouddata, sfservicecloud, sftp, shopify, silverpop, slack, stripe, surveymonkey, trello, twilio, watsondiscovery, watsonlt, wufoo, yammer, yapily, zendesk, zohocrm, zuora, msdynamicscrm-mock, amazondynamodb, amazonsqs, iiboc, googlecloudstorage, coewm, jenkins, ldap, azuread, msdynamicsfando, oraclehcm, salesforcemc}}
    
    Output just the triplet(s) of API calls without explanations!
    
    utterance: <{utterance} {tokenizer.eos_token}>
    """

    prompt1 = f""" Given the user utterance in <> below, produce an API call or calls 
    where a single api call can be similar or one of the following API calls:
    {split_search_result(docs[0])}
    {split_search_result(docs[1])}
    {split_search_result(docs[2])}
    {split_search_result(docs[3])}
    {split_search_result(docs[4])}
    
    Output just the triplet(s) of API calls without explanations!
    
    utterance: <{utterance} {tokenizer.eos_token}>
    """
    prompt2 = f""" utterance: <{utterance} {tokenizer.eos_token}>
    Output can be one of those:
    {split_search_result(docs[0])}
    {split_search_result(docs[1])}
    {split_search_result(docs[2])}
    {split_search_result(docs[3])}
    {split_search_result(docs[4])}
    """

    #print(prompt)
    response = get_completion(prompt2)
    response = response.replace("\\", "")
    response = remove_free_text(response)
    response = unify_varnames(response).replace("\t", " ")

    if normalized_token_edit_distance(tokenizer.encode(response, return_tensors="pt"), tokenizer.encode(label), tokenizer) <= 0.0001:
        counter += 1
    #else:
    #    print(test['utterances'][i], test['type'][i], test['subtype'][i])
    #    print('%%%%%%%%%%%%%%%%%%%%%%%%% Extected answer:')
    #    print(test['commands'][i])
    #    print('%%%%%%%%%%%%%%%%%%%%%%%%% Response:')
    #    print(response)
    #    print('%%%%%%%%%%%%%%%%%%%%%%%%% Prompt:')
    #    print(prompt)
    result.append(response)

print(f'{counter} api calls are initially correct out of {n_test}')
print(f'{counter_semantic} api calls out of {n_test} where among the top-5 semantic search results')
evaluation['generations'] = result

filehandler = open('llama_results_sentence_embed_promt2.pkl', 'wb')
pkl.dump(evaluation, filehandler)
