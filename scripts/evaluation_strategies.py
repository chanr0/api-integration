__author__ = 'Katya Mirylenka'
'''The script produce eal.pkl file using different strategies:
- no prompting
- prompting
- conversational simulation'''


import pickle as pkl
from tqdm import tqdm
import openai
from python.eval_util import normalized_token_edit_distance, unify_varnames

openai.api_key = "EMPTY" # Not support yet
openai.api_base = "http://missy.tryit.zurich.co.com:8000/v1"
model = "openllama-enr-join_api"
from transformers import AutoTokenizer

test_file_path = '.../0728_paraphrased_test_combined.pkl' # test file with utterance - api flow pairs, is df wwith columns 'utterances', 'commands'
eval_file_path = 'llama_results_df_prompting.pkl' # resulting eval file with eval df, which has the same structure as the test df but with additional columns 'generations'
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-3b")
conversational_simulation = False # parameter showing if the conversational simulation should be evaluated
prompting = True



def get_completion(prompt, model=model, _temp=0): # get the answer to a model from an utterance
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=_temp, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def remove_free_text(response):
    response = unify_varnames(response).replace("\t", " ")
    res = response.replace("\n", " ")
    ar = res.split(':')
    if len(ar) == 1:
        return response
    elif 'if ' in ar[0].lower() or 'for ' in ar[0].lower():
        return response
    return ar[1].strip()


with open(test_file_path, 'rb') as f:
    test = pkl.load(f)
n_test = len(test)

counter = 0
evaluation = test.copy()
result = []

for i in tqdm(range(n_test)):
    prompt_conversational = f""" Given the user utterance in <> below, produce an API call or calls in a form of a triplet:
    application.object.operation
    
    where the application takes value from the following application set:
    
    applications = {{amazons3, amazonsns, asana, azureblobstorage, bigcommerce, box, ciscospark, cloudantdb, cmis, concur, confluence, coupa, domino, dropbox, email, eventbrite, flexengage, freshdesk, gmail, googleanalytics, googledrive, googlesheet, hubspotcrm, hubspotmarketing, cocoss3, codb2, costerlingiv, costerlingoms, costerlingsciv1, cotwc, ift, insightly, intacct, jira, kronos, magento, mailchimp, marketo, maximo, msad, msdynamicscrmrest, msoffice365, msonedrive, mssharepoint, netsuitecrm, netsuiteerp, netsuitefinance, oraclesalescloud, quickbooksonline, rediscache, salesforce, salesforcepardot, saphybris, sapodata, servicenow, sfcommerceclouddata, sfservicecloud, sftp, shopify, silverpop, slack, stripe, surveymonkey, trello, twilio, watsondiscovery, watsonlt, wufoo, yammer, yapily, zendesk, zohocrm, zuora, msdynamicscrm-mock, amazondynamodb, amazonsqs, iiboc, googlecloudstorage, coewm, jenkins, ldap, azuread, msdynamicsfando, oraclehcm, salesforcemc}}
    
    utterance = <{test['utterances'][i]} {tokenizer.eos_token}>
    
    Output just the triplet(s) with if or for statements if needed without any comments or additional explanations. In case of if statement if condition is not known - state if CONDITION.
    Output just triplet(s).
    
    """
    prompt = f""" Given the user utterance in <> below, produce an API call or calls in a form of a triplet:
    [application.object.operation]
    
    where application takes value from the following application set:
    
    applications = {{amazons3, amazonsns, asana, azureblobstorage, bigcommerce, box, ciscospark, cloudantdb, cmis, concur, confluence, coupa, domino, dropbox, email, eventbrite, flexengage, freshdesk, gmail, googleanalytics, googledrive, googlesheet, hubspotcrm, hubspotmarketing, cocoss3, codb2, costerlingiv, costerlingoms, costerlingsciv1, cotwc, ift, insightly, intacct, jira, kronos, magento, mailchimp, marketo, maximo, msad, msdynamicscrmrest, msoffice365, msonedrive, mssharepoint, netsuitecrm, netsuiteerp, netsuitefinance, oraclesalescloud, quickbooksonline, rediscache, salesforce, salesforcepardot, saphybris, sapodata, servicenow, sfcommerceclouddata, sfservicecloud, sftp, shopify, silverpop, slack, stripe, surveymonkey, trello, twilio, watsondiscovery, watsonlt, wufoo, yammer, yapily, zendesk, zohocrm, zuora, msdynamicscrm-mock, amazondynamodb, amazonsqs, iiboc, googlecloudstorage, coewm, jenkins, ldap, azuread, msdynamicsfando, oraclehcm, salesforcemc}}
    
    Output just the triplet of API calls without explanations!
    utterance: <{test['utterances'][i]} {tokenizer.eos_token}>
    """

    if prompting:
        utterance = prompt
    else:
        utterance = f"{test['utterances'][i]} {tokenizer.eos_token}"
    response = get_completion(utterance)
    response = response.replace("\\", "").replace('[', "").replace(']', "")
    response = remove_free_text(response)
    init_response = response
    response = unify_varnames(response).replace("\t", " ")
    label = unify_varnames(test['commands'][i]).replace("\t", " ")
    if normalized_token_edit_distance(tokenizer.encode(response, return_tensors="pt"), tokenizer.encode(label), tokenizer) <= 0.0001:
        counter += 1
    elif conversational_simulation:
        app = (test['commands'][i].split('.')[0]).split(' ')[-1]
        if app.lower() not in test['utterances'][i].lower():
            response = get_completion(prompt_conversational)
            response = response.replace("\\", "")
            response = unify_varnames(response).replace("\t", " ")
            prompt_response = response
            if normalized_token_edit_distance(tokenizer.encode(response, return_tensors="pt"), tokenizer.encode(label), tokenizer) <=0.0001:
                counter += 1
            else:
                response = get_completion(f"{test['utterances'][i]}, I mean for {app}. Output just triplet(s).")
                response = response.replace("\\", "")
                response = unify_varnames(response).replace("\t", " ")
                if normalized_token_edit_distance(tokenizer.encode(response, return_tensors="pt"), tokenizer.encode(label), tokenizer) <=0.0001:
                    counter += 1
                else:
                    response = init_response
    result.append(response)

print(f'{counter} api calls are initially correct out of {n_test}')
evaluation['generations'] = result

filehandler = open(eval_file_path, 'wb')
pkl.dump(evaluation, filehandler)
