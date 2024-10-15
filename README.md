# join_api-data-generation
Documented and dense repository for generating tuning data for join_api from the IKG.

# Documentation

## Data generation

The following section documents what kind of samples are generated in `generate_sample_data.py`. More details are stored in the docstrings.

1. Simple description of a `APP.OBJECT.METHOD` triplet, e.g. `amazons3.buckettags.DELETEALL`. We create utterances in 4 different ways, the following example may illustrate the differences for the above command.
```python
amazons3.buckettags.DELETEALL
1. "It deletes bucket tags"  # object description from IKG
2. "Delete bucket tags"  # method display_name from IKG
3. "Delete all amazons3 buckettags"  # created from a template.
4. "Delete all Amazon S3 Bucket tags" # created from a template, using app and object display_names, potentially more natural.
```
2. Trigger-action samples. We cross-join all triggers with all actions and combine them using a template, e.g. 
```python
yammer.Message.CREATED
"Triggers when a yammer Message is created."  # method description from IKG
amazons3.buckettags.DELETEALL
"Delete all Amazon S3 Bucket tags" # created from a template
--> 
yammer.Message.CREATED \n amazons3.buckettags.DELETEALL
"When a new message is posted in yammer, delete all Amazon S3 Bucket tags."
```
3. Trigger-action-action samples. Just add another action, e.g. 
```python
yammer.Message.CREATED
"Triggers when a yammer Message is created."  # method description from IKG
amazons3.buckettags.DELETEALL
"Delete all Amazon S3 Bucket tags" # created from a template
salesforce.Note.UPDATE
"Update a Note on Salesforce."
-->
yammer.Message.CREATED \n amazons3.buckettags.DELETEALL \n salesforce.Note.UPDATE
"When a new message is posted in yammer, delete all Amazon S3 Bucket tags and update a Note on Salesforce."
```
4. If-conditionals (very WIP), generally follows the structure "when TRIGGER, do ACTION1 if CONDITION else ACTION2". Few rephrasals of this are implemented.
```python
...  # same as above
-->
yammer.Message.CREATED \n if CONDITION: \n\t amazons3.buckettags.DELETEALL \n else: \n\t salesforce.Note.UPDATE
"When a new message is posted in yammer, delete all Amazon S3 Bucket tags if CONDITION, else update a Note on Salesforce."
"When a new message is posted in yammer, if CONDITION, delete all Amazon S3 Bucket tags, else update a Note on Salesforce."
```
5. Simple for statement: currently is implemented as RETRIEVEALL -> ACTION, where the iterator variable is the lemmatized object.
```python
amazons3.buckettags.RETRIEVEALL
"Retrieve all Amazon S3 Bucket tags."
trello.Member.CREATE
"Add a member to a Trello Card"
-->
for buckettags in amazons3.buckettags.RETRIEVEALL: \n\t trello.Member.CREATE
"For each amazons3 buckettags, add a member to a Trello Card"
```

6. SYNC: Synchronize newly created objects with any other objects. Just uses names of the apps and objects to be synchronized in the template.
```python
yammer.User.CREATED \n yammer.User.RETRIEVEALL \n trello.Member.CREATE
"Sync new yammer users with trello members"
```

7. MOVE: Move an object across apps, i.e. the object must be present in both apps. Just uses names of the apps and the object in the template.
```python
for comment in trello.Comment.RETRIEVEALL: \n\t confluence.Comment.CREATE \n trello.Comment.DELETEALL
"Move comments from Trello to Confluence"
```

8. COPY: Copy an object across apps, i.e. the object must be present in both apps. Just uses names of the apps and the object in the template. Same as MOVE, but without deleting the item afterwards.
```python
for comment in trello.Comment.RETRIEVEALL: \n\t confluence.Comment.CREATE
"Copy comments from Trello to Confluence"
```

## Paraphrasing

Since we don't want the model to rely on the template structure to decide which code to generate, we want to add variation to the synthetically generated utterances.

Simple variations are already implemented, such as an initial length-weighted sampling across the 4 types of triplet (`APP.OBJECT.METHOD`) descriptions, as well as applying different templates randomly for specific cases, but larger-scale paraphrasing is necessary to make the trained model more robust to human variability.

For this reason, for code types (2, 3, 5, 6, 7, 8), we have hand-generated a few examples of paraphrasing related to the sample type (stored in `data/paraphrase_templates/`), which are used as few-shot examples for prompting Cluster BloomZ, a much larger language model (using `scripts/paraphrase_utterances.py`).

Parphrased examples can be found as `data/0420_df_samples_paraphrased.pkl`.

# How to run
Create a virtual environment from `requirements.txt`.

```python
python3 -m venv venv  # create venv in folder `venv`
source venv/bin/activate  # activate venv
pip install -r requirements.txt
```

Generating join_api sample data from IKG:
```python
python3 scripts/generate_sample_data.py --output_dir "..."
```

Querying BloomZ for creating paraphrases for the generated data. The args have defaults, but may need to be changed. `data_path` is the `pkl` file where the output of `generate_sample_data.py` is stored (e.g. '../data/0412_length_weighted_sampling_full.pkl'), `output_path` is the name of the output file and `paraphrases_dir` is the directory where the txt files where the paraphrase prompts are stored.

```python
export Cluster_API_KEY='...'  # store Cluster API key as environment variable
python3 scripts/paraphrase_utterances.py --data_path "..." --output_path "..." --paraphrases_dir "..."
```
