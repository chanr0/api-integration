# %%
from ast import literal_eval
import re
from tqdm import tqdm
from typing import List

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

tqdm.pandas()


def add_end_token(s, end_token="<|endoftext|>"):
    if s[-len(end_token) :] == end_token:
        return s
    return s + end_token


def load_txt_dataset(
    filename: str, comments=True, multiline=True, variations=False, add_end=False
):
    """Load a dataset as a python list.

    Keyword arguments:
    comments -- allow and ignore comments in file
    multiline -- a sample in the dataset may extend over multiple
        lines (continuation lines must start with a space which will
        be removed from the sample
    variations -- allow variations of the first line
    add_end -- add an end-token at the end of the sample
    """
    data = []
    variants = set()
    dataitem = None
    with open(filename, encoding="utf-8") as f:
        for line in f:
            if line == "\n":
                continue
            if comments and line[0] == "#":
                continue
            if variations and line[0] == "@":
                if dataitem[0] != "@":
                    variants.add(dataitem)
                    dataitem = "@"
                variants.add(line[1:])
                # ignore variations for the time being
                continue
            if multiline and line[0] == " ":
                dataitem += line[1:]
            else:
                if dataitem:
                    if variants:
                        for v in variants:
                            data.append((v + dataitem[1:]).strip())
                        variants = set()
                    else:
                        data.append(dataitem.strip())
                dataitem = line
    if dataitem:
        if variants:
            for v in variants:
                data.append((v + dataitem[1:]).strip())
        else:
            data.append(dataitem.strip())
    if add_end:
        data = [add_end_token(d) for d in data]
    return data

# %%
data_path = '../data/join_api/utterances-orig.txt'
utterances_orig = load_txt_dataset(data_path, add_end=False)

utterances_orig
# %% load paraphrased data, as well as training data
import pandas as pd

df_paraphrased = pd.read_csv(
    '../data/0421_paraphrased.csv', index_col=0
)
df_commands = pd.read_csv(
    '../data/0324_all_commands.csv', index_col=0
)
# %%
from data_generation_utils import load_ikg_data, generate_IKG_triplet_utterances
# pd.set_option("display.max_rows", None)
# pd.set_option('display.max_colwidth', None)
source: str = "../data/join_api/join_api_fs.json"
product_database_path: str = "../data/join_api/deployment_targets.csv"
application_filter: List[str] = None

df_descriptions_orig = load_ikg_data(
        source, product_database_path, application_filter
    )
df_descriptions_orig['app_name'] = df_descriptions_orig.apply(
    lambda x: x.command.split('.')[0], axis=1
)
df_descriptions_orig['bo_name'] = df_descriptions_orig.command.apply(
    lambda x: x.split('.')[1]
)
df_descriptions_orig.rename(columns={'command': 'commands'}, inplace=True)

# Load base data
df_apps = df_descriptions_orig.groupby(
    'app_display_name', as_index=False
).first()[['app_display_name', 'app_name']]
df_descriptions_full = generate_IKG_triplet_utterances(random_capitalization=False)
df_descriptions_full.rename(columns={'command': 'commands'}, inplace=True)

df_app_aliases = pd.read_csv('../data/app_aliases.csv', index_col = 0)
df_app_aliases['aliases'] = df_app_aliases.aliases.apply(
    literal_eval
)

# %%
df_descriptions_templatized = df_descriptions_full.groupby(
    'commands'
)['description'].apply(list).reset_index()
df_descriptions_templatized['templatized_description'] = df_descriptions_templatized.apply(
    lambda x: x.description[-1], axis=1
)
df_descriptions_templatized.drop('description', axis=1, inplace=True)

# %%
df_descriptions_replaced = df_descriptions_templatized.merge(
    df_descriptions_full[['commands', 'app_display_name']].drop_duplicates(),
    how='left'
).merge(
    df_app_aliases[
            df_app_aliases.apply(
                lambda x: len(x.aliases) >= 1,
            axis=1)
    ][['app_display_name', 'aliases']],
    how='left',
    on='app_display_name'
)

# Drop the rows where the 
df_descriptions_replaced = df_descriptions_replaced[
    df_descriptions_replaced.aliases.notna()
]
df_descriptions_replaced = df_descriptions_replaced.explode('aliases').reset_index(drop=True)

df_descriptions_replaced['utterances'] = df_descriptions_replaced.explode('aliases').apply(
    lambda x: re.sub(x.app_display_name, x.aliases, x.templatized_description),
    axis=1
)
df_eval_1 = df_descriptions_replaced.groupby('aliases').sample(1, random_state=4)[['utterances', 'commands']]
commands_grouped = df_descriptions_orig.groupby(
    ['app_display_name', 'object_display_name']
)['commands'].apply(list).reset_index()

# %%$ NOTE: I don't think this selection is specific enough to filter the truly unique BOs.
commands_grouped['bo_exact_matches'] = commands_grouped.progress_apply(
    lambda x: [
        (x.object_display_name, j)
        for ct, j in enumerate(commands_grouped.object_display_name.str.lower().tolist())
        if x.object_display_name.lower().rstrip('s') == j.lower().rstrip('s') and ct != x.name
    ],
        axis=1
)
# %%
df_eval_2 = commands_grouped[
    commands_grouped.bo_exact_matches.str.len() == 0
].groupby('app_display_name').sample(1, random_state=0).sample(25, random_state=4)
df_eval_2 = df_eval_2.explode('commands').groupby(['app_display_name', 'object_display_name']).sample(1, random_state=4).reset_index(drop=True)
df_eval_2['utterances'] = df_eval_2.merge(
    df_descriptions_templatized,
    how='left',
    on='commands').apply(
        lambda x: re.sub(f' {x.app_display_name}', '', x.templatized_description),
        axis=1
)
df_eval_2 = df_eval_2[['utterances', 'commands']]
df_eval_2

# %%
commands_grouped_app_bo = df_descriptions_orig.groupby(
    ['app_name', 'bo_name']
)['commands'].apply(list).reset_index()
commands_grouped_app_bo.app_name.value_counts()[commands_grouped_app_bo.app_name.value_counts() == 1].index

# %%
single_object_apps = commands_grouped_app_bo[commands_grouped_app_bo.app_name.isin(
    commands_grouped_app_bo.app_name.value_counts()[commands_grouped_app_bo.app_name.value_counts() == 1].index
)].explode('commands')

single_object_apps_merged = single_object_apps.merge(
    df_descriptions_full[['commands', 'description']], how='left', on='commands'
)
# single_object_apps_merged.to_csv('df_eval_3_dev.csv')
# %% this is done manually, 
df_eval_3 = pd.read_csv('df_eval_3_censored_apps.csv', index_col=0)
df_eval_3 = df_eval_3.groupby('commands', as_index=False).first().rename(
    columns={'description_censored': 'utterances'}
)
df_eval_3

# %%
df_samples = pd.read_pickle('../data/0412_length_weighted_sampling_full.pkl')
df_samples_paraphrased = pd.read_pickle('../data/0420_df_samples_paraphrased.pkl')

# %%
from data_generation_utils import _command_is_trigger
df_descriptions_actions = df_descriptions_orig[df_descriptions_orig.apply(
    lambda x: (
        (x.app_display_name.lower() in x.description.lower())
        |
        (x.app_name.lower() in x.description.lower())
     ) & (not _command_is_trigger(x.commands)) if x.description is not None else False,
    axis=1
)]

# %%
df_descriptions_actions['description'] = df_descriptions_actions.apply(
    lambda x: lemmatizer.lemmatize(x.description.split()[0].lower(), pos='v') + ' ' + ' '.join(x.description.split()[1:]),
    axis=1
)

# %%
df_eval_4 = df_samples[df_samples.apply(
    lambda x: all([i in x.utterances for i in x.app_list]) & (x.type == 3),
    axis=1
)].reset_index(drop=True)

df_eval_4 = pd.concat([df_eval_4.sample(1000, replace=True).reset_index(drop=True), df_descriptions_actions.groupby('app_display_name').sample(4, replace=True)[['commands', 'description']].sample(
    1000, replace=True
).reset_index(drop=True).rename(columns={'commands': 'commands_new', 'descriptions': 'descriptions_new'})
], axis=1)



# %%
pd.set_option("display.max_colwidth", None)
df_eval_4['commands'] = df_eval_4.apply(
    lambda x: x.commands + '\n' + x.commands_new,
    axis=1
)

df_eval_4['utterances'] = df_eval_4.apply(
    lambda x: re.sub(' and ', ', ',x.utterances).rstrip('.') + ' and ' + x.description,
    axis=1
)

# %%
df_descriptions_triggers = df_descriptions_orig[df_descriptions_orig.apply(
    lambda x: (
        (x.app_display_name.lower() in x.description.lower())
        |
        (x.app_name.lower() in x.description.lower())
     ) & (_command_is_trigger(x.commands)) if x.description is not None else False,
    axis=1
)]
df_descriptions_triggers
# %%
# from data_generation_utils import preprocess_triggers

def preprocess_triggers(trigger_description: str) -> str:
    """WIP Utility to make trigger descriptions more natural.
    Rule-based, derived from observing IKG trigger descriptions.

    Args:
        trigger_description (str): Trigger description from the IKG.

    Returns:
        str: Slightly rephrased trigger description.
    """
    if trigger_description.startswith("Triggers when"):
        return f'When {trigger_description.lstrip("Triggers when").rstrip(".")}'.capitalize()
    elif trigger_description.startswith("Specifies"):
        return f'For{trigger_description.lstrip("Specifies").rstrip(".")}'.capitalize()
    else:
        if trigger_description.endswith("s"):
            return f'For {trigger_description.lower().rstrip(".")}'
        else:
            return f'For a {trigger_description.lower().rstrip(".")}'


df_descriptions_triggers['description'] = df_descriptions_triggers.description.apply(
    preprocess_triggers
)


# # %%
# df_eval_5 = df_samples[df_samples.apply(
#     lambda x: all([i in x.utterances for i in x.app_list]) & (x.type == 3),
#     axis=1
# )].reset_index(drop=True)

# df_eval_5 = pd.concat(
#     [
#         df_eval_5,
#         df_descriptions_triggers.sample(
#             len(df_eval_5), replace=True
#         ).reset_index(drop=True)[['commands', 'description']].rename(
#             columns={'commands': 'commands_new', 'description': 'description_new'}
#         )
#     ],
#     axis=1)

# # %%
# df_eval_5['commands'] = df_eval_5.apply(
#     lambda x: x.commands_new + '\n' + x.commands,
#     axis=1
# )
# df_eval_5['utterances'] = df_eval_5.apply(
#     lambda x: x.description_new + ' and ' + ' '.join(x.utterances.split()[1:]),
#     axis=1
# )
# df_eval_5[['utterances','commands']]


# %%
def load_dataset(
    filename, comments=True, multiline=True, variations=False, add_end=False
):
    """Load a dataset as a python list, if multiline==True, a dataset may
    extend over multiple lines.  In this case, a line ending each consecutive line must
    start with a space which will be removed from the data."""
    data = []
    variants = set()
    dataitem = None
    with open(filename, encoding="utf-8") as f:
        for line in f:
            if line == "\n":
                continue
            if comments and line[0] == "#":
                continue
            if variations and line[0] == "@":
                if dataitem[0] != "@":
                    variants.add(dataitem)
                    dataitem = "@"
                variants.add(line[1:])
                # ignore variations for the time being
                continue
            if multiline and line[0] == " ":
                dataitem += line[1:]
            else:
                if dataitem:
                    if variants:
                        for v in variants:
                            data.append((v + dataitem[1:]).strip())
                        variants = set()
                    else:
                        data.append(dataitem.strip())
                dataitem = line
    if dataitem:
        if variants:
            for v in variants:
                data.append((v + dataitem[1:]).strip())
        else:
            data.append(dataitem.strip())
    if add_end:
        data = [add_end_token(d) for d in data]
    return data

data_all = load_dataset('../data/join_api/utterances-orig-corrected.txt', add_end=False, variations=True)
_utterances, _commands = [], []
for sample in data_all:
    # remove `eos_token` if already present
    _utterances.append(sample.split("\n")[0])
    # get commands. Insert tabs instead of double-whitespaces after newlines
    _commands.append(
        sample.lstrip(sample.split("\n")[0]).lstrip().replace("\n  ", "\n\t").replace('()', '')
    )
df_eval_orig = pd.DataFrame({"utterances": _utterances, "commands": _commands})
df_eval_orig

# %%
# df_eval_5
df_eval_4

# %%
pd.reset_option('display.max_rows')
pd.reset_option('display.max_colwidth')
pd.concat(
    [
        df_eval_1[['commands', 'utterances']].assign(type=1),
        df_eval_2[['commands', 'utterances']].assign(type=2),
        df_eval_3[['commands', 'utterances']].assign(type=3),
        df_eval_4[['commands', 'utterances']].assign(type=4),
        # df_eval_5[['commands', 'utterances']].assign(type=5),
        df_eval_orig.assign(type=6)
    ],
    axis=0
).reset_index(drop=True).to_csv('../data/0630_evaluation.csv')

# %%

df_paraphrased



# %%
# This is for checking model calibration under uncertainty
# ########
pd.set_option('display.max_colwidth', None)
df_descriptions_full['object_display_name_lemmatized'] = df_descriptions_full.apply(
    lambda x: (' '.join(x.object_display_name.split()[:-1]) + " " + lemmatizer.lemmatize(x.object_display_name.split()[-1].lower(), pos='n')).lower()
    if len(x.object_display_name.split()) > 1
    else lemmatizer.lemmatize(x.object_display_name.lower(), pos='n'),
    axis=1
)
aa = df_descriptions_full.groupby(['app_display_name', 'object_display_name_lemmatized'])['commands'].apply(list).reset_index().groupby('object_display_name_lemmatized')['app_display_name'].apply(list)
aa

# %%
bb = df_descriptions_full.groupby(['app_display_name', 'object_display_name_lemmatized'])['commands'].apply(
    lambda x: list(set(list(x)))).reset_index().groupby(
    'object_display_name_lemmatized'
    )['commands'].apply(sum)
# %% these are grouped all commands, which hace the same app
aa[aa.str.len() > 1].merge(
    bb
)

# %%
pd.merge(
    aa[aa.str.len() > 1],
    bb,
    on='object_display_name_lemmatized',
    how='left'
)


# .groupby('object_display_name_lemmatized')['app_display_name'].apply(list)

# %%
df_descriptions_full.groupby(['app_display_name', 'object_display_name_lemmatized'])['commands'].apply(list).reset_index()

#%%
df_descriptions_full.groupby(['app_display_name', 'object_display_name_lemmatized'])['commands'].apply(
    lambda x: list(set(list(x)))).reset_index().groupby(
    'object_display_name_lemmatized'
    )['app_display_name'].apply(list)


# %%
df_descriptions_full.groupby('object_display_name_lemmatized')['commands'].apply(list)
# %%
bb = df_descriptions_full.groupby(['app_display_name', 'object_display_name_lemmatized'])['commands'].apply(lambda x: list(set(list(x)))).reset_index().groupby('object_display_name_lemmatized')['command'].apply(sum)

# %%
pd.set_option('display.max_rows', None)
aaa = pd.merge(
    aa[aa.str.len() > 1], bb, on='object_display_name_lemmatized', how='left'
)
# %%
aaaa = aaa.explode('command')
aaaa['method'] = aaaa.apply(
    lambda x: x.command.split('.')[-1],
    axis=1
)
# %%

aaaa
# %%
aa[aa.str.len() > 1]
# %%
df_descriptions_full.groupby(['app_display_name', 'object_display_name'])['command'].apply(lambda x: list(set(list(x)))).reset_index()
# %%
df_descriptions_full
# %%
