import argparse
from collections import Counter
import json
import logging
import os
import random
import sys
from typing import Any, List
from tqdm import tqdm
from tqdm.dask import TqdmCallback
import warnings

import dask.dataframe as dd
import nltk

try:
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
except LookupError:
    nltk.download("wordnet")
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
import numpy as np
import pandas as pd
from thefuzz import fuzz

# append parent dir to path to import from parent (`join_api`).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import join_api
from join_api.common import *
import join_api.model
import util

lemmatizer = WordNetLemmatizer()
l = logging.getLogger("ikg")
logging.basicConfig(
    format="%(created)s %(name)s %(filename)s:%(lineno)s %(levelno)s %(message)s",
    level=logging.INFO,
)
tqdm.pandas()
# pd.set_option('display.max_colwidth', None)


def _load_json(filename: str) -> Any:
    l.log(25, f"loading {filename}")
    with open(filename) as fd:
        return json.load(fd)


def _get_iterator_variable(x: pd.Series) -> str:
    return lemmatizer.lemmatize(x.command_action1.split(".")[1].lower(), wordnet.NOUN)


def _command_is_trigger(
    command: str,
    triggers: List[str] = [
        "CREATED",
        "UPDATED",
        "DELETED",
        "CREATED_POLLER",
        "UPDATED_POLLER",
        "CREATEDORUPDATED_POLLER",
    ],
) -> bool:
    return command.split(".")[-1] in triggers


def contains_appname_in_description_fuzzy(
    command: str, description: str, thresh: int = 80
) -> bool:
    """Utility function for fuzzy-checking whether the appname contained in the `command` triplet
    is likely also in the `description` string. Threshold determines tolerance of fuzziness.
    Internally uses `thefuzz.fuzz.partial_ratio`, based on Levinshtein distance.

    Args:
        command (str): Command string in the `APP.OBJECT.METHOD` format.
        description (str): String to check whether it contains a variant of the app in the command.
        thresh (int, optional): Threshold for the `fuzz.partial_ratio` value, higher means more selective. Defaults to 80.

    Returns:
        bool: Boolean value indicating whether the partial match ratio is above the `threshold` value.
    """
    if (
        (command is None)
        or (description is None)
        or (command.strip() == "")
        or (description.strip() == "")
    ):  # i.e. if either of the strings is empty or missing
        warnings.warn("Some commands and/or descriptions are None or empty.")
        return False
    else:
        return (
            fuzz.partial_ratio(command.split(".")[0].lower(), description.lower())
            > thresh
        )


def generate_templatized_utterance(
    app_name: str,
    object_name: str,
    operation_name: str,
    actions: List[str] = [
        "CREATE",
        "RETRIEVEALL",
        "UPDATE",
        "UPDATEALL",
        "DELETEALL",
        "UPSERTWITHWHERE",
    ],
    triggers: List[str] = [
        "CREATED",
        "UPDATED",
        "DELETED",
        "CREATED_POLLER",
        "UPDATED_POLLER",
        "CREATEDORUPDATED_POLLER",
    ],
    random_capitalization: bool = True,
) -> str:
    """Utility function for creating a templatized utterance from appname, business object and operation.
    Formulations of the utterances are rule-based depending on the currently supported operations and
    inspired by the descriptions of actual operations.

    Args:
        app_name (str): Name of the app.
        object_name (str): Name of the business object.
        operation_name (str): Name of the operation / object method.
        actions (List[str], optional): List of currently supported actions. Defaults to
            [ "CREATE", "RETRIEVEALL", "UPDATE", "UPDATEALL", "DELETEALL", "UPSERTWITHWHERE"].
        triggers (List[str], optional): List of currently supported triggers. Defaults to
            [ "CREATED", "UPDATED", "DELETED", "CREATED_POLLER", "UPDATED_POLLER", "CREATEDORUPDATED_POLLER"].
        random_capitalization (bool, optional): Boolean flag whether to do randomly lowercase 50% of the samples. Defaults to True.

    Raises:
        ValueError: _description_

    Returns:
        str: _description_
    """
    if random_capitalization and (random.random() < 0.5):
        app_name = app_name.lower()
        object_name = object_name.lower()
    # RETRIEVEALL, UPDATEALL, DELETEALL
    if operation_name.endswith("ALL"):
        return f'{operation_name.split("ALL")[0].capitalize()} all {app_name} {object_name}.'
    # UPSERTWITHWHERE
    elif operation_name.endswith("UPSERTWITHWHERE"):
        return f"Update or create a {app_name} {object_name}."
    elif operation_name.endswith("POLLER"):
        # CREATEDORUPDATED_POLLER
        if "OR" in operation_name.split("_")[0]:
            return f"Triggers when a {app_name} {object_name} is created or updated."
        # CREATED_POLLER, UPDATED_POLLER
        else:
            return f'Triggers when a {app_name} {object_name} is {operation_name.split("_")[0].lower()}.'
    # CREATE, UPDATE
    elif operation_name in actions:
        return f"{operation_name.capitalize()} a {app_name} {object_name}."
    # CREATED, UPDATED, DELETED
    elif operation_name in triggers:
        return f"Triggers when a {app_name} {object_name} is {operation_name.lower()}."
    else:
        raise ValueError(
            f"Operation {operation_name} is not yet supported. Supported operations are {actions + triggers}."
        )


def get_if_else_utterance(
    description_action_1: str,
    description_action_2: str,
    description_trigger: str,
    condition: str = "CONDITION",
) -> str:
    """WIP utility to randomly choose between three templates to phrase if-else conditional flows.
    Generally structured as: "When TRIGGER, do ACTION1 if CONDITION, else do ACTION2", but with different
    phrasal structures.

    Args:
        description_action_1 (str): Utterance describing the first action.
        description_action_2 (str): Utterance describing the second action.
        description_trigger (str): Utterance describing the trigger.
        condition (str, optional): Word to use as condition after `if`. Defaults to "CONDITION".

    Returns:
        str: One of the three template utterances describing the if-else flow.
    """
    rfloat = random.random()

    if rfloat < 1 / 3:
        return (
            preprocess_triggers(description_trigger)
            + f', {" ".join([lemmatizer.lemmatize(description_action_1.split()[0].lower(), wordnet.VERB)] + description_action_1.split()[1:])}'.rstrip(
                "."
            )
            + f" if {condition}, else"
            + f' {" ".join([lemmatizer.lemmatize(description_action_2.split()[0].lower(), wordnet.VERB)] + description_action_2.split()[1:])}'.rstrip(
                "."
            )
        )
    elif rfloat > 2 / 3:
        return (
            f"If {condition}, "
            + preprocess_triggers(description_trigger).lower()
            + " "
            + description_action_1.lower().rstrip(".")
            + ", else "
            + description_action_2.lower().rstrip(".")
        )
    else:
        return (
            description_action_1.capitalize().rstrip(".")
            + " "
            + preprocess_triggers(description_trigger).lower()
            + f" if {condition}, else "
            + description_action_2.lower().rstrip(".")
        )


def load_ikg_data(
    source: str, product_database_path: str, application_filter: List[str]
) -> pd.DataFrame:
    """Utility for querying the IKG and retrieve all `APP.OBJECT.METHOD` triplets (="commands") and
    the descriptions given in the IKG as basic utterances.

    Args:
        source (str): Path to `join_api_fs.json`.
        product_database_path (str): Path to `deployment_targets.csv`.
        application_filter (List[str]): Filter on applications.

    Returns:
        pd.DataFrame: Dataframe containing all commands, utterances and `display_names` from the IKG.
    """
    l.info("parsing data")
    t = util.Timer()
    ikg = join_api.model.JoinApiKG_Base(product_database_path)
    source_config = _load_json(source)
    join_api.parser.parse(
        source_config, ikg, application_filter=application_filter, max_workers=1
    )
    ikg.postprocess()
    ikg.postcheck(flags=join_api.F_PATCH)
    l.info(f"parsed data in {t()}s")

    # Create mapping from `OpenAPI` command to actual operation description.
    commands, descriptions = [], []
    app_dn, object_dn, operation_dn = [], [], []  # display_names
    for _app in ikg.applications.values():
        for _obj in _app.objects.values():
            for _op in _obj.operations.values():
                app_dn.append(_app.display_name)
                object_dn.append(_obj.display_name)
                operation_dn.append(_op.display_name)
                commands.append(f"{_app.name}.{_obj.name}.{_op.name}")
                descriptions.append(_op.description)

    df_descriptions_orig = pd.DataFrame(
        {
            "command": commands,
            "description": descriptions,
            "app_display_name": app_dn,
            "object_display_name": object_dn,
            "operation_display_name": operation_dn,
        }
    )
    return df_descriptions_orig


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
    else:
        if trigger_description.endswith("s"):
            return f'For {trigger_description.lower().rstrip(".")}'
        else:
            return f'For a {trigger_description.lower().rstrip(".")}'


def sample_df_descriptions(df_descriptions_full: pd.DataFrame) -> pd.DataFrame:
    """Takes the output of `generate_templatized_utterance`, where each triplet has
    around 4 descriptions and does a length-weighted (exponential) groupby-sample. This means that for each
    command there will be at least one utterance, and the longest are most likely to be chosen.

    Args:
        df_descriptions_full (pd.DataFrame): Output dataframe of the `generate_templatized_utterance` method.

    Returns:
        pd.DataFrame: dataframe of samples, where each command represented at least once.
    """
    df_1 = df_descriptions_full.groupby("command").sample(
        1,
        weights=np.exp(
            df_descriptions_full.description.apply(lambda x: len(x.split())).values
        ),
    )
    df_rest = df_descriptions_full.loc[~df_descriptions_full.index.isin(df_1.index)]
    df_2 = (
        df_rest.groupby("command")
        .sample(
            1,
            weights=df_rest.description.apply(lambda x: len(x.split())).values,
        )
        .sample(frac=0.1)
    )
    return pd.concat([df_1, df_2], axis=0)


def sample_df_descriptions_cross(df_descriptions_full: pd.DataFrame) -> pd.DataFrame:
    """Samples from the output dataframe of `generate_templatized_utterance`, and does
    a cross-join between all actions and triggers to create all possible trigger -> action combinations.
    Utterances are generated using the action and trigger triplet descriptions + templates to combine them.

    To avoid generating too many samples, we only generate one sample per (trigger_app, action_app). This
    also results in approximately equal representation of apps in the samples.

    Args:
        df_descriptions_full (pd.DataFrame): Output dataframe of the `generate_templatized_utterance` method.

    Returns:
        pd.DataFrame: dataframe of trigger-action type samples.
    """
    df_descriptions = sample_df_descriptions(df_descriptions_full=df_descriptions_full)
    df_triggers = dd.from_pandas(
        df_descriptions[
            df_descriptions.apply(lambda x: _command_is_trigger(x.command), axis=1)
        ],
        npartitions=5,
    )
    df_actions = dd.from_pandas(
        df_descriptions[
            ~df_descriptions.apply(lambda x: _command_is_trigger(x.command), axis=1)
        ],
        npartitions=5,
    )
    df_triggers["key"] = 0
    df_actions["key"] = 0

    # cross join actions with triggers in dask
    with TqdmCallback(desc="compute"):
        df_descriptions_cross = (
            dd.merge(
                df_triggers, df_actions, on=["key"], suffixes=("_triggers", "_actions")
            )
            .compute()
            .drop("key", 1)
        )

    # sample one command per app
    df_descriptions_cross = df_descriptions_cross.groupby(
        ["app_triggers", "app_actions"]
    ).sample(1)
    # update app list
    df_descriptions_cross["app_list"] = df_descriptions_cross.progress_apply(
        lambda x: x.app_list_triggers + x.app_list_actions, axis=1
    )
    df_descriptions_cross = df_descriptions_cross.drop(
        ["app_list_triggers", "app_list_actions"], axis=1
    )
    # Single-command triplets with utterances
    df_descriptions_cross["commands_1"] = df_descriptions_cross.apply(
        lambda x: f"{x.command_triggers}\n{x.command_actions}",
        axis=1,
    )
    # lemmatize the first verb of the action. These are not always gramatically correct / fluent.
    df_descriptions_cross["utterances_1"] = df_descriptions_cross.apply(
        lambda x: preprocess_triggers(x.description_triggers)
        + f', {" ".join([lemmatizer.lemmatize(x.description_actions.split()[0].lower(), wordnet.VERB)] + x.description_actions.split()[1:])}'.rstrip(
            "."
        ),
        axis=1,
    )
    return df_descriptions_cross


def sample_df_descriptions_cross_cross(
    df_descriptions_full: pd.DataFrame,
) -> pd.DataFrame:
    """Re-samples a set of simple triplet samples and trigger-action samples and re-cross-joins them to
    create trigger-action-action type samples.
    To avoid generating too many samples, we agaoin only take one random sample per
    (trigger_app, action1_app, action2_app) set of apps.

    Args:
        df_descriptions_full (pd.DataFrame): Output dataframe of the `generate_templatized_utterance` method.

    Returns:
        pd.DataFrame: dataframe of trigger-action-action type samples.
    """
    _df_descriptions = sample_df_descriptions(df_descriptions_full=df_descriptions_full)
    l.info("df_descriptions generated")
    _df_descriptions_cross = sample_df_descriptions_cross(
        df_descriptions_full=df_descriptions_full
    )
    l.info("df_descriptions_cross generated")
    _df_descriptions_cross_cross = _df_descriptions_cross.merge(
        _df_descriptions[
            ~_df_descriptions.apply(lambda x: _command_is_trigger(x.command), axis=1)
        ],
        how="cross",
    )
    df_descriptions_cross_cross_subsampled = _df_descriptions_cross_cross.groupby(
        ["app_triggers", "app_actions", "app"]
    ).sample(1)
    df_descriptions_cross_cross_subsampled[
        "commands_2"
    ] = df_descriptions_cross_cross_subsampled.progress_apply(
        lambda x: f"{x.commands_1}\n{x.command}", axis=1
    )
    df_descriptions_cross_cross_subsampled[
        "utterances_2"
    ] = df_descriptions_cross_cross_subsampled.progress_apply(
        lambda x: x.utterances_1.rstrip(".")
        + f' and {" ".join([lemmatizer.lemmatize(x.description_actions.split()[0].lower(), wordnet.VERB)] + x.description.split()[1:])}',
        axis=1,
    )
    # update app list
    df_descriptions_cross_cross_subsampled[
        "app_list"
    ] = df_descriptions_cross_cross_subsampled.progress_apply(
        lambda x: x.app_list_x + x.app_list_y, axis=1
    )
    df_descriptions_cross_cross_subsampled = (
        df_descriptions_cross_cross_subsampled.drop(
            ["app_list_x", "app_list_y"], axis=1
        )
    )
    return df_descriptions_cross_cross_subsampled


def generate_IKG_triplet_utterances(
    source: str = "../data/join_api/join_api_fs.json",
    product_database_path: str = "../data/join_api/deployment_targets.csv",
    application_filter: List[str] = None,
) -> pd.DataFrame:
    """Function for generating four different types of utterances describing `APP.OBJECT.METHOD` triplets.
    First two are taken from the IKG, using method description and display name (calling `load_ikg_data`).
    The other two are using method-specific templates using the app and object `names` / `display_names`.

    Args:
        source (str, optional): Path to `join_api_fs.json`. Defaults to "../data/join_api/join_api_fs.json".
        product_database_path (str, optional): Path to `deployment_targets.csv`. Defaults to "../data/join_api/deployment_targets.csv".
        application_filter (List[str], optional): Filter on applications. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe containing at most 4 rows for each triplet, containing different utterances + metadata used for
        downstream processing (e.g. the apps which are used for each row).
    """
    df_descriptions_orig = load_ikg_data(
        source, product_database_path, application_filter
    )
    df_descriptions_orig.dropna(inplace=True)

    # Generate different triplet descriptions based on IKG data.
    # Create templatized utterance from triplet directly.
    df_descriptions_orig["templatized_description"] = df_descriptions_orig.apply(
        lambda x: generate_templatized_utterance(
            app_name=x.command.split(".")[0],
            object_name=x.command.split(".")[1],
            operation_name=x.command.split(".")[-1],
        ),
        axis=1,
    )
    df_descriptions_orig["command"] = df_descriptions_orig.command.apply(lambda x: x)

    # Use the app_display_name attribute instead, may be more common in natural utterances.
    df_descriptions_orig[
        "templatized_description_display_name"
    ] = df_descriptions_orig.apply(
        lambda x: generate_templatized_utterance(
            app_name=x.app_display_name,
            object_name=x.object_display_name,
            operation_name=x.command.split(".")[-1],
        ),
        axis=1,
    )

    # Collect the 4 different utterances for each triplet
    df_descriptions = pd.DataFrame(columns=["command", "description"])
    df_descriptions = pd.concat(
        [
            # Uses operation.description as utterance
            df_descriptions_orig[df_descriptions_orig["description"] != ""][
                ["command", "description"]
            ],
            # uses display_name of the operation function as utterance
            df_descriptions_orig[["command", "operation_display_name"]].rename(
                columns={"operation_display_name": "description"}
            ),
            # templatized utterance using app, bo, method from triplet
            df_descriptions_orig[["command", "templatized_description"]].rename(
                columns={"templatized_description": "description"}
            ),
            # templatized utterance using app, bo display_name from IKG
            df_descriptions_orig[
                ["command", "templatized_description_display_name"]
            ].rename(columns={"templatized_description_display_name": "description"}),
        ]
    )

    # add app_display_name and object_display_name to the description dataframe.
    df_descriptions = df_descriptions.merge(
        df_descriptions_orig[["command", "app_display_name", "object_display_name"]],
        how="left",
        on="command",
    )
    # Sample one utterance per command
    df_descriptions["app"] = df_descriptions.command.apply(lambda x: x.split(".")[0])
    df_descriptions["object"] = df_descriptions.command.apply(lambda x: x.split(".")[1])
    df_descriptions["app_list"] = df_descriptions.apply(lambda x: [x.app], axis=1)
    # remove whitespace descriptions
    df_descriptions = df_descriptions[
        df_descriptions.apply(lambda x: x.description.strip() != "", axis=1)
    ]
    # remove single-word descriptions
    df_descriptions = df_descriptions.loc[
        ~df_descriptions.description.apply(lambda x: len(x.split())) < 2
    ]
    return df_descriptions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="../../data/join_api/0421_length_weighted_sampling_full.pkl",
        help="The path to the file where the pickled dataframe should be stored.",
    )
    args = parser.parse_args()

    # TYPE 1: Load all triplets and types of descriptions.
    DF_DESCRIPTIONS_FULL = generate_IKG_triplet_utterances()
    # sample one kind of utterance for each triplet from DF_DESCRIPTIONS_FULL
    df_descriptions = sample_df_descriptions(df_descriptions_full=DF_DESCRIPTIONS_FULL)
    # TYPE 2: sample a cross-join between all action- and trigger-triplets
    # note, this internally again calls sample_df_descriptions to avoid bias.
    df_descriptions_cross = sample_df_descriptions_cross(
        df_descriptions_full=DF_DESCRIPTIONS_FULL
    )
    # TYPE 3: Sample a cross-join between trigger-actions and actions.
    # again, internally again resamples df_descriptions and df_descriptions_cross to avoid bias.
    df_descriptions_cross_cross_subsampled = sample_df_descriptions_cross_cross(
        df_descriptions_full=DF_DESCRIPTIONS_FULL
    )

    # TYPE 4: `ìf-else` statements: Send Gmail message for a new Salesforce Lead or a Slack message if email address is missing.
    # TODO: can we generate conditions relating to if statement: properties of operations
    _condition = "CONDITION"
    df_descriptions_cross_cross_subsampled_ = sample_df_descriptions_cross_cross(
        df_descriptions_full=DF_DESCRIPTIONS_FULL
    )

    df_descriptions_cross_cross_subsampled_[
        "commands_3"
    ] = df_descriptions_cross_cross_subsampled_.progress_apply(
        lambda x: f"{x.command_triggers}"
        + f'\nif {_condition.replace(" ", "_")}:'
        + f"\n\t{x.command_actions}"
        + "\nelse:"
        + f"\n\t{x.command}",
        axis=1,
    )
    df_descriptions_cross_cross_subsampled_[
        "utterances_3"
    ] = df_descriptions_cross_cross_subsampled_.progress_apply(
        lambda x: get_if_else_utterance(
            x.description_actions,
            x.description,
            x.description_triggers,
            condition=_condition,
        ),
        axis=1,
    )

    # TYPE 5: for-loops: Copy all files from GoogleDrive to Dropbox. Might only include RETRIEVEALL actions.
    # Reduce complexity a bit, shuffle and take top 1000, only affects around 5 apps.
    _df_descriptions = sample_df_descriptions(df_descriptions_full=DF_DESCRIPTIONS_FULL)
    df_retrieveall = _df_descriptions[
        ~_df_descriptions.apply(lambda x: _command_is_trigger(x.command), axis=1)
    ][
        _df_descriptions[
            ~_df_descriptions.apply(lambda x: _command_is_trigger(x.command), axis=1)
        ].apply(lambda x: x.command.split(".")[-1] == "RETRIEVEALL", axis=1)
    ]
    df_create = _df_descriptions[
        ~_df_descriptions.apply(lambda x: _command_is_trigger(x.command), axis=1)
    ][
        _df_descriptions[
            ~_df_descriptions.apply(lambda x: _command_is_trigger(x.command), axis=1)
        ].apply(lambda x: x.command.split(".")[-1] == "CREATE", axis=1)
    ]
    dd_retrieveall = dd.from_pandas(df_retrieveall, npartitions=5)
    dd_create = dd.from_pandas(df_create, npartitions=5)
    dd_retrieveall["key"] = 0
    dd_create["key"] = 0

    with TqdmCallback(desc="compute"):
        df_descriptions_cross_actions = (
            dd.merge(
                dd_retrieveall, dd_create, on=["key"], suffixes=("_action1", "_action2")
            )
            .compute()
            .drop("key", 1)
        )

    # For loops only for RETRIEVEALL-CREATE action combinations for now.
    df_descriptions_cross_retrieveall_create = df_descriptions_cross_actions.groupby(
        ["app_action1", "app_action2"]
    ).sample(1)
    df_descriptions_cross_retrieveall_create[
        "utterances_4"
    ] = df_descriptions_cross_retrieveall_create.progress_apply(
        lambda x: f'For each {x.command_action1.split(".")[0]} {lemmatizer.lemmatize(x.command_action1.split(".")[1].lower(), wordnet.NOUN)}, '
        + f'{" ".join([lemmatizer.lemmatize(x.description_action2.split()[0].lower(), wordnet.VERB)] + x.description_action2.split()[1:])}'.rstrip(
            "."
        ),
        axis=1,
    )
    df_descriptions_cross_retrieveall_create[
        "commands_4"
    ] = df_descriptions_cross_retrieveall_create.progress_apply(
        lambda x: f"for {_get_iterator_variable(x)} in {x.command_action1}:"
        + f"\n\t{x.command_action2}",
        axis=1,
    )

    # Update app list.
    df_descriptions_cross_retrieveall_create[
        "app_list"
    ] = df_descriptions_cross_retrieveall_create.progress_apply(
        lambda x: x.app_list_action1 + x.app_list_action2, axis=1
    )
    df_descriptions_cross_retrieveall_create = (
        df_descriptions_cross_retrieveall_create.drop(
            ["app_list_action1", "app_list_action2"], axis=1
        )
    )

    # TYPE 6: SYNC: such as `Sync Salesforce new cases with ServiceNow tickets`. It's a sync between objects.
    # We don't use the sampled utterances, just the object names. No need to resample `df_descriptions`.
    df_objects = pd.DataFrame(
        df_descriptions.apply(
            lambda x: ".".join(x.command.split(".")[:2]), axis=1
        ).unique(),
        columns=["objects"],
    )
    df_objects["app"] = df_objects.objects.apply(lambda x: x.split(".")[0])
    df_objects["object"] = df_objects.objects.apply(lambda x: x.split(".")[-1])

    # add display names for utterance generation
    df_objects = df_objects.merge(
        df_descriptions.drop_duplicates(subset=["app", "object"])[
            ["app", "object", "app_display_name", "object_display_name"]
        ],
        how="left",
        on=["app", "object"],
    )
    # cross-merge apps
    df_objects_cross = df_objects.merge(
        df_objects, how="cross", suffixes=("_new", "_to_sync")
    )
    df_objects_cross_sample = df_objects_cross.groupby(["app_new", "app_to_sync"]).sample(1)
    df_objects_cross_sample["utterances_5"] = df_objects_cross_sample.progress_apply(
        lambda x: f'Sync new {x.app_display_name_new} {x.object_display_name_new.rstrip("s").lower()}s with {x.app_display_name_to_sync} {x.object_display_name_to_sync.rstrip("s").lower()}s',
        axis=1,
    )
    df_objects_cross_sample["commands_5"] = df_objects_cross_sample.progress_apply(
        lambda x: f"{x.objects_new}.CREATED\n{x.objects_new}.RETRIEVEALL\n{x.objects_to_sync}.CREATE",
        axis=1,
    )
    df_objects_cross_sample["app_list"] = df_objects_cross_sample.apply(
        lambda x: [x.objects_new.split(".")[0]] + [x.objects_to_sync.split(".")[0]], axis=1
    )

    # TYPE 7: MOVE: same objects, but moved across different apps.
    # get names of business objects which appear in different apps
    _reoccurring_bos = [
        item
        for item, count in Counter(
            [o.split(".")[-1].lower() for o in df_objects.objects.unique()]
        ).items()
        if count > 1
    ]
    df_objects_reoccurring = df_objects[
        df_objects.objects.apply(lambda x: x.split(".")[-1].lower()).isin(_reoccurring_bos)
    ]
    df_objects_reoccurring["bo"] = df_objects_reoccurring.apply(
        lambda x: x.objects.split(".")[-1].lower(), axis=1
    )
    df_objects_cross_sample_match = df_objects_reoccurring.merge(
        df_objects_reoccurring, how="inner", on="bo", suffixes=("_new", "_to_sync")
    )
    df_objects_cross_sample_match = df_objects_cross_sample_match[
        df_objects_cross_sample_match.apply(lambda x: x.app_new != x.app_to_sync, axis=1)
    ]
    df_objects_cross_sample_match[
        "utterances_6"
    ] = df_objects_cross_sample_match.progress_apply(
        lambda x: f'Move {x.object_display_name_new.rstrip("s").lower()}s from {x.app_display_name_new} to {x.app_display_name_to_sync}',
        axis=1,
    )
    df_objects_cross_sample_match["commands_6"] = df_objects_cross_sample_match.apply(
        lambda x: f'for {x.objects_new.split(".")[-1]} in {x.objects_new}.RETRIEVEALL:\n\t{x.objects_to_sync}.CREATE\n{x.objects_new}.DELETEALL',
        axis=1,
    )
    df_objects_cross_sample_match["app_list"] = df_objects_cross_sample_match.apply(
        lambda x: [x.objects_new.split(".")[0]] + [x.objects_to_sync.split(".")[0]], axis=1
    )

    # TYPE 8: COPY: Same as MOVE, but without the DELETEALL.
    df_objects_cross_sample_match[
        "utterances_7"
    ] = df_objects_cross_sample_match.progress_apply(
        lambda x: f'Copy {x.object_display_name_new.rstrip("s").lower()}s from {x.app_display_name_new} to {x.app_display_name_to_sync}',
        axis=1,
    )
    df_objects_cross_sample_match["commands_7"] = df_objects_cross_sample_match.apply(
        lambda x: f'for {x.objects_new.split(".")[-1].lower()} in {x.objects_new}.RETRIEVEALL:\n\t{x.objects_to_sync}.CREATE',
        axis=1,
    )

    _num_samples_per_app = 250
    # Compile all samples, shuffle and save.
    df_out = pd.concat(
        [
            df_descriptions[["description", "command", "app_list"]]
            .rename(columns={"description": "utterances", "command": "commands"})
            .assign(type=1),
            df_descriptions_cross[["utterances_1", "commands_1", "app_list"]]
            .rename(columns={"utterances_1": "utterances", "commands_1": "commands"})
            .assign(type=2),
            df_descriptions_cross_cross_subsampled.groupby("app_triggers")
            .sample(_num_samples_per_app, random_state=0)[["utterances_2", "commands_2", "app_list"]]
            .rename(columns={"utterances_2": "utterances", "commands_2": "commands"})
            .assign(type=3),
            df_descriptions_cross_cross_subsampled_.groupby("app_triggers")
            .sample(_num_samples_per_app, random_state=0)[["utterances_3", "commands_3", "app_list"]]
            .rename(columns={"utterances_3": "utterances", "commands_3": "commands"})
            .assign(type=4),
            df_descriptions_cross_retrieveall_create[
                ["utterances_4", "commands_4", "app_list"]
            ]
            .rename(columns={"utterances_4": "utterances", "commands_4": "commands"})
            .assign(type=5),
            df_objects_cross_sample[["utterances_5", "commands_5", "app_list"]]
            .rename(columns={"utterances_5": "utterances", "commands_5": "commands"})
            .assign(type=6),
            df_objects_cross_sample_match[["utterances_6", "commands_6", "app_list"]]
            .rename(columns={"utterances_6": "utterances", "commands_6": "commands"})
            .assign(type=7),
            df_objects_cross_sample_match[["utterances_7", "commands_7", "app_list"]]
            .rename(columns={"utterances_7": "utterances", "commands_7": "commands"})
            .assign(type=8),
        ]
    ).reset_index(drop=True)

    df_out.sample(frac=1.0, random_state=0).to_pickle(
        args.output_dir
    )
