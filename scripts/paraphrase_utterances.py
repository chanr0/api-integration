import argparse
import os
import glob
import sys

import pandas as pd
import requests
from tqdm import tqdm
from thefuzz import fuzz


# append parent dir to path to import from parent (`join_api`).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from util.bam_client import BamClient

tqdm.pandas()
# set the API key beforehand using `export`
API_KEY = os.getenv("Cluster_API_KEY")


def get_paraphrase(
    utterance: str,
    prompt_template_path: str,
    temperature: float = 0.8,
) -> str:
    """Function which queries the CO Cluster BloomZ API with the utterance to be paraphrased as prompt,
    prepended with the examples stored in the txt file stored at `prompt_template_path`.

    Args:
        utterance (str): Utterance to be paraphrased.
        prompt_template_path (str): path to the txt file storing examples for few-shot paraphrase prompting.
        temperature (float, optional): Temperature used by BloomZ. Defaults to 0.8.

    Returns:
        str: _description_
    """
    bam_client = BamClient(
        api_key=API_KEY,
        prompt_template_path=prompt_template_path,
        temperature=temperature,
    )
    try:
        # Add a trigger to the utterance.
        utterance = "\n\nPhrase:\n" + utterance + "\nParaphrase:\n"
        result = bam_client.generate(utterance)
        # remove any additional sentences that might have been generated.
        result = result.split(".")[0] + "."
        return result
    except requests.exceptions.HTTPError as err:
        return utterance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="../data/0412_length_weighted_sampling_full.pkl",
        help="The path to the pickled dataframe, output of `generate_sample_data`.",
    )
    parser.add_argument(
        "--output_path",
        default="0421_df_samples_paraphrased.pkl",
        help="The path to the file where the pickled dataframe should be stored after paraphrasing.",
    )
    parser.add_argument(
        "--paraphrases_dir",
        default="../data/paraphrase_templates/",
        help="The path to the directory where the premade paraphrases are stored.",
    )
    args = parser.parse_args()

    df_samples = pd.read_pickle(args.data_path)
    df_samples["paraphrased_utterances"] = None
    df_samples["partial_ratios"] = None
    for fname in tqdm(
        sorted(glob.glob(args.paraphrases_dir + "join_api_paraphrases*.txt"))
    ):
        template_type = int(fname.rstrip(".txt").split("_")[-1])
        df_samples["paraphrased_utterances"] = df_samples.progress_apply(
            lambda x: get_paraphrase(
                x.utterances.rstrip(".") + ".", fname, temperature=0.8
            )
            if x.type == template_type
            else x.paraphrased_utterances,
            axis=1,
        )
        df_samples["partial_ratios"] = df_samples.progress_apply(
            lambda x: [
                fuzz.partial_ratio(x.paraphrased_utterances, app) for app in x.app_list
            ]
            if x.type == template_type
            else x.partial_ratios,
            axis=1,
        )

    df_samples.to_pickle(args.output_dir)

# %%
