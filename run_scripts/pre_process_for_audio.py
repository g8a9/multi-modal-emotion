import string
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torchaudio
from datasets import load_dataset

from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2Processor
from tqdm import tqdm
tqdm.pandas()

SAVE_PATH = "/Users/pranavsood/Documents/DDI/Gitlab/multi-modal-emotion/src/run_scripts/save_files_from_run"
ORIG = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
NEW = {v: k for k, v in ORIG.items()} # reverse orig


def get_dataset(dataset:string) -> pd.DataFrame:
    """
    Get the dataset and perform some operations to process the dataset into what the code expects later on
    """
    df = pd.read_pickle(dataset)
    # print(df)
    

    """
    **REMOVED DIA125_UTT3 AS ITS NOT IN VAL DATASET**
    **REMOVED DIA110_UTT7 AS ITS NOT IN VAL DATASET**
    """


    folder_name = "data/"
    df.rename(columns = {'y':'emotion'}, inplace = True)
    df['emotion'] = df['emotion'].map(NEW)
    
    df['path'] = folder_name + df['split'] + "_splits_wav/" + "dia" + df['dialog'].astype(str) + '_utt' + df['utterance'].astype(str) + ".wav"
    df['name'] = "dia" + df['dialog'].astype(str) + '_utt' + df['utterance'].astype(str) 
    # df.drop(columns = ["dialog" , "utterance" , "text"  , "num_words"] , inplace=True)

    df = df[ df["name"] != "dia110_utt7"] 
    df = df[ df["name"] != "dia125_utt3"] 

    


    return df

def createCSV(df):

    df = df[df['num_channels'] == 6]

    train_df = df[df['split'] == "train"]
    # train_df.drop(columns = 'split', inplace = True)

    test_df = df[df['split'] == "test"]
    # test_df.drop(columns = 'split', inplace = True)

    val_df = df[df['split'] == "val"]
    # val_df.drop(columns = 'split', inplace = True)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_df.to_csv(f"{SAVE_PATH}/train.csv", sep="\t", encoding="utf-8", index=False)
    test_df.to_csv(f"{SAVE_PATH}/test.csv", sep="\t", encoding="utf-8", index=False)
    val_df.to_csv(f"{SAVE_PATH}/val.csv", sep="\t", encoding="utf-8", index=False)

def createDatasets():
    data_files = {
    "train": f"{SAVE_PATH}/train.csv", 
    "test": f"{SAVE_PATH}/test.csv",
    "val": f"{SAVE_PATH}/val.csv",
    }

    dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
    train = dataset["train"]
    test = dataset["test"]
    eval = dataset["val"]
    return train , test , eval




def speech_file_to_array_fn(path , target_sampling_rate):
    
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def label_to_id(label, label_dict):

    if len(label_dict) > 0:
        return label_dict[label] if label in label_dict.keys() else -1

    return -1

def flatten_array_in_dict(Dict , str_):
    
  tmp = [ np.array(arr).flatten() for arr in Dict[str_] ]
  Dict[str_] = tmp
  return Dict


def preprocess_function(examples):
    
    input_column = "path"
    output_column = "emotion"
    model_path = "facebook/wav2vec2-large-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    target_sampling_rate = processor.feature_extractor.sampling_rate
    
    
    speech_list = [speech_file_to_array_fn(path , target_sampling_rate) for path in examples[input_column]]
    # print("\n \n \n \n",speech_list,"\n \n \n")
    target_list = [label_to_id(label, ORIG) for label in examples[output_column]]

    result = processor(speech_list, sampling_rate=target_sampling_rate)
    

    

    # print(f" \n \n \n \n shape:  {result['input_values'].shape} \n")
    result["labels"] = list(target_list)

    # result['input_values'] = np.array(result['input_values']).flatten()
    # print(result.keys())

    return flatten_array_in_dict(result , "input_values")





def main():
    # df = get_dataset("data/emotion_pd.pkl")

    createCSV(pd.read_pickle(f"data/emotion_pd_raw.pkl"))
    train_dataset, eval_dataset, test_dataset = createDatasets()

    train_dataset = train_dataset.map(
        preprocess_function,
        batch_size=1,
        batched=True,
        num_proc=8
    )
    train_dataset.to_pandas().to_pickle(f"{SAVE_PATH}/train_pkl.pkl")

    # eval_dataset = eval_dataset.map(
    #     preprocess_function,
    #     batch_size=1,
    #     batched=True,
    #     num_proc=8
    # )
    # eval_dataset.to_pandas().to_pickle(f"{SAVE_PATH}/val_pkl.pkl")

    # test_dataset = test_dataset.map(
    #     preprocess_function,
    #     batch_size=1,
    #     batched=True,
    #     num_proc=8
    # )
    # test_dataset.to_pandas().to_pickle(f"{SAVE_PATH}/test_pkl.pkl")

main()