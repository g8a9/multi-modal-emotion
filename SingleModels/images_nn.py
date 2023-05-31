if __name__ == '__main__' and not __package__:
    import sys
    sys.path.insert(0, "/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/")
    __package__ = 'SingleModels'
import torch

from num2words import num2words
import wandb
import numpy as np
import pandas as pd
import re


from .models.image import ImageClassification, ResnetClassification
from utils.data_loaders import ImageDataset
from .train_model.image_training import img_train, evaluate_img
from utils.global_functions import arg_parse , hidden_layer_count, Metrics
from torch.utils.data import DataLoader
from .models.image import collate_batch
from sklearn.model_selection import train_test_split

import os
from dataclasses import dataclass
import pandas as pd
import torch
import torchaudio

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start

def replace_ordinal_numbers(text):
    re_results = re.findall('(\d+(st|nd|rd|th))', text)
    for enitre_result, suffix in re_results:
        num = int(enitre_result[:-len(suffix)])
        text = text.replace(enitre_result, num2words(num, ordinal=True))
    return text


def replace_numbers(text):
    re_results = re.findall('\d+', text)
    for term in re_results:
        num = int(term)
        text = text.replace(term, num2words(num))
    return text


def convert_numbers(text):
    text = replace_ordinal_numbers(text)
    text = replace_numbers(text)

    return text

def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.full((num_frame+1, num_tokens+1), -float('inf'))
    trellis[:, 0] = 0
    for t in range(num_frame):
        trellis[t+1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
    )
    return trellis

def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When refering to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when refering to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t-1, j] + emission[t-1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t-1, j-1] + emission[t-1, tokens[j-1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t-1, tokens[j-1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j-1, t-1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        return None
        # raise ValueError('Failed to align')
    return path[::-1]

def merge_repeats(path , ratio , transcript , sr ):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(Segment(transcript[path[i1].token_index], path[i1].time_index, path[i2-1].time_index + 1, score))
        i1 = i2
    return (segments[0].start * ratio)/sr , (segments[-1].end * ratio)/sr

def get_times(path , text , model , labels , sample_rate):
    with torch.inference_mode():
        waveform, _ = torchaudio.load(path)
        # print("here" , flush = True)
        emissions, _ = model(waveform.to("cuda"))
        # print("here" , flush = True)
        emissions = torch.log_softmax(emissions, dim=-1)
        # print("here" , flush = True)
    emission = emissions[0].cpu().detach()
    rep = {',': "", 
            '!': "",
            '"': "",
            '#': "",
            '$': "",
            '%': "",
            '&': "",
            '(': "",
            ')': "",
            '*': "",
            '+': "",
            '.': "",
            '/': "",
            ':': "",
            ';': "",
            '<': "",
            '=': "",
            '>': "",
            '?': "",
            '@': "",
            '[': "",
            '\\': "",
            ']': "",
            '^': "",
            '_': "",
            '`': "",
            '{': "",
            '}': "",
            '~': "",
            '\'':"",
            }

    rep = dict((re.escape(k), v) for k, v in rep.items()) 
    #Python 3 renamed dict.iteritems to dict.items so use rep.items() for latest versions
    pattern = re.compile("|".join(rep.keys()))
    transcript = pattern.sub(lambda m: rep[re.escape(m.group(0))], text.upper())
    transc_list = transcript.split()
    for i , word in enumerate(transc_list):
        if word.isnumeric() or any(map(str.isdigit, word)):
            transc_list[i] = convert_numbers(word).replace("-" , " ").replace("," , " ").replace(" " , "|").upper()
            # transc_list[i] = str(num2words(word)).replace("-" , " ").upper()
    transcript = '|'.join(transc_list)
    transcript = transcript.replace(" " , "|")
    print(transcript , "\n" , flush=True)
    
    dictionary  = {c: i for i, c in enumerate(labels)}
    tokens = [dictionary[c] for c in transcript] # .replace(" " , "|")


    trellis = get_trellis(emission, tokens)

    path = backtrack(trellis, emission, tokens)
    if path == None:
        return None
    ratio = waveform[0].size(0) / (trellis.size(0) - 1)
    return merge_repeats(path , ratio , transcript , sample_rate )



def prepare_dataloader(df , batch_size, pin_memory=True, num_workers=4):

    # CHANGE DATASET
    dataset = ImageDataset(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, 
            num_workers=num_workers, drop_last=False, shuffle=True, 
            collate_fn=collate_batch
        )
    return dataloader

def runModel(rank , df_train , df_val , df_test, param_dict , model_param , wandb):
 

    epoch = param_dict['epoch']
    lr = param_dict['lr']
    patience = param_dict['patience']
    clip = param_dict['clip']
    T_max = param_dict['T_max']
    batch_size = param_dict['batch_size']
    weight_decay = param_dict['weight_decay']
    weights = param_dict['weights']
    label2id = param_dict['label2id']
    id2label = param_dict['id2label']

    num_labels = model_param['output_dim']
    input_dim = model_param['input_dim']
    lstm_layers = model_param['lstm_layers']
    hidden_layers = model_param['hidden_layers']

    # IMPLEMENT MODEL HERE
    #
    #
    ###

    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(rank)).to(rank) #TODO: maybe error here?
    Metric = Metrics(num_classes = num_labels, id2label = id2label , rank = rank)
    df_train = prepare_dataloader(df_train  , batch_size = batch_size ) #TODO: maybe prepare dataloader?
    df_val = prepare_dataloader(df_val  , batch_size = batch_size )
    df_test = prepare_dataloader(df_test , batch_size = batch_size )

    # MODEL NAME HERE
    # if param_dict['model'] == MODELNAME:
    model = ResnetClassification(model_param).to(rank)
    # Different Model?
    # else:
    #         model = ImageClassification(model_param)

   
    # print(f"config = \n {config} \n" , flush = True)
   
    wandb.watch(model, log = "all")  
    model = img_train(model, df_train, df_val, criterion , lr, epoch ,  weight_decay,T_max, Metric , patience , clip )
    evaluate_img(model , df_test ,  Metric)


def main():
    project_name = "MLP_test_images"
    args =  arg_parse(project_name)
   
    run = wandb.init(project=project_name, entity="ddi" , config = args)
    config = wandb.config
    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)

    # print(f"CONFIG AFTER INIT {config}")

    param_dict = {
        'epoch':config.epoch ,
        'patience':config.patience ,
        'lr': config.learning_rate ,
        'clip': config.clip ,
        'batch_size': 8,#config.batch_size,
        'weight_decay':config.weight_decay ,
        'model': config.model,
        'T_max':config.T_max ,
        'seed': config.seed,
    }

    model_param = {
        'input_dim':config.input_dim ,
        'output_dim':config.output_dim ,
        'lstm_layers':config.lstm_layers ,
        # Need to add the hidden layer count for each modality for their hidden layers 
        'hidden_layers':hidden_layer_count(config.hidden_layers) ,
    }

    df = pd.read_pickle(f"{args.dataset}.pkl")  
    #
    try:
        weights = torch.Tensor(list(dict(sorted((dict(1 - (df['emotion'].value_counts()/len(df))).items()))).values()))
        # weights = torch.Tensor([1,1,1,1,1,1,1])
        df_train, df_test,_,__ = train_test_split(df, df["emotion"], test_size = 0.25, random_state = param_dict['seed'] , stratify=df["emotion"])
        df_train, df_val,_ ,__  = train_test_split(df_train, df_train["emotion"], test_size = 0.25, random_state = param_dict['seed'] , stratify=df_train["emotion"])
        label2id = df.drop_duplicates('label').set_index('label').to_dict()['emotion']
    except:
        weights = torch.Tensor(list(dict(sorted((dict(1 - (df['label'].value_counts()/len(df))).items()))).values()))
        # weights = torch.Tensor([1,1,1,1,1,1,1])
        df_train, df_test, _, __ = train_test_split(df, df["label"], test_size = 0.25, random_state = param_dict['seed'] , stratify=df["label"])
        df_train, df_val, _,__  = train_test_split(df_train, df_train["label"], test_size = 0.25, random_state = param_dict['seed'] , stratify=df_train["label"])
        label2id = df.drop_duplicates('label').set_index('label').to_dict()['sentiment']
        label2id = {v: k for k, v in label2id.items()}
    #
    id2label = {v: k for k, v in label2id.items()}
    
    param_dict['weights'] = weights
    param_dict['label2id'] = label2id
    param_dict['id2label'] = id2label

    param_dict = {'epoch': 7, 'patience': 10, 'lr': 0.05204322366887127, 'clip': 1, 'batch_size': 8, 'weight_decay': 1e-07, 'model': 'Images', 'T_max': 10, 'seed': 96, 'weights': torch.Tensor([0.3667, 0.6333]), 'label2id': {'Negative': 0, 'Positive': 1}, 'id2label': {0: 'Negative', 1: 'Positive'}}
    model_param = {'input_dim': 3, 'output_dim': 2, 'lstm_layers': 1, 'hidden_layers': [32, 32]}

    print(f" in main \n param_dict = {param_dict} \n model_param = {model_param} \n df {args.dataset}")

    runModel("cuda",df_train , df_val , df_test ,param_dict , model_param , run )



# if __name__ == '__main__':
#     main()
from tqdm import tqdm
if __name__ == "__main__":
    df = pd.read_pickle("../../data/text_audio_video_emotion_data.pkl")
    df = df[df["size_padding"] < 999426]
    
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to("cuda")
    labels = bundle.get_labels()
    sr = bundle.sample_rate
    tqdm.pandas()
    df['timings'] = df.progress_apply(lambda x: get_times(x.audio_path , x.text, model , labels , sr) , axis = 1)
    df.to_pickle(f"../../data/text_audio_video_emotion_data_1.pkl")
    



    # print(f"Our audio file is {df.iloc[0]['audio_path']} \n ")
    # print(f"and our start and end time are {get_times(df.iloc[0]['audio_path'] , df.iloc[0]['text']  , model , labels , sr)}")
    



