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
        raise ValueError('Failed to align')
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
        emissions, _ = model(waveform.to("cuda"))
        emissions = torch.log_softmax(emissions, dim=-1)
    emission = emissions[0].cpu().detach()

    dictionary  = {c: i for i, c in enumerate(labels)}
    transcript = text.upper().replace(" " , "|")
    tokens = [dictionary[c] for c in transcript]


    trellis = get_trellis(emission, tokens)

    path = backtrack(trellis, emission, tokens)
    ratio = waveform[0].size(0) / (trellis.size(0) - 1)
    return merge_repeats(path , ratio , transcript , sample_rate )

if __name__ == "__main__":
    df = pd.read_pickle("../../data/text_audio_video_emotion_data.pkl")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to("cuda")
    labels = bundle.get_labels()
    sr = bundle.sample_rate
    print(f"Our audio file is {df.iloc[0]['audio_path']} \n ")
    print(f"and our start and end time are {get_times(df.iloc[0]['audio_path'][3:] , df.iloc[0]['text'] , bundle , model , labels , sr)}")
    