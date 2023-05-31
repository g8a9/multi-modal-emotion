import os
import sys
sys.path.insert(0,"/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/") 
__package__ = 'TripleModels'
from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")

from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import pandas as pd
from tqdm import tqdm


def main2():


    class CustomDiskWriter(KeyFrameDiskWriter):
        """Custom disk writer to save filename differently

        :param KeyFrameDiskWriter: Writer class to overwrite
        :type KeyFrameDiskWriter: Writer
        """

        # This method is used to generate output filename for a keyframe
        # Here we call the super on te base class and add a suffix

        def __init__(self ,  location, file_ext=".jpeg"):
            self.output_dir_path = location
            self.file_ext = file_ext
            self.check = 0

        def generate_output_filename(self, filepath, keyframe_number):
            """Custom output filename method

            :param filepath: [description]
            :type filepath: [type]
            """
            filename = keyframe_number

            vid_name = filepath.split("/")[-1].split(".")[0]

            isExist = os.path.exists(f"{self.output_dir_path}/{vid_name}")
            if not isExist:

            # Create a new directory because it does not exist
                os.makedirs(f"{self.output_dir_path}/{vid_name}")
            

            return f"{vid_name}/{filename}"

    # initialize video module
    vd = Video()

    # number of images to be returned
    no_of_frames_to_returned = 16

    df = pd.read_pickle("/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/data/text_audio_video_emotion_data_1.pkl")

    df_false = df[df["folder_created_check"] == False]
    vid_path_list  = df_false['video_path'].tolist()


    test_writer = CustomDiskWriter(f "/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/data/test_KeyFrameFolder")
    train_writer = CustomDiskWriter(f"/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/data/train_KeyFrameFolder")
    val_writer = CustomDiskWriter(f"/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/data/val_KeyFrameFolder")

    for path in tqdm(vid_path_list):
        name = path.split("/")[-2].split("_")[0]
        writer = train_writer if name == "train" else val_writer if name == "val" else test_writer
        vd.extract_video_keyframes(
        no_of_frames=no_of_frames_to_returned, file_path=path,
        writer=writer
        )




    # # extract keyframes and process data with diskwriter
    # dir = ["/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/data/train_video","/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/data/val_video","/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/data/test_video"]

    # for path in dir:
    #     name = path.split("/")[-1].split("_")[0]
    #     vd.extract_keyframes_from_videos_dir(
    #     no_of_frames=no_of_frames_to_returned, dir_path=path,
    #     writer=CustomDiskWriter(f"/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/data/{name}_KeyFrameFolder")
    #     )

    
if __name__ == '__main__':
    main2()


# [14, 7, 13, 9, 11, 3, 19, 2, 6, 14, 6, 9, 12, 4, 5, 22, 6, 10, 10, 1, 12, 13, 4, 8, 9, 14, 14, 2, 6, 13, 15, 4, 15, 11, 14, 5, 6, 2]
# [14, 21, 34, 43, 54, 57, 76, 78, 84, 98, 104, 113, 125, 129, 134, 156, 162, 172, 182, 183, 195, 208, 212, 220, 229, 243, 257, 259, 265, 278, 293, 297, 312, 323, 337, 342, 348, 350]

