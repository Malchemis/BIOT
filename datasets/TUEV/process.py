import mne
import numpy as np
import os
import pickle
from tqdm import tqdm

"""
https://github.com/Abhishaike/EEG_Event_Classification
"""


def BuildEvents(signals, times, EventData):
    # numEvents is equal to the number of rows of the .rec file
    [numEvents, z] = EventData.shape
    # fs is the sampling frequency
    fs = 250.0
    # numChan is the number of channels
    [numChan, numPoints] = signals.shape

    # standardize each channel
    # for i in range(numChan):
    #     if np.std(signals[i, :]) > 0:
    #         signals[i, :] = (signals[i, :] - np.mean(signals[i, :])) / np.std(signals[i, :])

    # features is a 3D array of shape (numEvents, numChan, fs * 5) for 5-second windows
    features = np.zeros([numEvents, numChan, int(fs) * 5])
    # offending_channel is the channel that caused the event
    offending_channel = np.zeros([numEvents, 1])
    # As many labels as events
    labels = np.zeros([numEvents, 1])
    # offset is the number of points in the signal
    offset = numPoints
    # Repeat the signals 3 times along the time axis to handle beginning and ending windows
    signals = np.concatenate([signals, signals, signals], axis=1)
    # For each event
    for i in range(numEvents):
        # Get the channel id, start and end times
        chan = int(EventData[i, 0])
        start = np.where((times) >= EventData[i, 1])[0][0] # take the first timestamp which is greater than the event start time
        end = np.where((times) >= EventData[i, 2])[0][0] # take the first timestamp which is greater than the event end time
        # print (offset + start - 2 * int(fs), offset + end + 2 * int(fs), signals.shape)
        # Extract a 5-second window in the "middle" signal (the second one) for each channel
        features[i, :] = signals[
            :, offset + start - 2 * int(fs) : offset + end + 2 * int(fs)
        ]
        offending_channel[i, :] = int(chan)
        labels[i, :] = int(EventData[i, 3]) # for this event, get
    return [features, offending_channel, labels]


def convert_signals(signals, Rawdata):
    signal_names = {
        k: v
        for (k, v) in zip(
            Rawdata.info["ch_names"], list(range(len(Rawdata.info["ch_names"])))
        )
    }
    # Creates a bipolar montage
    new_signals = np.vstack(
        (
            signals[signal_names["EEG FP1-REF"]]
            - signals[signal_names["EEG F7-REF"]],  # 0
            (
                signals[signal_names["EEG F7-REF"]]
                - signals[signal_names["EEG T3-REF"]]
            ),  # 1
            (
                signals[signal_names["EEG T3-REF"]]
                - signals[signal_names["EEG T5-REF"]]
            ),  # 2
            (
                signals[signal_names["EEG T5-REF"]]
                - signals[signal_names["EEG O1-REF"]]
            ),  # 3
            (
                signals[signal_names["EEG FP2-REF"]]
                - signals[signal_names["EEG F8-REF"]]
            ),  # 4
            (
                signals[signal_names["EEG F8-REF"]]
                - signals[signal_names["EEG T4-REF"]]
            ),  # 5
            (
                signals[signal_names["EEG T4-REF"]]
                - signals[signal_names["EEG T6-REF"]]
            ),  # 6
            (
                signals[signal_names["EEG T6-REF"]]
                - signals[signal_names["EEG O2-REF"]]
            ),  # 7
            (
                signals[signal_names["EEG FP1-REF"]]
                - signals[signal_names["EEG F3-REF"]]
            ),  # 14
            (
                signals[signal_names["EEG F3-REF"]]
                - signals[signal_names["EEG C3-REF"]]
            ),  # 15
            (
                signals[signal_names["EEG C3-REF"]]
                - signals[signal_names["EEG P3-REF"]]
            ),  # 16
            (
                signals[signal_names["EEG P3-REF"]]
                - signals[signal_names["EEG O1-REF"]]
            ),  # 17
            (
                signals[signal_names["EEG FP2-REF"]]
                - signals[signal_names["EEG F4-REF"]]
            ),  # 18
            (
                signals[signal_names["EEG F4-REF"]]
                - signals[signal_names["EEG C4-REF"]]
            ),  # 19
            (
                signals[signal_names["EEG C4-REF"]]
                - signals[signal_names["EEG P4-REF"]]
            ),  # 20
            (
                signals[signal_names["EEG P4-REF"]]
                - signals[signal_names["EEG O2-REF"]]
            ),  # 21
        )
    )
    return new_signals


def readEDF(fileName):
    # Read the EDF file
    Rawdata = mne.io.read_raw_edf(fileName)
    # Extract the signals and the timestamps
    signals, times = Rawdata[:]
    # Read the corresponding events
    RecFile = fileName[0:-3] + "rec"
    eventData = np.genfromtxt(RecFile, delimiter=",")
    Rawdata.close()
    # Return the signals, the timestamps, the events and the raw data
    return [signals, times, eventData, Rawdata]


def load_up_objects(BaseDir, Features, OffendingChannels, Labels, OutDir):
    for dirName, subdirList, fileList in tqdm(os.walk(BaseDir)):
        print("Found directory: %s" % dirName)
        for fname in fileList:
            if fname[-4:] == ".edf":
                print("\t%s" % fname)
                try:
                    # read
                    [signals, times, event, Rawdata] = readEDF(
                        dirName + "/" + fname
                    )  # event is the .rec file in the form of an array
                    # bipolar montage
                    signals = convert_signals(signals, Rawdata)
                except (ValueError, KeyError):
                    print("something funky happened in " + dirName + "/" + fname)
                    continue

                # get the features (for each event and channel, we have a 5-second window), offending channels and labels
                signals, offending_channels, labels = BuildEvents(signals, times, event)

                for idx, (signal, offending_channel, label) in enumerate(
                    zip(signals, offending_channels, labels)
                ):
                    sample = {
                        "signal": signal,
                        "offending_channel": offending_channel,
                        "label": label,
                    }
                    save_pickle(
                        sample,
                        os.path.join(
                            OutDir, fname.split(".")[0] + "-" + str(idx) + ".pkl"
                        ),
                    )

    return Features, Labels, OffendingChannels


def save_pickle(object, filename):
    with open(filename, "wb") as f:
        pickle.dump(object, f)


"""
TUEV dataset is downloaded from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
"""

root = "/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf"
train_out_dir = os.path.join(root, "processed_train")
eval_out_dir = os.path.join(root, "processed_eval")
if not os.path.exists(train_out_dir):
    os.makedirs(train_out_dir)
if not os.path.exists(eval_out_dir):
    os.makedirs(eval_out_dir)

BaseDirTrain = os.path.join(root, "train")
fs = 250
TrainFeatures = np.empty(
    (0, 16, fs)
)  # 0 for lack of intialization, 22 for channels, fs for num of points
TrainLabels = np.empty([0, 1])
TrainOffendingChannel = np.empty([0, 1])
load_up_objects(
    BaseDirTrain, TrainFeatures, TrainLabels, TrainOffendingChannel, train_out_dir
)

BaseDirEval = os.path.join(root, "eval")
fs = 250
EvalFeatures = np.empty(
    (0, 16, fs)
)  # 0 for lack of intialization, 22 for channels, fs for num of points
EvalLabels = np.empty([0, 1])
EvalOffendingChannel = np.empty([0, 1])
load_up_objects(
    BaseDirEval, EvalFeatures, EvalLabels, EvalOffendingChannel, eval_out_dir
)
