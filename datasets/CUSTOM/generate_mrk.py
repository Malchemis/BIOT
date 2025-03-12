import mne
import pandas as pd
import os
import os.path as op
import numpy as np
import sys
import pickle

def save_mrk_file(filepath,raw_name,onset_list_detected_spikes,onset_list_true_spikes):
    """if os.path.exists(filepath+'/MarkerFile.mrk'):
        with open(filepath+'/MarkerFile.mrk', 'r') as f:
            data = f.read()
            if data != "":
                print(data)
                if "/BrainTV" in data:
                    print("renaming "+filepath)
                    os.rename(filepath+'/MarkerFile.mrk', filepath+'/OldMarkerFile.mrk')"""
    print(filepath+'/'+raw_name[:-3]+'.mrk')
    with open(filepath+'/'+raw_name[:-3]+'.mrk', 'w') as f:
        if (len(onset_list_detected_spikes)>0 and len(onset_list_true_spikes)>0):
            f.write('PATH OF DATASET:\n'+filepath+' \n\n\nNUMBER OF MARKERS:\n2\n\n\n')
            f.write('CLASSGROUPID:\n3\nNAME:\ntrue_spike\nCOMMENT:\n\nCOLOR:\ngreen\nEDITABLE:\nYes\nCLASSID:\n1\nNUMBER OF SAMPLES:\n' + str(len(onset_list_true_spikes)) + '\nLIST OF SAMPLES:\nTRIAL NUMBER		TIME FROM SYNC POINT (in seconds)\n')
            for annot in onset_list_true_spikes:
                f.write('                  +0				     +'+str(annot))
                f.write('\n')
            f.write('\n\n')
            f.write('CLASSGROUPID:\n+3\nNAME:\ndetected_spike\nCOMMENT:\n\nCOLOR:\nred\nEDITABLE:\nYes\nCLASSID:\n+2\nNUMBER OF SAMPLES:\n+' + str(len(onset_list_detected_spikes)) + '\nLIST OF SAMPLES:\nTRIAL NUMBER		TIME FROM SYNC POINT (in seconds)\n')
            for annot in onset_list_detected_spikes:
                f.write('                  +0				     +'+str(annot))
                f.write('\n')
            f.write('\n\n')
        elif (len(onset_list_detected_spikes) == 0 or len(onset_list_true_spikes) == 0):
            f.write('PATH OF DATASET:\n'+filepath+' \n\n\nNUMBER OF MARKERS:\n1\n\n\n')
            if len(onset_list_detected_spikes) == 0:
                onset_list = onset_list_true_spikes
                annot_name = 'true_spike'
            elif len(onset_list_true_spikes) == 0:
                onset_list = onset_list_detected_spikes
                annot_name = 'detected_spike'
            f.write('CLASSGROUPID:\n3\nNAME:\n'+annot_name+'\nCOMMENT:\n\nCOLOR:\ngreen\nEDITABLE:\nYes\nCLASSID:\n1\nNUMBER OF SAMPLES:\n' + str(len(onset_list_true_spikes)) + '\nLIST OF SAMPLES:\nTRIAL NUMBER		TIME FROM SYNC POINT (in seconds)\n')
            for annot in onset_list:
                f.write('                  +0				     +'+str(annot))
                f.write('\n')


#df = pd.read_csv("C:/Users/pauli/Documents/MEGBase/results/150/imbalanced data/OneStageTrainingFL/100_patients/b3/sfcn/fullytrained2DbigK5/IterativeLearning/9thModel/sfcn_time__cvpredictions.csv")#"C:/Users/pauli/Documents/MEGBase/testsAgnes/max10_ae1D_4block.csv")#"/Users/pauli/Documents/MEGBase/data/outputANR2/sfcn_time__cvpredictions.csv")
df = pd.read_csv("C:/Users/pauli/Documents/MEGBase/results/ReannotPaper/poubelle4/features_only_fl_newvalid_0/_cvpredictions_features_only_last_model.csv")

# open a file containing the good 274 channels
with open('/Users/pauli/Documents/MEGBase/scripts/good_channels', 'rb') as fp:
    good_channels = pickle.load(fp)

#file containing the data_raw_nb for all test subjects
with open("C:/Users/pauli/Documents/MEGBase/ALlists48/it3/testing_data33_AL.txt", 'r') as f:#'E:/BlackHardDrive/testAgnes/p_list.txt'
    sub_ids = [line.strip().split('_')[2] for line in f]
#sub_ids = ['242','320']
sub_ids = ['122']
sub_ids=sorted(list(map(int, sub_ids)))

for sub in sub_ids:
    print("sub")
    print(sub)
    sub_df = df.loc[df['subject'] == sub]

    if(len(sub_df.index)>=0):
        with open("E:/BlackHardDrive/MEG_PHRC_2006/patient_list.txt") as f:
            lines = f.readlines()
        path = lines[int(sub)].rstrip()
        raw_names = [r for r in os.listdir(path) if (('Epi' in r) and ('.ds' in r) and not ('._' in r))]
        # Specific to ANR
        #path = '/Users/pauli/Documents/MEGBase/data/ANR/P'+str(sub)
        #raw_names = [r for r in os.listdir(path) if ('S.ds' in r)]
        raw_names_pkl = raw_names.copy()

        #######################################keep only blocks with annots in raw_names######################
        #######################################starting from batch 7, we keep also blocks with no annot so these below lines of code are commented
        """raw_names.sort()
        print(raw_names)
        for r in raw_names:
            try:
                print(r)
                raw = mne.io.read_raw_ctf(path+'/'+r, preload=True, verbose=False)
                raw.pick_types(meg=True, ref_meg=False)
                raw.pick_channels(ch_names=good_channels)
                # get events and keep spikes only
                events = mne.events_from_annotations(raw)
                with open('C:/Users/pauli/Documents/MEGBase/results/All_labels.txt') as f:
                    spike_labels = f.readlines()
                spike_labels = [sub.replace('\n', '') for sub in spike_labels]
                #spike_labels = ['event']
                print(len(raw.info['ch_names']))
                spike_annot = list(set(spike_labels).intersection(events[1]))
                #Comment out if section below if some blocks with no spikes were kept but .ds with errors at opening were not considered
                #if (not spike_annot):
                #	raw_names_pkl.remove(r)
                #elif (len(raw.info['ch_names'])!=274):
                #	raw_names_pkl.remove(r)
                #	print("REMOVED:", r)

            except:
                err_type, error, traceback = sys.exc_info()
                print('error: {err}'.format(err=error))
                raw_names_pkl.remove(r)
                pass"""

        raw_names_pkl.sort()
        print("raw_names_pkl")
        print(raw_names_pkl)
        for block in sub_df["block"].unique().tolist():
            print(raw_names_pkl)
            print("block")
            print(raw_names_pkl[block])
            raw = mne.io.read_raw_ctf(path+'/'+raw_names_pkl[block], preload=True, verbose=False)
            raw.resample(150)
            # get events and keep spikes only
            events = mne.events_from_annotations(raw)
            #spike_labels = ['event']
            with open('C:/Users/pauli/Documents/MEGBase/results/All_labels.txt') as f:
                spike_labels = f.readlines()
            spike_labels = [sub.replace('\n', '') for sub in spike_labels]
            spike_annot = list(set(spike_labels).intersection(events[1]))
            print("############# annotations:", spike_annot)
            events_spikes=np.empty(0)
            if spike_annot:
                for i,iname in enumerate(spike_annot):
                    spike_trigger = events[1][spike_annot[i]]
                    print(spike_trigger)
                    events_spikes=np.concatenate((events_spikes,(events[0][np.where(events[0][:, 2] == spike_trigger)][:, 0])),axis=0)
            print(events_spikes)

            current_block_df = sub_df.loc[sub_df['block'] == block]
            detected_spike_only = current_block_df.loc[current_block_df['pred'] == 1]
            true_spike_only = current_block_df.loc[current_block_df['test'] == 1]

            onset_true_spike = (true_spike_only['timing']/150).tolist()#(events_spikes/150).tolist()
            onset_annotated_spikes = (events_spikes/150).tolist()
            onset_detected_spikes = (detected_spike_only['timing']/150).tolist()
            """my_onset = onset_detected_spikes+onset_true_spike
            onset_original_annot = onset_annotated_spikes
            my_duration=[0]*(len(detected_spike_only['timing'])+len(true_spike_only['timing']))#+len(events_spikes))
            duration_original_annot = [0]*len(events_spikes)
            my_description=(detected_spike_only['pred'].astype(int)).tolist()+(['spike']*len((true_spike_only['test'].astype(int)).tolist()))
            description_original_annot = (['spike']*len((onset_detected_spikes)))

            my_annot = mne.Annotations(onset=my_onset, duration=my_duration, description=my_description)
            original_annot = mne.Annotations(onset=onset_original_annot, duration=duration_original_annot, description=description_original_annot)
            """
            print(path)
            path_to_save = path.replace('MEG_PHRC_2006','IterativeLearningNewAnnots/testPapier')#'IterativeLearningFeedback9'
            #path_to_save = path
            print(raw_names_pkl[block])
            save_mrk_file(path_to_save,raw_names_pkl[block],onset_detected_spikes,onset_true_spike)#onset_annotated_spikes)

            """raw.set_annotations(my_annot)
            raw.plot()
            input("Press Enter to continue...")"""