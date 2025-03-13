import mne
import pickle
import random
from mne.io.ctf import RawCTF


# tentative function to interpolate missing channels using mne
def interpolate_missing_channels(raw: RawCTF, good_channels: list, loc_meg_channels: dict) -> RawCTF:
	# returns the list of chanel names that are present in the data
	existing_channels = raw.info['ch_names']
	# gets the list of missing channels by comparing the existing channel names with the list of good channels
	missing_channels = list(set(good_channels) - set(existing_channels))
	new_raw: RawCTF = raw.copy()

	# creates fake channels and set them to "bad channels", rename them with the name of the missing channels;
	# then mne is supposed to be able to reconstruct bad channels with "interpolate_bads"
	for miss in missing_channels:
		to_copy = random.choice(existing_channels) 	 # take the name of a random existing channel
		new_channel = raw.copy().pick([to_copy]) 	 # copy it
		new_channel.rename_channels({to_copy: miss}) # rename it with the name of the missing channel
		new_raw.add_channels([new_channel], force_update_info=True) # add the missing channel to the data

		# specifies the location of the missing channel
		for i in range(len(new_raw.info['chs'])): # chs contains a list of channel information dictionaries, one per channel.
			if new_raw.info['chs'][i]['ch_name'] == miss: # if the channel name is the name of the missing channel
				new_raw.info['chs'][i]['loc'] = loc_meg_channels[miss] # set the location of the missing channel

	# reorder the channels to have the good channels first
	new_raw.reorder_channels(good_channels)
	new_raw.info['bads'] = missing_channels # set the bad channels to the missing channels for interpolate_bads to work

	new_raw.interpolate_bads(origin=(0, 0, 0.04), reset_bads=True)
	return new_raw


if __name__ == "__main__":
	# Handle a particular subject that has strange signal on some channels
	strange_channels = ['MRO22-2805', 'MRO23-2805', 'MRO24-2805']

	# path to .ds
	from glob import glob
	raw_filenames = glob('/home/malchemis/PycharmProjects/bio-sig-analysis/data/raw/crnl-meg/sample-data/Liogier_AllDataset1200Hz/*.ds')

	# path to the file.pkl containing the list of good channels
	with open('/home/malchemis/PycharmProjects/BIOT/datasets/CUSTOM/good_channels', 'rb') as fp:
		good_channels = pickle.load(fp)

	# path to the file.pkl containing for each channel name its location
	with open('/home/malchemis/PycharmProjects/BIOT/datasets/CUSTOM/loc_meg_channels.pkl', 'rb') as fp:
		loc_meg_channels = pickle.load(fp)

	# load the raw files
	raw_files = [mne.io.read_raw_ctf(raw_filename, preload=True).pick('meg') for raw_filename in raw_filenames]

	# Drop the strange channels
	for raw in raw_files:
		raw.drop_channels(strange_channels)

	# interpolate the missing channels
	interp_raw = []
	for raw in raw_files:
		interp_raw.append(interpolate_missing_channels(raw, good_channels, loc_meg_channels))