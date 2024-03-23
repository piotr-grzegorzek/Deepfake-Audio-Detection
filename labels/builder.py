import os
from CONFIG import DATA_PATH
'''
This label generator is based on the ASVspoof dataset and txt files which makes it easier to get data set labels.

Read a sound source (from txt file) in raw dir -> read if sound is fake or real -> 0 = real, 1 = fake -> 
Append correct value to a list and a txt file to reuse labels quicker (if data set doesnt change)
'''
labels_array = []


def build_labels_from(root):
    global labels_array
    labels_array.clear()  # Reset a labels list (global, it must be cleared)
    file_saver = open("labels\\clean\\"+root+".txt", "w+")  # labels reuse file init

    dir_sounds = os.listdir(DATA_PATH + root)
    # Realize which txt file we need to read(Based on a sound source data set - Train/Develop/Eval)
    for sound_name in dir_sounds:
        if sound_name[3] == "T":
            find_label_from("train", sound_name, file_saver)
        elif sound_name[3] == "E":
            find_label_from("eval", sound_name, file_saver)
        else:
            find_label_from("dev", sound_name, file_saver)

    file_saver.close()
    return labels_array[:]


def find_label_from(labels_raw_name, sound_name, file_saver):
    global labels_array
    sound_name = sound_name[:12]  # remove an extension from a file name

    labels_file = open("labels\\raw\\" + labels_raw_name+".txt")
    labels_lines = labels_file.readlines()
    labels_file.close()

    found_already = False
    for line in labels_lines:
        if found_already: break
        # if sound is spoof, mark as spoof
        if line.find(sound_name) != -1 and line.find("spoof") != -1:
            labels_array.append(1)
            file_saver.write("1\n")
            found_already = True
        elif line.find(sound_name) != -1 and line.find("bonafide") != -1:
            labels_array.append(0)
            file_saver.write("0\n")
            found_already = True


def read_labels_from(root):
    labels_read = []
    file_reader = open("labels\\clean\\"+root+".txt", "r")
    samples = 0
    for line in file_reader:
        labels_read.append(int(line))
        samples += 1
    print(str(samples)+" " + root + " label samples")
    file_reader.close()
    return labels_read
