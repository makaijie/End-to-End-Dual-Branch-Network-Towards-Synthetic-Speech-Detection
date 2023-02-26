import librosa.display
import soundfile

if __name__ == '__main__':
    with open('LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt', 'r') as f:
        audio_info = [info.strip().split() for info in f.readlines()]

    out = 'LA_Slience/train/'
    for i in range(len(audio_info)):
        print(i)
        _, utterance_id, _, _, _ = audio_info[i]
        utterance_path = 'LA/ASVspoof2019_LA_train/flac/' + utterance_id + '.flac'

        x,sr = librosa.load(utterance_path,sr=16000)
        x, _ = librosa.effects.trim(x,top_db=40)
        out_path = out + utterance_id + '.flac'
        soundfile.write(out_path,x,sr)


