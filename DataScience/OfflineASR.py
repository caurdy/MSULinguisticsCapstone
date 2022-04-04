"""
Using NeMo to produce timestamps for transcription
reference: https://github.com/NVIDIA/NeMo/blob/stable/tutorials/asr/Offline_ASR.ipynb

to install proper packages for Nemo see https://github.com/NVIDIA/NeMo
can subsitute apt-get w/ 'pip install ffmpeg' and 'pip install SoundFile'

Nemo_Asr requires torch~=1.8.1 however, Pyannote requires torch>=1.9
However, it seems nemo is okay to run on a newer version of torch (at least for this script)
Currently im running torch=1.10.2
"""
import nemo.collections.asr as nemo_asr
import numpy as np
import librosa
from IPython.display import Audio, display
from plotly import graph_objects as go

# print(nemo_asr.models.EncDecCTCModel.list_available_models())
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name='QuartzNet15x5Base-En', strict=False)
AUDIO_FILENAME = '../Data/wav/f86f93f2-d49a-456d-96c5-0b3858605736.wav'


def produceTimeAlignedTranscript(audio_file: str, speaker_file: str) -> None:
    """
    Takes in an audio file path and a .rttm file path (the product of speaker diarization) to produce
    a series of json objects representing sentences with attributes
        (start, end, content, speaker, wordcount, confidence)
    :param audio_file: Path to audio file
    :param speaker_file: Path to speaker diarization file
    :return: None
    """

    # load audio signal with librosa
    signal, sample_rate = librosa.load(audio_file, sr=16000)
    transcript = ASR_MODEL.transcribe(paths2audio_files=[audio_file])[0]  # get txt transcript
    transcript = RPUNCT.punctuate(transcript)
    logits = ASR_MODEL.transcribe([audio_file], logprobs=True)[0]  # get vector of probabilities for word predictions
    probs = softmax(logits)
    # 20ms is duration of a timestep at output of the model
    time_stride = 0.02
    # get model's alphabet
    labels = list(ASR_MODEL.decoder.vocabulary) + ['blank']
    labels[0] = 'space'
    # get timestamps for space symbols
    spaces = []
    state = ''
    idx_state = 0

    if np.argmax(probs[0]) == 0:
        state = 'space'
    for idx in range(1, probs.shape[0]):
        current_char_idx = np.argmax(probs[idx])
        if state == 'space' and current_char_idx != 0 and current_char_idx != 28:
            spaces.append([idx_state, idx - 1])
            state = ''
        if state == '':
            if current_char_idx == 0:
                state = 'space'
                idx_state = idx
    if state == 'space':
        spaces.append([idx_state, len(probs) - 1])

    # calibration offset for timestamps: 180 ms
    offset = -0.18
    # split the transcript into words
    words = transcript.split()
    # cut words
    pos_prev = 0
    timestampList = []
    for j, spot in enumerate(spaces):
        pos_end = offset + (spot[0] + spot[1]) / 2 * time_stride
        timestampList.append((round(pos_prev, 3), round(pos_end, 3), words[j]))
        pos_prev = pos_end

    df = pd.read_csv(speaker_file, delimiter=' ', header=None)
    df.columns = ['Type', 'Audio File', 'IDK', 'Start Time', 'Duration', 'N/A', 'N/A', 'ID', 'N/A', 'N/A']
    df = df.drop(['IDK', 'N/A', 'N/A', 'N/A', 'N/A'], axis=1)
    sentences = []
    for _, row in df.iterrows():
        start = row.loc['Start Time']
        duration = row.loc['Duration']
        sentence = []
        for word_begin, word_end, word in timestampList:
            if start > word_begin:  # this word is before this speaker, continue
                continue
            elif start + duration < word_begin:  # this word is after the speaker, break
                break
            sentence.append(word)  # this word is in the sentence
        sentences.append(" ".join(sentence))
    df = df.assign(content=sentences)
    df.to_csv('../Data/test.csv')


def displaySignalandSpectrum(audio_file: str) -> None:
    """
    Takes in an audio file and plots the signal and spectogram in your browser.
    """
    # load audio signal with librosa
    signal, sample_rate = librosa.load(AUDIO_FILENAME, sr=16000)

    # display audio player for the signal
    display(Audio(data=signal, rate=sample_rate))

    # plot the signal in time domain
    fig_signal = go.Figure(
        go.Scatter(x=np.arange(signal.shape[0]) / sample_rate,
                   y=signal, line={'color': 'green'},
                   name='Waveform',
                   hovertemplate='Time: %{x:.2f} s<br>Amplitude: %{y:.2f}<br><extra></extra>'),
        layout={
            'height': 300,
            'xaxis': {'title': 'Time, s'},
            'yaxis': {'title': 'Amplitude'},
            'title': 'Audio Signal',
            'margin': dict(l=0, r=0, t=40, b=0, pad=0),
        }
    )
    fig_signal.show()

    # calculate amplitude spectrum
    time_stride = 0.01
    hop_length = int(sample_rate * time_stride)
    n_fft = 512
    # linear scale spectrogram
    s = librosa.stft(y=signal,
                     n_fft=n_fft,
                     hop_length=hop_length)
    s_db = librosa.power_to_db(np.abs(s) ** 2, ref=np.max, top_db=100)

    # plot the signal in frequency domain
    fig_spectrum = go.Figure(
        go.Heatmap(z=s_db,
                   colorscale=[
                       [0, 'rgb(30,62,62)'],
                       [0.5, 'rgb(30,128,128)'],
                       [1, 'rgb(30,255,30)'],
                   ],
                   colorbar=dict(
                       ticksuffix=' dB'
                   ),
                   dx=time_stride, dy=sample_rate / n_fft / 1000,
                   name='Spectrogram',
                   hovertemplate='Time: %{x:.2f} s<br>Frequency: %{y:.2f} kHz<br>Magnitude: %{z:.2f} dB<extra></extra>'),
        layout={
            'height': 300,
            'xaxis': {'title': 'Time, s'},
            'yaxis': {'title': 'Frequency, kHz'},
            'title': 'Spectrogram',
            'margin': dict(l=0, r=0, t=40, b=0, pad=0),
        }
    )
    fig_spectrum.show()


# softmax implementation in NumPy
def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum(axis=-1).reshape([logits.shape[0], 1])


# Convert our audio sample to text
files = [AUDIO_FILENAME]

# load audio signal with librosa
signal, sample_rate = librosa.load(AUDIO_FILENAME, sr=None)

transcript = asr_model.transcribe(paths2audio_files=files)[0]
# print(f'Transcript: "{transcript}"')

# let's do inference once again but without decoder
logits = asr_model.transcribe(files, logprobs=True)[0]
probs = softmax(logits)

# 20ms is duration of a timestep at output of the model
time_stride = 0.02

# get model's alphabet
labels = list(asr_model.decoder.vocabulary) + ['blank']
labels[0] = 'space'

# plot probability distribution over characters for each timestep
fig_probs = go.Figure(
    go.Heatmap(z=probs.transpose(),
               colorscale=[
                   [0, 'rgb(30,62,62)'],
                   [1, 'rgb(30,255,30)'],
               ],
               y=labels,
               dx=time_stride,
               name='Probs',
               hovertemplate='Time: %{x:.2f} s<br>Character: %{y}<br>Probability: %{z:.2f}<extra></extra>'),
    layout={
        'height': 300,
        'xaxis': {'title': 'Time, s'},
        'yaxis': {'title': 'Characters'},
        'title': 'Character Probabilities',
        'margin': dict(l=0, r=0, t=40, b=0, pad=0),
    }
)
fig_probs.show()

# get timestamps for space symbols
spaces = []

state = ''
idx_state = 0

if np.argmax(probs[0]) == 0:
    state = 'space'

for idx in range(1, probs.shape[0]):
    current_char_idx = np.argmax(probs[idx])
    if state == 'space' and current_char_idx != 0 and current_char_idx != 28:
        spaces.append([idx_state, idx - 1])
        state = ''
    if state == '':
        if current_char_idx == 0:
            state = 'space'
            idx_state = idx

if state == 'space':
    spaces.append([idx_state, len(probs) - 1])

# calibration offset for timestamps: 180 ms
offset = -0.18

# split the transcript into words
words = transcript.split()

# cut words
pos_prev = 0
for j, spot in enumerate(spaces):
    display(words[j])
    pos_end = offset + (spot[0] + spot[1]) / 2 * time_stride
    print(pos_prev, pos_end * sample_rate)
    display(Audio(signal[int(pos_prev * sample_rate):int(pos_end * sample_rate)],
                  rate=sample_rate))
    pos_prev = pos_end

display(words[j + 1])
display(Audio(signal[int(pos_prev * sample_rate):],
              rate=sample_rate))
