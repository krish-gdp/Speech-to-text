import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import soundfile as sf
import tempfile
import threading
import time

class AudioRecorder:
    def __init__(self):
        self.frames = []
        self._recording = False
        self.saved_audio_file = None  # Initialize saved_audio_file attribute

    def start_recording(self):
        self._recording = True
        self.frames = []
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.start()

    def stop_recording(self):
        self._recording = False

    def _record_audio(self):
        audio_generator = webrtc_streamer(
            key="audio-recorder",
            mode=WebRtcMode.SENDONLY,
            audio=True,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            async_processing=True,
        )

        for audio_chunk in audio_generator:
            if not self._recording:
                break
            if audio_chunk:
                self.frames.append(audio_chunk.to_ndarray(dtype="int16"))
            time.sleep(0.1)

        audio_array = self._flatten_audio_frames()
        self._save_audio(audio_array)

    def _flatten_audio_frames(self):
        flat_audio = [audio for frame in self.frames for audio in frame]
        return flat_audio

    def _save_audio(self, audio_array):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_array, 16000)
            self.saved_audio_file = f.name

def main():
    st.title("Audio Recorder")

    recorder = AudioRecorder()

    if st.button("Start Recording"):
        recorder.start_recording()
    
    if st.button("Stop Recording"):
        recorder.stop_recording()
        if recorder.saved_audio_file:
            st.audio(recorder.saved_audio_file, format="audio/wav")

if __name__ == "__main__":
    main()