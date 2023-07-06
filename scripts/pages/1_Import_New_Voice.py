# AGPL: a notification must be added stating that changes have been made to that file.

import os
import shutil
from pathlib import Path

import streamlit as st
from random import randint
from math import floor

from tortoise.api import MODELS_DIR
from tortoise.inference import (
    infer_on_texts,
    run_and_save_tts,
    split_and_recombine_text,
)
from tortoise.utils.diffusion import SAMPLERS
from app_utils.filepicker import st_file_selector
from app_utils.conf import TortoiseConfig

from app_utils.funcs import (
    timeit,
    load_model,
    list_voices,
    load_voice_conditionings,
)

st.set_page_config(
    page_title="Tortoise TTS WebUI"
    )


LATENT_MODES = [
    "Tortoise original (bad)",
    "average per 4.27s (broken on small files)",
    "average per voice file (broken on small files)",
]

ncol = 0

def main():

    st.title('Tortoise TTS Fast WebUI')

    conf = TortoiseConfig()

    with st.expander("Create New Voice", expanded=True):
        if "file_uploader_key" not in st.session_state:
            st.session_state["file_uploader_key"] = str(randint(1000, 100000000))
            st.session_state["text_input_key"] = str(randint(1000, 100000000))

        uploaded_files = st.file_uploader(
            "Upload Audio Samples for a New Voice",
            accept_multiple_files=True,
            type=["wav"],
            key=st.session_state["file_uploader_key"]
        )

        voice_name = st.text_input(
            "New Voice Name",
            help="Enter a name for your new voice.",
            value="",
            key=st.session_state["text_input_key"]
        )

        create_voice_button = st.button(
            "Create Voice",
            disabled = ((voice_name.strip() == "") | (len(uploaded_files) == 0))
        )
        if create_voice_button:
            st.write(st.session_state)
            with st.spinner(f"Creating new voice: {voice_name}"):
                new_voice_name = voice_name.strip().replace(" ", "_")

                voices_dir = f'./tortoise/voices/{new_voice_name}/'
                if os.path.exists(voices_dir):
                    shutil.rmtree(voices_dir)
                os.makedirs(voices_dir)

                for index, uploaded_file in enumerate(uploaded_files):
                    bytes_data = uploaded_file.read()
                    with open(f"{voices_dir}voice_sample{index}.wav", "wb") as wav_file:
                        wav_file.write(bytes_data)

                st.session_state["text_input_key"] = str(randint(1000, 100000000))
                st.session_state["file_uploader_key"] = str(randint(1000, 100000000))
                st.experimental_rerun()

    voices = [v for v in os.listdir("tortoise/voices") if v != "cond_latent_example"]

    # Print voices as List in 3 columns

    st.divider()
    match (floor(len(voices) / 12) + 1):
        case 1:
            col1 = st.columns(1)
        case 2:
            col1, col2 = st.columns(2)
        case 3:
            col1, col2, col3 = st.columns(3)
        case 4:
            col1, col2, col3, col4 = st.columns(4)
        case _:
            col1, col2, col3, col4, col5 = st.columns(5)

    ncol = 0

    # Iterate for each column and print the voices

    for voice in voices :
        if ncol < 12:
            with col1:
                st.markdown("• " + str(voice))
                ncol += 1

        if (ncol >= 12) & (ncol < 24):
            with col2:
                st.markdown("• " + str(voice))
                ncol += 1

        if (ncol >= 24) & (ncol < 38):
            with col3:
                st.markdown("• " + str(voice))
                ncol += 1

        if (ncol >= 38) & (ncol < 50):
            with col4:
                st.markdown("• " + str(voice))
                ncol += 1
        if (ncol > 50):
            with col5:
                st.markdown("• " + str(voice))
                ncol += 1

if __name__ == "__main__":
    main()
