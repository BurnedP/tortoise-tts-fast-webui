# AGPL: a notification must be added stating that changes have been made to that file.

import os
import shutil
from pathlib import Path
import writer_saver as config

import streamlit as st
from random import randint

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
    page_title="Tortoise TTS WebUI",
    layout="wide",
    )


LATENT_MODES = [
    "Tortoise original (bad)",
    "Average per 4.27s (broken on small files)",
    "Average per voice file (broken on small files)",
]

output_path = config.read("paths", "user", "output_path")
diff_models = config.read("paths","user","diff_models_path")
gpt_models = config.read("paths","user","gpt_models_path")
model_dir = config.read("paths","sys","models_path_default")
conditional_free = True
autoreg_samples = 10


def main():

    st.title('Tortoise TTS Fast WebUI')

    st.divider()


    conf = TortoiseConfig()

    text = st.text_area(
        "Text",
        help="Text to speak.",
        value="The expressiveness of autoregressive transformers is literally nuts! I absolutely adore them.",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Checkpoints")
        gptchecks = [v for v in os.listdir(gpt_models) if (v[-4:] == ".pth")]
        ar_checkpoint = st.selectbox(
            "GPT Checkpoint",
            gptchecks,
            help="Selects the GPT Checkpoint (finetune) to use in generation. Loads from \"/models/GPT Checkpoints\" by default. ",
            index=0,
            key="pth",
        ) 

        diffchecks = [d for d in os.listdir(diff_models) if (d[-4:] == ".pth")]
        diff_checkpoint = st.selectbox(
            "Diffusion Checkpoint",
            diffchecks,
            help="Selects the Diffusion Checkpoint to use in generation. Loads from \"/models/Diffusion Checkpoints\" by default. ",
            index=0,
            key="diff"
        )
    with col2:
        st.subheader("Voices")
        voices = [v for v in os.listdir("tortoise/voices") if v != "cond_latent_example"]
        voice = st.selectbox(
            "Voice",
            voices,
            help="Selects the voice to use for generation. See options in voices/ directory (and add your own!) "
            "Use the & character to join two voices together. Use a comma to perform inference on multiple voices.",
            index=0,
        )
        preset = st.selectbox(
            "Preset",
            (
                "single_sample",
                "ultra_fast",
                "very_fast",
                "ultra_fast_old",
                "fast",
                "standard",
                "high_quality",
                "custom",
            ),
            help="Which voice preset to use.",
            index=1,
        )
    with col3:
        st.subheader("Generation Settings")



        seed = st.number_input(
            "Seed",
            help="Random seed which can be used to reproduce results.",
            value=int(config.read("user_settings", "tortoise", "seed")) if (config.read("user_settings", "tortoise", "seed") != None) else -1,
        )
        if seed == -1:
            seed = None

    if preset == "custom":
        st.subheader("Custom Settings")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            conditional_free = st.checkbox("Conditonal Free", value=config.read("user_settings", "tortoise", "conditional_free"))
            col11, col12 = st.columns(2)
            with col11:
                autoreg_samples = int(st.number_input("Samples", 0, value=config.read("user_settings", "tortoise", "autoreg_samples")))
                candidates = st.number_input(
                    "Candidates",
                    help="How many output candidates to produce per-voice.",
                    value=config.read("user_settings", "tortoise", "candidates")
                )
            with col12:
                iterations = int(st.number_input("Iterations", 0, value=config.read("user_settings", "tortoise", "iterations")))
                min_chars_to_split = st.number_input(
                    "Min Chars to Split",
                    help="Minimum number of characters to split text on",
                    min_value=50,
                    value=config.read("user_settings", "tortoise", "min_chars_to_split"),
                    step=1,
                )
            temperature = st.slider("Temperature", 0.0, 1.0, step=0.1, format="%.1f",
                value=config.read("user_settings", "tortoise", "temperature"))
            diff_temp = st.slider("Diffusion Temperature", 0.0, 1.0, step=0.1, format="%.1f",
                value=config.read("user_settings", "tortoise", "diff_temp"))
        with col2:
            cvvp = st.slider("CVVP Amount", 0.0, 1.0,
                             value=config.read("user_settings", "tortoise", "cvvp"))    
            len_penalty = st.slider("Length Penalty", 0.0, 1.0, step=0.1, format="%.1f",
                                    value=config.read("user_settings", "tortoise", "len_penalty"))
            rep_penalty = st.slider("Repetition Penalty", 0.0, 5.0, step=0.1, format="%.1f",
                                    value=config.read("user_settings", "tortoise", "rep_penalty"))
            top_p = st.slider("Top P", 0.1, 1.0, step=0.1, format="%.1f",
                              value=config.read("user_settings", "tortoise", "top_p"))          
            
        with col3:
            sampler = st.radio(
                "Sampler",
                #SAMPLERS,
                SAMPLERS,
                help="Diffusion sampler. Note that dpm++2m is experimental and typically requires more steps.",
                index=1,
            )
            latent_averaging_mode = st.radio(
                "Latent averaging mode",
                LATENT_MODES,
                help="How voice samples should be averaged together.",
                index=0,
            )
    else:
        latent_averaging_mode = LATENT_MODES[0]
        sampler = "p"
        candidates = 1
        min_chars_to_split = 200


            
    with st.sidebar:
        st.header("Optimizations")
        high_vram = not st.checkbox(
            "Low VRAM",
            help="Re-enable default offloading behaviour of tortoise",
            value=not config.read("user_settings", "tortoise", "high_vram"),
        )
        half = st.checkbox(
            "Half-Precision",
            help="Enable autocast to half precision for autoregressive model",
            value=config.read("user_settings", "tortoise", "half"),
        )
        kv_cache = st.checkbox(
            "Key-Value Cache",
            help="Enable kv_cache usage, leading to drastic speedups but worse memory usage",
            value=config.read("user_settings", "tortoise", "kv_cache"),
        )
        voice_fixer = st.checkbox(
            "Voice fixer",
            help="Use `voicefixer` to improve audio quality. This is a post-processing step which can be applied to any output.",
            value=config.read("user_settings", "tortoise", "voice_fixer"),
        )

        st.divider()
        
        st.header("Debug")
        produce_debug_state = st.checkbox(
            "Produce Debug State",
            help="Whether or not to produce debug_state.pth, which can aid in reproducing problems. Defaults to true.",
            value=config.read("user_settings", "tortoise", "produce_debug_state"),
        )
    
    

    if diff_checkpoint == "" or diff_checkpoint == "default.pth":
        diff_checkpoint = ""
    if ar_checkpoint == ""or ar_checkpoint == "default.pth":
        ar_checkpoint = ""

    ar_checkpoint = (None if ar_checkpoint[-4:] != ".pth" else (gpt_models + "/" + ar_checkpoint)) # type: ignore
    diff_checkpoint = (None if diff_checkpoint[-4:] != ".pth" else (gpt_models + "/" + diff_checkpoint)) # type: ignore
    tts = load_model(model_dir, high_vram, kv_cache, ar_checkpoint, diff_checkpoint)
    
    # No clue why tf this was here. Just takes the custom checkpoints back to none BEFORE loading
    #ar_checkpoint = None
    #diff_checkpoint = None
    #tts = load_model(MODELS_DIR, high_vram, kv_cache, ar_checkpoint, diff_checkpoint)
    #ar_checkpoint


    if st.button("Start", type="primary"):
        assert latent_averaging_mode
        assert preset
        assert voice


        def show_generation(fp, filename: str):
            """
            audio_buffer = BytesIO()
            save_gen_with_voicefix(g, audio_buffer, squeeze=False)
            torchaudio.save(audio_buffer, g, 24000, format='wav')
            """
            st.audio(str(fp), format="audio/wav")
            st.download_button(
                "Download sample",
                str(fp),
                file_name=filename,  # this doesn't actually seem to work lol
            )
        
        #Adds waiting animation whilst code inside is running
        with st.spinner(    
            "Generating {candidates} candidates for voice {voice} (seed={seed}). You can see progress in the terminal"
        ):
            os.makedirs(output_path, exist_ok=True)

            # Splits voices into a list, but I don't understand half of it             
            selected_voices = voice.split(",")
            for k, selected_voice in enumerate(selected_voices):
                if "&" in selected_voice:
                    voice_sel = selected_voice.split("&")
                else:
                    voice_sel = [selected_voice]
                voice_samples, conditioning_latents = load_voice_conditionings(
                    voice_sel, []
                )

                voice_path = Path(os.path.join(output_path, selected_voice))

                # Times how long to run
                with timeit(
                    f"Generating {candidates} candidates for voice {selected_voice} (seed={seed})"
                ):
                    nullable_kwargs = {
                        k: v
                        for k, v in zip(
                            ["sampler", "diffusion_iterations", "cond_free"],
                            [sampler, autoreg_samples, conditional_free],
                        )
                        if v is not None
                    }

                    # Defines the call_tts function, then sets settings for the generation itself with preset
                    # k is number of candidates
                    def call_tts(text: str):
                        if preset != "custom":

                            # Saves current settings before running

                            for key in config.read("user_settings", "tortoise"):
                                if key in locals():
                                    config.write("user_settings", "tortoise", key, locals()[key])
                                if key in globals():
                                    config.write("user_settings", "tortoise", key, globals()[key])

                            return tts.tts_with_preset(
                                text,
                                k=candidates,
                                voice_samples=voice_samples,
                                conditioning_latents=conditioning_latents,
                                preset=str(preset), 
                                use_deterministic_seed=seed,
                                return_deterministic_state=True,
                                cvvp_amount=0.0,
                                half=half,
                                latent_averaging_mode=LATENT_MODES.index(
                                    latent_averaging_mode
                                ),
                                **nullable_kwargs,
                            )
                        else:
                            # Saves current settings before running
                            for key in config.read("user_settings", "tortoise"):
                                if key in locals():
                                    config.write("user_settings", "tortoise",key, locals()[key])
                                if key in globals():
                                    config.write("user_settings", "tortoise",key, globals()[key])


                            return tts.tts(
                                text,
                                k=int(candidates),
                                voice_samples=voice_samples,
                                conditioning_latents=conditioning_latents,
                                use_deterministic_seed=seed,
                                return_deterministic_state=True,
                                cvvp_amount=cvvp,
                                half=half,
                                #Deterministic Parameters
                                num_autoregressive_samples=autoreg_samples,
                                temperature=temperature,
                                length_penalty=len_penalty,
                                repetition_penalty=rep_penalty,
                                top_p=top_p,
                                #Diffusion Parameters
                                diffusion_iterations=iterations,
                                cond_free=conditional_free,
                                diffusion_temperature=diff_temp,
                                latent_averaging_mode=LATENT_MODES.index(
                                    latent_averaging_mode
                                ),
                            )

                    # Takes care of text split and runs tts
                    if len(text) < min_chars_to_split:
                        filepaths = run_and_save_tts(
                            call_tts,
                            text,
                            voice_path,
                            return_deterministic_state=True,
                            return_filepaths=True,
                            voicefixer=voice_fixer,
                        )
                        for i, fp in enumerate(filepaths):
                            show_generation(fp, f"{selected_voice}-text-{i}.wav")
                    else:
                        desired_length = int(min_chars_to_split)
                        texts = split_and_recombine_text(
                            text, desired_length, desired_length + 100
                        )
                        filepaths = infer_on_texts(
                            call_tts,
                            texts,
                            voice_path,
                            return_deterministic_state=True,
                            return_filepaths=True,
                            lines_to_regen=set(range(len(texts))),
                            voicefixer=voice_fixer,
                        )
                        for i, fp in enumerate(filepaths):
                            show_generation(fp, f"{selected_voice}-text-{i}.wav")
        if produce_debug_state:
            """Debug states can be found in the output directory"""


if __name__ == "__main__":
    main()
