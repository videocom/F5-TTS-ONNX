import re
import time
import numpy as np
import onnxruntime
import soundfile as sf
from pydub import AudioSegment
import onnxruntime.tools.add_openvino_win_libs as utils
import argparse

utils.add_openvino_libs_to_path()


# Exported models  https://drive.google.com/drive/folders/1NxvDDDU0VmcySbbknfaUG5Aj5NH7qUBX
# For model A, C we should Linux_x64_CPU_F32/F5_Preprocess.ort , F5_Decode.ort    
# For model B we use Linux_GPU/FP16/F5_Transformer.onnx
# We take vocab.txt from  https://github.com/SWivid/F5-TTSdata/Emilia_ZH_EN_pinyin/vocab.txt 
# These models and vocab.txt is expected to be located in model_dir

parser = argparse.ArgumentParser(description="F5-TTS ONNX Inference")
parser.add_argument('--args_file', type=str, help='Path to args.toml file', default=None)
parser.add_argument('--model_dir', type=str, default="c:/Test/F5/models/mixed")
parser.add_argument('--ref_audio', type=str, default="c:/Test/F5/basic_ref_en.wav")
parser.add_argument('--ref_text', type=str, default="Some call me nature, others call me mother nature")
parser.add_argument('--gen_text', type=str, default="Let's try to generate some audio, its going to be interesting")
parser.add_argument('--gen_audio', type=str, default="c:/Test/F5/generated.wav")
parser.add_argument('--cache_dir', type=str, default="c:/temp/ov")

args = parser.parse_args()

model_dir = args.model_dir
ref_audio = args.ref_audio
ref_text = args.ref_text
gen_text = args.gen_text
gen_audio = args.gen_audio
cache_dir = args.cache_dir

onnx_model_A = f"{model_dir}/F5_Preprocess.ort"
onnx_model_B = f"{model_dir}/F5_Transformer.onnx"
onnx_model_C = f"{model_dir}/F5_Decode.ort"


HOP_LENGTH = 256                        # Number of samples between successive frames in the STFT
SAMPLE_RATE = 24000                     # The generated audio sample rate
RANDOM_SEED = 9527                      # Set seed to reproduce the generated audio
NFE_STEP = 32                           # F5-TTS model setting
SPEED = 1.0                             # Set for talking speed. Only works with dynamic_axes=True
MAX_THREADS = 8                         # Max CPU parallel threads.
DEVICE_ID = 0                           # The GPU id, default to 0.

OpenVINO_provider_options = [
        {
            'device_type': 'GPU',
            #'precision': 'ACCURACY',
            'precision': 'FP16',
            'num_of_threads': MAX_THREADS,
            'num_streams': 1,
            'enable_opencl_throttling': True,
            'enable_qdq_optimizer': False,
            'cache_dir': cache_dir
        }

    ]


with open(f"{model_dir}/vocab.txt", "r", encoding="utf-8") as f:
    vocab_char_map = {}
    for i, char in enumerate(f):
        vocab_char_map[char[:-1]] = i
vocab_size = len(vocab_char_map)



#Change def list_str_to_idx to get rid of Torch depedency

def list_str_to_idx(text, vocab_char_map, padding_value=0):
    get_idx = vocab_char_map.get
    list_idx_tensors = [np.array([get_idx(c, 0) for c in t], dtype=np.int32) for t in text]

    max_len = max(len(seq) for seq in list_idx_tensors)
    padded_text = np.full((len(list_idx_tensors), max_len), padding_value, dtype=np.int32)

    for i, seq in enumerate(list_idx_tensors):
        padded_text[i, :len(seq)] = seq

    return padded_text
    


# ONNX Runtime settings
onnxruntime.set_seed(RANDOM_SEED)
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 3                 # error level, it an adjustable value.
session_opts.inter_op_num_threads = MAX_THREADS     # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = MAX_THREADS     # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True            # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")

session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
model_type = ort_session_A._inputs_meta[0].type
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
in_name_A1 = in_name_A[1].name
in_name_A2 = in_name_A[2].name
out_name_A0 = out_name_A[0].name
out_name_A1 = out_name_A[1].name
out_name_A2 = out_name_A[2].name
out_name_A3 = out_name_A[3].name
out_name_A4 = out_name_A[4].name
out_name_A5 = out_name_A[5].name
out_name_A6 = out_name_A[6].name

#session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC

session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
#https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#onnxruntime-graph-level-optimization
#session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL



ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=['DmlExecutionProvider'],provider_options = [{'device_id': 0}]  )
#use this instead for OpenVino
#ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=OpenVINOExecutionProvider, provider_options=OpenVINO_provider_options)

print(f"\nUsable Providers Session B: {ort_session_B.get_providers()}")
model_B_dtype = ort_session_B._inputs_meta[0].type
in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
in_name_B0 = in_name_B[0].name
in_name_B1 = in_name_B[1].name
in_name_B2 = in_name_B[2].name
in_name_B3 = in_name_B[3].name
in_name_B4 = in_name_B[4].name
in_name_B5 = in_name_B[5].name
in_name_B6 = in_name_B[6].name
out_name_B0 = out_name_B[0].name

session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
in_name_C = ort_session_C.get_inputs()
out_name_C = ort_session_C.get_outputs()
in_name_C0 = in_name_C[0].name
in_name_C1 = in_name_C[1].name
out_name_C0 = out_name_C[0].name


# Load the input audio
print(f"\nReference Audio: {ref_audio}")
audio = np.array(AudioSegment.from_file(ref_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples())
audio_len = len(audio)
audio = audio.reshape(1, 1, -1)

ref_text_len = len(ref_text.encode('utf-8'))
gen_text_len = len(gen_text.encode('utf-8'))
ref_audio_len = audio_len // HOP_LENGTH + 1
max_duration = np.array(ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / SPEED), dtype=np.int64)

# wrap to a list since we removed the helper function convert_char_to_pinyin([ref_text + gen_text]) which did that
gen_text = [ref_text + gen_text]
text_ids = list_str_to_idx(gen_text, vocab_char_map)

print("\n\nRun F5-TTS by ONNX Runtime.")
start_count = time.time()
noise, rope_cos, rope_sin, cat_mel_text, cat_mel_text_drop, qk_rotated_empty, ref_signal_len = ort_session_A.run(
        [out_name_A0, out_name_A1, out_name_A2, out_name_A3, out_name_A4, out_name_A5, out_name_A6],
        {
            in_name_A0: audio,
            in_name_A1: text_ids,
            in_name_A2: max_duration
        })

if 'float16' in model_B_dtype:
    noise = noise.astype(np.float16)
    rope_cos = rope_cos.astype(np.float16)
    rope_sin = rope_sin.astype(np.float16)
    cat_mel_text = cat_mel_text.astype(np.float16)
    cat_mel_text_drop = cat_mel_text_drop.astype(np.float16)
    qk_rotated_empty = qk_rotated_empty.astype(np.float16)



for i in range(NFE_STEP):
    print(f"NFE_STEP: {i} of {NFE_STEP}")
    noise = ort_session_B.run(
        [out_name_B0],
        {
            in_name_B0: noise,
            in_name_B1: rope_cos,
            in_name_B2: rope_sin,
            in_name_B3: cat_mel_text,
            in_name_B4: cat_mel_text_drop,
            in_name_B5: qk_rotated_empty,
            in_name_B6: np.array(i, dtype=np.int32)
        })[0]
            
if 'float16' in model_B_dtype:    # when using fp32 for model C
    noise = noise.astype(np.float32)
        
generated_signal = ort_session_C.run(
        [out_name_C0],
        {
            in_name_C0: noise,
            in_name_C1: ref_signal_len
        })[0]
end_count = time.time()

# Save to audio
sf.write(gen_audio, generated_signal.reshape(-1), SAMPLE_RATE, format='WAVEX')
print(f"\nAudio generation is complete.\n\nONNXRuntime Time Cost in Seconds:\n{end_count - start_count:.3f}")
