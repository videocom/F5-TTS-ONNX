import re
import time
import jieba
import numpy as np
import torch
import onnxruntime
import soundfile as sf
from pydub import AudioSegment
from pypinyin import lazy_pinyin, Style
import onnxruntime.tools.add_openvino_win_libs as utils
utils.add_openvino_libs_to_path()


# Exported models  https://drive.google.com/drive/folders/1NxvDDDU0VmcySbbknfaUG5Aj5NH7qUBX

F5_project_path      = "c:/git/F5-TTS"   # The F5-TTS Github project download path.  URL: https://github.com/SWivid/F5-TTS

onnx_model_A         = "c:/Test/F5/models/Linux_x64_CPU_F32/F5_Preprocess.ort"                     
onnx_model_B         = "C:/Test/F5/models/Linux_GPU/FP16/F5_Transformer.onnx"                      
onnx_model_C         = "c:/Test/F5/models/Linux_x64_CPU_F32/F5_Decode.ort"                         
cache_dir            = "c:/temp/ov"
 

reference_audio      = "c:/Test/F5/basic_ref_en.wav"     # The reference audio path.
ref_text             = "Some call me nature, others call me mother nature" 
gen_text             = "Let's try to generate some audio, its going to be interesting"       # The target TTS.
generated_audio      = "c:/Test/F5/generated.wav"        # The generated audio path.


ORT_Accelerate_Providers = ['OpenVINOExecutionProvider'] # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                                         # else keep empty.
HOP_LENGTH = 256                        # Number of samples between successive frames in the STFT
SAMPLE_RATE = 24000                     # The generated audio sample rate
RANDOM_SEED = 9527                      # Set seed to reproduce the generated audio
NFE_STEP = 32                           # F5-TTS model setting
SPEED = 1.0                             # Set for talking speed. Only works with dynamic_axes=True
MAX_THREADS = 8                         # Max CPU parallel threads.
DEVICE_ID = 0                           # The GPU id, default to 0.

with open(f"{F5_project_path}/data/Emilia_ZH_EN_pinyin/vocab.txt", "r", encoding="utf-8") as f:
    vocab_char_map = {}
    for i, char in enumerate(f):
        vocab_char_map[char[:-1]] = i
vocab_size = len(vocab_char_map)


if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
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
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'gpu_mem_limit': 8 * 1024 * 1024 * 1024,  # 8 GB
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'cudnn_conv_use_max_workspace': '1',
            'do_copy_in_default_stream': '1',
            'cudnn_conv1d_pad_to_nc1d': '1',
            'enable_cuda_graph': '0',  # Set to '0' to avoid potential errors when enabled.
            'use_tf32': '0'          
        }
    ]
else:
    provider_options = None


def is_chinese_char(c):
    cp = ord(c)
    return (
        0x4E00 <= cp <= 0x9FFF or    # CJK Unified Ideographs
        0x3400 <= cp <= 0x4DBF or    # CJK Unified Ideographs Extension A
        0x20000 <= cp <= 0x2A6DF or  # CJK Unified Ideographs Extension B
        0x2A700 <= cp <= 0x2B73F or  # CJK Unified Ideographs Extension C
        0x2B740 <= cp <= 0x2B81F or  # CJK Unified Ideographs Extension D
        0x2B820 <= cp <= 0x2CEAF or  # CJK Unified Ideographs Extension E
        0xF900 <= cp <= 0xFAFF or    # CJK Compatibility Ideographs
        0x2F800 <= cp <= 0x2FA1F     # CJK Compatibility Ideographs Supplement
    )


def convert_char_to_pinyin(text_list, polyphone=True):
    final_text_list = []
    merged_trans = str.maketrans({
        '“': '"', '”': '"', '‘': "'", '’': "'",
        ';': ','
    })
    chinese_punctuations = set("。，、；：？！《》【】—…")
    for text in text_list:
        char_list = []
        text = text.translate(merged_trans)
        for seg in jieba.cut(text):
            if seg.isascii():
                if char_list and len(seg) > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and all(is_chinese_char(c) for c in seg):
                pinyin_list = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for c in pinyin_list:
                    if c not in chinese_punctuations:
                        char_list.append(" ")
                    char_list.append(c)
            else:
                for c in seg:
                    if c.isascii():
                        char_list.append(c)
                    elif c in chinese_punctuations:
                        char_list.append(c)
                    else:
                        char_list.append(" ")
                        pinyin = lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True)
                        char_list.extend(pinyin)
        final_text_list.append(char_list)
    return final_text_list


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
#ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)

print(f"\nUsable Providers Session B: {ort_session_B.get_providers()}")
model_dtype = ort_session_B._inputs_meta[0].type
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
print(f"\nReference Audio: {reference_audio}")
audio = np.array(AudioSegment.from_file(reference_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples())
audio_len = len(audio)
audio = audio.reshape(1, 1, -1)

zh_pause_punc = r"。，、；：？！"
ref_text_len = len(ref_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, ref_text))
gen_text_len = len(gen_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, gen_text))
ref_audio_len = audio_len // HOP_LENGTH + 1
max_duration = np.array(ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / SPEED), dtype=np.int64)
gen_text = convert_char_to_pinyin([ref_text + gen_text])
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

if 'float16' in model_dtype:
    noise = noise.astype(np.float16)
    rope_cos = rope_cos.astype(np.float16)
    rope_sin = rope_sin.astype(np.float16)
    cat_mel_text = cat_mel_text.astype(np.float16)
    cat_mel_text_drop = cat_mel_text_drop.astype(np.float16)
    qk_rotated_empty = qk_rotated_empty.astype(np.float16)


if "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    noise = onnxruntime.OrtValue.ortvalue_from_numpy(noise, 'cuda', DEVICE_ID)
    rope_cos = onnxruntime.OrtValue.ortvalue_from_numpy(rope_cos, 'cuda', DEVICE_ID)
    rope_sin = onnxruntime.OrtValue.ortvalue_from_numpy(rope_sin, 'cuda', DEVICE_ID)
    cat_mel_text = onnxruntime.OrtValue.ortvalue_from_numpy(cat_mel_text, 'cuda', DEVICE_ID)
    cat_mel_text_drop = onnxruntime.OrtValue.ortvalue_from_numpy(cat_mel_text_drop, 'cuda', DEVICE_ID)
    qk_rotated_empty = onnxruntime.OrtValue.ortvalue_from_numpy(qk_rotated_empty, 'cuda', DEVICE_ID)

    inputs = {
        in_name_B1: (rope_cos.element_type(), rope_cos.data_ptr(), rope_cos.shape()),
        in_name_B2: (rope_sin.element_type(), rope_sin.data_ptr(), rope_sin.shape()),
        in_name_B3: (cat_mel_text.element_type(), cat_mel_text.data_ptr(), cat_mel_text.shape()),
        in_name_B4: (cat_mel_text_drop.element_type(), cat_mel_text_drop.data_ptr(), cat_mel_text_drop.shape()),
        in_name_B5: (qk_rotated_empty.element_type(), qk_rotated_empty.data_ptr(), qk_rotated_empty.shape()),
    }

    io_binding = ort_session_B.io_binding()
    for name, (dtype, buffer_ptr, shape) in inputs.items():
        io_binding.bind_input(
            name=name,
            device_type='cuda',
            device_id=DEVICE_ID,
            element_type=dtype,
            shape=shape,
            buffer_ptr=buffer_ptr
        )
    io_binding.bind_output(out_name_B0, 'cuda')

    for i in range(NFE_STEP):
        print(f"NFE_STEP: {i}")
        time_step = onnxruntime.OrtValue.ortvalue_from_numpy(np.array(i, dtype=np.int32), 'cuda', DEVICE_ID)
        io_binding.bind_input(
            name=in_name_B6,
            device_type='cuda',
            device_id=DEVICE_ID,
            element_type=time_step.element_type(),
            shape=time_step.shape(),
            buffer_ptr=time_step.data_ptr()
        )
        io_binding.bind_input(
            name=in_name_B0,
            device_type='cuda',
            device_id=DEVICE_ID,
            element_type=noise.element_type(),
            shape=noise.shape(),
            buffer_ptr=noise.data_ptr()
        )
        ort_session_B.run_with_iobinding(io_binding)
        noise = io_binding.get_outputs()[0]
    noise = onnxruntime.OrtValue.numpy(noise)
else:
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
            
if 'float16' in model_dtype:    # when using fp32 for model C
    noise = noise.astype(np.float32)
        
generated_signal = ort_session_C.run(
        [out_name_C0],
        {
            in_name_C0: noise,
            in_name_C1: ref_signal_len
        })[0]
end_count = time.time()

# Save to audio
sf.write(generated_audio, generated_signal.reshape(-1), SAMPLE_RATE, format='WAVEX')
print(f"\nAudio generation is complete.\n\nONNXRuntime Time Cost in Seconds:\n{end_count - start_count:.3f}")
