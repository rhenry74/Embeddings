import onnxruntime as ort
from transformers import AutoTokenizer
from pathlib import Path

import numpy
import builtins


quantize_output_model_path = "model_quantized.onnx"

# providers = ['CPUExecutionProvider']
# provider_options = [{}]

providers = ['VitisAIExecutionProvider']  
cache_dir = Path(__file__).parent.resolve()
provider_options = [{
                'config_file': 'vaip_config.json',
                'cacheDir': str(cache_dir),
                'cacheKey': 'ipucachekey'
            }]

session_options = ort.SessionOptions()
#session_options.log_verbosity_level = 1
#session_options.log_severity_level = 0

builtins.impl = "v0"
builtins.quant_mode = "w8a8"
session = ort.InferenceSession(quantize_output_model_path, 
                               providers=providers,
                               sess_options=session_options,                               
                               provider_options=provider_options)

input_shape = session.get_inputs()

tokenizer = AutoTokenizer.from_pretrained('UAE-Large-V1')

tokens = tokenizer.encode_plus(text='what are we going to do today', 
                                  return_attention_mask=True,
                                  return_token_type_ids=True
                                  )

print(input_shape[0])
print(input_shape[1])
print(input_shape[2])

input_ids = numpy.array([tokens['input_ids']], dtype=numpy.int64)
attention_mask = numpy.array([tokens['attention_mask']], dtype=numpy.int64)
token_type_ids = numpy.array([tokens['token_type_ids']], dtype=numpy.int64)

shape = {'input_ids': input_ids
        ,'attention_mask': attention_mask
        ,'token_type_ids': token_type_ids}

try:
        print('about to run')
        outputs = session.run(None, shape)
        print('run success')
        print(outputs)
except Exception as e:
        print("There was an error: ", e)                      

