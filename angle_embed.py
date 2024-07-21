import onnxruntime as ort
from transformers import AutoTokenizer
from pathlib import Path

import numpy
import builtins

from flask import Flask, request
import orjson


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

tokenizer = AutoTokenizer.from_pretrained('UAE-Large-V1')

def getVectorFroModel(phrase):
     tokens = tokenizer.encode_plus(text=phrase, 
                                  return_attention_mask=True,
                                  return_token_type_ids=True
                                  )

     input_ids = numpy.array([tokens['input_ids']], dtype=numpy.int64)
     attention_mask = numpy.array([tokens['attention_mask']], dtype=numpy.int64)
     token_type_ids = numpy.array([tokens['token_type_ids']], dtype=numpy.int64)

     shape = {'input_ids': input_ids
        ,'attention_mask': attention_mask
        ,'token_type_ids': token_type_ids}
   
     try:
                #print('about to run: ' + phrase)
                outputs = session.run(None, shape)
                #print('run success')
                v1 = outputs[0][0][0]
                #print(v1)
                v1_norm = v1 / numpy.linalg.norm(v1)
                return v1_norm
     except Exception as e:
                print("There was an error: ", e)     

app = Flask(__name__)

@app.route("/")
def home():
    return "angle is loaded V0.4", 200

@app.route("/embedding/<phrase>")
def getEmbedding(phrase):
    
   v1_norm = getVectorFroModel(phrase)
    
   data = {'Vector': v1_norm}
   s = orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY,).decode('UTF-8')
   return s, 200

@app.route("/compare")
def compareEmbedding():
    phrase1 = request.args.get("phrase1")
    phrase2 = request.args.get("phrase2")
    vec1 = getVectorFroModel(phrase1)
    vec2 = getVectorFroModel(phrase2)
    
    data = {'dot': vec1.dot(vec2), 'v1': vec1,'v2': vec2}
    s = orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY,).decode('UTF-8')
    return s, 200

if __name__ == "__main__":
    app.run(host='::', port=7770)









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

