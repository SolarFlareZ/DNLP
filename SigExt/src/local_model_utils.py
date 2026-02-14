import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_models = {}
_tokenizers = {}

# Model config
MODEL_CONFIGS = {
    "mistral": {
        "name": "mistralai/Mistral-7B-Instruct-v0.2",
        "trust_remote_code": False,
    },
    "llama": {
        "name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "trust_remote_code": False,
    },
    "qwen": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "trust_remote_code": True,
    },
}


def load_model(model_type):
    global _tokenizers, _models # need to fix
    
    if model_type not in _models:
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}. Choose from {list(MODEL_CONFIGS.keys())}")
        
        config = MODEL_CONFIGS[model_type]
        logging.info(f"Loading {model_type} model: {config['name']}...")
        
        # Load tokenizer
        _tokenizers[model_type] = AutoTokenizer.from_pretrained(
            config["name"],
            trust_remote_code=config["trust_remote_code"]
        )
        
        # Set pad token if not present
        if _tokenizers[model_type].pad_token is None:
            _tokenizers[model_type].pad_token = _tokenizers[model_type].eos_token
        
        # Load model
        _models[model_type] = AutoModelForCausalLM.from_pretrained(
            config["name"],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=config["trust_remote_code"]
        )
        _models[model_type].eval()
        
        logging.info(f"{model_type} model loaded successfully.")
    
    return _tokenizers[model_type], _models[model_type]


def _generate_with_model(model_type, prompt, **gen_kwargs):
    tokenizer, model = load_model(model_type)
    
    # Format using chat template
    messages = [{"role": "user", "content": prompt}]
    
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            truncation=True,
            max_length=4096
        )
    except Exception as e:
        # Fallback if apply_chat_template fails
        logging.warning(f"Chat template failed for {model_type}, using direct tokenization: {e}")
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).input_ids
    
    # Handle both cases: encoded might be a tensor or BatchEncoding
    if not isinstance(encoded, torch.Tensor):
        encoded = encoded.input_ids
    
    encoded = encoded.to(model.device)
    
    # Store the input length for later
    input_length = encoded.shape[1]
    
    # Default generation parameters
    default_kwargs = {
        "max_new_tokens": 512,
        "temperature": 1.0,
        "top_p": 0.8,
        "top_k": 10,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    # Override with any provided kwargs
    default_kwargs.update(gen_kwargs)
    
    with torch.no_grad():
        output = model.generate(
            encoded,
            **default_kwargs
        )
    
    # Decode only the generated part (skip the input)
    response = tokenizer.decode(
        output[0][input_length:],
        skip_special_tokens=True
    )
    
    return response.strip()


def predict_one_eg_mistral(x):
    try:
        prompt = x["prompt_input"]
        response = _generate_with_model("mistral", prompt)
        #logging.info(f"Mistral generated: {response[:100]}...")
        return response
    except Exception as e:
        logging.error(f"Error in Mistral inference: {e}")
        import traceback
        traceback.print_exc()
        return ""


def predict_one_eg_llama(x):
    try:
        prompt = x["prompt_input"]
        response = _generate_with_model("llama", prompt)
        # logging.info(f"Llama generated: {response[:100]}...")
        return response
    except Exception as e:
        logging.error(f"Error in Llama inference: {e}")
        import traceback
        traceback.print_exc()
        return ""


def predict_one_eg_qwen(x):
    try:
        prompt = x["prompt_input"]
        response = _generate_with_model("qwen", prompt)
        # logging.info(f"Qwen generated: {response[:100]}...")
        return response
    except Exception as e:
        logging.error(f"Error in Qwen inference: {e}")
        import traceback
        traceback.print_exc()
        return ""


def predict_one_eg_claude_instant(x):
    raise NotImplementedError("Claude not available - use Mistral, Llama, or Qwen instead")