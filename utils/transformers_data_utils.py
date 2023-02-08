from transformers import AutoTokenizer, AutoModel
from typing import Tuple


def get_model_weight_dict(model_name: str = "bert-base-uncased") -> Tuple[dict, dict]:
    model = AutoModel.from_pretrained(model_name)

    for param in model.parameters():
        param.requires_grad = False

    parameters_weight_dict = {
        "embedding": {
            "word_embeddings": model.embeddings.word_embeddings.weight,
            "position_embeddings": model.embeddings.position_embeddings.weight
        }
    }
    parameters_bias_dict = {}

    params = model.state_dict()
    number_layers = len(model.encoder.layer)

    for layer_idx in range(number_layers):
        weight_dict = {}  # dict storing weight parameters only for this layer
        bias_dict = {}  # dict storing bias parameters only for this layer
        initial_string_match = "encoder.layer.%s." % layer_idx
        weight_parameters_names = [i.split(initial_string_match)[1] for i in params if
                                   i.startswith(initial_string_match)
                                   and "bias" not in i and "LayerNorm" not in i]
        bias_parameters_names = [i.split(initial_string_match)[1] for i in params if i.startswith(initial_string_match)
                                 and "bias" in i and "LayerNorm" not in i]
        for j in weight_parameters_names:
            if "attention.output" not in j:
                weight_dict[j.replace("attention.", "").replace("self.", "").replace("weight", "")[:-1].
                    replace(".", "_")] = params["%s%s" % (initial_string_match, j)]
            else:
                weight_dict["attention_output"] = params["%s%s" % (initial_string_match, j)]

        for j in bias_parameters_names:
            if "attention.output" not in j:
                bias_dict[j.replace("attention.", "").replace("self.", "").replace("bias", "")[:-1].
                    replace(".", "_")] = params["%s%s" % (initial_string_match, j)]
            else:
                bias_dict["attention_output"] = params["%s%s" % (initial_string_match, j)]

        parameters_weight_dict["_".join(initial_string_match.split(".")[1:-1])] = weight_dict
        parameters_bias_dict["_".join(initial_string_match.split(".")[1:-1])] = bias_dict

    return parameters_weight_dict, parameters_bias_dict
