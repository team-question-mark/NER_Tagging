from fastapi import FastAPI
from pydantic import BaseModel
import torch

# FastAPI 애플리케이션 생성
app = FastAPI()

# 요청 바디 모델 정의
class TextRequest(BaseModel):
    sentence: str

# from ratsnlp.nlpbook.ner import NERTrainArguments

from ratsnlp.nlpbook.ner import NERDeployArguments
args = NERDeployArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_model_dir="./",
    max_seq_length=64,
)

from transformers import BertConfig, BertForTokenClassification
fine_tuned_model_ckpt = torch.load(
    args.downstream_model_checkpoint_fpath,
    map_location=torch.device("cpu")
)
pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
    num_labels=fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel(),
)
model = BertForTokenClassification(pretrained_model_config)
model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
model.eval()

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)

labels = [label.strip() for label in open(args.downstream_model_labelmap_fpath, "r").readlines()]
id_to_label = {}
for idx, label in enumerate(labels):
    if "PER" in label:
        label = "인명"
    elif "LOC" in label:
        label = "지명"
    elif "ORG" in label:
        label = "기관명"
    id_to_label[idx] = label


def inference_fn(sentence):
    inputs = tokenizer(
        [sentence],
        max_length=args.max_seq_length,
        padding="max_length",
        truncation=True,
    )
    with torch.no_grad():
        outputs = model(**{k: torch.tensor(v) for k, v in inputs.items()})
        probs = outputs.logits[0].softmax(dim=1)
        top_probs, preds = torch.topk(probs, dim=1, k=1)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_tags = [id_to_label[pred.item()] for pred in preds]
        result = []
        for token, predicted_tag, top_prob in zip(tokens, predicted_tags, top_probs):
            if token not in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]:
                token_result = {
                    "token": token,
                    "predicted_tag": predicted_tag,
                    "top_prob": str(round(top_prob[0].item(), 4)),
                }
                result.append(token_result)
    return {
        "sentence": sentence,
        "result": result,
    }


def perform_ner(text):
    sentences = text.split('\n')
    results = []

    for sentence in sentences:
        result = inference_fn(sentence)
        results.append(result)

    return results

def reconstruct_sentences(results):
    reconstructed_sentences = []

    for result in results:
        sentence = result['sentence']
        tokens = []
        predicted_tags = []

        for token_result in result['result']:
            token = token_result['token']
            predicted_tag = token_result['predicted_tag']
            tokens.append(token)
            predicted_tags.append(predicted_tag)

        reconstructed_tokens = []
        current_tag = None

        for token, tag in zip(tokens, predicted_tags):
            if tag in ['인명', '지명', '기관명']:
                current_tag = tag
                reconstructed_tokens.append(f"{tag}")
            else:
                if token.startswith("##"):
                    reconstructed_tokens.append(token[2:])
                else:
                    reconstructed_tokens.append(token)

        reconstructed_sentence = " ".join(reconstructed_tokens).strip()
        reconstructed_sentences.append(reconstructed_sentence)

    return reconstructed_sentences

def filter_result(result):
    sentence = result["sentence"]
    tokens = result["result"]

    named_entities = {
        "인명": [], "지명": [],"기관명": []
    }

    for token in tokens:
        token_text = token["token"]
        predicted_tag = token["predicted_tag"]

        if predicted_tag in ["인명", "지명", "기관명"]:
            named_entities[predicted_tag].append(token_text)

    filtered_result = {
        "NER_entities": named_entities
    }

    return filtered_result



@app.post("/ner")
def ner(text_request: TextRequest):
    text = text_request.sentence
    ner_results = perform_ner(text)
    reconstructed_sentences = reconstruct_sentences(ner_results)

    # 필터링된 결과 생성
    filtered_ner_results = [filter_result(result) for result in ner_results]

    # 필터링된 결과에서 인명, 지명, 기관명 추출
    named_entities = {
        "인명": [],
        "지명": [],
        "기관명": []
    }

    for result in filtered_ner_results:
        for entity_type, entity_list in result['NER_entities'].items():
            named_entities[entity_type].extend(entity_list)

    # 응답 준비
    response = {
        "request_text": text,
        "NER_result": reconstructed_sentences[0],
        "NER_tag": named_entities
        
    }
    return response

# FastAPI 서버 시작
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)