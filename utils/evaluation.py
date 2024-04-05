import evaluate
from pycocoevalcap.cider.cider import Cider

bleu = evaluate.load("bleu")
rouge = evaluate.load('rouge')
meteor = evaluate.load('meteor')
cider_scorer = Cider()

def ignore_padding(labels, outputs, padding_values):
	mask = labels != padding_values
	new_outputs = []
	for i, each in enumerate(mask) :
		ignore_outputs = outputs[i][each].tolist()
		new_outputs.append(ignore_outputs)

	return new_outputs

def compute_metrics(pred, gt):
    bleu1_score = bleu.compute(predictions=pred, references=gt, max_order=1)['bleu']
    bleu2_score = bleu.compute(predictions=pred, references=gt, max_order=2)['bleu']
    bleu3_score = bleu.compute(predictions=pred, references=gt, max_order=3)['bleu']
    bleu4_score = bleu.compute(predictions=pred, references=gt, max_order=4)['bleu']
    rouge_score = rouge.compute(predictions=pred, references=gt)['rougeL']
    meteor_score = meteor.compute(predictions=pred, references=gt)['meteor']

    # Compute the CIDEr score
    hypotheses_dict = {i: [h] for i, h in enumerate(pred)}
    references_dict = {i: r for i, r in enumerate(gt)}
    cider_score, _ = cider_scorer.compute_score(references_dict, hypotheses_dict)

    return {
        "bleu1": bleu1_score, 
        "bleu2": bleu2_score, 
        "bleu3": bleu3_score, 
        "bleu4": bleu4_score, 
        "rouge": rouge_score, 
        "meteor": meteor_score, 
        "cider": cider_score
    }
