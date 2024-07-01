import rapidfuzz
from rouge_score import rouge_scorer

class Scorer():
    """Class for scoring two texts."""

    def __init__(self, original_text:str, prediction_text:str) -> None:
        self.original = original_text
        self.prediction = prediction_text 
        self.scores = self.compare_strings()
        return 

    def compare_strings(self):
        scores = {}
        scores['levenshtein_similarity_score'] = Scorer.calculate_fuzz_score(self.original, 
                                                                             self.prediction)
        scores['rouge_score'] = Scorer.calculate_rougel_score(target=self.original, 
                                                              prediction=self.prediction)
        return scores


    @staticmethod
    def calculate_fuzz_score(t1, t2, threshold=None):
        """Calculate fuzzy matching score between two strings."""
        # return rapidfuzz.fuzz.partial_ratio(
        return rapidfuzz.distance.Levenshtein.normalized_similarity(
            s1=t1,
            s2=t2,
            # processor=rapidfuzz.utils.default_process,
            score_cutoff=threshold,
        )

    @staticmethod
    def calculate_rougel_score(target, prediction):
        """Calculate RougeL score between two strings."""
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2","rouge3","rouge4","rougeL"])
        return scorer.score(target=target, prediction=prediction)
    
    @staticmethod
    def calculate_metrics(tp:int, 
                          tn:int, 
                          fp:int, 
                          fn:int, 
                          count_common:int = None, 
                          count_total:int = None):
        """Computes several metrics and returns dict containing scores. Metrics include:
        - Accuracy (if tn provided)
        - Precision
        - Recall
        - F1 score
        - False Positive Rate (if tn provided)
        - False Negative Rate 

        Args:
            tp (int): Number of true positives
            tn (int, optional): Number of true negatives. Defaults to None.
            fp (int): Number of false positives
            fn (int): Number of false negatives
            count_common (int, optional): For IOU: common items between set of prediction and truth. Defaults to None.
            count_total (int, optional): For IOU: total items in set of predictions and truth . Defaults to None.
        """
        scores = {}
        scores['precision'] = None
        scores['recall'] = None
        scores['fmeasure'] = None
        scores['accuracy'] = None
        scores['intersection_over_union'] = None
        scores['false_positive_rate'] = None
        scores['false_negative_rate'] = None
        
        try:
            precision = tp/(tp+fp)
            scores['precision'] = precision
        except Exception:
            pass
        
        try:         
            recall = tp/(tp+fn) 
            fnr = fn/(tp + fn)
            # f1 = 2/(1/recall + 1/precision)
            f1 = (2*tp)/(2*tp + fp + fn)
            accuracy = (tp+tn)/(tp+tn+fp+fn)
            scores['accuracy'] = accuracy
            scores['recall'] = recall
            scores['fmeasure'] = f1
            scores['false_negative_rate'] = fnr
        except Exception:
            pass

        try:
            fpr = fp/(fp+tn)
            scores['false_positive_rate'] = fpr
        except Exception:
            pass

        if count_common is not None and count_total is not None and count_total!=0:
            scores['intersection_over_union'] = count_common/count_total
        
        return scores

    @staticmethod
    def count_tp_tn_fp_fn(ground_truth:list, model_prediction:list):
        tp = 0
        tn = 0
        fp = 0 
        fn = 0

        for item in model_prediction:
            if item in ground_truth:
                tp +=1
            else:
                fp +=1
            
        for item in ground_truth:
            if item in model_prediction:
                pass
            elif item not in model_prediction:
                fn+=1

        gt = set(ground_truth)
        pred = set(model_prediction)
        count_common = len(gt.intersection(pred))
        count_total = len(gt.union(pred))

        return (tp, tn, fp, fn, count_common, count_total)
