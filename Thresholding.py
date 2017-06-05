def M_Cut(all_scores):
    predictions = []
    for scores in all_scores:
        #implement M-Cut thresholding
        sorted_scores = list(scores)
        sorted_scores.sort()
        max_diff = scr1 = scr2 = 0
        for i in range(len(sorted_scores)-1):
            if max_diff < abs(sorted_scores[i]-sorted_scores[i+1]):
                scr1 = sorted_scores[i]
                scr2 = sorted_scores[i+1]
                max_diff = abs(sorted_scores[i]-sorted_scores[i+1])
        thresh = (scr1+scr2)/2
        #print "Threshold: ", thresh, scr1, scr2
        pred = [0]*len(scores)
        classes = [i for i, x in enumerate(scores) if x > thresh]
        for i in classes:
            pred[i] = 1
        predictions.append(pred)
    return predictions


def M_Cut_mod(all_scores):
    predictions = []
    for scores in all_scores:
        #implement M-Cut thresholding
        sorted_scores = list(scores)
        sorted_scores.sort()
        diff = []
        for i in range(len(sorted_scores)-1):
            diff.append(abs(sorted_scores[i]-sorted_scores[i+1]))
        sorted_diff = list(diff)
        temp = diff1 = diff2 = 0
        for i in range(len(sorted_diff)-1):
            if temp < abs(sorted_diff[i]-sorted_diff[i+1]):
                diff1 = sorted_diff[i]
                diff2 = sorted_diff[i+1]
                temp = abs(sorted_diff[i]-sorted_diff[i+1])

        thresh = (scr1+scr2)/2
        #print "Threshold: ", thresh, scr1, scr2
        pred = [0]*len(scores)
        classes = [i for i, x in enumerate(scores) if x > thresh]
        for i in classes:
            pred[i] = 1
        predictions.append(pred)
    return predictions




"""
def P_Cut(all_scores):
    for scores in all_scores:
"""
