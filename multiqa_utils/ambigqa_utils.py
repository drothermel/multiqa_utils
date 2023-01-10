


## Goal: Create Ground Truth from AmbigQA Data File
# (less obvious how to do this bc the questions have multiple
#  conflicting annotations often)

# Convert all annotations into a list of lists of answer aliases
# (necessary bc single answers aren't lists initially)
def process_ambigqa_annotation(ann):
    if ann['type'] == 'singleAnswer':
        return [ann['answer']]
    elif ann['type'] == 'multipleQAs':
        return [qap['answer'] for qap in ann['qaPairs']]
    else:
        assert False

        
# For each question choose the (last) MultiQA annotation if exists
# (clearly, room for improvement here)
def process_ambigqa_qdata(qdata):
    the_answers = (None, None)
    for ann in qdata['annotations']:
        ans = process_ambigqa_annotation(ann)
        if ann['type'] == 'multipleQAs':
            return ans
        elif the_answers[0] == None:
            the_answers = (ann['type'], ans)
    return the_answers[1]


# Takes the original file in list form and converts to ground truth
# [(question, answer_alias_list), ...]
def convert_original_to_ground_truth(original_data):
    ground_truth = {}
    for qdata in original_data:
        ground_truth[qdata['question']] = process_ambigqa_qdata(qdata)
    return ground_truth