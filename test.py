import pickle
from tqdm import tqdm
from nltk.tokenize.treebank import TreebankWordDetokenizer
detokenize = TreebankWordDetokenizer().detokenize
import re 
import argparse
from accent_utils import process_line
import numpy as np
from multiprocessing import Pool

def get_arg():
  parse = argparse.ArgumentParser()
  parse.add_argument('test_data_file')
  parse.add_argument('ground_truth_file')
  parse.add_argument('model_dir')
  return parse.parse_args()

def remove_vn_accent(word):
    word = re.sub('[áàảãạăắằẳẵặâấầẩẫậ]', 'a', word)
    word = re.sub('[éèẻẽẹêếềểễệ]', 'e', word)
    word = re.sub('[óòỏõọôốồổỗộơớờởỡợ]', 'o', word)
    word = re.sub('[íìỉĩị]', 'i', word)
    word = re.sub('[úùủũụưứừửữự]', 'u', word)
    word = re.sub('[ýỳỷỹỵ]', 'y', word)
    word = re.sub('đ', 'd', word)
    return word
    
def vn_dict():
  with open('vn_syllables.txt', encoding='utf8') as fin:
    vnword = fin.read().splitlines()
  vndict = {}
  for word in vnword:
    no_accent = remove_vn_accent(word)
    if not no_accent in vndict.keys():
      vndict.setdefault(no_accent, [word])
    else:
      vndict[no_accent].append(word)
  return vndict

vndict = vn_dict()

def gen_accents(word):
  word = remove_vn_accent(word.lower())
  if word in vndict.keys():
    return vndict[word]
  else:
    return [word]

# beam search
def beam_search(words, model, k=3):
  #preprocess
  sequences = []
  for idx, word in enumerate(words):
    if idx == 0:
      sequences = [([x], 0.0) for x in gen_accents(word)]
    else:
      all_sequences = []
      for seq in sequences:
        for next_word in gen_accents(word):
          current_word = seq[0][-1]
          try:
              previous_word = seq[0][-2]
              score = model.logscore(next_word, [previous_word, current_word])
          except:
              score = model.logscore(next_word, [current_word])
          new_seq = seq[0].copy()
          new_seq.append(next_word)
          all_sequences.append((new_seq, seq[1] + score))
      all_sequences = sorted(all_sequences,key=lambda x: x[1], reverse=True)
      sequences = all_sequences[:k]
  return sequences

# args = get_arg()

# with open(args.model_dir,'rb') as fin:
#   model = pickle.load(fin)
#   print('load model done')


# if __name__=='__main__':

def translate(sent, model_sent, k):
  sent = sent.replace('\n','')
  result = beam_search(sent.lower().split(), model_sent, k)
  return detokenize(result[0][0])

def evaluate(pred, label):
    _, pred_punc = process_line(pred)
    _, label_punc = process_line(label)

    pred_punc = np.array(pred_punc)
    label_punc = np.array(label_punc)

    true_values = np.sum(pred_punc==label_punc)
    n_values = len(pred_punc)

    return true_values, n_values

args = get_arg()
with open(args.model_dir, 'rb') as fin:
  model = pickle.load(fin)
  print('load model done')

# beam_search_param = 1
# beam_search_param = 2
beam_search_param = 3
def translate1(sent):
  return translate(sent, model, beam_search_param)

def pool_handler(data):
  print('compute val_acc for beam_search_param = %s, process with 4 pool'%beam_search_param)
  with Pool(4) as p:
    temp_result = list(tqdm(p.imap(translate1, data), total=len(data)))
  return temp_result
  

if __name__=='__main__':
  

  # with open(args.model_dir,'rb') as fin:
  #   model = pickle.load(fin)
  # print('load model done')

  test_data_lines = None
  ground_truth_lines = None
  with open(args.test_data_file, "r", encoding='utf-8') as f:
      test_data_lines = f.readlines()

  print('load data done')

  # beam_search_param = 1
  # def translate1(sent):
  #   return translate(sent, model, beam_search_param)

  # with Pool(4) as p:
  #   temp_result = list(tqdm(p.imap(translate1, test_data_lines), total=len(test_data_lines)))
  temp_result = pool_handler(test_data_lines)
  del test_data_lines

  total_true_values = 0
  total_values = 0
  with open(args.ground_truth_file, "r", encoding='utf-8') as f:
      ground_truth_lines = f.readlines()

  for i in range(len(temp_result)):
    true_values, n_values = evaluate(temp_result[i], ground_truth_lines[i])
    total_true_values += true_values
    total_values += n_values

  print("Avg. Accuracy: {}".format(total_true_values / total_values))



  
  
