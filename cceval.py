import os
import re
import json
import sys
import argparse
import openai
from abc import ABC, abstractmethod
from tqdm import tqdm
import time


class BaseAPIWrapper(ABC):
    @abstractmethod
    def get_completion(self, user_prompt, system_prompt=None):
        pass

class OpenAIAPIWrapper(BaseAPIWrapper):
    def __init__(self, caller_name="default", key_pool=None, temperature=0, model="gpt-4-32k-0613", time_out=30):
        self.key_pool = key_pool
        self.temperature = temperature
        self.model = model
        self.time_out = time_out
        openai.api_base = ""
        openai.api_type = "azure"
        openai.api_version = "2023-06-01-preview" 
        openai.api_key = key_pool[0]

    def request(self, system_content, usr_question):
        response = openai.ChatCompletion.create(
            engine="gpt_openapi",
            messages=[
                {"role": "system", "content": f"{system_content}"},
                {"role": "user", "content": f"{usr_question}"}
            ],
            temperature=self.temperature, 
            model=self.model
        )
        resp = response.choices[0]['message']['content']
        total_tokens = response.usage['total_tokens']

        return resp, total_tokens
    
    def get_completion(self, user_prompt=None, system_prompt=None,max_try=10):
        gpt_cv_nlp = '[]'
        key_i = 0
        total_tokens = 0
        while max_try > 0:
            try:
                gpt_cv_nlp, total_tokens = self.request(system_prompt, user_prompt)
                max_try = 0
                break
            except:
                print("fail ", max_try)
                key = self.key_pool[key_i%2]
                openai.api_key = key
                key_i += 1
                time.sleep(self.time_out)
                max_try -= 1
    
        return gpt_cv_nlp, total_tokens



def load_generated_captions(cap_file):      
   caps = json.load(open(cap_file))
   try:
       metrics = {}
       caps = caps
       imids = set([cap['image_id'] for cap in caps])
   except:
       raise Exception("Expect caption file to consist of a dectionary with sentences correspdonding to the key 'imgToEval'")

   return caps, imids, metrics


class CHAIR(object):
    def __init__(self, key):
        self.openai_obj = OpenAIAPIWrapper(key_pool=[key])
        with open('./prompt/region_cap2obj_prompt.txt', 'r') as file:
            content = file.read()
        self.region_user_prompt = content
        with open('./prompt/cap2obj_prompt_bracket.txt', 'r') as file:
            content = file.read()
        self.cap_user_prompt = content
        with open('./prompt/hallucination_prompt.txt', 'r') as file:
            content = file.read()
        self.hall_user_prompt = content
        with open('./prompt/coverage_prompt.txt', 'r') as file:
            content = file.read()
        self.coverage_user_prompt = content
        self.system_prompt = (
            "I am ChatGPT, a virtual assistant based on OpenAI's GPT-4 model." 
            "I'm designed to understand and generate human-like text based on the input I receive." 
            "My main purpose is to assist with information, answer questions, help with tasks that involve" 
            "natural language processing, and engage in conversations with users."
            "Please note that while I aim to provide accurate and reliable information."
        )
    
    def list_region2cap(self, list_regions):
        user_prompt = self.region_user_prompt.format_map({'list_of_regions':list(set(list_regions))})
        gpt_ret, total_tokens = self.openai_obj.get_completion(user_prompt=user_prompt, system_prompt=self.system_prompt,max_try=10)
        match = re.search(r"\[(.*?)\]", gpt_ret)
        if match:
            objects_list_str = match.group(1)
            objects_in_image = [item.strip(" '") for item in objects_list_str.split(",")]
            return list(set(objects_in_image))
        else:
            return []
    
    def cap2objs_gpt4(self, cap):
        user_prompt = self.cap_user_prompt.format_map({'cap':cap})
        gpt_ret, total_tokens = self.openai_obj.get_completion(user_prompt=user_prompt, system_prompt=self.system_prompt,max_try=10)
        match = re.search(r"\[(.*?)\]", gpt_ret)
        if match:
            objects_list_str = match.group(1)
            objects_in_image = [item.strip(" '") for item in objects_list_str.split(",")]
            return objects_in_image
        else:
            return []
    
    def cap2objs_spacy(self, cap):
        return []
    
    def get_hall_gpt4(self, gt, cap_obj):
        user_prompt = self.hall_user_prompt.format_map({'gt':gt, 'cap_obj':cap_obj})
        gpt_ret, total_tokens = self.openai_obj.get_completion(user_prompt=user_prompt, system_prompt=self.system_prompt,max_try=10)
        match = re.search(r"\[(.*?)\]", gpt_ret)
        if match:
            objects_list_str = match.group(1)
            objects_in_image = [item.strip(" '") for item in objects_list_str.split(",")]
            return list(set(objects_in_image))
        else:
            return []
    
    def compute_chair_vg(self, cap_file, vg_path='./vg_info_100.json'):
        image_infos = json.load(open(vg_path))

        num_caps = 0.
        num_hallucinated_caps = 0.
        hallucinated_word_count = 0.
        coco_word_count = 0.
        all_param_words = 0.0
        coco_part_word_count = 0.
        caps = json.load(open(cap_file))
        caps = caps[:100]
        output = {'sentences': []} 
        avg_len = 0
    
        for i, cap_eval in tqdm(enumerate(caps), total=len(caps)):
            cap = cap_eval['text']
            imid = cap_eval['image_id']
            if str(i+1) in image_infos or (i+1 in image_infos):
                gt_objects = image_infos[str(i+1)]['gt_objs']
            else:
                exit()
            raw_words = self.cap2objs_gpt4(cap)
            param_words = re.findall(r'\[(.*?)\]', cap)
            param_words = list(set(param_words))
            raw_words = list(set(raw_words))
            all_objects = list(set(raw_words[:] + param_words))

            hallucinated_words = self.get_hall_gpt4(gt_objects, raw_words)
            hallucinated_words = [item for item in hallucinated_words if item != '']
            sent_len = len(cap.split(' '))
            avg_len += sent_len

            all_param_words += len(param_words)
            coco_part_word_count += len(raw_words)
            coco_word_count += len(all_objects)
            hallucinated_word_count += len(hallucinated_words)
            num_caps += 1
            if len(hallucinated_words) > 0:
                num_hallucinated_caps += 1
        
        with open("./vg_info_100.json", "w") as file:
            json.dump(image_infos, file, indent=4)
        chair_s = (num_hallucinated_caps/num_caps)
        chair_i = (hallucinated_word_count/coco_word_count)
        chair_i_n = (hallucinated_word_count/coco_part_word_count)
        output['overall_metrics'] = {
                                     'CHAIRs': chair_s,
                                     'CHAIRi': chair_i,
                                     'CHAIRi_n': chair_i_n,
                                     'sentence len':avg_len / num_caps,
                                     'avg brack objects': all_param_words / num_caps,
                                     'avg objects': coco_word_count / num_caps,
                                     'avg all objects': coco_part_word_count / num_caps
                                     }
    
        return output
    
    def get_uncover_gpt4(self, gt, cap_obj):
        user_prompt = self.coverage_user_prompt.format_map({'cap_obj':cap_obj, 'gt':gt})
        gpt_ret, total_tokens = self.openai_obj.get_completion(user_prompt=user_prompt, system_prompt=self.system_prompt,max_try=10)

        match = re.search(r"\[(.*?)\]", gpt_ret)
        # print(gpt_ret)
        # print('Not covered words are ', match)
        # print('gt is: ', gt)
        # print('cap_obj is: ', cap_obj)
        if match:
            objects_list_str = match.group(1)
            # Split the string into a list of items
            objects_in_image = [item.strip(" '") for item in objects_list_str.split(",")]
            return list(set(objects_in_image))
        else:
            return []
    
    def converage(self, cap_file, vg_path='./vg_info_100.json'):
        image_infos = json.load(open(vg_path))
        num_caps = 0.
        caps = json.load(open(cap_file))
        caps = caps[:100]

        output = {'sentences': []} 
        avg_len = 0
        coco_word_count = 0.
        uncover_word_counts = 0
        num_uncovered_count = 0
        sent_objects = 0

        for i, cap_eval in tqdm(enumerate(caps), total=len(caps)):

            # Remove instructions
            cap = cap_eval['text']
            imid = cap_eval['image_id']
            if str(i+1) in image_infos or (i+1 in image_infos):
                gt_objects = image_infos[str(i+1)]['gt_objs']
            else:
                exit()
            raw_words = self.cap2objs_gpt4(cap)
            param_words = re.findall(r'\[(.*?)\]', cap)
            raw_words = list(set(raw_words[:] + param_words))
            gt_objects = list(set(gt_objects))

            uncover_words = self.get_uncover_gpt4(gt_objects, raw_words)
            uncover_words = [item for item in uncover_words if item != '']
            sent_len = len(cap.split(' '))
            avg_len += sent_len

            sent_objects += len(raw_words)
            coco_word_count += len(gt_objects) 
            # print('total len: ', coco_word_count)
            # print('cap items len: ', len(raw_words))
            # print('uncovered words len: ', len(uncover_words), uncover_words, type(uncover_words))
            uncover_word_counts += len(uncover_words)
            # print('total uncovered word is ', uncover_word_counts)
            # print('coverage is ', (coco_word_count-uncover_word_counts)/coco_word_count)
            num_caps += 1
            if len(uncover_words) > 0:
                num_uncovered_count += 1
        
        uncover_s = (num_uncovered_count/num_caps)
        uncover_i = (uncover_word_counts/coco_word_count)
        output['overall_metrics'] = {
                                     'Uncovers': uncover_s,
                                     'Uncoveri': uncover_i,
                                     'Coveri': 1 - uncover_i,
                                     'sentence len':avg_len / num_caps,
                                     'avg gt objects': coco_word_count / num_caps,
                                     'avg cap objects': sent_objects / num_caps
                                     }
        return output




        
def print_metrics(hallucination_cap_dict, quiet=False):
    sentence_metrics = hallucination_cap_dict['overall_metrics']
    metric_string = "CHAIRs: %0.01f\t CHAIRi: %0.01f\t Sentence length: %0.01f\t Avg objects: %0.01f\t" %(sentence_metrics['CHAIRs']*100, 
                                       sentence_metrics['CHAIRi']*100, sentence_metrics['sentence len'], sentence_metrics['avg objects'])
    print(metric_string)
    return metric_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap_file", type=str, default='CAPTION FILE PATH')
    parser.add_argument("--coverage", type=bool, default=False)
    parser.add_argument("--key", type=str, default='OPENAI_API_KEY')
    args = parser.parse_args()

    _, imids, _ = load_generated_captions(args.cap_file)
    evaluator = CHAIR(args.key) 
    if not args.coverage:
        cap_dict = evaluator.compute_chair_vg(args.cap_file, vg_path='./vg_info_100.json') 
        print_metrics(cap_dict)
    else:
        coverage_dict = evaluator.converage(args.cap_file, vg_path='./vg_info_100.json')
        print(coverage_dict)
    
