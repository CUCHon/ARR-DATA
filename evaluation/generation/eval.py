import argparse
import json
import os
import time
import numpy as np
import sys
from openai import AsyncOpenAI
from tqdm import tqdm
import asyncio
from typing import Any, List, Dict
import logging

import tiktoken
gpt_encoder = tiktoken.get_encoding("cl100k_base")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def dispatch_openai_requests(
    client,
    messages_list: List[List[Dict[str, Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> List[Dict]:
    """Dispatches requests to OpenAI API asynchronously using the new client."""
    async_responses = []
    for x in messages_list:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=x,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                timeout=10,  # Reduced timeout to 10 seconds
            )
            async_responses.append(response)
        except Exception as e:
            print(f"Error in request: {e}")
            raise
    
    return async_responses

def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            raise Exception("Invalid score pair.")
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return [-1, -1]

def gen_prompt(ques, ans1, ans2):
    sys_prompt = "You are a helpful and precise assistant for checking the quality of the answer."
    prompt_template = "[Question]\n{question}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{criteria}\n\n"
    criteria = "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."
    prompt = prompt_template.format(
        question=ques, answer_1=ans1, answer_2=ans2, criteria=criteria
    )
    return sys_prompt, prompt

def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wraped_file", default='')
    parser.add_argument("--api_key", type=str, default='')
    parser.add_argument("--api_model", type=str, default='gpt-3.5-turbo')
    parser.add_argument("--api_base", type=str, default='')
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Batch size to call OpenAI GPT",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    args = parser.parse_args()
    
    # Initialize the async client with the new format
    client_args = {"api_key": args.api_key}
    if args.api_base != '':
        client_args["base_url"] = args.api_base
    
    client = AsyncOpenAI(**client_args)
    
    print('Begin:', args.wraped_file)
    print(f'Using model: {args.api_model}')
    print(f'Batch size: {args.batch_size}')

    # Load the wrapped data
    try:
        with open(args.wraped_file, 'r') as f:
            wraped_info = json.load(f)
        print(f"Successfully loaded wrapped file with {len(wraped_info['data'])} items")
    except Exception as e:
        print(f"Error loading wrapped file: {e}")
        sys.exit(1)

    meta_info = wraped_info['Meta_Info']
    dataset_name = meta_info['dataset_name']
    qa_jsons = wraped_info['data']

    if dataset_name == "vicuna":
        prompt_key = 'text'
    elif dataset_name == "koala":
        prompt_key = 'prompt'
    elif dataset_name == "sinstruct":
        prompt_key = 'instruction'
    elif dataset_name == "wizardlm":
        prompt_key = 'Instruction'
    elif dataset_name == "lima":
        prompt_key = 'conversations'
    else:
        print(f"Warning: Unknown dataset name {dataset_name}, using 'text' as default prompt key")
        prompt_key = 'text'

    total_len = len(qa_jsons)
    question_idx_list = list(range(total_len))

    predictions_all = []
    for reverse in range(2):  # reverse or not
        print(f"Processing {'reverse' if reverse else 'normal'} order evaluations")
        message_list = []
        token_len_list = []

        for i in question_idx_list:
            instruction = qa_jsons[i][prompt_key]
            ques = instruction

            if reverse:  # reverse = 1, secondly
                ans1 = qa_jsons[i]['Answer2']
                ans2 = qa_jsons[i]['Answer1']
            else:  # reverse = 0, firstly
                ans1 = qa_jsons[i]['Answer1']
                ans2 = qa_jsons[i]['Answer2']
            
            sys_prompt, prompt = gen_prompt(ques, ans1, ans2)

            message = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ]
            message_list.append(message)
            token_len_list.append(len(gpt_encoder.encode(prompt)))

        predictions = []
        i = 0
        wait_base = 1  # Reduced wait time to 1 second
        retry_limit = 2  # Reduced retry limit
        error = 0
        pbar = tqdm(total=len(message_list))
        batch_size = min(args.batch_size, len(message_list))
        
        print(f"Starting API calls with {len(message_list)} messages in batches of {batch_size}")
        
        while i < len(message_list):
            end_idx = min(i + batch_size, len(message_list))
            current_batch = message_list[i:end_idx]
            current_token_lens = token_len_list[i:end_idx]
            
            if not current_batch:
                print("Warning: Empty batch, skipping")
                i += batch_size
                continue
                
            token_limit_in_current_batch = min(args.max_tokens, 4070 - max(current_token_lens))
            
            retry_count = 0
            while retry_count < retry_limit:
                try:
                    print(f"Calling API for batch {i} to {end_idx-1} (size: {len(current_batch)})")
                    
                    batch_predictions = asyncio.run(
                        dispatch_openai_requests(
                            client=client,
                            messages_list=current_batch,
                            model=args.api_model,
                            temperature=0.0,
                            max_tokens=token_limit_in_current_batch,
                            top_p=1.0,
                        )
                    )
                    
                    predictions.extend(batch_predictions)
                    i = end_idx
                    pbar.update(len(current_batch))
                    print(f"Successfully processed batch with {len(current_batch)} items")
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    retry_count += 1
                    error += 1
                    print(f"Batch error ({i} to {end_idx-1}), attempt {retry_count}/{retry_limit}")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {str(e)}")
                    
                    if retry_count >= retry_limit:
                        print(f"Reached retry limit for batch {i} to {end_idx-1}, skipping batch")
                        # Skip this batch after max retries
                        i = end_idx
                        pbar.update(len(current_batch))
                    else:
                        print(f"Waiting {wait_base} seconds before retry...")
                        time.sleep(wait_base)
                        wait_base = min(wait_base * 2, 2)  # Exponential backoff with max 2s
            
        pbar.close()
        predictions_all.append(predictions)
        print(f"Completed {'reverse' if reverse else 'normal'} order with {len(predictions)} successful responses")

    # Process scores and reviews
    print("Processing scores and reviews...")
    all_scores = []
    for reverse in range(2):
        scores_list = []
        predictions = predictions_all[reverse]
        
        if not predictions:
            print(f"Warning: No predictions for {'reverse' if reverse else 'normal'} order")
            continue
            
        for idx, prediction in enumerate(predictions):
            if idx >= len(qa_jsons):
                print(f"Warning: prediction index {idx} exceeds qa_jsons length {len(qa_jsons)}")
                continue
                
            try:
                # Adapt to new API response format
                review = prediction.choices[0].message.content
                scores = parse_score(review)
                review_key = 'review' if not reverse else 'review_reverse'
                scores_key = 'scores' if not reverse else 'scores_reverse'
                qa_jsons[idx][review_key] = review
                qa_jsons[idx][scores_key] = str(scores)
                scores_list.append(scores)
            except Exception as e:
                print(f"Error processing prediction {idx}: {e}")
                print(f"Prediction structure: {prediction}")
        
        if scores_list:
            all_scores.append(scores_list)
            avg_scores = np.array(scores_list).mean(0)
            avg_key = 'average_scores' if not reverse else 'average_scores_reverse'
            meta_info[avg_key] = str(avg_scores.tolist())
            print(f"Average scores for {'reverse' if reverse else 'normal'} order: {avg_scores}")

    # Determine output filename based on model
    wraped_info['Meta_Info'] = meta_info
    wraped_info['data'] = qa_jsons
    
    if 'gpt-4-turbo' in args.api_model:
        output_review_file = args.wraped_file.strip('.json') + '_reviews_gpt4_turbo.json'
    elif 'gpt-4.1-mini' in args.api_model:
        output_review_file = args.wraped_file.strip('.json') + '_reviews_gpt4.1-mini.json'
    elif 'gpt-4.1' in args.api_model:
        output_review_file = args.wraped_file.strip('.json') + '_reviews_gpt4.1.json'
    elif 'gpt-4o' in args.api_model :
        output_review_file = args.wraped_file.strip('.json') + '_reviews_gpt4o.json'
    elif 'gpt-4' in args.api_model:
        output_review_file = args.wraped_file.strip('.json') + '_reviews_gpt4.json'
    elif 'gpt-3.5' in args.api_model:
        output_review_file = args.wraped_file.strip('.json') + '_reviews_gpt3.5.json'
    else:
        # Use model name directly if it doesn't match known patterns
        output_review_file = args.wraped_file.strip('.json') + f'_reviews_{args.api_model.replace("-", "_")}.json'
    
    print(f"Saving results to {output_review_file}")
    with open(f"{output_review_file}", "w") as f:
        json.dump(wraped_info, f, indent=4)

    print('Finish:', args.wraped_file)