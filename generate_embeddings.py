import json
import time
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

t0 = time.time()
candidates_file = r"c:\all\indiaruns-hackathon\data-and-ai-challange\India_runs_data_and_ai_challenge\candidates.jsonl"
output_embeddings_file = r"c:\all\indiaruns-hackathon\embeddings.npy"
output_ids_file = r"c:\all\indiaruns-hackathon\candidate_ids.json"
model_dir = r"c:\all\indiaruns-hackathon\model_weights"

print("Loading local model...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir)
# Set evaluation mode and disable grads for speed/memory
model.eval()
torch.set_grad_enabled(False)

# Check if CUDA is available (though rules say CPU only, let's check anyway as fallback)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

print("Extracting candidate profiles...")
texts = []
candidate_ids = []

with open(candidates_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        cid = obj["candidate_id"]
        candidate_ids.append(cid)
        
        profile = obj.get("profile", {})
        headline = profile.get("headline", "")
        summary = profile.get("summary", "")
        yoe = profile.get("years_of_experience", 0)
        curr_title = profile.get("current_title", "")
        curr_company = profile.get("current_company", "")
        
        skills = [sk.get("name", "") for sk in obj.get("skills", []) if sk.get("name")]
        skills_str = ", ".join(skills)
        
        jobs = []
        for job in obj.get("career_history", []):
            title = job.get("title", "")
            company = job.get("company", "")
            desc = job.get("description", "")
            jobs.append(f"{title} at {company}: {desc}")
        jobs_str = " | ".join(jobs)
        
        text = f"Headline: {headline}. Summary: {summary}. Total Experience: {yoe} years. Current Role: {curr_title} at {curr_company}. Skills: {skills_str}. Career History: {jobs_str}."
        texts.append(text)

print(f"Extracted {len(texts)} candidates in {time.time()-t0:.2f}s. Starting embedding generation...")

# Mean Pooling function
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

batch_size = 128
all_embeddings = []

t_start = time.time()
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=256)
    # Move batch to device
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    
    all_embeddings.append(sentence_embeddings.cpu().numpy())
    
    if (i // batch_size) % 50 == 0:
        elapsed = time.time() - t_start
        processed = min(i + batch_size, len(texts))
        rate = processed / elapsed
        eta = (len(texts) - processed) / rate
        print(f"Embedded {processed}/{len(texts)} ({processed/len(texts)*100:.1f}%) | Speed: {rate:.1f} sent/sec | ETA: {eta/60:.1f} min")

print("Saving embeddings...")
embeddings_np = np.vstack(all_embeddings)
np.save(output_embeddings_file, embeddings_np)

with open(output_ids_file, "w", encoding="utf-8") as f_ids:
    json.dump(candidate_ids, f_ids)

print(f"Pre-computation complete! Saved {embeddings_np.shape} to {output_embeddings_file}. Total time: {(time.time()-t0)/60:.2f} min.")
