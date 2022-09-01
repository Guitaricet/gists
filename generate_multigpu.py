# torchrun --nproc_per_node 2 generate_multigpu.py
# tested on torch==1.12.1 and transformers==
import os
import json

import torch
import torch.distributed as dist

import transformers
import datasets

from tqdm import tqdm

MODEL_NAME = "facebook/opt-350m"
PER_DEVICE_BATCH_SIZE = 8
MAX_BATCHES = 2


if __name__ == '__main__':
    dist.init_process_group(backend="nccl")

    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = dist.get_rank()  # in case you use multiple nodes

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)  # required for dist.gather_object
    model.to(device)

    dataset = datasets.load_dataset("wikitext", "wikitext-2-v1")["train"]
    ddp_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=PER_DEVICE_BATCH_SIZE, sampler=ddp_sampler)

    all_predictions_str = []
    print(f"RANK {global_rank}: Starting generation")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, disable=local_rank != 0)):
            if i >= MAX_BATCHES:
                break

            encoded = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True, max_length=128)
            encoded = encoded.to(device)

            predictions = model.generate(**encoded, do_sample=True, max_length=256)
            predictions = predictions.cpu().numpy()
            predictions_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            all_predictions_str.extend(predictions_str)

    print(f"RANK {global_rank}: Finished generation")

    # collect all of the predicted strings from all GPUs and send them to GPU 0
    all_predictions_str_gathered = None
    if global_rank == 0:
        all_predictions_str_gathered = dist.get_world_size() * [None]

    dist.gather_object(all_predictions_str, all_predictions_str_gathered, dst=0)

    if global_rank == 0:
        # flatten list
        all_predictions_str_gathered = [item for sublist in all_predictions_str_gathered for item in sublist]

        with open("predictions.txt", "w") as f:
            f.write("\n".join(all_predictions_str_gathered))
        print(f"RANK {global_rank}: Predictions saved to predictions.txt")
    print(f"RANK {global_rank}: Done")
