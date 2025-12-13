import os
import json
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


def filter_for_wikipedia(row):
    metadata = row["metadata"]
    urls = metadata["urls"]

    for url in urls:
        if "https://en.wikipedia.org/" in url:
            return True
    return False


def clean_urls(example):
    metadata = example["metadata"]
    urls = metadata["urls"]

    refined_urls = []
    for url in urls:
        if url.count("wikipedia.org") > 1 or url.count("https") > 1 or url.count(" ") > 0 or url.count("\n") > 0:
            continue
        refined_urls.append(url)
    metadata["urls"] = refined_urls
    example["metadata"] = metadata
    return example


def main():
    dataset = load_dataset(
        "OpenEvals/SimpleQA", split="test").filter(filter_for_wikipedia)
    dataset = dataset.map(clean_urls)

    dataset.push_to_hub("mfirth/simplewikiqa", split="train",
                        token=os.getenv("HF_TOKEN"))


if __name__ == "__main__":
    main()
