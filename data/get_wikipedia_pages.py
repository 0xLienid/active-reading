import os
import wikipedia
from urllib.parse import parse_qs, unquote, urlparse
from datasets import load_dataset, Dataset
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


def wikipedia_url_to_title(url: str) -> str:
    """
    Convert a Wikipedia URL into a human-readable page title that works with
    the `wikipedia` Python package.

    Example:
        "https://en.wikipedia.org/wiki/2021%E2%80%9322_Premier_League#League_table"
        -> "2021–22 Premier League"
    """
    if url is None:
        raise TypeError("url must be a string, got None")

    url = url.strip()
    if not url:
        raise ValueError("url must be a non-empty string")

    parsed = urlparse(url)

    # Supports URLs like:
    # - https://en.wikipedia.org/wiki/Foo_Bar
    # - https://en.wikipedia.org/w/index.php?title=Foo_Bar&oldid=123
    raw_title: str | None = None

    qs = parse_qs(parsed.query)
    if "title" in qs and qs["title"]:
        raw_title = qs["title"][0]
    else:
        path = parsed.path or ""
        wiki_prefix = "/wiki/"
        if wiki_prefix in path:
            raw_title = path.split(wiki_prefix, 1)[1]

    if not raw_title:
        raise ValueError(
            f"Could not extract Wikipedia title from URL: {url!r}")

    # Wikipedia encodes spaces as underscores in titles; URLs may also be percent-encoded.
    title = unquote(raw_title)
    title = title.replace("_", " ").strip()

    if not title:
        raise ValueError(f"Extracted empty Wikipedia title from URL: {url!r}")

    return title


def get_wikipedia_page(url: str) -> str:
    """
    Get the Wikipedia page for the given title.
    """
    try:
        title = wikipedia_url_to_title(url)
        if not title:
            raise ValueError(
                f"Could not extract Wikipedia title from URL: {url!r}")

        return title, wikipedia.page(title=title, auto_suggest=False, redirect=True).content
    except Exception as e:
        print(f"Page error for URL: {url} and title: {title}: {e}")
        return None, None


def main():
    CHECKPOINT_SIZE = 100

    print(f"Setting Wikipedia language to en and rate limiting to True")
    wikipedia.set_lang("en")
    wikipedia.set_rate_limiting(True)

    print(f"Loading dataset")
    dataset = load_dataset("mfirth/simplewikiqa",
                           split="train")

    print(f"Building set of all Wikipedia URLs")
    all_urls = []
    for row in dataset:
        urls = row["metadata"]["urls"]
        for url in urls:
            if "https://en.wikipedia.org/" in url:
                all_urls.append(url)

    all_urls = list(set(all_urls))

    print(f"Getting Wikipedia pages for {len(all_urls)} URLs")
    data = []
    for url in tqdm(all_urls):
        title, page = get_wikipedia_page(url)
        if title and page:
            data.append({
                "url": url,
                "title": title,
                "page": page
            })

        if len(data) % CHECKPOINT_SIZE == 0:
            print(f"Saving checkpoint {len(data) // CHECKPOINT_SIZE}")
            dataset = Dataset.from_list(data)
            dataset.push_to_hub("mfirth/simplewikiqa-pages", split="train",
                                token=os.getenv("HF_TOKEN"))

    print(f"Saving final dataset")
    dataset = Dataset.from_list(data)
    dataset.push_to_hub("mfirth/simplewikiqa-pages", split="train",
                        token=os.getenv("HF_TOKEN"))


if __name__ == "__main__":
    main()
