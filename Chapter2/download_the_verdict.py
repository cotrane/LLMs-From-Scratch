import urllib.request


def download_the_verdict():
    """Download dataset 'the-verdict.txt' from GitHub.

    Returns:
        str: Path to the downloaded file
    """
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )

    file_path = "the-verdict.txt"

    urllib.request.urlretrieve(url, file_path)

    print(f"Downloaded the verdict to {file_path}")
    return file_path


if __name__ == "__main__":
    download_the_verdict()
