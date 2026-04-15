import httpx
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; PMDigestBot/1.0; "
        "+https://github.com/mounicasirineni/pm-ai-agents-2026)"
    )
}

MIN_WORD_THRESHOLD = 50

test_urls = [
    # Amazon News - known 0-word failures
    "https://www.aboutamazon.com/news/company-news/amazon-globalstar-apple",
    "https://www.aboutamazon.com/news/entertainment/your-fault-london-prime-video",
    # Google Blog - Low Confidence today
    "https://blog.google/technology/ai/gemini-robotics-er-1-6/",  # adjust to actual URL
    # Microsoft Blog - verify
    "https://blogs.microsoft.com/blog/2026/04/14/surface-hub-discontinued/",  # adjust to actual URL
]

for url in test_urls:
    try:
        response = httpx.get(url, headers=HEADERS, timeout=10, follow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        result = {"url": url, "status": response.status_code, "text": None, "words": 0}

        for selector in ["article", "main", "[role='main']", "body"]:
            container = soup.select_one(selector)
            if container:
                text = container.get_text(separator=" ", strip=True)
                words = len(text.split())
                if words >= MIN_WORD_THRESHOLD:
                    result["text"] = text[:200]  # preview only
                    result["words"] = words
                    result["selector"] = selector
                    break

        print(f"\nURL: {url}")
        print(f"Status: {result['status']}")
        print(f"Words: {result['words']}")
        print(f"Selector: {result.get('selector', 'none matched')}")
        print(f"Preview: {result['text'][:200] if result['text'] else 'EMPTY'}")

    except Exception as e:
        print(f"\nURL: {url}")
        print(f"FAILED: {e}")
