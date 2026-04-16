import httpx
from bs4 import BeautifulSoup

url = "https://blogs.microsoft.com/blog/2026/04/14/surface-hub-discontinued/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/",
}

response = httpx.get(url, headers=headers, timeout=10, follow_redirects=True)
print(f"Status: {response.status_code}")
soup = BeautifulSoup(response.text, "html.parser")
for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
    tag.decompose()
for selector in ["article", "main", "[role='main']", "body"]:
    container = soup.select_one(selector)
    if container:
        text = container.get_text(separator=" ", strip=True)
        words = len(text.split())
        if words >= 50:
            print(f"Words: {words}, Selector: {selector}")
            print(f"Preview: {text[:200]}")
            break
