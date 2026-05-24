"""
Web arama modülü - DuckDuckGo (ücretsiz, API key gerektirmez)
"""
from duckduckgo_search import DDGS


def web_search(query: str, max_results: int = 4) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results, region="tr-tr"))
        if not results:
            # Try without region restriction
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "Arama sonucu bulunamadı."
        lines = []
        for r in results:
            lines.append(f"📌 {r['title']}\n{r['body']}\n🔗 {r['href']}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Arama hatası: {str(e)[:150]}"
