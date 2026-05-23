import httpx
import xml.etree.ElementTree as ET
from pathlib import Path


ARXIV_API = "https://export.arxiv.org/api/query"


def _strip_version(arxiv_id: str) -> str:
    return arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id


def fetch_arxiv_papers(arxiv_ids: list[str]) -> list[dict]:
    id_list = ",".join(arxiv_ids)
    url = f"{ARXIV_API}?id_list={id_list}&max_results={len(arxiv_ids)}"

    response = httpx.get(url, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    ns = {"a": "http://www.w3.org/2005/Atom"}

    papers = []
    for entry in root.findall("a:entry", ns):
        id_el = entry.find("a:id", ns)
        raw_id = id_el.text.strip().split("/")[-1] if id_el is not None else ""

        title_el = entry.find("a:title", ns)
        title = title_el.text.strip().replace("\n", " ") if title_el is not None else ""

        summary_el = entry.find("a:summary", ns)
        summary = summary_el.text.strip().replace("\n", " ") if summary_el is not None else ""

        published_el = entry.find("a:published", ns)
        published = published_el.text[:10] if published_el is not None else ""

        papers.append({
            "id": _strip_version(raw_id),
            "title": title,
            "summary": summary,
            "authors": [
                author.find("a:name", ns).text for author in entry.findall("a:author", ns)
            ],
            "published": published,
        })

    return papers


def extract_text_from_pdf(pdf_path: str) -> str:
    import fitz
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def download_arxiv_pdf(arxiv_id: str, output_dir: str) -> str | None:
    base_id = _strip_version(arxiv_id)
    pdf_url = f"https://arxiv.org/pdf/{base_id}.pdf"
    output_path = Path(output_dir) / f"{base_id}.pdf"

    if output_path.exists():
        return str(output_path)

    response = httpx.get(pdf_url, follow_redirects=True, timeout=60)
    if response.status_code != 200:
        return None

    output_path.write_bytes(response.content)
    return str(output_path)
