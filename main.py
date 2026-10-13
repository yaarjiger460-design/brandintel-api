import io
import os
import re
from collections import Counter
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI(
    title="BrandIntel API",
    description="High-speed B2B Company & Domain Intelligence API. Enrich company domains with high-res logos, brand colors, detected tech stack, social media profiles, and contact info.",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RAPIDAPI_PROXY_SECRET = os.getenv("RAPIDAPI_PROXY_SECRET", "")

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 BrandIntel/1.2",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def verify_rapidapi_request(secret_header: Optional[str]):
    if RAPIDAPI_PROXY_SECRET and secret_header != RAPIDAPI_PROXY_SECRET:
        raise HTTPException(
            status_code=403,
            detail="Access forbidden: Direct requests not allowed. Please access via RapidAPI marketplace.",
        )


def clean_domain(raw_domain: str) -> str:
    raw = raw_domain.strip().lower()
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        domain = parsed.netloc
    else:
        domain = raw.split("/")[0]

    domain = re.sub(r"^www\.", "", domain)
    if not re.match(r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", domain):
        raise HTTPException(status_code=400, detail=f"Invalid domain format: '{raw_domain}'")
    return domain


TECH_SIGNATURES: Dict[str, Dict[str, List[str]]] = {
    "Shopify": {
        "html": ["cdn.shopify.com", "shopify-buy", "Shopify.theme"],
    },
    "WordPress": {
        "html": ["wp-content", "wp-includes"],
    },
    "Next.js": {
        "html": ["__NEXT_DATA__", "/_next/static"],
    },
    "Webflow": {
        "html": ["assets.website-files.com", "webflow.js"],
    },
    "Squarespace": {
        "html": ["static1.squarespace.com"],
    },
    "Wix": {
        "html": ["static.parastorage.com", "wix.com"],
    },
    "React": {
        "html": ["data-reactroot", "react-dom", "_react"],
    },
    "Vue.js": {
        "html": ["data-v-", "vue.min.js"],
    },
    "Tailwind CSS": {
        "html": ["tailwind", "text-sm font-medium", "flex items-center justify-between"],
    },
    "Stripe": {
        "html": ["js.stripe.com"],
    },
    "Google Analytics / Tag Manager": {
        "html": ["googletagmanager.com/gtm.js", "google-analytics.com/analytics.js", "gtag("],
    },
    "HubSpot": {
        "html": ["js.hs-scripts.com", "hbspt.forms"],
    },
    "Intercom": {
        "html": ["widget.intercom.io"],
    },
}


def detect_tech_stack(html: str, headers: httpx.Headers) -> List[str]:
    detected = []
    html_lower = html.lower()

    server = headers.get("server", "").lower()
    if "cloudflare" in server or "cf-ray" in headers:
        detected.append("Cloudflare")
    if "vercel" in headers or "x-vercel-id" in headers:
        detected.append("Vercel")
    if "netlify" in headers or "x-nf-request-id" in headers:
        detected.append("Netlify")

    for tech_name, rules in TECH_SIGNATURES.items():
        for pattern in rules.get("html", []):
            if pattern.lower() in html_lower:
                detected.append(tech_name)
                break

    return sorted(list(set(detected)))


def extract_socials(soup: BeautifulSoup, base_url: str) -> Dict[str, Optional[str]]:
    socials: Dict[str, Optional[str]] = {
        "twitter": None,
        "linkedin": None,
        "github": None,
        "instagram": None,
        "youtube": None,
        "facebook": None,
    }

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        lower = href.lower()

        if ("twitter.com/" in lower or "x.com/" in lower) and not socials["twitter"]:
            if not any(x in lower for x in ["/share", "/intent", "/search"]):
                socials["twitter"] = href
        elif "linkedin.com/company" in lower and not socials["linkedin"]:
            socials["linkedin"] = href
        elif "github.com/" in lower and not socials["github"]:
            if not any(x in lower for x in ["/issues", "/pull", "/login", "/pricing"]):
                socials["github"] = href
        elif "instagram.com/" in lower and not socials["instagram"]:
            if not any(x in lower for x in ["/p/", "/explore"]):
                socials["instagram"] = href
        elif ("youtube.com/@" in lower or "youtube.com/channel" in lower or "youtube.com/c/" in lower) and not socials["youtube"]:
            socials["youtube"] = href
        elif "facebook.com/" in lower and not socials["facebook"]:
            if not any(x in lower for x in ["/sharer", "/dialog"]):
                socials["facebook"] = href

    return socials


async def extract_dominant_colors_from_image(client: httpx.AsyncClient, image_url: str) -> List[str]:
    """Downloads brand logo/icon and analyzes pixel RGB values to detect primary vibrant brand colors."""
    try:
        resp = await client.get(image_url, headers=BROWSER_HEADERS, timeout=4.0)
        if resp.status_code != 200 or not resp.content:
            return []

        img = Image.open(io.BytesIO(resp.content))
        img = img.convert("RGBA")
        img.thumbnail((64, 64))

        pixels = list(img.getdata())
        color_counts = Counter()

        for r, g, b, a in pixels:
            # Skip transparent pixels
            if a < 100:
                continue

            brightness = (r * 299 + g * 587 + b * 114) / 1000
            saturation = max(r, g, b) - min(r, g, b)

            # Ignore pure whites and dark grays
            if brightness > 240 or brightness < 20:
                continue

            # Quantize color to 16-step bins
            qr = min(255, (r // 16) * 16 + 8)
            qg = min(255, (g // 16) * 16 + 8)
            qb = min(255, (b // 16) * 16 + 8)

            # Heavy bonus weight for vivid, colorful pixels
            weight = 5 if saturation > 50 else (2 if saturation > 20 else 1)
            color_counts[(qr, qg, qb)] += weight

        if not color_counts:
            return []

        top_hex_colors = []
        for (r, g, b), _ in color_counts.most_common(6):
            hex_val = f"#{r:02X}{g:02X}{b:02X}"
            if hex_val not in top_hex_colors:
                top_hex_colors.append(hex_val)
            if len(top_hex_colors) >= 3:
                break

        return top_hex_colors
    except Exception:
        return []


async def determine_brand_colors(
    client: httpx.AsyncClient,
    domain: str,
    soup: BeautifulSoup,
    logo_candidates: List[Optional[str]],
) -> List[str]:
    """Extracts genuine brand colors, prioritizing pixel analysis over generic white/black meta tags."""
    extracted = []

    # Priority 1: Analyze pixels of the logo and high-res icons
    urls_to_try = [url for url in logo_candidates if url and not url.lower().endswith(".svg")]
    urls_to_try.append(f"https://www.google.com/s2/favicons?domain={domain}&sz=128")

    for url in urls_to_try:
        pixel_colors = await extract_dominant_colors_from_image(client, url)
        if pixel_colors:
            extracted.extend(pixel_colors)
            break

    # Priority 2: Check meta theme-color if not pure black/white
    theme_color = soup.find("meta", attrs={"name": "theme-color"})
    if theme_color and theme_color.get("content"):
        c = theme_color["content"].strip().upper()
        if re.match(r"^#(?:[0-9A-F]{3}){1,2}$", c) and c not in ["#FFFFFF", "#000000", "#FFF", "#000"]:
            if c not in extracted:
                extracted.append(c)

    # Return top vibrant colors, or default monochrome
    return extracted[:4] if extracted else ["#0A0A0A"]


def extract_logos(soup: BeautifulSoup, base_url: str, domain: str) -> Dict[str, Optional[str]]:
    favicon = None
    icon = None
    og_image = None

    og_tag = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name": "og:image"})
    if og_tag and og_tag.get("content"):
        og_image = urljoin(base_url, og_tag["content"].strip())

    apple_icon = soup.find("link", rel=lambda r: r and "apple-touch-icon" in r.lower())
    if apple_icon and apple_icon.get("href"):
        icon = urljoin(base_url, apple_icon["href"].strip())

    fav_tag = soup.find("link", rel=lambda r: r and any(x in r.lower() for x in ["icon", "shortcut icon"]))
    if fav_tag and fav_tag.get("href"):
        favicon = urljoin(base_url, fav_tag["href"].strip())

    default_favicon = f"https://www.google.com/s2/favicons?domain={domain}&sz=128"
    high_res_logo = icon or og_image or default_favicon

    return {
        "logo_url": high_res_logo,
        "favicon_url": favicon or default_favicon,
        "og_image_url": og_image,
        "apple_touch_icon": icon,
    }


def extract_contacts(soup: BeautifulSoup, html: str, base_url: str) -> Dict[str, Optional[object]]:
    """Intelligently detects customer support emails, phones, help centers, and contact pages."""
    raw_emails = set()
    raw_phones = set()

    # 1. Search mailto: links
    for mailto in soup.find_all("a", href=re.compile(r"^mailto:", re.I)):
        raw = mailto["href"].replace("mailto:", "").split("?")[0].strip().lower()
        if re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", raw):
            raw_emails.add(raw)

    # 2. Search raw text email patterns
    text_matches = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b", html)
    for email in text_matches:
        lower = email.lower()
        if not any(lower.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif", ".js", ".css"]):
            raw_emails.add(lower)

    # 3. Search tel: links
    for tel in soup.find_all("a", href=re.compile(r"^tel:", re.I)):
        raw = tel["href"].replace("tel:", "").strip()
        if len(raw) >= 7:
            raw_phones.add(raw)

    # 4. Search Contact Us & Help Center Links
    contact_page = None
    help_center = None

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text().strip().lower()
        href_lower = href.lower()

        if not contact_page and any(k in href_lower or k in text for k in ["contact-us", "/contact", "contact"]):
            contact_page = urljoin(base_url, href)
        elif not help_center and any(k in href_lower or k in text for k in ["help-center", "support", "/help", "knowledgebase"]):
            help_center = urljoin(base_url, href)

    # 5. Prioritize Customer Care / Support Email
    support_email = None
    general_emails = []

    for email in sorted(list(raw_emails)):
        if not support_email and any(email.startswith(prefix) for prefix in ["support@", "help@", "care@", "customercare@", "service@", "info@", "contact@"]):
            support_email = email
        else:
            general_emails.append(email)

    if not support_email and general_emails:
        support_email = general_emails.pop(0)

    # 6. Prioritize Customer Support Phone
    support_phone = sorted(list(raw_phones))[0] if raw_phones else None

    return {
        "customer_support_email": support_email,
        "customer_support_phone": support_phone,
        "contact_page_url": contact_page,
        "help_center_url": help_center,
        "all_emails_found": sorted(list(raw_emails)),
        "all_phones_found": sorted(list(raw_phones)),
    }


@app.get("/")
def root():
    return {
        "service": "BrandIntel API",
        "status": "operational",
        "documentation": "/docs",
        "version": "1.2.0",
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/v1/enrich")
async def enrich_company(
    domain: str = Query(..., description="Target company domain (e.g. stripe.com, shopify.com)"),
    x_rapidapi_proxy_secret: Optional[str] = Header(None),
):
    """Full company intelligence: Logos, brand colors, detected tech stack, socials, and support contacts."""
    verify_rapidapi_request(x_rapidapi_proxy_secret)
    cleaned_domain = clean_domain(domain)
    target_url = f"https://{cleaned_domain}"

    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True, verify=False) as client:
            response = await client.get(target_url, headers=BROWSER_HEADERS)
            html = response.text
            soup = BeautifulSoup(html, "html.parser")
            final_url = str(response.url)

            # Title & Description
            title = None
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            elif soup.find("meta", property="og:title"):
                title = soup.find("meta", property="og:title").get("content", "").strip()

            description = None
            desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", property="og:description")
            if desc_tag and desc_tag.get("content"):
                description = desc_tag["content"].strip()

            logos = extract_logos(soup, final_url, cleaned_domain)
            socials = extract_socials(soup, final_url)
            tech_stack = detect_tech_stack(html, response.headers)
            contacts = extract_contacts(soup, html, final_url)

            # Advanced pixel brand color extraction
            brand_colors = await determine_brand_colors(
                client=client,
                domain=cleaned_domain,
                soup=soup,
                logo_candidates=[logos["apple_touch_icon"], logos["logo_url"], logos["favicon_url"]],
            )

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Unable to reach domain '{cleaned_domain}': {str(exc)}",
        )

    return {
        "domain": cleaned_domain,
        "name": title.split(" - ")[0].split(" | ")[0] if title else cleaned_domain.capitalize(),
        "title": title,
        "description": description,
        "brand_assets": {
            "logo": logos["logo_url"],
            "favicon": logos["favicon_url"],
            "og_image": logos["og_image_url"],
            "colors": brand_colors,
        },
        "social_profiles": socials,
        "technology_stack": tech_stack,
        "contacts": contacts,
        "meta": {
            "status_code": response.status_code,
            "final_url": final_url,
            "secured_ssl": final_url.startswith("https://"),
        },
    }


@app.get("/v1/logo")
async def get_company_logo(
    domain: str = Query(..., description="Target company domain (e.g. stripe.com)"),
    x_rapidapi_proxy_secret: Optional[str] = Header(None),
):
    enrichment = await enrich_company(domain, x_rapidapi_proxy_secret)
    return {
        "domain": enrichment["domain"],
        "name": enrichment["name"],
        "brand_assets": enrichment["brand_assets"],
    }


@app.get("/v1/tech")
async def get_company_tech(
    domain: str = Query(..., description="Target company domain (e.g. stripe.com)"),
    x_rapidapi_proxy_secret: Optional[str] = Header(None),
):
    enrichment = await enrich_company(domain, x_rapidapi_proxy_secret)
    return {
        "domain": enrichment["domain"],
        "technology_stack": enrichment["technology_stack"],
        "social_profiles": enrichment["social_profiles"],
    }
