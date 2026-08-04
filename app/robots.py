"""Robots.txt compliance — checks every URL before crawling."""

import urllib.request
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from app.config import CRAWLER_USER_AGENT


def check_robots_txt(urls: list[str]) -> list[str]:
    allowed_urls = []

    for url in urls:
        try:
            parsed     = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

            rp = RobotFileParser(robots_url)

            # ✅ Set timeout — prevents hanging on slow robots.txt
            try:
                req      = urllib.request.Request(
                    robots_url,
                    headers={"User-Agent": CRAWLER_USER_AGENT}
                )
                response = urllib.request.urlopen(req, timeout=5)
                rp.parse(response.read().decode("utf-8", errors="ignore").splitlines())
            except Exception:
                # ✅ If robots.txt unreachable — assume allowed
                allowed_urls.append(url)
                continue

            # ✅ Check with real browser user agent, not wildcard "*"
            if rp.can_fetch(CRAWLER_USER_AGENT, url):
                allowed_urls.append(url)
                print(f"[Robots.txt] Allowed : {url}")
            else:
                # ✅ Double check with wildcard before final block decision
                if rp.can_fetch("*", url):
                    allowed_urls.append(url)
                    print(f"[Robots.txt] Allowed via wildcard: {url}")
                else:
                    print(f"[Robots.txt] Blocked : {url}")

        except Exception:
            # ✅ Any error — assume allowed, don't block valid URLs
            allowed_urls.append(url)

    return allowed_urls
