from wpipe import step
from .schemas.rss_schema import RSSInput
import requests
import xml.etree.ElementTree as ET
from loguru import logger

@step(name="RSSParserStep")
class RSSParserStep:
    """Basic example of a wpipe step that parses an RSS feed."""
    
    def __init__(self, config=None):
        self.config = config

    def __call__(self, data: RSSInput):
        logger.info(f"Fetching RSS feed from: {data.url}")
        response = requests.get(str(data.url))
        root = ET.fromstring(response.content)
        
        items = []
        for item in root.findall('.//item')[:data.limit]:
            items.append({
                "title": item.find('title').text,
                "link": item.find('link').text
            })
        
        return {"items": items}
