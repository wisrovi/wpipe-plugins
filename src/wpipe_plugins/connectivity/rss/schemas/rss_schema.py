from pydantic import BaseModel, HttpUrl

class RSSInput(BaseModel):
    url: HttpUrl
    limit: int = 5
