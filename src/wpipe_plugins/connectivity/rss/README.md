# RSS Parser Step

This state allows for easy parsing of RSS feeds and integration into a `wpipe` pipeline.

## Usage

The state requires a valid URL and optionally a limit for the number of items to retrieve.

```python
from wpipe_plugins.connectivity.rss import RSSParserStep
from wpipe import Pipeline

step = RSSParserStep()
pipe = Pipeline(pipeline_name="rss_test")
pipe.set_steps([step])

result = pipe.run({"url": "https://news.google.com/rss", "limit": 5})
```

## Requirements
- `requests`
- `pydantic`
