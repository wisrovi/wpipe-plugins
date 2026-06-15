# RSS Parser Step (Community Supported)

> **Community Support**: This state is part of the `wpipe-plugins` community repository. It is built and supported by the community. For official author-maintained states, visit [wpipe-steps](https://github.com/wisrovi/wpipe-steps).

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
