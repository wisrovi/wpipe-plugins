from wpipe_plugins.connectivity.rss.states.rss_state import RSSParserStep
from wpipe import Pipeline

def run_example():
    step = RSSParserStep()
    pipe = Pipeline(pipeline_name="rss_test")
    pipe.set_steps([step])
    
    result = pipe.run({"url": "https://news.google.com/rss", "limit": 2})
    print(result)

if __name__ == "__main__":
    run_example()
