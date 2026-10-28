# Market data

Raw trades and book-ticker data are not distributed in Git. The baseline configuration `configs/robust_study.yaml` defines the market and date ranges. Its `download` stage retrieves the required sessions through `cjr.download`.

Default raw path: `data/raw/dot/`. Default causal cache: `data/cache/`. Use `CJR_TEST_RAW` and `CJR_VALIDATION_CACHE` to override the final pipeline paths. The baseline downloader reads the directories in its YAML configuration.

Download availability and provider access conditions apply. Generated figures require local experiment scores; they do not require the raw data once those scores exist.
