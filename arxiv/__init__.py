""".. include:: ../README.md"""

from __future__ import annotations

import logging
import time
import itertools
import feedparser
import requests

from urllib.parse import urlencode
from datetime import datetime, timedelta
from typing import Generator, Iterator

# These symbols were moved to `schema`, but are re-exported here to
# preserve backward compatibility with existing imports.
from .schema import (
    Result,
    Search,
    SortCriterion,
    SortOrder,
    ArxivError,
    UnexpectedEmptyPageError,
    HTTPError,
    _classname,
)

__all__ = [
    "Result",
    "Search",
    "SortCriterion",
    "SortOrder",
    "ArxivError",
    "UnexpectedEmptyPageError",
    "HTTPError",
    "Client",
]

logger = logging.getLogger(__name__)


class Client:
    """
    Specifies a strategy for fetching results from arXiv's API.

    This class obscures pagination and retry logic, and exposes
    `Client.results`.
    """

    query_url_format = "https://export.arxiv.org/api/query?{}"
    """
    The arXiv query API endpoint format.
    """
    page_size: int
    """
    Maximum number of results fetched in a single API request. Smaller pages can
    be retrieved faster, but may require more round-trips.

    The API's limit is 2000 results per page.
    """
    delay_seconds: float
    """
    Number of seconds to wait between API requests.

    [arXiv's Terms of Use](https://arxiv.org/help/api/tou) ask that you "make no
    more than one request every three seconds."
    """
    num_retries: int
    """
    Number of times to retry a failing API request before raising an Exception.
    """

    _last_request_dt: datetime | None
    _session: requests.Session

    def __init__(self, page_size: int = 100, delay_seconds: float = 3.0, num_retries: int = 3):
        """
        Constructs an arXiv API client with the specified options.

        Note: the default parameters should provide a robust request strategy
        for most use cases. Extreme page sizes, delays, or retries risk
        violating the arXiv [API Terms of Use](https://arxiv.org/help/api/tou),
        brittle behavior, and inconsistent results.
        """
        self.page_size = page_size
        self.delay_seconds = delay_seconds
        self.num_retries = num_retries
        self._last_request_dt = None
        self._session = requests.Session()

    def __str__(self) -> str:
        return f"Client(page_size={self.page_size}, delay={self.delay_seconds}s, retries={self.num_retries})"

    def __repr__(self) -> str:
        return "{}(page_size={}, delay_seconds={}, num_retries={})".format(
            _classname(self),
            repr(self.page_size),
            repr(self.delay_seconds),
            repr(self.num_retries),
        )

    def results(self, search: Search, offset: int = 0) -> Iterator[Result]:
        """
        Uses this client configuration to fetch one page of the search results
        at a time, yielding the parsed `Result`s, until `max_results` results
        have been yielded or there are no more search results.

        If all tries fail, raises an `UnexpectedEmptyPageError` or `HTTPError`.

        Setting a nonzero `offset` discards leading records in the result set.
        When `offset` is greater than or equal to `search.max_results`, the full
        result set is discarded.

        For more on using generators, see
        [Generators](https://wiki.python.org/moin/Generators).
        """
        limit = search.max_results - offset if search.max_results else None
        if limit and limit < 0:
            return iter(())
        return itertools.islice(self._results(search, offset), limit)

    def _results(self, search: Search, offset: int = 0) -> Generator[Result, None, None]:
        page_url = self._format_url(search, offset, self.page_size)
        feed = self._parse_feed(page_url, first_page=True)
        if not feed.entries:
            logger.info("Got empty first page; stopping generation")
            return
        total_results = int(feed.feed.opensearch_totalresults)
        logger.info(
            "Got first page: %d of %d total results",
            len(feed.entries),
            total_results,
        )

        while feed.entries:
            for entry in feed.entries:
                try:
                    yield Result._from_feed_entry(entry)
                except Result.MissingFieldError as e:
                    logger.warning("Skipping partial result: %s", e)
            offset += len(feed.entries)
            if offset >= total_results:
                break
            page_url = self._format_url(search, offset, self.page_size)
            feed = self._parse_feed(page_url, first_page=False)

    def _format_url(self, search: Search, start: int, page_size: int) -> str:
        """
        Construct a request API for search that returns up to `page_size`
        results starting with the result at index `start`.
        """
        url_args = search._url_args()
        url_args.update(
            {
                "start": str(start),
                "max_results": str(page_size),
            }
        )
        return self.query_url_format.format(urlencode(url_args))

    def _parse_feed(
        self, url: str, first_page: bool = True, _try_index: int = 0
    ) -> feedparser.FeedParserDict:
        """
        Fetches the specified URL and parses it with feedparser.

        If a request fails or is unexpectedly empty, retries the request up to
        `self.num_retries` times.
        """
        try:
            return self.__try_parse_feed(url, first_page=first_page, try_index=_try_index)
        except (
            HTTPError,
            UnexpectedEmptyPageError,
            requests.exceptions.ConnectionError,
        ) as err:
            if _try_index < self.num_retries:
                logger.debug("Got error (try %d): %s", _try_index, err)
                return self._parse_feed(url, first_page=first_page, _try_index=_try_index + 1)
            logger.debug("Giving up (try %d): %s", _try_index, err)
            raise err

    def __try_parse_feed(
        self,
        url: str,
        first_page: bool,
        try_index: int,
    ) -> feedparser.FeedParserDict:
        """
        Recursive helper for _parse_feed. Enforces `self.delay_seconds`: if that
        number of seconds has not passed since `_parse_feed` was last called,
        sleeps until delay_seconds seconds have passed.
        """
        # If this call would violate the rate limit, sleep until it doesn't.
        if self._last_request_dt is not None:
            required = timedelta(seconds=self.delay_seconds)
            since_last_request = datetime.now() - self._last_request_dt
            if since_last_request < required:
                to_sleep = (required - since_last_request).total_seconds()
                logger.info("Sleeping: %f seconds", to_sleep)
                time.sleep(to_sleep)

        logger.info("Requesting page (first: %r, try: %d): %s", first_page, try_index, url)

        resp = self._session.get(url, headers={"user-agent": "arxiv.py/2.3.2"})
        self._last_request_dt = datetime.now()
        if resp.status_code != requests.codes.OK:
            raise HTTPError(url, try_index, resp.status_code)

        feed = feedparser.parse(resp.content)
        if len(feed.entries) == 0 and not first_page:
            raise UnexpectedEmptyPageError(url, try_index, feed)

        if feed.bozo:
            logger.warning(
                "Bozo feed; consider handling: %s",
                feed.bozo_exception if "bozo_exception" in feed else None,
            )

        return feed
