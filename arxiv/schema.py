from __future__ import annotations

import logging
import time
import feedparser
import os
import math
import re
import warnings
from urllib.parse import urlparse
from urllib.request import urlretrieve
from calendar import timegm
from enum import Enum
from datetime import datetime, timezone

from typing import TYPE_CHECKING, Iterator

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from typing_extensions import TypedDict
    import feedparser

    class FeedParserDict(TypedDict, total=False):
        id: str
        title: str
        summary: str
        authors: list[dict[str, str]]
        links: list[dict[str, str]]
        tags: list[dict[str, str]]
        updated_parsed: time.struct_time
        published_parsed: time.struct_time
        arxiv_comment: str
        arxiv_journal_ref: str
        arxiv_doi: str
        arxiv_primary_category: dict[str, str]


_DEFAULT_TIME = datetime.min


class Result:
    """
    An entry in an arXiv query results feed.

    See [the arXiv API User's Manual: Details of Atom Results
    Returned](https://arxiv.org/help/api/user-manual#_details_of_atom_results_returned).
    """

    entry_id: str
    """A url of the form `https://arxiv.org/abs/{id}`."""
    updated: datetime
    """When the result was last updated."""
    published: datetime
    """When the result was originally published."""
    title: str
    """The title of the result."""
    authors: list[Result.Author]
    """The result's authors."""
    summary: str
    """The result abstract."""
    comment: str | None
    """The authors' comment if present."""
    journal_ref: str | None
    """A journal reference if present."""
    doi: str | None
    """A URL for the resolved DOI to an external resource if present."""
    primary_category: str
    """
    The result's primary arXiv category. See [arXiv: Category
    Taxonomy](https://arxiv.org/category_taxonomy).
    """
    categories: list[str]
    """
    All of the result's categories. See [arXiv: Category
    Taxonomy](https://arxiv.org/category_taxonomy).
    """
    links: list[Result.Link]
    """Up to three URLs associated with this result."""
    pdf_url: str | None
    """The URL of a PDF version of this result if present among links."""
    _raw: feedparser.FeedParserDict
    """
    The raw feedparser result object if this Result was constructed with
    Result._from_feed_entry.
    """

    def __init__(
        self,
        entry_id: str,
        updated: datetime = _DEFAULT_TIME,
        published: datetime = _DEFAULT_TIME,
        title: str = "",
        authors: list[Result.Author] | None = None,
        summary: str = "",
        comment: str = "",
        journal_ref: str = "",
        doi: str = "",
        primary_category: str = "",
        categories: list[str] | None = None,
        links: list[Result.Link] | None = None,
        _raw: feedparser.FeedParserDict | None = None,
    ):
        """
        Constructs an arXiv search result item.

        In most cases, prefer using `Result._from_feed_entry` to parsing and
        constructing `Result`s yourself.
        """
        self.entry_id = entry_id
        self.updated = updated
        self.published = published
        self.title = title
        self.authors = authors or []
        self.summary = summary
        self.comment = comment
        self.journal_ref = journal_ref
        self.doi = doi
        self.primary_category = primary_category
        self.categories = categories or []
        self.links = links or []
        # Calculated members
        self.pdf_url = Result._get_pdf_url(self.links)
        # Debugging
        self._raw = _raw

    @classmethod
    def _from_feed_entry(cls, entry: feedparser.FeedParserDict) -> Result:
        """
        Converts a feedparser entry for an arXiv search result feed into a
        Result object.
        """
        if not hasattr(entry, "id"):
            raise Result.MissingFieldError("id")
        # Title attribute may be absent for certain titles. Defaulting to "0" as
        # it's the only title observed to cause this bug.
        # https://github.com/lukasschwab/arxiv.py/issues/71
        # title = entry.title if hasattr(entry, "title") else "0"
        title = "0"
        if hasattr(entry, "title"):
            title = entry.title
        else:
            logger.warning("Result %s is missing title attribute; defaulting to '0'", entry.id)
        return Result(
            entry_id=entry.id,
            updated=Result._to_datetime(entry.updated_parsed),
            published=Result._to_datetime(entry.published_parsed),
            title=re.sub(r"\s+", " ", title),
            authors=[Result.Author._from_feed_author(a) for a in entry.authors],
            summary=entry.summary,
            comment=entry.get("arxiv_comment"),
            journal_ref=entry.get("arxiv_journal_ref"),
            doi=entry.get("arxiv_doi"),
            primary_category=entry.arxiv_primary_category.get("term"),
            categories=[tag.get("term") for tag in entry.tags],
            links=[Result.Link._from_feed_link(link) for link in entry.links],
            _raw=entry,
        )

    def __str__(self) -> str:
        return self.entry_id

    def __repr__(self) -> str:
        return (
            "{}(entry_id={}, updated={}, published={}, title={}, authors={}, "
            "summary={}, comment={}, journal_ref={}, doi={}, "
            "primary_category={}, categories={}, links={})"
        ).format(
            _classname(self),
            repr(self.entry_id),
            repr(self.updated),
            repr(self.published),
            repr(self.title),
            repr(self.authors),
            repr(self.summary),
            repr(self.comment),
            repr(self.journal_ref),
            repr(self.doi),
            repr(self.primary_category),
            repr(self.categories),
            repr(self.links),
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Result):
            return self.entry_id == other.entry_id
        return False

    def get_short_id(self) -> str:
        """
        Returns the short ID for this result.

        + If the result URL is `"https://arxiv.org/abs/2107.05580v1"`,
        `result.get_short_id()` returns `2107.05580v1`.

        + If the result URL is `"https://arxiv.org/abs/quant-ph/0201082v1"`,
        `result.get_short_id()` returns `"quant-ph/0201082v1"` (the pre-March
        2007 arXiv identifier format).

        For an explanation of the difference between arXiv's legacy and current
        identifiers, see [Understanding the arXiv
        identifier](https://arxiv.org/help/arxiv_identifier).
        """
        return self.entry_id.split("arxiv.org/abs/")[-1]

    def _get_default_filename(self, extension: str = "pdf") -> str:
        """
        A default `to_filename` function for the extension given.
        """
        nonempty_title = self.title if self.title else "UNTITLED"
        return ".".join(
            [
                self.get_short_id().replace("/", "_"),
                re.sub(r"[^\w]", "_", nonempty_title),
                extension,
            ]
        )

    def download_pdf(
        self,
        dirpath: str = "./",
        filename: str = "",
        download_domain: str = "export.arxiv.org",
    ) -> str:
        """
        Downloads the PDF for this result to the specified directory.

        The filename is generated by calling `to_filename(self)`.

        **Deprecated:** future versions of this client library will not provide
        download helpers (out of scope). Use `result.pdf_url` directly.
        """
        if not filename:
            filename = self._get_default_filename()
        path = os.path.join(dirpath, filename)
        if self.pdf_url is None:
            raise ValueError("No PDF URL available for this result")
        pdf_url = Result._substitute_domain(self.pdf_url, download_domain)
        written_path, _ = urlretrieve(pdf_url, path)
        return written_path

    def download_source(
        self,
        dirpath: str = "./",
        filename: str = "",
        download_domain: str = "export.arxiv.org",
    ) -> str:
        """
        Downloads the source tarfile for this result to the specified
        directory.

        The filename is generated by calling `to_filename(self)`.

        **Deprecated:** future versions of this client library will not provide
        download helpers (out of scope). Use `result.source_url` directly.
        """
        if not filename:
            filename = self._get_default_filename("tar.gz")
        path = os.path.join(dirpath, filename)
        source_url_str = self.source_url()
        if source_url_str is None:
            raise ValueError("No source URL available for this result")
        source_url = Result._substitute_domain(source_url_str, download_domain)
        written_path, _ = urlretrieve(source_url, path)
        return written_path

    def source_url(self) -> str | None:
        """
        Derives a URL for the source tarfile for this result.
        """
        if self.pdf_url is None:
            return None
        return self.pdf_url.replace("/pdf/", "/src/")

    @staticmethod
    def _get_pdf_url(links: list[Result.Link]) -> str | None:
        """
        Finds the PDF link among a result's links and returns its URL.

        Should only be called once for a given `Result`, in its constructor.
        After construction, the URL should be available in `Result.pdf_url`.
        """
        pdf_urls = [link.href for link in links if link.title == "pdf"]
        if len(pdf_urls) == 0:
            return None
        elif len(pdf_urls) > 1:
            logger.warning("Result has multiple PDF links; using %s", pdf_urls[0])
        return pdf_urls[0]

    @staticmethod
    def _to_datetime(ts: time.struct_time) -> datetime:
        """
        Converts a UTC time.struct_time into a time-zone-aware datetime.

        This will be replaced with feedparser functionality [when it becomes
        available](https://github.com/kurtmckee/feedparser/issues/212).
        """
        return datetime.fromtimestamp(timegm(ts), tz=timezone.utc)

    @staticmethod
    def _substitute_domain(url: str, domain: str) -> str:
        """
        Replaces the domain of the given URL with the specified domain.

        This is useful for testing purposes.
        """
        parsed_url = urlparse(url)
        return parsed_url._replace(netloc=domain).geturl()

    class Author:
        """
        A light inner class for representing a result's authors.
        """

        name: str
        """The author's name."""

        def __init__(self, name: str):
            """
            Constructs an `Author` with the specified name.

            In most cases, prefer using `Author._from_feed_author` to parsing
            and constructing `Author`s yourself.
            """
            self.name = name

        @classmethod
        def _from_feed_author(cls, feed_author: feedparser.FeedParserDict) -> Result.Author:
            """
            Constructs an `Author` with the name specified in an author object
            from a feed entry.

            See usage in `Result._from_feed_entry`.
            """
            return Result.Author(feed_author.name)

        def __str__(self) -> str:
            return self.name

        def __repr__(self) -> str:
            return "{}({})".format(_classname(self), repr(self.name))

        def __eq__(self, other: object) -> bool:
            if isinstance(other, Result.Author):
                return self.name == other.name
            return False

    class Link:
        """
        A light inner class for representing a result's links.
        """

        href: str
        """The link's `href` attribute."""
        title: str | None
        """The link's title."""
        rel: str
        """The link's relationship to the `Result`."""
        content_type: str | None
        """The link's HTTP content type."""

        def __init__(
            self,
            href: str,
            title: str | None = None,
            rel: str = "",
            content_type: str | None = None,
        ):
            """
            Constructs a `Link` with the specified link metadata.

            In most cases, prefer using `Link._from_feed_link` to parsing and
            constructing `Link`s yourself.
            """
            self.href = href
            self.title = title
            self.rel = rel
            self.content_type = content_type

        @classmethod
        def _from_feed_link(cls, feed_link: feedparser.FeedParserDict) -> Result.Link:
            """
            Constructs a `Link` with link metadata specified in a link object
            from a feed entry.

            See usage in `Result._from_feed_entry`.
            """
            return Result.Link(
                href=feed_link.href,
                title=feed_link.get("title"),
                rel=feed_link.get("rel") or "",
                content_type=feed_link.get("content_type"),
            )

        def __str__(self) -> str:
            return self.href

        def __repr__(self) -> str:
            return "{}({}, title={}, rel={}, content_type={})".format(
                _classname(self),
                repr(self.href),
                repr(self.title),
                repr(self.rel),
                repr(self.content_type),
            )

        def __eq__(self, other: object) -> bool:
            if isinstance(other, Result.Link):
                return self.href == other.href
            return False

    class MissingFieldError(Exception):
        """
        An error indicating an entry is unparseable because it lacks required
        fields.
        """

        missing_field: str
        """The required field missing from the would-be entry."""
        message: str
        """Message describing what caused this error."""

        def __init__(self, missing_field: str):
            self.missing_field = missing_field
            self.message = "Entry from arXiv missing required info"

        def __repr__(self) -> str:
            return "{}({})".format(_classname(self), repr(self.missing_field))


class SortCriterion(Enum):
    """
    A SortCriterion identifies a property by which search results can be
    sorted.

    See [the arXiv API User's Manual: sort order for return
    results](https://arxiv.org/help/api/user-manual#sort).
    """

    Relevance = "relevance"
    LastUpdatedDate = "lastUpdatedDate"
    SubmittedDate = "submittedDate"


class SortOrder(Enum):
    """
    A SortOrder indicates order in which search results are sorted according
    to the specified arxiv.SortCriterion.

    See [the arXiv API User's Manual: sort order for return
    results](https://arxiv.org/help/api/user-manual#sort).
    """

    Ascending = "ascending"
    Descending = "descending"


class Search:
    """
    A specification for a search of arXiv's database.

    To run a search, use `Search.run` to use a default client or `Client.run`
    with a specific client.
    """

    query: str
    """
    A query string.

    This should be unencoded. Use `au:del_maestro AND ti:checkerboard`, not
    `au:del_maestro+AND+ti:checkerboard`.

    See [the arXiv API User's Manual: Details of Query
    Construction](https://arxiv.org/help/api/user-manual#query_details).
    """
    id_list: list[str]
    """
    A list of arXiv article IDs to which to limit the search.

    See [the arXiv API User's
    Manual](https://arxiv.org/help/api/user-manual#search_query_and_id_list)
    for documentation of the interaction between `query` and `id_list`.
    """
    max_results: int | None
    """
    The maximum number of results to be returned in an execution of this
    search. To fetch every result available, set `max_results=None`.

    The API's limit is 300,000 results per query.
    """
    sort_by: SortCriterion
    """The sort criterion for results."""
    sort_order: SortOrder
    """The sort order for results."""

    def __init__(
        self,
        query: str = "",
        id_list: list[str] | None = None,
        max_results: int | None = None,
        sort_by: SortCriterion = SortCriterion.Relevance,
        sort_order: SortOrder = SortOrder.Descending,
    ):
        """
        Constructs an arXiv API search with the specified criteria.
        """
        self.query = query
        self.id_list = id_list or []
        # Handle deprecated v1 default behavior.
        self.max_results = None if max_results == math.inf else max_results
        self.sort_by = sort_by
        self.sort_order = sort_order

    def __str__(self) -> str:
        if self.query and self.id_list:
            return f"Search(query='{self.query}', id_list={len(self.id_list)} items)"
        elif self.query:
            return f"Search(query='{self.query}')"
        elif self.id_list:
            return f"Search(id_list={len(self.id_list)} items)"
        else:
            return "Search(empty)"

    def __repr__(self) -> str:
        return ("{}(query={}, id_list={}, max_results={}, sort_by={}, sort_order={})").format(
            _classname(self),
            repr(self.query),
            repr(self.id_list),
            repr(self.max_results),
            repr(self.sort_by),
            repr(self.sort_order),
        )

    def _url_args(self) -> dict[str, str]:
        """
        Returns a dict of search parameters that should be included in an API
        request for this search.
        """
        return {
            "search_query": self.query,
            "id_list": ",".join(self.id_list),
            "sortBy": self.sort_by.value,
            "sortOrder": self.sort_order.value,
        }

    def results(self, offset: int = 0) -> Iterator[Result]:
        """
        Executes the specified search using a default arXiv API client. For info
        on default behavior, see `Client.__init__` and `Client.results`.

        **Deprecated** after 2.0.0; use `Client.results`.
        """
        from . import Client

        warnings.warn(
            "The 'Search.results' method is deprecated, use 'Client.results' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return Client().results(self, offset=offset)


class ArxivError(Exception):
    """This package's base Exception class."""

    url: str
    """The feed URL that could not be fetched."""
    retry: int
    """
    The request try number which encountered this error; 0 for the initial try,
    1 for the first retry, and so on.
    """
    message: str
    """Message describing what caused this error."""

    def __init__(self, url: str, retry: int, message: str):
        """
        Constructs an `ArxivError` encountered while fetching the specified URL.
        """
        self.url = url
        self.retry = retry
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return "{} ({})".format(self.message, self.url)


class UnexpectedEmptyPageError(ArxivError):
    """
    An error raised when a page of results that should be non-empty is empty.

    This should never happen in theory, but happens sporadically due to
    brittleness in the underlying arXiv API; usually resolved by retries.

    See `Client.results` for usage.
    """

    raw_feed: feedparser.FeedParserDict
    """
    The raw output of `feedparser.parse`. Sometimes this contains useful
    diagnostic information, e.g. in 'bozo_exception'.
    """

    def __init__(self, url: str, retry: int, raw_feed: feedparser.FeedParserDict):
        """
        Constructs an `UnexpectedEmptyPageError` encountered for the specified
        API URL after `retry` tries.
        """
        self.url = url
        self.raw_feed = raw_feed
        super().__init__(url, retry, "Page of results was unexpectedly empty")

    def __repr__(self) -> str:
        return "{}({}, {}, {})".format(
            _classname(self), repr(self.url), repr(self.retry), repr(self.raw_feed)
        )


class HTTPError(ArxivError):
    """
    A non-200 status encountered while fetching a page of results.

    See `Client.results` for usage.
    """

    status: int
    """The HTTP status reported by feedparser."""

    def __init__(self, url: str, retry: int, status: int):
        """
        Constructs an `HTTPError` for the specified status code, encountered for
        the specified API URL after `retry` tries.
        """
        self.url = url
        self.status = status
        super().__init__(
            url,
            retry,
            "Page request resulted in HTTP {}".format(self.status),
        )

    def __repr__(self) -> str:
        return "{}({}, {}, {})".format(
            _classname(self), repr(self.url), repr(self.retry), repr(self.status)
        )


def _classname(o: object) -> str:
    """A helper function for use in __repr__ methods: arxiv.Result.Link."""
    return "arxiv.{}".format(o.__class__.__qualname__)
