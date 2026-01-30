import unittest
import asyncio
from unittest.mock import MagicMock, call, patch, AsyncMock
import arxiv
from arxiv.async_client import AsyncClient
from datetime import datetime, timedelta
from pytest import approx
import httpx

def empty_response(code: int) -> httpx.Response:
    return httpx.Response(status_code=code, content=b"")

class TestAsyncClient(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_format_id(self):
        with self.assertRaises(arxiv.HTTPError):
            async for _ in AsyncClient(num_retries=0).results(arxiv.Search(id_list=["abc"])):
                pass

    async def test_invalid_id(self):
        results = []
        async for r in AsyncClient().results(arxiv.Search(id_list=["0000.0000"])):
            results.append(r)
        self.assertEqual(len(results), 0)

    async def test_nonexistent_id_in_list(self):
        client = AsyncClient()
        # Assert thrown error is handled and hidden by generator.
        results = []
        async for r in client.results(arxiv.Search(id_list=["0808.05394"])):
            results.append(r)
        self.assertEqual(len(results), 0)
        # Generator should still yield valid entries.
        results = []
        async for r in client.results(arxiv.Search(id_list=["0808.05394", "1707.08567"])):
            results.append(r)
        self.assertEqual(len(results), 1)

    async def test_max_results(self):
        client = AsyncClient(page_size=10)
        search = arxiv.Search(query="testing", max_results=2)
        results = []
        async for r in client.results(search):
            results.append(r)
        self.assertEqual(len(results), 2)

    async def test_query_page_count(self):
        client = AsyncClient(page_size=10)
        client._parse_feed = MagicMock(wraps=client._parse_feed)
        
        # We need to await the generator
        results = []
        async for r in client.results(arxiv.Search(query="testing", max_results=55)):
            results.append(r)

        # NOTE: don't directly assert on call count; allow for retries.
        unique_urls = set()
        for parse_call in client._parse_feed.call_args_list:
            args, _kwargs = parse_call
            unique_urls.add(args[0])

        self.assertEqual(len(results), 55)
        self.assertSetEqual(
            unique_urls,
            {
                "https://export.arxiv.org/api/query?search_query=testing&id_list=&sortBy=relevance&sortOrder=descending&start=0&max_results=10",
                "https://export.arxiv.org/api/query?search_query=testing&id_list=&sortBy=relevance&sortOrder=descending&start=10&max_results=10",
                "https://export.arxiv.org/api/query?search_query=testing&id_list=&sortBy=relevance&sortOrder=descending&start=20&max_results=10",
                "https://export.arxiv.org/api/query?search_query=testing&id_list=&sortBy=relevance&sortOrder=descending&start=30&max_results=10",
                "https://export.arxiv.org/api/query?search_query=testing&id_list=&sortBy=relevance&sortOrder=descending&start=40&max_results=10",
                "https://export.arxiv.org/api/query?search_query=testing&id_list=&sortBy=relevance&sortOrder=descending&start=50&max_results=10",
            },
        )

    async def test_offset(self):
        max_results = 10
        search = arxiv.Search(query="testing", max_results=max_results)
        client = AsyncClient(page_size=10)

        default = []
        async for r in client.results(search):
            default.append(r)
            
        no_offset = []
        async for r in client.results(search):
            no_offset.append(r)
            
        self.assertListEqual(default, no_offset)

        offset = max_results // 2
        half_offset = []
        async for r in client.results(search, offset=offset):
            half_offset.append(r)
            
        self.assertListEqual(default[offset:], half_offset)

        offset_above_max_results = []
        async for r in client.results(search, offset=max_results):
            offset_above_max_results.append(r)
            
        self.assertListEqual(offset_above_max_results, [])

    async def test_search_results_offset(self):
        # NOTE: page size is irrelevant here.
        client = AsyncClient(page_size=15)
        search = arxiv.Search(query="testing", max_results=10)
        
        all_results = []
        async for r in client.results(search, offset=0):
            all_results.append(r)
        self.assertEqual(len(all_results), 10)

        for offset in [0, 5, 9, 10, 11]:
            client_results = []
            async for r in client.results(search, offset=offset):
                client_results.append(r)
            self.assertEqual(len(client_results), max(0, search.max_results - offset))
            if client_results:
                self.assertEqual(all_results[offset].entry_id, client_results[0].entry_id)

    async def test_no_duplicates(self):
        search = arxiv.Search("testing", max_results=100)
        ids = set()
        client = AsyncClient()
        async for r in client.results(search):
            self.assertFalse(r.entry_id in ids)
            ids.add(r.entry_id)

    @patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=empty_response(500))
    @patch("asyncio.sleep", new_callable=AsyncMock, return_value=None)
    async def test_retry(self, mock_sleep, mock_get):
        broken_client = AsyncClient()

        async def broken_get():
            search = arxiv.Search(query="quantum")
            async for r in broken_client.results(search):
                return r

        with self.assertRaises(arxiv.HTTPError):
            await broken_get()

        for num_retries in [2, 5]:
            broken_client.num_retries = num_retries
            try:
                await broken_get()
                self.fail("broken_get didn't throw HTTPError")
            except arxiv.HTTPError as e:
                self.assertEqual(e.status, 500)
                self.assertEqual(e.retry, broken_client.num_retries)

    @patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=empty_response(200))
    @patch("asyncio.sleep", new_callable=AsyncMock, return_value=None)
    async def test_sleep_standard(self, mock_sleep, mock_get):
        client = AsyncClient()
        url = client._format_url(arxiv.Search(query="quantum"), 0, 1)
        # A client should sleep until delay_seconds have passed.
        await client._parse_feed(url)
        mock_sleep.assert_not_called()
        # Overwrite _last_request_dt to minimize flakiness: different
        # environments will have different page fetch times.
        client._last_request_dt = datetime.now()
        await client._parse_feed(url)
        mock_sleep.assert_called_once_with(approx(client.delay_seconds, rel=1e-3))

    @patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=empty_response(200))
    @patch("asyncio.sleep", new_callable=AsyncMock, return_value=None)
    async def test_sleep_multiple_requests(self, mock_sleep, mock_get):
        client = AsyncClient()
        url1 = client._format_url(arxiv.Search(query="quantum"), 0, 1)
        url2 = client._format_url(arxiv.Search(query="testing"), 0, 1)
        # Rate limiting is URL-independent; expect same behavior as in
        # `test_sleep_standard`.
        await client._parse_feed(url1)
        mock_sleep.assert_not_called()
        client._last_request_dt = datetime.now()
        await client._parse_feed(url2)
        mock_sleep.assert_called_once_with(approx(client.delay_seconds, rel=1e-3))

    @patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=empty_response(200))
    @patch("asyncio.sleep", new_callable=AsyncMock, return_value=None)
    async def test_sleep_elapsed(self, mock_sleep, mock_get):
        client = AsyncClient()
        url = client._format_url(arxiv.Search(query="quantum"), 0, 1)
        # If _last_request_dt is less than delay_seconds ago, sleep.
        client._last_request_dt = datetime.now() - timedelta(seconds=client.delay_seconds - 1)
        await client._parse_feed(url)
        mock_sleep.assert_called_once()
        mock_sleep.reset_mock()
        # If _last_request_dt is at least delay_seconds ago, don't sleep.
        client._last_request_dt = datetime.now() - timedelta(seconds=client.delay_seconds)
        await client._parse_feed(url)
        mock_sleep.assert_not_called()

    @patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=empty_response(200))
    @patch("asyncio.sleep", new_callable=AsyncMock, return_value=None)
    async def test_sleep_zero_delay(self, mock_sleep, mock_get):
        client = AsyncClient(delay_seconds=0)
        url = client._format_url(arxiv.Search(query="quantum"), 0, 1)
        await client._parse_feed(url)
        await client._parse_feed(url)
        mock_sleep.assert_not_called()

    @patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=empty_response(500))
    @patch("asyncio.sleep", new_callable=AsyncMock, return_value=None)
    async def test_sleep_between_errors(self, mock_sleep, mock_get):
        client = AsyncClient()
        url = client._format_url(arxiv.Search(query="quantum"), 0, 1)
        try:
            await client._parse_feed(url)
        except arxiv.HTTPError:
            pass
        # Should sleep between retries.
        mock_sleep.assert_called()
        self.assertEqual(mock_sleep.call_count, client.num_retries)
        mock_sleep.assert_has_calls(
            [
                call(approx(client.delay_seconds, abs=1e-2)),
            ]
            * client.num_retries
        )

    @patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=empty_response(200))
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_concurrency_locking(self, mock_sleep, mock_get):
        """
        Test that multiple concurrent requests obey the rate limit (delay_seconds).
        Without the lock, both requests might see '_last_request_dt' as None (or old)
        and skip sleeping, violating the rate limit.
        """
        client = AsyncClient(delay_seconds=1.0)
        url = client._format_url(arxiv.Search(query="concurrent"), 0, 1)

        # Run two requests concurrently
        await asyncio.gather(
            client._parse_feed(url),
            client._parse_feed(url)
        )

        # The first request should proceed immediately (no sleep).
        # The second request must wait for the lock, see the updated _last_request_dt,
        # and sleep for the remainder of the delay_seconds.
        self.assertTrue(mock_sleep.called, "Should have slept for the second concurrent request")
        self.assertEqual(mock_sleep.call_count, 1, "Should have successfully locked and slept exactly once")
        
        mock_sleep.assert_called_with(approx(1.0, rel=0.1))
